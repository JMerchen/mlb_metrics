"""Per-player projected stat lines for NFL player props - volume x
efficiency, replacing the direction-only opponent ratio that
`nfl_player_props._skill_category_edges` has used until now.

WHY THIS EXISTS (2026-09-17 user report). The props board showed an
`opponent_allowed_rate` of 202.4 next to Jauan Jennings' 31.7
receiving yards per game. That number is the total receiving yards the
opponent allows to ALL opposing WRs per game, so it is not at player
scale - but the deeper problem is that it is BIASED, not merely
mis-scaled. Yards-allowed-PER-GAME confounds two unrelated things:

  1. how many targets a defense faces (their offense's pace, game
     script, how often they trail), which the defense does not control
     and which says nothing about how well they cover; and
  2. how many yards they surrender on each of those targets, which is
     the actual defensive quality the prop cares about.

Measured on the same 2025+2026 history the board runs on, Chicago
allowed 202.4 receiving yards per game to WRs against a league rate of
138.2 - a 1.47x "matchup." Per TARGET they allowed 9.32 against a
league 7.91, a 1.17x matchup. The per-game framing overstated the
edge by roughly the amount Chicago's pace deviates from average, and
`config.NFL_PROP_MATCHUP_CLIP` then hid the overstatement by pinning
the ratio at its 1.2 ceiling. Every per-game opponent adjustment in
the props path carried that same confound.

The second failure was structural: the ratio model never multiplied
anything, so it produced a DIRECTION ("Over") without a MAGNITUDE. It
could not answer "how many yards do we think this player gets?", which
is the only question a prop actually poses. Jefferson and Jennings -
same team, same opponent, target shares of 0.314 and 0.162 - received
an identical 1.2 ratio and an identical "Over", because the sole
input driving the call was the defense they happened to share.

THE MODEL. A projection is volume times efficiency:

    projected_targets   = target_share x team_targets_per_game
    projected_receptions = projected_targets x catch_rate x catch_multiplier
    projected_rec_yards  = projected_targets x yards_per_target x ypt_multiplier

and the rushing mirror of it through carry_share and yards per carry.
Three properties of that decomposition drive the design here:

* Volume carries the signal and is stable. Target share is a
  team-normalised usage rate that settles within a few games and is
  what actually separates two players facing the same defense (the
  Jefferson/Jennings 2:1 split above is entirely a volume effect). It
  is therefore used essentially as observed, recency-weighted only.

* Efficiency is noisy and must be shrunk. Yards per target needs
  something like 50-100 attempts before it carries much signal, and
  early in a season a player may have a dozen. Both player and defense
  efficiency go through the empirical-Bayes shrinkage in
  `_shrink_rate` - a weighted average of the observed rate and the
  position's league rate, where the league prior carries the weight of
  `config.NFL_PROP_EFFICIENCY_PRIOR_TARGETS` (resp.
  `..._DEFENSE_PRIOR_TARGETS`) synthetic attempts. With one week of
  2026 played this is what stops a 2-target, 60-yard week from
  projecting a 30 yard per target receiver.

* The opponent adjustment belongs on efficiency ONLY. Multiplying a
  volume estimate by a per-game allowed rate double-counts pace, which
  is precisely the 1.47-vs-1.17 error above. Volume comes from the
  player's own team; the defense moves only the per-play terms.

WHERE EPA COMES IN. Defensive EPA allowed per target is a better-
behaved quality signal than raw yards allowed per target - it is
situation-aware (down, distance, field position) and settles faster,
where a yards rate can be dragged around by two broken tackles. It
cannot be the spine of the projection, because EPA is not in yards and
`targets x EPA` is not a stat line. It is used instead to stabilise
the ORDERING of the opponent multiplier: `_blend_multiplier`
standardises the yards-based and EPA-based views across the league
within a position, blends the z-scores at
`config.NFL_PROP_DEFENSE_EPA_BLEND`, and maps the result back onto the
yards multiplier's own mean and spread. The multiplier stays in yards
units and stays centred on 1.0; only its rank order borrows EPA's
steadier hand.

DATA SOURCE. Everything is computed from the weekly player table that
`nfl_player_props.build_prop_edges` is already handed - no new fetch.
Play-by-play would also serve (and was checked against: pbp-derived
targets match the weekly `targets` column exactly on 98.1% of
player-weeks, and pbp-derived defensive rates land within ~2% of the
weekly-derived ones), but weekly is self-consistent with the
`target_share` column whose denominator the volume layer depends on,
and it avoids making the props path carry a second large table.

KNOWN LIMITATION, stated rather than hidden: team volume is the
team's own recency-weighted rate and is NOT adjusted for the upcoming
game's script. A heavy favourite throws less than its season rate and
a heavy underdog throws more, and `spread_line`/`total_line` are
available on the schedule to model that. It is left out of this
version deliberately - it is a real effect but an untested one, and
this module is already replacing the board's central calculation.
"""

import numpy as np
import pandas as pd

from mlb_metrics import config, nfl_rush_rec


# Per-category plumbing: which weekly columns carry the numerator and
# denominator of each per-play rate, so the receiving and rushing sides
# can share one shrinkage/blending path instead of duplicating it.
RECEIVING_ATTEMPT_COLUMN = "targets"
RUSHING_ATTEMPT_COLUMN = "carries"

# Every column `project_player_stats` guarantees, so an empty result can
# still be selected from by name (see that function's own empty guard).
PROJECTION_OUTPUT_COLUMNS = [
    "player_id", "position", "team", "opponent", "games",
    "target_share", "carry_share", "team_targets_per_game", "team_carries_per_game",
    "yards_per_target", "catch_rate", "yards_per_carry",
    "ypt_multiplier", "catch_multiplier", "ypc_multiplier",
    "projected_targets", "projected_carries",
    "projected_receptions", "projected_receiving_yards", "projected_rushing_yards",
]


def _recency_weights(ranked: pd.Series, windows) -> pd.Series:
    """Per-row weight from the same (games_back, weight) window structure
    `config.NFL_SKILL_WINDOWS` and friends already use, where `ranked` is
    a 0-based recency rank (0 = most recent game).

    The existing rolling helpers in this project average a per-game rate
    inside each window and then blend those averages. That is correct for
    a counting stat per game, but wrong for a RATIO whose denominator
    varies game to game: a 1-target game and a 12-target game would get
    equal say in a yards-per-target average. Weighting each row and then
    dividing summed numerator by summed denominator keeps the ratio
    attempt-weighted, which is what a per-play rate means. With the
    default [(None, 0.20), (8, 0.30), (4, 0.50)] a row's weight decays
    1.00 -> 0.50 -> 0.20 as it ages past the 4- and 8-game marks.
    """
    weights = pd.Series(0.0, index=ranked.index)
    for games_back, weight in windows:
        if games_back is None:
            weights += weight
        else:
            weights += np.where(ranked < games_back, weight, 0.0)
    return weights


def _shrink_rate(numerator: pd.Series, denominator: pd.Series, prior_rate, prior_strength: float) -> pd.Series:
    """Empirical-Bayes shrinkage of `numerator/denominator` toward
    `prior_rate`, with the prior carrying the weight of `prior_strength`
    synthetic attempts: (num + k*prior) / (den + k).

    A player with zero attempts lands exactly on the prior instead of
    NaN, and one with many attempts is left essentially alone - the
    pull is strongest exactly where the sample is thinnest, which is
    the whole point early in a season.
    """
    return (numerator + prior_strength * prior_rate) / (denominator + prior_strength)


def _blend_multiplier(
    rates: pd.DataFrame, multiplier_col: str, epa_col: str, group_col: str, blend: float
) -> pd.Series:
    """Combine a yards-based opponent multiplier with an EPA-based view
    of the same defense, returning a multiplier still expressed in the
    yards multiplier's own units and centred on its own mean.

    Both views are standardised within `group_col` (always position
    here), blended as z-scores at `blend` weight on EPA, then mapped back
    through the yards multiplier's own mean and standard deviation. The
    output therefore keeps the centre and spread of the pure-yards
    multiplier and can be multiplied straight into a yards projection;
    only the ORDER of the teams changes, to reflect EPA's steadier read.

    Degenerate groups - a single team, or zero variance in either view -
    keep the unblended yards multiplier rather than dividing by zero.

    MEASURED RESULT, recorded here because it is the honest answer to
    "should EPA be in this model at all": blending EPA in does not
    improve the projection. Replaying 2025 with
    scripts/backtest_nfl_prop_projections.py, mean directional hit rate
    across categories came out 0.5607 / 0.5598 / 0.5634 / 0.5598 at EPA
    weights of 0.0 / 0.35 / 0.7 / 1.0 over the thin-sample early weeks
    (2-5), where EPA's faster settling should have helped most, and
    0.5414 / 0.5414 / 0.5407 / 0.5414 over the mature weeks (10-18).
    That is flat to within noise everywhere. `config.NFL_PROP_DEFENSE_EPA_BLEND` is
    therefore shipped at 0.0, and this machinery is retained as a tested
    lever rather than removed - EPA remains theoretically the better
    behaved signal and may earn its place against a real book line,
    which is a different target than the baseline measured here.
    """
    multiplier = rates[multiplier_col]
    if blend <= 0:
        return multiplier

    grouped_multiplier = rates.groupby(group_col)[multiplier_col]
    grouped_epa = rates.groupby(group_col)[epa_col]
    multiplier_mean, multiplier_std = grouped_multiplier.transform("mean"), grouped_multiplier.transform("std")
    epa_mean, epa_std = grouped_epa.transform("mean"), grouped_epa.transform("std")

    usable = (multiplier_std > 0) & (epa_std > 0) & multiplier_std.notna() & epa_std.notna()
    z_blend = (
        (1.0 - blend) * (multiplier - multiplier_mean) / multiplier_std
        + blend * (rates[epa_col] - epa_mean) / epa_std
    )
    return pd.Series(
        np.where(usable, multiplier_mean + z_blend * multiplier_std, multiplier), index=rates.index
    )


def _skill_rows(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Regular-season rows for the skill positions, recency-ranked per
    player, with the row weights the usage/efficiency layers share."""
    scoped = weekly_df[weekly_df["position"].isin(nfl_rush_rec.SKILL_POSITIONS)].copy()
    if "season_type" in scoped.columns:
        scoped = scoped[scoped["season_type"] == "REG"]
    return scoped


def compute_team_volume(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per team: recency-weighted targets and carries per game.

    This is the denominator the usage shares multiply back against, so
    it is deliberately built by summing the SAME weekly player rows that
    `target_share` is defined over (player targets / team targets in
    that game). Taking it from play-by-play instead would introduce a
    handful of laterals and two-point plays that `target_share`'s own
    denominator excludes, and the shares would no longer reconstitute
    the team total.
    """
    scoped = _skill_rows(weekly_df)
    if scoped.empty:
        return pd.DataFrame(columns=["team", "games", "team_targets_per_game", "team_carries_per_game"])

    team_week = scoped.groupby(["team", "season", "week"], as_index=False).agg(
        targets=(RECEIVING_ATTEMPT_COLUMN, "sum"), carries=(RUSHING_ATTEMPT_COLUMN, "sum")
    )
    ordered = team_week.sort_values(["season", "week"], ascending=False)
    ordered["_recency_rank"] = ordered.groupby("team").cumcount()
    ordered["_weight"] = _recency_weights(ordered["_recency_rank"], config.NFL_SKILL_WINDOWS)

    ordered["_wt_targets"] = ordered["targets"] * ordered["_weight"]
    ordered["_wt_carries"] = ordered["carries"] * ordered["_weight"]
    agg = ordered.groupby("team", as_index=False).agg(
        games=("_recency_rank", "size"),
        _wt=("_weight", "sum"),
        _wt_targets=("_wt_targets", "sum"),
        _wt_carries=("_wt_carries", "sum"),
    )
    agg["team_targets_per_game"] = agg["_wt_targets"] / agg["_wt"]
    agg["team_carries_per_game"] = agg["_wt_carries"] / agg["_wt"]
    return agg[["team", "games", "team_targets_per_game", "team_carries_per_game"]]


def compute_player_usage(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per player: recency-weighted target share and carry share.

    Normalising matters because it separates a player's role from his
    offense's volume: a 20% target share means the same thing on a
    run-heavy team and a pass-happy one, where raw targets per game does
    not.

    Both shares are DERIVED here (player attempts over the team's
    attempts in that same game) rather than read from the weekly table's
    own `target_share` column, for two reasons. The shares must share a
    denominator with `compute_team_volume`, because the projection
    multiplies one back against the other and a mismatch would silently
    mis-scale every projected target; and the shipped column's
    denominator counts targets to non-skill players (a lineman on a
    trick play, a quarterback on a pass from a receiver) that this
    module's team total deliberately excludes. Checked against the
    shipped column across 5,676 player-weeks of 2025-2026: correlation
    0.9992, mean absolute difference 0.0009, with the handful of larger
    gaps all attributable to exactly that denominator difference.
    """
    scoped = _skill_rows(weekly_df)
    if scoped.empty:
        return pd.DataFrame(columns=["player_id", "position", "games", "target_share", "carry_share"])

    team_week = scoped.groupby(["team", "season", "week"], as_index=False).agg(
        _team_targets=(RECEIVING_ATTEMPT_COLUMN, "sum"), _team_carries=(RUSHING_ATTEMPT_COLUMN, "sum")
    )
    scoped = scoped.merge(team_week, on=["team", "season", "week"], how="left")
    scoped["_carry_share"] = np.where(
        scoped["_team_carries"] > 0, scoped[RUSHING_ATTEMPT_COLUMN] / scoped["_team_carries"], 0.0
    )
    scoped["_target_share"] = np.where(
        scoped["_team_targets"] > 0, scoped[RECEIVING_ATTEMPT_COLUMN] / scoped["_team_targets"], 0.0
    )

    ordered = scoped.sort_values(["season", "week"], ascending=False)
    ordered["_recency_rank"] = ordered.groupby("player_id").cumcount()
    ordered["_weight"] = _recency_weights(ordered["_recency_rank"], config.NFL_SKILL_WINDOWS)

    ordered["_wt_target_share"] = ordered["_target_share"] * ordered["_weight"]
    ordered["_wt_carry_share"] = ordered["_carry_share"] * ordered["_weight"]
    agg = ordered.groupby("player_id", as_index=False).agg(
        position=("position", "first"),
        games=("_recency_rank", "size"),
        _wt=("_weight", "sum"),
        _wt_target_share=("_wt_target_share", "sum"),
        _wt_carry_share=("_wt_carry_share", "sum"),
    )
    agg["target_share"] = agg["_wt_target_share"] / agg["_wt"]
    agg["carry_share"] = agg["_wt_carry_share"] / agg["_wt"]
    return agg[["player_id", "position", "games", "target_share", "carry_share"]]


def league_efficiency_rates(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per position: the league's own attempt-weighted yards per
    target, catch rate, and yards per carry. These are the shrinkage
    priors every player and defense rate is pulled toward, so they are
    pooled across all attempts rather than averaged over players (an
    average of player rates would let a 2-target receiver count as much
    as a 150-target one).
    """
    scoped = _skill_rows(weekly_df)
    if scoped.empty:
        return pd.DataFrame(columns=["position", "yards_per_target", "catch_rate", "yards_per_carry"])

    agg = scoped.groupby("position", as_index=False).agg(
        targets=(RECEIVING_ATTEMPT_COLUMN, "sum"),
        receptions=("receptions", "sum"),
        receiving_yards=("receiving_yards", "sum"),
        carries=(RUSHING_ATTEMPT_COLUMN, "sum"),
        rushing_yards=("rushing_yards", "sum"),
    )
    agg["yards_per_target"] = np.where(agg["targets"] > 0, agg["receiving_yards"] / agg["targets"], 0.0)
    agg["catch_rate"] = np.where(agg["targets"] > 0, agg["receptions"] / agg["targets"], 0.0)
    agg["yards_per_carry"] = np.where(agg["carries"] > 0, agg["rushing_yards"] / agg["carries"], 0.0)
    return agg[["position", "yards_per_target", "catch_rate", "yards_per_carry"]]


def compute_player_efficiency(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per player: shrunk yards per target, catch rate, and yards
    per carry, each pulled toward that player's own position prior.

    Unlike the usage layer these are recency-weighted AND shrunk. The
    weighting keeps a player's recent form relevant; the shrinkage stops
    a small denominator from producing a nonsense rate. Both matter -
    without shrinkage a receiver with 3 targets and a 55-yard catch
    projects at 18 yards per target forever.
    """
    scoped = _skill_rows(weekly_df)
    league = league_efficiency_rates(weekly_df).set_index("position")
    if scoped.empty:
        return pd.DataFrame(
            columns=["player_id", "position", "targets", "carries", "yards_per_target", "catch_rate", "yards_per_carry"]
        )

    ordered = scoped.sort_values(["season", "week"], ascending=False)
    ordered["_recency_rank"] = ordered.groupby("player_id").cumcount()
    ordered["_weight"] = _recency_weights(ordered["_recency_rank"], config.NFL_SKILL_WINDOWS)

    for column in (RECEIVING_ATTEMPT_COLUMN, "receptions", "receiving_yards", RUSHING_ATTEMPT_COLUMN, "rushing_yards"):
        ordered[f"_wt_{column}"] = ordered[column].fillna(0.0) * ordered["_weight"]

    agg = ordered.groupby("player_id", as_index=False).agg(
        position=("position", "first"),
        targets=(f"_wt_{RECEIVING_ATTEMPT_COLUMN}", "sum"),
        receptions=("_wt_receptions", "sum"),
        receiving_yards=("_wt_receiving_yards", "sum"),
        carries=(f"_wt_{RUSHING_ATTEMPT_COLUMN}", "sum"),
        rushing_yards=("_wt_rushing_yards", "sum"),
    )

    prior_ypt = agg["position"].map(league["yards_per_target"])
    prior_catch = agg["position"].map(league["catch_rate"])
    prior_ypc = agg["position"].map(league["yards_per_carry"])
    strength = config.NFL_PROP_EFFICIENCY_PRIOR_TARGETS

    agg["yards_per_target"] = _shrink_rate(agg["receiving_yards"], agg["targets"], prior_ypt, strength)
    agg["catch_rate"] = _shrink_rate(agg["receptions"], agg["targets"], prior_catch, strength)
    agg["yards_per_carry"] = _shrink_rate(agg["rushing_yards"], agg["carries"], prior_ypc, strength)
    return agg[
        ["player_id", "position", "targets", "carries", "yards_per_target", "catch_rate", "yards_per_carry"]
    ]


def compute_position_defense_per_play_rates(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (defending team, position): what that defense allows
    PER TARGET and PER CARRY to opposing players at that position,
    shrunk toward the position's league rate.

    This is the module's central correction. The per-game equivalent in
    `nfl_player_props.compute_position_defense_rolling_rates` answers
    "how many yards does this defense give up to WRs in a game", which
    moves with how many WR targets the defense happens to face. This
    answers "how many yards does this defense give up on each WR
    target", which does not. Retaining the attempt counts alongside the
    rates lets the caller see how thin a defense's sample is.
    """
    scoped = _skill_rows(weekly_df)
    league = league_efficiency_rates(weekly_df).set_index("position")
    columns = [
        "team", "position", "targets_faced", "carries_faced",
        "yards_allowed_per_target", "catch_rate_allowed", "yards_allowed_per_carry",
        "receiving_epa_per_target", "rushing_epa_per_carry",
    ]
    if scoped.empty:
        return pd.DataFrame(columns=columns)

    stat_columns = [
        RECEIVING_ATTEMPT_COLUMN, "receptions", "receiving_yards",
        RUSHING_ATTEMPT_COLUMN, "rushing_yards", "receiving_epa", "rushing_epa",
    ]
    for column in stat_columns:
        if column not in scoped.columns:
            scoped[column] = 0.0

    # Sum to one row per (defense, position, GAME) BEFORE ranking by
    # recency. Ranking the raw player rows instead would make the window
    # count rows rather than games - a defense that faced five receivers
    # in a week would burn its whole 4-deep full-weight window on that
    # single week, so "the last four games" would silently mean "the last
    # four opposing players." Every other rolling rate in this module
    # ranks one row per game already; this is the only place where the
    # grain differs, and collapsing first is what keeps them consistent.
    defense_game = scoped.groupby(
        ["opponent_team", "position", "season", "week"], as_index=False
    )[stat_columns].sum()

    ordered = defense_game.sort_values(["season", "week"], ascending=False)
    ordered["_recency_rank"] = ordered.groupby(["opponent_team", "position"]).cumcount()
    ordered["_weight"] = _recency_weights(ordered["_recency_rank"], config.NFL_DEFENSE_WINDOWS)

    for column in stat_columns:
        ordered[f"_wt_{column}"] = ordered[column].fillna(0.0) * ordered["_weight"]

    agg = ordered.groupby(["opponent_team", "position"], as_index=False).agg(
        targets_faced=(f"_wt_{RECEIVING_ATTEMPT_COLUMN}", "sum"),
        receptions_allowed=("_wt_receptions", "sum"),
        receiving_yards_allowed=("_wt_receiving_yards", "sum"),
        carries_faced=(f"_wt_{RUSHING_ATTEMPT_COLUMN}", "sum"),
        rushing_yards_allowed=("_wt_rushing_yards", "sum"),
        receiving_epa_allowed=("_wt_receiving_epa", "sum"),
        rushing_epa_allowed=("_wt_rushing_epa", "sum"),
    ).rename(columns={"opponent_team": "team"})

    prior_ypt = agg["position"].map(league["yards_per_target"])
    prior_catch = agg["position"].map(league["catch_rate"])
    prior_ypc = agg["position"].map(league["yards_per_carry"])
    strength = config.NFL_PROP_DEFENSE_PRIOR_TARGETS

    agg["yards_allowed_per_target"] = _shrink_rate(
        agg["receiving_yards_allowed"], agg["targets_faced"], prior_ypt, strength
    )
    agg["catch_rate_allowed"] = _shrink_rate(
        agg["receptions_allowed"], agg["targets_faced"], prior_catch, strength
    )
    agg["yards_allowed_per_carry"] = _shrink_rate(
        agg["rushing_yards_allowed"], agg["carries_faced"], prior_ypc, strength
    )
    # EPA priors are 0.0 by construction: EPA is already defined relative
    # to an average outcome, so "no information" for a defense is neutral
    # EPA, not the league's mean allowed EPA.
    agg["receiving_epa_per_target"] = _shrink_rate(
        agg["receiving_epa_allowed"], agg["targets_faced"], 0.0, strength
    )
    agg["rushing_epa_per_carry"] = _shrink_rate(
        agg["rushing_epa_allowed"], agg["carries_faced"], 0.0, strength
    )
    return agg[columns]


def compute_opponent_multipliers(defense_rates: pd.DataFrame, league_rates: pd.DataFrame) -> pd.DataFrame:
    """Turn the per-play allowed rates into multipliers centred on 1.0,
    with EPA blended into the ordering per `_blend_multiplier`.

    Each multiplier is computed WITHIN a position, so a defense is
    measured against how the league as a whole handles that position,
    never against a blend dominated by whichever position sees the most
    volume league-wide.
    """
    if defense_rates.empty:
        return defense_rates.assign(ypt_multiplier=[], catch_multiplier=[], ypc_multiplier=[])

    league = league_rates.set_index("position")
    rates = defense_rates.copy()
    rates["ypt_multiplier"] = rates["yards_allowed_per_target"] / rates["position"].map(league["yards_per_target"])
    rates["catch_multiplier"] = rates["catch_rate_allowed"] / rates["position"].map(league["catch_rate"])
    rates["ypc_multiplier"] = rates["yards_allowed_per_carry"] / rates["position"].map(league["yards_per_carry"])

    blend = config.NFL_PROP_DEFENSE_EPA_BLEND
    rates["ypt_multiplier"] = _blend_multiplier(
        rates, "ypt_multiplier", "receiving_epa_per_target", "position", blend
    )
    rates["ypc_multiplier"] = _blend_multiplier(
        rates, "ypc_multiplier", "rushing_epa_per_carry", "position", blend
    )

    low, high = config.NFL_PROP_PROJECTION_MULTIPLIER_CLIP
    for column in ("ypt_multiplier", "catch_multiplier", "ypc_multiplier"):
        rates[column] = rates[column].clip(low, high)
    return rates


def project_player_stats(
    weekly_df: pd.DataFrame, opponents_df: pd.DataFrame, latest_team_df: pd.DataFrame = None
) -> pd.DataFrame:
    """Projected receptions, receiving yards, and rushing yards for every
    skill player with an upcoming opponent.

    `opponents_df` is `nfl_matchup.team_opponents`' own output (team ->
    opponent for the week being projected). `latest_team_df` optionally
    supplies player_id -> team; when omitted it is derived from the most
    recent weekly row, the same mid-season-trade-tolerant "current team"
    rule the rest of the props path uses.

    The returned frame keeps every intermediate term - projected
    targets/carries, the shrunk player efficiency, and the opponent
    multiplier actually applied - so a projection on the board can be
    explained rather than merely asserted.
    """
    usage = compute_player_usage(weekly_df)
    efficiency = compute_player_efficiency(weekly_df)
    volume = compute_team_volume(weekly_df)
    league = league_efficiency_rates(weekly_df)
    multipliers = compute_opponent_multipliers(compute_position_defense_per_play_rates(weekly_df), league)

    # An empty result still has to carry the full schema: callers select
    # the projection columns by name, so returning a bare DataFrame here
    # turns "no qualifying players" into a KeyError several frames away
    # instead of an empty board.
    if usage.empty:
        return pd.DataFrame(columns=PROJECTION_OUTPUT_COLUMNS)

    if latest_team_df is None:
        ordered = _skill_rows(weekly_df).sort_values(["season", "week"], ascending=False)
        latest_team_df = ordered.groupby("player_id", as_index=False).first()[["player_id", "team"]]

    players = usage.merge(
        efficiency.drop(columns=["position"]), on="player_id", how="left"
    ).merge(latest_team_df, on="player_id", how="left")
    players = players.merge(volume, on="team", how="left", suffixes=("", "_team"))
    players = players.merge(opponents_df, on="team", how="left")

    opponent_multipliers = multipliers.rename(columns={"team": "opponent"})
    players = players.merge(
        opponent_multipliers[[
            "opponent", "position", "ypt_multiplier", "catch_multiplier", "ypc_multiplier",
            "yards_allowed_per_target", "catch_rate_allowed", "yards_allowed_per_carry",
        ]],
        on=["opponent", "position"], how="left",
    )
    # A defense with no history at all for a position (possible in week 1
    # of an expansion-thin sample) is treated as exactly average rather
    # than dropped - the player's own rate still projects fine.
    for column in ("ypt_multiplier", "catch_multiplier", "ypc_multiplier"):
        players[column] = players[column].fillna(1.0)

    players["projected_targets"] = players["target_share"] * players["team_targets_per_game"]
    players["projected_carries"] = players["carry_share"] * players["team_carries_per_game"]
    players["projected_receptions"] = (
        players["projected_targets"] * players["catch_rate"] * players["catch_multiplier"]
    )
    players["projected_receiving_yards"] = (
        players["projected_targets"] * players["yards_per_target"] * players["ypt_multiplier"]
    )
    players["projected_rushing_yards"] = (
        players["projected_carries"] * players["yards_per_carry"] * players["ypc_multiplier"]
    )
    return players


def compute_team_pass_volume(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per team: recency-weighted pass attempts per game, and the
    same for the attempts each DEFENSE faces.

    Built from the `attempts` column rather than the `targets` the
    receiving side uses, because a QB prop is scored on official pass
    attempts - the two differ by throwaways and a handful of plays with
    no recorded receiver, which matters at the QB's own scale even though
    it barely moves a target share.
    """
    scoped = weekly_df if "season_type" not in weekly_df.columns else weekly_df[weekly_df["season_type"] == "REG"]
    columns = ["team", "games", "team_attempts_per_game", "attempts_faced_per_game"]
    if scoped.empty:
        return pd.DataFrame(columns=columns)

    offense = scoped.groupby(["team", "season", "week"], as_index=False).agg(attempts=("attempts", "sum"))
    defense = scoped.groupby(["opponent_team", "season", "week"], as_index=False).agg(
        attempts_faced=("attempts", "sum")
    ).rename(columns={"opponent_team": "team"})
    team_week = offense.merge(defense, on=["team", "season", "week"], how="outer").fillna(
        {"attempts": 0.0, "attempts_faced": 0.0}
    )

    ordered = team_week.sort_values(["season", "week"], ascending=False)
    ordered["_recency_rank"] = ordered.groupby("team").cumcount()
    ordered["_weight"] = _recency_weights(ordered["_recency_rank"], config.NFL_SKILL_WINDOWS)
    ordered["_wt_attempts"] = ordered["attempts"] * ordered["_weight"]
    ordered["_wt_faced"] = ordered["attempts_faced"] * ordered["_weight"]

    agg = ordered.groupby("team", as_index=False).agg(
        games=("_recency_rank", "size"), _wt=("_weight", "sum"),
        _wt_attempts=("_wt_attempts", "sum"), _wt_faced=("_wt_faced", "sum"),
    )
    agg["team_attempts_per_game"] = agg["_wt_attempts"] / agg["_wt"]
    agg["attempts_faced_per_game"] = agg["_wt_faced"] / agg["_wt"]
    return agg[columns]


def compute_pass_defense_per_play_rates(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per defending team: passing yards allowed PER ATTEMPT,
    shrunk toward the league rate.

    The per-GAME equivalent this replaces is confounded by volume in
    exactly the way the receiving side was, and measurably so: across
    2024-2025, a defense's passing yards allowed per game correlates
    +0.659 with the pass attempts it faces, while its yards allowed per
    attempt correlates -0.229 with the same quantity. Per game, the
    league spans 198 to 261 yards - a 1.32x spread that reads as a large
    matchup difference but is substantially a measure of how often each
    defense's own offense forces opponents to throw.
    """
    scoped = weekly_df if "season_type" not in weekly_df.columns else weekly_df[weekly_df["season_type"] == "REG"]
    columns = ["team", "attempts_faced", "yards_allowed_per_attempt", "sacks_per_attempt_faced"]
    if scoped.empty:
        return pd.DataFrame(columns=columns)

    defense_game = scoped.groupby(["opponent_team", "season", "week"], as_index=False).agg(
        attempts=("attempts", "sum"), passing_yards=("passing_yards", "sum"),
        sacks=("sacks_suffered", "sum"),
    ).rename(columns={"opponent_team": "team"})

    ordered = defense_game.sort_values(["season", "week"], ascending=False)
    ordered["_recency_rank"] = ordered.groupby("team").cumcount()
    ordered["_weight"] = _recency_weights(ordered["_recency_rank"], config.NFL_DEFENSE_WINDOWS)
    for column in ("attempts", "passing_yards", "sacks"):
        ordered[f"_wt_{column}"] = ordered[column].fillna(0.0) * ordered["_weight"]

    agg = ordered.groupby("team", as_index=False).agg(
        attempts_faced=("_wt_attempts", "sum"),
        yards_allowed=("_wt_passing_yards", "sum"),
        sacks_forced=("_wt_sacks", "sum"),
    )
    league_ypa = agg["yards_allowed"].sum() / agg["attempts_faced"].sum() if agg["attempts_faced"].sum() else 0.0
    league_sack_rate = agg["sacks_forced"].sum() / agg["attempts_faced"].sum() if agg["attempts_faced"].sum() else 0.0
    strength = config.NFL_PROP_DEFENSE_PRIOR_ATTEMPTS

    agg["yards_allowed_per_attempt"] = _shrink_rate(
        agg["yards_allowed"], agg["attempts_faced"], league_ypa, strength
    )
    agg["sacks_per_attempt_faced"] = _shrink_rate(
        agg["sacks_forced"], agg["attempts_faced"], league_sack_rate, strength
    )
    return agg[columns]


def project_qb_stats(
    weekly_df: pd.DataFrame, opponents_df: pd.DataFrame, latest_team_df: pd.DataFrame = None
) -> pd.DataFrame:
    """Projected passing yards per QB: attempt share x team attempts x
    shrunk yards per attempt x the opponent's per-attempt multiplier.

    The same volume x efficiency decomposition the receiving side uses,
    with attempt share standing in for target share. A starter takes
    essentially all of his team's attempts so the share is near 1.0 and
    does little work, but expressing it as a share is what keeps a QB who
    split time mid-season - or one who just took over - from projecting
    on a per-game average that mixes both roles.
    """
    columns = [
        "player_id", "team", "opponent", "games", "attempt_share",
        "team_attempts_per_game", "projected_attempts", "yards_per_attempt",
        "ypa_multiplier", "projected_passing_yards",
    ]
    scoped = weekly_df if "season_type" not in weekly_df.columns else weekly_df[weekly_df["season_type"] == "REG"]
    scoped = scoped[scoped["position"] == "QB"]
    if scoped.empty:
        return pd.DataFrame(columns=columns)

    team_week = scoped.groupby(["team", "season", "week"], as_index=False).agg(_team_attempts=("attempts", "sum"))
    scoped = scoped.merge(team_week, on=["team", "season", "week"], how="left")
    scoped["_attempt_share"] = np.where(
        scoped["_team_attempts"] > 0, scoped["attempts"] / scoped["_team_attempts"], 0.0
    )

    ordered = scoped.sort_values(["season", "week"], ascending=False)
    ordered["_recency_rank"] = ordered.groupby("player_id").cumcount()
    ordered["_weight"] = _recency_weights(ordered["_recency_rank"], config.NFL_QB_WINDOWS)
    ordered["_wt_share"] = ordered["_attempt_share"] * ordered["_weight"]
    for column in ("attempts", "passing_yards"):
        ordered[f"_wt_{column}"] = ordered[column].fillna(0.0) * ordered["_weight"]

    players = ordered.groupby("player_id", as_index=False).agg(
        games=("_recency_rank", "size"), _wt=("_weight", "sum"),
        _wt_share=("_wt_share", "sum"),
        attempts=("_wt_attempts", "sum"), passing_yards=("_wt_passing_yards", "sum"),
    )
    players["attempt_share"] = players["_wt_share"] / players["_wt"]

    league_ypa = (
        players["passing_yards"].sum() / players["attempts"].sum() if players["attempts"].sum() else 0.0
    )
    players["yards_per_attempt"] = _shrink_rate(
        players["passing_yards"], players["attempts"], league_ypa, config.NFL_PROP_QB_PRIOR_ATTEMPTS
    )

    if latest_team_df is None:
        latest_team_df = ordered.groupby("player_id", as_index=False).first()[["player_id", "team"]]
    volume = compute_team_pass_volume(weekly_df)
    defense = compute_pass_defense_per_play_rates(weekly_df)

    players = players.merge(latest_team_df, on="player_id", how="left")
    players = players.merge(volume[["team", "team_attempts_per_game"]], on="team", how="left")
    players = players.merge(opponents_df, on="team", how="left")

    league_defense_ypa = (
        defense["yards_allowed_per_attempt"].mean() if not defense.empty else league_ypa
    )
    multiplier = defense.rename(columns={"team": "opponent"})[["opponent", "yards_allowed_per_attempt"]]
    players = players.merge(multiplier, on="opponent", how="left")
    players["ypa_multiplier"] = (
        players["yards_allowed_per_attempt"] / league_defense_ypa
    ).fillna(1.0) if league_defense_ypa else 1.0

    low, high = config.NFL_PROP_PROJECTION_MULTIPLIER_CLIP
    players["ypa_multiplier"] = players["ypa_multiplier"].clip(low, high)
    players["projected_attempts"] = players["attempt_share"] * players["team_attempts_per_game"]
    players["projected_passing_yards"] = (
        players["projected_attempts"] * players["yards_per_attempt"] * players["ypa_multiplier"]
    )
    return players[columns]


def compute_sacks_allowed_per_game(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per OFFENSE: sacks allowed per game, shrunk toward the
    league rate, plus the per-dropback rate behind it.

    Deliberately kept PER GAME, unlike every other opponent rate in this
    module, and the reason is a real asymmetry rather than an oversight.
    Everywhere else, a per-game allowed rate is confounded because the
    volume a defense faces is set by its own offense and says nothing
    about the player being projected. For a pass rusher the opponent's
    dropbacks are not a confound - they ARE his opportunity, so a rate
    that already includes them is the quantity the prop turns on.

    Measured rather than assumed, over 2024-2025: sacks allowed per game
    correlates only +0.153 with dropbacks per game, so the per-game
    figure is barely volume-inflated in the first place. Compare the
    +0.659 the passing-yards-per-game rate showed on the same data, which
    is why THAT one moved to a per-attempt basis and this one did not.

    What was genuinely wrong here was the clipping, not the basis. The
    board's Sacks column was pinned to a config.NFL_PROP_MATCHUP_CLIP
    boundary on 63% of rows - 16 distinct values across 135 - so it
    reported "at least 20%" while appearing to report a measurement.
    Replayed over 2025 weeks 4-18, removing the clip changed 54.5% of
    Sacks rows and left the directional hit rate EXACTLY unchanged at
    0.5490, which is the expected result rather than a disappointing
    one: direction depends on which side of 1.0 a ratio falls, and
    clipping is a positive monotonic transform that cannot move it.
    Unclipping is therefore a pure honesty fix - the number starts
    meaning what it says - bought at no cost in accuracy.

    NO PROJECTION IS OFFERED FOR SACKS, and that is a measured decision.
    A `projected_sacks = shrunk player rate x opponent multiplier` was
    built and backtested exactly like the other categories, and it was
    materially WORSE than the ratio model it would have replaced: a
    directional hit rate of 0.3845 against 0.5490, a difference of
    -0.1645 with a player-clustered 95% interval of [-0.1883, -0.1413],
    and a worse MAE too (0.2947 against 0.2805). Sacks are a rare,
    heavily zero-inflated counting event with no usage-share
    decomposition available in the weekly table - there is no
    pass-rush-snap denominator to play the role targets play for a
    receiver - so shrinking the per-game rate mostly destroyed the
    signal that separates rushers. The code for it was deleted rather
    than left dormant, so that nobody wires up a projection this project
    has already measured as harmful.
    """
    scoped = weekly_df if "season_type" not in weekly_df.columns else weekly_df[weekly_df["season_type"] == "REG"]
    columns = ["team", "games", "sacks_allowed_per_game", "sacks_allowed_per_dropback"]
    if scoped.empty:
        return pd.DataFrame(columns=columns)

    team_week = scoped.groupby(["team", "season", "week"], as_index=False).agg(
        sacks=("sacks_suffered", "sum"), attempts=("attempts", "sum")
    )
    ordered = team_week.sort_values(["season", "week"], ascending=False)
    ordered["_recency_rank"] = ordered.groupby("team").cumcount()
    ordered["_weight"] = _recency_weights(ordered["_recency_rank"], config.NFL_DEFENSE_WINDOWS)
    ordered["_wt_sacks"] = ordered["sacks"].fillna(0.0) * ordered["_weight"]
    ordered["_wt_attempts"] = ordered["attempts"].fillna(0.0) * ordered["_weight"]

    agg = ordered.groupby("team", as_index=False).agg(
        games=("_recency_rank", "size"), _wt=("_weight", "sum"),
        sacks=("_wt_sacks", "sum"), attempts=("_wt_attempts", "sum"),
    )
    league_per_game = agg["sacks"].sum() / agg["_wt"].sum() if agg["_wt"].sum() else 0.0
    dropbacks = agg["attempts"] + agg["sacks"]
    league_per_dropback = agg["sacks"].sum() / dropbacks.sum() if dropbacks.sum() else 0.0

    agg["sacks_allowed_per_game"] = _shrink_rate(
        agg["sacks"], agg["_wt"], league_per_game, config.NFL_PROP_SACK_PRIOR_GAMES
    )
    agg["sacks_allowed_per_dropback"] = _shrink_rate(
        agg["sacks"], dropbacks, league_per_dropback, config.NFL_PROP_DEFENSE_PRIOR_ATTEMPTS
    )
    return agg[columns]


# Column name each prop category's projection lives under, so callers can
# map a category straight onto its projected value without a branch.
PROP_CATEGORY_PROJECTION_COLUMNS = {
    "Receptions": "projected_receptions",
    "Receiving Yards": "projected_receiving_yards",
    "Rushing Yards": "projected_rushing_yards",
}

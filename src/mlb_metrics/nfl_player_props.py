"""Real, matchup-based NFL player prop rankings - direct user request
(2026-09-16): "Puka Nacua was targeted in almost half of his routes run
in week 1, so if he goes against a defense that's weak against the
pass, he'd be a good target to choose the over on receptions" - rank
real player prop bets (receptions, receiving yards, rushing yards,
passing yards, sacks) by how favorable each player's real upcoming
matchup is, not by any real sportsbook prop line (user-confirmed scope
via AskUserQuestion: "rank by matchup quality only, no real lines" -
these are real "good spots to look," not a priced +EV pick against a
specific number).

Reuses this project's own existing real building blocks rather than
duplicating them:
- `nfl_rush_rec.compute_skill_rolling_stats` / `nfl_passing.compute_qb_rolling_stats`
  for a player's own real recent per-game rate (receptions/receiving
  yards/rushing yards/passing yards).
- `nfl_teams.compute_defense_rolling_rates` for the opponent's real
  allowed-rate at those same categories.
- `nfl_matchup.compute_opponent_adjustment_ratio` for the actual ratio
  math (opponent's real allowed-rate vs the real league average,
  blended toward neutral by a real, backtested weight, clipped).
- `nfl_matchup.team_opponents`-style real "this week's real opponent,
  never a stale last-played one" lookup (the exact bug class that
  module's own docstring warns about).

Sacks is a genuinely new real category this module adds (a pass-RUSHER
prop, not a skill-player one) - `compute_pass_rush_rolling_stats`/
`compute_sacks_allowed_rolling_rates` below, built the same real way,
sourced from real, confirmed nflreadpy weekly-stats columns
(`def_sacks` for an individual defender's own real sacks, `sacks_suffered`
for a team's own real sacks-allowed-while-on-offense - both confirmed
live via this project's own cached `data/raw/nfl/weekly_2025.parquet`,
2026-09-16, not assumed from memory).

A DELIBERATELY SEPARATE clip/weight from `nfl_matchup.py`'s own
(`config.NFL_PROP_MATCHUP_CLIP`/`NFL_PROP_MATCHUP_WEIGHT`, not
`NFL_MATCHUP_OFFENSE_CLIP`/`NFL_MATCHUP_WEIGHT`) - see that config
block's own docstring for why a DK-fantasy-point-blending clip would be
the wrong tool here."""

import os

import numpy as np
import pandas as pd

from mlb_metrics import config, nfl_matchup, nfl_passing, nfl_prop_projections, nfl_rush_rec, nfl_teams


# The real weekly-stats column each real prop category is ultimately
# settled against - the single real place this mapping lives, so
# nfl_prop_predictions.resolve_prop_predictions can score a real logged
# pick against what the player ACTUALLY did that week without
# re-deriving (and drifting from) the category names the builders below
# emit. Deliberately keyed by the exact real category strings
# `build_prop_edges` itself produces, so a renamed or newly-added
# category surfaces as a real, visible KeyError-shaped gap at resolution
# time rather than silently scoring nothing.
PROP_CATEGORY_STAT_COLUMNS = {
    "Receptions": "receptions",
    "Receiving Yards": "receiving_yards",
    "Rushing Yards": "rushing_yards",
    "Passing Yards": "passing_yards",
    "Sacks": "def_sacks",
}


def compute_current_season_snap_share(
    snap_counts_df: pd.DataFrame, rosters_df: pd.DataFrame, season: int
) -> pd.DataFrame:
    """[player_id, snap_share] - each player's real total snaps THIS
    season as a share of their team's own real season snap total.

    Real, confirmed bug this exists to fix (2026-09-17 user report:
    "we need a snap share filter for the current season... there's four
    different running backs just from AZ"): this module builds a player's
    own per-game rate from `history["weekly"]`, which spans the PRIOR
    season plus the current one, so a player who put up real rates all of
    last season still carries a rate, still resolves to a real
    `latest_team`, and still gets matched to that team's real upcoming
    opponent - even if they have not taken a single real snap this year.
    Confirmed live on the real shipped week-2 board: all four of the real
    Arizona running backs it surfaced (Trey Benson, James Conner, Michael
    Carter, Bam Knight) had ZERO real 2026 appearances; Arizona's actual
    current backs were Tyler Allgeier and Jeremiyah Love. The existing
    `config.NFL_PROP_MIN_GAMES` floor cannot catch this, because those
    games are real - they just happened last season.

    Deliberately measured for BOTH sides of the ball, unlike
    `nfl_bestball.compute_player_snap_share` (offense only, which is all
    that ranking needs): this module's own Sacks category is about real
    pass RUSHERS, who take no real offensive snaps at all, so an
    offense-only share would silently filter every real defender off the
    board. Each player's real share is taken as the larger of their real
    offensive and defensive shares, which needs no position list and
    handles a real two-way or special-teams-only player without a special
    case.

    Denominator convention is `nfl_bestball.compute_player_snap_share`'s
    own, reused deliberately rather than reinvented: a team's real snap
    total for one game is the max real snap count among its players that
    game (in practice the real total - some lineman plays every snap),
    summed across every real game that team played this season, NOT just
    the games this player appeared in. So a real one-game cameo at a high
    per-game rate correctly reads as a LOW season share rather than
    looking like an every-week starter.

    `snap_counts_df` is keyed by `pfr_player_id`, a different real id
    space than `weekly_df`'s own gsis `player_id` - crossed over via
    `rosters_df`'s real `gsis_id`/`pfr_id`, the same real crosswalk
    `nfl_bestball.compute_player_snap_share`/
    `nfl_team_strength.compute_qb_continuity_adjustment` already use. A
    player absent from the result has no real snaps this season at all,
    which `build_prop_edges` treats as exactly what it is - not playing -
    rather than as a real 0.0 it would then have to compare."""
    season_snaps = snap_counts_df[
        (snap_counts_df["season"] == season) & (snap_counts_df["game_type"] == "REG")
    ]
    if season_snaps.empty:
        return pd.DataFrame(columns=["player_id", "snap_share"])

    shares = {}
    for column in ("offense_snaps", "defense_snaps"):
        team_game_totals = season_snaps.groupby(["team", "game_id"])[column].max()
        team_season_totals = team_game_totals.groupby("team").sum().rename("team_season_snaps")

        player_team = season_snaps.groupby(["pfr_player_id", "team"])[column].sum().rename("player_snaps")
        player_team = player_team.reset_index().merge(team_season_totals, on="team", how="left")

        totals = player_team.groupby("pfr_player_id")[["player_snaps", "team_season_snaps"]].sum()
        shares[column] = (totals["player_snaps"] / totals["team_season_snaps"]).rename(column)

    combined = pd.concat(shares.values(), axis=1)
    combined["snap_share"] = combined.max(axis=1)

    season_rosters = rosters_df[rosters_df["season"] == season]
    crosswalk = season_rosters.dropna(subset=["pfr_id"]).drop_duplicates("gsis_id")[["gsis_id", "pfr_id"]]

    result = combined.reset_index().merge(crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    return result.rename(columns={"gsis_id": "player_id"})[["player_id", "snap_share"]]


def _latest_player_names(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """[player_id, player_name]: each player's most recent real
    `player_display_name` - a player's own display name is stable
    game-to-game, but this mirrors nfl_rush_rec.compute_skill_rolling_stats'
    own "most recent row wins" convention for consistency (and correctly
    picks up a real mid-career name change, however rare)."""
    ordered = weekly_df.sort_values(["season", "week"], ascending=False)
    latest = ordered.groupby("player_id", as_index=False).first()
    return latest[["player_id", "player_display_name"]].rename(columns={"player_display_name": "player_name"})


def _rank_by_recency(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Adds `_recency_rank`: 0 for a player's own most recent game, 1 for
    their next-most-recent, etc. - independent per player_id. Identical
    to nfl_rush_rec._rank_by_recency/nfl_passing._rank_by_recency - not
    imported from either since both are private, module-local helpers
    there, same "small enough to duplicate rather than expose a private
    function across modules" call every other NFL rolling-stat module
    already makes."""
    ordered = weekly_df.sort_values(["season", "week"], ascending=False)
    return ordered.assign(_recency_rank=ordered.groupby("player_id").cumcount())


def compute_pass_rush_rolling_stats(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per real defensive player with any real recorded
    `def_sacks` history: [player_id, team, position, games,
    sacks_per_game], blended across config.NFL_PASS_RUSH_WINDOWS - same
    real windowing shape as nfl_rush_rec.compute_skill_rolling_stats,
    applied to a real individual pass-rusher's own real sacks instead of
    a skill player's real receiving/rushing stats. `weekly_df` is NOT
    pre-filtered by position here (real confirmed positions recording a
    real `def_sacks` value span DE/DT/LB/OLB/ILB/MLB/NT/DL/CB/SAF/S/FS/
    SS/DB - a real, occasionally cross-positional stat, e.g. a real
    blitzing DB), so this filters on `has_recorded_a_sack` (any real
    positive `def_sacks` value, ever) rather than a hand-maintained
    position allowlist that could quietly exclude a real, valid pass
    rusher.

    `weekly_df` may span multiple seasons - windows are purely games-back
    over each player's own rows, season boundaries don't reset or
    interrupt them."""
    has_recorded_a_sack = weekly_df.groupby("player_id")["def_sacks"].transform("max") > 0
    pass_rush_df = weekly_df[has_recorded_a_sack].copy()
    ranked = _rank_by_recency(pass_rush_df)

    blended = None
    full_games = None

    for games_back, weight in config.NFL_PASS_RUSH_WINDOWS:
        window_df = ranked if games_back is None else ranked[ranked["_recency_rank"] < games_back]
        agg = window_df.groupby("player_id", as_index=False).agg(
            games=("game_id", "nunique"), def_sacks=("def_sacks", "sum")
        )
        rate = (agg["def_sacks"] / agg["games"]).where(agg["games"] > 0, 0)
        contribution = agg[["player_id"]].assign(rate=rate.values).set_index("player_id")["rate"] * weight
        blended = contribution if blended is None else blended.add(contribution, fill_value=0)
        if games_back is None:
            full_games = agg[["player_id", "games"]]

    result = blended.rename("sacks_per_game").reset_index()
    result = result.merge(full_games, on="player_id", how="left")

    latest_info = ranked[ranked["_recency_rank"] == 0][["player_id", "team", "position"]]
    result = result.merge(latest_info, on="player_id", how="left")

    return result[["player_id", "team", "position", "games", "sacks_per_game"]]


def compute_sacks_allowed_rolling_rates(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per team: [team, games, sacks_allowed_per_game], blended
    across config.NFL_PASS_RUSH_WINDOWS - same real windowing shape as
    nfl_teams.compute_defense_rolling_rates, but grouped by a team's OWN
    `team` column (not `opponent_team`), since real `sacks_suffered` is
    already an OFFENSE-side stat recorded on the sacked team's own real
    weekly row - no opponent-perspective flip needed (unlike
    nfl_teams.compute_team_week_allowed's own `pass_yards_allowed` etc.,
    which start life on the OPPOSING offense's row and must be regrouped
    by `opponent_team` to become "what this defense allowed"). Summed
    across EVERY real player on a team-week, not filtered to QB alone -
    a real, rare non-QB `sacks_suffered` row (e.g. a real trick-play
    sack) still counts toward that team's real total sacks allowed that
    game, confirmed live to occur (2026-09-16, this project's own cached
    `data/raw/nfl/weekly_2025.parquet`)."""
    team_week = weekly_df.groupby(["team", "season", "week"], as_index=False).agg(
        sacks_allowed=("sacks_suffered", "sum")
    )
    ordered = team_week.sort_values(["season", "week"], ascending=False)
    ranked = ordered.assign(_recency_rank=ordered.groupby("team").cumcount())

    blended = None
    full_games = None

    for games_back, weight in config.NFL_PASS_RUSH_WINDOWS:
        window_df = ranked if games_back is None else ranked[ranked["_recency_rank"] < games_back]
        agg = window_df.groupby("team", as_index=False).agg(
            games=("_recency_rank", "size"), sacks_allowed=("sacks_allowed", "sum")
        )
        rate = (agg["sacks_allowed"] / agg["games"]).where(agg["games"] > 0, 0)
        contribution = agg[["team"]].assign(rate=rate.values).set_index("team")["rate"] * weight
        blended = contribution if blended is None else blended.add(contribution, fill_value=0)
        if games_back is None:
            full_games = agg[["team", "games"]]

    result = blended.rename("sacks_allowed_per_game").reset_index()
    result = result.merge(full_games, on="team", how="left")
    return result[["team", "games", "sacks_allowed_per_game"]]


def compute_position_defense_rolling_rates(weekly_df: pd.DataFrame, stat_col: str) -> pd.DataFrame:
    """One row per (defending team, position) among `nfl_rush_rec.SKILL_POSITIONS`
    (RB/WR/TE): per-game rate of `stat_col` allowed to opposing players AT
    THAT POSITION, blended across config.NFL_DEFENSE_WINDOWS - the same
    real windowed-blend machinery nfl_teams.compute_defense_rolling_rates
    already uses for its own team-wide (every position summed together)
    version, but split by the opposing player's own real position.

    Real, necessary fix (2026-09-16 user report): the team-wide version
    is dominated by whichever position gets the most real volume
    league-wide (WR), so "this defense allows 14.1 receptions/game" said
    nothing about how that SAME defense performs specifically against a
    TE or RB - real different positions face real different real
    matchups (a run-funnel defense might be generous to RBs but stingy
    to WRs) that a team-wide blend erases. Splitting by position, the
    same real windowed-blend shape every other defense rate in this
    project already uses, keeps a TE prop compared against real
    TE-allowed history, not a blanket team number that happens to be
    dominated by WR volume."""
    scoped = weekly_df[weekly_df["position"].isin(nfl_rush_rec.SKILL_POSITIONS)]
    team_week = scoped.groupby(["opponent_team", "position", "season", "week"], as_index=False).agg(
        allowed=(stat_col, "sum")
    ).rename(columns={"opponent_team": "team"})

    ordered = team_week.sort_values(["season", "week"], ascending=False)
    ranked = ordered.assign(_recency_rank=ordered.groupby(["team", "position"]).cumcount())

    blended = None
    full_games = None

    for games_back, weight in config.NFL_DEFENSE_WINDOWS:
        window_df = ranked if games_back is None else ranked[ranked["_recency_rank"] < games_back]
        agg = window_df.groupby(["team", "position"], as_index=False).agg(
            games=("_recency_rank", "size"), allowed=("allowed", "sum")
        )
        rate = (agg["allowed"] / agg["games"]).where(agg["games"] > 0, 0)
        contribution = agg[["team", "position"]].assign(rate=rate.values).set_index(["team", "position"])["rate"] * weight
        blended = contribution if blended is None else blended.add(contribution, fill_value=0)
        if games_back is None:
            full_games = agg[["team", "position", "games"]]

    result = blended.rename("allowed_per_game").reset_index()
    result = result.merge(full_games, on=["team", "position"], how="left")
    return result[["team", "position", "games", "allowed_per_game"]]


def _skill_category_edges(
    weekly_df: pd.DataFrame, current_week_schedule_df: pd.DataFrame, weight: float
) -> pd.DataFrame:
    """Receptions/Receiving Yards/Rushing Yards - all three come from ONE
    real call to nfl_rush_rec.compute_skill_rolling_stats (it already
    returns all three per-game rates for RB/WR/TE together). Each
    category's own opponent-allowed rate is looked up POSITION-SPECIFIC
    via `compute_position_defense_rolling_rates` (real, necessary fix,
    see that function's own docstring) - a WR's prop is compared against
    what that defense allows to opposing WRs, a TE's against opposing
    TEs, never one blanket team-wide number dominated by whichever
    position gets the most real volume league-wide."""
    skill_rolling = nfl_rush_rec.compute_skill_rolling_stats(weekly_df)
    names = _latest_player_names(weekly_df)
    opponents = nfl_matchup.team_opponents(current_week_schedule_df)

    # Real, latest team per player - nfl_rush_rec's own output has no
    # team column (see its module docstring: only player_id/position/
    # games/rates), so this is attached here the same "most recent real
    # row wins" way nfl_bestball.compute_player_games_played already
    # establishes for a mid-season-trade-tolerant "current team."
    ordered = weekly_df.sort_values(["season", "week"], ascending=False)
    latest_team = ordered.groupby("player_id", as_index=False).first()[["player_id", "team"]]

    players = skill_rolling.merge(names, on="player_id", how="left").merge(latest_team, on="player_id", how="left")
    players = players.merge(opponents, on="team", how="left")

    projections = nfl_prop_projections.project_player_stats(weekly_df, opponents, latest_team)
    defense_rates = nfl_prop_projections.compute_opponent_multipliers(
        nfl_prop_projections.compute_position_defense_per_play_rates(weekly_df),
        nfl_prop_projections.league_efficiency_rates(weekly_df),
    )
    league_rates = nfl_prop_projections.league_efficiency_rates(weekly_df).set_index("position")

    # (category, per-game baseline column, projection column, the defense's
    # own per-PLAY allowed column, that column's league prior, and what one
    # "play" means for this category - reported as `rate_basis` so the
    # allowed rate on the board is self-describing rather than silently
    # changing units between categories.
    category_specs = [
        ("Receptions", "receptions_per_game", "projected_receptions",
         "catch_rate_allowed", "catch_rate", "per target"),
        ("Receiving Yards", "receiving_yards_per_game", "projected_receiving_yards",
         "yards_allowed_per_target", "yards_per_target", "per target"),
        ("Rushing Yards", "rushing_yards_per_game", "projected_rushing_yards",
         "yards_allowed_per_carry", "yards_per_carry", "per carry"),
    ]

    projection_columns = ["player_id", "projected_targets", "projected_carries"] + [
        spec[2] for spec in category_specs
    ]

    frames = []
    for category, player_col, projection_col, allowed_col, league_col, basis in category_specs:
        rows = players.rename(columns={player_col: "player_rate"}).copy()
        rows["category"] = category
        rows["rate_basis"] = basis

        rows = rows.merge(projections[projection_columns], on="player_id", how="left")
        rows = rows.rename(columns={projection_col: "projection"})

        # The opponent's own PER-PLAY allowed rate, not the per-game one
        # the ratio model used. See nfl_prop_projections' module docstring
        # for the measured reason: per-game confounds defensive quality
        # with how many plays the defense faces, and the live board's
        # 202.4 receiving-yards-per-game figure was that confound showing
        # through at a scale no individual receiver could ever reach.
        allowed = defense_rates.rename(columns={"team": "opponent", allowed_col: "opponent_allowed_rate"})
        rows = rows.merge(
            allowed[["opponent", "position", "opponent_allowed_rate"]], on=["opponent", "position"], how="left"
        )
        rows["league_rate"] = rows["position"].map(league_rates[league_col])
        rows["opponent_allowed_rate"] = rows["opponent_allowed_rate"].fillna(rows["league_rate"])

        # `ratio` stays in the schema as the opponent's per-play multiplier
        # so downstream consumers (nfl_prop_predictions' log, the docs
        # table) keep working, but it is now a per-play quantity and is no
        # longer what the board ranks on - `top_prop_bets` ranks on how far
        # the projection departs from the player's own baseline.
        # A league rate of zero is possible for a category no one in the
        # pool has attempted yet (no TE has taken a carry, say). That is
        # "no information", which is a ratio of 1.0 - dividing would give
        # inf and let an empty category outrank every real matchup.
        rows["ratio"] = np.where(
            rows["league_rate"] > 0, rows["opponent_allowed_rate"] / rows["league_rate"], 1.0
        )

        frames.append(rows[[
            "player_id", "player_name", "team", "position", "opponent", "games",
            "category", "player_rate", "projection", "projected_targets", "projected_carries",
            "opponent_allowed_rate", "league_rate", "rate_basis", "ratio",
        ]])

    return pd.concat(frames, ignore_index=True)


def _passing_category_edges(
    weekly_df: pd.DataFrame, current_week_schedule_df: pd.DataFrame, weight: float
) -> pd.DataFrame:
    """Passing Yards - a QB's own real rolling passing_yards_per_game
    against the real opponent's own pass_yards_allowed_per_game (the
    SAME real allowed-rate table `_skill_category_edges` already uses
    for the Receiving Yards category - a real pass defense's own
    weakness is exactly the same real number whether it shows up as a
    WR's real receiving yards or the opposing QB's real passing yards)."""
    qb_rolling = nfl_passing.compute_qb_rolling_stats(weekly_df)
    names = _latest_player_names(weekly_df)
    defense_rates = nfl_teams.compute_defense_rolling_rates(weekly_df)
    opponents = nfl_matchup.team_opponents(current_week_schedule_df)

    ordered = weekly_df.sort_values(["season", "week"], ascending=False)
    latest_info = ordered.groupby("player_id", as_index=False).first()[["player_id", "team", "position"]]

    players = qb_rolling.merge(names, on="player_id", how="left").merge(latest_info, on="player_id", how="left")
    players = players.merge(opponents, on="team", how="left")

    rows = players.rename(columns={"passing_yards_per_game": "player_rate"}).copy()
    rows["category"] = "Passing Yards"

    defense_by_opponent = defense_rates.rename(
        columns={"team": "opponent", "pass_yards_allowed_per_game": "opponent_allowed_rate"}
    )
    rows = rows.merge(defense_by_opponent[["opponent", "opponent_allowed_rate"]], on="opponent", how="left")

    league_rate = defense_rates["pass_yards_allowed_per_game"].mean()
    rows["opponent_allowed_rate"] = rows["opponent_allowed_rate"].fillna(league_rate)
    rows["league_rate"] = league_rate
    rows["ratio"] = nfl_matchup.compute_opponent_adjustment_ratio(
        rows["opponent_allowed_rate"], league_rate, weight, clip=config.NFL_PROP_MATCHUP_CLIP
    )
    # Passing Yards has no projection: nfl_prop_projections models
    # receiving and rushing from usage share x team volume, and a QB has
    # no equivalent share to take of his own team's attempts. The columns
    # are carried as NaN so every category shares one schema, and
    # `top_prop_bets` falls back to ranking this category on its matchup
    # ratio - the same basis it always used, left deliberately unchanged
    # because nothing has been measured that would justify altering it.
    rows["projection"] = float("nan")
    rows["projected_targets"] = float("nan")
    rows["projected_carries"] = float("nan")
    rows["rate_basis"] = "per game"

    return rows[[
        "player_id", "player_name", "team", "position", "opponent", "games",
        "category", "player_rate", "projection", "projected_targets", "projected_carries",
        "opponent_allowed_rate", "league_rate", "rate_basis", "ratio",
    ]]


def _sacks_category_edges(
    weekly_df: pd.DataFrame, current_week_schedule_df: pd.DataFrame, weight: float
) -> pd.DataFrame:
    """Sacks - a real individual pass-rusher's own rolling
    sacks_per_game against the real opponent OFFENSE's own
    sacks_allowed_per_game (compute_sacks_allowed_rolling_rates above) -
    the only category here where "opponent" for a defensive player
    means the OFFENSE they're about to face, not another defense."""
    pass_rush_rolling = compute_pass_rush_rolling_stats(weekly_df)
    names = _latest_player_names(weekly_df)
    sacks_allowed = compute_sacks_allowed_rolling_rates(weekly_df)
    opponents = nfl_matchup.team_opponents(current_week_schedule_df)

    players = pass_rush_rolling.merge(names, on="player_id", how="left")
    players = players.merge(opponents, on="team", how="left")

    rows = players.rename(columns={"sacks_per_game": "player_rate"}).copy()
    rows["category"] = "Sacks"

    sacks_allowed_by_opponent = sacks_allowed.rename(
        columns={"team": "opponent", "sacks_allowed_per_game": "opponent_allowed_rate"}
    )
    rows = rows.merge(sacks_allowed_by_opponent[["opponent", "opponent_allowed_rate"]], on="opponent", how="left")

    league_rate = sacks_allowed["sacks_allowed_per_game"].mean()
    rows["opponent_allowed_rate"] = rows["opponent_allowed_rate"].fillna(league_rate)
    rows["league_rate"] = league_rate
    rows["ratio"] = nfl_matchup.compute_opponent_adjustment_ratio(
        rows["opponent_allowed_rate"], league_rate, weight, clip=config.NFL_PROP_MATCHUP_CLIP
    )
    # No projection for Sacks, for the same reason as Passing Yards above
    # (and more so: a sack is a rare counting event with no usage-share
    # decomposition at all). Ranked on its matchup ratio, unchanged.
    rows["projection"] = float("nan")
    rows["projected_targets"] = float("nan")
    rows["projected_carries"] = float("nan")
    rows["rate_basis"] = "per game"

    return rows[[
        "player_id", "player_name", "team", "position", "opponent", "games",
        "category", "player_rate", "projection", "projected_targets", "projected_carries",
        "opponent_allowed_rate", "league_rate", "rate_basis", "ratio",
    ]]


def build_prop_edges(
    weekly_df: pd.DataFrame,
    current_week_schedule_df: pd.DataFrame,
    weight: float = None,
    snap_share: pd.DataFrame = None,
    min_snap_share: float = None,
) -> pd.DataFrame:
    """One real row per (player, category) across all 5 real categories
    (Receptions, Receiving Yards, Rushing Yards, Passing Yards, Sacks):
    [player_id, player_name, team, position, opponent, games, category,
    player_rate, opponent_allowed_rate, league_rate, ratio]. A player
    with no real upcoming opponent this week (a real bye) gets dropped
    (an inner-join-shaped result via the `opponent` merge producing NaN,
    filtered by `top_prop_bets`'s own qualifiers) rather than a
    fabricated matchup.

    `weight` defaults to config.NFL_PROP_MATCHUP_WEIGHT (real,
    backtested - see that constant's own docstring) when not given -
    same "explicit override for backtesting, config default for live
    use" pattern every other NFL matchup weight in this project uses.

    `snap_share` (compute_current_season_snap_share's own output, see
    that function for the real bug this fixes) restricts the pool to
    players actually playing THIS season: a player below
    `min_snap_share` (config.NFL_PROP_MIN_SNAP_SHARE when not given), or
    absent from `snap_share` entirely (no real snaps this season at all),
    is dropped. Applied HERE rather than in `top_prop_bets` alongside the
    other qualifiers, deliberately: `top_prop_bets` ranks on a
    WITHIN-CATEGORY percentile, so leaving players who aren't playing in
    the pool would shift every remaining real player's own percentile -
    they have to be gone before anything is ranked, not filtered out
    afterwards.

    Omitting `snap_share` (the default) skips this filter entirely, and
    so does passing a real EMPTY frame - which is the honest behavior at
    real week 1, when no current-season snaps exist yet for anyone and
    filtering on them would empty the whole board rather than narrow it."""
    weight = config.NFL_PROP_MATCHUP_WEIGHT if weight is None else weight
    min_snap_share = config.NFL_PROP_MIN_SNAP_SHARE if min_snap_share is None else min_snap_share

    edges = pd.concat(
        [
            _skill_category_edges(weekly_df, current_week_schedule_df, weight),
            _passing_category_edges(weekly_df, current_week_schedule_df, weight),
            _sacks_category_edges(weekly_df, current_week_schedule_df, weight),
        ],
        ignore_index=True,
    )

    if snap_share is None or snap_share.empty:
        if snap_share is not None:
            print(
                "[nfl_player_props] No real current-season snap data yet - skipping the snap-share filter "
                "for this run (expected at real week 1, a real problem to look at otherwise)."
            )
        return edges

    qualified = snap_share[snap_share["snap_share"] >= min_snap_share]["player_id"]
    return edges[edges["player_id"].isin(set(qualified))].reset_index(drop=True)


def top_prop_bets(edges_df: pd.DataFrame, n: int = 10, min_games: int = None) -> pd.DataFrame:
    """Filters `edges_df` to real, meaningfully-rostered players
    (`games >= min_games` - config.NFL_PROP_MIN_GAMES if not given -
    AND `player_rate` at or above that category's own real
    config.NFL_PROP_MIN_USAGE floor, so a real garbage-time cameo
    catching one pass a game can never crowd out a real starter just
    because a tiny sample let a stray ratio run hot), then ranks by a
    real, WITHIN-CATEGORY percentile rank of the matchup ratio - NOT raw
    `abs(ratio - 1)` (a real, confirmed necessary fix: Sacks' own real
    allowed-rate is a far noisier per-game counting stat than receiving/
    rushing/passing yardage, so its raw ratio naturally swings much
    further from 1.0 for the SAME real relative rarity of matchup,
    letting Sacks alone crowd out every other real category's top
    matchups) and NOT a within-category z-score either (a real, ALSO
    confirmed problem: this real slate's own clip-saturated ties cluster
    several real players at the exact same ratio per opponent, and a
    category that happens to have a tighter real std - unrelated to how
    genuinely extreme any one real matchup is - gets an unfairly
    amplified z-score for the same real tie). Percentile rank
    (`pandas.Series.rank(pct=True)`, which averages ties fairly) is
    bounded in [0, 1] regardless of a category's own real distribution
    shape, so "how far into today's real slate's own tail does this
    matchup fall, for players in this category" is genuinely comparable
    across categories - confirmed live to actually diversify the top 10
    across this project's own 2025 week 10 real slate, unlike either
    prior approach. Returns the top `n` real rows with a `direction`
    column ("Over" when the real ratio > 1 - the opponent allows more
    than league average at this real category - "Under" when ratio < 1),
    sorted by certainty: primarily the real matchup `edge_percentile`
    above, with a real secondary tiebreak on `usage_percentile` (each
    player's own `player_rate`, ranked as a percentile WITHIN their
    category for the same cross-category-comparability reason as
    `edge_percentile`), and a real tertiary tiebreak on
    `raw_edge_magnitude` (the UNCLIPPED `|opponent_allowed_rate /
    league_rate - 1|`, ignoring `config.NFL_PROP_MATCHUP_CLIP` entirely).
    This secondary key is real, necessary signal, not cosmetic:
    `opponent_allowed_rate` is a real TEAM-level (or, since the 2026-09-16
    position-split fix, team+position-level) stat, so every player who
    shares that bucket shares the exact same real ratio and thus the
    exact same `edge_percentile` - a real, confirmed problem (2026-09-16
    user report: "the sort seems to be by team") where, without a real
    tiebreak, a stable sort's own incidental original row order (grouped
    by category from how `edges_df` is built) silently decided the order
    instead of anything real about certainty. Preferring the higher-usage
    player within a genuine tie is itself a real certainty signal (a
    bigger real per-game workload is a more stable, less boom/bust
    estimate of what a player will do), not an arbitrary tiebreaker.

    The tertiary key exists for a real, SEPARATE reason (also confirmed
    live, 2026-09-16, against this project's own cached real NFL data):
    on a real slate where several teams' matchup ratios all saturate the
    SAME clip boundary (a real, common occurrence, not an edge case),
    percentile rank alone can't distinguish "just barely hit the clip"
    from "would have been 3x more extreme without it" - both get the
    exact same edge_percentile, and usage_percentile doesn't help either
    since it's normalized WITHIN each row's own category, not across
    categories. The unclipped raw magnitude breaks that residual tie with
    a real, if noisier, signal - safe to use only here, as a last resort
    after two better-behaved keys are already exhausted, unlike using it
    as the PRIMARY key (which is exactly the "Sacks crowds out everything
    else" bug this function's own percentile-rank design already fixed)."""
    min_games = config.NFL_PROP_MIN_GAMES if min_games is None else min_games

    qualified = edges_df[edges_df["games"] >= min_games].copy()
    qualified["min_usage"] = qualified["category"].map(config.NFL_PROP_MIN_USAGE)
    qualified = qualified[qualified["player_rate"] >= qualified["min_usage"]]
    qualified = qualified.dropna(subset=["opponent"])

    # The board's confidence signal, per category:
    #
    #   skill categories (Receptions/Receiving Yards/Rushing Yards) - how
    #     far the PROJECTION departs from the player's own recency-blended
    #     per-game baseline, as a fraction of that baseline.
    #   Passing Yards / Sacks - the old |ratio - 1|, because those two
    #     categories have no projection (see their own edge functions).
    #
    # Both are "larger = more confident" and both are percentile-ranked
    # within their own category before being compared, so the cross-
    # category comparability the previous design established is preserved.
    #
    # Ranking on the projection gap is measured, not assumed
    # (scripts/backtest_nfl_prop_projections.py, replaying 2025 weeks
    # 4-18 with strict no-lookahead history). Directional accuracy rises
    # monotonically with the size of the gap - for Receptions, quintiles
    # of the gap hit 48.6% / 50.5% / 53.7% / 56.9% / 61.7% - and taking
    # only the top 10 rows per week by gap, which is exactly what this
    # function serves, hit 74.0% on Receiving Yards, 76.7% on Receptions
    # and 70.7% on Rushing Yards against 55.1% / 54.3% / 56.6% for all
    # qualified rows in those same categories.
    #
    # Against the ratio model this replaces, on a player-clustered paired
    # bootstrap over 419 receivers and 218 backs: Receiving Yards +4.5
    # points of hit rate (95% CI +2.2 to +6.8), Receptions +4.1 (+2.1 to
    # +6.2), Rushing Yards +3.0 (-1.1 to +7.0). The rushing interval
    # spans zero - the gain there is NOT established, and the projection
    # is used for that category on the strength of its (also
    # not-quite-significant) error reduction and the two categories where
    # the effect is solid, not because rushing itself was proven.
    #
    # WHAT THAT NUMBER ACTUALLY IS, because it is easy to misread in two
    # separate ways, and the first is not obvious:
    #
    # 1. It is mostly SELECTION, not direction. On those same top-10
    #    rows, simply betting Over every time scores 74.0% on Receiving
    #    Yards and 76.7% on Receptions - identical to the model, because
    #    the model calls Over on nearly all of them. For the two
    #    receiving categories the model is therefore not out-predicting a
    #    coin flip on WHICH SIDE; it is picking players whose trailing
    #    average understates them, after which the side is automatic.
    #    That is still real skill and it is the skill this ranking is
    #    for, but it is selection skill and is described as such. Only
    #    Rushing Yards adds direction on top (70.7% against 66.0% for
    #    always-Over on the same rows). A skewed base rate is NOT the
    #    explanation: across all qualified rows the actual beats the
    #    trailing mean just 45.4% / 50.0% / 49.0% of the time, so the
    #    stand-in line is fair and the effect is genuinely in the
    #    selection.
    #
    # 2. The line is a stand-in. These hit rates score against the
    #    player's own blended per-game rate, because this repo has no
    #    book prop lines (see nfl_prop_projections' module docstring). A
    #    real sportsbook line is far sharper than a trailing average and
    #    already prices in most of the role change the selection above
    #    is detecting. These figures establish that the ranking ORDERS
    #    bets by genuine confidence; they do NOT imply a 74% win rate
    #    against a real market, and must not be quoted as though they do.
    # Coerced explicitly: Passing Yards and Sacks contribute an all-NaN
    # `projection`, and concatenating those with the skill categories'
    # real floats can leave the column as object dtype, which numpy's
    # log rejects outright.
    projection = pd.to_numeric(qualified["projection"], errors="coerce")
    baseline = pd.to_numeric(qualified["player_rate"], errors="coerce")
    has_projection = projection.notna() & (baseline > 0) & (projection > 0)
    # The gap is measured in LOG space, deliberately. A projection is a
    # product of three ratios (share x volume x efficiency), so its
    # distribution is right-skewed: measured on the live 2026 week-2
    # slate, projection/baseline ran a q10 of 0.72 against a q90 of 1.58
    # for Receiving Yards, and 0.58 against 2.85 for Rushing Yards. A
    # plain |projection - baseline| / baseline therefore scores a 2x
    # overshoot as a gap of 1.0 and a 2x undershoot as only 0.5, and it
    # divides by a baseline that is itself smallest for the least
    # established players. Both effects push the same way, and the board
    # showed it: ranked on the percentage gap, the top 10 came back 100%
    # "Over" on Receiving Yards in backtest and 10 of 10 Over live. The
    # log ratio treats halving and doubling as equal departures, which is
    # the correct symmetry for a multiplicative quantity.
    #
    # It costs nothing measurable to do this. Top-10-per-week directional
    # accuracy over 2025 weeks 4-18 was 0.7333 / 0.7267 / 0.7067 on log
    # gap against 0.7400 / 0.7667 / 0.7067 on percentage gap - differences
    # of at most 1.1 standard errors on a 150-row sample, and that before
    # accounting for the clustering that makes the true interval wider
    # still - while the share of "Over" calls in the top 10 falls from
    # 100% / 96% / 82% to 92% / 82% / 70%.
    projection_gap = np.log(
        projection.where(has_projection) / baseline.where(has_projection)
    ).abs()
    ratio_gap = pd.Series(
        np.where(
            qualified["league_rate"] > 0,
            (qualified["opponent_allowed_rate"] / qualified["league_rate"] - 1).abs(),
            0.0,
        ),
        index=qualified.index,
    )
    qualified["confidence_signal"] = projection_gap.fillna(ratio_gap)

    # Over/Under follows whichever signal that category actually used: a
    # projection above the baseline is an Over, and where there is no
    # projection the opponent's matchup ratio decides as before.
    qualified["direction"] = np.where(
        has_projection,
        np.where(projection > baseline, "Over", "Under"),
        np.where(qualified["ratio"] > 1, "Over", "Under"),
    )
    # Ranked on the UNCLIPPED ratio, deliberately, even though `ratio`
    # (clipped to config.NFL_PROP_MATCHUP_CLIP) remains the real reported
    # signal - a real, measured fix (2026-09-16), not a preference:
    #
    # The clip earns its keep on MAGNITUDE (a real re-run of
    # scripts/backtest_nfl_player_props.py over all 10 real seasons, on
    # the corrected position-split rates, confirmed (0.8, 1.2) is still
    # the real correlation optimum - a tighter clip genuinely suppresses
    # noisy extremes). But this key is a RANK, and that same real
    # backtest proves clipping cannot help a rank: its own tercile spread
    # came out IDENTICAL for every single (weight, clip) candidate tested
    # (+0.0268 for Receptions, +0.0321 for Sacks, across the entire real
    # grid), because rank is invariant to any positive monotonic
    # transform. Clipping before ranking therefore adds nothing and
    # actively destroys real ordering: every row past the boundary
    # collapses to one identical value. Measured live on the real shipped
    # week-2 slate: 35.6% of qualified rows pinned to a boundary, 63% for
    # Sacks (just 16 distinct real ratio values across 135 rows) - so the
    # top of the real board became one giant tie, ordered by the usage
    # tiebreak (volume) rather than by matchup quality, which is exactly
    # what a real user reported seeing.
    #
    # `weight` is a positive constant, so rank(1 + w*(raw - 1)) == rank(raw)
    # - ranking the plain unclipped ratio here is the same real ordering
    # the weighted-but-unclipped value would give, without re-deriving it.
    qualified["edge_percentile"] = qualified.groupby("category")["confidence_signal"].rank(pct=True)
    qualified["usage_percentile"] = qualified.groupby("category")["player_rate"].rank(pct=True)

    ranked = qualified.sort_values(
        ["edge_percentile", "usage_percentile", "confidence_signal"], ascending=[False, False, False]
    ).head(n)
    return ranked.drop(
        columns=["min_usage", "edge_percentile", "usage_percentile"]
    ).reset_index(drop=True)


def write_prop_bets_csv(edges_df: pd.DataFrame, output_path: str, top_n: int = 10) -> pd.DataFrame:
    """Real `top_prop_bets` output written to `output_path` - if NO real
    row qualifies this week (a real bye-heavy week, or every real player
    fails config.NFL_PROP_MIN_GAMES/NFL_PROP_MIN_USAGE), nothing is
    written and the prior week's real CSV is left in place, the same
    "don't let one bad/quiet week erase real history" posture
    `board_runner.run_board` already establishes for the 3 consensus
    boards. Returns the real top-N DataFrame (empty if nothing
    qualified)."""
    top = top_prop_bets(edges_df, n=top_n)
    if top.empty:
        print("No real player props qualified this week - writing nothing, leaving the prior CSV in place.")
        return top

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    top.to_csv(output_path, index=False)
    print(f"Wrote {output_path}")
    return top

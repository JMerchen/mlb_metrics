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

import pandas as pd

from mlb_metrics import config, nfl_matchup, nfl_passing, nfl_rush_rec, nfl_teams


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


def _skill_category_edges(
    weekly_df: pd.DataFrame, current_week_schedule_df: pd.DataFrame, weight: float
) -> pd.DataFrame:
    """Receptions/Receiving Yards/Rushing Yards - all three come from
    ONE real call to nfl_rush_rec.compute_skill_rolling_stats (it
    already returns all three per-game rates for RB/WR/TE together) and
    ONE real call to nfl_teams.compute_defense_rolling_rates (it already
    returns pass_yards_allowed/rush_yards_allowed/receptions_allowed
    together) - not three separate real fetches."""
    skill_rolling = nfl_rush_rec.compute_skill_rolling_stats(weekly_df)
    names = _latest_player_names(weekly_df)
    defense_rates = nfl_teams.compute_defense_rolling_rates(weekly_df)
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

    category_specs = [
        ("Receptions", "receptions_per_game", "receptions_allowed_per_game"),
        ("Receiving Yards", "receiving_yards_per_game", "pass_yards_allowed_per_game"),
        ("Rushing Yards", "rushing_yards_per_game", "rush_yards_allowed_per_game"),
    ]

    frames = []
    for category, player_col, allowed_col in category_specs:
        rows = players.rename(columns={player_col: "player_rate"}).copy()
        rows["category"] = category

        defense_by_opponent = defense_rates.rename(columns={"team": "opponent", allowed_col: "opponent_allowed_rate"})
        rows = rows.merge(defense_by_opponent[["opponent", "opponent_allowed_rate"]], on="opponent", how="left")

        league_rate = defense_rates[allowed_col].mean()
        rows["opponent_allowed_rate"] = rows["opponent_allowed_rate"].fillna(league_rate)
        rows["league_rate"] = league_rate
        rows["ratio"] = nfl_matchup.compute_opponent_adjustment_ratio(
            rows["opponent_allowed_rate"], league_rate, weight, clip=config.NFL_PROP_MATCHUP_CLIP
        )

        frames.append(rows[[
            "player_id", "player_name", "team", "position", "opponent", "games",
            "category", "player_rate", "opponent_allowed_rate", "league_rate", "ratio",
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

    return rows[[
        "player_id", "player_name", "team", "position", "opponent", "games",
        "category", "player_rate", "opponent_allowed_rate", "league_rate", "ratio",
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

    return rows[[
        "player_id", "player_name", "team", "position", "opponent", "games",
        "category", "player_rate", "opponent_allowed_rate", "league_rate", "ratio",
    ]]


def build_prop_edges(
    weekly_df: pd.DataFrame, current_week_schedule_df: pd.DataFrame, weight: float = None
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
    use" pattern every other NFL matchup weight in this project uses."""
    weight = config.NFL_PROP_MATCHUP_WEIGHT if weight is None else weight

    return pd.concat(
        [
            _skill_category_edges(weekly_df, current_week_schedule_df, weight),
            _passing_category_edges(weekly_df, current_week_schedule_df, weight),
            _sacks_category_edges(weekly_df, current_week_schedule_df, weight),
        ],
        ignore_index=True,
    )


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
    than league average at this real category - "Under" when ratio < 1)."""
    min_games = config.NFL_PROP_MIN_GAMES if min_games is None else min_games

    qualified = edges_df[edges_df["games"] >= min_games].copy()
    qualified["min_usage"] = qualified["category"].map(config.NFL_PROP_MIN_USAGE)
    qualified = qualified[qualified["player_rate"] >= qualified["min_usage"]]
    qualified = qualified.dropna(subset=["opponent"])

    qualified["direction"] = qualified["ratio"].apply(lambda r: "Over" if r > 1 else "Under")
    percentile = qualified.groupby("category")["ratio"].rank(pct=True)
    qualified["edge_percentile"] = (percentile - 0.5).abs()

    ranked = qualified.sort_values("edge_percentile", ascending=False).head(n)
    return ranked.drop(columns=["min_usage", "edge_percentile"]).reset_index(drop=True)


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

"""Weekly NFL player-prop logging and outcome resolution - the props
analog of nfl_game_predictions.py, and the direct answer to a real,
structural gap found 2026-09-16 while reviewing this feature against its
own stated goal ("find good bets"): nfl_pipeline.run() wrote
docs/data/nfl_player_props.csv fresh every week, OVERWRITING it, and
nothing anywhere ever recorded whether last week's real 10 picks
actually came in. Every other pick this project makes (MLB game picks,
NFL game picks) has a real log -> resolve -> evaluate loop; props had
none, so the feature's real hit rate was unknowable and unimprovable by
construction - no amount of tuning could be validated, and a real
regression could never be detected.

What "resolved" honestly means here, given this feature's own
user-confirmed scope (rank by matchup quality alone, NO real sportsbook
prop line to settle against - see nfl_player_props.py's own module
docstring): a real pick claims that a favorable/unfavorable real matchup
will push a player ABOVE/BELOW their OWN recent per-game baseline
(`player_rate` at pick time, logged alongside the pick precisely so this
is checkable later). So an "Over" pick is a real hit when the player's
real stat that week came in above that baseline, an "Under" when it came
in below. This is deliberately the SAME real quantity
scripts/backtest_nfl_player_props.py already validates offline
(`relative_performance` = actual / prior rate), so a real live hit rate
is directly comparable to the real backtested claim rather than
measuring some subtly different thing.

A real exact tie (actual == baseline, genuinely possible for a low-count
stat like Sacks or Receptions) is recorded as a real PUSH (`hit` left
null, `actual_value` still filled in), not silently scored as a loss -
the same "don't fabricate a result you don't have" posture
game_predictions.resolve_game_predictions already takes for an
unfinished game.
"""

import os

import pandas as pd

from mlb_metrics import config, nfl_player_props

PROP_PREDICTION_COLUMNS = [
    "season", "week", "player_id", "player_name", "team", "position", "opponent",
    "category", "direction", "baseline_rate", "opponent_allowed_rate", "league_rate", "ratio",
    "model_version", "actual_value", "hit",
]


def select_prop_picks(
    top_props: pd.DataFrame,
    season: int,
    week: int,
    model_version: str = None,
) -> pd.DataFrame:
    """Turns `nfl_player_props.top_prop_bets`' own real output into
    loggable rows - one real row per (player, category) pick, stamped
    with the real `season`/`week` they were made for and the real
    `model_version` that produced them.

    `player_rate` is logged as `baseline_rate`: the real per-game number
    this pick is implicitly claiming the player will beat (Over) or fall
    short of (Under). Renamed on the way into the log specifically so
    what it MEANS at resolution time is unambiguous - it is the real
    settle-against baseline, not a live stat.

    Returns an empty, correctly-shaped frame for an empty input (a real
    bye-heavy week, or a week where nothing cleared the real
    games/usage floors) rather than raising - same graceful contract
    `nfl_player_props.write_prop_bets_csv` already uses for that case."""
    model_version = config.NFL_PROP_MODEL_VERSION if model_version is None else model_version
    if top_props is None or top_props.empty:
        return pd.DataFrame(columns=PROP_PREDICTION_COLUMNS)

    picks = top_props.rename(columns={"player_rate": "baseline_rate"}).copy()
    picks["season"] = season
    picks["week"] = week
    picks["model_version"] = model_version
    picks["actual_value"] = pd.NA
    picks["hit"] = pd.NA

    return picks.reindex(columns=PROP_PREDICTION_COLUMNS)


def append_prop_predictions(picks: pd.DataFrame, log_path: str) -> pd.DataFrame:
    """Appends `picks` to the real prop log at `log_path`, deduping on
    (season, week, player_id, category) with the NEWEST row winning -
    direct mirror of nfl_game_predictions.append_game_predictions' own
    contract, with a per-player-per-category key in place of that one's
    game_id (one real player can legitimately appear in several real
    categories in the same real week, so category is part of the key).

    Re-running a real week before its games are played simply refreshes
    that week's rows in place; re-running it AFTER they resolve would
    re-blank them, which is exactly why resolution writes back to this
    same file and the pipeline only ever logs the real UPCOMING week."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path)
        combined = pd.concat([picks, existing], ignore_index=True)
    else:
        combined = picks

    combined = combined.drop_duplicates(subset=["season", "week", "player_id", "category"], keep="first")
    combined = combined.sort_values(["season", "week", "category", "player_id"]).reset_index(drop=True)
    combined.to_csv(log_path, index=False)
    return combined


def resolve_prop_predictions(log_path: str, weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Fills in `actual_value`/`hit` for any still-pending real rows
    (`actual_value` null) whose real week now has real weekly stats in
    `weekly_df` - the props analog of
    nfl_game_predictions.resolve_game_predictions, and like that one a
    single already-fetched real frame covers every pending week at once
    (no per-date fetch loop needed).

    A real pending pick whose week genuinely hasn't been played yet
    simply stays pending - it is matched on (season, week, player_id),
    so a week with no real rows in `weekly_df` yet resolves nothing,
    rather than scoring a fabricated 0.

    A real pick whose player has NO real row for that week at all (a
    real inactive/injured/benched player - genuinely common, and exactly
    the kind of outcome that must NOT be silently scored as a loss)
    likewise stays pending rather than being recorded as a miss: this
    feature never claimed the player would play, only how their real
    matchup graded, and quietly counting real DNPs as losses would
    understate a real hit rate for a reason that has nothing to do with
    the real signal being measured."""
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=PROP_PREDICTION_COLUMNS)

    log = pd.read_csv(log_path)
    if log.empty:
        return log

    pending = log["actual_value"].isna()
    if not pending.any() or weekly_df is None or weekly_df.empty:
        return log

    stat_columns = nfl_player_props.PROP_CATEGORY_STAT_COLUMNS
    available = [c for c in set(stat_columns.values()) if c in weekly_df.columns]
    actuals = weekly_df[["player_id", "season", "week"] + available].copy()

    for category, stat_col in stat_columns.items():
        if stat_col not in available:
            continue
        rows = pending & (log["category"] == category)
        if not rows.any():
            continue
        merged = log.loc[rows, ["player_id", "season", "week"]].merge(
            actuals[["player_id", "season", "week", stat_col]], on=["player_id", "season", "week"], how="left",
        )
        log.loc[rows, "actual_value"] = merged[stat_col].to_numpy()

    scored = log["actual_value"].notna() & log["hit"].isna()
    actual = pd.to_numeric(log["actual_value"], errors="coerce")
    baseline = pd.to_numeric(log["baseline_rate"], errors="coerce")
    over = scored & (log["direction"] == "Over")
    under = scored & (log["direction"] == "Under")

    log.loc[over & (actual > baseline), "hit"] = 1
    log.loc[over & (actual < baseline), "hit"] = 0
    log.loc[under & (actual < baseline), "hit"] = 1
    log.loc[under & (actual > baseline), "hit"] = 0
    # A real exact tie stays a real push (hit left null), per module docstring.

    log.to_csv(log_path, index=False)
    return log


def summarize_prop_results(log_path: str) -> pd.DataFrame:
    """Real, honest per-(model_version, category) hit-rate summary over
    whatever real resolved picks the log holds - plus an "all" category
    row per version, so a real overall rate is always visible without
    re-aggregating by hand.

    `n_pending` is reported alongside the real resolved counts rather
    than hidden: a real hit rate over 6 resolved picks is a very
    different thing from one over 600, and this feature is new enough
    that the honest answer will be "not enough real picks yet" for a
    while. Returns an empty, correctly-shaped frame when nothing has
    resolved yet."""
    columns = ["model_version", "category", "n_resolved", "n_hits", "hit_rate", "n_pushes", "n_pending"]
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=columns)

    log = pd.read_csv(log_path)
    if log.empty:
        return pd.DataFrame(columns=columns)

    log["hit"] = pd.to_numeric(log["hit"], errors="coerce")
    log["_resolved"] = log["actual_value"].notna()

    def _summarize(group: pd.DataFrame, version: str, category: str) -> dict:
        resolved = group[group["_resolved"]]
        scored = resolved[resolved["hit"].notna()]
        return {
            "model_version": version,
            "category": category,
            "n_resolved": len(scored),
            "n_hits": int(scored["hit"].sum()) if len(scored) else 0,
            "hit_rate": scored["hit"].mean() if len(scored) else pd.NA,
            "n_pushes": len(resolved) - len(scored),
            "n_pending": len(group) - len(resolved),
        }

    rows = []
    for version, version_group in log.groupby("model_version", dropna=False):
        rows.append(_summarize(version_group, version, "all"))
        for category, category_group in version_group.groupby("category", dropna=False):
            rows.append(_summarize(category_group, version, category))

    return pd.DataFrame(rows, columns=columns)

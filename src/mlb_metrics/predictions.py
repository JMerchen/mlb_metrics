"""Daily pick logging and outcome resolution - the core of Phase B.

Nothing in the original script ever checked whether WAVE/Game_Hit_Probability
actually predicted hits. This module maintains an append-only log of
(date, player, predicted probability, realized outcome) that evaluation.py
scores: `select_picks` turns a computed hitters table into that day's
ranked, qualified picks; `append_predictions` logs them *before* the game is
played; `resolve_predictions` fills in whether the pick actually got a hit
once that date's outcome data is available.
"""

import os

import pandas as pd

from mlb_metrics import config, helpers

PREDICTION_COLUMNS = [
    "date", "key_mlbam", "name", "rank", "predicted_probability", "metric",
    "probability", "Matchup_Hit_Probability", "actual_hit", "at_bats", "model_version",
]

# Tag applied (via the migration guards in append_predictions/resolve_predictions)
# to any row logged before the model_version column existed, or reconstructed
# by a historical replay (see git_backtest.py) - distinguishes "we don't know
# which logic produced this" from a real current-version live pick.
LEGACY_MODEL_VERSION = "legacy"

# The set of probability-like signals select_picks jointly gates on (each
# must clear min_probability), whichever of them happen to be present on the
# `hitters` table passed in - see select_picks's docstring.
JOINT_PROBABILITY_GATE_COLUMNS = ["probability", "Game_Hit_Probability", "Matchup_Hit_Probability"]


def select_picks(
    hitters: pd.DataFrame,
    date,
    top_n: int = config.BACKTEST_TOP_N,
    min_plate_appearances: int = config.BACKTEST_MIN_PLATE_APPEARANCES,
    metric: str = "Game_Hit_Probability",
    rank_metric: str | None = None,
    min_probability: float = config.HITTER_MIN_PROBABILITY,
    max_avg_batting_order: float = config.LINEUP_TOP_HALF_MAX_SLOT,
    min_start_rate: float = config.LINEUP_MIN_START_RATE,
    teams_playing_today: set[str] | None = None,
    model_version: str = config.HITTER_MODEL_VERSION,
) -> pd.DataFrame:
    """Rank a computed hitters table (the wave.csv-equivalent output of
    hitters.assemble_hitters) by `rank_metric` (defaults to `metric`) and
    return the top `top_n` qualified picks for `date`, in PREDICTION_COLUMNS
    shape with `actual_hit` left null (unresolved). `predicted_probability`
    and the logged `metric` name always come from `metric`, regardless of
    which column was used to rank - `rank_metric` only changes *which*
    qualified hitters get chosen, not what probability gets reported/scored.

    `max_avg_batting_order`/`min_start_rate` only take effect if `hitters`
    has avg_batting_order/start_rate columns (see hitters.assemble_hitters's
    optional `lineup_consistency` param) - absent columns mean a no-op, so
    old wave.csv snapshots in git history (which predate this feature) are
    unaffected. A null avg_batting_order (never started for their current
    team) fails the comparison and is correctly excluded, not treated as 0.

    `min_probability` requires EVERY column in JOINT_PROBABILITY_GATE_COLUMNS
    that's actually present on `hitters` to clear this bar (see
    config.HITTER_MIN_PROBABILITY) - a hitter can look good on one signal
    while being unreliable on another (see config.py for what each
    divergence means). On a normal day that's `probability` and
    `Game_Hit_Probability`; on a day `Matchup_Hit_Probability` has been
    merged in too (see pipeline.run), a good matchup is required just as
    much as the other two. Column-gated like the lineup qualifiers, so a
    missing column is simply skipped, not a required 0.

    `teams_playing_today`, if given, additionally requires a batter's team
    to be in the set - unlike the lineup qualifiers this isn't column-gated
    (whether a team is playing today isn't a property of the hitters table
    itself); defaults to None, i.e. off.

    `model_version` is stamped onto every returned row (see
    config.HITTER_MODEL_VERSION) so evaluation.py/the dashboard can segment
    accuracy stats by which selection logic actually produced a given pick -
    without this, a qualifier/ranking change never visibly moves the
    dashboard's stats until the (much larger) pre-change history stops
    dominating the aggregate.

    `probability` and `Matchup_Hit_Probability` are logged alongside
    `predicted_probability` (which stays Game_Hit_Probability - a real,
    calibratable per-game rate, needed as-is for Brier/log-loss scoring) so
    evaluation.py's Beat the Streak "recommended" gate can blend all three
    signals instead of thresholding Game_Hit_Probability alone (see
    evaluation._combined_probability). `Matchup_Hit_Probability` is NaN
    whenever `hitters` doesn't carry it (no schedule/matchup data that
    day, or a historical wave.csv-only replay - see git_backtest.py).
    """
    qualified = hitters[(hitters["PA_L"] + hitters["PA_R"]) >= min_plate_appearances].copy()
    if "avg_batting_order" in qualified.columns:
        qualified = qualified[qualified["avg_batting_order"] <= max_avg_batting_order]
    if "start_rate" in qualified.columns:
        qualified = qualified[qualified["start_rate"] >= min_start_rate]
    if teams_playing_today is not None:
        qualified = qualified[qualified["team"].isin(teams_playing_today)]
    for gate_column in JOINT_PROBABILITY_GATE_COLUMNS:
        if gate_column in qualified.columns:
            qualified = qualified[qualified[gate_column] >= min_probability]

    picks = qualified.sort_values(rank_metric or metric, ascending=False).head(top_n).reset_index(drop=True)

    picks["rank"] = picks.index + 1
    picks["date"] = pd.Timestamp(date)
    picks["name"] = picks["name_first"].fillna("").astype(str) + " " + picks["name_last"].fillna("").astype(str)
    picks["predicted_probability"] = picks[metric]
    picks["metric"] = metric
    for optional_column in ("probability", "Matchup_Hit_Probability"):
        if optional_column not in picks.columns:
            picks[optional_column] = pd.NA
    picks["actual_hit"] = pd.NA
    picks["at_bats"] = pd.NA
    picks["model_version"] = model_version

    return picks[PREDICTION_COLUMNS]


def append_predictions(picks: pd.DataFrame, log_path: str) -> pd.DataFrame:
    """Append `picks` to the predictions log at `log_path`, deduping on
    (date, key_mlbam, metric) so re-running a day's pipeline - or a `picks`
    batch that already contains duplicates itself, e.g. from git history
    replaying the same date via more than one commit - doesn't create
    duplicate log entries. Existing rows (including already-resolved
    actual_hit values) always win over a re-logged pick for the same key."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path, parse_dates=["date"])
        if "model_version" not in existing.columns:
            existing["model_version"] = LEGACY_MODEL_VERSION  # migrate a log written before model_version existed
        combined = pd.concat([picks, existing], ignore_index=True)
    else:
        combined = picks

    combined = combined.drop_duplicates(subset=["date", "key_mlbam", "metric"], keep="last")
    combined = combined.sort_values(["date", "rank"]).reset_index(drop=True)
    combined.to_csv(log_path, index=False)
    return combined


def resolve_predictions(log_path: str, completed_events_by_date: pd.DataFrame) -> pd.DataFrame:
    """Fill in `at_bats`/`actual_hit` for any still-pending rows in the
    predictions log whose date is covered by `completed_events_by_date`
    (columns: game_date, batter, events - e.g. the persisted raw Statcast
    data). A row is "still pending" if its `at_bats` is null - not
    `actual_hit`, since a batter can be fully resolved with zero at-bats
    (rained out, DNP, etc: at_bats=0, actual_hit stays null because there's
    no hit/miss to score) which must stay distinguishable from a date we
    simply haven't seen outcome data for yet. A row only gets resolved once
    its date is <= the latest date present anywhere in
    `completed_events_by_date` - not just once *that batter* appears in it -
    so a confirmed zero-at-bats day is never mistaken for "not checked yet".
    Already-resolved rows (at_bats already set) are left untouched."""
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=PREDICTION_COLUMNS)

    log = pd.read_csv(log_path, parse_dates=["date"])
    if log.empty:
        return log
    if "at_bats" not in log.columns:
        log["at_bats"] = pd.NA  # migrate a log written before at_bats existed
    if "model_version" not in log.columns:
        log["model_version"] = LEGACY_MODEL_VERSION  # migrate a log written before model_version existed

    events = completed_events_by_date.copy()
    events["had_hit"] = helpers.is_hit(events["events"])
    known_through = events["game_date"].max() if len(events) else pd.NaT

    per_batter_day = (
        events.groupby(["game_date", "batter"])
        .agg(resolved_at_bats=("events", "size"), resolved_hit=("had_hit", "max"))
        .reset_index()
        .rename(columns={"game_date": "date", "batter": "key_mlbam"})
    )

    log = log.merge(per_batter_day, on=["date", "key_mlbam"], how="left")

    still_pending = log["at_bats"].isna()
    knowable = pd.notna(known_through) & (log["date"] <= known_through)
    resolvable = still_pending & knowable

    resolved_at_bats = log["resolved_at_bats"].fillna(0)
    log.loc[resolvable, "at_bats"] = resolved_at_bats[resolvable]

    got_hit = resolvable & (resolved_at_bats > 0) & (log["resolved_hit"] == 1)
    got_out = resolvable & (resolved_at_bats > 0) & (log["resolved_hit"] != 1)
    log.loc[got_hit, "actual_hit"] = 1
    log.loc[got_out, "actual_hit"] = 0
    # resolvable & resolved_at_bats == 0 (no_game): at_bats is now 0, but
    # actual_hit is intentionally left null - there's no hit/miss to record.

    log = log.drop(columns=["resolved_at_bats", "resolved_hit"])
    log.to_csv(log_path, index=False)
    return log

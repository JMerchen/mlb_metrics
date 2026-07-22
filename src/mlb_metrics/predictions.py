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

PREDICTION_COLUMNS = ["date", "key_mlbam", "name", "rank", "predicted_probability", "metric", "actual_hit", "at_bats"]


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

    `min_probability` requires BOTH `probability` and `Game_Hit_Probability`
    to clear this bar (see config.HITTER_MIN_PROBABILITY) - a hitter can look
    good on one of those two signals while being unreliable on the other
    (see config.py for what each divergence means). Column-gated like the
    lineup qualifiers, so it's a no-op against a table missing either column.

    `teams_playing_today`, if given, additionally requires a batter's team
    to be in the set - unlike the lineup qualifiers this isn't column-gated
    (whether a team is playing today isn't a property of the hitters table
    itself); defaults to None, i.e. off.
    """
    qualified = hitters[(hitters["PA_L"] + hitters["PA_R"]) >= min_plate_appearances].copy()
    if "avg_batting_order" in qualified.columns:
        qualified = qualified[qualified["avg_batting_order"] <= max_avg_batting_order]
    if "start_rate" in qualified.columns:
        qualified = qualified[qualified["start_rate"] >= min_start_rate]
    if teams_playing_today is not None:
        qualified = qualified[qualified["team"].isin(teams_playing_today)]
    if {"probability", "Game_Hit_Probability"}.issubset(qualified.columns):
        qualified = qualified[
            (qualified["probability"] >= min_probability)
            & (qualified["Game_Hit_Probability"] >= min_probability)
        ]

    picks = qualified.sort_values(rank_metric or metric, ascending=False).head(top_n).reset_index(drop=True)

    picks["rank"] = picks.index + 1
    picks["date"] = pd.Timestamp(date)
    picks["name"] = picks["name_first"].fillna("").astype(str) + " " + picks["name_last"].fillna("").astype(str)
    picks["predicted_probability"] = picks[metric]
    picks["metric"] = metric
    picks["actual_hit"] = pd.NA
    picks["at_bats"] = pd.NA

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

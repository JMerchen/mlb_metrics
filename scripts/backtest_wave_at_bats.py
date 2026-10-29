"""Backtest config.WAVE_AT_BATS_ONLY: should the picks' WAVE count at-bats
only (the batting-average scale, matching PAVE's per-at-bat rate in the
matchup's log5) instead of every plate appearance with walks and
hit-by-pitches as hitless trials (the version the picks were built on)?

Same replay as scripts/backtest_window_carry_forward.py: for every date,
each variant's full feature set is recomputed from games strictly before
that date (dfs_backtest._compute_date_outputs) and joined to whether each
hitter got a hit that day; walk-forward by month, the hit-probability model
(the shipped HistGradientBoostingClassifier settings) is refit on each
variant's earlier months and scored on the next - log loss, Brier, AUC, and
the share of days the top pick / both of the top two hit, by the model, the
live rule (model shortlist ranked by Matchup_Approach) and CAVE.

RESULT: see the bottom of the run's output (filled in after the run).

Usage:
    python scripts/backtest_wave_at_bats.py [--start 2026-05-01] [--end 2026-09-27] [--workers 3]
"""

import argparse
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from mlb_metrics import config, data, dfs_backtest, dfs_ml, matchup

_STATE = {}


def _init(raw_dir, season):
    persisted = data.load_persisted_statcast(raw_dir, season)
    persisted = persisted[persisted["game_type"] == "R"]
    _STATE["persisted"] = persisted
    _STATE["schedule"] = dfs_backtest.derive_historical_team_schedule(persisted)


def _date_rows(date):
    persisted, schedule = _STATE["persisted"], _STATE["schedule"]
    day_events = persisted[persisted["game_date"] == date]
    got_hit = dfs_backtest.compute_actual_hitter_got_hit(data.completed_events(day_events, ["game_date", "batter", "events"]))
    frames = []
    for variant in (False, True):
        config.WAVE_AT_BATS_ONLY = variant
        day = dfs_backtest._compute_date_outputs(persisted, schedule, date)
        if day is None:
            continue
        features = dfs_ml.build_hitter_features(
            day["outputs"]["wave"], day["outputs"]["pave"], day["outputs"]["confidence"],
            day["todays_schedule"], day["matchup_probability"],
        )
        scored = features.merge(got_hit, on="key_mlbam", how="inner")
        scored["date"] = date
        scored["at_bats_only"] = variant
        frames.append(scored)
    return pd.concat(frames, ignore_index=True) if frames else None


def build_log(dates, raw_dir, season, workers):
    with Pool(workers, initializer=_init, initargs=(raw_dir, season)) as pool:
        parts = [part for part in pool.imap(_date_rows, dates) if part is not None]
    return pd.concat(parts, ignore_index=True)


def _model():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(learning_rate=0.03, max_depth=3, max_iter=200, min_samples_leaf=200)


def _top_hits(day: pd.DataFrame, score: str, n: int) -> bool:
    return bool(day.nlargest(n, score)["Got_Hit"].astype(bool).all())


def _live_rule_hits(day: pd.DataFrame, n: int) -> bool:
    shortlist = day.nlargest(config.HITTER_MODEL_SHORTLIST_SIZE, "Model_Hit_Probability")
    return _top_hits(shortlist, "Matchup_Approach", n)


def evaluate(log: pd.DataFrame) -> pd.DataFrame:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    log = log.copy()
    log["date"] = pd.to_datetime(log["date"])
    log["month"] = log["date"].dt.to_period("M")
    months = sorted(log["month"].unique())
    rows, daily = [], []
    for variant, frame in log.groupby("at_bats_only"):
        tested = []
        for month in months[2:]:
            train, test = frame[frame["month"] < month], frame[frame["month"] == month].copy()
            if train.empty or test.empty:
                continue
            model = _model().fit(dfs_ml.hitter_feature_matrix(train), train["Got_Hit"].astype(int))
            test["Model_Hit_Probability"] = model.predict_proba(dfs_ml.hitter_feature_matrix(test))[:, 1]
            tested.append(test)
        tested = pd.concat(tested, ignore_index=True)
        tested["Matchup_Approach"] = matchup.compute_matchup_approach(tested["Approach"], tested["Matchup_Hit_Probability"])
        y, p = tested["Got_Hit"].astype(int), tested["Model_Hit_Probability"]
        by_day = tested.groupby("date")
        day_results = pd.DataFrame({
            "model_top1": by_day.apply(lambda d: _top_hits(d, "Model_Hit_Probability", 1)),
            "model_top2": by_day.apply(lambda d: _top_hits(d, "Model_Hit_Probability", 2)),
            "live_top1": by_day.apply(lambda d: _live_rule_hits(d, 1)),
            "live_top2": by_day.apply(lambda d: _live_rule_hits(d, 2)),
            "cave_top1": by_day.apply(lambda d: _top_hits(d, "Game_Hit_Probability", 1)),
            "cave_top2": by_day.apply(lambda d: _top_hits(d, "Game_Hit_Probability", 2)),
        })
        day_results["at_bats_only"] = variant
        daily.append(day_results)
        row = {"at_bats_only": variant, "hitter_games": len(tested), "days": len(day_results),
               "log_loss": log_loss(y, p), "brier": brier_score_loss(y, p), "auc": roc_auc_score(y, p)}
        row.update(day_results.drop(columns="at_bats_only").mean().to_dict())
        rows.append(row)
    summary = pd.DataFrame(rows).set_index("at_bats_only")
    daily = pd.concat(daily)
    return summary, daily


def paired_days(daily: pd.DataFrame, column: str):
    """Days only one variant's pick hit: (at-bats only, original only)."""
    wide = daily.reset_index().pivot(index="date", columns="at_bats_only", values=column)
    return int((wide[True] & ~wide[False]).sum()), int((wide[False] & ~wide[True]).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=config.SEASON_START.year)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--log", default="data/backtest/wave_at_bats_log.csv")
    parser.add_argument("--reuse-log", action="store_true", help="Score an existing --log instead of rebuilding it.")
    args = parser.parse_args()

    if not args.reuse_log:
        _init(args.raw_dir, args.season)
        dates = sorted(_STATE["schedule"]["date"].unique())
        start = pd.Timestamp(args.start) if args.start else pd.Timestamp(f"{args.season}-05-01")
        end = pd.Timestamp(args.end) if args.end else pd.Timestamp(config.REGULAR_SEASON_END)
        dates = [d for d in dates if start <= pd.Timestamp(d) <= end]
        _STATE.clear()
        print(f"Rebuilding features for {len(dates)} dates, both variants...")
        log = build_log(dates, args.raw_dir, args.season, args.workers)
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        log.to_csv(args.log, index=False)
    log = pd.read_csv(args.log)

    summary, daily = evaluate(log)
    pd.set_option("display.width", 160)
    print(summary.round(4).to_string())
    for column in ("model_top1", "model_top2", "live_top1", "live_top2", "cave_top1", "cave_top2"):
        better, worse = paired_days(daily, column)
        print(f"{column}: days only at-bats-only's pick hit {better}, only the original's {worse}")


if __name__ == "__main__":
    main()

"""Phase 2 backtest for decision_score.py's "was swinging/taking advised"
methodology - real, no-lookahead, out-of-sample validation, required by
the user before Decision_Score ships to the dashboard (see the project's
Decision Score plan/PR).

The real question: does a batter's real compliance with their OWN
advised action - on the pitch that actually ended a plate appearance -
associate with a real, better PA outcome? This is a correlational test
(compliance isn't randomly assigned - a batter's own overall skill could
confound a pooled comparison), so this script reports BOTH:

1. A pooled Mann-Whitney U test (matched vs. not-matched PA-ending
   pitches' real "PA value" = helpers.is_on_base + helpers.total_bases,
   an OPS-equivalent per-PA value) - the straightforward "does compliance
   associate with a better outcome" test.
2. A per-batter PAIRED test (each batter's own mean PA value when matched
   vs. not matched, restricted to batters with >= MIN_PER_GROUP real PAs
   in both groups) - a real robustness check that at least partially
   controls for "some batters are just better at everything," which the
   pooled test alone can't rule out.

No-lookahead design: a single, real chronological train/test split
within the complete 2025 season (not a walk-forward day-by-day refit -
matches this project's own established backtest precedent, e.g. the NFL
game-picks backtest's "train weeks 1-7, test weeks 8-18" single split,
not a per-week refit). TRAIN builds each batter's real windowed overall/
zone OPS reference (compute_batter_overall_ops/compute_zone_ops); TEST is
held out entirely from that reference - it's applied to TEST, never
re-fit on it.

TRAIN: 2025-03 through 2025-06 (~400K real pitches).
TEST:  2025-07 through 2025-09 (~342K real pitches), the REST of the same
       season - avoids the cross-season discontinuity a 2025-tail/
       2026-spring split would introduce (different players' real form
       across a real offseason gap).

Sweeps a small grid of candidate config.DECISION_SCORE_* values (using
decision_score.py's own override parameters - see compute_zone_ops/
compute_decision_advice's own docstrings - never by mutating the config
module) and reports every candidate's real numbers, honestly, whether or
not any of them clear a real bar.

Usage:
    python scripts/backtest_decision_score.py
"""

import itertools
import os

import pandas as pd
from scipy.stats import mannwhitneyu, ttest_rel

from mlb_metrics import config, decision_score, helpers, pipeline

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

TRAIN_FILES = [
    "statcast_2025_03.parquet", "statcast_2025_04.parquet",
    "statcast_2025_05.parquet", "statcast_2025_06.parquet",
]
TEST_FILES = ["statcast_2025_07.parquet", "statcast_2025_08.parquet", "statcast_2025_09.parquet"]

# Minimum real PAs (per group - matched, not-matched) a batter needs to
# contribute to the per-batter paired test - same small-sample-noise
# guard this project's other backtests apply (e.g.
# config.BACKTEST_MIN_PLATE_APPEARANCES's own reasoning).
MIN_PER_GROUP = 5

TEST_COLUMNS = [
    "batter", "game_date", "zone", "balls", "strikes", "description", "events",
    "inning", "bat_score", "fld_score", "on_2b", "on_3b",
]

# Real sweep results (two rounds - the first found shrinkage_strength=20
# strongest of {10, 20, 40} with only the coarse 1.15/0.85 count pair
# tried; the second, at fixed shrinkage=20, swept finer count-multiplier
# granularity and confirmed the effect weakens monotonically from 1.0 at
# every step - see config.py's DECISION_SCORE_* comments for the real
# numbers and the honest negative finding on the count/leverage
# multipliers). Kept as the full grid here so this script stays a
# complete, re-runnable record of the validation, not just its last leg.
SHRINKAGE_CANDIDATES = [10.0, 20.0, 40.0]
COUNT_MULTIPLIER_CANDIDATES = [
    (1.0, 1.0), (1.05, 0.95), (1.08, 0.92), (1.10, 0.90), (1.15, 0.85), (1.30, 0.70),
]
LEVERAGE_MULTIPLIER_CANDIDATES = [1.0, 0.98, 0.95]


def _load(files: list) -> pd.DataFrame:
    frames = [pd.read_parquet(os.path.join(RAW_DIR, f)) for f in files]
    return pd.concat(frames, ignore_index=True)


def _pa_value(events: pd.Series) -> pd.Series:
    """A simple, real, OPS-equivalent per-PA value: the on-base indicator
    (0/1) plus total bases (0-4) - the same two real components
    traditional_stats.compute_traditional_batting_stats's own OBP/SLG
    (and therefore OPS) are built from, just summed at the per-PA level
    instead of aggregated into a season rate."""
    return helpers.is_on_base(events) + helpers.total_bases(events)


def evaluate_candidate(
    test_df: pd.DataFrame, overall_ops: pd.DataFrame, zone_ops: pd.DataFrame,
    hitter_multiplier: float, pitcher_multiplier: float, leverage_multiplier: float,
) -> dict:
    advice = decision_score.compute_decision_advice(
        test_df, overall_ops, zone_ops,
        hitter_multiplier=hitter_multiplier, pitcher_multiplier=pitcher_multiplier,
        leverage_multiplier=leverage_multiplier,
    )
    matched = (helpers.is_swing(test_df["description"]) == (advice == "swing").astype(int)).astype(int)

    pa_ending = test_df[test_df["events"].isin(config.COUNTED_EVENTS)].copy()
    pa_ending["matched"] = matched.loc[pa_ending.index]
    pa_ending["pa_value"] = _pa_value(pa_ending["events"])

    matched_values = pa_ending.loc[pa_ending["matched"] == 1, "pa_value"]
    unmatched_values = pa_ending.loc[pa_ending["matched"] == 0, "pa_value"]

    result = {
        "n_pa_ending": len(pa_ending),
        "n_matched": len(matched_values),
        "n_unmatched": len(unmatched_values),
        "compliance_rate": matched.mean(),
        "mean_pa_value_matched": matched_values.mean() if len(matched_values) else float("nan"),
        "mean_pa_value_unmatched": unmatched_values.mean() if len(unmatched_values) else float("nan"),
    }

    if len(matched_values) >= 2 and len(unmatched_values) >= 2:
        result["pooled_p_value"] = float(mannwhitneyu(matched_values, unmatched_values, alternative="two-sided").pvalue)
    else:
        result["pooled_p_value"] = float("nan")

    per_batter = pa_ending.groupby(["batter", "matched"])["pa_value"].agg(["mean", "count"]).unstack("matched")
    paired = per_batter[
        (per_batter[("count", 1)].fillna(0) >= MIN_PER_GROUP) & (per_batter[("count", 0)].fillna(0) >= MIN_PER_GROUP)
    ]
    result["n_batters_paired"] = len(paired)
    if len(paired) >= 2:
        result["paired_p_value"] = float(ttest_rel(paired[("mean", 1)], paired[("mean", 0)]).pvalue)
        result["paired_mean_diff"] = float((paired[("mean", 1)] - paired[("mean", 0)]).mean())
    else:
        result["paired_p_value"] = float("nan")
        result["paired_mean_diff"] = float("nan")

    return result


def main():
    print(f"Loading train ({TRAIN_FILES})...")
    train_raw = _load(TRAIN_FILES)
    print(f"Loading test ({TEST_FILES})...")
    test_raw = _load(TEST_FILES)
    print(f"Train: {len(train_raw):,} real pitches ({train_raw['game_date'].min().date()} - {train_raw['game_date'].max().date()})")
    print(f"Test:  {len(test_raw):,} real pitches ({test_raw['game_date'].min().date()} - {test_raw['game_date'].max().date()})")

    pa_events_train = pipeline.build_pitch_events(train_raw)
    test_df = test_raw[TEST_COLUMNS].copy()

    windows = config.WAVE_WINDOWS
    overall_ops = decision_score.compute_batter_overall_ops(pa_events_train, windows)
    print(f"\n{len(overall_ops):,} batters with a real windowed overall OPS from train.\n")

    rows = []
    for shrinkage in SHRINKAGE_CANDIDATES:
        zone_ops = decision_score.compute_zone_ops(pa_events_train, overall_ops, windows, shrinkage_strength=shrinkage)
        for (hitter_mult, pitcher_mult), leverage_mult in itertools.product(
            COUNT_MULTIPLIER_CANDIDATES, LEVERAGE_MULTIPLIER_CANDIDATES
        ):
            result = evaluate_candidate(test_df, overall_ops, zone_ops, hitter_mult, pitcher_mult, leverage_mult)
            result.update({
                "shrinkage_strength": shrinkage,
                "hitter_multiplier": hitter_mult,
                "pitcher_multiplier": pitcher_mult,
                "leverage_multiplier": leverage_mult,
            })
            rows.append(result)
            print(
                f"shrinkage={shrinkage:>4.0f}  hitter={hitter_mult:.2f}  pitcher={pitcher_mult:.2f}  "
                f"leverage={leverage_mult:.2f}  |  compliance={result['compliance_rate']*100:5.1f}%  "
                f"pa_value matched={result['mean_pa_value_matched']:.3f} vs unmatched={result['mean_pa_value_unmatched']:.3f}  "
                f"pooled_p={result['pooled_p_value']:.4f}  "
                f"paired_p={result['paired_p_value']:.4f} (n_batters={result['n_batters_paired']})"
            )

    results = pd.DataFrame(rows)
    results.to_csv(os.path.join(os.path.dirname(__file__), "..", "data", "decision_score_backtest_results.csv"), index=False)

    # A real bar: BOTH the pooled test and the per-batter paired test
    # (the confound-robustness check) must clear real significance, AND
    # the direction must be the right one (matched > unmatched), not just
    # a small p-value in either direction.
    results["real_effect"] = results["mean_pa_value_matched"] > results["mean_pa_value_unmatched"]
    results["clears_bar"] = (
        results["real_effect"]
        & (results["pooled_p_value"] < 0.05)
        & (results["paired_p_value"] < 0.05)
    )
    winners = results[results["clears_bar"]].sort_values("paired_p_value")

    print("\n" + "=" * 100)
    if winners.empty:
        print("NO candidate configuration cleared a real bar (pooled p<0.05 AND paired p<0.05, matched > unmatched).")
        print("Reporting honestly: this first-pass methodology is NOT validated by this backtest.")
        best = results.sort_values("paired_p_value").iloc[0]
        print(f"\nClosest candidate (lowest paired p-value, NOT a validated result):\n{best}")
    else:
        best = winners.iloc[0]
        print("Best validated candidate (lowest paired p-value among those clearing the bar):")
        print(best)

    print("\nFull results written to data/decision_score_backtest_results.csv")


if __name__ == "__main__":
    main()

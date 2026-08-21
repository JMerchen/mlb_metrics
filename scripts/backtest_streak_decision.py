"""Real go/no-go backtest for quant-analytics item #4, slice 1's
streak-aware optimal-stopping "play or sit" rule (decision_theory.py) -
does treating the daily pick decision as a real finite-horizon dynamic
program (streak- and days-remaining-aware) beat today's fixed
`config.DAILY_PICK_MIN_PROBABILITY=0.77` bar on real historical data?

Unlike every prior slice, this needs NO GitHub Actions dispatch and NO
network access: `data/predictions/predictions.csv` is already checked
into the repo, real, resolved history - readable locally.

Real data pipeline: `evaluation.graded_daily_picks` (max_picks=
`config.DAILY_PICK_MAX`, min_probability=0.0 - every logged candidate day,
not pre-filtered by today's own status-quo bar) gives every day's real
candidate picks and their real `combined_probability`; collapsed to one
row per day (mean `combined_probability` as that day's `p`, real
picks-that-hit count, whether any pick missed). `p` is treated as a
single real number per day - deliberately NOT an attempt at a true joint
probability of two correlated picks both hitting (item #4's OTHER,
separate sub-problem, explicitly out of scope here - see
decision_theory.py's own module docstring).

Two decision rules replay the SAME real historical day sequence:
- status quo: play iff `p >= config.DAILY_PICK_MIN_PROBABILITY`.
- DP rule: `decision_theory.should_play`, solved once against this same
  real historical `p` sample and a real empirical `gain` (average
  picks-that-hit on a real non-reset day, `decision_theory.estimate_gain`
  against the real `evaluation.streak_progression` history - not assumed
  to always be `DAILY_PICK_MAX`).

Reports both rules' real single-path outcome over the actual historical
order, AND a bootstrap (resampling real (p, outcome) day-pairs WITH
replacement, `--bootstrap-samples` times) - this intentionally reshuffles
day order (a fresh draw of horizon-many real day-pairs, not an
order-preserving resample), consistent with the DP's own stated
days-are-i.i.d. simplification. Real go/no-go bar: does the DP rule's
mean bootstrapped final streak beat the status quo's - reported honestly
either direction, the same discipline as item #3's slices.

Usage:
    python scripts/backtest_streak_decision.py
    python scripts/backtest_streak_decision.py --bootstrap-samples 5000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from mlb_metrics import config, decision_theory, evaluation


def _load_day_level(predictions_path: str) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path, parse_dates=["date"])
    graded = evaluation.graded_daily_picks(
        predictions, metric=None, max_picks=config.DAILY_PICK_MAX, min_probability=0.0
    )
    graded["outcome"] = evaluation._classify_outcome(graded)
    day_level = (
        graded.groupby("date")
        .agg(
            p=("combined_probability", "mean"),
            n_hit=("outcome", lambda s: (s == "hit").sum()),
            any_miss=("outcome", lambda s: (s == "miss").any()),
            any_pending=("outcome", lambda s: (s == "pending").any()),
        )
        .reset_index()
    )
    # Matches streak_progression's own rule: a day isn't processed at all
    # until every pick logged for it is resolved.
    day_level = day_level[~day_level["any_pending"]].sort_values("date").reset_index(drop=True)
    return day_level


def _simulate(day_level: pd.DataFrame, decide_fn) -> dict:
    """decide_fn(streak, days_remaining, p) -> bool (play today?). Walks
    the real day sequence forward once, applying the REAL logged outcome
    on any day the rule chooses to play."""
    streak = 0
    longest = 0
    n_played = 0
    n = len(day_level)
    for i, row in day_level.iterrows():
        days_remaining = n - i
        if decide_fn(streak, days_remaining, row["p"]):
            n_played += 1
            streak = 0 if row["any_miss"] else streak + row["n_hit"]
        longest = max(longest, streak)
    return {"final_streak": streak, "longest_streak": longest, "n_played": n_played, "n_days": n}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--predictions-path", default="data/predictions/predictions.csv")
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    day_level = _load_day_level(args.predictions_path)
    if day_level.empty:
        print("No resolved days in the predictions log - nothing to backtest.")
        return
    print(f"Loaded {len(day_level)} resolved days from {args.predictions_path}")

    raw_predictions = pd.read_csv(args.predictions_path, parse_dates=["date"])
    status_quo_progression = evaluation.streak_progression(
        raw_predictions, max_picks=config.DAILY_PICK_MAX, min_probability=config.DAILY_PICK_MIN_PROBABILITY
    )
    gain = decision_theory.estimate_gain(status_quo_progression)
    print(f"Real empirical gain (avg picks-that-hit on a real non-reset day): {gain:.4f}")
    if gain <= 0:
        print("No real non-reset history to estimate a gain from - can't solve the DP. Nothing further to report.")
        return

    horizon = len(day_level)
    solved = decision_theory.solve_reservation_thresholds(day_level["p"].tolist(), horizon=horizon, gain=gain)

    def status_quo_decide(streak, days_remaining, p):
        return p >= config.DAILY_PICK_MIN_PROBABILITY

    def dp_decide(streak, days_remaining, p):
        return decision_theory.should_play(streak, days_remaining, p, solved)

    sq = _simulate(day_level, status_quo_decide)
    dp = _simulate(day_level, dp_decide)

    print(f"\n=== Real single-path replay over {horizon} resolved days ===")
    print(
        f"  Status quo (play iff p>={config.DAILY_PICK_MIN_PROBABILITY}): "
        f"final_streak={sq['final_streak']}, longest_streak={sq['longest_streak']}, "
        f"days_played={sq['n_played']}/{horizon}"
    )
    print(
        f"  DP rule (streak/horizon-aware):                 "
        f"final_streak={dp['final_streak']}, longest_streak={dp['longest_streak']}, "
        f"days_played={dp['n_played']}/{horizon}"
    )

    rng = np.random.RandomState(args.seed)
    sq_finals, dp_finals = [], []
    for _ in range(args.bootstrap_samples):
        idx = rng.randint(0, horizon, size=horizon)
        resampled = day_level.iloc[idx].reset_index(drop=True)
        sq_finals.append(_simulate(resampled, status_quo_decide)["final_streak"])
        dp_finals.append(_simulate(resampled, dp_decide)["final_streak"])

    print(f"\n=== Bootstrap ({args.bootstrap_samples} resamples of the real historical (p, outcome) day-pairs) ===")
    print(f"  Status quo: mean final streak={np.mean(sq_finals):.3f}, median={np.median(sq_finals):.1f}")
    print(f"  DP rule:    mean final streak={np.mean(dp_finals):.3f}, median={np.median(dp_finals):.1f}")
    dp_wins = sum(1 for a, b in zip(dp_finals, sq_finals) if a > b)
    ties = sum(1 for a, b in zip(dp_finals, sq_finals) if a == b)
    print(f"  DP beats status quo in {dp_wins}/{args.bootstrap_samples} resamples ({ties} ties)")

    if np.mean(dp_finals) > np.mean(sq_finals):
        print(
            "  -> GO: the DP rule shows a real improvement on this bootstrap - "
            "worth further validation before any live wiring (see decision_theory.py's non-goals)."
        )
    else:
        print(
            "  -> NO-GO (or inconclusive): the DP rule does not show a real improvement "
            "on this real (thin) history - reported honestly."
        )


if __name__ == "__main__":
    main()

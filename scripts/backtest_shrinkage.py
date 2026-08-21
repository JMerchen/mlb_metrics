"""Real go/no-go backtest for quant-analytics item #3, slice 1's Bayesian-
shrinkage constants (config.WAVE_SHRINKAGE_STRENGTH,
config.GAME_HIT_PROB_SHRINKAGE_STRENGTH) - see helpers.shrink_rate and
hitters.compute_wave/compute_game_hit_probability's own docstrings for the
real mechanism this validates.

For each candidate strength in a small grid, monkeypatches
config.WAVE_SHRINKAGE_STRENGTH/config.GAME_HIT_PROB_SHRINKAGE_STRENGTH for
the duration of one full dfs_backtest.assemble_hitter_hit_log(...) call -
the SAME real no-lookahead historical reconstruction
scripts/train_hitter_hit_model.py already uses. This is the simplest way
to sweep a config-driven constant through the WHOLE pipeline.compute_outputs
chain (hitters.assemble_hitters, compute_wave, compute_game_hit_probability)
without threading a new parameter through every intermediate function
signature - those functions only ever read the live config value in
production too, exactly what this reproduces per grid point. The original
config values are restored in a `finally` block regardless of outcome, so
a crash mid-sweep can never leave a stale monkeypatched value behind.

Scores BOTH the full unfiltered population (every hitter-date in the log,
including the sub-BACKTEST_MIN_PLATE_APPEARANCES rows that are excluded
from live picks today - this most directly demonstrates shrinkage's real
effect, since those are exactly the small-sample rows item #3's own
headline example names) AND the PA-gated subset (Total_PA >=
config.BACKTEST_MIN_PLATE_APPEARANCES - the population actually exposed
to live Beat the Streak picks today, unaffected by this slice's explicit
decision not to remove that hard gate). The real go/no-go bar for a
nonzero live default is the GATED subset beating strength=0's own
log_loss by a real margin - that's the population this change would
actually affect once shipped; the unfiltered numbers are reported for
context, not as the ship/no-ship bar.

WAVE_SHRINKAGE_STRENGTH and GAME_HIT_PROB_SHRINKAGE_STRENGTH are swept
INDEPENDENTLY (the other stays at its own live config default for every
run in a given sweep), not jointly - a joint grid would be
len(wave_grid) * len(game_grid) real historical reconstructions, expensive
for a marginal benefit over validating each signal's own shrinkage effect
in isolation first.

Needs data/raw/statcast_<season>.parquet (see scripts/wave.py).

Usage:
    python scripts/backtest_shrinkage.py
    python scripts/backtest_shrinkage.py --season 2026 --wave-grid 0,10,25,50,100
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, dfs_backtest, ml_models

DEFAULT_GRID = [0.0, 10.0, 25.0, 50.0, 100.0]


def _score(rows, column):
    if rows.empty:
        return None
    return ml_models.evaluate_classifier_predictions(rows["Got_Hit"], rows[column].clip(0, 1))


def _sweep(raw_dir: str, season: int, days: int | None, grid: list[float], target: str) -> dict:
    """target is "wave" or "game_hit_prob" - selects which config constant
    this sweep varies and which resulting column gets scored. Returns
    {strength: {"full": scored_dict_or_None, "gated": scored_dict_or_None}}."""
    column = "probability" if target == "wave" else "Game_Hit_Probability"
    results = {}
    original_wave = config.WAVE_SHRINKAGE_STRENGTH
    original_game = config.GAME_HIT_PROB_SHRINKAGE_STRENGTH
    try:
        for strength in grid:
            if target == "wave":
                config.WAVE_SHRINKAGE_STRENGTH = strength
            else:
                config.GAME_HIT_PROB_SHRINKAGE_STRENGTH = strength

            rows = dfs_backtest.assemble_hitter_hit_log(raw_dir, season=season, days=days)
            gated = rows[rows["Total_PA"] >= config.BACKTEST_MIN_PLATE_APPEARANCES] if not rows.empty else rows
            results[strength] = {"full": _score(rows, column), "gated": _score(gated, column)}
    finally:
        config.WAVE_SHRINKAGE_STRENGTH = original_wave
        config.GAME_HIT_PROB_SHRINKAGE_STRENGTH = original_game
    return results


def _print_scored(label: str, scored: dict | None) -> None:
    if scored is None:
        print(f"    {label}: no scored rows")
        return
    print(
        f"    {label}: log_loss={scored['log_loss']:.4f}, brier={scored['brier_score']:.4f}, "
        f"roc_auc={scored['roc_auc']:.4f}, accuracy={scored['accuracy']:.4f} (n={scored['n']})"
    )


def report(title: str, results: dict) -> None:
    print(f"\n=== {title} ===")
    for strength, by_population in results.items():
        print(f"  strength={strength}")
        _print_scored("full population", by_population["full"])
        _print_scored("PA-gated (live-eligible) population", by_population["gated"])

    baseline = results.get(0.0, {}).get("gated")
    if baseline is None:
        print("  -> strength=0 baseline unscored on the gated population - cannot judge go/no-go")
        return

    candidates = {
        s: r["gated"] for s, r in results.items()
        if s != 0.0 and r["gated"] is not None
    }
    if not candidates:
        print("  -> no nonzero candidate produced scored rows on the gated population")
        return

    best_strength = min(candidates, key=lambda s: candidates[s]["log_loss"])
    if candidates[best_strength]["log_loss"] < baseline["log_loss"]:
        print(
            f"  -> GO: strength={best_strength} beats strength=0 (unshrunk) on the PA-gated "
            f"population's log_loss ({candidates[best_strength]['log_loss']:.4f} vs. "
            f"{baseline['log_loss']:.4f}) - candidate for a nonzero live default"
        )
    else:
        print(
            f"  -> NO-GO: strength=0 (unshrunk) remains best on the PA-gated population's log_loss "
            f"({baseline['log_loss']:.4f}) - reported honestly, no live default earned on this holdout"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument(
        "--days", type=int, default=None,
        help="Trim to the most recent N dates instead of the full persisted history - each date "
             "recomputes the full pipeline once PER grid value, so this is expensive. Full history "
             "is the default (None), matching train_hitter_hit_model.py's own --days flag.",
    )
    parser.add_argument("--wave-grid", default=",".join(str(x) for x in DEFAULT_GRID))
    parser.add_argument("--game-hit-prob-grid", default=",".join(str(x) for x in DEFAULT_GRID))
    args = parser.parse_args()

    season = args.season or config.SEASON_START.year
    wave_grid = [float(x) for x in args.wave_grid.split(",")]
    game_grid = [float(x) for x in args.game_hit_prob_grid.split(",")]

    print(f"Sweeping WAVE_SHRINKAGE_STRENGTH (season={season}, days={args.days or 'all'})...")
    wave_results = _sweep(args.raw_dir, season, args.days, wave_grid, "wave")
    report("WAVE shrinkage sweep (probability column vs. real Got_Hit)", wave_results)

    print(f"\nSweeping GAME_HIT_PROB_SHRINKAGE_STRENGTH (season={season}, days={args.days or 'all'})...")
    game_results = _sweep(args.raw_dir, season, args.days, game_grid, "game_hit_prob")
    report("Game_Hit_Probability shrinkage sweep (vs. real Got_Hit)", game_results)


if __name__ == "__main__":
    main()

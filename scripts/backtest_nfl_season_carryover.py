"""Real, no-lookahead, MULTI-SEASON backtest for the season-carryover
shrinkage + home-field-advantage + composite-reweighting follow-up
(2026-09-04 - "every season should only carry over a portion of the
team's score from the previous season... weeks 1-6... calibrating the
features to that individual season" / "a little push or pull from
home/away" / "it comes down to offensive efficiency, defensive
efficiency, and turnover ratio, for the most part").

Uses nfl_game_picks_backtest.build_multi_season_history/
score_multi_season_snapshots (NOT replay_season, which only ever replays
ONE season in isolation with no real prior-season data fed in at all, so
it cannot exercise nfl_team_strength._season_aware_blend's cross-season
carryover mechanism at all) across the real, complete 2016-2025 dataset
(all 10 seasons now cached locally, including play-by-play - see PR #98).
Real weeks 1 and 2 of every season after 2016 ARE replayed here (unlike
replay_season's own week 1/2 exclusion) - a real prior season's history
is available, so there's a real prediction to make.

Grid design (kept deliberately bounded, not exhaustive - same "hand-tune
candidates, then honestly sweep" discipline as
scripts/backtest_decision_score.py): the EXPENSIVE step
(nfl_team_strength.assemble_team_metrics/compute_qb_continuity_adjustment,
rebuilt every real replayed week) only depends on
(carryover_regression, carryover_prior_strength) - build_multi_season_history
is called ONCE per pair and cached; composite_weights/home_field_weight
only affect the CHEAP nfl_game_picks.compute_game_win_probabilities
re-score step (score_multi_season_snapshots), swept freely on top without
re-running team-strength assembly.

**The real bar, reported honestly either way**: a candidate must beat the
TRUE baseline (season_aware=False - today's actual flat-concat, no-
home-field, live-composite-weights live behavior) on BOTH the overall
replay AND, specifically, the weeks-1-6-of-each-season slice - the exact
target this whole feature exists for. Winning only on the full season
while losing early weeks (or vice versa) does not clear the bar. Nothing
in config.py is touched unless a real candidate clears it.

Usage:
    python scripts/backtest_nfl_season_carryover.py
"""

import itertools
import os

import pandas as pd

from mlb_metrics import config, nfl_game_picks_backtest as backtest

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nfl")
SEASONS = list(range(2016, 2026))  # 2016-2025, all real cached data (incl. pbp - PR #98)

EARLY_WEEKS = set(range(1, 7))  # weeks 1-6 - the exact slice this feature targets

REGRESSION_CANDIDATES = [0.3, 0.5, 0.7]
PRIOR_STRENGTH_CANDIDATES = [3.0, 6.0, 10.0]
HOME_FIELD_CANDIDATES = [0.0, 0.02, 0.05]
COMPOSITE_CANDIDATES = {
    "live": config.NFL_GAME_PICK_COMPOSITE_WEIGHTS,
    "core_only": config.NFL_GAME_PICK_COMPOSITE_WEIGHTS_CORE_ONLY,
    "core_heavy": config.NFL_GAME_PICK_COMPOSITE_WEIGHTS_CORE_HEAVY,
}


def _load(table: str, season: int) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(RAW_DIR, f"{table}_{season}.parquet"))


def _load_all(table: str) -> pd.DataFrame:
    return pd.concat([_load(table, season) for season in SEASONS], ignore_index=True)


def _score(replay: pd.DataFrame, label: str) -> dict:
    """One real scored row for `replay` (already real REPLAY_COLUMNS
    shape) - overall AND weeks-1-6-only, model vs. market vs.
    beat-closing-line, reusing backtest.score_predictions/beat_closing_line_rate
    directly (no reimplementation)."""
    early = replay[replay["week"].isin(EARLY_WEEKS)]

    def _row(scope_name: str, scope_df: pd.DataFrame) -> dict:
        model = backtest.score_predictions(scope_df, "home_win_probability")
        market = backtest.score_predictions(scope_df, "market_home_win_probability")
        closing = backtest.beat_closing_line_rate(scope_df)
        return {
            "candidate": label, "scope": scope_name,
            "n": model["n"], "model_accuracy": model["accuracy"], "model_log_loss": model["log_loss"],
            "model_brier": model["brier_score"],
            "market_accuracy": market["accuracy"], "market_log_loss": market["log_loss"],
            "beat_closing_line_rate": closing["rate"], "beat_closing_line_n": closing["n_compared"],
        }

    return [_row("overall", replay), _row("weeks_1_6", early)]


def main():
    print(f"Loading real {SEASONS[0]}-{SEASONS[-1]} NFL data (schedules/team_stats/weekly/snap_counts/rosters/pbp)...")
    schedules = _load_all("schedules")
    team_stats = _load_all("team_stats")
    weekly = _load_all("weekly")
    snap_counts = _load_all("snap_counts")
    rosters = _load_all("rosters_weekly")
    pbp = _load_all("pbp")

    all_rows = []

    print("\n=== Baseline: today's real live behavior (no season-carryover, no home-field, live composite weights) ===")
    baseline_replay = backtest.replay_multi_season(
        schedules, team_stats, weekly, snap_counts, rosters, pbp, SEASONS,
        composite_weights=config.NFL_GAME_PICK_COMPOSITE_WEIGHTS, home_field_weight=0.0, season_aware=False,
    )
    print(f"  {len(baseline_replay):,} real replayed games.")
    all_rows.extend(_score(baseline_replay, "baseline"))

    print("\n=== Season-carryover x home-field x composite-weight grid ===")
    for regression, prior_strength in itertools.product(REGRESSION_CANDIDATES, PRIOR_STRENGTH_CANDIDATES):
        print(f"\nBuilding real history for regression={regression}, prior_strength={prior_strength} "
              f"(expensive - rebuilds team strength for every real replayed week)...")
        snapshots = backtest.build_multi_season_history(
            schedules, team_stats, weekly, snap_counts, rosters, pbp, SEASONS,
            carryover_regression=regression, carryover_prior_strength=prior_strength,
        )
        print(f"  {len(snapshots)} real replayed weeks.")

        for home_field, (composite_name, composite_weights) in itertools.product(
            HOME_FIELD_CANDIDATES, COMPOSITE_CANDIDATES.items()
        ):
            replay = backtest.score_multi_season_snapshots(
                snapshots, composite_weights=composite_weights, home_field_weight=home_field
            )
            label = f"reg={regression}_prior={prior_strength}_home={home_field}_wt={composite_name}"
            rows = _score(replay, label)
            for row in rows:
                row.update({
                    "regression": regression, "prior_strength": prior_strength,
                    "home_field_weight": home_field, "composite": composite_name,
                })
            all_rows.extend(rows)
            overall = rows[0]
            print(
                f"  {label}: overall acc={overall['model_accuracy']:.3f} "
                f"log_loss={overall['model_log_loss']:.4f} beat_line={overall['beat_closing_line_rate']:.3f}"
            )

    results = pd.DataFrame(all_rows)
    results.to_csv(os.path.join(os.path.dirname(__file__), "..", "data", "nfl_season_carryover_backtest_results.csv"), index=False)

    baseline_overall = results[(results["candidate"] == "baseline") & (results["scope"] == "overall")].iloc[0]
    baseline_early = results[(results["candidate"] == "baseline") & (results["scope"] == "weeks_1_6")].iloc[0]
    print("\n" + "=" * 100)
    print(f"BASELINE - overall: acc={baseline_overall['model_accuracy']:.3f} log_loss={baseline_overall['model_log_loss']:.4f}")
    print(f"BASELINE - weeks 1-6: acc={baseline_early['model_accuracy']:.3f} log_loss={baseline_early['model_log_loss']:.4f} (n={baseline_early['n']})")

    candidates = results[results["candidate"] != "baseline"]
    overall = candidates[candidates["scope"] == "overall"].set_index("candidate")
    early = candidates[candidates["scope"] == "weeks_1_6"].set_index("candidate")

    # The real bar: beats baseline log_loss on BOTH scopes - the whole
    # point is fixing weeks 1-6 without wrecking the rest of the season.
    clears_bar = overall.index[
        (overall["model_log_loss"] < baseline_overall["model_log_loss"])
        & (early.loc[overall.index, "model_log_loss"] < baseline_early["model_log_loss"])
    ]

    print(f"\n{len(clears_bar)} of {len(overall)} candidates beat the baseline's log_loss on BOTH overall AND weeks 1-6.")
    if len(clears_bar) == 0:
        print("NO candidate configuration cleared a real bar - reporting honestly: this backtest does NOT")
        print("validate shipping the season-carryover/home-field/composite changes as tested. config.py is")
        print("NOT being changed based on this run.")
        best = early.loc[overall.index].sort_values("model_log_loss").iloc[0]
        print(f"\nClosest candidate by weeks-1-6 log_loss (NOT validated):\n{best}")
    else:
        winner = early.loc[clears_bar].sort_values("model_log_loss").iloc[0]
        print(f"\nBest validated candidate (lowest weeks-1-6 log_loss among those clearing the bar):\n{winner}")
        print(f"\nSame candidate's overall numbers:\n{overall.loc[winner.name]}")

    print("\nFull results written to data/nfl_season_carryover_backtest_results.csv")


if __name__ == "__main__":
    main()

"""Real, no-lookahead, multi-season backtest for the opponent-adjustment
follow-up (2026-09-11 - "the way that we figure offensive and defensive
efficiency and strength for the nfl needs to be with respect to how
other teams did... if a team only throws for 200 yards in a game...
but if they were facing a historically good pass defense, that might
actually indicate a really good pass attack").

`nfl_team_strength.compute_team_offense_defense_edge` now nets each real
game's own_epa/epa_allowed against the SPECIFIC opponent's own known
prior baseline heading into that game (see that function's own docstring
for the mechanism - `_prior_rolling_series`, a real per-game, no-
lookahead-as-of-that-specific-game rolling series, mirroring
`teams.compute_offensive_edge`'s own already-shipped MLB precedent),
gated by `config.NFL_OPPONENT_ADJUSTMENT_WEIGHT` (0.0 = today's live,
unadjusted behavior - the real null hypothesis).

Uses `nfl_game_picks_backtest.build_multi_season_history`/
`score_multi_season_snapshots` (NOT `replay_season`, which never feeds a
real prior season at all) across all 10 real cached seasons
(2016-2025, including play-by-play - PR #98) - the exact same real infra
`scripts/backtest_nfl_season_carryover.py` already established.
`opponent_adjustment_weight` is the EXPENSIVE axis here (it changes
`assemble_team_metrics`'s own team-strength assembly, rebuilt once per
candidate weight and cached), unlike `composite_weights`/
`home_field_weight` (cheap re-scoring only) - so each weight in
`NFL_OPPONENT_ADJUSTMENT_WEIGHT_GRID` triggers exactly one real,
full-history rebuild, no more.

**The real bar, reported honestly either way**: a candidate weight must
beat the TRUE baseline (weight=0.0 - today's actual live behavior, live
composite weights, validated home-field term) on real accuracy/log_loss
across all replayed weeks. Nothing in config.py is touched unless a real
candidate clears it.

**Complaint-specific verification**: for each nonzero weight, the real
games where a team's RAW offensive_edge rank differs most from its
ADJUSTED rank - i.e. a team that looks mediocre by raw EPA but strong
once the opponents it actually faced are netted out, or vice versa -
printed side by side, a direct concrete check against the user's own
stated example.

Usage:
    python scripts/backtest_nfl_opponent_adjustment.py
"""

import os

import pandas as pd

from mlb_metrics import config, nfl_game_picks_backtest as backtest, nfl_team_strength

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nfl")
SEASONS = list(range(2016, 2026))  # 2016-2025, all real cached data (incl. pbp - PR #98)

WEIGHT_GRID = config.NFL_OPPONENT_ADJUSTMENT_WEIGHT_GRID


def _load(table: str, season: int) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(RAW_DIR, f"{table}_{season}.parquet"))


def _load_all(table: str) -> pd.DataFrame:
    return pd.concat([_load(table, season) for season in SEASONS], ignore_index=True)


def _score(replay: pd.DataFrame, label: str) -> dict:
    model = backtest.score_predictions(replay, "home_win_probability")
    market = backtest.score_predictions(replay, "market_home_win_probability")
    closing = backtest.beat_closing_line_rate(replay)
    return {
        "weight": label, "n": model["n"],
        "model_accuracy": model["accuracy"], "model_log_loss": model["log_loss"], "model_brier": model["brier_score"],
        "market_accuracy": market["accuracy"], "market_log_loss": market["log_loss"],
        "beat_closing_line_rate": closing["rate"], "beat_closing_line_n": closing["n_compared"],
    }


def _rank_divergence_report(team_stats: pd.DataFrame, season: int, weight: float, top_n: int = 5) -> None:
    """Real, complaint-specific check: teams whose offensive_edge RANK
    moves the most between raw (weight=0.0) and adjusted (this `weight`),
    using the full real `season`'s own final snapshot (as-of end of
    season - a real, readable illustration of the mechanism, not a
    no-lookahead claim in itself; the backtest's own accuracy numbers
    above are what actually validate no-lookahead real predictive
    value)."""
    season_stats = team_stats[team_stats["season"] == season]
    raw = nfl_team_strength.compute_team_offense_defense_edge(
        season_stats, current_season=season, opponent_adjustment_weight=0.0
    ).set_index("team")
    adjusted = nfl_team_strength.compute_team_offense_defense_edge(
        season_stats, current_season=season, opponent_adjustment_weight=weight
    ).set_index("team")

    combined = pd.DataFrame({
        "raw_offensive_edge": raw["offensive_edge"], "adjusted_offensive_edge": adjusted["offensive_edge"],
    })
    combined["raw_rank"] = combined["raw_offensive_edge"].rank(ascending=False)
    combined["adjusted_rank"] = combined["adjusted_offensive_edge"].rank(ascending=False)
    combined["rank_change"] = combined["raw_rank"] - combined["adjusted_rank"]

    print(f"\n  Real {season} offensive_edge rank movers at weight={weight} (raw vs. opponent-adjusted):")
    movers = combined.reindex(combined["rank_change"].abs().sort_values(ascending=False).index).head(top_n)
    for team, row in movers.iterrows():
        direction = "UP" if row["rank_change"] > 0 else "DOWN"
        print(
            f"    {team}: raw_rank={int(row['raw_rank'])} -> adjusted_rank={int(row['adjusted_rank'])} "
            f"({direction} {abs(row['rank_change']):.0f}) - raw={row['raw_offensive_edge']:.3f}, "
            f"adjusted={row['adjusted_offensive_edge']:.3f}"
        )


def main():
    print(f"Loading real {SEASONS[0]}-{SEASONS[-1]} NFL data (schedules/team_stats/weekly/snap_counts/rosters/pbp)...")
    schedules = _load_all("schedules")
    team_stats = _load_all("team_stats")
    weekly = _load_all("weekly")
    snap_counts = _load_all("snap_counts")
    rosters = _load_all("rosters_weekly")
    pbp = _load_all("pbp")

    all_rows = []
    for weight in WEIGHT_GRID:
        print(f"\nBuilding real history for opponent_adjustment_weight={weight} "
              f"(expensive - rebuilds team strength for every real replayed week)...")
        replay = backtest.replay_multi_season(
            schedules, team_stats, weekly, snap_counts, rosters, pbp, SEASONS,
            opponent_adjustment_weight=weight,
        )
        print(f"  {len(replay):,} real replayed games.")
        row = _score(replay, weight)
        all_rows.append(row)
        print(f"  weight={weight}: acc={row['model_accuracy']:.4f} log_loss={row['model_log_loss']:.4f} "
              f"brier={row['model_brier']:.4f} beat_line={row['beat_closing_line_rate']:.3f}")

        if weight != 0.0:
            _rank_divergence_report(team_stats, SEASONS[-1], weight)

    results = pd.DataFrame(all_rows)
    results.to_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "nfl_opponent_adjustment_backtest_results.csv"),
        index=False,
    )

    baseline = results[results["weight"] == 0.0].iloc[0]
    print("\n" + "=" * 100)
    print(f"BASELINE (weight=0.0, today's live behavior): acc={baseline['model_accuracy']:.4f} "
          f"log_loss={baseline['model_log_loss']:.4f} brier={baseline['model_brier']:.4f}")

    candidates = results[results["weight"] != 0.0]
    clears_bar = candidates[
        (candidates["model_log_loss"] < baseline["model_log_loss"])
        & (candidates["model_accuracy"] >= baseline["model_accuracy"])
    ]

    print(f"\n{len(clears_bar)} of {len(candidates)} candidate weights beat the baseline's log_loss "
          f"AND matched/beat its accuracy.")
    if clears_bar.empty:
        print("NO candidate weight cleared a real bar - reporting honestly: this backtest does NOT validate")
        print("shipping a nonzero NFL_OPPONENT_ADJUSTMENT_WEIGHT as tested. config.py is NOT being changed")
        print("based on this run.")
        best = candidates.sort_values("model_log_loss").iloc[0]
        print(f"\nClosest candidate by log_loss (NOT validated):\n{best}")
    else:
        winner = clears_bar.sort_values("model_log_loss").iloc[0]
        print(f"\nBest validated candidate (lowest log_loss among those clearing the bar):\n{winner}")

    print("\nFull results written to data/nfl_opponent_adjustment_backtest_results.csv")


if __name__ == "__main__":
    main()

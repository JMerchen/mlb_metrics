"""Validates the real production change made after round 1/2
(predictions.select_picks's hard avg_batting_order gate REMOVED,
replaced by lineup.compute_expected_plate_appearances feeding a
per-batter Expected_PA into hitters.assemble_hitters/
matchup.compute_matchup_hit_probability's trials count - see
config.LINEUP_TOP_HALF_MAX_SLOT's docstring) against the same real,
no-lookahead, full-season methodology round 1 used.

Two candidates, both now running through the NEW pipeline code (so c1's
own numbers shift slightly too vs round 1's - Expected_PA is baked into
WAVE/Matchup_Hit_Probability themselves now, not gated at selection time,
so it affects every candidate that uses Approach/Matchup_Approach):

  c1_wave_vs_pave    - same as round 1: PA qualifier only, ranked by
                        Matchup_Approach - the reference point.
  c3_new_heuristic    - today's REAL current selection logic post-change:
                        PA + start_rate + recency qualifiers (no more
                        avg_batting_order gate), ranked by Matchup_Approach.
                        This is what predictions.select_picks now actually
                        does live, unmodified - not a hypothetical.

If c3 now matches or beats c1 (unlike round 1's c2, which trailed it),
the fix worked.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import config, data, dfs_backtest, helpers, matchup, predictions

LINEUP_QUALIFIER_COLUMNS = ["avg_batting_order", "start_rate", "Last_Game_Date", "Recent_Avg_Batting_Order", "Expected_PA"]


def _resolve_picks(picks: pd.DataFrame, persisted: pd.DataFrame) -> pd.DataFrame:
    events = data.completed_events(persisted, ["game_date", "batter", "events"]).copy()
    events["had_hit"] = helpers.is_hit(events["events"])
    per_batter_day = (
        events.groupby(["game_date", "batter"])
        .agg(resolved_at_bats=("events", "size"), resolved_hit=("had_hit", "max"))
        .reset_index()
        .rename(columns={"game_date": "date", "batter": "key_mlbam"})
    )
    picks = picks.merge(per_batter_day, on=["date", "key_mlbam"], how="left")
    picks["at_bats"] = picks["resolved_at_bats"].fillna(0)
    got_hit = (picks["at_bats"] > 0) & (picks["resolved_hit"] == 1)
    got_out = (picks["at_bats"] > 0) & (picks["resolved_hit"] != 1)
    picks["actual_hit"] = pd.NA
    picks.loc[got_hit, "actual_hit"] = 1
    picks.loc[got_out, "actual_hit"] = 0
    return picks.drop(columns=["resolved_at_bats", "resolved_hit"])


def _select(pool: pd.DataFrame, date, *, rank_metric: str, teams_playing_today: set[str] | None,
            model_version: str) -> pd.DataFrame:
    return predictions.select_picks(
        pool, date,
        top_n=config.BACKTEST_TOP_N,
        min_plate_appearances=config.BACKTEST_MIN_PLATE_APPEARANCES,
        metric="Game_Hit_Probability",
        rank_metric=rank_metric,
        min_probability=0.0,
        teams_playing_today=teams_playing_today,
        model_version=model_version,
    )


def run_backtest(raw_dir: str = "data/raw", season: int | None = None) -> pd.DataFrame:
    season = season or config.SEASON_START.year
    persisted = data.load_persisted_statcast(raw_dir, season)
    if persisted is None:
        raise SystemExit("No persisted Statcast history found.")

    team_schedule = dfs_backtest.derive_historical_team_schedule(persisted)
    dates = sorted(team_schedule["date"].unique())
    print(f"Backtesting {len(dates)} real historical dates: {dates[0]} to {dates[-1]}", flush=True)

    all_picks = []
    start = time.time()
    for i, current_date in enumerate(dates):
        day = dfs_backtest._compute_date_outputs(persisted, team_schedule, current_date)
        if day is None:
            continue

        wave = day["outputs"]["wave"]
        todays_schedule = day["todays_schedule"]
        matchup_probability = day["matchup_probability"]
        teams_playing_today = set(todays_schedule["team"])

        matched_pool = wave.merge(matchup_probability, on="key_mlbam", how="inner")
        matched_pool["Matchup_Approach"] = matched_pool["Approach"] * matched_pool["Matchup_Hit_Probability"]

        c1_pool = matched_pool.drop(columns=[c for c in LINEUP_QUALIFIER_COLUMNS if c in matched_pool.columns])
        c1_picks = _select(c1_pool, current_date, rank_metric="Matchup_Approach",
                            teams_playing_today=teams_playing_today, model_version="backtest-v3-c1-wave-vs-pave")
        c1_picks["strategy"] = "c1_wave_vs_pave"

        c3_picks = _select(matched_pool, current_date, rank_metric="Matchup_Approach",
                            teams_playing_today=teams_playing_today, model_version="backtest-v3-c3-new-heuristic")
        c3_picks["strategy"] = "c3_new_heuristic"

        all_picks.append(pd.concat([c1_picks, c3_picks], ignore_index=True))

        if (i + 1) % 10 == 0 or i == len(dates) - 1:
            elapsed = time.time() - start
            print(f"  [{i+1}/{len(dates)}] {current_date} done ({elapsed:.0f}s elapsed)", flush=True)

    result = pd.concat(all_picks, ignore_index=True) if all_picks else pd.DataFrame()
    if result.empty:
        return result
    return _resolve_picks(result, persisted)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=config.SEASON_START.year)
    parser.add_argument("--output", default="data/backtest/pick_strategy_backtest_v3.csv")
    args = parser.parse_args()

    result = run_backtest(args.raw_dir, args.season)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    result.to_csv(args.output, index=False)
    print(f"Wrote {args.output} ({len(result)} rows).")


if __name__ == "__main__":
    main()

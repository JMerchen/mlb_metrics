"""Phase 3: does a team's real average Decision_Score (of the batters who
actually played for them) associate with real game outcomes - and,
specifically, does it add anything BEYOND what the existing team model
already captures (teams.compute_strength_metrics's pyth_strength)? Per
the user's explicit instruction on this feature: if not significant (or
not incremental beyond the existing team model), NO team-level Decision
Score display gets built at all - this script reports honestly either
way, and nothing downstream acts on its result until a human reads it.

Real, no-lookahead design - the same train (2025-03 to 06) / test
(2025-07 to 09) split as scripts/backtest_decision_score.py:
- Each batter's Decision_Score comes ENTIRELY from TRAIN
  (decision_score.compute_decision_score, config.WAVE_WINDOWS blend) -
  never re-fit on test.
- Each team's real per-game roster (which batters actually hit for them)
  and the real game outcome (teams.build_team_record's `win`) come
  entirely from TEST.
- The existing team model's signal (pyth_strength) is a SINGLE snapshot -
  teams.compute_strength_metrics run once on the full TRAIN-period team
  record ("team strength as of the day testing starts"), not a per-game
  walk-forward recompute across the test window - a documented
  simplification, same spirit as this project's other single-split
  backtests (e.g. scripts/backtest_decision_score.py itself).

Two real statsmodels.Logit fits (same significance-report shape
scripts/train_game_pick_model.py already establishes): team
Decision_Score alone, then team Decision_Score + pyth_strength together -
the second is the real "incremental signal" test.

Usage:
    python scripts/backtest_decision_score_team_level.py
"""

import os

import pandas as pd
import statsmodels.api as sm

from mlb_metrics import config, data, decision_score, pipeline, teams

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

TRAIN_FILES = [
    "statcast_2025_03.parquet", "statcast_2025_04.parquet",
    "statcast_2025_05.parquet", "statcast_2025_06.parquet",
]
TEST_FILES = ["statcast_2025_07.parquet", "statcast_2025_08.parquet", "statcast_2025_09.parquet"]


def _load(files: list) -> pd.DataFrame:
    frames = [pd.read_parquet(os.path.join(RAW_DIR, f)) for f in files]
    return pd.concat(frames, ignore_index=True)


def _batting_team(df: pd.DataFrame) -> pd.Series:
    """The real team at bat on this row: home team in the bottom of the
    inning, away team in the top - both already-real Statcast columns,
    no join/lookup needed."""
    return df["home_team"].where(df["inning_topbot"] == "Bot", df["away_team"])


def _zscore(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std()


def main():
    print(f"Loading train ({TRAIN_FILES})...")
    train_raw = _load(TRAIN_FILES)
    print(f"Loading test ({TEST_FILES})...")
    test_raw = _load(TEST_FILES)

    # Player-level Decision_Score, entirely from TRAIN.
    pa_events_train = pipeline.build_pitch_events(train_raw)
    all_pitches_train = pipeline.build_all_pitch_events(train_raw)
    player_scores = decision_score.compute_decision_score(all_pitches_train, pa_events_train)
    player_scores = player_scores.rename(columns={"key_mlbam": "batter"})
    print(f"{len(player_scores):,} batters with a real TRAIN-period Decision_Score.")

    # Existing team model signal, as of train-end (a single snapshot).
    train_with_game_id = data.assign_game_ids(train_raw)
    team_record_train = teams.build_team_record(train_with_game_id)
    current_strength, _sos = teams.compute_strength_metrics(team_record_train)

    # Real game outcomes, entirely from TEST.
    test_with_game_id = data.assign_game_ids(test_raw)
    team_record_test = teams.build_team_record(test_with_game_id)

    # Real per-(team, game) roster of batters who actually hit, from TEST.
    test_pa = test_with_game_id[["batter", "game_id", "home_team", "away_team", "inning_topbot", "events"]].copy()
    test_pa = test_pa[test_pa["events"].isin(config.COUNTED_EVENTS)]
    test_pa["team"] = _batting_team(test_pa)
    roster = test_pa[["team", "game_id", "batter"]].drop_duplicates()
    roster = roster.merge(player_scores[["batter", "Decision_Score"]], on="batter", how="inner")

    team_game_score = roster.groupby(["team", "game_id"], as_index=False)["Decision_Score"].mean()

    analysis = team_game_score.merge(team_record_test[["team", "game_id", "win"]], on=["team", "game_id"], how="inner")
    analysis = analysis.merge(current_strength[["team", "pyth_strength"]], on="team", how="left")
    analysis = analysis.dropna(subset=["Decision_Score", "win", "pyth_strength"])

    print(f"\n{len(analysis):,} real (team, game) rows with a Decision_Score, a real outcome, and a real prior team strength.")
    print(f"Real win rate in this sample: {analysis['win'].mean():.3f}")

    # .astype(float) only retags the dtype (every value is already
    # numeric) - same object-dtype guard scripts/train_game_pick_model.py
    # needs, in case the merge above upcasts a column to object.
    y = analysis["win"].astype(float)

    print("\n=== Univariate: team Decision_Score alone ===")
    X_uni = sm.add_constant(_zscore(analysis[["Decision_Score"]].astype(float)))
    uni = sm.Logit(y, X_uni).fit(disp=0)
    print(uni.summary2().tables[1])

    print("\n=== Multivariate: team Decision_Score + existing pyth_strength (the real incremental test) ===")
    X_multi = sm.add_constant(_zscore(analysis[["Decision_Score", "pyth_strength"]].astype(float)))
    multi = sm.Logit(y, X_multi).fit(disp=0)
    print(multi.summary2().tables[1])

    decision_p = multi.pvalues["Decision_Score"]
    print("\n" + "=" * 90)
    if decision_p < 0.05:
        print(f"Team Decision_Score IS significant (p={decision_p:.4f}) even controlling for pyth_strength - real incremental signal.")
        print("Per the plan: build the informational team-level display; wiring into the win-probability")
        print("composite is a separate follow-up decision, not decided by this script.")
    else:
        print(f"Team Decision_Score is NOT significant after controlling for pyth_strength (p={decision_p:.4f}).")
        print("Per the user's explicit instruction: no team-level Decision Score display will be built.")

    analysis.to_csv(
        os.path.join(os.path.dirname(__file__), "..", "data", "decision_score_team_level_backtest_results.csv"),
        index=False,
    )


if __name__ == "__main__":
    main()

import subprocess

import pandas as pd
import pytest

from mlb_metrics import game_picks_backtest


def _statcast_game(game_pk, date, home_team, away_team, home_pitcher, away_pitcher, home_score, away_score):
    """A minimal 4-row single-game Statcast fixture: Top(home pitching)/
    Bot(away pitching) alternating, score climbing to the given final."""
    return [
        {"game_pk": game_pk, "game_date": date, "home_team": home_team, "away_team": away_team,
         "pitcher": home_pitcher, "inning_topbot": "Top", "at_bat_number": 1,
         "post_home_score": 0, "post_away_score": 0},
        {"game_pk": game_pk, "game_date": date, "home_team": home_team, "away_team": away_team,
         "pitcher": away_pitcher, "inning_topbot": "Bot", "at_bat_number": 2,
         "post_home_score": min(1, home_score), "post_away_score": 0},
        {"game_pk": game_pk, "game_date": date, "home_team": home_team, "away_team": away_team,
         "pitcher": home_pitcher, "inning_topbot": "Top", "at_bat_number": 3,
         "post_home_score": min(1, home_score), "post_away_score": min(1, away_score)},
        {"game_pk": game_pk, "game_date": date, "home_team": home_team, "away_team": away_team,
         "pitcher": away_pitcher, "inning_topbot": "Bot", "at_bat_number": 4,
         "post_home_score": home_score, "post_away_score": away_score},
    ]


def test_derive_historical_schedule_games_uses_real_game_pk_not_reconstructed_id():
    # Two real doubleheader games (same two teams, same date, different
    # game_pk and different final scores) - a synthetic per-run game_id
    # reconstruction could conflate or fragment these; using Statcast's own
    # game_pk directly must not.
    rows = (
        _statcast_game(100, pd.Timestamp("2026-06-01"), "NYY", "BOS", 101, 201, 5, 2)
        + _statcast_game(101, pd.Timestamp("2026-06-01"), "NYY", "BOS", 102, 202, 1, 3)
    )
    games = game_picks_backtest.derive_historical_schedule_games(pd.DataFrame(rows)).set_index("game_pk")

    assert len(games) == 2
    assert games.loc[100, "home_score"] == 5 and games.loc[100, "away_score"] == 2
    assert games.loc[100, "home_probable_pitcher_key_mlbam"] == 101
    assert games.loc[100, "away_probable_pitcher_key_mlbam"] == 201
    assert games.loc[101, "home_score"] == 1 and games.loc[101, "away_score"] == 3
    assert games.loc[101, "status"] == "Final"


def test_derive_historical_schedule_games_excludes_zero_zero_artifacts():
    # MLB games can't end 0-0 - any reconstructed 0-0 "final" is a data
    # artifact (see module docstring) and must be dropped, not treated as a
    # real result to backtest against.
    rows = [
        {"game_pk": 999, "game_date": pd.Timestamp("2026-06-01"), "home_team": "NYY", "away_team": "BOS",
         "pitcher": 101, "inning_topbot": "Top", "at_bat_number": 1, "post_home_score": 0, "post_away_score": 0},
    ]
    games = game_picks_backtest.derive_historical_schedule_games(pd.DataFrame(rows))

    assert games.empty


def _git(args, cwd):
    subprocess.run(["git"] + args, cwd=cwd, check=True, capture_output=True, text=True)


def _confidence_csv(rows):
    return pd.DataFrame(
        [
            {
                "team": team, "pyth_Strength": s, "pyth_Confidence": c,
                "suppression_resistance": sr, "true_power": tp, "Bullpen_PAVE_PLUS": bp,
            }
            for team, s, c, sr, tp, bp in rows
        ]
    )


def _init_repo_with_confidence_history(tmp_path):
    repo = tmp_path / "repo"
    data_dir = repo / "docs" / "data"
    data_dir.mkdir(parents=True)
    confidence_path = data_dir / "confidence.csv"
    pave_path = data_dir / "pave.csv"

    _git(["init", "-q"], repo)
    _git(["config", "user.email", "test@example.com"], repo)
    _git(["config", "user.name", "Test"], repo)

    # Day 1: NYY (home) favored over BOS - faces BOS's weaker starter+bullpen.
    _confidence_csv(
        [("NYY", 1.1, 1.05, 1.0, 1.0, 0.9), ("BOS", 0.9, 0.95, 1.0, 1.0, 1.1)]
    ).to_csv(confidence_path, index=False)
    pd.DataFrame([{"key_mlbam": 101, "PAVE_PLUS": 0.8}, {"key_mlbam": 201, "PAVE_PLUS": 1.2}]).to_csv(
        pave_path, index=False
    )
    _git(["add", "docs/data/confidence.csv", "docs/data/pave.csv"], repo)
    _git(["commit", "-q", "-m", "day1", "--date=2026-06-01T12:00:00"], repo)

    return repo, confidence_path, pave_path


def test_reconstruct_historical_game_picks_replays_and_resolves_immediately(tmp_path):
    repo, _, _ = _init_repo_with_confidence_history(tmp_path)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _statcast_game(555, pd.Timestamp("2026-06-01"), "NYY", "BOS", 101, 201, 5, 2)
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    picks = game_picks_backtest.reconstruct_historical_game_picks(
        repo_dir=str(repo), raw_dir=str(raw_dir), season=2026, days=40
    )

    assert len(picks) == 1
    row = picks.iloc[0]
    assert row["game_pk"] == 555
    assert row["predicted_winner"] == "NYY"  # better composite + faces BOS's weaker pitching
    # Already resolved - a backtested pick's outcome is known at reconstruction time.
    assert row["game_played"] == 1
    assert row["actual_winner"] == "NYY"


def test_reconstruct_historical_game_picks_skips_commits_missing_required_columns(tmp_path):
    repo = tmp_path / "repo"
    data_dir = repo / "docs" / "data"
    data_dir.mkdir(parents=True)
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "test@example.com"], repo)
    _git(["config", "user.name", "Test"], repo)

    # A commit predating pyth_Strength/true_power etc. - old schema, must be skipped.
    pd.DataFrame([{"team": "NYY", "current": 1.0}]).to_csv(data_dir / "confidence.csv", index=False)
    pd.DataFrame([{"key_mlbam": 101, "PAVE_PLUS": 0.8}]).to_csv(data_dir / "pave.csv", index=False)
    _git(["add", "docs/data/confidence.csv", "docs/data/pave.csv"], repo)
    _git(["commit", "-q", "-m", "old-schema", "--date=2026-06-01T12:00:00"], repo)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _statcast_game(555, pd.Timestamp("2026-06-01"), "NYY", "BOS", 101, 201, 5, 2)
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    picks = game_picks_backtest.reconstruct_historical_game_picks(
        repo_dir=str(repo), raw_dir=str(raw_dir), season=2026, days=40
    )

    assert picks.empty


def test_reconstruct_historical_game_picks_days_limits_the_replay_window(tmp_path):
    repo = tmp_path / "repo"
    data_dir = repo / "docs" / "data"
    data_dir.mkdir(parents=True)
    _git(["init", "-q"], repo)
    _git(["config", "user.email", "test@example.com"], repo)
    _git(["config", "user.name", "Test"], repo)

    pave = pd.DataFrame([{"key_mlbam": 101, "PAVE_PLUS": 0.8}, {"key_mlbam": 201, "PAVE_PLUS": 1.2}])

    # Slightly different content each day - identical content would leave
    # nothing to commit (git commit fails with no changes staged).
    for day, date, nyy_strength in [(1, "2026-06-01", 1.1), (2, "2026-06-02", 1.2)]:
        _confidence_csv(
            [("NYY", nyy_strength, 1.05, 1.0, 1.0, 0.9), ("BOS", 0.9, 0.95, 1.0, 1.0, 1.1)]
        ).to_csv(data_dir / "confidence.csv", index=False)
        pave.to_csv(data_dir / "pave.csv", index=False)
        _git(["add", "docs/data/confidence.csv", "docs/data/pave.csv"], repo)
        _git(["commit", "-q", "-m", f"day{day}", f"--date={date}T12:00:00"], repo)

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    rows = _statcast_game(555, pd.Timestamp("2026-06-01"), "NYY", "BOS", 101, 201, 5, 2) + _statcast_game(
        556, pd.Timestamp("2026-06-02"), "NYY", "BOS", 101, 201, 3, 1
    )
    pd.DataFrame(rows).to_parquet(raw_dir / "statcast_2026.parquet", index=False)

    picks = game_picks_backtest.reconstruct_historical_game_picks(
        repo_dir=str(repo), raw_dir=str(raw_dir), season=2026, days=1
    )

    assert len(picks) == 1
    assert picks.iloc[0]["game_pk"] == 556  # only the most recent day's commit replayed

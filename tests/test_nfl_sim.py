"""Tests for nfl_sim (the What if? game simulation's team profiles) and
scripts/build_nfl_sim_profiles.py."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

from mlb_metrics import nfl_sim

REPO_ROOT = Path(__file__).resolve().parents[1]


def _play(**kw):
    row = {c: np.nan for c in nfl_sim.PBP_COLUMNS}
    row.update({"game_id": "g1", "season": 2026, "season_type": "REG", "week": 1, "qtr": 1, "down": 1,
                "ydstogo": 10, "score_differential": 0, "sack": 0, "interception": 0, "complete_pass": 0,
                "fumble_lost": 0, "pass_attempt": 0, "drive": 1})
    row.update(kw)
    return row


def _pbp():
    """Two teams' plays: KC throws a lot, BUF runs; plus last season's league
    plays, kicks and drive starts."""
    rng = np.random.default_rng(0)
    rows = []
    for season in (2025, 2026):
        for i in range(400):
            team = "KC" if i % 2 else "BUF"
            pass_rate = 0.7 if team == "KC" else 0.35
            if rng.random() < pass_rate:
                sack = rng.random() < 0.07
                complete = not sack and rng.random() < 0.65
                yards = -7 if sack else (float(rng.integers(0, 30)) if complete else 0.0)
                rows.append(_play(season=season, posteam=team, defteam="X", play_type="pass", sack=int(sack),
                                  complete_pass=int(complete), pass_attempt=1, yards_gained=yards,
                                  game_id=f"{season}_{i // 100}"))
            else:
                rows.append(_play(season=season, posteam=team, defteam="X", play_type="run",
                                  yards_gained=float(rng.integers(-2, 12)), game_id=f"{season}_{i // 100}"))
        for d in range(60):
            team = "KC" if d % 2 else "BUF"
            side = team if d % 3 else "OPP"
            rows.append(_play(season=season, posteam=team, play_type="run", yards_gained=1.0, drive=d,
                              game_id=f"k{season}", drive_start_transition="KICKOFF",
                              drive_start_yard_line=f"{side} {20 + d % 20}"))
        for k in range(80):
            dist = 20 + k % 40
            rows.append(_play(season=season, posteam="KC", play_type="field_goal", kick_distance=float(dist),
                              field_goal_result="made" if rng.random() < 1 / (1 + np.exp(-(6 - 0.1 * dist))) else "missed"))
            rows.append(_play(season=season, posteam="BUF", play_type="punt", kick_distance=45.0,
                              return_yards=5.0, touchback=float(k % 10 == 0)))
            rows.append(_play(season=season, posteam="KC", play_type="extra_point",
                              extra_point_result="good" if k % 20 else "failed"))
    pbp = pd.DataFrame(rows)
    pbp.loc[pbp["season"] == 2026, "week"] = 3
    return pbp


def test_profiles_give_each_team_its_style_but_the_league_mean():
    profiles = nfl_sim.build_profiles(_pbp(), 2026)
    lg = profiles["league"]
    assert profiles["through_week"] == 3 and set(profiles["teams"]) == {"KC", "BUF"}
    assert len(lg["rush_yards"]) == len(nfl_sim.QUANTILES) == len(lg["quantile_probs"])
    assert 0.5 < lg["comp_rate"] < 0.8 and 0.03 < lg["sack_rate"] < 0.12
    kc, buf = profiles["teams"]["KC"], profiles["teams"]["BUF"]
    # Pulled toward the league, but still apart.
    assert kc["neutral_pass_rate"] > lg["neutral_pass_rate"] > buf["neutral_pass_rate"]
    assert kc["neutral_pass_rate"] < 0.7
    # Style, not strength: every team's draws average the league's gain.
    for team in (kc, buf):
        for key in ("rush_yards", "completion_yards"):
            assert abs(nfl_sim._sampler_mean(team[key]) - nfl_sim._sampler_mean(lg[key])) < 0.02
    # Kicking: the field-goal curve falls with distance; punts net 40.
    assert lg["field_goal"]["b"] < 0
    assert lg["punt_net"][len(lg["punt_net"]) // 2] == 40.0
    assert 0.9 < lg["xp_rate"] < 1


def test_drive_starts_read_either_side_of_midfield():
    pbp = pd.DataFrame([
        {"game_id": "g", "drive": 1, "posteam": "KC", "drive_start_transition": "KICKOFF", "drive_start_yard_line": "KC 30"},
        {"game_id": "g", "drive": 2, "posteam": "KC", "drive_start_transition": "KICKOFF", "drive_start_yard_line": "BUF 40"},
        {"game_id": "g", "drive": 3, "posteam": "KC", "drive_start_transition": "KICKOFF", "drive_start_yard_line": "50"},
        {"game_id": "g", "drive": 3, "posteam": "KC", "drive_start_transition": "KICKOFF", "drive_start_yard_line": "50"},
        {"game_id": "g", "drive": 4, "posteam": "KC", "drive_start_transition": "PUNT", "drive_start_yard_line": "KC 10"},
    ])
    assert list(nfl_sim._drive_starts(pbp, "KICKOFF")) == [30, 60, 50]


def test_shrinking_a_small_sample_stays_near_the_league():
    assert nfl_sim._shrunk_rate(10, 10, 0.5) == (10 + 150 * 0.5) / 160
    league = np.linspace(-5, 40, len(nfl_sim.QUANTILES))
    assert nfl_sim._shrunk_shape([], league) == [round(float(v), 2) for v in league]


def _load_script():
    path = REPO_ROOT / "scripts" / "build_nfl_sim_profiles.py"
    spec = importlib.util.spec_from_file_location("build_nfl_sim_profiles", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_writes_the_profiles(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    raw.mkdir()
    pbp = _pbp()
    for season in (2025, 2026):
        pbp[pbp["season"] == season].to_parquet(raw / f"pbp_{season}.parquet", index=False)
    out = tmp_path / "out"
    out.mkdir()
    monkeypatch.setattr("sys.argv", ["build", "--raw-dir", str(raw), "--data-dir", str(out), "--season", "2026"])
    _load_script().main()
    profiles = json.loads((out / "nfl_sim_profiles.json").read_text())
    assert profiles["season"] == 2026 and set(profiles["teams"]) == {"KC", "BUF"}


def test_build_without_play_by_play_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["build", "--raw-dir", str(tmp_path), "--data-dir", str(tmp_path), "--season", "2026"])
    _load_script().main()
    assert not (tmp_path / "nfl_sim_profiles.json").exists()


def test_team_players_split_recent_targets_and_carries_among_the_active():
    rows = []
    for week in range(1, 7):
        gid = f"2026_{week:02d}_KC"
        for i in range(10):
            # Rice gets 6 of 10 targets; Kelce 4 - but Kelce has been cut.
            receiver, name = ("r1", "R.Rice") if i < 6 else ("t1", "T.Kelce")
            rows.append(_play(game_id=gid, week=week, posteam="KC", play_type="pass", pass_attempt=1,
                              passer_player_id="q1", passer_player_name="P.Mahomes",
                              receiver_player_id=receiver, receiver_player_name=name,
                              complete_pass=int(i % 2 == 0), yards_gained=12.0 if i % 2 == 0 else 0.0))
        for i in range(6):
            rusher, name = ("b1", "I.Pacheco") if i < 4 else ("q1", "P.Mahomes")
            rows.append(_play(game_id=gid, week=week, posteam="KC", play_type="run",
                              rusher_player_id=rusher, rusher_player_name=name, yards_gained=5.0))
        rows.append(_play(game_id=gid, week=week, posteam="KC", play_type="field_goal", kick_distance=40.0,
                          field_goal_result="made", kicker_player_id="k1", kicker_player_name="H.Butker"))
    current = pd.DataFrame(rows)
    roster = pd.DataFrame([
        {"season": 2026, "week": 6, "team": "KC", "position": p, "full_name": n, "gsis_id": g, "pfr_id": f, "status": st}
        for g, n, p, f, st in [("q1", "Patrick Mahomes", "QB", "MahoPa00", "ACT"), ("r1", "Rashee Rice", "WR", "RiceRa00", "ACT"),
                               ("t1", "Travis Kelce", "TE", "KelcTr00", "CUT"), ("b1", "Isiah Pacheco", "RB", "PachIs00", "ACT"),
                               ("k1", "Harrison Butker", "K", "ButkHa00", "ACT")]
    ])
    out = nfl_sim.team_players(current, "KC", nfl_sim._active_roster(roster, 2026))
    assert out["qb"] == {"gsis": "q1", "name": "Patrick Mahomes"}
    assert out["kicker"]["name"] == "Harrison Butker"
    by_name = {p["name"]: p for p in out["players"]}
    assert "Travis Kelce" not in by_name
    rice = by_name["Rashee Rice"]
    assert rice["pfr"] == "RiceRa00" and rice["targets"] == 1.0
    # Pacheco has 4 of the 6 carries a game, the QB's own runs the other 2.
    assert by_name["Isiah Pacheco"]["carries"] == round(4 / 6, 4)
    assert by_name["QB"]["role"] == "QB" and by_name["QB"]["carries"] == round(2 / 6, 4)
    # Rice catches half his targets, as the team does: a catch rate of 1x.
    assert rice["catch"] == 1.0


def test_penalty_profile_rates_types_and_how_each_is_enforced():
    rows = []
    for i in range(400):
        row = _play(posteam="KC", defteam="BUF", play_type="run", yards_gained=4.0, desc="run", game_id=f"g{i // 50}")
        rows.append(row)
    def flag(team, kind, yards, play_type="no_play", first=0, desc="PENALTY"):
        return _play(posteam="KC", defteam="BUF", play_type=play_type, penalty=1, penalty_team=team,
                     penalty_type=kind, penalty_yards=float(yards), first_down_penalty=first, desc=desc)
    rows += [flag("KC", "False Start", 5) for _ in range(20)]
    rows += [flag("KC", "Offensive Holding", 10) for _ in range(10)]
    rows += [flag("BUF", "Defensive Pass Interference", y, first=1) for y in [8, 12, 15, 20, 30] * 4]
    rows += [flag("BUF", "Unnecessary Roughness", 15, play_type="run", first=1) for _ in range(8)]
    rows += [flag("BUF", "Defensive Holding", 5, desc="PENALTY declined") for _ in range(30)]
    snaps, flags = nfl_sim._snaps_with_flags(pd.DataFrame(rows))
    league = nfl_sim._penalty_league(snaps, flags)
    n = len(snaps)
    assert league["offense_rate"] == round(30 / n, 4)
    # Declined flags don't count.
    assert league["defense_rate"] == round(28 / n, 4)
    by = {t["name"]: t for t in league["types"]}
    assert set(by) == {"False Start", "Offensive Holding", "Defensive Pass Interference", "Unnecessary Roughness"}
    assert by["False Start"]["presnap"] and by["False Start"]["share"] == round(20 / 30, 4)
    assert by["Offensive Holding"]["yards"] == 10 and not by["Offensive Holding"]["presnap"]
    # A personal foul on a play that stood is added on after it.
    assert by["Unnecessary Roughness"]["after"] and not by["Defensive Pass Interference"]["after"]
    assert by["Defensive Pass Interference"]["first_down"] == 1.0
    assert len(league["dpi_yards"]) == len(nfl_sim.QUANTILES)

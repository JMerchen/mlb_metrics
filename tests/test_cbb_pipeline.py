import datetime
import json

import pandas as pd
import pytest

from tests.cbb_helpers import box_for, bracket_schedule, game, round_robin
from mlb_metrics import cbb_data, cbb_pipeline, cbb_ratings, config

STRENGTH = {1: (10.0, 8.0), 2: (4.0, 2.0), 3: (0.0, 0.0), 4: (-3.0, -5.0), 5: (-11.0, -5.0)}
TODAY = datetime.date(2027, 1, 10)


def _season():
    played = round_robin(STRENGTH)
    upcoming = pd.DataFrame([
        game(5001, "2027-01-10", 1, 5, 0, 0, done=False),
        game(5002, "2027-01-11", 3, 2, 0, 0, done=False, neutral=True),
        game(5003, "2027-01-12", 4, 99, 0, 0, done=False, away_conf=None),     # not a Division I opponent
        game(5004, "2027-01-20", 2, 1, 0, 0, done=False),                      # past the window
        game(5005, "2027-01-11", 2, -1, 0, 0, done=False),                     # opponent to be decided
    ])
    return pd.concat([played, upcoming], ignore_index=True)


def test_upcoming_games_are_unplayed_division_one_games_in_the_next_few_days():
    games = cbb_pipeline.upcoming_games(_season(), TODAY, days=2)
    assert games["game_id"].tolist() == ["5001", "5002"]
    assert games["date"].tolist() == ["2027-01-10", "2027-01-11"] and games["neutral"].tolist() == [False, True]


def test_price_games_gives_a_win_chance_and_a_score_that_agree():
    schedules = _season()
    games = cbb_ratings.prepare_games(schedules, box_for(schedules[schedules["status_type_completed"]]))
    ratings, league = cbb_ratings.fit_ratings(games)
    names = cbb_ratings.team_names(schedules).set_index("team_id")["team"]
    unknown = pd.DataFrame([{"game_id": "x", "date": "2027-01-10", "home_id": 1, "away_id": 77, "neutral": False}])
    picks = cbb_pipeline.price_games(pd.concat([cbb_pipeline.upcoming_games(schedules, TODAY, 2), unknown], ignore_index=True),
                                     ratings, league, names, 2027)
    assert list(picks.columns) == cbb_pipeline.LOG_COLUMNS and picks["game_id"].tolist() == ["5001", "5002"]
    best_at_home = picks.iloc[0]
    assert best_at_home["home_team"] == "Team 1" and best_at_home["home_win_probability"] > 0.95
    assert best_at_home["predicted_home_points"] - best_at_home["predicted_away_points"] == pytest.approx(
        best_at_home["predicted_margin"], abs=0.11)
    assert picks.iloc[1]["home_win_probability"] < 0.5 and picks.iloc[1]["predicted_margin"] < 0     # team 3 vs team 2, neutral
    assert picks["home_score"].isna().all() and (picks["model_version"] == config.CBB_MODEL_VERSION).all()


def _pick(game_id, probability, margin=5.0, date="2027-01-10", season=2027):
    return {"date": date, "season": season, "game_id": game_id, "home_id": 1, "away_id": 2, "home_team": "Team 1",
            "away_team": "Team 2", "neutral": False, "home_win_probability": probability, "predicted_margin": margin,
            "predicted_home_points": 75.0, "predicted_away_points": 70.0, "model_version": "v1",
            "home_score": float("nan"), "away_score": float("nan")}


def test_update_log_keeps_a_games_first_pick_and_fills_in_final_scores(tmp_path):
    path = str(tmp_path / "log.csv")
    nothing_played = cbb_ratings.prepare_games(pd.DataFrame([game(1, "2027-01-01", 3, 4, 70, 60)]))
    cbb_pipeline.update_log(path, pd.DataFrame([_pick("5001", 0.70), _pick("5002", 0.40)]), nothing_played)
    final = cbb_ratings.prepare_games(pd.DataFrame([game(5001, "2027-01-10", 1, 2, 81, 77)]))
    log = cbb_pipeline.update_log(path, pd.DataFrame([_pick("5001", 0.99), _pick("5003", 0.55, date="2027-01-12")]), final)
    assert log["game_id"].tolist() == ["5001", "5002", "5003"]
    by_id = log.set_index("game_id")
    assert by_id.at["5001", "home_win_probability"] == 0.70                  # not repriced
    assert by_id.at["5001", "home_score"] == 81 and by_id.at["5001", "away_score"] == 77
    assert pd.isna(by_id.at["5002", "home_score"])
    assert pd.read_csv(path, dtype={"game_id": str})["game_id"].tolist() == ["5001", "5002", "5003"]


def test_pick_record_counts_graded_picks_by_how_sure_they_were():
    log = pd.DataFrame([
        {**_pick("1", 0.65, margin=4.0), "home_score": 80.0, "away_score": 70.0},     # right, off by 6
        {**_pick("2", 0.35, margin=-4.0), "home_score": 80.0, "away_score": 70.0},    # picked the visitor: wrong, off by 14
        {**_pick("3", 0.95, margin=20.0), "home_score": 90.0, "away_score": 60.0},    # right, off by 10
        _pick("4", 0.80),                                                              # not played
        {**_pick("5", 0.80, season=2026), "home_score": 90.0, "away_score": 60.0},    # last season
    ])
    record = cbb_pipeline.pick_record(log, 2027)
    assert (record["games"], record["right"], record["margin_miss"]) == (3, 2, 10.0)
    bands = {b["label"]: (b["games"], b["right"]) for b in record["bands"]}
    assert bands["60–70%"] == (2, 1) and bands["90–100%"] == (1, 1) and bands["50–60%"] == (0, 0)
    assert cbb_pipeline.pick_record(log.iloc[0:0], 2027) == {"games": 0, "right": 0, "margin_miss": None, "bands": record["bands"][:0] or [
        {**b, "games": 0, "right": 0} for b in record["bands"]]}


def _fetch_from(tables):
    def fetch(table, season, **kwargs):
        if (table, season) not in tables:
            raise RuntimeError("no such season")
        return tables[(table, season)]
    return fetch


def test_run_writes_the_page_data_and_the_log(tmp_path, monkeypatch):
    schedules = _season()
    box = box_for(schedules[schedules["status_type_completed"]])
    last = round_robin({1: (2.0, 2.0), 2: (8.0, 8.0), 3: (0.0, 0.0), 4: (-2.0, -2.0), 5: (-8.0, -8.0)}, season=2026)
    tables = {("schedules", 2027): schedules, ("team_box", 2027): box, ("schedules", 2026): last, ("team_box", 2026): box_for(last)}
    monkeypatch.setattr(cbb_data, "fetch_table", _fetch_from(tables))
    monkeypatch.setattr(config, "CBB_BRACKET_SIMS", 200)
    out, log_path = tmp_path / "out", str(tmp_path / "log.csv")
    payload = cbb_pipeline.run(season=2027, raw_dir=str(tmp_path / "raw"), output_dir=str(out), log_path=log_path, today=TODAY)
    saved = json.loads((out / "cbb.json").read_text())           # strict JSON: a browser can read it
    assert "NaN" not in (out / "cbb.json").read_text()
    assert saved["season"] == 2027 and saved["updated"] == "2027-01-10" and saved["games_played"] == 20
    assert saved["through"] == "2026-12-20"
    assert [t["team"] for t in saved["teams"]] == ["Team 1", "Team 2", "Team 3", "Team 4", "Team 5"]
    top = saved["teams"][0]
    assert top["rank"] == 1 and top["offense_rank"] == 1 and top["wins"] == 8 and top["conference"] == "America East"
    assert top["last_season_net"] is not None and saved["teams"][1]["last_season_net"] > top["last_season_net"]
    assert [g["game_id"] for g in saved["games"]] == ["5001", "5002"] and saved["games"][0]["home_score"] is None
    assert saved["record"]["games"] == 0
    # Five teams can't fill a bracket: the page gets an empty projected one.
    assert saved["bracket"]["kind"] == "projected" and saved["bracket"]["teams"] == []
    assert payload["league"]["sigma"] == config.CBB_MARGIN_SIGMA
    assert pd.read_csv(log_path)["game_id"].astype(str).tolist() == ["5001", "5002"]
    # The next day one game is final: it's graded, and nothing is repriced.
    tables[("schedules", 2027)] = pd.concat([schedules[schedules["game_id"] != 5001],
                                             pd.DataFrame([game(5001, "2027-01-10", 1, 5, 90, 60)])], ignore_index=True)
    again = cbb_pipeline.run(season=2027, raw_dir=str(tmp_path / "raw"), output_dir=str(out), log_path=log_path,
                             today=datetime.date(2027, 1, 11))
    assert again["record"]["games"] == 1 and again["record"]["right"] == 1
    assert [g["game_id"] for g in again["games"]] == ["5001", "5002"] and again["games"][0]["home_score"] == 90


def test_run_does_nothing_when_the_data_cant_be_fetched(tmp_path, monkeypatch):
    monkeypatch.setattr(cbb_data, "fetch_table", _fetch_from({}))
    assert cbb_pipeline.run(season=2027, raw_dir=str(tmp_path), output_dir=str(tmp_path / "out"),
                            log_path=str(tmp_path / "log.csv"), today=TODAY) is None
    assert not (tmp_path / "out" / "cbb.json").exists()


def test_build_bracket_uses_the_real_bracket_once_it_is_posted(monkeypatch):
    monkeypatch.setattr(config, "CBB_BRACKET_SIMS", 300)
    schedule, where = bracket_schedule(first_four_done=True)
    games = cbb_ratings.prepare_games(schedule)
    ratings = pd.DataFrame({"team_id": list(where), "offense": 0.0, "defense": 0.0,
                            "net": [20.0 - 2.5 * seed for _, seed in where.values()], "tempo": 68.0,
                            "games": 30, "wins": 20, "losses": 10})
    league = {"efficiency": 105.0, "home_edge": 2.5, "tempo": 68.0, "games": 5000}
    block = cbb_pipeline.build_bracket(2027, schedule, games, ratings, league, cbb_ratings.team_names(schedule))
    assert block["kind"] == "official" and len(block["teams"]) == 68 and block["final_four_pairs_known"] is False
    by_team = {t["team_id"]: t for t in block["teams"]}
    # The First Four is played: each winner is in the round of 64 for sure, each loser is out.
    assert by_team[116]["round_of_64"] == 1.0 and by_team[117]["round_of_64"] == 0.0
    assert len(block["results"]) == 4 and block["results"][0]["round"] == "First Four"
    assert sum(t["champion"] for t in block["teams"]) == pytest.approx(1.0, abs=1e-3)


def test_fetch_table_keeps_the_columns_used_and_retries(monkeypatch):
    import io

    frame = pd.DataFrame({"game_id": [1], "team_id": [2], "field_goals_attempted": [50], "extra": ["x"]})
    buffer = io.BytesIO()
    frame.to_parquet(buffer)
    calls = []

    class Response:
        content = buffer.getvalue()

        def raise_for_status(self):
            pass

    def get(url, timeout):
        calls.append(url)
        if len(calls) == 1:
            raise OSError("network down")
        return Response()

    monkeypatch.setattr(cbb_data.requests, "get", get)
    monkeypatch.setattr(cbb_data.time, "sleep", lambda s: None)
    out = cbb_data.fetch_table("team_box", 2027)
    assert list(out.columns) == ["game_id", "team_id", "field_goals_attempted"] and len(calls) == 2
    assert calls[0].endswith("/team_box/parquet/team_box_2027.parquet")
    assert cbb_data.table_url("schedules", 2026).endswith("/schedules/parquet/mbb_schedule_2026.parquet")


def test_load_table_reads_the_cache_before_the_network(tmp_path, monkeypatch):
    frame = pd.DataFrame({"game_id": [1]})
    fetched = []
    monkeypatch.setattr(cbb_data, "fetch_table", lambda table, season: fetched.append((table, season)) or frame)
    first = cbb_data.load_table("schedules", 2026, str(tmp_path))
    second = cbb_data.load_table("schedules", 2026, str(tmp_path))
    assert fetched == [("schedules", 2026)] and first.equals(second)
    cbb_data.load_table("schedules", 2026, str(tmp_path), refresh=True)
    assert len(fetched) == 2

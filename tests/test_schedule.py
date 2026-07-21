import pandas as pd

from mlb_metrics import schedule


def test_team_id_to_abbrev_has_30_teams_and_matches_docs_app_js():
    # Spot-check against the table docs/app.js's teamLogo() ports from -
    # catches drift if either table is edited without the other.
    assert len(schedule.TEAM_ID_TO_ABBREV) == 30
    assert schedule.TEAM_ID_TO_ABBREV[133] == "ATH"
    assert schedule.TEAM_ID_TO_ABBREV[120] == "WSH"
    assert schedule.TEAM_ID_TO_ABBREV[147] == "NYY"


def _raw_schedule(games):
    """games: list of (date, home_id, away_id, home_probable_id, away_probable_id)."""
    by_date = {}
    for date, home_id, away_id, home_probable, away_probable in games:
        entry = by_date.setdefault(date, {"date": date, "games": []})
        game = {
            "teams": {
                "home": {"team": {"id": home_id}},
                "away": {"team": {"id": away_id}},
            }
        }
        if home_probable is not None:
            game["teams"]["home"]["probablePitcher"] = {"id": home_probable}
        if away_probable is not None:
            game["teams"]["away"]["probablePitcher"] = {"id": away_probable}
        entry["games"].append(game)
    return {"dates": list(by_date.values())}


def test_normalize_schedule_parses_probable_pitchers_and_opponents():
    # NYY (147) home vs BOS (111) away; NYY's probable is announced, BOS's isn't yet.
    raw = _raw_schedule([("2026-07-21", 147, 111, 592789, None)])

    result = schedule.normalize_schedule(raw).set_index("team")

    assert result.loc["NYY", "opponent"] == "BOS"
    assert result.loc["NYY", "probable_pitcher_key_mlbam"] == 592789
    assert result.loc["BOS", "opponent"] == "NYY"
    assert pd.isna(result.loc["BOS", "probable_pitcher_key_mlbam"])
    assert (result["date"] == pd.Timestamp("2026-07-21")).all()


def test_normalize_schedule_skips_unknown_team_ids():
    raw = _raw_schedule([("2026-07-21", 147, 999999, 592789, 111111)])

    result = schedule.normalize_schedule(raw)

    assert result.empty  # unknown away team id (999999) -> whole game skipped, not guessed at


def test_normalize_schedule_keeps_only_first_game_of_a_doubleheader():
    raw = _raw_schedule(
        [
            ("2026-07-21", 147, 111, 592789, 111111),
            ("2026-07-21", 147, 111, 605400, 222222),  # game 2, same two teams
        ]
    )

    result = schedule.normalize_schedule(raw)

    assert len(result[result["team"] == "NYY"]) == 1
    assert result[result["team"] == "NYY"].iloc[0]["probable_pitcher_key_mlbam"] == 592789

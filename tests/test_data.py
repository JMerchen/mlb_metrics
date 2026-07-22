import pandas as pd

from mlb_metrics import data


def test_persist_raw_statcast_appends_and_dedupes_by_pitch(tmp_path):
    raw_dir = str(tmp_path / "raw")
    season = 2026

    day1 = pd.DataFrame({
        "game_pk": [1, 1],
        "at_bat_number": [1, 2],
        "pitch_number": [1, 1],
        "game_date": pd.to_datetime(["2026-04-01", "2026-04-01"]),
        "events": ["single", "field_out"],
    })
    combined1 = data.persist_raw_statcast(day1, raw_dir, season)
    assert len(combined1) == 2

    day2 = pd.DataFrame({
        "game_pk": [1, 2],
        "at_bat_number": [2, 1],  # first row shares day1's pitch key
        "pitch_number": [1, 1],
        "game_date": pd.to_datetime(["2026-04-01", "2026-04-02"]),
        "events": ["walk", "single"],
    })
    combined2 = data.persist_raw_statcast(day2, raw_dir, season)

    assert len(combined2) == 3
    dup_row = combined2[(combined2["game_pk"] == 1) & (combined2["at_bat_number"] == 2)]
    assert dup_row.iloc[0]["events"] == "walk"  # keep="last" -> day2's value wins on conflict

    reloaded = data.load_persisted_statcast(raw_dir, season)
    assert len(reloaded) == 3


def test_load_persisted_statcast_returns_none_when_absent(tmp_path):
    assert data.load_persisted_statcast(str(tmp_path / "raw"), 2026) is None


def _pitch(game_pk, game_date, home, away, at_bat_number, pitch_number=1):
    return {
        "game_pk": game_pk, "game_date": pd.Timestamp(game_date), "home_team": home, "away_team": away,
        "at_bat_number": at_bat_number, "pitch_number": pitch_number,
    }


def test_assign_game_ids_groups_by_game_pk_not_reconstructed_boundaries():
    """Two real games on the same date, same two teams (a doubleheader) -
    game_pk already disambiguates them directly, no at_bat_number-reset
    heuristic needed."""
    rows = [
        _pitch(100, "2026-06-01", "NYY", "BOS", 1),
        _pitch(100, "2026-06-01", "NYY", "BOS", 2),
        _pitch(101, "2026-06-01", "NYY", "BOS", 1),
        _pitch(101, "2026-06-01", "NYY", "BOS", 2),
    ]
    result = data.assign_game_ids(pd.DataFrame(rows))

    ids_by_pk = result.groupby("game_pk")["game_id"].nunique()
    assert (ids_by_pk == 1).all()  # each real game gets exactly one game_id
    assert result[result["game_pk"] == 100]["game_id"].iloc[0] != result[result["game_pk"] == 101]["game_id"].iloc[0]


def test_assign_game_ids_is_correct_even_when_simultaneous_games_interleave_in_row_order():
    """Regression test for a real bug: the old implementation reconstructed
    game boundaries by walking a single global counter over the whole
    table, which only worked if one real game's rows were contiguous.
    persist_raw_statcast only sorts by game_date, and MLB plays many games
    in parallel each day, so rows from different simultaneous games can be
    interleaved in arbitrary order - confirmed empirically to badly
    fragment/conflate real games at scale. This fixture reproduces that
    interleaving directly: two different games' at-bat rows alternate row
    by row, and every row must still land in the correct one of exactly two
    game_id groups."""
    rows = [
        _pitch(100, "2026-06-01", "NYY", "BOS", 1),   # game A start
        _pitch(200, "2026-06-01", "SEA", "TEX", 1),   # game B start (interleaved)
        _pitch(100, "2026-06-01", "NYY", "BOS", 2),   # game A continues
        _pitch(200, "2026-06-01", "SEA", "TEX", 2),   # game B continues
        _pitch(100, "2026-06-01", "NYY", "BOS", 3),
        _pitch(200, "2026-06-01", "SEA", "TEX", 3),
    ]
    result = data.assign_game_ids(pd.DataFrame(rows))

    game_a_ids = result[result["game_pk"] == 100]["game_id"].unique()
    game_b_ids = result[result["game_pk"] == 200]["game_id"].unique()
    assert len(game_a_ids) == 1  # not fragmented across multiple game_ids
    assert len(game_b_ids) == 1
    assert game_a_ids[0] != game_b_ids[0]  # not conflated into the same game_id


def test_assign_game_ids_orders_game_id_chronologically():
    """Downstream code (teams.py, lineup.py) relies on a higher game_id
    meaning a more recent game (e.g. groupby(...)['game_id'].max() for
    "latest game") - game_pk itself isn't chronologically monotonic (it's
    assigned by MLB's scheduling system, not in play order), so game_id
    must be independently re-derived in date order, not just copied from
    game_pk."""
    rows = [
        _pitch(500, "2026-06-02", "NYY", "BOS", 1),  # later date, lower game_pk
        _pitch(100, "2026-06-01", "SEA", "TEX", 1),  # earlier date, higher game_pk
    ]
    result = data.assign_game_ids(pd.DataFrame(rows))

    id_earlier = result[result["game_pk"] == 100]["game_id"].iloc[0]
    id_later = result[result["game_pk"] == 500]["game_id"].iloc[0]
    assert id_earlier < id_later


def test_label_pitcher_roles_picks_lowest_at_bat_number_as_starter():
    # Top half = home_team pitching to away_team's batters; Bot half =
    # away_team pitching to home_team's batters (matches the convention
    # used throughout data.py's other team-lookup helpers).
    rows = [
        {"game_id": 1, "inning_topbot": "Top", "home_team": "A", "away_team": "B", "pitcher": 101, "at_bat_number": 1},
        {"game_id": 1, "inning_topbot": "Bot", "home_team": "A", "away_team": "B", "pitcher": 201, "at_bat_number": 2},
        {"game_id": 1, "inning_topbot": "Top", "home_team": "A", "away_team": "B", "pitcher": 102, "at_bat_number": 3},
        {"game_id": 1, "inning_topbot": "Bot", "home_team": "A", "away_team": "B", "pitcher": 202, "at_bat_number": 4},
    ]
    data_with_game_id = pd.DataFrame(rows)

    roles = data.label_pitcher_roles(data_with_game_id).set_index(["team", "pitcher"])

    assert roles.loc[("A", 101), "is_starter"]
    assert not roles.loc[("A", 102), "is_starter"]
    assert roles.loc[("B", 201), "is_starter"]
    assert not roles.loc[("B", 202), "is_starter"]


def test_extract_game_results_finds_final_score_per_game():
    rows = [
        # Game 1: NYY (home) beats BOS (away) 5-2, score climbing over 4 rows.
        {"game_id": 1, "game_date": pd.Timestamp("2026-06-01"), "home_team": "NYY", "away_team": "BOS",
         "post_home_score": 0, "post_away_score": 0},
        {"game_id": 1, "game_date": pd.Timestamp("2026-06-01"), "home_team": "NYY", "away_team": "BOS",
         "post_home_score": 1, "post_away_score": 0},
        {"game_id": 1, "game_date": pd.Timestamp("2026-06-01"), "home_team": "NYY", "away_team": "BOS",
         "post_home_score": 1, "post_away_score": 2},
        {"game_id": 1, "game_date": pd.Timestamp("2026-06-01"), "home_team": "NYY", "away_team": "BOS",
         "post_home_score": 5, "post_away_score": 2},
        # Game 2: LAD (home) loses to SF (away) 1-3.
        {"game_id": 2, "game_date": pd.Timestamp("2026-06-01"), "home_team": "LAD", "away_team": "SF",
         "post_home_score": 0, "post_away_score": 1},
        {"game_id": 2, "game_date": pd.Timestamp("2026-06-01"), "home_team": "LAD", "away_team": "SF",
         "post_home_score": 1, "post_away_score": 3},
    ]
    results = data.extract_game_results(pd.DataFrame(rows)).set_index("game_id")

    assert results.loc[1, "home_score"] == 5
    assert results.loc[1, "away_score"] == 2
    assert results.loc[2, "home_score"] == 1
    assert results.loc[2, "away_score"] == 3


def test_assign_batting_order_uses_away_bats_top_home_bats_bottom():
    # A batter who only appears during Top-half at-bats must be credited to
    # the AWAY team (they're batting while the home team pitches) - this is
    # the OPPOSITE convention from label_pitcher_roles's pitching-team split
    # above, and easy to invert by accident since the two functions look
    # nearly identical.
    rows = [
        {"game_id": 1, "inning_topbot": "Top", "home_team": "A", "away_team": "B", "batter": 501, "at_bat_number": 1},
        {"game_id": 1, "inning_topbot": "Bot", "home_team": "A", "away_team": "B", "batter": 601, "at_bat_number": 2},
    ]
    data_with_game_id = pd.DataFrame(rows)

    order = data.assign_batting_order(data_with_game_id)

    top_row = order[order["batter"] == 501].iloc[0]
    assert top_row["team"] == "B"  # away team bats during Top

    bot_row = order[order["batter"] == 601].iloc[0]
    assert bot_row["team"] == "A"  # home team bats during Bot


def test_assign_batting_order_ranks_by_first_at_bat_and_flags_late_entries():
    rows = []
    # Away team B's 9 starters, first at-bats at 1, 3, 5, ..., 17.
    for i, batter in enumerate(range(901, 910)):
        rows.append({
            "game_id": 1, "inning_topbot": "Top", "home_team": "A", "away_team": "B",
            "batter": batter, "at_bat_number": 1 + 2 * i,
        })
    # Home team A's 9 starters, first at-bats at 2, 4, 6, ..., 18.
    for i, batter in enumerate(range(801, 810)):
        rows.append({
            "game_id": 1, "inning_topbot": "Bot", "home_team": "A", "away_team": "B",
            "batter": batter, "at_bat_number": 2 + 2 * i,
        })
    # A substitute for team B enters after all 9 starters have batted once.
    rows.append({
        "game_id": 1, "inning_topbot": "Top", "home_team": "A", "away_team": "B",
        "batter": 999, "at_bat_number": 19,
    })

    order = data.assign_batting_order(pd.DataFrame(rows)).set_index(["team", "batter"])

    assert order.loc[("B", 901), "batting_order"] == 1
    assert order.loc[("B", 909), "batting_order"] == 9
    assert order.loc[("B", 999), "batting_order"] == 10  # substitute, not part of the starting 9
    assert order.loc[("A", 801), "batting_order"] == 1
    assert order.loc[("A", 809), "batting_order"] == 9

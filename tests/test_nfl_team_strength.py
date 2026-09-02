import pandas as pd
import pytest

from mlb_metrics import config, nfl_team_strength


def _game(game_id, season, week, home_team, away_team, home_score, away_score, game_type="REG"):
    return {
        "game_id": game_id, "season": season, "week": week, "game_type": game_type,
        "home_team": home_team, "away_team": away_team,
        "home_score": home_score, "away_score": away_score,
    }


def test_build_team_record_splits_home_and_away_correctly():
    schedules = pd.DataFrame([
        _game("2025_01_KC_DET", 2025, 1, "DET", "KC", 21, 20),  # DET (home) wins
        _game("2025_02_KC_PHI", 2025, 2, "PHI", "KC", 17, 24),  # KC (away) wins
    ])

    record = nfl_team_strength.build_team_record(schedules).set_index("team")

    det = record.loc["DET"]
    assert det["opp"] == "KC" and det["ps"] == 21 and det["pa"] == 20
    assert det["win"] == 1 and det["loss"] == 0

    # KC has two rows (one per game) - check both explicitly.
    kc = nfl_team_strength.build_team_record(schedules)
    kc_rows = kc[kc["team"] == "KC"].sort_values("week")
    assert kc_rows.iloc[0][["opp", "ps", "pa", "win", "loss"]].tolist() == ["DET", 20, 21, 0, 1]
    assert kc_rows.iloc[1][["opp", "ps", "pa", "win", "loss"]].tolist() == ["PHI", 24, 17, 1, 0]


def test_build_team_record_excludes_playoffs_and_unplayed_games():
    schedules = pd.DataFrame([
        _game("2025_01_KC_DET", 2025, 1, "DET", "KC", 21, 20),
        _game("2025_19_KC_DET", 2025, 19, "DET", "KC", 30, 27, game_type="WC"),  # real playoff game
        _game("2025_02_KC_PHI", 2025, 2, "PHI", "KC", None, None),  # real unplayed/future game
    ])

    record = nfl_team_strength.build_team_record(schedules)

    assert len(record) == 2  # only the one real completed REG game (both team rows)
    assert set(record["game_id"]) == {"2025_01_KC_DET"}


def test_build_team_record_handles_a_real_tie():
    # NFL games can end in a real tie - win/loss both 0, not fabricated.
    schedules = pd.DataFrame([_game("2025_01_A_B", 2025, 1, "A", "B", 17, 17)])

    record = nfl_team_strength.build_team_record(schedules).set_index("team")

    assert record.loc["A", "win"] == 0 and record.loc["A", "loss"] == 0
    assert record.loc["B", "win"] == 0 and record.loc["B", "loss"] == 0


def test_compute_strength_metrics_survives_chained_groupby_apply():
    # Regression guard mirroring test_teams.py's identically-named test -
    # compute_strength_metrics chains two separate groupby("team").apply()
    # calls; pandas excludes the grouping column from what's passed to the
    # applied function, so the helpers must re-derive "team" from
    # group.name rather than relying on it staying a column in `group`.
    record = pd.DataFrame({
        "opp": ["BOS", "NYY", "BOS", "NYY"],
        "team": ["NYY", "BOS", "NYY", "BOS"],
        "season": [2025, 2025, 2025, 2025],
        "week": [1, 1, 2, 2],
        "game_id": ["2025_01_A", "2025_01_A", "2025_02_B", "2025_02_B"],
        "win": [1, 0, 0, 1],
        "loss": [0, 1, 1, 0],
        "ps": [24, 17, 14, 21],
        "pa": [17, 24, 21, 14],
    })

    current_strength, sos = nfl_team_strength.compute_strength_metrics(record)

    assert set(current_strength["team"]) == {"NYY", "BOS"}
    assert set(sos["team"]) == {"NYY", "BOS"}
    # Both teams are 1-1 overall, but current_strength as-of a team's most
    # recent game is built from STRICTLY EARLIER games only (no lookahead
    # onto the outcome of that most recent game itself) - so with only two
    # games each, "current" reduces to each team's week-1 result alone.
    # NYY won week 1 (and lost week 2); BOS lost week 1 (and won week 2).
    nyy_current = current_strength.set_index("team").loc["NYY", "current_strength"]
    bos_current = current_strength.set_index("team").loc["BOS", "current_strength"]
    assert nyy_current > bos_current


def test_compute_strength_metrics_pyth_strength_favors_larger_point_differential():
    # Two teams both 1-0, but A won by a lot more than B did - A's
    # pyth_strength (point-differential-based) should exceed B's even
    # though their real win% is identical.
    record = pd.DataFrame({
        "opp": ["X", "Y", "A", "B"],
        "team": ["A", "B", "X", "Y"],
        "season": [2025, 2025, 2025, 2025],
        "week": [1, 1, 1, 1],
        "game_id": ["g1", "g2", "g1", "g2"],
        "win": [1, 1, 0, 0],
        "loss": [0, 0, 1, 1],
        "ps": [40, 21, 10, 17],
        "pa": [10, 17, 40, 21],
    })
    # Add a second real game so each team has a real "current" (as-of
    # most-recent-game) rolling window to report - compute_strength_metrics
    # reports each team's rating AS OF their most recent game, built from
    # STRICTLY EARLIER games (via the shift), so a team's very first game
    # ever contributes 0 history to its own "current" rating; a second
    # game is needed for the first game's real point differential to show up.
    record2 = pd.DataFrame({
        "opp": ["Z", "Z", "A", "B"],
        "team": ["A", "B", "Z", "Z"],
        "season": [2025, 2025, 2025, 2025],
        "week": [2, 2, 2, 2],
        "game_id": ["g3", "g4", "g3", "g4"],
        "win": [1, 1, 0, 0],
        "loss": [0, 0, 1, 1],
        "ps": [17, 17, 10, 10],
        "pa": [10, 10, 17, 17],
    })
    full = pd.concat([record, record2], ignore_index=True)

    current_strength, _ = nfl_team_strength.compute_strength_metrics(full)
    by_team = current_strength.set_index("team")

    assert by_team.loc["A", "pyth_strength"] > by_team.loc["B", "pyth_strength"]


def test_compute_team_offense_defense_edge_uses_opponents_own_row():
    # Real pattern: what team B's offense produced against A that week IS
    # what A's defense allowed - taken from B's own team_stats row via
    # opponent_team, not recomputed.
    team_stats = pd.DataFrame([
        {"team": "A", "opponent_team": "B", "season": 2025, "week": 1, "game_id": "g1",
         "season_type": "REG", "passing_epa": 5.0, "rushing_epa": 2.0, "receiving_epa": 0.0},
        {"team": "B", "opponent_team": "A", "season": 2025, "week": 1, "game_id": "g1",
         "season_type": "REG", "passing_epa": -3.0, "rushing_epa": 1.0, "receiving_epa": 0.0},
        # A real playoff row that must be excluded entirely.
        {"team": "A", "opponent_team": "B", "season": 2025, "week": 19, "game_id": "g2",
         "season_type": "POST", "passing_epa": 99.0, "rushing_epa": 99.0, "receiving_epa": 99.0},
    ])

    result = nfl_team_strength.compute_team_offense_defense_edge(team_stats).set_index("team")

    assert result.loc["A", "offensive_edge"] == pytest.approx(7.0)  # A's own real EPA produced
    assert result.loc["A", "defensive_edge"] == pytest.approx(-2.0)  # B's own real EPA that game (what A allowed)
    assert result.loc["B", "offensive_edge"] == pytest.approx(-2.0)
    assert result.loc["B", "defensive_edge"] == pytest.approx(7.0)


def test_nfl_team_strength_windows_sum_to_one():
    assert sum(weight for _, weight in config.NFL_TEAM_STRENGTH_WINDOWS) == pytest.approx(1.0)


def _snap_row(team, season, week, pfr_player_id, offense_snaps, offense_pct=0.9, game_type="REG", position="QB"):
    return {
        "game_id": f"{season}_{week:02d}_{team}", "season": season, "week": week, "game_type": game_type,
        "team": team, "position": position, "pfr_player_id": pfr_player_id,
        "offense_snaps": offense_snaps, "offense_pct": offense_pct,
    }


def _weekly_row(player_id, season, week, passing_epa):
    return {
        "player_id": player_id, "position": "QB", "season": season, "week": week,
        "game_id": f"{season}_{week:02d}_{player_id}",
        "attempts": 30, "completions": 20, "passing_yards": 200, "passing_tds": 1,
        "passing_interceptions": 0, "carries": 2, "rushing_yards": 5, "rushing_tds": 0,
        "passing_epa": passing_epa,
    }


def _roster_row(season, gsis_id, pfr_id):
    return {"season": season, "gsis_id": gsis_id, "pfr_id": pfr_id}


def test_compute_qb_continuity_adjustment_identifies_real_recent_starter():
    # KC's QB1 (pfr "qb1_pfr") started weeks 1-3, then got hurt - QB2
    # ("qb2_pfr") took over weeks 4-6. Over the smallest games-back window
    # (4 games), QB2 has played MORE recent snaps than QB1, so QB2 should
    # be identified as the real recent primary starter, not QB1 even
    # though QB1 played more total games.
    snap_rows = (
        [_snap_row("KC", 2025, w, "qb1_pfr", offense_snaps=60) for w in range(1, 4)]
        + [_snap_row("KC", 2025, w, "qb2_pfr", offense_snaps=60) for w in range(4, 7)]
        # A real playoff row that must be excluded.
        + [_snap_row("KC", 2025, 19, "qb1_pfr", offense_snaps=60, game_type="WC")]
        # A real non-QB snap row (a WR) that must be excluded.
        + [_snap_row("KC", 2025, 6, "wr1_pfr", offense_snaps=55, position="WR")]
    )
    snap_counts = pd.DataFrame(snap_rows)
    weekly = pd.DataFrame([
        _weekly_row("qb1_gsis", 2025, w, passing_epa=5.0) for w in range(1, 4)
    ] + [
        _weekly_row("qb2_gsis", 2025, w, passing_epa=-3.0) for w in range(4, 7)
    ])
    rosters = pd.DataFrame([
        _roster_row(2025, "qb1_gsis", "qb1_pfr"),
        _roster_row(2025, "qb2_gsis", "qb2_pfr"),
    ])

    result = nfl_team_strength.compute_qb_continuity_adjustment(snap_counts, weekly, rosters).set_index("team")

    assert result.loc["KC", "recent_primary_qb_id"] == "qb2_gsis"
    assert result.loc["KC", "recent_primary_qb_epa"] == pytest.approx(-3.0)
    assert result.loc["KC", "recent_primary_qb_games"] == 3


def test_compute_qb_continuity_adjustment_missing_crosswalk_yields_zero_epa():
    # A QB with real snap-count rows but no real pfr_id<->gsis_id roster
    # crosswalk row at all - a real, honest missing value, not a
    # fabricated 0% share (mirrors nfl_bestball.compute_player_snap_share's
    # own documented behavior for this exact situation).
    snap_counts = pd.DataFrame([_snap_row("NYJ", 2025, w, "unmatched_pfr", offense_snaps=60) for w in range(1, 4)])
    weekly = pd.DataFrame([_weekly_row("some_other_gsis", 2025, 1, passing_epa=5.0)])
    rosters = pd.DataFrame([_roster_row(2025, "some_other_gsis", "some_other_pfr")])

    result = nfl_team_strength.compute_qb_continuity_adjustment(snap_counts, weekly, rosters).set_index("team")

    assert pd.isna(result.loc["NYJ", "recent_primary_qb_id"])
    assert result.loc["NYJ", "recent_primary_qb_epa"] == pytest.approx(0.0)
    assert result.loc["NYJ", "recent_primary_qb_games"] == 0


def test_assemble_team_metrics_real_season_shape():
    # Real, small multi-team/multi-week fixture (not the full season) -
    # checks the end-to-end assembly produces the right shape/columns
    # with no NaNs, real z-normalized values (mean should land near 1.0
    # across a large-enough real slate), matching teams.assemble_team_metrics'
    # own output contract. A genuine round-robin (each team faces a
    # DIFFERENT opponent each week, repeating the cycle) - not a fixed
    # A-vs-B/C-vs-D pairing every week, which would make every team's
    # only-ever opponent the one excluded by the opponent-exclusion
    # rolling window and degenerate `strength` to zero for everyone.
    teams = ["A", "B", "C", "D"]
    weekly_pairings = [
        [("A", "B"), ("C", "D")],
        [("A", "C"), ("B", "D")],
        [("A", "D"), ("B", "C")],
        [("A", "B"), ("C", "D")],
    ]
    # Each team has its own real, fixed EPA quality level (not identical
    # across teams) so offensive_edge/defensive_edge have real variance to
    # z-normalize - an all-teams-identical EPA fixture would give every
    # team a zero-std, all-NaN z-score, same failure mode as a degenerate
    # single-opponent schedule above.
    passing_epa_by_team = {"A": 2.0, "B": 1.0, "C": 0.5, "D": -0.5}
    rows = []
    ts_rows = []
    for week, pairings in enumerate(weekly_pairings, start=1):
        for game_num, (home, away) in enumerate(pairings, start=1):
            gid = f"2025_{week:02d}_{game_num}"
            rows.append(_game(gid, 2025, week, home, away, 24, 17))
            for team, opp in [(home, away), (away, home)]:
                ts_rows.append({
                    "team": team, "opponent_team": opp, "season": 2025, "week": week, "game_id": gid,
                    "season_type": "REG", "passing_epa": passing_epa_by_team[team],
                    "rushing_epa": 0.5, "receiving_epa": 0.0,
                })
    schedules = pd.DataFrame(rows)
    team_stats = pd.DataFrame(ts_rows)

    master = nfl_team_strength.assemble_team_metrics(schedules, team_stats)

    assert set(master["team"]) == set(teams)
    expected_cols = {
        "team", "current", "Strength", "pyth_Strength", "SOS", "pyth_SOS",
        "Confidence", "pyth_Confidence", "Confidence_Delta", "true_power",
        "offensive_edge", "defensive_edge",
        "games_played", "win_rate", "win_rate_CI_Low", "win_rate_CI_High",
    }
    assert set(master.columns) == expected_cols
    assert not master.isna().any().any()
    assert (master["games_played"] == 4).all()

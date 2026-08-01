import pandas as pd
import pytest

from mlb_metrics import config, nfl_dfs, nfl_passing, nfl_rush_rec


def _qb_row(player_id, season, week, passing_yards, passing_tds=2, ints=1, carries=3, rush_yards=15, rush_tds=0, two_pt=0):
    return {
        "player_id": player_id,
        "position": "QB",
        "season": season,
        "week": week,
        "game_id": f"{season}_{week:02d}_{player_id}",
        "attempts": 30,
        "completions": 20,
        "passing_yards": passing_yards,
        "passing_tds": passing_tds,
        "passing_interceptions": ints,
        "carries": carries,
        "rushing_yards": rush_yards,
        "rushing_tds": rush_tds,
        "passing_2pt_conversions": two_pt,
    }


def test_compute_qb_dk_points_matches_hand_computed_formula():
    # 10 games, weeks 1-10. Linear categories held CONSTANT across every
    # game (passing_tds=2, ints=1, rush_yards=15, rush_tds=0) so their
    # blended per-game rate trivially equals the constant regardless of
    # window weights - isolates the test to verifying the FULL formula
    # assembly (linear terms + bonus terms), not re-deriving window math
    # already covered by test_nfl_passing.py. Only passing_yards (for the
    # 300+ bonus) and 2pt conversions vary.
    rows = []
    for week in range(1, 9):
        rows.append(_qb_row("qb1", 2025, week, passing_yards=200))
    rows.append(_qb_row("qb1", 2025, 9, passing_yards=320, two_pt=1))
    rows.append(_qb_row("qb1", 2025, 10, passing_yards=340))
    weekly_df = pd.DataFrame(rows)

    qb_rolling = nfl_passing.compute_qb_rolling_stats(weekly_df)
    result = nfl_dfs.compute_qb_dk_points(qb_rolling, weekly_df).set_index("player_id")

    # Recency ranks: week10=0 (340), week9=1 (320,2pt=1), week8..1=2..9 (200 each).
    windows = dict(config.NFL_QB_WINDOWS)
    py_full, py_8, py_4 = (340 + 320 + 200 * 8) / 10, (340 + 320 + 200 * 6) / 8, (340 + 320 + 200 * 2) / 4
    blended_yards = py_full * windows[None] + py_8 * windows[8] + py_4 * windows[4]

    # cleared_300: week10 and week9 only.
    bonus_full, bonus_8, bonus_4 = 2 / 10, 2 / 8, 2 / 4
    blended_bonus_rate = bonus_full * windows[None] + bonus_8 * windows[8] + bonus_4 * windows[4]

    # 2pt: week9 only (value 1).
    twopt_full, twopt_8, twopt_4 = 1 / 10, 1 / 8, 1 / 4
    blended_2pt = twopt_full * windows[None] + twopt_8 * windows[8] + twopt_4 * windows[4]

    expected = (
        blended_yards * config.NFL_DK_PASS_YARD_POINTS
        + 2 * config.NFL_DK_PASS_TD_POINTS
        + 1 * config.NFL_DK_INTERCEPTION_POINTS
        + 15 * config.NFL_DK_RUSH_YARD_POINTS
        + 0 * config.NFL_DK_RUSH_TD_POINTS
        + blended_bonus_rate * config.NFL_DK_300_PASS_YARD_BONUS
        + blended_2pt * config.NFL_DK_2PT_POINTS
    )

    assert result.loc["qb1", "DK_Points_QB"] == pytest.approx(expected)
    assert result.loc["qb1", "Expected_300_Bonus_Rate"] == pytest.approx(blended_bonus_rate)


def _skill_row(player_id, position, season, week, rush_yards=0, rec_yards=0, receptions=0, targets=0, rush_tds=0, rec_tds=0, rush_fum_lost=0, rec_fum_lost=0, rush_2pt=0, rec_2pt=0):
    return {
        "player_id": player_id,
        "position": position,
        "season": season,
        "week": week,
        "game_id": f"{season}_{week:02d}_{player_id}",
        "carries": 10,
        "rushing_yards": rush_yards,
        "rushing_tds": rush_tds,
        "targets": targets,
        "receptions": receptions,
        "receiving_yards": rec_yards,
        "receiving_tds": rec_tds,
        "rushing_fumbles_lost": rush_fum_lost,
        "receiving_fumbles_lost": rec_fum_lost,
        "rushing_2pt_conversions": rush_2pt,
        "receiving_2pt_conversions": rec_2pt,
    }


def test_compute_skill_dk_points_scores_rush_and_rec_100_bonuses_separately():
    # A single game where the player clears BOTH the rush-100 and rec-100
    # bonus in the same game - both must be scored, not just one.
    rows = [_skill_row("rb1", "RB", 2025, 1, rush_yards=120, rec_yards=110, receptions=5, rush_tds=1)]
    weekly_df = pd.DataFrame(rows)

    skill_rolling = nfl_rush_rec.compute_skill_rolling_stats(weekly_df)
    result = nfl_dfs.compute_skill_dk_points(skill_rolling, weekly_df).set_index("player_id")

    expected = (
        120 * config.NFL_DK_RUSH_YARD_POINTS
        + 1 * config.NFL_DK_RUSH_TD_POINTS
        + 110 * config.NFL_DK_RECEIVING_YARD_POINTS
        + 5 * config.NFL_DK_RECEPTION_POINTS
        + 1.0 * config.NFL_DK_100_YARD_BONUS  # rush 100+
        + 1.0 * config.NFL_DK_100_YARD_BONUS  # rec 100+
    )
    assert result.loc["rb1", "DK_Points_Skill"] == pytest.approx(expected)
    assert result.loc["rb1", "Expected_Rush_100_Bonus_Rate"] == pytest.approx(1.0)
    assert result.loc["rb1", "Expected_Rec_100_Bonus_Rate"] == pytest.approx(1.0)


def test_compute_skill_dk_points_fumble_lost_is_negative():
    rows = [_skill_row("rb1", "RB", 2025, 1, rush_yards=50, rush_fum_lost=1)]
    weekly_df = pd.DataFrame(rows)

    skill_rolling = nfl_rush_rec.compute_skill_rolling_stats(weekly_df)
    result = nfl_dfs.compute_skill_dk_points(skill_rolling, weekly_df).set_index("player_id")

    expected = 50 * config.NFL_DK_RUSH_YARD_POINTS + 1 * config.NFL_DK_FUMBLE_LOST_POINTS
    assert result.loc["rb1", "DK_Points_Skill"] == pytest.approx(expected)

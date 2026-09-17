import numpy as np
import pandas as pd
import pytest

from mlb_metrics import config, nfl_prop_projections


def _row(
    player_id, team, opponent_team, position, season, week,
    targets=0, receptions=0, receiving_yards=0, carries=0, rushing_yards=0,
    receiving_epa=0.0, rushing_epa=0.0,
):
    return {
        "player_id": player_id,
        "team": team,
        "opponent_team": opponent_team,
        "position": position,
        "season": season,
        "week": week,
        "season_type": "REG",
        "targets": targets,
        "receptions": receptions,
        "receiving_yards": receiving_yards,
        "carries": carries,
        "rushing_yards": rushing_yards,
        "receiving_epa": receiving_epa,
        "rushing_epa": rushing_epa,
    }


def _weeks(player_id, team, opponent, position, weeks, **stats):
    return [_row(player_id, team, opponent, position, 2025, week, **stats) for week in weeks]


def test_recency_weights_decay_with_the_configured_windows():
    ranked = pd.Series([0, 3, 4, 7, 8, 20])
    weights = nfl_prop_projections._recency_weights(ranked, [(None, 0.20), (8, 0.30), (4, 0.50)])

    # Inside all three windows, then inside two, then only the open one.
    assert list(weights) == [1.0, 1.0, 0.5, 0.5, 0.2, 0.2]


def test_shrink_rate_pulls_small_samples_toward_the_prior():
    # 1 target, 30 yards - an unshrunk 30.0 yards per target. With a prior
    # of 7.0 carrying 20 synthetic attempts it lands near the prior.
    shrunk = nfl_prop_projections._shrink_rate(pd.Series([30.0]), pd.Series([1.0]), 7.0, 20.0)
    assert shrunk.iloc[0] == pytest.approx((30.0 + 20.0 * 7.0) / 21.0)
    assert 7.0 < shrunk.iloc[0] < 9.0

    # A large sample is left essentially alone.
    barely = nfl_prop_projections._shrink_rate(pd.Series([3000.0]), pd.Series([300.0]), 7.0, 20.0)
    assert barely.iloc[0] == pytest.approx((3000.0 + 140.0) / 320.0)
    assert 9.5 < barely.iloc[0] < 10.0


def test_shrink_rate_returns_the_prior_for_zero_attempts():
    shrunk = nfl_prop_projections._shrink_rate(pd.Series([0.0]), pd.Series([0.0]), 6.5, 20.0)
    assert shrunk.iloc[0] == pytest.approx(6.5)


def test_player_usage_shares_sum_to_one_across_a_team():
    rows = []
    rows += _weeks("wr1", "SF", "SEA", "WR", range(1, 6), targets=12, carries=0)
    rows += _weeks("wr2", "SF", "SEA", "WR", range(1, 6), targets=8, carries=0)
    rows += _weeks("rb1", "SF", "SEA", "RB", range(1, 6), targets=0, carries=20)

    usage = nfl_prop_projections.compute_player_usage(pd.DataFrame(rows)).set_index("player_id")

    assert usage.loc["wr1", "target_share"] == pytest.approx(0.6)
    assert usage.loc["wr2", "target_share"] == pytest.approx(0.4)
    assert usage["target_share"].sum() == pytest.approx(1.0)
    assert usage.loc["rb1", "carry_share"] == pytest.approx(1.0)


def test_team_volume_and_usage_share_a_denominator():
    """The projection multiplies a share back against team volume, so the
    two must be built from the same denominator or every projected target
    is silently mis-scaled."""
    rows = []
    rows += _weeks("wr1", "SF", "SEA", "WR", range(1, 6), targets=12)
    rows += _weeks("wr2", "SF", "SEA", "WR", range(1, 6), targets=8)
    weekly = pd.DataFrame(rows)

    usage = nfl_prop_projections.compute_player_usage(weekly).set_index("player_id")
    volume = nfl_prop_projections.compute_team_volume(weekly).set_index("team")

    assert volume.loc["SF", "team_targets_per_game"] == pytest.approx(20.0)
    reconstructed = usage["target_share"] * volume.loc["SF", "team_targets_per_game"]
    assert reconstructed.loc["wr1"] == pytest.approx(12.0)
    assert reconstructed.loc["wr2"] == pytest.approx(8.0)


def test_defense_per_play_rates_are_volume_invariant():
    """The defect this whole module exists to fix: a defense that merely
    FACES more targets must not look worse per play than one that faces
    fewer at identical efficiency."""
    rows = []
    # SEA faces two receivers a week, NE faces one - identical 7.0 yards
    # per target in both cases, double the per-game yardage for SEA.
    rows += _weeks("sf1", "SF", "SEA", "WR", range(1, 6), targets=10, receiving_yards=70, receptions=6)
    rows += _weeks("sf2", "SF", "SEA", "WR", range(1, 6), targets=10, receiving_yards=70, receptions=6)
    rows += _weeks("kc1", "KC", "NE", "WR", range(1, 6), targets=10, receiving_yards=70, receptions=6)

    rates = nfl_prop_projections.compute_position_defense_per_play_rates(pd.DataFrame(rows))
    rates = rates.set_index(["team", "position"])

    assert rates.loc[("SEA", "WR"), "yards_allowed_per_target"] == pytest.approx(7.0)
    assert rates.loc[("NE", "WR"), "yards_allowed_per_target"] == pytest.approx(7.0)
    # SEA genuinely faced twice the volume - the rate is invariant to it,
    # but the attempt count still records it.
    assert rates.loc[("SEA", "WR"), "targets_faced"] == pytest.approx(
        2 * rates.loc[("NE", "WR"), "targets_faced"]
    )


def test_defense_recency_window_counts_games_not_rows():
    """A defense that faces many receivers in one week must not burn its
    whole full-weight window on that single week."""
    rows = []
    # CHI faces 5 different receivers every week for 5 weeks: 25 rows, but
    # only 5 games. Every game is identical, so any correct weighting
    # returns exactly the per-target rate regardless of the window.
    for index in range(5):
        rows += _weeks(f"wr{index}", "GB", "CHI", "WR", range(1, 6), targets=4, receiving_yards=28, receptions=3)

    rates = nfl_prop_projections.compute_position_defense_per_play_rates(pd.DataFrame(rows))
    rates = rates.set_index(["team", "position"])

    assert rates.loc[("CHI", "WR"), "yards_allowed_per_target"] == pytest.approx(7.0)
    # 5 games x 20 targets, weighted 1.0/1.0/1.0/1.0/0.5 - if the rank had
    # counted rows instead of games this would come out far lower.
    assert rates.loc[("CHI", "WR"), "targets_faced"] == pytest.approx(20 * 4.5)


def test_blend_multiplier_is_a_no_op_at_zero_weight():
    rates = pd.DataFrame({
        "position": ["WR", "WR", "WR"],
        "ypt_multiplier": [0.9, 1.0, 1.2],
        "epa": [0.5, -0.2, 0.1],
    })
    blended = nfl_prop_projections._blend_multiplier(rates, "ypt_multiplier", "epa", "position", 0.0)
    assert list(blended) == [0.9, 1.0, 1.2]


def test_blend_multiplier_reorders_but_preserves_centre_and_spread():
    rates = pd.DataFrame({
        "position": ["WR"] * 4,
        "ypt_multiplier": [0.90, 0.98, 1.02, 1.10],
        # EPA ranks the same four defenses in the OPPOSITE order.
        "epa": [0.30, 0.10, -0.10, -0.30],
    })
    blended = nfl_prop_projections._blend_multiplier(rates, "ypt_multiplier", "epa", "position", 1.0)

    # Pure EPA weighting flips the order outright ...
    assert list(blended.rank()) == [4.0, 3.0, 2.0, 1.0]
    # ... while keeping the yards multiplier's own centre and spread, so
    # the result is still something that can scale a yards projection.
    assert blended.mean() == pytest.approx(rates["ypt_multiplier"].mean())
    assert blended.std() == pytest.approx(rates["ypt_multiplier"].std())


def test_blend_multiplier_survives_a_zero_variance_group():
    rates = pd.DataFrame({
        "position": ["TE", "TE"],
        "ypt_multiplier": [1.05, 0.95],
        "epa": [0.2, 0.2],
    })
    blended = nfl_prop_projections._blend_multiplier(rates, "ypt_multiplier", "epa", "position", 0.5)
    assert list(blended) == [1.05, 0.95]


def test_projection_is_volume_times_efficiency_times_matchup():
    rows = []
    # SF: wr1 takes 12 of 20 targets at 8.0 yards each; SEA is exactly
    # league average, so the matchup multiplier is 1.0 and the projection
    # reduces to share x volume x efficiency.
    rows += _weeks("wr1", "SF", "SEA", "WR", range(1, 6), targets=12, receiving_yards=96, receptions=8)
    rows += _weeks("wr2", "SF", "SEA", "WR", range(1, 6), targets=8, receiving_yards=64, receptions=4)
    weekly = pd.DataFrame(rows)
    opponents = pd.DataFrame([{"team": "SF", "opponent": "SEA"}])

    projections = nfl_prop_projections.project_player_stats(weekly, opponents).set_index("player_id")

    assert projections.loc["wr1", "projected_targets"] == pytest.approx(12.0)
    # Only one defense exists in this fixture, so it IS the league and its
    # multiplier must be exactly neutral.
    assert projections.loc["wr1", "ypt_multiplier"] == pytest.approx(1.0)
    expected = 12.0 * projections.loc["wr1", "yards_per_target"]
    assert projections.loc["wr1", "projected_receiving_yards"] == pytest.approx(expected)


def test_projection_separates_players_who_share_a_defense():
    """Jefferson and Jennings: same team, same opponent, different usage.
    The ratio model gave them an identical call; the projection must
    not."""
    rows = []
    rows += _weeks("hog", "MIN", "CHI", "WR", range(1, 6), targets=16, receiving_yards=112, receptions=10)
    rows += _weeks("complementary", "MIN", "CHI", "WR", range(1, 6), targets=4, receiving_yards=28, receptions=3)
    weekly = pd.DataFrame(rows)
    opponents = pd.DataFrame([{"team": "MIN", "opponent": "CHI"}])

    projections = nfl_prop_projections.project_player_stats(weekly, opponents).set_index("player_id")

    assert projections.loc["hog", "ypt_multiplier"] == pytest.approx(
        projections.loc["complementary", "ypt_multiplier"]
    )
    assert projections.loc["hog", "projected_receiving_yards"] == pytest.approx(
        4 * projections.loc["complementary", "projected_receiving_yards"]
    )


def test_project_player_stats_returns_the_full_schema_when_empty():
    empty = pd.DataFrame(columns=list(_row("x", "SF", "SEA", "WR", 2025, 1).keys()))
    projections = nfl_prop_projections.project_player_stats(empty, pd.DataFrame(columns=["team", "opponent"]))

    assert projections.empty
    # Callers select projection columns by name - an empty result still
    # has to be selectable, or "nobody qualified" becomes a KeyError.
    for column in nfl_prop_projections.PROJECTION_OUTPUT_COLUMNS:
        assert column in projections.columns


def test_projection_clip_bounds_the_matchup_multiplier():
    low, high = config.NFL_PROP_PROJECTION_MULTIPLIER_CLIP
    rows = []
    # One absurdly generous defense and one absurdly stingy one.
    rows += _weeks("a", "SF", "SEA", "WR", range(1, 6), targets=20, receiving_yards=600, receptions=18)
    rows += _weeks("b", "KC", "NE", "WR", range(1, 6), targets=20, receiving_yards=20, receptions=2)
    weekly = pd.DataFrame(rows)

    rates = nfl_prop_projections.compute_opponent_multipliers(
        nfl_prop_projections.compute_position_defense_per_play_rates(weekly),
        nfl_prop_projections.league_efficiency_rates(weekly),
    )

    assert rates["ypt_multiplier"].max() <= high + 1e-9
    assert rates["ypt_multiplier"].min() >= low - 1e-9


def test_league_efficiency_rates_are_attempt_weighted_not_player_averaged():
    """A 1-target cameo must not count as much as a 100-target starter
    when setting the prior every other rate is shrunk toward."""
    rows = []
    rows += _weeks("starter", "SF", "SEA", "WR", [1], targets=100, receiving_yards=700, receptions=60)
    rows += _weeks("cameo", "SF", "SEA", "WR", [1], targets=1, receiving_yards=80, receptions=1)

    league = nfl_prop_projections.league_efficiency_rates(pd.DataFrame(rows)).set_index("position")

    # Attempt-weighted: 780 yards on 101 targets.
    assert league.loc["WR", "yards_per_target"] == pytest.approx(780 / 101)
    # An average of the two players' rates would have been (7.0 + 80.0)/2.
    assert league.loc["WR", "yards_per_target"] < 10.0

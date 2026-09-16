import pandas as pd
import pytest

from mlb_metrics import config, nfl_player_props


def _row(
    player_id, player_name, team, opponent_team, position, season, week,
    passing_yards=0, rushing_yards=0, receiving_yards=0, receptions=0, targets=0, carries=0,
    def_sacks=0.0, sacks_suffered=0.0,
):
    return {
        "player_id": player_id,
        "player_display_name": player_name,
        "team": team,
        "opponent_team": opponent_team,
        "position": position,
        "season": season,
        "week": week,
        "season_type": "REG",
        "game_id": f"{season}_{week:02d}_{team}_{opponent_team}",
        "attempts": 0,
        "completions": 0,
        "passing_yards": passing_yards,
        "passing_tds": 0,
        "passing_interceptions": 0,
        "passing_epa": 0.0,
        "carries": carries,
        "rushing_yards": rushing_yards,
        "rushing_tds": 0,
        "targets": targets,
        "receptions": receptions,
        "receiving_yards": receiving_yards,
        "receiving_tds": 0,
        "rushing_fumbles_lost": 0,
        "receiving_fumbles_lost": 0,
        "def_sacks": def_sacks,
        "sacks_suffered": sacks_suffered,
    }


def test_nfl_pass_rush_windows_sum_to_one():
    assert sum(weight for _, weight in config.NFL_PASS_RUSH_WINDOWS) == pytest.approx(1.0)


def test_compute_pass_rush_rolling_stats_blends_windows():
    rows = [
        _row("edge1", "Edge One", "SF", "SEA", "DE", 2025, week, def_sacks=1.0)
        for week in range(1, 6)
    ]
    weekly_df = pd.DataFrame(rows)

    result = nfl_player_props.compute_pass_rush_rolling_stats(weekly_df).set_index("player_id")

    assert result.loc["edge1", "games"] == 5
    assert result.loc["edge1", "team"] == "SF"
    assert result.loc["edge1", "position"] == "DE"
    # Every window sees the same 5 games at 1.0 sacks/game each - the
    # blend is exactly 1.0 regardless of window weights.
    assert result.loc["edge1", "sacks_per_game"] == pytest.approx(1.0)


def test_compute_pass_rush_rolling_stats_excludes_players_who_never_recorded_a_sack():
    rows = [
        _row("edge1", "Edge One", "SF", "SEA", "DE", 2025, 1, def_sacks=1.0),
        _row("cb1", "Corner One", "SF", "SEA", "CB", 2025, 1, def_sacks=0.0),
    ]
    weekly_df = pd.DataFrame(rows)

    result = nfl_player_props.compute_pass_rush_rolling_stats(weekly_df)

    assert list(result["player_id"]) == ["edge1"]


def test_compute_sacks_allowed_rolling_rates_blends_windows():
    rows = [
        _row("qb1", "QB One", "NYJ", "MIA", "QB", 2025, week, sacks_suffered=2.0)
        for week in range(1, 6)
    ]
    weekly_df = pd.DataFrame(rows)

    result = nfl_player_props.compute_sacks_allowed_rolling_rates(weekly_df).set_index("team")

    assert result.loc["NYJ", "games"] == 5
    assert result.loc["NYJ", "sacks_allowed_per_game"] == pytest.approx(2.0)


def test_compute_sacks_allowed_rolling_rates_sums_every_real_player_on_the_team_week():
    # A real, rare non-QB sacks_suffered row (e.g. a trick play) still
    # counts toward that team's total real sacks allowed that week.
    rows = [
        _row("qb1", "QB One", "NYJ", "MIA", "QB", 2025, 1, sacks_suffered=2.0),
        _row("wr1", "WR One", "NYJ", "MIA", "WR", 2025, 1, sacks_suffered=1.0),
    ]
    weekly_df = pd.DataFrame(rows)

    result = nfl_player_props.compute_sacks_allowed_rolling_rates(weekly_df).set_index("team")

    assert result.loc["NYJ", "sacks_allowed_per_game"] == pytest.approx(3.0)


def _multi_week_skill_rows(player_id, name, team, opponent, position, weeks, **stat_kwargs):
    return [_row(player_id, name, team, opponent, position, 2025, week, **stat_kwargs) for week in weeks]


def test_build_prop_edges_covers_all_five_categories():
    rows = []
    rows += _multi_week_skill_rows("wr1", "WR One", "SF", "SEA", "WR", range(1, 6), receptions=5, receiving_yards=60)
    rows += _multi_week_skill_rows("rb1", "RB One", "SF", "SEA", "RB", range(1, 6), rushing_yards=80, carries=15)
    rows += _multi_week_skill_rows("qb1", "QB One", "SF", "SEA", "QB", range(1, 6), passing_yards=250)
    rows += _multi_week_skill_rows("edge1", "Edge One", "SF", "SEA", "DE", range(1, 6), def_sacks=1.0)
    # A real opposing defense (SEA) whose players allowed the above -
    # give SEA's own offense a real sacks_suffered history too.
    rows += _multi_week_skill_rows("seaqb", "SEA QB", "SEA", "SF", "QB", range(1, 6), sacks_suffered=2.0)
    weekly_df = pd.DataFrame(rows)

    current_week_schedule = pd.DataFrame([{"home_team": "SF", "away_team": "SEA"}])

    edges = nfl_player_props.build_prop_edges(weekly_df, current_week_schedule)

    assert set(edges["category"]) == {"Receptions", "Receiving Yards", "Rushing Yards", "Passing Yards", "Sacks"}
    assert set(edges.columns) == {
        "player_id", "player_name", "team", "position", "opponent", "games",
        "category", "player_rate", "opponent_allowed_rate", "league_rate", "ratio",
    }


def test_build_prop_edges_favorable_matchup_produces_a_ratio_above_one():
    # SEA is a real, comparatively weak pass defense (allows more
    # passing yards than NE) - a WR on the team facing SEA should get
    # ratio > 1. A defense's own real "allowed" rate comes from what the
    # OFFENSES FACING IT produced (nfl_teams.compute_team_week_allowed
    # groups by `opponent_team`) - so it's SF's own QB passing_yards
    # (team=SF, opponent_team=SEA) that sets SEA's real allowed rate,
    # NOT a row on SEA's own team.
    rows = []
    rows += _multi_week_skill_rows("wr_vs_weak", "WR Vs Weak", "SF", "SEA", "WR", range(1, 6), receiving_yards=60, receptions=5)
    rows += _multi_week_skill_rows("wr_vs_strong", "WR Vs Strong", "KC", "NE", "WR", range(1, 6), receiving_yards=60, receptions=5)
    # SEA allows a lot of real passing yards; NE allows very little.
    rows += _multi_week_skill_rows("sf_qb", "SF QB", "SF", "SEA", "QB", range(1, 6), passing_yards=350)
    rows += _multi_week_skill_rows("kc_qb", "KC QB", "KC", "NE", "QB", range(1, 6), passing_yards=50)
    weekly_df = pd.DataFrame(rows)

    current_week_schedule = pd.DataFrame([{"home_team": "SF", "away_team": "SEA"}, {"home_team": "KC", "away_team": "NE"}])

    edges = nfl_player_props.build_prop_edges(weekly_df, current_week_schedule)
    rec_yards = edges[edges["category"] == "Receiving Yards"].set_index("player_id")

    assert rec_yards.loc["wr_vs_weak", "ratio"] > 1.0
    assert rec_yards.loc["wr_vs_strong", "ratio"] < 1.0


def test_build_prop_edges_drops_no_opponent_for_a_bye_team():
    rows = _multi_week_skill_rows("wr1", "WR One", "SF", "SEA", "WR", range(1, 6), receptions=5, receiving_yards=60)
    weekly_df = pd.DataFrame(rows)
    # SF has no game in this week's real schedule (a bye).
    current_week_schedule = pd.DataFrame([{"home_team": "KC", "away_team": "DEN"}])

    edges = nfl_player_props.build_prop_edges(weekly_df, current_week_schedule)
    receptions = edges[edges["category"] == "Receptions"].set_index("player_id")

    assert pd.isna(receptions.loc["wr1", "opponent"])


def test_top_prop_bets_filters_below_min_games():
    rows = _multi_week_skill_rows("wr1", "WR One", "SF", "SEA", "WR", [1], receptions=10, receiving_yards=150)
    weekly_df = pd.DataFrame(rows)
    current_week_schedule = pd.DataFrame([{"home_team": "SF", "away_team": "SEA"}])

    edges = nfl_player_props.build_prop_edges(weekly_df, current_week_schedule)
    top = nfl_player_props.top_prop_bets(edges, n=10, min_games=3)

    assert top.empty


def test_top_prop_bets_filters_below_category_min_usage():
    # A real, low-usage cameo (1 catch/game) should never qualify for
    # the Receptions category, however extreme its matchup ratio.
    rows = _multi_week_skill_rows("wr_scrub", "WR Scrub", "SF", "SEA", "WR", range(1, 6), receptions=1, receiving_yards=10)
    weekly_df = pd.DataFrame(rows)
    current_week_schedule = pd.DataFrame([{"home_team": "SF", "away_team": "SEA"}])

    edges = nfl_player_props.build_prop_edges(weekly_df, current_week_schedule)
    top = nfl_player_props.top_prop_bets(edges, n=10)

    assert "wr_scrub" not in set(top["player_id"]) if "player_id" in top.columns else True
    assert top.empty


def test_top_prop_bets_labels_over_and_under_correctly():
    rows = []
    rows += _multi_week_skill_rows("wr_vs_weak", "WR Vs Weak", "SF", "SEA", "WR", range(1, 6), receiving_yards=60, receptions=5)
    rows += _multi_week_skill_rows("wr_vs_strong", "WR Vs Strong", "KC", "NE", "WR", range(1, 6), receiving_yards=60, receptions=5)
    rows += _multi_week_skill_rows("sf_qb", "SF QB", "SF", "SEA", "QB", range(1, 6), passing_yards=350)
    rows += _multi_week_skill_rows("kc_qb", "KC QB", "KC", "NE", "QB", range(1, 6), passing_yards=50)
    weekly_df = pd.DataFrame(rows)
    current_week_schedule = pd.DataFrame([{"home_team": "SF", "away_team": "SEA"}, {"home_team": "KC", "away_team": "NE"}])

    edges = nfl_player_props.build_prop_edges(weekly_df, current_week_schedule)
    top = nfl_player_props.top_prop_bets(edges, n=10).set_index("player_id")

    rec_yards_rows = top[top["category"] == "Receiving Yards"]
    assert rec_yards_rows.loc["wr_vs_weak", "direction"] == "Over"
    assert rec_yards_rows.loc["wr_vs_strong", "direction"] == "Under"


def test_top_prop_bets_percentile_ranking_does_not_let_one_category_dominate():
    # Real, confirmed necessary fix (2026-09-16): Sacks' own real
    # allowed-rate is a far noisier per-game counting stat than
    # receiving yardage, so its raw ratio swings much further from 1.0
    # for a comparably "unusual" real matchup - ranking on raw
    # abs(ratio - 1) let Sacks alone crowd out every other real
    # category.
    #
    # This fixture gives Receiving Yards a real, but modest (well within
    # the clip range) ratio swing, and Sacks a much LARGER, fully
    # clip-saturated real ratio swing for its own single real favorable
    # matchup - a real, deliberately unfair raw-magnitude gap the old
    # abs(ratio - 1) ranking would always resolve in Sacks' favor. Each
    # category has exactly one real favorable and one real unfavorable
    # matchup, so with n=2, a percentile-based ranking (each category's
    # own real most-favorable row reaches the exact same top real
    # percentile regardless of its own raw ratio's magnitude) picks
    # exactly one row from EACH category - proving raw magnitude no
    # longer decides the outcome.
    rows = []
    # Receiving Yards: a real, modest, NOT clip-saturated ratio swing.
    # Each QB row ALSO carries a real, neutral-ish sacks_suffered value
    # so these 4 teams don't distort the Sacks league average below
    # (each real category's league rate is computed over the WHOLE
    # real weekly_df, so a team with no real value for the other
    # category's stat would otherwise drag that other category's real
    # league average toward zero, an unwanted cross-category artifact
    # of this fixture, not a real property of either category).
    rows += _multi_week_skill_rows("wr_fav", "WR Favorable", "SF", "SEA", "WR", range(1, 6), receiving_yards=60, receptions=5)
    rows += _multi_week_skill_rows("wr_unfav", "WR Unfavorable", "KC", "NE", "WR", range(1, 6), receiving_yards=60, receptions=5)
    rows += _multi_week_skill_rows("sf_qb", "SF QB", "SF", "SEA", "QB", range(1, 6), passing_yards=230, sacks_suffered=2.0)
    rows += _multi_week_skill_rows("kc_qb", "KC QB", "KC", "NE", "QB", range(1, 6), passing_yards=180, sacks_suffered=2.0)
    # Sacks: a real, fully clip-saturated (much larger raw) ratio swing.
    rows += _multi_week_skill_rows("edge_fav", "Edge Favorable", "GB", "CHI", "DE", range(1, 6), def_sacks=1.0)
    rows += _multi_week_skill_rows("edge_unfav", "Edge Unfavorable", "LAR", "ARI", "DE", range(1, 6), def_sacks=1.0)
    rows += _multi_week_skill_rows("chi_qb", "CHI QB", "CHI", "GB", "QB", range(1, 6), sacks_suffered=6.0, passing_yards=200)
    rows += _multi_week_skill_rows("ari_qb", "ARI QB", "ARI", "LAR", "QB", range(1, 6), sacks_suffered=1.0, passing_yards=200)
    # A real, neutral passing_yards row for GB/LAR too - without this,
    # CHI/ARI would each show a real 0.0 pass_yards_allowed_per_game
    # (nobody else in this fixture ever plays AGAINST them with a real
    # passing_yards value), an artificial, unwanted drag on the
    # Receiving Yards league average that has nothing to do with either
    # real category this test is actually about.
    rows += _multi_week_skill_rows("gb_qb", "GB QB", "GB", "CHI", "QB", range(1, 6), passing_yards=200)
    rows += _multi_week_skill_rows("lar_qb", "LAR QB", "LAR", "ARI", "QB", range(1, 6), passing_yards=200)
    weekly_df = pd.DataFrame(rows)
    current_week_schedule = pd.DataFrame(
        [
            {"home_team": "SF", "away_team": "SEA"},
            {"home_team": "KC", "away_team": "NE"},
            {"home_team": "GB", "away_team": "CHI"},
            {"home_team": "LAR", "away_team": "ARI"},
        ]
    )

    edges = nfl_player_props.build_prop_edges(weekly_df, current_week_schedule)
    receiving_yards = edges[edges["category"] == "Receiving Yards"]
    sacks = edges[edges["category"] == "Sacks"]
    # Confirm the fixture actually produces the intended real gap in
    # raw magnitude before trusting the ranking assertion below.
    assert (receiving_yards["ratio"] - 1).abs().max() < (sacks["ratio"] - 1).abs().max()

    # This fixture's own Passing Yards rows also happen to reach the
    # exact same top real edge_percentile as Receiving Yards/Sacks (a
    # real 3-way tie at the top, not just 2) - n=3 (not 2) so the
    # secondary usage-based tiebreak added for real ties (see
    # top_prop_bets' own docstring) doesn't arbitrarily bump one of the
    # three out of an artificially small top-n window; the real point
    # of this test - that Sacks' own huge raw ratio magnitude doesn't
    # buy it anything over a category with a far smaller raw gap - still
    # holds regardless of which of the tied three the tiebreak orders
    # first.
    top = nfl_player_props.top_prop_bets(edges, n=3)

    assert set(top["category"]) == {"Receiving Yards", "Sacks", "Passing Yards"}
    assert set(top["direction"]) == {"Over"}


def test_top_prop_bets_breaks_ties_by_usage_not_incidental_team_order():
    # Real, confirmed user report (2026-09-16): "the sort seems to be by
    # team." Root cause: opponent_allowed_rate is a real TEAM-level stat,
    # so every player on the same team in the same category shares the
    # exact same real ratio/edge_percentile - without a real tiebreak,
    # a stable sort's own incidental row order (grouped by team from how
    # build_prop_edges assembles rows) silently decided the order.
    # These three SF receivers all face the same real SEA defense in the
    # same real category, so their real ratios tie exactly - only their
    # own real usage (receptions/game) should decide the order among
    # them, highest usage first.
    rows = []
    rows += _multi_week_skill_rows("wr_low", "WR Low", "SF", "SEA", "WR", range(1, 6), receptions=3, receiving_yards=40)
    rows += _multi_week_skill_rows("wr_high", "WR High", "SF", "SEA", "WR", range(1, 6), receptions=9, receiving_yards=40)
    rows += _multi_week_skill_rows("wr_mid", "WR Mid", "SF", "SEA", "WR", range(1, 6), receptions=6, receiving_yards=40)
    rows += _multi_week_skill_rows("sea_qb", "SEA QB", "SEA", "SF", "QB", range(1, 6), sacks_suffered=2.0, passing_yards=200)
    weekly_df = pd.DataFrame(rows)
    current_week_schedule = pd.DataFrame([{"home_team": "SF", "away_team": "SEA"}])

    edges = nfl_player_props.build_prop_edges(weekly_df, current_week_schedule)
    receptions_only = edges[edges["category"] == "Receptions"]
    # Confirm the fixture actually produces the intended real tie before
    # trusting the ordering assertion below.
    assert receptions_only["ratio"].nunique() == 1

    top = nfl_player_props.top_prop_bets(edges, n=10)
    ordered_ids = list(top[top["category"] == "Receptions"]["player_id"])

    assert ordered_ids == ["wr_high", "wr_mid", "wr_low"]

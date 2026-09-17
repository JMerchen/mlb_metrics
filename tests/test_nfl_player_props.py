import numpy as np
import pandas as pd
import pytest

from mlb_metrics import config, nfl_player_props


def _row(
    player_id, player_name, team, opponent_team, position, season, week,
    passing_yards=0, rushing_yards=0, receiving_yards=0, receptions=0, targets=0, carries=0,
    def_sacks=0.0, sacks_suffered=0.0, attempts=0,
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
        "attempts": attempts,
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


def test_compute_position_defense_rolling_rates_splits_allowed_rate_by_position():
    # Real, confirmed user report (2026-09-16): "opp allows... neither
    # seems accurate, nor at a player level, and is not separated by
    # position" - MIN allowing a real 14.1 receptions/game (a blanket
    # team-wide number covering WRs+TEs+RBs together) said nothing about
    # how MIN specifically defends TEs. Here, MIN allows a real, much
    # BIGGER volume to opposing TEs than opposing WRs - a team-wide
    # number would average the two together and hide this real split.
    rows = []
    rows += _multi_week_skill_rows("te1", "TE One", "CHI", "MIN", "TE", range(1, 6), receptions=8)
    rows += _multi_week_skill_rows("wr1", "WR One", "CHI", "MIN", "WR", range(1, 6), receptions=2)
    weekly_df = pd.DataFrame(rows)

    result = nfl_player_props.compute_position_defense_rolling_rates(weekly_df, "receptions").set_index(["team", "position"])

    assert result.loc[("MIN", "TE"), "allowed_per_game"] == pytest.approx(8.0)
    assert result.loc[("MIN", "WR"), "allowed_per_game"] == pytest.approx(2.0)


def test_compute_position_defense_rolling_rates_only_covers_skill_positions():
    # A real QB row (e.g. a scramble/garbage-time stat) should never
    # pollute a real RB/WR/TE-facing defense's own allowed rate.
    rows = _multi_week_skill_rows("qb1", "QB One", "CHI", "MIN", "QB", range(1, 6), rushing_yards=20)
    weekly_df = pd.DataFrame(rows)

    result = nfl_player_props.compute_position_defense_rolling_rates(weekly_df, "rushing_yards")

    assert result.empty


def test_build_prop_edges_covers_all_five_categories():
    rows = []
    rows += _multi_week_skill_rows("wr1", "WR One", "SF", "SEA", "WR", range(1, 6), receptions=5, receiving_yards=60)
    rows += _multi_week_skill_rows("rb1", "RB One", "SF", "SEA", "RB", range(1, 6), rushing_yards=80, carries=15)
    rows += _multi_week_skill_rows("qb1", "QB One", "SF", "SEA", "QB", range(1, 6), passing_yards=250, attempts=35)
    rows += _multi_week_skill_rows("edge1", "Edge One", "SF", "SEA", "DE", range(1, 6), def_sacks=1.0)
    # A real opposing defense (SEA) whose players allowed the above -
    # give SEA's own offense a real sacks_suffered history too.
    rows += _multi_week_skill_rows(
        "seaqb", "SEA QB", "SEA", "SF", "QB", range(1, 6),
        sacks_suffered=2.0, passing_yards=200, attempts=30,
    )
    weekly_df = pd.DataFrame(rows)

    current_week_schedule = pd.DataFrame([{"home_team": "SF", "away_team": "SEA"}])

    edges = nfl_player_props.build_prop_edges(weekly_df, current_week_schedule)

    assert set(edges["category"]) == {"Receptions", "Receiving Yards", "Rushing Yards", "Passing Yards", "Sacks"}
    assert set(edges.columns) == {
        "player_id", "player_name", "team", "position", "opponent", "games",
        "category", "player_rate", "projection", "projected_targets", "projected_carries",
        "opponent_allowed_rate", "league_rate", "rate_basis", "ratio",
    }
    # Four of the five categories carry a projection. Sacks alone does
    # not, and that is measured rather than missing: a projected-sacks
    # model was built and backtested and came out materially WORSE than
    # the ratio it would have replaced (hit rate 0.3845 vs 0.5490), so it
    # was deleted. Sacks still has to carry the column, as NaN, so every
    # category shares one schema.
    projections = edges.set_index("category")["projection"]
    assert projections.loc[["Sacks"]].isna().all()
    assert projections.loc["Passing Yards"].notna().all()
    assert (projections.loc["Passing Yards"] > 0).all()


def test_build_prop_edges_favorable_matchup_produces_a_ratio_above_one():
    # SEA is a real, comparatively weak WR pass defense (allows more real
    # receiving yards to opposing WRs than NE) - a WR on the team facing
    # SEA should get ratio > 1. A defense's own real "allowed" rate comes
    # from what the OFFENSES FACING IT produced (nfl_teams.compute_team_week_allowed
    # groups by `opponent_team`) - so it's SF's own WR receiving_yards
    # (team=SF, opponent_team=SEA) that sets SEA's real allowed rate, NOT
    # a row on SEA's own team. Real, necessary fix (2026-09-16): this
    # opponent-allowed rate is now POSITION-specific (see
    # compute_position_defense_rolling_rates's own docstring), so the
    # real differentiating signal must be each WR's own real receiving
    # yards - passing_yards (a team-wide, position-blind stat) no longer
    # feeds the Receiving Yards category at all.
    # Both subjects are deliberately LEAGUE-AVERAGE per target (6.5), so
    # nothing about the players themselves separates them and the ratio
    # can only come from the defense. Their team-mates set each defense's
    # allowed rate: SEA gives up 9.0 yards per target, NE 4.0, against a
    # 6.5 league rate.
    rows = []
    rows += _multi_week_skill_rows("wr_vs_weak", "WR Vs Weak", "SF", "SEA", "WR", range(1, 6), targets=10, receiving_yards=65, receptions=5)
    rows += _multi_week_skill_rows("sf_filler", "SF Filler", "SF", "SEA", "WR", range(1, 6), targets=10, receiving_yards=115, receptions=5)
    rows += _multi_week_skill_rows("wr_vs_strong", "WR Vs Strong", "KC", "NE", "WR", range(1, 6), targets=10, receiving_yards=65, receptions=5)
    rows += _multi_week_skill_rows("kc_filler", "KC Filler", "KC", "NE", "WR", range(1, 6), targets=10, receiving_yards=15, receptions=5)
    weekly_df = pd.DataFrame(rows)

    current_week_schedule = pd.DataFrame([{"home_team": "SF", "away_team": "SEA"}, {"home_team": "KC", "away_team": "NE"}])

    edges = nfl_player_props.build_prop_edges(weekly_df, current_week_schedule)
    rec_yards = edges[edges["category"] == "Receiving Yards"].set_index("player_id")

    assert rec_yards.loc["wr_vs_weak", "ratio"] > 1.0
    assert rec_yards.loc["wr_vs_strong", "ratio"] < 1.0
    # The rate on the board is now PER TARGET, not per game - the whole
    # point of the projection model (see nfl_prop_projections' module
    # docstring). 9.0 and 4.0 shrunk toward the 6.5 league rate.
    assert rec_yards.loc["wr_vs_weak", "rate_basis"] == "per target"
    assert rec_yards.loc["wr_vs_weak", "opponent_allowed_rate"] == pytest.approx(8.0)
    assert rec_yards.loc["wr_vs_strong", "opponent_allowed_rate"] == pytest.approx(5.0)


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
    # Real, necessary fix (2026-09-16): opponent-allowed rate is now
    # POSITION-specific (see compute_position_defense_rolling_rates's
    # own docstring), so the differentiating signal must be each WR's
    # own real receiving yards, not a team-wide QB passing_yards stat.
    # Same fixture shape as the ratio test above: both subjects sit at
    # exactly the 6.5 league yards per target, so shrinkage is a no-op on
    # them and the direction is decided purely by the defense. Each
    # projects 10 targets (a 0.5 share of his team's 20), so the
    # projections come out at exactly 10 * 6.5 * 1.2308 = 80.0 against a
    # 65.0 baseline (Over) and 10 * 6.5 * 0.7692 = 50.0 against the same
    # 65.0 baseline (Under).
    rows = []
    rows += _multi_week_skill_rows("wr_vs_weak", "WR Vs Weak", "SF", "SEA", "WR", range(1, 6), targets=10, receiving_yards=65, receptions=5)
    rows += _multi_week_skill_rows("sf_filler", "SF Filler", "SF", "SEA", "WR", range(1, 6), targets=10, receiving_yards=115, receptions=5)
    rows += _multi_week_skill_rows("wr_vs_strong", "WR Vs Strong", "KC", "NE", "WR", range(1, 6), targets=10, receiving_yards=65, receptions=5)
    rows += _multi_week_skill_rows("kc_filler", "KC Filler", "KC", "NE", "WR", range(1, 6), targets=10, receiving_yards=15, receptions=5)
    weekly_df = pd.DataFrame(rows)
    current_week_schedule = pd.DataFrame([{"home_team": "SF", "away_team": "SEA"}, {"home_team": "KC", "away_team": "NE"}])

    edges = nfl_player_props.build_prop_edges(weekly_df, current_week_schedule)
    top = nfl_player_props.top_prop_bets(edges, n=20).set_index("player_id")

    rec_yards_rows = top[top["category"] == "Receiving Yards"]
    assert rec_yards_rows.loc["wr_vs_weak", "projection"] == pytest.approx(80.0)
    assert rec_yards_rows.loc["wr_vs_strong", "projection"] == pytest.approx(50.0)
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
    # Real, necessary fix (2026-09-16): opponent-allowed rate is now
    # POSITION-specific (see compute_position_defense_rolling_rates's
    # own docstring), so each WR's own real receiving yards is what must
    # differentiate SEA/NE's allowed rate now, not a team-wide QB
    # passing_yards stat. Each QB row ALSO carries a real, neutral-ish
    # sacks_suffered value so these 4 teams don't distort the Sacks
    # league average below (each real category's league rate is computed
    # over the WHOLE real weekly_df, so a team with no real value for
    # the other category's stat would otherwise drag that other
    # category's real league average toward zero, an unwanted
    # cross-category artifact of this fixture, not a real property of
    # either category).
    # Per target these come out at 7.8 and 4.2 against a 6.0 league rate,
    # which after shrinkage is a matchup swing of +/-0.13 - comfortably
    # smaller than the clipped +/-0.20 Sacks reaches below, which is the
    # gap this test's own precondition asserts before trusting the
    # ranking.
    rows += _multi_week_skill_rows("wr_fav", "WR Favorable", "SF", "SEA", "WR", range(1, 6), targets=10, receiving_yards=78, receptions=5)
    rows += _multi_week_skill_rows("wr_unfav", "WR Unfavorable", "KC", "NE", "WR", range(1, 6), targets=10, receiving_yards=42, receptions=5)
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

    # With the diversity cap lifted, the full ordering is visible and is
    # by usage, highest first.
    top = nfl_player_props.top_prop_bets(edges, n=10, max_per_team_category=99)
    ordered_ids = list(top[top["category"] == "Receptions"]["player_id"])
    assert ordered_ids == ["wr_high", "wr_mid", "wr_low"]

    # And under the shipped cap of one row per (team, category), the one
    # that survives is the highest-usage receiver - not `wr_low`, which is
    # first in the fixture's own row order. That is the same property the
    # full ordering above asserts, and it is the one that actually reaches
    # the board.
    capped = nfl_player_props.top_prop_bets(edges, n=10)
    assert list(capped[capped["category"] == "Receptions"]["player_id"]) == ["wr_high"]


def test_opponent_allowed_rate_is_per_play_not_per_game():
    """Regression test for the 2026-09-17 user report: a board showing
    `opponent_allowed_rate` 202.4 next to a 31.7 yards-per-game receiver.

    That number was the opponent's total receiving yards allowed to ALL
    opposing WRs per GAME, which is not a quantity any single receiver
    could ever reach - but the real defect was that it was BIASED, not
    just mis-scaled, because yards-allowed-per-game rises with how many
    targets a defense faces even when it defends each one exactly as
    well as everyone else.

    Both defenses here allow an identical 7.0 yards per target. SEA
    simply FACES twice the volume (20 targets a game to NE's 10), so
    per game it concedes 140 yards to NE's 70 - a 2x "matchup" that is
    pure pace and nothing to do with coverage. The per-play rate the
    board now reports must see through that and call both defenses
    average.
    """
    rows = []
    rows += _multi_week_skill_rows("sf_wr1", "SF WR One", "SF", "SEA", "WR", range(1, 6), targets=10, receiving_yards=70, receptions=5)
    rows += _multi_week_skill_rows("sf_wr2", "SF WR Two", "SF", "SEA", "WR", range(1, 6), targets=10, receiving_yards=70, receptions=5)
    rows += _multi_week_skill_rows("kc_wr1", "KC WR One", "KC", "NE", "WR", range(1, 6), targets=10, receiving_yards=70, receptions=5)
    weekly_df = pd.DataFrame(rows)
    current_week_schedule = pd.DataFrame([{"home_team": "SF", "away_team": "SEA"}, {"home_team": "KC", "away_team": "NE"}])

    edges = nfl_player_props.build_prop_edges(weekly_df, current_week_schedule)
    rec_yards = edges[edges["category"] == "Receiving Yards"].set_index("player_id")

    # The per-GAME view that used to drive this board would have made SEA
    # look twice as generous as NE. Confirm the fixture really does carry
    # that 2x volume gap, so the assertion below is meaningful.
    per_game = nfl_player_props.compute_position_defense_rolling_rates(weekly_df, "receiving_yards")
    per_game = per_game.set_index(["team", "position"])["allowed_per_game"]
    assert per_game.loc[("SEA", "WR")] == pytest.approx(2 * per_game.loc[("NE", "WR")])

    # Per play, both defenses are identical and both are exactly league
    # average, so neither player gets a matchup edge in either direction.
    assert rec_yards.loc["sf_wr1", "opponent_allowed_rate"] == pytest.approx(7.0)
    assert rec_yards.loc["kc_wr1", "opponent_allowed_rate"] == pytest.approx(7.0)
    assert rec_yards.loc["sf_wr1", "ratio"] == pytest.approx(1.0)
    assert rec_yards.loc["kc_wr1", "ratio"] == pytest.approx(1.0)


def test_top_prop_bets_ranks_skill_categories_on_the_projection_gap():
    """The board ranks skill categories on how far the PROJECTION departs
    from the player's own per-game baseline, not on the opponent ratio.

    Both receivers below face the exact same defense, so the ratio is
    identical for the two of them and could not order them at all. What
    separates them is usage: `wr_rising` takes 80% of his team's targets
    while `wr_steady` takes 20% of his, so the projection moves
    `wr_rising` much further off his trailing average. The old
    ratio-based ranking was blind to this - it is exactly why two
    Minnesota receivers with target shares of 0.314 and 0.162 came back
    with an identical 1.2 ratio and an identical Over on the live board.
    """
    rows = []
    # SF faces SEA. wr_rising dominates his team's targets; his low-volume
    # team-mate barely features.
    rows += _multi_week_skill_rows("wr_rising", "WR Rising", "SF", "SEA", "WR", range(1, 6), targets=16, receiving_yards=112, receptions=8)
    rows += _multi_week_skill_rows("sf_minor", "SF Minor", "SF", "SEA", "WR", range(1, 6), targets=4, receiving_yards=28, receptions=2)
    # KC also faces SEA this week, so both subjects share one defense and
    # one identical matchup ratio.
    rows += _multi_week_skill_rows("wr_steady", "WR Steady", "KC", "SEA", "WR", range(1, 6), targets=4, receiving_yards=28, receptions=2)
    rows += _multi_week_skill_rows("kc_major", "KC Major", "KC", "SEA", "WR", range(1, 6), targets=16, receiving_yards=112, receptions=8)
    weekly_df = pd.DataFrame(rows)
    current_week_schedule = pd.DataFrame([{"home_team": "SF", "away_team": "SEA"}, {"home_team": "KC", "away_team": "SEA"}])

    edges = nfl_player_props.build_prop_edges(weekly_df, current_week_schedule)
    rec_yards = edges[edges["category"] == "Receiving Yards"].set_index("player_id")

    # Confirm the ratio genuinely cannot separate them before asserting
    # that something else did.
    assert rec_yards.loc["wr_rising", "ratio"] == pytest.approx(rec_yards.loc["wr_steady", "ratio"])
    # But their projections differ, because their target shares do.
    assert rec_yards.loc["wr_rising", "projected_targets"] > rec_yards.loc["wr_steady", "projected_targets"]
    assert rec_yards.loc["wr_rising", "projection"] > rec_yards.loc["wr_steady", "projection"]


def _snap_row(pfr_id, team, week, offense=0.0, defense=0.0, season=2026, game_type="REG"):
    return {
        "season": season, "game_type": game_type, "week": week, "team": team,
        "game_id": f"{season}_{week:02d}_{team}", "pfr_player_id": pfr_id,
        "offense_snaps": offense, "defense_snaps": defense,
    }


def _roster(gsis_id, pfr_id, season=2026):
    return {"season": season, "gsis_id": gsis_id, "pfr_id": pfr_id}


def test_compute_current_season_snap_share_measures_share_of_team_season_snaps():
    # Team plays 2 real games at 60 snaps each = 120 real team snaps.
    # The starter takes 60+60; the backup takes 12 in one game only.
    snaps = pd.DataFrame([
        _snap_row("pfr_start", "ARI", 1, offense=60.0), _snap_row("pfr_start", "ARI", 2, offense=60.0),
        _snap_row("pfr_back", "ARI", 1, offense=12.0),
    ])
    rosters = pd.DataFrame([_roster("starter", "pfr_start"), _roster("backup", "pfr_back")])

    result = nfl_player_props.compute_current_season_snap_share(snaps, rosters, 2026).set_index("player_id")

    assert result.loc["starter", "snap_share"] == pytest.approx(1.0)
    # 12/120 - a real one-game cameo must read LOW against the team's own
    # full season, not high against only the game the player appeared in.
    assert result.loc["backup", "snap_share"] == pytest.approx(0.1)


def test_compute_current_season_snap_share_covers_defenders_too():
    # The Sacks category is about real pass RUSHERS, who take no real
    # offensive snaps - an offense-only share would filter every real
    # defender off the board entirely.
    snaps = pd.DataFrame([
        _snap_row("pfr_edge", "GB", 1, defense=55.0),
        _snap_row("pfr_qb", "GB", 1, offense=60.0),
    ])
    rosters = pd.DataFrame([_roster("edge1", "pfr_edge"), _roster("qb1", "pfr_qb")])

    result = nfl_player_props.compute_current_season_snap_share(snaps, rosters, 2026).set_index("player_id")

    assert result.loc["edge1", "snap_share"] == pytest.approx(1.0)
    assert result.loc["qb1", "snap_share"] == pytest.approx(1.0)


def test_compute_current_season_snap_share_ignores_other_seasons():
    # The whole point is CURRENT-season participation - last season's
    # snaps are exactly what must not count.
    snaps = pd.DataFrame([
        _snap_row("pfr_gone", "ARI", 1, offense=60.0, season=2025),
        _snap_row("pfr_here", "ARI", 1, offense=60.0, season=2026),
    ])
    rosters = pd.DataFrame([
        _roster("departed", "pfr_gone", season=2025), _roster("departed", "pfr_gone", season=2026),
        _roster("current", "pfr_here"),
    ])

    result = nfl_player_props.compute_current_season_snap_share(snaps, rosters, 2026)

    assert list(result["player_id"]) == ["current"]


def test_compute_current_season_snap_share_is_empty_with_no_current_season_data():
    snaps = pd.DataFrame([_snap_row("pfr_a", "ARI", 1, offense=60.0, season=2025)])
    rosters = pd.DataFrame([_roster("a", "pfr_a")])

    assert nfl_player_props.compute_current_season_snap_share(snaps, rosters, 2026).empty


def _two_receiver_edges():
    rows = []
    rows += _multi_week_skill_rows("plays_now", "Plays Now", "SF", "SEA", "WR", range(1, 6), receptions=5)
    rows += _multi_week_skill_rows("departed", "Departed Last Year", "SF", "SEA", "WR", range(1, 6), receptions=5)
    return pd.DataFrame(rows), pd.DataFrame([{"home_team": "SF", "away_team": "SEA"}])


def test_build_prop_edges_drops_players_with_no_current_season_snaps():
    # The real reported bug (2026-09-17): all four Arizona backs on the
    # real week-2 board had ZERO real current-season appearances. A player
    # absent from the snap-share frame entirely is not playing.
    weekly_df, schedule = _two_receiver_edges()
    snap_share = pd.DataFrame([{"player_id": "plays_now", "snap_share": 0.80}])

    edges = nfl_player_props.build_prop_edges(weekly_df, schedule, snap_share=snap_share)

    assert set(edges["player_id"]) == {"plays_now"}


def test_build_prop_edges_drops_players_below_the_snap_share_floor():
    weekly_df, schedule = _two_receiver_edges()
    snap_share = pd.DataFrame([
        {"player_id": "plays_now", "snap_share": 0.80},
        {"player_id": "departed", "snap_share": 0.05},
    ])

    edges = nfl_player_props.build_prop_edges(weekly_df, schedule, snap_share=snap_share, min_snap_share=0.25)

    assert set(edges["player_id"]) == {"plays_now"}


def test_build_prop_edges_skips_the_filter_entirely_at_week_one():
    # At real week 1 nobody has a current-season snap yet - filtering on
    # them would empty the whole board rather than narrow it.
    weekly_df, schedule = _two_receiver_edges()

    empty = nfl_player_props.build_prop_edges(
        weekly_df, schedule, snap_share=pd.DataFrame(columns=["player_id", "snap_share"])
    )
    unfiltered = nfl_player_props.build_prop_edges(weekly_df, schedule)

    assert set(empty["player_id"]) == set(unfiltered["player_id"]) == {"plays_now", "departed"}


def test_top_prop_bets_ranks_halving_and_doubling_as_equal_departures():
    """The confidence gap is symmetric in log space, so a projection at
    half the baseline is ranked exactly as confidently as one at double.

    A plain percentage gap is not symmetric - doubling scores 1.0 while
    halving scores only 0.5 - and because a projection is a product of
    three ratios its distribution is right-skewed, so that asymmetry fed
    the board an almost entirely "Over" top 10 (see top_prop_bets' own
    comment for the measured figures).
    """
    edges = pd.DataFrame([
        # Same category, same baseline. One projects at 2x, one at 0.5x.
        {"player_id": "doubler", "player_name": "Doubler", "team": "SF", "position": "WR",
         "opponent": "SEA", "games": 5, "category": "Receiving Yards", "player_rate": 50.0,
         "projection": 100.0, "projected_targets": 8.0, "projected_carries": 0.0,
         "opponent_allowed_rate": 8.0, "league_rate": 8.0, "rate_basis": "per target", "ratio": 1.0},
        {"player_id": "halver", "player_name": "Halver", "team": "KC", "position": "WR",
         "opponent": "NE", "games": 5, "category": "Receiving Yards", "player_rate": 50.0,
         "projection": 25.0, "projected_targets": 4.0, "projected_carries": 0.0,
         "opponent_allowed_rate": 8.0, "league_rate": 8.0, "rate_basis": "per target", "ratio": 1.0},
    ])

    top = nfl_player_props.top_prop_bets(edges, n=2).set_index("player_id")

    assert top.loc["doubler", "direction"] == "Over"
    assert top.loc["halver", "direction"] == "Under"
    # Equal and opposite in log space, so neither outranks the other.
    assert top.loc["doubler", "confidence_signal"] == pytest.approx(np.log(2.0))
    assert top.loc["halver", "confidence_signal"] == pytest.approx(np.log(2.0))


def test_top_prop_bets_caps_correlated_rows_from_one_team_and_category():
    """Every player in one (team, category) bucket shares an opponent
    rate, so where no projection separates them - Sacks - they receive an
    identical ranking signal. Observed live on the 2026 week-2 slate:
    three Chicago rushers facing Minnesota, all at +32%, took three of
    ten slots. That is one opinion about one offensive line sold as three
    bets, the same defect a user reported when four Arizona running backs
    filled the board.
    """
    rows = []
    for name, sacks in (("edge_a", 1.2), ("edge_b", 0.9), ("edge_c", 0.7)):
        rows += _multi_week_skill_rows(name, name, "CHI", "MIN", "DE", range(1, 6), def_sacks=sacks)
    rows += _multi_week_skill_rows(
        "min_qb", "MIN QB", "MIN", "CHI", "QB", range(1, 6),
        sacks_suffered=6.0, passing_yards=200, attempts=30,
    )
    edges = nfl_player_props.build_prop_edges(
        pd.DataFrame(rows), pd.DataFrame([{"home_team": "CHI", "away_team": "MIN"}])
    )
    sacks_rows = edges[edges["category"] == "Sacks"]
    # Confirm the fixture really does hand all three the same signal.
    assert sacks_rows["ratio"].nunique() == 1

    top = nfl_player_props.top_prop_bets(edges, n=10)
    chicago_sacks = top[(top["team"] == "CHI") & (top["category"] == "Sacks")]
    assert len(chicago_sacks) == config.NFL_PROP_MAX_PER_TEAM_CATEGORY
    # The survivor is the best of the tied group, not an arbitrary one.
    assert chicago_sacks.iloc[0]["player_id"] == "edge_a"


def test_top_prop_bets_cap_removes_rather_than_reorders():
    """The cap is applied AFTER ranking, so it can only ever drop the
    weaker member of a correlated pair - it must never promote a row
    above one that outranked it."""
    rows = []
    rows += _multi_week_skill_rows("wr_a", "WR A", "SF", "SEA", "WR", range(1, 6), targets=12, receiving_yards=120, receptions=8)
    rows += _multi_week_skill_rows("wr_b", "WR B", "SF", "SEA", "WR", range(1, 6), targets=8, receiving_yards=20, receptions=2)
    rows += _multi_week_skill_rows("wr_c", "KC WR", "KC", "NE", "WR", range(1, 6), targets=10, receiving_yards=70, receptions=5)
    rows += _multi_week_skill_rows("wr_d", "NE WR", "NE", "KC", "WR", range(1, 6), targets=10, receiving_yards=70, receptions=5)
    edges = nfl_player_props.build_prop_edges(
        pd.DataFrame(rows),
        pd.DataFrame([{"home_team": "SF", "away_team": "SEA"}, {"home_team": "KC", "away_team": "NE"}]),
    )

    uncapped = nfl_player_props.top_prop_bets(edges, n=20, max_per_team_category=99)
    capped = nfl_player_props.top_prop_bets(edges, n=20)

    # Capped output is a subsequence of the uncapped ranking: same
    # relative order throughout, only removals.
    uncapped_keys = list(zip(uncapped["player_id"], uncapped["category"]))
    capped_keys = list(zip(capped["player_id"], capped["category"]))
    iterator = iter(uncapped_keys)
    assert all(key in iterator for key in capped_keys)

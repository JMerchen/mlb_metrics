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
    rows = []
    rows += _multi_week_skill_rows("wr_vs_weak", "WR Vs Weak", "SF", "SEA", "WR", range(1, 6), receiving_yards=100, receptions=5)
    rows += _multi_week_skill_rows("wr_vs_strong", "WR Vs Strong", "KC", "NE", "WR", range(1, 6), receiving_yards=20, receptions=5)
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
    # Real, necessary fix (2026-09-16): opponent-allowed rate is now
    # POSITION-specific (see compute_position_defense_rolling_rates's
    # own docstring), so the differentiating signal must be each WR's
    # own real receiving yards, not a team-wide QB passing_yards stat.
    rows = []
    rows += _multi_week_skill_rows("wr_vs_weak", "WR Vs Weak", "SF", "SEA", "WR", range(1, 6), receiving_yards=100, receptions=5)
    rows += _multi_week_skill_rows("wr_vs_strong", "WR Vs Strong", "KC", "NE", "WR", range(1, 6), receiving_yards=30, receptions=5)
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
    rows += _multi_week_skill_rows("wr_fav", "WR Favorable", "SF", "SEA", "WR", range(1, 6), receiving_yards=66, receptions=5)
    rows += _multi_week_skill_rows("wr_unfav", "WR Unfavorable", "KC", "NE", "WR", range(1, 6), receiving_yards=54, receptions=5)
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


def test_top_prop_bets_breaks_a_cross_category_tie_by_unclipped_raw_magnitude():
    # Real, confirmed live finding (2026-09-16, against this project's
    # own cached real NFL data): on a real slate where several teams'
    # ratios all saturate the same clip boundary, BOTH edge_percentile
    # (each category's own top row reaches the exact same 0.5) AND
    # usage_percentile (each row here is also the higher-usage one in
    # its own 2-row category, so it also reaches the exact same 1.0) can
    # legitimately tie across categories - the tertiary raw_edge_magnitude
    # key exists specifically for this residual case.
    #
    # Receptions: a real, modest raw ratio swing (1.3x) that still clips
    # to the same 1.2 as Rushing Yards below.
    rows = []
    rows += _multi_week_skill_rows("wr_fav", "WR Favorable", "SF", "SEA", "WR", range(1, 6), receptions=6.5)
    rows += _multi_week_skill_rows("wr_unfav", "WR Unfavorable", "KC", "NE", "WR", range(1, 6), receptions=3.5)
    # Rushing Yards: a real, much LARGER raw ratio swing (1.54x) - same
    # clipped ratio (1.2) as Receptions above, but a real, bigger true
    # edge underneath the clip.
    rows += _multi_week_skill_rows("rb_fav", "RB Favorable", "GB", "CHI", "RB", range(1, 6), rushing_yards=100)
    rows += _multi_week_skill_rows("rb_unfav", "RB Unfavorable", "LAR", "ARI", "RB", range(1, 6), rushing_yards=30)
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
    rec = edges[(edges["category"] == "Receptions") & (edges["player_id"] == "wr_fav")].iloc[0]
    rush = edges[(edges["category"] == "Rushing Yards") & (edges["player_id"] == "rb_fav")].iloc[0]
    # Confirm the fixture actually produces the intended real clipped tie
    # before trusting the ordering assertion below.
    assert rec["ratio"] == pytest.approx(rush["ratio"]) == pytest.approx(1.2)

    top = nfl_player_props.top_prop_bets(edges, n=2)

    assert list(top["player_id"]) == ["rb_fav", "wr_fav"]


def test_top_prop_bets_ranks_on_the_unclipped_ratio_not_the_clipped_one():
    # Real, measured bug (2026-09-16): ranking on the CLIPPED ratio
    # collapsed every matchup past the clip boundary into one identical
    # value - 35.6% of real qualified rows on the real shipped week-2
    # board, a single 49-row tie at the top, which then fell through to
    # the usage tiebreak and ordered the board by volume instead of by
    # matchup quality. scripts/backtest_nfl_player_props.py proves a clip
    # cannot help a rank anyway (its rank-based tercile spread is
    # IDENTICAL for every clip candidate tested) - so ranking uses the
    # unclipped ratio while `ratio` still reports the clipped signal.
    #
    # Both WRs below face defenses well past the clip ceiling, so both
    # report the SAME clipped ratio - only the unclipped matchup tells
    # them apart, and the genuinely more extreme one must rank first
    # even though it is the LOWER-usage player (so usage can't be what
    # produced the ordering).
    rows = []
    rows += _multi_week_skill_rows("wr_extreme", "WR Extreme", "SF", "SEA", "WR", range(1, 6), receiving_yards=40)
    rows += _multi_week_skill_rows("wr_mild", "WR Mild", "KC", "NE", "WR", range(1, 6), receiving_yards=45)
    # SEA allows a huge amount to WRs (240/wk); NE allows less but is
    # still above average (150/wk) - BOTH clip to the same ceiling.
    rows += _multi_week_skill_rows("sea_feeder", "SEA Feeder", "ARI", "SEA", "WR", range(1, 6), receiving_yards=200)
    rows += _multi_week_skill_rows("ne_feeder", "NE Feeder", "BUF", "NE", "WR", range(1, 6), receiving_yards=105)
    # Two real stingy defenses, purely to pull the real league average
    # down far enough that both subject matchups land ABOVE it (rather
    # than straddling it and clipping in opposite directions).
    rows += _multi_week_skill_rows("low_one", "Low One", "DAL", "DEN", "WR", range(1, 6), receiving_yards=30)
    rows += _multi_week_skill_rows("low_two", "Low Two", "NYG", "MIA", "WR", range(1, 6), receiving_yards=30)
    weekly_df = pd.DataFrame(rows)
    current_week_schedule = pd.DataFrame([
        {"home_team": "SF", "away_team": "SEA"},
        {"home_team": "KC", "away_team": "NE"},
    ])

    edges = nfl_player_props.build_prop_edges(weekly_df, current_week_schedule)
    subject = edges[edges["player_id"].isin(["wr_extreme", "wr_mild"])]
    rec_yards = subject[subject["category"] == "Receiving Yards"].set_index("player_id")
    # Confirm the fixture really does saturate both to the same clipped
    # ratio before trusting the ordering assertion below.
    assert rec_yards.loc["wr_extreme", "ratio"] == pytest.approx(rec_yards.loc["wr_mild", "ratio"])
    unclipped = rec_yards["opponent_allowed_rate"] / rec_yards["league_rate"]
    assert unclipped.loc["wr_extreme"] > unclipped.loc["wr_mild"]

    top = nfl_player_props.top_prop_bets(edges, n=10)
    ordered = [p for p in top["player_id"] if p in ("wr_extreme", "wr_mild")]

    assert ordered[0] == "wr_extreme"


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

import numpy as np
import pandas as pd
import pytest

from mlb_metrics import config, prop_market


def _quote(book, line, over_price=-110, under_price=-110, player_id="wr1",
           category="Receptions", fetched_at="2026-09-17T12:00:00Z", player_name="WR One"):
    return {
        "fetched_at": fetched_at, "book": book, "player_id": player_id,
        "player_name": player_name, "category": category, "line": line,
        "over_price": over_price, "under_price": under_price,
    }


def test_american_to_decimal_both_signs():
    assert prop_market.american_to_decimal(100) == pytest.approx(2.0)
    assert prop_market.american_to_decimal(-110) == pytest.approx(1.9090909, abs=1e-6)
    assert prop_market.american_to_decimal(150) == pytest.approx(2.5)
    assert prop_market.american_to_decimal(-200) == pytest.approx(1.5)


def test_devig_two_way_sums_to_one_and_removes_the_overround():
    # -110 / -110 is 52.38% each side, an overround of 4.76%.
    p_over, p_under = prop_market.devig_two_way(-110, -110)
    assert float(p_over) == pytest.approx(0.5)
    assert float(p_over) + float(p_under) == pytest.approx(1.0)

    # An asymmetric quote keeps its lean after de-vigging.
    p_over, _ = prop_market.devig_two_way(-200, +160)
    assert 0.6 < float(p_over) < 0.7


def test_outcome_sigma_grows_with_the_square_root_of_the_level():
    """The measured form: quadrupling the level doubles the spread. A
    constant sigma would badly misprice both tails of the board."""
    small = float(prop_market.outcome_sigma("Receiving Yards", 25.0))
    large = float(prop_market.outcome_sigma("Receiving Yards", 100.0))
    assert large == pytest.approx(2 * small)
    assert small == pytest.approx(config.PROP_OUTCOME_SIGMA_K["Receiving Yards"] * 5.0)


def test_outcome_sigma_floors_the_level():
    """A 0.5-reception line must not imply a near-zero spread - that
    would make every price look like a certainty and manufacture huge
    fake edges at the bottom of the board."""
    floored = float(prop_market.outcome_sigma("Receptions", 0.01))
    assert floored == pytest.approx(
        config.PROP_OUTCOME_SIGMA_K["Receptions"] * np.sqrt(config.PROP_SIGMA_MIN_LEVEL)
    )


def test_outcome_sigma_is_nan_for_a_category_with_no_fitted_k():
    """Sacks has no projection to take residuals against, so no k was
    fitted. That gap stays visible rather than being filled with a
    guess."""
    assert np.isnan(float(prop_market.outcome_sigma("Sacks", 1.0)))


def test_implied_mean_of_a_fair_price_is_the_line_itself():
    """A book at its own line with a 50/50 de-vigged price is saying the
    line IS its central estimate."""
    mean = prop_market.implied_mean(4.5, 0.5, "Receptions")
    assert float(mean) == pytest.approx(4.5)


def test_implied_mean_moves_with_the_price_at_a_fixed_line():
    """PRICE disagreement: two books on the same line disagree, and the
    one pricing the over as a favourite implies a higher mean."""
    favoured = float(prop_market.implied_mean(4.5, 0.60, "Receptions"))
    against = float(prop_market.implied_mean(4.5, 0.40, "Receptions"))
    assert favoured > 4.5 > against
    # Symmetric about the line for symmetric probabilities.
    assert favoured - 4.5 == pytest.approx(4.5 - against)


def test_implied_mean_and_probability_over_are_inverses():
    mean = prop_market.implied_mean(70.0, 0.62, "Receiving Yards")
    back = prop_market.probability_over(mean, 70.0, "Receiving Yards")
    assert float(back) == pytest.approx(0.62, abs=1e-9)


def test_line_and_price_disagreement_land_on_one_scale():
    """The unification that makes cross-book comparison possible: a book
    that moved its LINE and a book that only moved its PRICE become
    comparable numbers."""
    moved_line = float(prop_market.implied_mean(5.5, 0.50, "Receptions"))
    moved_price = float(prop_market.implied_mean(4.5, 0.50, "Receptions"))
    assert moved_line > moved_price
    # And a big enough price move at the lower line can match the higher
    # line outright, which is exactly the equivalence being modelled.
    p_needed = float(prop_market.probability_over(moved_line, 4.5, "Receptions"))
    assert float(prop_market.implied_mean(4.5, p_needed, "Receptions")) == pytest.approx(moved_line)


def test_consensus_is_robust_to_one_broken_quote():
    """A weighted MEDIAN, not a mean - one fat-fingered or stale quote
    must not drag the consensus."""
    quotes = pd.DataFrame([
        _quote("A", 4.5), _quote("B", 4.5), _quote("C", 4.5),
        _quote("D", 12.5),  # nonsense
    ])
    consensus = prop_market.build_consensus(quotes)
    assert consensus.loc[0, "consensus_mean"] == pytest.approx(4.5)
    assert consensus.loc[0, "n_books"] == 4
    # The disagreement measure still SEES the broken quote, which is the
    # point - it just does not let it move the centre.
    assert consensus.loc[0, "book_disagreement"] > 3.0


def test_consensus_reports_disagreement_in_comparable_units():
    """`disagreement_sigmas` exists so categories can be compared: half a
    reception of disagreement is a lot, half a passing yard is nothing."""
    receptions = prop_market.build_consensus(pd.DataFrame([
        _quote("A", 4.0), _quote("B", 4.5), _quote("C", 5.0),
    ]))
    passing = prop_market.build_consensus(pd.DataFrame([
        _quote("A", 250.0, category="Passing Yards"),
        _quote("B", 250.5, category="Passing Yards"),
        _quote("C", 251.0, category="Passing Yards"),
    ]))
    # Similar raw spread in each, but wildly different significance.
    assert receptions.loc[0, "disagreement_sigmas"] > 10 * passing.loc[0, "disagreement_sigmas"]


def test_consensus_flags_too_few_books_rather_than_dropping_them():
    """Two books that disagree are two opinions, not a consensus and an
    outlier. The row stays visible and is marked."""
    quotes = pd.DataFrame([_quote("A", 4.5), _quote("B", 5.5)])
    consensus = prop_market.build_consensus(quotes)
    assert len(consensus) == 1
    assert consensus.loc[0, "has_consensus"] is np.False_ or consensus.loc[0, "has_consensus"] is False
    assert consensus.loc[0, "n_books"] < config.PROP_MIN_BOOKS_FOR_CONSENSUS


def test_sharpness_weighting_pulls_the_consensus_toward_the_sharp_book(monkeypatch):
    monkeypatch.setattr(config, "PROP_BOOK_SHARPNESS", {"SHARP": 100.0})
    quotes = pd.DataFrame([
        _quote("SHARP", 3.5), _quote("RETAIL_A", 5.5), _quote("RETAIL_B", 5.5),
    ])
    consensus = prop_market.build_consensus(quotes)
    # Unweighted, the median of (3.5, 5.5, 5.5) is 5.5; the sharp book's
    # weight has to drag it down.
    assert consensus.loc[0, "consensus_mean"] < 5.0


def test_outlier_scoring_finds_the_favourable_side_of_the_off_book():
    """The core use: three books agree at 4.5 and one hangs 6.5, so the
    consensus says take the UNDER at the outlier."""
    quotes = pd.DataFrame([
        _quote("A", 4.5), _quote("B", 4.5), _quote("C", 4.5), _quote("OFF", 6.5),
    ])
    scored = prop_market.score_book_disagreement(quotes)
    top = scored.iloc[0]
    assert top["book"] == "OFF"
    assert top["best_side"] == prop_market.UNDER
    assert top["expected_value"] > 0
    assert top["deviation"] > 0  # the outlier's implied mean sits above consensus
    # And the books that agree with the consensus offer no edge.
    agreeing = scored[scored["book"] != "OFF"]
    assert (agreeing["expected_value"] < 0).all()


def test_outlier_scoring_prices_consensus_at_the_outliers_own_line():
    """When books post different handicaps, the only apples-to-apples
    comparison is the consensus probability evaluated at the line being
    bet - not at the consensus line."""
    quotes = pd.DataFrame([
        _quote("A", 60.0, category="Receiving Yards"),
        _quote("B", 60.0, category="Receiving Yards"),
        _quote("C", 60.0, category="Receiving Yards"),
        _quote("OFF", 80.0, category="Receiving Yards"),
    ])
    scored = prop_market.score_book_disagreement(quotes).set_index("book")
    expected = float(prop_market.probability_over(60.0, 80.0, "Receiving Yards"))
    # Consensus mean is 60; at the outlier's 80 line the over is unlikely,
    # so the under is favoured at roughly 1 - expected.
    assert scored.loc["OFF", "best_side"] == prop_market.UNDER
    assert scored.loc["OFF", "consensus_probability"] == pytest.approx(1 - expected, abs=1e-9)


def test_outlier_scoring_marks_when_the_sharpest_book_is_the_one_out_of_line(monkeypatch):
    """If the sharpest book in the market is the outlier, that is
    information, not opportunity - it likely knows something."""
    monkeypatch.setattr(config, "PROP_BOOK_SHARPNESS", {"SHARP": 10.0})
    quotes = pd.DataFrame([
        _quote("SHARP", 6.5), _quote("A", 4.5), _quote("B", 4.5), _quote("C", 4.5),
    ])
    scored = prop_market.score_book_disagreement(quotes).set_index("book")
    assert bool(scored.loc["SHARP", "outlier_is_sharpest_book"]) is True
    assert bool(scored.loc["A", "outlier_is_sharpest_book"]) is False


def test_stale_quotes_are_flagged():
    quotes = pd.DataFrame([
        _quote("FRESH", 4.5, fetched_at="2026-09-17T12:00:00Z"),
        _quote("STALE", 4.5, fetched_at="2026-09-17T11:00:00Z"),
    ])
    flagged = prop_market.flag_stale_quotes(quotes, as_of="2026-09-17T12:01:00Z").set_index("book")
    assert bool(flagged.loc["STALE", "is_stale"]) is True
    assert bool(flagged.loc["FRESH", "is_stale"]) is False
    assert flagged.loc["STALE", "age_minutes"] == pytest.approx(61.0)


def test_middle_detection_finds_a_winnable_integer_gap():
    """Over 3.5 at one book and Under 4.5 at another both cash on exactly
    4 - genuinely available because receptions are low integers."""
    quotes = pd.DataFrame([
        _quote("A", 3.5), _quote("B", 4.5), _quote("C", 4.0),
    ])
    middles = prop_market.detect_middles(quotes)
    best = middles.iloc[0]
    assert best["over_line"] < best["under_line"]
    assert best["middle_width"] == pytest.approx(1.0)
    assert 0.0 < best["middle_probability"] < 1.0
    # Exactly one side always wins, so the pair can only lose the vig.
    assert best["expected_value"] > -1.0


def test_middle_detection_ignores_a_reversed_or_same_book_pair():
    quotes = pd.DataFrame([_quote("A", 4.5), _quote("A", 3.5)])
    assert prop_market.detect_middles(quotes).empty


def test_arbitrage_detection_requires_a_genuine_guaranteed_profit():
    # Two heavily favourable one-sided prices at the same line.
    quotes = pd.DataFrame([
        _quote("A", 4.5, over_price=+150, under_price=-400),
        _quote("B", 4.5, over_price=-400, under_price=+150),
    ])
    arbs = prop_market.detect_arbitrage(quotes)
    assert not arbs.empty
    assert (arbs["profit_fraction"] > 0).all()

    # A normal vigged market has none.
    fair = pd.DataFrame([_quote("A", 4.5), _quote("B", 4.5)])
    assert prop_market.detect_arbitrage(fair).empty


def test_closing_line_value_rewards_beating_the_close():
    """A bet on the over at 3.5 when the market closes at 4.5 got value:
    the market moved toward the bet."""
    quotes = pd.DataFrame([
        _quote("A", 3.5, fetched_at="2026-09-17T12:00:00Z"),
        _quote("A", 4.5, fetched_at="2026-09-17T16:00:00Z"),
    ])
    bets = pd.DataFrame([{
        "placed_at": "2026-09-17T12:00:00Z", "book": "A", "player_id": "wr1",
        "category": "Receptions", "line": 3.5, "side": prop_market.OVER, "price": -110,
    }])
    clv = prop_market.closing_line_value(bets, quotes)
    assert len(clv) == 1
    assert clv.loc[0, "clv"] > 0
    assert bool(clv.loc[0, "beat_close"]) is True


def test_closing_line_value_penalises_a_bet_the_market_moved_against():
    quotes = pd.DataFrame([
        _quote("A", 4.5, fetched_at="2026-09-17T12:00:00Z"),
        _quote("A", 3.5, fetched_at="2026-09-17T16:00:00Z"),
    ])
    bets = pd.DataFrame([{
        "placed_at": "2026-09-17T12:00:00Z", "book": "A", "player_id": "wr1",
        "category": "Receptions", "line": 4.5, "side": prop_market.OVER, "price": -110,
    }])
    clv = prop_market.closing_line_value(bets, quotes)
    assert clv.loc[0, "clv"] < 0
    assert bool(clv.loc[0, "beat_close"]) is False


def test_closing_line_value_sees_a_pure_price_move():
    """A line move is not the only way to get value - re-pricing the
    close at the BET's line is what makes a price-only move visible."""
    quotes = pd.DataFrame([
        _quote("A", 4.5, over_price=-110, fetched_at="2026-09-17T12:00:00Z"),
        _quote("A", 4.5, over_price=-160, under_price=+130, fetched_at="2026-09-17T16:00:00Z"),
    ])
    bets = pd.DataFrame([{
        "placed_at": "2026-09-17T12:00:00Z", "book": "A", "player_id": "wr1",
        "category": "Receptions", "line": 4.5, "side": prop_market.OVER, "price": -110,
    }])
    clv = prop_market.closing_line_value(bets, quotes)
    assert clv.loc[0, "clv"] > 0


def test_every_entry_point_survives_an_empty_frame():
    """The board has to build on a week where no feed responded."""
    empty = pd.DataFrame(columns=prop_market.QUOTE_COLUMNS)
    assert prop_market.build_consensus(empty).empty
    assert prop_market.score_book_disagreement(empty).empty
    assert prop_market.detect_middles(empty).empty
    assert prop_market.detect_arbitrage(empty).empty
    assert prop_market.closing_line_value(pd.DataFrame(columns=prop_market.BET_COLUMNS), empty).empty


def test_append_quote_snapshots_keeps_history_and_dedupes(tmp_path):
    """CLV cannot be reconstructed from a file that only holds the latest
    lines, so snapshots accumulate rather than overwrite."""
    path = str(tmp_path / "snapshots" / "quotes.csv")
    first = pd.DataFrame([_quote("A", 4.5, fetched_at="2026-09-17T12:00:00Z")])
    second = pd.DataFrame([_quote("A", 3.5, fetched_at="2026-09-17T16:00:00Z")])

    prop_market.append_quote_snapshots(first, path)
    combined = prop_market.append_quote_snapshots(second, path)
    assert len(combined) == 2

    # Re-appending the same observation does not duplicate it.
    again = prop_market.append_quote_snapshots(second, path)
    assert len(again) == 2


def test_outlier_scoring_uses_a_leave_one_out_consensus():
    """A book must be scored against the REST of the market, not against
    a consensus it helped set.

    At the three-book minimum this is not a refinement but a correctness
    requirement: the outlier can BE the median of the three, so an
    included-self consensus would score it as agreeing with what is
    mostly its own quote.
    """
    quotes = pd.DataFrame([_quote("A", 4.5), _quote("B", 4.5), _quote("OFF", 6.5)])
    scored = prop_market.score_book_disagreement(quotes).set_index("book")

    # The rest of the market (A and B) sits at 4.5, so that is what OFF
    # is measured against - NOT the 4.5-median-of-three that would also
    # include OFF itself.
    assert scored.loc["OFF", "consensus_mean"] == pytest.approx(4.5)
    assert scored.loc["OFF", "consensus_books"] == 2
    assert scored.loc["OFF", "deviation"] == pytest.approx(2.0)

    # And each agreeing book is scored against a consensus containing the
    # outlier, which correctly makes them look slightly off in the other
    # direction rather than perfectly neutral.
    assert scored.loc["A", "consensus_books"] == 2
    assert scored.loc["A", "consensus_mean"] > 4.5


def test_leave_one_out_changes_the_answer_when_it_matters():
    """With three books the included-self median is the middle quote; the
    leave-one-out median of the other two is materially different."""
    quotes = pd.DataFrame([_quote("LOW", 3.5), _quote("MID", 4.5), _quote("HIGH", 5.5)])
    pooled = prop_market.build_consensus(quotes).loc[0, "consensus_mean"]
    scored = prop_market.score_book_disagreement(quotes).set_index("book")

    assert pooled == pytest.approx(4.5)
    # MID is scored against LOW and HIGH, whose own median is also 4.5 -
    # so MID genuinely has no edge, and the deviation is zero.
    assert scored.loc["MID", "deviation"] == pytest.approx(0.0, abs=1e-9)
    # LOW is scored against MID and HIGH, a consensus ABOVE the pooled
    # one, so its deviation is larger than a pooled comparison would say.
    assert scored.loc["LOW", "consensus_mean"] > pooled
    assert scored.loc["LOW", "deviation"] < 0

"""Unit tests for kelly.py - hand-computed Kelly-criterion cases."""

import pytest

from mlb_metrics.kelly import kelly_fraction, moneyline_to_net_odds


def test_moneyline_to_net_odds_negative_favorite():
    # -150 -> risk $150 to win $100, net odds b = 100/150 = 2/3.
    assert moneyline_to_net_odds(-150) == pytest.approx(2 / 3)


def test_moneyline_to_net_odds_positive_underdog():
    # +150 -> risk $100 to win $150, net odds b = 1.5.
    assert moneyline_to_net_odds(150) == pytest.approx(1.5)


def test_moneyline_to_net_odds_rejects_zero():
    with pytest.raises(ValueError):
        moneyline_to_net_odds(0)


def test_kelly_fraction_textbook_even_money_case():
    # p=0.6 at even money (+100, b=1): f* = (0.6*1 - 0.4)/1 = 0.2 exactly -
    # the classic textbook full-Kelly example.
    assert kelly_fraction(0.6, 100) == pytest.approx(0.2)


def test_kelly_fraction_no_real_edge_at_a_vigged_line_clips_to_zero():
    # A true coin flip (p=0.5) against a real vigged line (-110, b=100/110)
    # has a NEGATIVE raw edge - clipped to 0, not a negative stake. This is
    # exactly why "just bet 50/50 games" is never +EV against a real price.
    b = 100 / 110
    raw_edge = (0.5 * b - 0.5) / b
    assert raw_edge < 0
    assert kelly_fraction(0.5, -110) == 0.0


def test_kelly_fraction_realistic_positive_edge_case():
    # p=0.55 at -110 (b=100/110): f* = (0.55*b - 0.45)/b.
    b = 100 / 110
    expected = (0.55 * b - 0.45) / b
    assert expected > 0
    assert kelly_fraction(0.55, -110) == pytest.approx(expected)


def test_kelly_fraction_multiplier_scales_the_stake():
    full = kelly_fraction(0.6, 100, fraction=1.0)
    half = kelly_fraction(0.6, 100, fraction=0.5)
    assert half == pytest.approx(full / 2)
    assert half == pytest.approx(0.1)

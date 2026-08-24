"""Kelly-criterion bet sizing - a follow-up to the market-benchmark work
(quant-analytics item #6): turns "the model's probability disagrees with
the market's real price" into an actual recommended stake, once
`game_predictions.advise_bets` (used by both `pipeline.run()` and
`scripts/recommend_bets.py`) has computed that edge for real.

Deliberately pure, no `mlb_metrics.config` import - config wiring (which
Kelly fraction multiplier, which minimum edge) happens in the CALLER, not
here: either a script's CLI args, or a library function's own
config-defaulted parameters (`game_predictions.select_game_picks`'s
`kelly_fraction_multiplier`/`min_edge` params) - never hardcoded inside
this module, same separation `decision_theory.py` already establishes for
this project's other real decision-theory module. Needs nothing beyond
stdlib math - no `pandas` dependency either, since both functions here
operate on plain scalars.
"""


def moneyline_to_net_odds(moneyline: float) -> float:
    """American moneyline -> net odds b (profit per $1 staked, NOT the
    total payout multiple - a $100 stake at +150 nets $150 profit, so
    b=1.5, not 2.5). 0 is not a real American-odds value - raises rather
    than silently producing garbage, same "raise on nonsensical input"
    convention decision_theory.py already uses (e.g. gain <= 0)."""
    if moneyline == 0:
        raise ValueError("moneyline cannot be 0")
    if moneyline < 0:
        return 100 / (-moneyline)
    return moneyline / 100


def kelly_fraction(probability: float, moneyline: float, fraction: float = 1.0) -> float:
    """The Kelly-optimal fraction of bankroll to stake on a bet with a
    real `moneyline` price, given `probability` (this project's own,
    independently-estimated true probability of that side winning - NOT
    the market's own implied probability, which would always give exactly
    0 by construction).

    f* = (p*b - (1-p)) / b, where b is the net odds from
    moneyline_to_net_odds. Clipped at 0 - a negative result means the
    real price doesn't offer a real edge at this probability estimate,
    not "stake a negative amount." `fraction` (default 1.0 = full Kelly)
    scales the result down - see config.KELLY_FRACTION_MULTIPLIER's own
    docstring for why this project always calls this at less than 1.0 in
    practice (full Kelly assumes the probability estimate is exactly
    right, which it never is)."""
    b = moneyline_to_net_odds(moneyline)
    f_star = (probability * b - (1 - probability)) / b
    return max(f_star, 0.0) * fraction

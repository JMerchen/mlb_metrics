"""Evidence that a bet wins, measured so it can be checked every week.

Why (2026-11-20): every bet claim on the site rests on a win rate over a
few dozen to a few hundred bets, and we look at each one again every week.
An ordinary significance test is only honest if you look once, at a sample
size fixed in advance: look every week and stop when it looks good and a
coin flip passes sooner or later. The methods here ("anytime-valid"
inference: Shafer 2021, "Testing by betting"; Waudby-Smith and Ramdas 2024,
"Estimating means of bounded random variables by betting"; Shin, Ramdas and
Rinaldo 2024, "E-detectors") keep their guarantee however often and
whenever we look.

The idea is a bet against the claim "this has no edge". Start with 1 unit
and stake a fixed fraction of the bankroll on every bet in the series. If
the bets truly have no edge (average return zero or worse), the bankroll
is expected to shrink, and the chance it EVER reaches 20 is at most 1 in
20 (Ville's inequality) - at any moment, with no fixed sample size. So the
bankroll itself is the evidence: 20 is the usual bar (the equal of
p < 0.05), 100 is strong, and anything near 1 or below is none. Not
knowing the right fraction in advance, the bankroll here is split evenly
over a range of fractions (FRACTIONS); an even split of fair bets is still
a fair bet.

Three tools, all on a series of bets in the order they were made:

  wealth_path          the bankroll after each bet (evidence of an edge).
  confidence_sequence  the win rates not yet ruled out, a range that
                       holds at every look at once (for bets at one price).
  fade_detector        an alarm for "the win rate has dropped below a
                       stated level at some point" - started fresh at
                       every bet, so it does not matter when the drop
                       began or that we went looking after a bad run.

What these cannot do: a rule CHOSEN by looking at a set of results is
flattered by those results however they are tested. For a replay the
bankroll is an upper bound on the evidence; only bets made after the rule
was fixed are a clean test.
"""

import numpy as np

# Fractions of the bankroll the evidence bet is split over. A bet that wins
# 55-58% at -110 is best staked at about 6-12%; the range covers smaller
# and larger edges.
FRACTIONS = np.linspace(0.01, 0.30, 30)
# The usual bar for the bankroll: reaching it has a chance of at most 1 in
# 20 if there is no edge.
EVIDENCE_BAR = 20.0
# How sure the confidence sequence is: every range it ever reports holds
# the true rate with this probability.
CONFIDENCE = 0.95
# The fade alarm goes off at this level, which a series whose win rate has
# NOT dropped takes this many bets or more to reach on average.
ALARM_LEVEL = 1000.0
WIN_AT_110 = 100.0 / 110.0


def _log_mix(log_paths: np.ndarray) -> np.ndarray:
    """Log of the even split's value from each part's log value (rows:
    time). In logs throughout: a long winning series overflows otherwise."""
    top = log_paths.max(axis=1)
    return top + np.log(np.exp(log_paths - top[:, None]).mean(axis=1))


def wealth_path(returns, fractions=FRACTIONS) -> np.ndarray:
    """The evidence bankroll after each bet, starting from 1. returns: the
    profit on each bet per unit staked, in the order made (a loss is -1, a
    win at -110 is +0.909, a push is 0). Under "no edge" - an expected
    return of zero or less on every bet given all before it - the chance
    this ever reaches x is at most 1/x."""
    r = np.asarray(returns, dtype=float)
    if r.size == 0:
        return np.array([])
    if (r < -1).any():
        raise ValueError("a bet cannot lose more than its stake")
    return np.exp(np.minimum(_log_mix(np.cumsum(np.log1p(np.outer(r, fractions)), axis=0)), 700.0))


def evidence(returns) -> dict:
    """{bets, units, wealth (now), peak (highest so far), peak_at (index of
    the bet that reached it)}. The peak counts as evidence too: the 1-in-x
    bound covers the whole path, not just its end."""
    r = np.asarray(returns, dtype=float)
    path = wealth_path(r)
    if path.size == 0:
        return {"bets": 0, "units": 0.0, "wealth": 1.0, "peak": 1.0, "peak_at": None}
    peak_at = int(path.argmax())
    peak = float(max(1.0, path[peak_at]))
    return {"bets": int(r.size), "units": float(r.sum()), "wealth": float(path[-1]), "peak": peak,
            "peak_at": peak_at if path[peak_at] > 1 else None}


def _rate_log_wealth(won: np.ndarray, rates: np.ndarray, direction: int) -> np.ndarray:
    """For each candidate rate, the highest log bankroll reached by a bet
    that the true rate is higher (direction +1) or lower (-1) than it."""
    room = np.where(direction > 0, rates, 1.0 - rates)         # the most one bet can cost per unit
    out = np.empty(rates.size)
    for j, rate in enumerate(rates):
        steps = np.outer(won - rate, direction * FRACTIONS * 1.6 / room[j])   # up to 48% of the allowed stake
        out[j] = _log_mix(np.cumsum(np.log1p(steps), axis=0)).max()
    return out


def confidence_sequence(won, confidence=CONFIDENCE, step=0.001) -> dict | None:
    """{low, high}: the win rates the series has not ruled out. Unlike an
    ordinary confidence interval this holds at every bet at once - it can
    be read after any bet, as often as wanted - which makes it wider. A
    rate is ruled out once a bet against it has multiplied its bankroll by
    2 / (1 - confidence) at any point so far. won: 1 or 0 per bet (pushes
    left out). None with no bets."""
    x = np.asarray(won, dtype=float)
    x = x[~np.isnan(x)]
    if x.size == 0:
        return None
    rates = np.arange(step, 1.0, step)
    bar = np.log(2.0 / (1.0 - confidence))
    out_low = _rate_log_wealth(x, rates, +1) >= bar      # the truth is higher than these
    out_high = _rate_log_wealth(x, rates, -1) >= bar     # ... and lower than these
    kept = rates[~(out_low | out_high)]
    return {"low": float(kept.min()), "high": float(kept.max())}


def fade_detector(won, level, alarm=ALARM_LEVEL) -> dict:
    """An alarm for "the win rate has fallen below `level`" that may start
    at any bet (a Shiryaev-Roberts e-detector): before each bet a fresh
    unit joins a bankroll betting that the rate is below `level`. If the
    rate never drops below `level`, the bankroll takes `alarm` bets or
    more, on average, to reach `alarm`. Returns {level, value (now), peak,
    tripped_at (index of the first bet at or over the alarm, or None)}."""
    x = np.asarray(won, dtype=float)
    x = x[~np.isnan(x)]
    stakes = FRACTIONS * 1.6 / (1.0 - level)
    held = np.zeros(stakes.size)
    path = np.empty(x.size)
    for i, outcome in enumerate(x):
        held = (held + 1.0) * (1.0 - stakes * (outcome - level))
        path[i] = held.mean()
    over = np.flatnonzero(path >= alarm)
    return {"level": float(level), "value": float(path[-1]) if x.size else 0.0,
            "peak": float(path.max()) if x.size else 0.0, "tripped_at": int(over[0]) if over.size else None}


def at_110(won) -> np.ndarray:
    """Returns per unit for bets at -110 from 1 / 0 / NaN (a push)."""
    x = np.asarray(won, dtype=float)
    return np.where(np.isnan(x), 0.0, np.where(x > 0, WIN_AT_110, -1.0))

"""Ranges with a hit rate that holds: conformal prediction.

Why (2026-11-20): the site gave one range, the point total "plus or minus
9", sized once by hand to catch about half of games, and nothing at all
around a player's projected yards. A range is only useful if "4 times in
5" means 4 times in 5. The usual way to get one assumes the misses follow
a bell curve; a receiver's yards do not (most games near the projection, a
few far over, none below zero).

Conformal prediction (Vovk, Gammerman and Shafer 2005; Lei et al. 2018;
Angelopoulos and Bates 2023 for a plain introduction) assumes no shape. It
takes the model's own past misses - on games it had not seen when it
predicted them - and reads the range straight off them: if 80% of past
misses fell between -19 and +31 yards, the next range is the projection
-19 to +31. The one assumption is that the next game is like the past ones
(exchangeable), and under it the stated hit rate is guaranteed for any
model, good or bad. A worse model just gets wider ranges.

Two refinements used here:

  A scale. A 90-yard projection misses by more than a 15-yard one, so
  misses are measured in units of a size fitted to the projection
  (fit_scale), and the range widens with it ("locally weighted" conformal).

  Groups. The promise above is an average over all predictions. Keeping
  the misses of small, middling and large projections apart (calibrate's
  `groups`) makes it hold within each.

If the game changes, the past stops being like the future and the promise
weakens; the player calibration is rebuilt every week from the latest
three seasons so it follows.

The same past misses also give the chance of clearing any number
(chance_over: a conformal predictive distribution, Vovk et al. 2019),
which is what a prop line or an over/under asks.
"""

import math

import numpy as np


def fit_scale(predictions, misses, floor_share=0.25) -> dict:
    """{intercept, slope, floor}: the typical size of a miss as a straight
    line in the prediction (least squares on the absolute misses), never
    below `floor_share` of the average size. With a flat line every
    prediction gets the same width."""
    p = np.asarray(predictions, dtype=float)
    size = np.abs(np.asarray(misses, dtype=float))
    if p.size < 3 or np.ptp(p) == 0:
        mean = float(size.mean()) if size.size else 1.0
        return {"intercept": mean or 1.0, "slope": 0.0, "floor": (mean or 1.0) * floor_share}
    slope, intercept = np.polyfit(p, size, 1)
    return {"intercept": float(intercept), "slope": float(slope), "floor": float(size.mean() * floor_share)}


def scale_of(scale: dict | None, predictions) -> np.ndarray:
    """The size of a typical miss at each prediction (1 with no scale)."""
    p = np.asarray(predictions, dtype=float)
    if not scale:
        return np.ones_like(p)
    return np.maximum(scale["intercept"] + scale["slope"] * p, scale["floor"])


def scores(predictions, actuals, scale: dict | None = None) -> np.ndarray:
    """Past misses (actual minus predicted) in units of their scale, sorted."""
    p = np.asarray(predictions, dtype=float)
    return np.sort((np.asarray(actuals, dtype=float) - p) / scale_of(scale, p))


def _rank_value(sorted_values: np.ndarray, rank: int) -> float:
    """The rank-th smallest (1-based), or the end if the rank is past it."""
    if rank < 1:
        return -math.inf
    if rank > sorted_values.size:
        return math.inf
    return float(sorted_values[rank - 1])


def _ranks(n: int, coverage: float) -> tuple[int, int]:
    """Which of n sorted misses bound the range: the n + 1 is the small-
    sample correction (the next miss is one more, and could fall anywhere
    among them). Rounded first, so 0.9 x 10 is 9 and not 9.000000000000002."""
    tail = (1.0 - coverage) / 2.0
    return math.floor(round(tail * (n + 1), 9)), math.ceil(round((1.0 - tail) * (n + 1), 9))


def band(sorted_scores, coverage: float) -> tuple[float, float]:
    """(low, high) offsets, in scale units, that the next miss falls
    between with probability `coverage` or more: equal shares of the
    leftover on each side, with the small-sample correction that makes the
    promise exact (the n+1 in the ranks). Infinite on a side when there
    are too few past misses to promise that much."""
    s = np.asarray(sorted_scores, dtype=float)
    n = s.size
    low_rank, high_rank = _ranks(n, coverage)
    return _rank_value(s, low_rank), _rank_value(s, high_rank)


def interval(predictions, sorted_scores, coverage: float, scale: dict | None = None, lower=None):
    """(low, high) arrays: each prediction's range at `coverage`. `lower`
    clips the low end (0 for yards and points)."""
    p = np.asarray(predictions, dtype=float)
    low, high = band(sorted_scores, coverage)
    size = scale_of(scale, p)
    lo, hi = p + low * size, p + high * size
    return (np.maximum(lo, lower) if lower is not None else lo), hi


def chance_over(prediction: float, line: float, sorted_scores, scale: dict | None = None) -> float:
    """The chance the result is over `line`: the share of past misses that
    would have carried this prediction past it, counted so that it can
    never say 0 or 1 (the result itself is one more unknown miss)."""
    s = np.asarray(sorted_scores, dtype=float)
    needed = (line - prediction) / float(scale_of(scale, [prediction])[0])
    over = s.size - np.searchsorted(s, needed, side="right")
    tied = np.searchsorted(s, needed, side="right") - np.searchsorted(s, needed, side="left")
    return float((over + 0.5 * tied + 0.5) / (s.size + 1))


# ---- A calibration small enough to ship to the page -----------------------
#
# The page needs the past misses to draw a range and to answer "what is
# the chance he clears this line?". Rather than every miss, each group's
# misses are kept as their values at GRID's evenly spaced shares (every
# half percent), which loses nothing a reader could see.
GRID = np.linspace(0.0, 1.0, 201)


def calibrate(predictions, actuals, groups: int = 1, scaled: bool = False) -> dict:
    """{n, scale, edges, groups: [{n, scores}]} from past predictions and
    results the model had not seen when it made them.

    groups > 1 splits the predictions into that many equal-sized bands by
    size and keeps each band's misses apart, so the hit rate holds within
    small projections and within large ones, not just on average (a
    "Mondrian" split). Without it a range sized for everyone was too wide
    for a 15-yard projection and too narrow for a 90-yard one: 80% ranges
    caught 91% of the smallest third and 60% of the largest."""
    p = np.asarray(predictions, dtype=float)
    a = np.asarray(actuals, dtype=float)
    keep = ~(np.isnan(p) | np.isnan(a))
    p, a = p[keep], a[keep]
    scale = fit_scale(p, a - p) if scaled else None
    edges = np.quantile(p, np.linspace(0, 1, groups + 1)[1:-1]) if groups > 1 and p.size else np.array([])
    member = np.searchsorted(edges, p)
    out = []
    for g in range(len(edges) + 1):
        s = scores(p[member == g], a[member == g], scale)
        out.append({"n": int(s.size), "scores": [round(float(v), 4) for v in np.quantile(s, GRID)] if s.size else []})
    return {"n": int(p.size), "scale": scale, "edges": [round(float(e), 4) for e in edges], "groups": out}


def _group(calibration: dict, prediction: float) -> dict:
    return calibration["groups"][int(np.searchsorted(np.asarray(calibration["edges"]), prediction))]


def _offset(group: dict, share: float) -> float:
    """The miss (in scale units) that `share` of the group's fall below."""
    return float(np.interp(min(1.0, max(0.0, share)), GRID, group["scores"]))


def range_for(calibration: dict, prediction: float, coverage: float, lower=None) -> tuple[float, float] | None:
    """(low, high) for one prediction at `coverage`, from its group's past
    misses. None when the group has no misses to go on."""
    group = _group(calibration, prediction)
    n = group["n"]
    if not n:
        return None
    size = float(scale_of(calibration.get("scale"), [prediction])[0])
    low_rank, high_rank = _ranks(n, coverage)
    low = prediction + _offset(group, low_rank / n) * size
    high = prediction + _offset(group, high_rank / n) * size
    return (max(low, lower) if lower is not None else low), high


def chance_over_line(calibration: dict, prediction: float, line: float) -> float | None:
    """The chance the result is over `line`, from the group's past misses
    (see chance_over). Never exactly 0 or 1."""
    group = _group(calibration, prediction)
    n = group["n"]
    if not n:
        return None
    size = float(scale_of(calibration.get("scale"), [prediction])[0])
    below = float(np.interp((line - prediction) / size, group["scores"], GRID, left=0.0, right=1.0))
    return (n * (1.0 - below) + 0.5) / (n + 1)

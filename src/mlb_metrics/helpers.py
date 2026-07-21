"""Vectorized event classifiers.

The original script redefined these same six classifiers (hit, bases, stk,
homer, ob, ab - `bases` and `sv` were literally identical functions) inline,
once per metric section, each applied row-by-row via `.apply(fn, axis=1)`.
They're consolidated here as vectorized `Series -> Series` functions over the
`events` column, which is both faster and removes ~150 lines of duplication.
Classification logic (which events count as a hit, an official at-bat, etc.)
is unchanged from the original.
"""

import pandas as pd

HIT_EVENTS = {"single", "double", "triple", "home_run"}
ON_BASE_EVENTS = HIT_EVENTS | {"walk", "hit_by_pitch"}
NON_AT_BAT_EVENTS = {"walk", "hit_by_pitch"}
STRIKEOUT_WALK_HBP_EVENTS = {"strikeout", "strikeout_double_play", "walk", "hit_by_pitch"}
BASES_BY_EVENT = {"single": 1, "double": 2, "triple": 3, "home_run": 4}


def is_hit(events: pd.Series) -> pd.Series:
    return events.isin(HIT_EVENTS).astype(int)


def total_bases(events: pd.Series) -> pd.Series:
    """Bases gained on the event (0 for outs/walks/etc). Also used as `sv` (slugging value)."""
    return events.map(BASES_BY_EVENT).fillna(0).astype(int)


def is_strikeout_walk_hbp(events: pd.Series) -> pd.Series:
    return events.isin(STRIKEOUT_WALK_HBP_EVENTS).astype(int)


def is_home_run(events: pd.Series) -> pd.Series:
    return (events == "home_run").astype(int)


def is_on_base(events: pd.Series) -> pd.Series:
    return events.isin(ON_BASE_EVENTS).astype(int)


def is_official_at_bat(events: pd.Series) -> pd.Series:
    return (~events.isin(NON_AT_BAT_EVENTS)).astype(int)

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
STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}
STRIKEOUT_WALK_HBP_EVENTS = STRIKEOUT_EVENTS | {"walk", "hit_by_pitch"}
BASES_BY_EVENT = {"single": 1, "double": 2, "triple": 3, "home_run": 4}

# Outs recorded on the play, for innings-pitched counting (age_curve.py's
# pitcher metrics - K9/BB9/HR9/FIP need a per-9-innings denominator).
# "fielders_choice" (no out on THIS batter, but not necessarily 0 outs on
# the overall play - Statcast doesn't distinguish from a single row) is
# conservatively counted as 0, same simplification NON_AT_BAT_EVENTS
# already makes elsewhere in this project rather than modeling every edge
# case exactly.
OUTS_BY_EVENT = {
    "field_out": 1,
    "force_out": 1,
    "strikeout": 1,
    "fielders_choice_out": 1,
    "grounded_into_double_play": 2,
    "double_play": 2,
    "strikeout_double_play": 2,
}


def is_hit(events: pd.Series) -> pd.Series:
    return events.isin(HIT_EVENTS).astype(int)


def total_bases(events: pd.Series) -> pd.Series:
    """Bases gained on the event (0 for outs/walks/etc). Also used as `sv` (slugging value)."""
    return events.map(BASES_BY_EVENT).fillna(0).astype(int)


def is_strikeout_walk_hbp(events: pd.Series) -> pd.Series:
    return events.isin(STRIKEOUT_WALK_HBP_EVENTS).astype(int)


def is_strikeout(events: pd.Series) -> pd.Series:
    return events.isin(STRIKEOUT_EVENTS).astype(int)


def is_walk(events: pd.Series) -> pd.Series:
    return (events == "walk").astype(int)


# DK Classic MLB fantasy scoring counts an intentional walk as a walk too -
# a genuinely different rule from is_walk above (used by pitcher_form.py's
# BB9, traditional_stats.py's Age Curves BB9, and dfs_backtest.py's
# pitcher-actual BB - all three already validated against the plain "walk"
# definition). Deliberately a SEPARATE classifier rather than widening
# is_walk, so this doesn't silently perturb those three already-shipped
# numbers. In practice "intent_walk" rows never reach this function today
# anyway - config.COUNTED_EVENTS (the upstream completed-PA filter every
# caller's input already passed through) doesn't include "intent_walk", so
# real intentional walks are NOT currently credited to DK_Points_Hitter - a
# known, small (~0.3% of real plate appearances) gap, flagged honestly
# rather than fixed by touching COUNTED_EVENTS, which many other
# already-validated metrics (WAVE, PAVE, WHOPS) share and would need
# re-validating if its PA population changed.
def is_walk_for_dk_scoring(events: pd.Series) -> pd.Series:
    return events.isin({"walk", "intent_walk"}).astype(int)


def is_hit_by_pitch(events: pd.Series) -> pd.Series:
    return (events == "hit_by_pitch").astype(int)


def outs_recorded(events: pd.Series) -> pd.Series:
    """Outs recorded on the play (0, 1, or 2) - see OUTS_BY_EVENT."""
    return events.map(OUTS_BY_EVENT).fillna(0).astype(int)


def is_home_run(events: pd.Series) -> pd.Series:
    return (events == "home_run").astype(int)


def is_on_base(events: pd.Series) -> pd.Series:
    return events.isin(ON_BASE_EVENTS).astype(int)


def is_official_at_bat(events: pd.Series) -> pd.Series:
    return (~events.isin(NON_AT_BAT_EVENTS)).astype(int)


def estimate_rbi(df: pd.DataFrame) -> pd.Series:
    """Runs driven in on this completed plate appearance, approximated as
    `post_bat_score - bat_score` on the PA's own final-pitch row (the one
    row per PA data.completed_events already collapses every caller's
    input to) - the standard Statcast RBI-approximation technique. Needs
    the whole row (not just `events`), unlike every other classifier here.

    Two known, accepted simplifications, same documented-tradeoff category
    as this project's FIP-for-ER estimate:
    - This is the LAST pitch's own score delta, not a true first-pitch-of-
      PA to last-pitch-of-PA delta - a run that scored on an EARLIER pitch
      of the same PA (e.g. a wild pitch with the bases loaded before ball
      four) is missed. Rare in practice.
    - Cannot distinguish a legitimate RBI from a run that scored on a
      fielding error or the batter's own GIDP third-out (no RBI credited
      under official rules in either case) - both look identical to a bare
      score delta under this technique.
    Clipped at 0 (a score can only increase within one PA - a negative
    delta would only ever indicate a data artifact)."""
    return (df["post_bat_score"] - df["bat_score"]).clip(lower=0).astype(int)

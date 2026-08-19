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


def is_non_at_bat_event(events: pd.Series) -> pd.Series:
    """Walk or HBP - the only two event types excluded from the AB
    denominator when converting a per-PA rate to a per-AB rate (a
    strikeout IS an official at-bat, unlike a walk/HBP - see
    pitchers.py's PAVE for the real bug this distinction fixed: an
    earlier version used is_strikeout_walk_hbp here instead, which
    wrongly excluded strikeouts from the AB count too, inflating
    hit-rate-against for exactly the pitchers who rack up the most
    strikeouts). Equivalent to `1 - is_official_at_bat`, exposed as its
    own classifier for the same reason is_official_at_bat is: callers
    read the intent directly rather than re-deriving it from a negation."""
    return events.isin(NON_AT_BAT_EVENTS).astype(int)


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


def is_batted_ball(type_col: pd.Series) -> pd.Series:
    """A pitch that was put in play ("X", Statcast's own pitch-result code
    for the `type` column) - the subset of completed-PA rows
    (data.completed_events's own output) that carry real batted-ball data
    (launch_speed/launch_angle/estimated_ba_using_speedangle/
    estimated_woba_using_speedangle/launch_speed_angle). A strikeout/walk/
    HBP has none of these populated - averaging over them unfiltered would
    silently bias every quality-of-contact rate toward 0/NaN, not measure
    contact quality at all.

    Returns a boolean mask, unlike every other classifier in this file
    (which return 0/1 int Series meant to be summed via a stat_fns entry) -
    this is meant for FILTERING `dt` down to batted-ball rows before
    windowing/blending (see hitters.compute_quality_of_contact), not for
    counting."""
    return type_col == "X"


def is_barrel(launch_speed_angle: pd.Series) -> pd.Series:
    """Statcast's own real quality-of-contact bucket (`launch_speed_angle`,
    an integer 0-6 assigned to every batted ball) - 6 is Statcast's own
    definition of a "barrel" (the launch-speed/launch-angle combination
    most correlated with extra-base value). No literal `barrel` boolean
    column exists on a real Statcast row; this IS the real barrel
    classification (not an approximation reconstructed from the raw
    launch_speed/launch_angle formula).

    A real gap confirmed against the actual persisted data (not just a
    synthetic-test edge case): some real batted-ball rows have a null
    launch_speed_angle even though Statcast otherwise tracked the batted
    ball - the `== 6` comparison against a pandas nullable-dtype column
    returns pd.NA (not False) for those rows (three-valued comparison
    logic), which `.astype(int)` alone cannot convert - fillna(False)
    resolves "no tracked quality bucket" to "not a barrel" before casting,
    the same "missing data reads as a real, honest 0, never a crash or a
    fabricated value" precedent every other classifier here follows."""
    return (launch_speed_angle == 6).fillna(False).astype(int)


PITCH_TYPE_FAMILY = {
    # Fastballs: thrown hard and relatively straight, or with modest cut/sink.
    "FF": "fastball", "SI": "fastball", "FC": "fastball", "FA": "fastball",
    # Breaking balls: sharp lateral/downward break off a fastball-speed arm action.
    "SL": "breaking", "ST": "breaking", "CU": "breaking", "KC": "breaking",
    "SV": "breaking", "CS": "breaking",
    # Offspeed: same arm action as a fastball, thrown noticeably slower to
    # disrupt timing (changeups/splitters) or a handful of rare novelty
    # pitches with no real fastball/breaking analog.
    "CH": "offspeed", "FS": "offspeed", "FO": "offspeed", "EP": "offspeed",
    "KN": "offspeed",
    # Deliberately unmapped (pitch_type_family returns NaN for these, same
    # as any code not listed at all): "PO" (pitchout - not a real pitch to
    # the batter) and "UN"/null (Statcast couldn't classify it). A rare/
    # unclassifiable pitch_type is dropped from arsenal-mix/pitch-family
    # windowing rather than guessed into a family - see
    # pitchers.compute_pitch_arsenal and hitters.compute_pitch_family_rates.
}


def pitch_type_family(pitch_type: pd.Series) -> pd.Series:
    """Groups Statcast's real `pitch_type` codes into the same three-bucket
    fastball/breaking/offspeed scheme Baseball Savant's own pitch-arsenal
    pages use, confirmed against every real code observed in the actual
    persisted `data/raw/statcast_2026.parquet` (FF/SI/SL/CH/ST/FC/CU/FS/KC/
    SV/EP/FA/FO/KN/CS/PO/UN, plus real nulls - see PITCH_TYPE_FAMILY above
    for the exact mapping). Returns NaN (not a fabricated fourth bucket) for
    any code not in the mapping, including real nulls - about 0.4% of real
    pitches in the persisted data, mostly pitchouts and Statcast's own rare
    "couldn't classify" rows.

    Unlike `is_batted_ball`/`is_barrel` above, this returns the family
    label itself (a string, or NaN), not a 0/1 count - it's meant for
    grouping/filtering (`dt.assign(pitch_family=...)`, then window by that
    column) rather than direct summation."""
    return pitch_type.map(PITCH_TYPE_FAMILY)


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

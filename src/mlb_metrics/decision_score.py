"""Decision Score: a plate-discipline metric quantifying, per pitch,
whether swinging or taking was the "advised" choice for THIS batter right
now - given the count, the pitch's real Statcast zone, their own recency-
windowed OPS-equivalent production in that specific zone, and the game
situation - then scoring how consistently they actually do the advised
thing.

Deliberately self-referential: "advised" is built from the batter's OWN
windowed OPS (overall, and per zone, each shrunk toward their own
overall rate via helpers.shrink_rate - never a league prior), not a
league-average run-value model. "Is this a good pitch FOR ME, right now"
- not "is this objectively a good pitch to hit."

Two input frames, both from pipeline.py, both widened specifically for
this module:
- `pa_events` (pipeline.build_pitch_events's output): one row per
  completed plate appearance, carrying the REAL outcome (`events`) and
  the zone/count the PA ended on - used to compute a batter's real
  windowed OBP/SLG/OPS, overall and per zone (compute_batter_overall_ops/
  compute_zone_ops).
- `all_pitches` (pipeline.build_all_pitch_events's output): every real
  pitch a batter saw, carrying `description` (swing/take, via
  helpers.is_swing), `zone`, `balls`/`strikes`, and real situational
  columns (`inning`/`bat_score`/`fld_score`/`on_2b`/`on_3b`) - used to
  classify each pitch's advised action and whether the batter's real
  choice matched it (compute_decision_advice/compute_decision_score).

VALIDATED (with an honest caveat): scripts/backtest_decision_score.py's
real, no-lookahead, out-of-sample backtest confirms the core zone signal
- PA-ending pitches where the batter's real choice matched the zone-
based advice score a real, statistically significant better outcome
(both pooled and per-batter-paired, p<0.0001) than mismatched ones. The
count-context/situational-leverage multipliers, however, did NOT
independently improve on that result at any magnitude swept - both ship
at 1.0 (no-op); see config.py's own DECISION_SCORE_* comments for the
real numbers."""

import numpy as np
import pandas as pd

from mlb_metrics import config, helpers, hitters

# Statcast's own real zone classification (helpers.OUT_OF_ZONE_CODES):
# 1-9 is the actual 3x3 strike zone grid, 11-14 are the four real "chase"
# quadrants just outside it - no other codes exist on a real classified
# pitch (confirmed against the actual persisted zone value_counts, same
# as helpers.py's own OUT_OF_ZONE_CODES comment).
REAL_ZONE_CODES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14]

# Standard, real baseball count buckets (not invented here): 3-0/3-1/2-0
# are "hitter's counts" (a walk is valuable, no strikeout risk on a
# take); 0-2/1-2 are "pitcher's counts" (a called third strike ends the
# PA outright on a take). Everything else - including the symmetric 3-2
# full count - is neutral.
HITTER_COUNTS = {(3, 0), (3, 1), (2, 0)}
PITCHER_COUNTS = {(0, 2), (1, 2)}


def classify_count_context(balls: pd.Series, strikes: pd.Series) -> pd.Series:
    """"hitter"/"pitcher"/"neutral" per pitch - see HITTER_COUNTS/
    PITCHER_COUNTS above. A missing balls/strikes value (never happens on
    a real Statcast row, but degrades safely on an older/narrower
    fixture) reads as "neutral" - comparisons against NaN are False, so
    it simply never matches either special bucket, the same "missing
    data gets a safe default, not a crash" precedent this project's other
    classifiers already follow."""
    hitter_mask = (
        ((balls == 3) & (strikes == 0)) | ((balls == 3) & (strikes == 1)) | ((balls == 2) & (strikes == 0))
    )
    pitcher_mask = ((balls == 0) & (strikes == 2)) | ((balls == 1) & (strikes == 2))
    result = pd.Series("neutral", index=balls.index)
    result[hitter_mask] = "hitter"
    result[pitcher_mask] = "pitcher"
    return result


def _ops_stat_fns() -> dict:
    return {
        "ob": lambda df: helpers.is_on_base(df["events"]),
        "ab": lambda df: helpers.is_official_at_bat(df["events"]),
        "tb": lambda df: helpers.total_bases(df["events"]),
    }


def compute_batter_overall_ops(pa_events: pd.DataFrame, windows: list) -> pd.DataFrame:
    """Each batter's own windowed-blended OBP/SLG/OPS - no zone/side
    split. Used both as compute_zone_ops's shrinkage prior and as
    compute_decision_advice's swing-threshold baseline.

    Reuses hitters.blend_windows (the exact same windowing/blending
    primitive WAVE/WHOPS/WTB already use) via a dummy constant `_all`
    column instead of a real side/zone split - `blend_windows` always
    needs *some* categorical `column`/`side` pair to filter on, and this
    is the cleanest way to reuse that exact, already-tested machinery for
    an unsplit rate rather than re-deriving the same window/blend logic
    a second time.

    A batter with zero real plate appearances in `pa_events` never
    appears at all (no row, not a fabricated 0) - matches every other
    windowed rate in this project."""
    dt = pa_events.assign(_all="all")
    stat_fns = _ops_stat_fns()
    rate_fns = {
        "OBP": lambda agg: agg["ob"] / agg["n"],
        "SLG": lambda agg: agg["tb"] / agg["ab"],
    }
    blended, _ = hitters.blend_windows(dt, windows, "all", stat_fns, rate_fns, column="_all")
    blended["OPS"] = blended["OBP"].fillna(0) + blended["SLG"].fillna(0)
    return blended.rename(columns={"batter": "key_mlbam"})[["key_mlbam", "OBP", "SLG", "OPS"]]


def compute_zone_ops(
    pa_events: pd.DataFrame, overall_ops: pd.DataFrame, windows: list, shrinkage_strength: float | None = None
) -> pd.DataFrame:
    """One row per (batter, zone) that batter has ended at least one real
    plate appearance in: windowed-blended OBP/SLG/OPS for PAs that ended
    on a pitch in that real Statcast zone, each per-window rate shrunk
    (helpers.shrink_rate, `shrinkage_strength` real-PA pseudo-
    observations; None -> config.DECISION_SCORE_ZONE_SHRINKAGE_STRENGTH -
    same override-with-a-config-default shape hitters.compute_wave's own
    `shrinkage_strength` param already uses, needed here so
    scripts/backtest_decision_score.py can sweep candidate values without
    mutating the config module) toward THAT SAME BATTER'S OWN overall
    windowed OBP/SLG from `overall_ops` BEFORE the windows are blended -
    the exact same shrink-then-blend order hitters.compute_wave already
    establishes (shrink_rate is called inside each rate_fn, using that
    window's own raw count/n, not applied after blending).

    Deliberately self-referential - their own average, never a league
    prior - "their windowed ops in that zone" is a relative-to-themselves
    signal.

    A (batter, zone) pair with zero real PAs across every window (a real
    absence, not a fabricated 0) never gets a row - a caller merging this
    onto a wider frame gets a real NaN for that batter/zone, not a
    fabricated 0 (compute_decision_advice's own fillna(0) is the
    intentional point where that absence becomes "advise taking" by
    default, not this function's job)."""
    overall_lookup = overall_ops.set_index("key_mlbam")[["OBP", "SLG"]]
    strength = config.DECISION_SCORE_ZONE_SHRINKAGE_STRENGTH if shrinkage_strength is None else shrinkage_strength

    def _shrunk_obp(agg):
        prior = agg["batter"].map(overall_lookup["OBP"]).fillna(0)
        return helpers.shrink_rate(agg["ob"], agg["n"], prior, strength)

    def _shrunk_slg(agg):
        prior = agg["batter"].map(overall_lookup["SLG"]).fillna(0)
        return helpers.shrink_rate(agg["tb"], agg["ab"], prior, strength)

    stat_fns = _ops_stat_fns()
    rate_fns = {"OBP": _shrunk_obp, "SLG": _shrunk_slg}

    rows = []
    for zone in REAL_ZONE_CODES:
        blended, full_counts = hitters.blend_windows(pa_events, windows, zone, stat_fns, rate_fns, column="zone")
        if full_counts is None or full_counts.empty:
            continue
        blended = blended.assign(zone=zone)
        rows.append(blended)

    if not rows:
        return pd.DataFrame(columns=["key_mlbam", "zone", "Zone_OPS"])

    long = pd.concat(rows, ignore_index=True)
    long["Zone_OPS"] = long["OBP"].fillna(0) + long["SLG"].fillna(0)
    return long.rename(columns={"batter": "key_mlbam"})[["key_mlbam", "zone", "Zone_OPS"]]


def compute_decision_advice(
    all_pitches: pd.DataFrame,
    overall_ops: pd.DataFrame,
    zone_ops: pd.DataFrame,
    hitter_multiplier: float | None = None,
    pitcher_multiplier: float | None = None,
    leverage_multiplier: float | None = None,
    leverage_min_inning: int | None = None,
    leverage_max_score_diff: int | None = None,
) -> pd.Series:
    """Per pitch: "swing" or "take", whichever the data says was better
    FOR THIS BATTER right now. Returns a Series aligned to `all_pitches`'s
    index.

    threshold = batter's own overall OPS * count_multiplier *
    situation_multiplier; advised = "swing" if that batter's shrunk
    Zone_OPS for THIS pitch's zone clears the threshold, else "take".

    Every keyword arg here defaults to its config.DECISION_SCORE_* value
    (None means "use the config default") - explicit overrides exist so
    scripts/backtest_decision_score.py can sweep candidate values without
    mutating the config module, the same shape hitters.compute_wave's own
    `shrinkage_strength` param already establishes.

    A batter/pitch with no real OPS/Zone_OPS signal at all (a name not
    yet seen in `pa_events`, or a null zone - ~0.4% of real pitches,
    mostly pitchouts/Statcast's own "couldn't classify" rows) fills to 0
    on both sides - both compare to 0, which reads as "take" (0 >= a
    positive threshold is False) unless the batter's own overall OPS is
    also 0, a safe, honest default rather than a fabricated 0.5.

    `merge` always returns a fresh 0..n-1 index regardless of
    `all_pitches`'s own - both merges here are left joins against a
    frame keyed uniquely on (batter) / (batter, zone), so row order and
    count are preserved exactly; the final Series is still built against
    `all_pitches.index` explicitly (not the merged frame's own reset
    index) so a caller comparing/assigning it back onto `all_pitches`
    (compute_decision_score does exactly this) can never silently
    misalign even if `all_pitches` itself doesn't already have a plain
    RangeIndex."""
    df = all_pitches.reset_index(drop=True).merge(
        overall_ops.rename(columns={"key_mlbam": "batter", "OPS": "OPS_overall"})[["batter", "OPS_overall"]],
        on="batter", how="left",
    )
    df = df.merge(
        zone_ops.rename(columns={"key_mlbam": "batter"})[["batter", "zone", "Zone_OPS"]],
        on=["batter", "zone"], how="left",
    )
    ops_overall = df["OPS_overall"].fillna(0)
    zone_ops_col = df["Zone_OPS"].fillna(0)

    hitter_multiplier = config.DECISION_SCORE_HITTER_COUNT_MULTIPLIER if hitter_multiplier is None else hitter_multiplier
    pitcher_multiplier = (
        config.DECISION_SCORE_PITCHER_COUNT_MULTIPLIER if pitcher_multiplier is None else pitcher_multiplier
    )
    leverage_multiplier = (
        config.DECISION_SCORE_HIGH_LEVERAGE_MULTIPLIER if leverage_multiplier is None else leverage_multiplier
    )
    leverage_min_inning = (
        config.DECISION_SCORE_HIGH_LEVERAGE_MIN_INNING if leverage_min_inning is None else leverage_min_inning
    )
    leverage_max_score_diff = (
        config.DECISION_SCORE_HIGH_LEVERAGE_MAX_SCORE_DIFF
        if leverage_max_score_diff is None else leverage_max_score_diff
    )

    count_context = classify_count_context(df["balls"], df["strikes"])
    count_multiplier = pd.Series(1.0, index=df.index)
    count_multiplier[count_context == "hitter"] = hitter_multiplier
    count_multiplier[count_context == "pitcher"] = pitcher_multiplier

    score_diff = (df["bat_score"] - df["fld_score"]).abs()
    runner_in_scoring_position = df["on_2b"].notna() | df["on_3b"].notna()
    high_leverage = (
        (df["inning"] >= leverage_min_inning)
        & (score_diff <= leverage_max_score_diff)
        & runner_in_scoring_position
    )
    situation_multiplier = pd.Series(1.0, index=df.index)
    situation_multiplier[high_leverage] = leverage_multiplier

    threshold = ops_overall * count_multiplier * situation_multiplier
    return pd.Series(np.where(zone_ops_col >= threshold, "swing", "take"), index=all_pitches.index)


def compute_decision_score(
    all_pitches: pd.DataFrame,
    pa_events: pd.DataFrame,
    windows: list | None = None,
    shrinkage_strength: float | None = None,
    hitter_multiplier: float | None = None,
    pitcher_multiplier: float | None = None,
    leverage_multiplier: float | None = None,
    leverage_min_inning: int | None = None,
    leverage_max_score_diff: int | None = None,
) -> pd.DataFrame:
    """Ties compute_batter_overall_ops/compute_zone_ops/
    compute_decision_advice together into one batter-level, windowed-
    blended Decision_Score (0-100): the % of real pitches where the
    batter's actual swing/take choice matched the advised action.

    `windows` defaults to config.WAVE_WINDOWS (reused, not a new
    dedicated window schedule - same "reuse unless a real reason to
    differ" precedent hitters.compute_quality_of_contact's own docstring
    already documents for config.WHOPS_WTB_WINDOWS). Every other keyword
    arg passes straight through to compute_zone_ops/compute_decision_advice
    (see their own docstrings) - all default to their config.DECISION_SCORE_*
    value, overridable for scripts/backtest_decision_score.py's sweep.

    Returns [key_mlbam, Decision_Score, Decision_Score_N] -
    Decision_Score_N is the real full-season pitch count backing the
    blend (mirrors WAVE's PA_L/PA_R - lets a reader/caller see the real
    sample size behind the number, not just the number itself). A batter
    with zero real pitches seen never gets a row."""
    windows = config.WAVE_WINDOWS if windows is None else windows

    overall_ops = compute_batter_overall_ops(pa_events, windows)
    zone_ops = compute_zone_ops(pa_events, overall_ops, windows, shrinkage_strength=shrinkage_strength)
    advice = compute_decision_advice(
        all_pitches, overall_ops, zone_ops,
        hitter_multiplier=hitter_multiplier, pitcher_multiplier=pitcher_multiplier,
        leverage_multiplier=leverage_multiplier, leverage_min_inning=leverage_min_inning,
        leverage_max_score_diff=leverage_max_score_diff,
    )

    matched = (helpers.is_swing(all_pitches["description"]) == (advice == "swing").astype(int)).astype(int)
    dt = all_pitches.assign(_all="all", matched=matched.values)

    stat_fns = {"matched_sum": lambda df: df["matched"]}
    rate_fns = {"Decision_Score": lambda agg: agg["matched_sum"] / agg["n"]}
    blended, full_counts = hitters.blend_windows(dt, windows, "all", stat_fns, rate_fns, column="_all")

    result = blended.rename(columns={"batter": "key_mlbam"})
    result["Decision_Score"] = (result["Decision_Score"] * 100).fillna(0)
    result = result.merge(
        full_counts.rename(columns={"batter": "key_mlbam", "n": "Decision_Score_N"})[["key_mlbam", "Decision_Score_N"]],
        on="key_mlbam", how="left",
    )
    return result[["key_mlbam", "Decision_Score", "Decision_Score_N"]]

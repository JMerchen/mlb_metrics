"""Hitter metrics: WAVE (at-bat based hit probability), Game_Hit_Probability
(game-level hit rate), WTB (weighted total bases), WHOPS/RC (weighted
OPS / runs-created proxy - computed but, as in the original script, not
currently written to any output; kept here as a reusable building block for
future matchup work rather than wired into the CSV pipeline), and
compute_quality_of_contact (real Statcast batted-ball-quality rates -
Exit_Velo/Barrel_Rate/xBA/xwOBA - added to widen dfs_ml.HITTER_FEATURE_COLUMNS
beyond outcome-rate-only signals; see that module's docstring for why), and
compute_pitch_family_rates (WAVE split by the PA-ending pitch's fastball/
breaking/offspeed family instead of by pitcher handedness - the batter half
of matchup.py's pitch-type-specific platoon matchup, see that function's
own docstring), compute_home_road_split (WAVE split by home/road instead
of by pitcher handedness - same wave-logic small-sample guard applied to
a batter's home/away performance, see that function's own docstring), and
compute_plate_discipline (real per-pitch Whiff_Rate/
Chase_Rate - unlike every other function here, built from EVERY pitch a
batter saw, not just the PA-ending one, since a swing decision is a
per-pitch signal, not a per-PA one).

All three of WAVE/WHOPS/WTB share the same shape: split at-bats by the
opposing pitcher's throwing hand, compute a rate per recency window, and
blend the windows with fixed weights (see config.py). `_blend_windows`
implements that shared shape once instead of the ~6x duplicated inline
blocks in the original script.

Note on population: each blended side-metric is anchored on batters who
have at least one full-season at-bat *against that side* (matching the
original's `rfull.merge(...)` anchor) - a batter who has only ever faced
left-handed pitching, for instance, is dropped from the R-side blend and
picks up a WAVE_R of 0 rather than being excluded outright, exactly as in
the original.
"""

import pandas as pd

from mlb_metrics import config, helpers


def _window_key(days_back):
    return "full" if days_back is None else f"d{days_back}"


def _slice_by_days(frame: pd.DataFrame, latest, days_back):
    if days_back is None:
        return frame
    cutoff = latest - pd.Timedelta(days=days_back)
    return frame[frame["game_date"] >= cutoff]


def _side_window_agg(
    dt: pd.DataFrame, latest, days_back, side: str, stat_fns: dict, column: str = "p_throws"
) -> pd.DataFrame:
    """One row per batter: `n` (at-bat count) plus the sum of each stat in
    `stat_fns`, restricted to the window and to rows where `column` equals
    `side`. `column` defaults to "p_throws" (every original caller here -
    WAVE/WHOPS/WTB/extended DK rates/quality-of-contact - splits by the
    opposing pitcher's throwing hand); compute_pitch_family_rates passes
    column="pitch_family" instead to split the exact same windowing/
    blending machinery by pitch-type family rather than by handedness - the
    "side" concept generalizes cleanly to any categorical column with a
    small, fixed set of values.

    Each `stat_fns` value is `fn(side_df) -> Series`, given the WHOLE
    windowed/sided frame (not just its `events` column) - needed by stats
    like `helpers.estimate_rbi` that read more than one column
    (`bat_score`/`post_bat_score`). Existing `events`-only classifiers
    (`helpers.is_hit`, etc.) are passed as `lambda df: helpers.is_hit(df["events"])`."""
    window_df = _slice_by_days(dt, latest, days_back)
    side_df = window_df[window_df[column] == side].copy()
    side_df["n"] = 1
    for name, fn in stat_fns.items():
        side_df[name] = fn(side_df)
    cols = ["batter", "n"] + list(stat_fns.keys())
    return side_df[cols].groupby("batter", as_index=False).sum()


def _blend_windows(dt: pd.DataFrame, windows, side: str, stat_fns: dict, rate_fns: dict, column: str = "p_throws"):
    """Blend one or more rate functions across `windows` for one side of `column`.

    rate_fns: dict of name -> function(window_agg_df) -> per-batter rate Series.
    Returns (blended_df, full_season_counts) where blended_df has columns
    [batter, *rate_fns.keys()] and full_season_counts has
    [batter, n, *stat_fns.keys()] - the RAW (unshrunk) full-season counts,
    exposed so a caller can compute a real confidence interval on the
    full-season rate (see compute_wave) without re-deriving them from
    `dt` a second time.
    """
    latest = dt["game_date"].max()
    blended = {name: None for name in rate_fns}
    full_counts = None

    for days_back, weight in windows:
        agg = _side_window_agg(dt, latest, days_back, side, stat_fns, column=column)
        base = agg[["batter"]]
        for name, rate_fn in rate_fns.items():
            rate = rate_fn(agg).fillna(0)
            contribution = base.assign(rate=rate.values).set_index("batter")["rate"] * weight
            blended[name] = (
                contribution if blended[name] is None else blended[name].add(contribution, fill_value=0)
            )
        if days_back is None:
            full_counts = agg[["batter", "n"] + list(stat_fns.keys())]

    result = pd.DataFrame(blended)
    result.index.name = "batter"
    result = result.reset_index()
    return result, full_counts


def compute_wave(dt: pd.DataFrame, shrinkage_strength: float | None = None) -> pd.DataFrame:
    """Recency-weighted at-bat hit rate vs L/R pitching, converted to a
    per-game hit probability. Returns key_mlbam, WAVE, WAVE_L, WAVE_R,
    l_at_bat, r_at_bat, probability, probability_L, probability_R,
    WAVE_CI_Low, WAVE_CI_High, probability_CI_Low, probability_CI_High.

    `shrinkage_strength` (real-unit at-bats; None -> config.WAVE_SHRINKAGE_STRENGTH)
    pulls each per-window rate toward the real league-average AB hit rate
    computed from THIS SAME `dt` (no new fetch, no lookahead - it's
    whatever historical data is already loaded for this run, the same
    "compute the league average from currently-loaded data" pattern
    pitcher_matchup.attach_opponent_offense's league_bases_pg already
    uses) via helpers.shrink_rate - see quant-analytics item #3's
    "Bayesian shrinkage" README section. 0.0 (the live default) is the
    exact null hypothesis: bit-for-bit identical to the unshrunk
    hit/n division this function always used before.

    `WAVE_CI_Low`/`WAVE_CI_High` (quant-analytics item #3, slice 3): a
    real Wilson-score confidence interval (helpers.wilson_ci) on the
    COMBINED (both-hands pooled), RAW full-season hit rate - the one
    component of this blend that's a single, well-defined binomial
    proportion, unlike WAVE itself (a blend of 4 overlapping, nested
    recency windows with no closed-form interval - see README's
    "Confidence intervals" section). Independent of `shrinkage_strength`:
    the interval describes the raw estimator's sampling uncertainty, not
    the shrunk point estimate. `probability_CI_Low`/`probability_CI_High`
    are the same interval pushed through the monotonic
    1-(1-rate)**trials transform (valid, since that transform is
    strictly increasing over [0,1])."""
    strength = config.WAVE_SHRINKAGE_STRENGTH if shrinkage_strength is None else shrinkage_strength
    league_rate = helpers.is_hit(dt["events"]).sum() / len(dt) if len(dt) else 0.0
    stat_fns = {"hit": lambda df: helpers.is_hit(df["events"])}
    rate_fns = {"rate": lambda agg: helpers.shrink_rate(agg["hit"], agg["n"], league_rate, strength)}

    r_blended, r_full = _blend_windows(dt, config.WAVE_WINDOWS, "R", stat_fns, rate_fns)
    l_blended, l_full = _blend_windows(dt, config.WAVE_WINDOWS, "L", stat_fns, rate_fns)

    # Renamed per-side up front (not just "n") - both r_full/l_full now
    # also carry a raw "hit" column (widened _blend_windows return), so
    # merging them without renaming would collide into pandas' hit_x/hit_y
    # merge suffixes.
    wave = r_full.rename(columns={"n": "r_at_bat", "hit": "r_hit"})
    wave = wave.merge(r_blended.rename(columns={"rate": "WAVE_R"}), on="batter", how="left")
    wave = wave.merge(l_full.rename(columns={"n": "l_at_bat", "hit": "l_hit"}), on="batter", how="left")
    wave = wave.merge(l_blended.rename(columns={"rate": "WAVE_L"}), on="batter", how="left")
    wave = wave.fillna(0)

    wave["abt"] = wave["l_at_bat"] + wave["r_at_bat"]
    wave["abtl"] = wave["l_at_bat"] / wave["abt"]
    wave["abtr"] = wave["r_at_bat"] / wave["abt"]
    wave["WAVE"] = wave["WAVE_R"] * wave["abtr"] + wave["WAVE_L"] * wave["abtl"]
    wave = wave.rename(columns={"batter": "key_mlbam"})

    wave["probability"] = 1 - (1 - wave["WAVE"]) ** config.WAVE_TRIALS_PER_GAME
    wave["probability_L"] = 1 - (1 - wave["WAVE_L"]) ** config.WAVE_TRIALS_PER_GAME
    wave["probability_R"] = 1 - (1 - wave["WAVE_R"]) ** config.WAVE_TRIALS_PER_GAME

    total_hit = wave["r_hit"] + wave["l_hit"]
    total_ab = wave["r_at_bat"] + wave["l_at_bat"]
    wave["WAVE_CI_Low"], wave["WAVE_CI_High"] = helpers.wilson_ci(total_hit, total_ab)
    wave["probability_CI_Low"] = 1 - (1 - wave["WAVE_CI_Low"]) ** config.WAVE_TRIALS_PER_GAME
    wave["probability_CI_High"] = 1 - (1 - wave["WAVE_CI_High"]) ** config.WAVE_TRIALS_PER_GAME

    return wave[
        [
            "key_mlbam", "WAVE", "WAVE_L", "WAVE_R", "l_at_bat", "r_at_bat",
            "probability", "probability_L", "probability_R",
            "WAVE_CI_Low", "WAVE_CI_High", "probability_CI_Low", "probability_CI_High",
        ]
    ]


def compute_whops(dt: pd.DataFrame) -> pd.DataFrame:
    """Weighted OPS and a runs-created proxy, split by throws and blended
    across windows. Computed for completeness but - as in the original
    script - not currently merged into any output table."""
    stat_fns = {
        "ob": lambda df: helpers.is_on_base(df["events"]),
        "sv": lambda df: helpers.total_bases(df["events"]),
        "ab": lambda df: helpers.is_official_at_bat(df["events"]),
    }

    def ops_rate(agg):
        return agg["ob"] / agg["n"] + agg["sv"] / agg["n"]

    def rc_rate(agg):
        obp = agg["ob"] / agg["n"]
        slg = agg["sv"] / agg["n"]
        return slg * obp * (agg["ab"] / agg["n"])

    rate_fns = {"ops": ops_rate, "rc": rc_rate}
    windows = config.WHOPS_WTB_WINDOWS

    r_blended, r_full = _blend_windows(dt, windows, "R", stat_fns, rate_fns)
    l_blended, l_full = _blend_windows(dt, windows, "L", stat_fns, rate_fns)

    whops = r_full.rename(columns={"n": "pa_rfull"})
    whops = whops.merge(r_blended.rename(columns={"ops": "whops_R", "rc": "rcw_R"}), on="batter", how="left")
    whops = whops.merge(l_full.rename(columns={"n": "pa_lfull"}), on="batter", how="left")
    whops = whops.merge(l_blended.rename(columns={"ops": "whops_L", "rc": "rcw_L"}), on="batter", how="left")
    whops = whops.fillna(0)

    whops["abt"] = whops["pa_lfull"] + whops["pa_rfull"]
    whops["abtl"] = whops["pa_lfull"] / whops["abt"]
    whops["abtr"] = whops["pa_rfull"] / whops["abt"]
    whops["whops"] = whops["whops_R"] * whops["abtr"] + whops["whops_L"] * whops["abtl"]
    whops["rcw"] = whops["rcw_R"] * whops["abtr"] + whops["rcw_L"] * whops["abtl"]
    whops["rope"] = whops["rcw"] + whops["whops"]
    whops = whops.rename(columns={"batter": "key_mlbam"})

    return whops[
        ["key_mlbam", "whops", "whops_R", "whops_L", "pa_lfull", "pa_rfull", "rcw_R", "rcw_L", "rcw", "rope"]
    ]


def compute_wtb(dt: pd.DataFrame) -> pd.DataFrame:
    """Recency-weighted slugging value, blended by throws, plus Expected_Bases."""
    stat_fns = {"sv": lambda df: helpers.total_bases(df["events"])}
    rate_fns = {"slg": lambda agg: agg["sv"] / agg["n"]}
    windows = config.WHOPS_WTB_WINDOWS

    r_blended, r_full = _blend_windows(dt, windows, "R", stat_fns, rate_fns)
    l_blended, l_full = _blend_windows(dt, windows, "L", stat_fns, rate_fns)

    wtb = r_full.rename(columns={"n": "pa_rfull"})
    wtb = wtb.merge(r_blended.rename(columns={"slg": "WTB_r"}), on="batter", how="left")
    wtb = wtb.merge(l_full.rename(columns={"n": "pa_lfull"}), on="batter", how="left")
    wtb = wtb.merge(l_blended.rename(columns={"slg": "WTB_l"}), on="batter", how="left")
    wtb = wtb.fillna(0)

    wtb["ab"] = wtb["pa_lfull"] + wtb["pa_rfull"]
    wtb["abr"] = wtb["pa_rfull"] / wtb["ab"]
    wtb["abl"] = wtb["pa_lfull"] / wtb["ab"]
    wtb["WTB"] = wtb["WTB_r"] * wtb["abr"] + wtb["WTB_l"] * wtb["abl"]
    wtb["Expected_Bases"] = wtb["WTB"] * config.WTB_TRIALS_PER_GAME
    wtb = wtb.rename(columns={"batter": "key_mlbam"})

    return wtb[["key_mlbam", "pa_lfull", "pa_rfull", "WTB_l", "WTB_r", "WTB", "Expected_Bases"]]


def compute_extended_dk_rates(dt: pd.DataFrame) -> pd.DataFrame:
    """Recency-windowed BB/HBP/RBI rates for DK Classic MLB scoring
    (dfs.py's compute_hitter_dk_points), reusing the exact same windowed
    blend machinery WAVE/WHOPS/WTB already use. Deliberately reuses
    config.WHOPS_WTB_WINDOWS (not a new dedicated window constant) - these
    draw from the identical per-PA event stream WTB already blends, at the
    same row granularity, so a separate window schedule would be an
    unjustified extra knob, not a real architectural need (contrast with
    pitcher_form.py's DFS_PITCHER_WINDOWS, which is deliberately separate
    because it blends START-level, not AT-BAT-level, data).

    `dt` must carry `bat_score`/`post_bat_score` (see
    pipeline.build_pitch_events) for helpers.estimate_rbi.

    Returns [key_mlbam, Expected_BB, Expected_HBP, Expected_RBI] - a
    per-game expected count for each, converted from the blended per-PA
    rate via config.WTB_TRIALS_PER_GAME (the same at-bat-per-game
    assumption Expected_Bases already uses)."""
    stat_fns = {
        "bb": lambda df: helpers.is_walk_for_dk_scoring(df["events"]),
        "hbp": lambda df: helpers.is_hit_by_pitch(df["events"]),
        "rbi": helpers.estimate_rbi,
    }
    rate_fns = {
        "bb_rate": lambda agg: agg["bb"] / agg["n"],
        "hbp_rate": lambda agg: agg["hbp"] / agg["n"],
        "rbi_rate": lambda agg: agg["rbi"] / agg["n"],
    }
    windows = config.WHOPS_WTB_WINDOWS

    r_blended, r_full = _blend_windows(dt, windows, "R", stat_fns, rate_fns)
    l_blended, l_full = _blend_windows(dt, windows, "L", stat_fns, rate_fns)

    r_blended = r_blended.rename(columns={"bb_rate": "BB_rate_R", "hbp_rate": "HBP_rate_R", "rbi_rate": "RBI_rate_R"})
    l_blended = l_blended.rename(columns={"bb_rate": "BB_rate_L", "hbp_rate": "HBP_rate_L", "rbi_rate": "RBI_rate_L"})

    result = r_full.rename(columns={"n": "pa_rfull"})
    result = result.merge(r_blended, on="batter", how="left")
    result = result.merge(l_full.rename(columns={"n": "pa_lfull"}), on="batter", how="left")
    result = result.merge(l_blended, on="batter", how="left")
    result = result.fillna(0)

    result["abt"] = result["pa_lfull"] + result["pa_rfull"]
    result["abtl"] = result["pa_lfull"] / result["abt"]
    result["abtr"] = result["pa_rfull"] / result["abt"]

    for stat in ("BB", "HBP", "RBI"):
        blended_rate = result[f"{stat}_rate_R"] * result["abtr"] + result[f"{stat}_rate_L"] * result["abtl"]
        result[f"Expected_{stat}"] = (blended_rate * config.WTB_TRIALS_PER_GAME).fillna(0)

    result = result.rename(columns={"batter": "key_mlbam"})
    return result[["key_mlbam", "Expected_BB", "Expected_HBP", "Expected_RBI"]]


def compute_quality_of_contact(dt: pd.DataFrame) -> pd.DataFrame:
    """Recency-windowed real batted-ball-quality rates - Exit_Velo (mean
    launch_speed), Barrel_Rate (share of batted balls that are real
    Statcast barrels, helpers.is_barrel), xBA (mean
    estimated_ba_using_speedangle), xwOBA (mean
    estimated_woba_using_speedangle) - each blended by throws-side and
    recency window, the exact same shape WAVE/WTB/compute_extended_dk_rates
    already use.

    Computed over BATTED BALLS ONLY (helpers.is_batted_ball filters `dt`
    down to real balls-in-play before windowing) - a strikeout/walk/HBP
    has none of these columns populated, and including them unfiltered
    would silently bias every rate toward 0 instead of measuring contact
    quality. This means `n` here is a batted-ball count, not a
    plate-appearance count, unlike every other function in this file - the
    shared `_blend_windows`/`_side_window_agg` machinery still applies
    completely unmodified, since it operates on whatever frame it's given.

    Deliberately reuses config.WHOPS_WTB_WINDOWS (not a new dedicated
    window constant) - same reasoning compute_extended_dk_rates's own
    docstring already gives: this draws from a subset of the identical
    per-PA event stream WTB blends, at the same row granularity (just
    narrowed to the balls-in-play rows), so a separate window schedule
    isn't a justified extra knob on a first pass - revisit if a real
    backtest suggests batted-ball sample size specifically needs different
    window weights.

    Returns [key_mlbam, Exit_Velo, Barrel_Rate, xBA, xwOBA]. A batter with
    zero batted balls in a window contributes 0 to that window (same
    fillna(0) precedent every other blended rate in this file uses), not a
    fabricated league-average value."""
    batted = dt[helpers.is_batted_ball(dt["type"])].copy()

    stat_fns = {
        "exit_velo_sum": lambda df: df["launch_speed"],
        "barrel": lambda df: helpers.is_barrel(df["launch_speed_angle"]),
        "xba_sum": lambda df: df["estimated_ba_using_speedangle"],
        "xwoba_sum": lambda df: df["estimated_woba_using_speedangle"],
    }
    rate_fns = {
        "exit_velo": lambda agg: agg["exit_velo_sum"] / agg["n"],
        "barrel_rate": lambda agg: agg["barrel"] / agg["n"],
        "xba": lambda agg: agg["xba_sum"] / agg["n"],
        "xwoba": lambda agg: agg["xwoba_sum"] / agg["n"],
    }
    windows = config.WHOPS_WTB_WINDOWS

    r_blended, r_full = _blend_windows(batted, windows, "R", stat_fns, rate_fns)
    l_blended, l_full = _blend_windows(batted, windows, "L", stat_fns, rate_fns)

    r_blended = r_blended.rename(
        columns={"exit_velo": "Exit_Velo_R", "barrel_rate": "Barrel_Rate_R", "xba": "xBA_R", "xwoba": "xwOBA_R"}
    )
    l_blended = l_blended.rename(
        columns={"exit_velo": "Exit_Velo_L", "barrel_rate": "Barrel_Rate_L", "xba": "xBA_L", "xwoba": "xwOBA_L"}
    )

    result = r_full.rename(columns={"n": "bb_rfull"})
    result = result.merge(r_blended, on="batter", how="left")
    result = result.merge(l_full.rename(columns={"n": "bb_lfull"}), on="batter", how="left")
    result = result.merge(l_blended, on="batter", how="left")
    result = result.fillna(0)

    result["bbt"] = result["bb_lfull"] + result["bb_rfull"]
    result["bbtl"] = result["bb_lfull"] / result["bbt"]
    result["bbtr"] = result["bb_rfull"] / result["bbt"]

    for stat in ("Exit_Velo", "Barrel_Rate", "xBA", "xwOBA"):
        result[stat] = result[f"{stat}_R"] * result["bbtr"] + result[f"{stat}_L"] * result["bbtl"]

    result = result.rename(columns={"batter": "key_mlbam"})
    return result[["key_mlbam", "Exit_Velo", "Barrel_Rate", "xBA", "xwOBA"]]


def compute_pitch_family_rates(dt: pd.DataFrame) -> pd.DataFrame:
    """Recency-windowed at-bat hit rate, split by the PA-ENDING pitch's
    family (fastball/breaking/offspeed - helpers.pitch_type_family) instead
    of by opposing-pitcher throwing hand - the same WAVE formula
    (hit_sum / n, config.WAVE_WINDOWS) computed a third way via
    _blend_windows/_side_window_agg's generalized `column` parameter
    (column="pitch_family" here instead of the default "p_throws").

    This is the batter half of the pitch-type-specific platoon matchup
    (matchup.py's _pitch_arsenal_multiplier is the other half, using the
    opposing starter's own arsenal mix - pitchers.compute_pitch_arsenal):
    a batter who reliably ends ABs on fastballs successfully is plausibly a
    real fastball hitter, independent of the handedness platoon WAVE_L/
    WAVE_R already capture. Known, deliberate simplification, same
    category as helpers.estimate_rbi's documented tradeoffs: this credits
    the PA's OUTCOME to whichever pitch happened to end it, not to every
    pitch actually seen during the PA (e.g. five sliders taken before a
    fastball is put in play credits the fastball alone) - a real proxy,
    not a full per-pitch swing-decision model (that's plate-discipline
    territory, a separate signal, not this one).

    Returns [key_mlbam, Fastball_WAVE, Breaking_WAVE, Offspeed_WAVE]. A
    batter who has never ended a PA on a given family contributes 0 to
    that family (same fillna(0) precedent every other blended rate in this
    file uses, not a fabricated league-average value)."""
    families = dt.assign(pitch_family=helpers.pitch_type_family(dt["pitch_type"]))
    families = families[families["pitch_family"].notna()]

    stat_fns = {"hit": lambda df: helpers.is_hit(df["events"])}
    rate_fns = {"rate": lambda agg: agg["hit"] / agg["n"]}
    windows = config.WAVE_WINDOWS

    blended = {}
    full = {}
    for family in ("fastball", "breaking", "offspeed"):
        family_blended, family_full = _blend_windows(
            families, windows, family, stat_fns, rate_fns, column="pitch_family"
        )
        blended[family] = family_blended.rename(columns={"rate": family})
        full[family] = family_full.rename(columns={"n": f"{family}_n"})

    result = full["fastball"]
    result = result.merge(blended["fastball"], on="batter", how="left")
    result = result.merge(full["breaking"], on="batter", how="outer")
    result = result.merge(blended["breaking"], on="batter", how="left")
    result = result.merge(full["offspeed"], on="batter", how="outer")
    result = result.merge(blended["offspeed"], on="batter", how="left")
    result = result.fillna(0)

    result = result.rename(
        columns={"fastball": "Fastball_WAVE", "breaking": "Breaking_WAVE", "offspeed": "Offspeed_WAVE"}
    )
    result = result.rename(columns={"batter": "key_mlbam"})
    return result[["key_mlbam", "Fastball_WAVE", "Breaking_WAVE", "Offspeed_WAVE"]]


def compute_home_road_split(dt: pd.DataFrame) -> pd.DataFrame:
    """Recency-windowed at-bat hit rate, split by whether the PA happened
    at HOME or on the ROAD instead of by opposing-pitcher throwing hand -
    the same WAVE formula (hit_sum / n, config.WAVE_WINDOWS) computed a
    fourth way via _blend_windows/_side_window_agg's generalized `column`
    parameter (column="home_road" here), same pattern
    compute_pitch_family_rates already established for pitch-type family.

    Wave-logic follow-up (2026-08-24): "for each of our features, they
    should be taken with wave logic (someone... might randomly struggle
    versus lefties or at home)" - a batter's home/road split needs the
    exact same small-sample-noise guard the platoon split already gets
    (a recency-blended rate across multiple windows), not a flat
    season-long home-vs-away average.

    `inning_topbot` ("Bot" = bottom of the inning = the HOME team is
    batting, "Top" = the AWAY team is batting) directly tells us, per PA,
    which side of the split that PA belongs to - no need to resolve the
    batter's own team identity or cross-reference home_team/away_team at
    all. A row with a missing/unrecognized inning_topbot value (an older/
    narrower synthetic fixture, or pipeline.build_pitch_events's own
    missing-column-degrades-to-null fallback) contributes to neither
    split - same "unclassifiable rows are excluded, not guessed" precedent
    compute_pitch_family_rates's own pitch_type handling establishes.

    Returns [key_mlbam, WAVE_Home, WAVE_Away]. A batter who has only ever
    been observed on one side of the split contributes a real 0 for the
    other - same fillna(0) precedent every other blended rate in this
    file uses, not a fabricated league-average value."""
    home_road = dt.assign(
        home_road=dt["inning_topbot"].map({"Bot": "home", "Top": "away"})
    )
    home_road = home_road[home_road["home_road"].notna()]

    stat_fns = {"hit": lambda df: helpers.is_hit(df["events"])}
    rate_fns = {"rate": lambda agg: agg["hit"] / agg["n"]}
    windows = config.WAVE_WINDOWS

    blended = {}
    full = {}
    for side, label in (("home", "Home"), ("away", "Away")):
        side_blended, side_full = _blend_windows(home_road, windows, side, stat_fns, rate_fns, column="home_road")
        blended[label] = side_blended.rename(columns={"rate": label})
        full[label] = side_full.rename(columns={"n": f"{label}_n"})

    result = full["Home"]
    result = result.merge(blended["Home"], on="batter", how="left")
    result = result.merge(full["Away"], on="batter", how="outer")
    result = result.merge(blended["Away"], on="batter", how="left")
    result = result.fillna(0)

    result = result.rename(columns={"Home": "WAVE_Home", "Away": "WAVE_Away"})
    result = result.rename(columns={"batter": "key_mlbam"})
    return result[["key_mlbam", "WAVE_Home", "WAVE_Away"]]


def compute_plate_discipline(all_pitches: pd.DataFrame) -> pd.DataFrame:
    """Recency-windowed plate discipline: Whiff_Rate (whiffs / swings -
    given the batter offers at a pitch, how often they miss entirely) and
    Chase_Rate (chases / real out-of-zone pitches seen - given a pitch is
    genuinely outside the strike zone, how often they swing at it anyway),
    blended across config.WAVE_WINDOWS (reused, not a new dedicated window
    scheme - same reasoning compute_quality_of_contact's own docstring
    gives for reusing an existing window set rather than inventing an
    unvalidated new one: this is a batter-intrinsic quality metric like
    WAVE, not matchup-dependent, so WAVE's own recency philosophy is the
    natural fit).

    `all_pitches` must be pipeline.build_all_pitch_events's output - EVERY
    real pitch the batter saw, not the PA-ending-pitch-only frame `dt`
    provides - swing/whiff/chase are fundamentally about every single
    pitch a batter saw, not just the one that happened to end the PA (the
    exact "that's plate-discipline territory, a separate signal" distinction
    compute_pitch_family_rates's own docstring already draws).

    No throws-side split (unlike WAVE/quality-of-contact/pitch-family
    rates) - mirrors pitchers.compute_pitch_arsenal's own "windowed rate
    from ALL pitches, single population" shape rather than hitters.py's
    platoon-split machinery, since a real per-pitch swing decision doesn't
    need a same/opposite-handed distinction to be a meaningful signal on
    its own (a real backtest could still surface a platoon split in
    swing decisions later, revisit if so).

    Returns [key_mlbam, Whiff_Rate, Chase_Rate]. A batter with zero real
    swings (or zero real out-of-zone pitches seen) in a window contributes
    0 to that rate - the same fillna(0) "no data reads as an honest 0,
    never a fabricated value" precedent every other blended rate in this
    project uses, even though a literal 0% miss/chase rate is an
    unavoidably ambiguous edge case for a genuinely tiny sample (same
    caveat PAVE-style ratios already carry when their own denominator is
    small)."""
    classified = all_pitches.assign(
        swing=helpers.is_swing(all_pitches["description"]),
        whiff=helpers.is_whiff(all_pitches["description"]),
        out_of_zone=helpers.is_out_of_zone(all_pitches["zone"]),
        chase=helpers.is_chase(all_pitches["description"], all_pitches["zone"]),
    )

    stats = ("swing", "whiff", "out_of_zone", "chase")
    latest = classified["game_date"].max()
    blended = {"Whiff_Rate": None, "Chase_Rate": None}

    for days_back, weight in config.WAVE_WINDOWS:
        window_df = _slice_by_days(classified, latest, days_back)
        agg = window_df.groupby("batter", as_index=False)[list(stats)].sum()
        base = agg[["batter"]]
        window_rates = {
            "Whiff_Rate": (agg["whiff"] / agg["swing"]).fillna(0),
            "Chase_Rate": (agg["chase"] / agg["out_of_zone"]).fillna(0),
        }
        for name, rate in window_rates.items():
            contribution = base.assign(value=rate.values).set_index("batter")["value"] * weight
            blended[name] = contribution if blended[name] is None else blended[name].add(contribution, fill_value=0)

    result = pd.DataFrame(blended)
    result.index.name = "batter"
    result = result.reset_index()
    result = result.rename(columns={"batter": "key_mlbam"})
    return result[["key_mlbam", "Whiff_Rate", "Chase_Rate"]]


def compute_game_hit_probability(data_with_game_id: pd.DataFrame, shrinkage_strength: float | None = None) -> pd.DataFrame:
    """Rate of *games* (not at-bats) with >=1 hit, blended across windows.
    This is the closest existing analog to a Beat-the-Streak pick probability.
    Returns key_mlbam, Game_Hit_Probability, Game_Hit_Probability_CI_Low,
    Game_Hit_Probability_CI_High.

    `shrinkage_strength` (real-unit games; None -> config.GAME_HIT_PROB_SHRINKAGE_STRENGTH)
    is the games-unit analog of compute_wave's own shrinkage param - see
    that function's docstring and quant-analytics item #3's "Bayesian
    shrinkage" README section. 0.0 (the live default) is the exact null
    hypothesis, bit-for-bit identical to this function's original
    had_hit/game division.

    `Game_Hit_Probability_CI_Low`/`_CI_High` (quant-analytics item #3,
    slice 3): a real Wilson-score confidence interval (helpers.wilson_ci)
    on the RAW full-season "games with a hit" rate - unlike
    Game_Hit_Probability itself (a blend of 4 overlapping, nested
    recency windows with no closed-form interval), the full-season
    component is a single, well-defined binomial proportion. Independent
    of `shrinkage_strength` - see compute_wave's own docstring for the
    same reasoning."""
    strength = config.GAME_HIT_PROB_SHRINKAGE_STRENGTH if shrinkage_strength is None else shrinkage_strength
    completed = data_with_game_id[data_with_game_id["events"].isin(config.COUNTED_EVENTS)][
        ["game_date", "batter", "events", "game_id"]
    ].copy()
    completed["had_hit"] = helpers.is_hit(completed["events"])

    game_hits = completed.groupby(["batter", "game_id", "game_date"], as_index=False)["had_hit"].max()
    game_hits["game"] = 1
    latest = game_hits["game_date"].max()
    league_rate = game_hits["had_hit"].sum() / len(game_hits) if len(game_hits) else 0.0

    blended = None
    full_counts = None
    for days_back, weight in config.GAME_HIT_PROB_WINDOWS:
        window_games = _slice_by_days(game_hits, latest, days_back)
        agg = window_games.groupby("batter", as_index=False)[["had_hit", "game"]].sum()
        rate = helpers.shrink_rate(agg["had_hit"], agg["game"], league_rate, strength).fillna(0)
        contribution = agg[["batter"]].assign(rate=rate.values).set_index("batter")["rate"] * weight
        blended = contribution if blended is None else blended.add(contribution, fill_value=0)
        if days_back is None:
            full_counts = agg[["batter", "had_hit", "game"]]

    result = blended.reset_index()
    result.columns = ["batter", "Game_Hit_Probability"]
    result = result.merge(full_counts, on="batter", how="left").fillna(0)
    result["Game_Hit_Probability_CI_Low"], result["Game_Hit_Probability_CI_High"] = helpers.wilson_ci(
        result["had_hit"], result["game"]
    )
    return result[
        ["batter", "Game_Hit_Probability", "Game_Hit_Probability_CI_Low", "Game_Hit_Probability_CI_High"]
    ].rename(columns={"batter": "key_mlbam"})


def compute_last_game_dates(data_with_game_id: pd.DataFrame) -> pd.DataFrame:
    """[key_mlbam, Last_Game_Date] - the most recent date each batter
    recorded a completed plate appearance. Unlike compute_current_hit_streaks's
    walk, finding the single most recent date doesn't need the game_id-safe
    chronological ordering (a doubleheader's two games share one game_date
    either way) - a plain per-batter max is correct.

    Feeds assemble_hitters's Last_Game_Date column, which
    predictions.select_picks gates official picks on (a hitter who hasn't
    played recently shouldn't be pick-eligible just because their
    season-long rates are still high)."""
    completed = data_with_game_id[data_with_game_id["events"].isin(config.COUNTED_EVENTS)][
        ["game_date", "batter"]
    ]
    result = completed.groupby("batter", as_index=False)["game_date"].max()
    return result.rename(columns={"batter": "key_mlbam", "game_date": "Last_Game_Date"})


def compute_current_hit_streaks(
    data_with_game_id: pd.DataFrame, recent_days: int = config.HIT_STREAK_RECENT_DAYS
) -> pd.DataFrame:
    """[key_mlbam, Current_Hit_Streak, last_game_date] - each recently-active
    batter's real current consecutive-games-with-a-hit streak, counted
    backward from their own most recent game. Same per-game hit indicator
    compute_game_hit_probability builds - grouped by (batter, game_id,
    game_date), NOT game_date alone (see data.assign_game_ids's docstring:
    date-only grouping fragments a doubleheader's two games and has already
    been a real bug here) - just walked in game_id order per batter instead
    of aggregated into a rate.

    A batter whose most recent game is more than `recent_days` before the
    latest game_date in the data is excluded entirely - a streak frozen by
    weeks of inactivity isn't a real "current" one. Batters within that
    window are still included even at Current_Hit_Streak=0 (their last game
    was a miss); it's the caller's job to decide what counts as "on a
    streak" for display purposes."""
    completed = data_with_game_id[data_with_game_id["events"].isin(config.COUNTED_EVENTS)][
        ["game_date", "batter", "events", "game_id"]
    ].copy()
    completed["had_hit"] = helpers.is_hit(completed["events"])

    game_hits = completed.groupby(["batter", "game_id", "game_date"], as_index=False)["had_hit"].max()
    if game_hits.empty:
        return pd.DataFrame(columns=["key_mlbam", "Current_Hit_Streak", "last_game_date"])

    latest = game_hits["game_date"].max()
    game_hits = game_hits.sort_values("game_id")

    rows = []
    for batter, group in game_hits.groupby("batter", sort=False):
        last_game_date = group["game_date"].iloc[-1]
        if (latest - last_game_date).days > recent_days:
            continue
        streak = 0
        for had_hit in group["had_hit"].iloc[::-1]:
            if not had_hit:
                break
            streak += 1
        rows.append({"key_mlbam": batter, "Current_Hit_Streak": streak, "last_game_date": last_game_date})

    return pd.DataFrame(rows, columns=["key_mlbam", "Current_Hit_Streak", "last_game_date"])


def assemble_hitters(
    dt: pd.DataFrame,
    data_with_game_id: pd.DataFrame,
    names: pd.DataFrame,
    latest_team: pd.DataFrame,
    lineup_consistency: pd.DataFrame | None = None,
    all_pitches: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the final hitter output table (equivalent to the original script's `WAVE` dataframe).

    `lineup_consistency` (see lineup.compute_lineup_consistency) is optional
    so existing callers/tests are unaffected; when given, it adds
    avg_batting_order/start_rate columns that predictions.select_picks's
    lineup qualifiers key off of. avg_batting_order is deliberately left
    null (not filled to 0) for a batter with no recorded starts - a null
    correctly fails the "top half of the order" qualifier downstream,
    whereas a 0 would look like the best possible batting slot.

    `all_pitches` (pipeline.build_all_pitch_events's output - EVERY real
    pitch, not `dt`'s PA-ending-only frame) is likewise optional so
    existing callers/tests are unaffected; when given, it adds
    Whiff_Rate/Chase_Rate (compute_plate_discipline) - the same optional-
    input shape pitchers.assemble_pitchers's own `pitch_arsenal` parameter
    uses, for the same reason (a per-PITCH signal genuinely needs a
    different input than every PA-level signal here).
    """
    wave = compute_wave(dt)
    wtb = compute_wtb(dt)
    extended_dk_rates = compute_extended_dk_rates(dt)
    quality_of_contact = compute_quality_of_contact(dt)
    pitch_family_rates = compute_pitch_family_rates(dt)
    home_road_split = compute_home_road_split(dt)
    game_hit_prob = compute_game_hit_probability(data_with_game_id)
    last_game = compute_last_game_dates(data_with_game_id)

    hitters = wave.merge(
        wtb[["key_mlbam", "pa_lfull", "pa_rfull", "Expected_Bases"]], on="key_mlbam", how="left"
    )
    hitters = hitters.merge(extended_dk_rates, on="key_mlbam", how="left")
    hitters = hitters.merge(quality_of_contact, on="key_mlbam", how="left")
    hitters = hitters.merge(pitch_family_rates, on="key_mlbam", how="left")
    hitters = hitters.merge(home_road_split, on="key_mlbam", how="left")
    hitters = hitters.merge(game_hit_prob, on="key_mlbam", how="left")
    hitters = hitters.merge(names, on="key_mlbam", how="left")
    hitters = hitters.merge(latest_team, on="key_mlbam", how="left")

    hitters["Consistency"] = hitters["Game_Hit_Probability"] - hitters["probability"]
    hitters["Approach"] = hitters["Game_Hit_Probability"] * hitters["probability"]

    columns = [
        "key_mlbam", "name_first", "name_last", "team",
        "pa_lfull", "pa_rfull",
        "WAVE", "WAVE_L", "WAVE_R", "probability_L", "probability_R", "probability",
        "WAVE_CI_Low", "WAVE_CI_High", "probability_CI_Low", "probability_CI_High",
        "Game_Hit_Probability", "Game_Hit_Probability_CI_Low", "Game_Hit_Probability_CI_High",
        "Consistency", "Approach", "Expected_Bases",
        "Expected_BB", "Expected_HBP", "Expected_RBI",
        "Exit_Velo", "Barrel_Rate", "xBA", "xwOBA",
        "Fastball_WAVE", "Breaking_WAVE", "Offspeed_WAVE",
        "WAVE_Home", "WAVE_Away",
    ]
    if all_pitches is not None:
        plate_discipline = compute_plate_discipline(all_pitches)
        hitters = hitters.merge(plate_discipline, on="key_mlbam", how="left")
        columns = columns + ["Whiff_Rate", "Chase_Rate"]

    hitters = hitters[columns].fillna(0)

    # Merged AFTER the fillna(0) block above (like lineup_consistency below)
    # rather than before - a missing/NaT Last_Game_Date must stay NaT, not
    # get coerced to 0 by that blanket fillna, since predictions.select_picks's
    # recency gate relies on NaT correctly failing the comparison (same
    # "null loses, isn't filled to a value that would wrongly pass"
    # precedent avg_batting_order already established).
    hitters = hitters.merge(last_game, on="key_mlbam", how="left")

    if lineup_consistency is not None:
        hitters = hitters.merge(lineup_consistency, on="key_mlbam", how="left")
        hitters["start_rate"] = hitters["start_rate"].fillna(0)

    hitters = hitters.sort_values("Game_Hit_Probability", ascending=False)
    return hitters.rename(columns={"pa_lfull": "PA_L", "pa_rfull": "PA_R"})

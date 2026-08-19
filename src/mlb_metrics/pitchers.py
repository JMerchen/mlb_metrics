"""Pitcher metrics: PAVE (batting-average-against, converting a per-plate-
appearance hit rate to a per-at-bat one by excluding walks/HBP from the
denominator) and PAVE_PLUS (normalized to the qualified-pitcher league
average), plus Bullpen_PAVE: the same PAVE formula pooled across a team's
relief appearances only.

**Real bug fixed here (found via a real production complaint - a hitter's
matchup against a dominant strikeout starter, Jacob Misiorowski, wasn't
being treated as tough at all)**: the original formula excluded
strikeouts from the AB denominator ALONGSIDE walks/HBP
(`helpers.is_strikeout_walk_hbp`), not just walks/HBP. A strikeout IS an
official at-bat - only walks and HBP aren't - so that formula shrank the
effective denominator for every strikeout a pitcher recorded, which
INFLATES the computed hit-rate for exactly the pitchers who rack up the
most strikeouts (the more good things a swing-and-miss ace does, the
worse this formula made him look). Confirmed against Misiorowski's real
2026 Statcast log (439 batters faced, 173 K, 28 BB, 8 HBP, 61 hits): the
old formula computed a full-season PAVE of 0.265 (PAVE_PLUS ~0.93 -
barely better than average); the corrected formula (excluding only
walks/HBP, `helpers.is_non_at_bat_event`) gives 61/(439-28-8) = 0.151 -
genuinely elite, matching his real Cy-Young-caliber season. This affected
every consumer of PAVE/PAVE_PLUS/Bullpen_PAVE project-wide (matchup.py's
Matchup_Hit_Probability, game_picks.py's susceptibility signal, dfs.py's
Expected_H_Allowed) - see each module's own backtest re-run after this
fix for the corrected numbers.

Bullpen_PAVE exists because a hitter's 3-5 at-bats in a game are not all
against the starter - typically only 2-3 are, with the rest against
whichever relievers come in. Evaluating a hitter's matchup against "today's
starter's PAVE" alone ignores that. Bullpen_PAVE is team-level (a bullpen is
a shared pool, not one pitcher), computed the same recency-weighted way as
PAVE, restricted to appearances where the pitcher was not that game's
starter (see data.label_pitcher_roles).

Power_A_PLUS (and its bullpen analog Bullpen_Power_A_PLUS) is a second,
complementary pitching-quality signal: PAVE treats every hit allowed
identically, so a pitcher who mostly allows singles and one who mostly
allows home runs can show the same PAVE despite very different
run-prevention profiles - closer to what ERA actually measures. Rather than
pulling in ERA (ties to sequencing/inherited runners, not just this
pitcher's own stuff, and isn't computed from this project's own Statcast
data), Power_A_PLUS normalizes the already-blended `power_a` (total bases
allowed per plate appearance, blended across the same config.PAVE_WINDOWS
recency windows as PAVE) the same way PAVE_PLUS normalizes PAVE - "who is
currently susceptible to allowing damage," not just hits.

`Throws` (compute_pitcher_throws) exposes each pitcher's own throwing hand -
used by matchup.py to select a batter's platoon-specific hit rate
(WAVE_L/WAVE_R, see hitters.py) against today's actual probable starter,
instead of a hand-blended overall rate that treats facing a lefty and a
righty as interchangeable.

compute_pitch_arsenal is a different kind of pitcher signal: not an
outcome rate at all, but a real windowed USAGE mix (what share of a
pitcher's actual pitches are fastball/breaking/offspeed - see
helpers.pitch_type_family). It's the pitcher half of matchup.py's
pitch-type-specific platoon adjustment - the batter half is
hitters.compute_pitch_family_rates. Unlike every other function here, it
needs EVERY pitch a pitcher threw, not just the PA-ending ones -
pipeline.build_all_pitch_events (not build_pitcher_events) is its input.
"""

import pandas as pd

from mlb_metrics import config, helpers


def _slice_by_days(frame: pd.DataFrame, latest, days_back):
    if days_back is None:
        return frame
    cutoff = latest - pd.Timedelta(days=days_back)
    return frame[frame["game_date"] >= cutoff]


_STAT_FNS = {
    "hit": helpers.is_hit,
    "non_ab": helpers.is_non_at_bat_event,
    "tba": helpers.total_bases,
    "hr": helpers.is_home_run,
}


def _pave_rate(agg: pd.DataFrame) -> pd.Series:
    """hits / AB, where AB = PA - walks - HBP (a strikeout stays IN the
    AB count - see module docstring's "real bug fixed here" for why this
    matters)."""
    baa = agg["hit"] / agg["n"]
    non_ab_rate = agg["non_ab"] / agg["n"]
    return baa / (1 - non_ab_rate)


_RATE_FNS = {
    "PAVE": _pave_rate,
    "baa": lambda agg: agg["hit"] / agg["n"],
    "power_a": lambda agg: agg["tba"] / agg["n"],
    "hr_per": lambda agg: agg["hr"] / agg["n"],
}


def _window_agg(pdf: pd.DataFrame, latest, days_back, group_col: str) -> pd.DataFrame:
    window_df = _slice_by_days(pdf, latest, days_back).copy()
    window_df["n"] = 1
    for name, fn in _STAT_FNS.items():
        window_df[name] = fn(window_df["events"])
    cols = [group_col, "n"] + list(_STAT_FNS.keys())
    return window_df[cols].groupby(group_col, as_index=False).sum()


def _blend_pave(pdf: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Returns [group_col, PAVE, baa, power_a, hr_per, at_bats, hits, TBA],
    blended across config.PAVE_WINDOWS and grouped by `group_col` (either
    "pitcher" for individual PAVE or "team" for Bullpen_PAVE)."""
    latest = pdf["game_date"].max()
    blended = {name: None for name in _RATE_FNS}
    full_counts = None
    full_stats = None

    for days_back, weight in config.PAVE_WINDOWS:
        agg = _window_agg(pdf, latest, days_back, group_col)
        base = agg[[group_col]]
        for name, rate_fn in _RATE_FNS.items():
            rate = rate_fn(agg).fillna(0)
            contribution = base.assign(rate=rate.values).set_index(group_col)["rate"] * weight
            blended[name] = (
                contribution if blended[name] is None else blended[name].add(contribution, fill_value=0)
            )
        if days_back is None:
            full_counts = agg[[group_col, "n"]].rename(columns={"n": "at_bats"})
            full_stats = agg[[group_col, "hit", "tba"]].rename(columns={"hit": "hits", "tba": "TBA"})

    result = pd.DataFrame(blended)
    result.index.name = group_col
    result = result.reset_index()
    result = result.merge(full_counts, on=group_col, how="left")
    return result.merge(full_stats, on=group_col, how="left")


def compute_pitch_arsenal(all_pitches: pd.DataFrame) -> pd.DataFrame:
    """Recency-windowed pitch-type-family usage mix: what real share of
    this pitcher's actual pitches are fastballs vs. breaking balls vs.
    offspeed (helpers.pitch_type_family), blended across
    config.PAVE_WINDOWS (the same recency weighting PAVE itself uses - no
    new, separately-validated window scheme introduced just for this).

    `all_pitches` must be pipeline.build_all_pitch_events's output - EVERY
    real pitch thrown, not the PA-ending-pitch-only frame every other
    function in this module uses (`pdf`/`_slice_by_days`'s usual input) -
    a usage mix computed from only PA-ending pitches would be badly
    distorted (a putaway pitch is disproportionately breaking/offspeed in
    the real game, not representative of the full mix a batter actually
    sees).

    This is the pitcher half of the pitch-type-specific platoon matchup
    (matchup.py's _pitch_arsenal_multiplier is the other half, using the
    batter's own Fastball_WAVE/Breaking_WAVE/Offspeed_WAVE -
    hitters.compute_pitch_family_rates).

    Returns [key_mlbam, Fastball_Rate, Breaking_Rate, Offspeed_Rate,
    pitches_thrown]. A pitch with no real, classifiable pitch_type
    (helpers.PITCH_TYPE_FAMILY) is already excluded by
    pipeline.build_all_pitch_events, so the three rates sum to 1.0 for
    every pitcher with at least one real classified pitch in a window; a
    pitcher absent from a window (no real pitches thrown in it at all)
    contributes 0 to every family there, same fillna(0) precedent every
    other blended rate in this project uses."""
    classified = all_pitches.assign(family=helpers.pitch_type_family(all_pitches["pitch_type"]))
    classified = classified[classified["family"].notna()]

    families = ("fastball", "breaking", "offspeed")
    latest = classified["game_date"].max()
    blended = {family: None for family in families}
    full_counts = None

    for days_back, weight in config.PAVE_WINDOWS:
        window_df = _slice_by_days(classified, latest, days_back).copy()
        window_df["n"] = 1
        for family in families:
            window_df[family] = (window_df["family"] == family).astype(int)
        agg = window_df[["pitcher", "n", *families]].groupby("pitcher", as_index=False).sum()
        base = agg[["pitcher"]]
        for family in families:
            rate = (agg[family] / agg["n"]).fillna(0)
            contribution = base.assign(rate=rate.values).set_index("pitcher")["rate"] * weight
            blended[family] = (
                contribution if blended[family] is None else blended[family].add(contribution, fill_value=0)
            )
        if days_back is None:
            full_counts = agg[["pitcher", "n"]].rename(columns={"n": "pitches_thrown"})

    result = pd.DataFrame(blended)
    result.index.name = "pitcher"
    result = result.reset_index()
    result = result.merge(full_counts, on="pitcher", how="left")
    result = result.rename(
        columns={
            "pitcher": "key_mlbam", "fastball": "Fastball_Rate",
            "breaking": "Breaking_Rate", "offspeed": "Offspeed_Rate",
        }
    )
    return result[["key_mlbam", "Fastball_Rate", "Breaking_Rate", "Offspeed_Rate", "pitches_thrown"]]


def compute_pave(pdf: pd.DataFrame) -> pd.DataFrame:
    """Returns key_mlbam, PAVE, baa, power_a, hr_per, at_bats, hits, TBA -
    the raw building blocks assemble_pitchers() turns into PAVE_PLUS and
    the Expected_* columns."""
    return _blend_pave(pdf, "pitcher").rename(columns={"pitcher": "key_mlbam"})


def compute_pitcher_throws(pdf: pd.DataFrame) -> pd.DataFrame:
    """One row per pitcher: [key_mlbam, Throws] ("L"/"R") - the pitcher's
    own throwing hand (Statcast's p_throws, constant for a given pitcher
    all season, unlike a batter's opponent-hand splits). Used by matchup.py
    to pick a batter's platoon-specific rate (WAVE_L/WAVE_R) against
    today's actual probable starter instead of a handedness-blended
    overall rate.

    Empty (with the right columns, not a crash) when `pdf` doesn't carry
    p_throws - older callers/test fixtures predating this column - so a
    merge onto it just leaves every Throws null rather than failing."""
    if "p_throws" not in pdf.columns:
        return pd.DataFrame(columns=["key_mlbam", "Throws"])
    throws = pdf[["pitcher", "p_throws"]].dropna().drop_duplicates(subset="pitcher", keep="last")
    return throws.rename(columns={"pitcher": "key_mlbam", "p_throws": "Throws"})


def compute_bullpen_pave(pdf_with_role: pd.DataFrame) -> pd.DataFrame:
    """Team-level PAVE computed only from relief appearances (`is_starter`
    False), so hitters can be evaluated against the bullpen they're likely
    to face in their later at-bats. `pdf_with_role` must carry `team`
    (the pitching team for that appearance) and `is_starter`, as produced by
    pipeline.build_pitcher_events_with_role().

    Bullpen_PAVE_PLUS is normalized to the mean across all teams (no
    at-bat-share qualifier is needed here, unlike individual PAVE_PLUS,
    since every team's bullpen accumulates innings continuously).
    """
    bullpen = pdf_with_role[~pdf_with_role["is_starter"]]
    result = _blend_pave(bullpen, "team")

    mean_bullpen_pave = result["PAVE"].mean()
    result["Bullpen_PAVE_PLUS"] = result["PAVE"] / mean_bullpen_pave
    mean_bullpen_power_a = result["power_a"].mean()
    result["Bullpen_Power_A_PLUS"] = result["power_a"] / mean_bullpen_power_a

    return result.rename(
        columns={
            "PAVE": "Bullpen_PAVE",
            "baa": "Bullpen_BAA",
            "power_a": "Bullpen_Power_A",
            "hr_per": "Bullpen_HR_Per",
            "at_bats": "Bullpen_AtBats",
            "hits": "Bullpen_Hits",
            "TBA": "Bullpen_TBA",
        }
    )[
        [
            "team", "Bullpen_AtBats", "Bullpen_PAVE", "Bullpen_PAVE_PLUS",
            "Bullpen_Power_A_PLUS", "Bullpen_BAA", "Bullpen_Power_A", "Bullpen_HR_Per",
        ]
    ]


def assemble_pitchers(
    pdf: pd.DataFrame,
    names: pd.DataFrame,
    latest_pitcher_team: pd.DataFrame,
    pitch_arsenal: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the final pitcher output table (equivalent to the original script's `pave` dataframe).

    `pdf` must carry a `p_throws` column (see pipeline.build_pitcher_events)
    for the `Throws` column below - a missing p_throws (old callers/test
    fixtures predating this) degrades to Throws=NaN for every pitcher via
    compute_pitcher_throws's dropna, not a crash.

    `pitch_arsenal` (compute_pitch_arsenal's output) is optional so
    existing callers/tests are unaffected; when given, it adds
    Fastball_Rate/Breaking_Rate/Offspeed_Rate - matchup.py's
    pitch-type-specific platoon adjustment reads these for today's
    probable starter. Deliberately left NULL (not filled to 0) for a
    pitcher with no real pitch_arsenal row - a real pitcher's three rates
    always sum to ~1.0, so a 0/0/0 fallback would misrepresent "we don't
    know this pitcher's mix" as "this pitcher throws nothing," the same
    "null loses, isn't coerced into a value that looks meaningful"
    precedent avg_batting_order/Last_Game_Date already establish
    elsewhere - matchup.py's own multiplier defaults a missing mix to
    neutral at the point of use instead."""
    pave = compute_pave(pdf)
    throws = compute_pitcher_throws(pdf)

    qualified_threshold = pave["at_bats"].max() * config.PAVE_QUALIFIED_AB_FRACTION
    qualified = pave["at_bats"] > qualified_threshold
    mean_qualified_pave = pave.loc[qualified, "PAVE"].mean()
    mean_qualified_power_a = pave.loc[qualified, "power_a"].mean()
    pave["PAVE_PLUS"] = (pave["PAVE"] / mean_qualified_pave).fillna(0)
    pave["Power_A_PLUS"] = (pave["power_a"] / mean_qualified_power_a).fillna(0)
    pave = pave[pave["PAVE_PLUS"] > 0]

    pave["Expected_Hits"] = pave["baa"] * config.PAVE_PA_PER_START
    pave["Expected_Bases"] = pave["power_a"] * config.PAVE_PA_PER_START
    pave["Expected_HRs"] = pave["hr_per"] * config.PAVE_PA_PER_START

    pave = pave.merge(names, on="key_mlbam", how="left")
    pave = pave.merge(latest_pitcher_team, on="key_mlbam", how="left")
    pave = pave.merge(throws, on="key_mlbam", how="left")

    columns = [
        "key_mlbam", "name_first", "name_last", "team", "at_bats", "Throws",
        "PAVE", "PAVE_PLUS", "Power_A_PLUS",
        "Expected_Hits", "Expected_Bases", "Expected_HRs",
    ]
    if pitch_arsenal is not None:
        pave = pave.merge(
            pitch_arsenal[["key_mlbam", "Fastball_Rate", "Breaking_Rate", "Offspeed_Rate"]],
            on="key_mlbam", how="left",
        )
        columns = columns + ["Fastball_Rate", "Breaking_Rate", "Offspeed_Rate"]

    pave = pave[columns]
    return pave.sort_values("PAVE_PLUS", ascending=False)

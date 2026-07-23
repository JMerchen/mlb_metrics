"""Traditional season stats (hitters: AVG/OBP/SLG/OPS; pitchers: K9/BB9/
HR9/FIP) - a season-only (not recency-windowed) line, unlike WAVE/PAVE/
WHOPS/WTB elsewhere in this project.

These exist specifically for age_curve.py: Lahman's historical player-
seasons carry no Statcast-derived signal at all (WAVE/PAVE can't be
computed for a 1975 season), so a current player has to be put on the
same, era-independent stat basis - a single full-season rate line - to be
comparable against history.

compute_traditional_batting_stats reuses the same event classifiers
hitters.py's compute_whops already uses internally for its own (unwired)
OPS-ish calculation, just aggregated once over the whole season instead of
blended across recency windows. OBP here ignores sacrifice flies (not
classified separately anywhere in this project - see helpers.py's
NON_AT_BAT_EVENTS) - a known, minor simplification, consistent with how
this project documents other first-pass simplifications rather than
silently ignoring them.

compute_traditional_pitching_stats deliberately excludes ERA (see
config.AGE_CURVE_PITCHER_METRICS's docstring) - K9/BB9/HR9/FIP instead,
the same defense-independent "own stuff" philosophy as pitchers.py's
Power_A_PLUS.
"""

import pandas as pd

from mlb_metrics import config, helpers


def compute_traditional_batting_stats(dt: pd.DataFrame, min_at_bats: int = 0) -> pd.DataFrame:
    """One row per batter: [key_mlbam, PA, AB, AVG, OBP, SLG, OPS], from
    `dt` (pipeline.build_pitch_events's completed-at-bat-events table - the
    same input hitters.compute_wave/compute_whops/compute_wtb take).
    `min_at_bats` drops batters below the threshold (a small-sample AB=1
    batter can otherwise show an AVG/OPS of 0 or 1.000+ on pure noise -
    same reasoning as config.BACKTEST_MIN_PLATE_APPEARANCES elsewhere)."""
    df = dt.copy()
    df["hit"] = helpers.is_hit(df["events"])
    df["bases"] = helpers.total_bases(df["events"])
    df["on_base"] = helpers.is_on_base(df["events"])
    df["ab"] = helpers.is_official_at_bat(df["events"])

    agg = df.groupby("batter", as_index=False).agg(
        PA=("events", "size"), AB=("ab", "sum"), H=("hit", "sum"), TB=("bases", "sum"), OB=("on_base", "sum"),
    )
    agg = agg[agg["AB"] >= min_at_bats]

    agg["AVG"] = (agg["H"] / agg["AB"]).fillna(0)
    agg["OBP"] = (agg["OB"] / agg["PA"]).fillna(0)
    agg["SLG"] = (agg["TB"] / agg["AB"]).fillna(0)
    agg["OPS"] = agg["OBP"] + agg["SLG"]

    return agg.rename(columns={"batter": "key_mlbam"})[["key_mlbam", "PA", "AB", "AVG", "OBP", "SLG", "OPS"]]


def compute_traditional_pitching_stats(pdf: pd.DataFrame, min_ip: float = 0) -> pd.DataFrame:
    """One row per pitcher: [key_mlbam, IP, K9, BB9, HR9, FIP], from `pdf`
    (pipeline.build_pitcher_events's completed-at-bat-events table, keyed
    by pitcher - the same input pitchers.compute_pave takes). `min_ip`
    drops pitchers below the threshold (a small-sample pitcher can
    otherwise show a K9/FIP of 0 or an extreme outlier on pure noise - same
    reasoning as compute_traditional_batting_stats's min_at_bats)."""
    df = pdf.copy()
    df["outs"] = helpers.outs_recorded(df["events"])
    df["so"] = helpers.is_strikeout(df["events"])
    df["bb"] = helpers.is_walk(df["events"])
    df["hbp"] = helpers.is_hit_by_pitch(df["events"])
    df["hr"] = helpers.is_home_run(df["events"])

    agg = df.groupby("pitcher", as_index=False).agg(
        outs=("outs", "sum"), SO=("so", "sum"), BB=("bb", "sum"), HBP=("hbp", "sum"), HR=("hr", "sum"),
    )
    agg["IP"] = agg["outs"] / 3
    agg = agg[agg["IP"] >= min_ip]

    ip_safe = agg["IP"].replace(0, pd.NA)
    agg["K9"] = (agg["SO"] * 9 / ip_safe).fillna(0)
    agg["BB9"] = (agg["BB"] * 9 / ip_safe).fillna(0)
    agg["HR9"] = (agg["HR"] * 9 / ip_safe).fillna(0)
    agg["FIP"] = (
        (13 * agg["HR"] + 3 * (agg["BB"] + agg["HBP"]) - 2 * agg["SO"]) / ip_safe + config.AGE_CURVE_FIP_CONSTANT
    ).fillna(0)

    return agg.rename(columns={"pitcher": "key_mlbam"})[["key_mlbam", "IP", "K9", "BB9", "HR9", "FIP"]]

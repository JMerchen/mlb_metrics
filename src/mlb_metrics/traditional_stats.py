"""Traditional season batting stats (AVG/OBP/SLG/OPS) - a season-only (not
recency-windowed) line, unlike WAVE/WHOPS/WTB elsewhere in this project.

These exist specifically for age_curve.py: Lahman's historical player-
seasons carry no Statcast-derived signal at all (WAVE/PAVE can't be
computed for a 1975 season), so a current player has to be put on the
same, era-independent stat basis - a single full-season rate line - to be
comparable against history. Reuses the same event classifiers hitters.py's
compute_whops already uses internally for its own (unwired) OPS-ish
calculation, just aggregated once over the whole season instead of blended
across recency windows.

OBP here ignores sacrifice flies (not classified separately anywhere in
this project - see helpers.py's NON_AT_BAT_EVENTS) - a known, minor
simplification, consistent with how this project documents other
first-pass simplifications rather than silently ignoring them.
"""

import pandas as pd

from mlb_metrics import helpers


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

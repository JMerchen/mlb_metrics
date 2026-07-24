"""Recency-windowed starter form: K9/BB9/HR9 and innings-per-start, blended
across config.DFS_PITCHER_WINDOWS - the same windowed-recency-blend shape
pitchers.compute_pave already uses for PAVE, applied to a different set of
underlying stats (strikeouts/walks/home-runs/outs instead of hits allowed)
because dfs.py needs them and nothing in this project computes them on a
live, recency-windowed basis (traditional_stats.compute_traditional_pitching_stats
is season-to-date, built for the Age Curves page, not this).

Restricted to starts only (pdf_with_role's is_starter == True rows, the
same input pitchers.compute_bullpen_pave takes the complement of) - IP per
appearance is only a meaningful "per start" number for a starter; a
reliever's appearance length varies by bullpen usage/matchup, not a
comparable "start."

Deliberately its own config.DFS_PITCHER_WINDOWS rather than reusing
config.PAVE_WINDOWS - see that constant's docstring for why (PAVE blends
at-bat-level data, this blends start-level data, so the same window
lengths would leave far too few real starts in the tightest window).
"""

import pandas as pd

from mlb_metrics import config, helpers


def _slice_by_days(frame: pd.DataFrame, latest, days_back):
    if days_back is None:
        return frame
    cutoff = latest - pd.Timedelta(days=days_back)
    return frame[frame["game_date"] >= cutoff]


def _window_agg(starts_df: pd.DataFrame, latest, days_back) -> pd.DataFrame:
    """One row per pitcher: outs/so/bb/hr summed, plus `starts` (nunique
    game_id) over the window."""
    window_df = _slice_by_days(starts_df, latest, days_back).copy()
    window_df["outs"] = helpers.outs_recorded(window_df["events"])
    window_df["so"] = helpers.is_strikeout(window_df["events"])
    window_df["bb"] = helpers.is_walk(window_df["events"])
    window_df["hr"] = helpers.is_home_run(window_df["events"])
    return window_df.groupby("pitcher", as_index=False).agg(
        outs=("outs", "sum"),
        so=("so", "sum"),
        bb=("bb", "sum"),
        hr=("hr", "sum"),
        starts=("game_id", "nunique"),
    )


def _rate(agg: pd.DataFrame, numerator_col: str) -> pd.Series:
    ip = agg["outs"] / 3
    return (agg[numerator_col] * 9 / ip).where(ip > 0, 0)


_RATE_FNS = {
    "K9": lambda agg: _rate(agg, "so"),
    "BB9": lambda agg: _rate(agg, "bb"),
    "HR9": lambda agg: _rate(agg, "hr"),
    "IP_per_start": lambda agg: (agg["outs"] / 3 / agg["starts"]).where(agg["starts"] > 0, 0),
}


def compute_pitcher_dfs_form(pdf_with_role: pd.DataFrame) -> pd.DataFrame:
    """Returns [key_mlbam, starts, IP, K9, BB9, HR9, IP_per_start], blended
    across config.DFS_PITCHER_WINDOWS. `pdf_with_role` must carry `game_id`
    and `is_starter` (pipeline.build_pitcher_events_with_role's output);
    only is_starter==True rows are used.

    `starts`/`IP` are the unweighted full-season (days_back=None) counts,
    used by dfs.compute_pitcher_dk_points's config.DFS_PITCHER_MIN_STARTS
    qualifier - a small-sample pitcher's K9/BB9/HR9 (especially HR9) can be
    pure noise off 1-2 starts, same reasoning as PAVE_QUALIFIED_AB_FRACTION
    elsewhere in this project."""
    starts_df = pdf_with_role[pdf_with_role["is_starter"]]
    latest = starts_df["game_date"].max()

    blended = {name: None for name in _RATE_FNS}
    full_counts = None

    for days_back, weight in config.DFS_PITCHER_WINDOWS:
        agg = _window_agg(starts_df, latest, days_back)
        base = agg[["pitcher"]]
        for name, rate_fn in _RATE_FNS.items():
            rate = rate_fn(agg).fillna(0)
            contribution = base.assign(rate=rate.values).set_index("pitcher")["rate"] * weight
            blended[name] = contribution if blended[name] is None else blended[name].add(contribution, fill_value=0)
        if days_back is None:
            full_counts = agg[["pitcher", "starts", "outs"]].copy()
            full_counts["IP"] = full_counts["outs"] / 3

    result = pd.DataFrame(blended)
    result.index.name = "pitcher"
    result = result.reset_index()
    result = result.merge(full_counts[["pitcher", "starts", "IP"]], on="pitcher", how="left")

    return result.rename(columns={"pitcher": "key_mlbam"})[
        ["key_mlbam", "starts", "IP", "K9", "BB9", "HR9", "IP_per_start"]
    ]

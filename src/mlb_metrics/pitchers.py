"""Pitcher metrics: PAVE (batting-average-against, adjusted for K/BB/HBP
rate) and PAVE_PLUS (normalized to the qualified-pitcher league average)."""

import pandas as pd

from mlb_metrics import config, helpers


def _slice_by_days(frame: pd.DataFrame, latest, days_back):
    if days_back is None:
        return frame
    cutoff = latest - pd.Timedelta(days=days_back)
    return frame[frame["game_date"] >= cutoff]


def _pitcher_window_agg(pdf: pd.DataFrame, latest, days_back, stat_fns: dict) -> pd.DataFrame:
    window_df = _slice_by_days(pdf, latest, days_back).copy()
    window_df["n"] = 1
    for name, fn in stat_fns.items():
        window_df[name] = fn(window_df["events"])
    cols = ["pitcher", "n"] + list(stat_fns.keys())
    return window_df[cols].groupby("pitcher", as_index=False).sum()


def compute_pave(pdf: pd.DataFrame) -> pd.DataFrame:
    """Returns key_mlbam, PAVE, baa, power_a, hr_per, at_bats, hits, TBA -
    the raw building blocks assemble_pitchers() turns into PAVE_PLUS and
    the Expected_* columns."""
    stat_fns = {
        "hit": helpers.is_hit,
        "stk": helpers.is_strikeout_walk_hbp,
        "tba": helpers.total_bases,
        "hr": helpers.is_home_run,
    }

    def pave_rate(agg):
        baa = agg["hit"] / agg["n"]
        stk_rate = agg["stk"] / agg["n"]
        return baa / (1 - stk_rate)

    rate_fns = {
        "PAVE": pave_rate,
        "baa": lambda agg: agg["hit"] / agg["n"],
        "power_a": lambda agg: agg["tba"] / agg["n"],
        "hr_per": lambda agg: agg["hr"] / agg["n"],
    }

    latest = pdf["game_date"].max()
    blended = {name: None for name in rate_fns}
    full_counts = None
    full_stats = None

    for days_back, weight in config.PAVE_WINDOWS:
        agg = _pitcher_window_agg(pdf, latest, days_back, stat_fns)
        base = agg[["pitcher"]]
        for name, rate_fn in rate_fns.items():
            rate = rate_fn(agg).fillna(0)
            contribution = base.assign(rate=rate.values).set_index("pitcher")["rate"] * weight
            blended[name] = (
                contribution if blended[name] is None else blended[name].add(contribution, fill_value=0)
            )
        if days_back is None:
            full_counts = agg[["pitcher", "n"]].rename(columns={"n": "at_bats"})
            full_stats = agg[["pitcher", "hit", "tba"]].rename(columns={"hit": "hits", "tba": "TBA"})

    pave = pd.DataFrame(blended)
    pave.index.name = "pitcher"
    pave = pave.reset_index()
    pave = pave.merge(full_counts, on="pitcher", how="left")
    pave = pave.merge(full_stats, on="pitcher", how="left")
    return pave.rename(columns={"pitcher": "key_mlbam"})


def assemble_pitchers(pdf: pd.DataFrame, names: pd.DataFrame, latest_pitcher_team: pd.DataFrame) -> pd.DataFrame:
    """Build the final pitcher output table (equivalent to the original script's `pave` dataframe)."""
    pave = compute_pave(pdf)

    qualified_threshold = pave["at_bats"].max() * config.PAVE_QUALIFIED_AB_FRACTION
    mean_qualified_pave = pave.loc[pave["at_bats"] > qualified_threshold, "PAVE"].mean()
    pave["PAVE_PLUS"] = (pave["PAVE"] / mean_qualified_pave).fillna(0)
    pave = pave[pave["PAVE_PLUS"] > 0]

    pave["Expected_Hits"] = pave["baa"] * config.PAVE_PA_PER_START
    pave["Expected_Bases"] = pave["power_a"] * config.PAVE_PA_PER_START
    pave["Expected_HRs"] = pave["hr_per"] * config.PAVE_PA_PER_START

    pave = pave.merge(names, on="key_mlbam", how="left")
    pave = pave.merge(latest_pitcher_team, on="key_mlbam", how="left")

    pave = pave[
        ["key_mlbam", "name_first", "name_last", "team", "at_bats", "PAVE_PLUS", "Expected_Hits", "Expected_Bases", "Expected_HRs"]
    ]
    return pave.sort_values("PAVE_PLUS", ascending=False)

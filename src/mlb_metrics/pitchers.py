"""Pitcher metrics: PAVE (batting-average-against, adjusted for K/BB/HBP
rate) and PAVE_PLUS (normalized to the qualified-pitcher league average),
plus Bullpen_PAVE: the same PAVE formula pooled across a team's relief
appearances only.

Bullpen_PAVE exists because a hitter's 3-5 at-bats in a game are not all
against the starter - typically only 2-3 are, with the rest against
whichever relievers come in. Evaluating a hitter's matchup against "today's
starter's PAVE" alone ignores that. Bullpen_PAVE is team-level (a bullpen is
a shared pool, not one pitcher), computed the same recency-weighted way as
PAVE, restricted to appearances where the pitcher was not that game's
starter (see data.label_pitcher_roles).
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
    "stk": helpers.is_strikeout_walk_hbp,
    "tba": helpers.total_bases,
    "hr": helpers.is_home_run,
}


def _pave_rate(agg: pd.DataFrame) -> pd.Series:
    baa = agg["hit"] / agg["n"]
    stk_rate = agg["stk"] / agg["n"]
    return baa / (1 - stk_rate)


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


def compute_pave(pdf: pd.DataFrame) -> pd.DataFrame:
    """Returns key_mlbam, PAVE, baa, power_a, hr_per, at_bats, hits, TBA -
    the raw building blocks assemble_pitchers() turns into PAVE_PLUS and
    the Expected_* columns."""
    return _blend_pave(pdf, "pitcher").rename(columns={"pitcher": "key_mlbam"})


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
            "Bullpen_BAA", "Bullpen_Power_A", "Bullpen_HR_Per",
        ]
    ]


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
        ["key_mlbam", "name_first", "name_last", "team", "at_bats", "PAVE", "PAVE_PLUS", "Expected_Hits", "Expected_Bases", "Expected_HRs"]
    ]
    return pave.sort_values("PAVE_PLUS", ascending=False)

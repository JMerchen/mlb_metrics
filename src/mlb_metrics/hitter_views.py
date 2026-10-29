"""Different views of a hitter's season, for the Stats page's hitter table.

WAVE (hitters.compute_wave) is his at-bat hit rate and CAVE ("Carried
Average" - hitters.compute_game_hit_probability's Game_Hit_Probability)
the share of his games with a hit, both blended across recency windows;
here both carry a hitter's longer window into a shorter one he has no
at-bats in (see season_wave_cave). This module adds:

  WOPS  the same window blend for OPS (OBP per plate appearance plus SLG
        per at-bat), over config.WAVE_WINDOWS with each window's rates
        shrunk toward the league by config.WAVE_SHRINKAGE_STRENGTH, as
        WAVE is.

  "vs. pitchers faced" versions of WAVE, CAVE and WOPS, which credit hits
  off tough pitchers more and soft ones less. For each window, every at-bat
  (or plate appearance) he took is matched with what that pitcher allowed
  all season (shrunk toward the league by config.PITCHERS_FACED_SHRINK);
  the factor league rate / rate his pitchers allowed says how much harder
  or easier his pitching was than average, and the window factors blend
  with the stat's own window weights, then rescale so the average hitter's
  factor is 1. The adjusted stat is the stat times that factor - 1.05 means
  his pitchers were 5% stingier than average, so his numbers count 5% more.

For CAVE the per-game comparison is the chance a league-average hitter
gets a hit in that game's at-bats against those pitchers,
1 - prod(1 - pitcher hit rate), against the same at-bats at the league
rate.
"""

import numpy as np
import pandas as pd

from . import config

STATCAST_COLUMNS = ["game_pk", "game_date", "game_type", "batter", "pitcher", "p_throws", "events", "at_bat_number", "pitch_number"]

HIT_EVENTS = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
ON_BASE_EVENTS = {"single", "double", "triple", "home_run", "walk", "intent_walk", "hit_by_pitch"}
NOT_AT_BAT = {"walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt", "sac_fly_double_play",
              "sac_bunt_double_play", "catcher_interf", "truncated_pa"}


def plate_appearances(statcast: pd.DataFrame) -> pd.DataFrame:
    """Completed regular-season plate appearances with H, TB, OB, AB."""
    df = statcast[statcast["events"].notna() & (statcast["events"] != "truncated_pa")]
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]
    df = df.copy()
    df["H"] = df["events"].isin(HIT_EVENTS).astype(int)
    df["TB"] = df["events"].map(HIT_EVENTS).fillna(0).astype(int)
    df["OB"] = df["events"].isin(ON_BASE_EVENTS).astype(int)
    df["AB"] = (~df["events"].isin(NOT_AT_BAT)).astype(int)
    df["PA"] = 1
    return df


def league_rates(pas: pd.DataFrame) -> dict:
    ab, pa = max(1, pas["AB"].sum()), max(1, pas["PA"].sum())
    return {"h": pas["H"].sum() / ab, "ob": pas["OB"].sum() / pa, "tb": pas["TB"].sum() / ab}


def pitcher_allowed(pas: pd.DataFrame, league: dict) -> pd.DataFrame:
    """What each pitcher allowed all season, shrunk toward the league:
    hits per at-bat (h_allowed), on-base per PA (ob_allowed), total bases
    per at-bat (tb_allowed)."""
    k = config.PITCHERS_FACED_SHRINK
    sums = pas.groupby("pitcher")[["H", "TB", "OB", "AB", "PA"]].sum()
    return pd.DataFrame({
        "h_allowed": (sums["H"] + league["h"] * k) / (sums["AB"] + k),
        "ob_allowed": (sums["OB"] + league["ob"] * k) / (sums["PA"] + k),
        "tb_allowed": (sums["TB"] + league["tb"] * k) / (sums["AB"] + k),
    })


def _window(frame: pd.DataFrame, latest, days_back):
    """hitters.slice_by_days's window: everything, or the last
    `days_back` days through `latest`."""
    if days_back is None:
        return frame
    return frame[frame["game_date"] >= latest - pd.Timedelta(days=days_back)]


def _shrunk(count, n, prior, strength):
    return (count + prior * strength) / (n + strength)


def _longest_first(windows):
    return sorted(windows, key=lambda w: float("-inf") if w[0] is None else -w[0])


def _carry(rate: pd.Series, previous: pd.Series | None) -> pd.Series:
    """A hitter missing from this window keeps his next-longer window's value."""
    if previous is None:
        return rate
    return rate.reindex(previous.index.union(rate.index)).fillna(previous)


def _blend_carried(per_window: list, windows) -> pd.Series:
    """Weighted blend of per-window Series (longest window first), each
    window carrying the previous one forward for hitters missing from it."""
    total, previous = None, None
    for rate, (_, weight) in zip(per_window, windows):
        rate = _carry(rate, previous)
        previous = rate
        part = rate * weight
        total = part if total is None else total.add(part, fill_value=0)
    return total / sum(weight for _, weight in windows)


def compute_wops(pas: pd.DataFrame, league: dict | None = None) -> pd.DataFrame:
    """[key_mlbam, WOPS, WOPS_OBP, WOPS_SLG]: OBP and SLG blended over
    config.WAVE_WINDOWS (each window shrunk toward the league, a window he
    has no plate appearances in carrying his longer one), summed."""
    league = league or league_rates(pas)
    strength = config.WAVE_SHRINKAGE_STRENGTH
    latest = pas["game_date"].max()
    windows = _longest_first(config.WAVE_WINDOWS)
    obp, slg = [], []
    for days_back, _ in windows:
        sums = _window(pas, latest, days_back).groupby("batter")[["OB", "PA", "TB", "AB"]].sum()
        obp.append(_shrunk(sums["OB"], sums["PA"], league["ob"], strength))
        slg.append(_shrunk(sums["TB"], sums["AB"], league["tb"], strength))
    out = pd.DataFrame({"WOPS_OBP": _blend_carried(obp, windows), "WOPS_SLG": _blend_carried(slg, windows)})
    out["WOPS"] = out["WOPS_OBP"] + out["WOPS_SLG"]
    return out.rename_axis("key_mlbam").reset_index()


def pitchers_faced_factors(pas: pd.DataFrame, league: dict | None = None) -> pd.DataFrame:
    """[key_mlbam, wave_factor, cave_factor, obp_factor, slg_factor]: how
    much tougher than average the pitching each hitter faced was, blended
    over the same windows as the stat each factor adjusts."""
    league = league or league_rates(pas)
    faced = pas.join(pitcher_allowed(pas, league), on="pitcher")
    latest = faced["game_date"].max()

    ab = faced[faced["AB"] == 1].copy()
    ab["log_miss"] = np.log1p(-ab["h_allowed"].clip(upper=0.999))
    games = ab.groupby(["batter", "game_pk", "game_date"]).agg(n_ab=("AB", "sum"), log_miss=("log_miss", "sum")).reset_index()
    games["expected"] = 1 - np.exp(games["log_miss"])                         # vs. the pitchers he faced
    games["baseline"] = 1 - (1 - league["h"]) ** games["n_ab"]                  # same at-bats, league pitching

    wave_windows = _longest_first(config.WAVE_WINDOWS)
    cave_windows = _longest_first(config.GAME_HIT_PROB_WINDOWS)
    wave_f, ob_f, tb_f, cave_f = [], [], [], []
    for days_back, _ in wave_windows:
        w = _window(faced, latest, days_back)
        w_ab = w[w["AB"] == 1].groupby("batter")
        wave_f.append(league["h"] / w_ab["h_allowed"].mean())
        tb_f.append(league["tb"] / w_ab["tb_allowed"].mean())
        ob_f.append(league["ob"] / w.groupby("batter")["ob_allowed"].mean())
    for days_back, _ in cave_windows:
        g = _window(games, latest, days_back).groupby("batter")[["baseline", "expected"]].sum()
        cave_f.append(g["baseline"] / g["expected"])

    out = pd.DataFrame({
        "wave_factor": _blend_carried(wave_f, wave_windows),
        "obp_factor": _blend_carried(ob_f, wave_windows),
        "slg_factor": _blend_carried(tb_f, wave_windows),
        "cave_factor": _blend_carried(cave_f, cave_windows),
    })
    # Averaging rates and then dividing leans every factor slightly above 1
    # (Jensen's inequality); rescale so the at-bat-weighted average hitter's
    # factor is exactly 1 and the adjusted views stay centered on the plain ones.
    weights = pas.groupby("batter")["AB"].sum().reindex(out.index).fillna(0)
    for column in out.columns:
        valid = out[column].notna() & (weights > 0)
        out[column] = out[column] / np.average(out.loc[valid, column], weights=weights[valid])
    return out.rename_axis("key_mlbam").reset_index()


def season_wave_cave(statcast: pd.DataFrame) -> pd.DataFrame:
    """[key_mlbam, WAVE, CAVE]: hitters.compute_wave and
    hitters.compute_game_hit_probability with carry_forward, so a window a
    hitter has no at-bats in (hurt, benched, or no lefties lately) takes his
    next-longer window instead of counting as zero. That makes them views of
    his season; the pick models still read wave.csv's original versions.

    WAVE here counts at-bats only, so it reads on the batting-average scale
    the site describes it on. wave.csv's (the picks') counts walks and
    hit-by-pitches as trials too - hits per plate appearance, about 10%
    lower (2026's league: .220 against a .244 batting average) - and its
    pick settings were tuned on that."""
    from . import data, helpers, hitters

    regular = statcast[statcast["game_type"] == "R"] if "game_type" in statcast.columns else statcast
    dt = data.completed_events(regular, ["game_date", "batter", "p_throws", "events"])
    dt = dt[~dt["events"].isin(helpers.NON_AT_BAT_EVENTS)]
    wave = hitters.compute_wave(dt, carry_forward=True)[["key_mlbam", "WAVE"]]
    cave = hitters.compute_game_hit_probability(data.assign_game_ids(regular), carry_forward=True)
    cave = cave[["key_mlbam", "Game_Hit_Probability"]].rename(columns={"Game_Hit_Probability": "CAVE"})
    return wave.merge(cave, on="key_mlbam", how="outer")


def hitter_views(statcast: pd.DataFrame) -> pd.DataFrame:
    """One row per hitter: WAVE, CAVE, WOPS and their "vs. pitchers faced"
    versions (WAVE_adj, CAVE_adj, WOPS_adj), plus the factors behind them."""
    pas = plate_appearances(statcast)
    league = league_rates(pas)
    out = season_wave_cave(statcast)
    out = out.merge(compute_wops(pas, league), on="key_mlbam", how="left")
    out = out.merge(pitchers_faced_factors(pas, league), on="key_mlbam", how="left")
    for column in ("wave_factor", "cave_factor", "obp_factor", "slg_factor"):
        out[column] = out[column].fillna(1.0)
    out["WAVE_adj"] = out["WAVE"] * out["wave_factor"]
    out["CAVE_adj"] = (out["CAVE"] * out["cave_factor"]).clip(upper=1.0)
    out["WOPS_adj"] = out["WOPS_OBP"] * out["obp_factor"] + out["WOPS_SLG"] * out["slg_factor"]
    return out[["key_mlbam", "WAVE", "WAVE_adj", "CAVE", "CAVE_adj", "WOPS", "WOPS_adj",
                "wave_factor", "cave_factor", "obp_factor", "slg_factor"]]

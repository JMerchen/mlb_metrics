"""Do individual NFL players play differently in bad weather, beyond the
league-wide weather effect - and is that difference real or luck?

User question (2026-11-04): "Some players play worse in the cold while
others seem unaffected by it. Some players tend to fumble more when the
ball gets wet."

For each kind of player, outcome and condition, every player's own
weather split is
    d = (his rate in the condition) - (his rate otherwise)
and from the spread of d across players, the spread that sampling alone
would give (a binomial or per-play variance over his plays) is taken out
(DerSimonian-Laird random effects): tau is the SD of TRUE player-to-player
differences in the weather effect; Q's p-value asks whether there is any
at all. Then a split-half check: d from even seasons against d from odd
seasons for the same players - a real trait repeats.

Conditions (outdoor or open-roof games): cold 35°F or below, wind 15 mph
or more, rain from the play-by-play's weather text. "Otherwise" is every
other play, indoors included.
Outcomes:
    QB          EPA per dropback (passer, or the scrambler)
    fumbles     fumbles per touch (carries and catches), the player's own
    catches     catch rate per target (the receiver)
    kickers     field goals made above the league's rate at that distance

RESULT (2026-11-04, 1999-2026: 527k dropbacks, 650k touches, 429k
targets, 27k field goals): no player carries his own weather effect.
    fumbles per touch   rain, cold, wind: no spread beyond luck (p 1.00);
                        even vs odd seasons r -0.20 / +0.03 / -0.05
    QB EPA per dropback cold and rain show a spread (p 0.01) but it doesn't
                        repeat: even vs odd seasons r -0.15 (cold), +0.04
                        (rain), -0.04 (wind)
    catch rate          spread in all three (p < 0.001), none repeats
                        (r +0.08, -0.01, +0.00)
    kickers             nothing (rain p 0.055 over 61 kickers, r -0.05)
The spreads that don't repeat are which seasons and teams a player's
bad-weather games fell in, not the player. As groups, QBs of dome teams
did no worse outdoors in the cold (+0.04 EPA per dropback better than
outdoor teams' QBs, +/- 0.06), rain (+0.03 +/- 0.05) or wind (+0.07
+/- 0.05). So the simulation and the points model keep one league-wide
weather effect (nfl_weather.py, SIM_WEATHER in docs/nfl_sim.js).

Usage:
    python scripts/backtest_nfl_player_weather.py [first_season]
    NFL_OLD_PBP_DIR=<dir> python scripts/backtest_nfl_player_weather.py 1999
"""

import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import nfl_weather  # noqa: E402

RAW_DIR = "data/raw/nfl"
# Seasons before 2016 aren't kept in the repo: nflverse's play_by_play_<year>.parquet
# files, downloaded to this directory if set.
OLD_DIR = os.environ.get("NFL_OLD_PBP_DIR")
COLUMNS = ["game_id", "season", "season_type", "roof", "temp", "wind", "weather", "home_team", "posteam",
           "play_type", "pass", "rush", "sack", "qb_dropback", "qb_epa", "complete_pass", "qb_kneel", "qb_spike",
           "passer_player_id", "passer_player_name", "rusher_player_id", "rusher_player_name",
           "receiver_player_id", "receiver_player_name", "fumbled_1_player_id",
           "field_goal_attempt", "field_goal_result", "kick_distance", "kicker_player_id", "kicker_player_name"]
CONDITIONS = {"cold": "temp <= 35", "wind": "wind >= 15", "rain": "rain"}


def _path(season: int) -> str | None:
    for path in [os.path.join(RAW_DIR, f"pbp_{season}.parquet"),
                 os.path.join(OLD_DIR or "", f"play_by_play_{season}.parquet")]:
        if os.path.exists(path):
            return path
    return None


def load(first: int) -> pd.DataFrame:
    seasons = [s for s in range(first, 2027) if _path(s)]
    frames = []
    for season in seasons:
        pbp = pd.read_parquet(_path(season), columns=COLUMNS)
        pbp = pbp[pbp["season_type"] == "REG"]
        # Weather fields repeat on every play of a game; take each game's.
        game = pbp.groupby("game_id").agg(roof=("roof", "first"), temp=("temp", "first"),
                                          wind=("wind", "first"), weather=("weather", "first"))
        outdoor = game["roof"].isin(["outdoors", "open"])
        kind = game["weather"].map(nfl_weather.precip_from_text)
        flags = pd.DataFrame({
            "cold": outdoor & (game["temp"] <= 35),
            "wind": outdoor & (game["wind"] >= 15),
            "rain": outdoor & (kind == "rain"),
        })
        frames.append(pbp.drop(columns=["roof", "temp", "wind", "weather"]).join(flags, on="game_id"))
    print(f"seasons {seasons[0]}-{seasons[-1]}")
    return pd.concat(frames, ignore_index=True)


def outcomes(plays: pd.DataFrame) -> dict:
    """{name: DataFrame(player, name, season, y, cold, wind, rain)}."""
    keep = ["season", "cold", "wind", "rain"]
    dropbacks = plays[(plays["qb_dropback"] == 1) & plays["qb_epa"].notna() & (plays["qb_spike"] != 1)]
    qb = dropbacks.assign(player=dropbacks["passer_player_id"].fillna(dropbacks["rusher_player_id"]),
                          name=dropbacks["passer_player_name"].fillna(dropbacks["rusher_player_name"]),
                          y=dropbacks["qb_epa"])
    runs = plays[(plays["rush"] == 1) & (plays["qb_kneel"] != 1) & plays["rusher_player_id"].notna()]
    catches = plays[(plays["complete_pass"] == 1) & plays["receiver_player_id"].notna()]
    touches = pd.concat([
        runs.assign(player=runs["rusher_player_id"], name=runs["rusher_player_name"]),
        catches.assign(player=catches["receiver_player_id"], name=catches["receiver_player_name"]),
    ])
    touches["y"] = (touches["fumbled_1_player_id"] == touches["player"]).astype(float)
    targets = plays[(plays["pass"] == 1) & (plays["sack"] != 1) & plays["receiver_player_id"].notna()]
    targets = targets.assign(player=targets["receiver_player_id"], name=targets["receiver_player_name"],
                             y=targets["complete_pass"].astype(float))
    kicks = plays[(plays["field_goal_attempt"] == 1) & plays["kick_distance"].notna()].copy()
    kicks["made"] = (kicks["field_goal_result"] == "made").astype(float)
    # The league's make rate at each distance (a logistic fit), so a kicker
    # is graded against the kicks he was given.
    import statsmodels.api as sm
    fit = sm.Logit(kicks["made"], sm.add_constant(kicks[["kick_distance"]])).fit(disp=0)
    kicks = kicks.assign(player=kicks["kicker_player_id"], name=kicks["kicker_player_name"],
                         y=kicks["made"] - fit.predict(sm.add_constant(kicks[["kick_distance"]])))
    return {name: df[["player", "name", "y", *keep]].dropna(subset=["player", "y"])
            for name, df in [("QB EPA/dropback", qb), ("fumbles/touch", touches),
                             ("catch rate", targets), ("kickers FG over expected", kicks)]}


def splits(df: pd.DataFrame, condition: str, min_plays: int) -> pd.DataFrame:
    """Per player: n and mean in the condition and otherwise, d and its
    sampling variance."""
    g = df.groupby(["player", df[condition].rename("in")])["y"].agg(["size", "mean", "var"]).unstack("in")
    g = g.dropna(subset=[("size", True), ("size", False)])
    g = g[(g[("size", True)] >= min_plays) & (g[("size", False)] >= 3 * min_plays)]
    pooled_var = df["y"].var()  # per-play noise; for 0/1 outcomes close to p(1-p)
    out = pd.DataFrame({
        "n_in": g[("size", True)], "n_out": g[("size", False)],
        "mean_in": g[("mean", True)], "mean_out": g[("mean", False)],
    })
    out["d"] = out["mean_in"] - out["mean_out"]
    out["se2"] = pooled_var / out["n_in"] + pooled_var / out["n_out"]
    return out


def random_effects(s: pd.DataFrame) -> dict:
    w = 1 / s["se2"]
    mean = (w * s["d"]).sum() / w.sum()
    q = (w * (s["d"] - mean) ** 2).sum()
    k = len(s)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    typical_se = float(np.sqrt(s["se2"].median()))
    return {"players": k, "league_d": mean, "tau": np.sqrt(tau2), "typical_se": typical_se,
            "p": 1 - stats.chi2.cdf(q, k - 1),
            "reliability": tau2 / (tau2 + typical_se ** 2) if tau2 > 0 else 0.0}


def split_half(df: pd.DataFrame, condition: str, min_plays: int) -> tuple:
    even = splits(df[df["season"] % 2 == 0], condition, min_plays)
    odd = splits(df[df["season"] % 2 == 1], condition, min_plays)
    both = even[["d"]].rename(columns={"d": "d_even"}).join(odd[["d"]].rename(columns={"d": "d_odd"}), how="inner")
    if len(both) < 8:
        return len(both), float("nan"), float("nan")
    r, p = stats.pearsonr(both["d_even"], both["d_odd"])
    return len(both), r, p


MIN_PLAYS = {"QB EPA/dropback": 60, "fumbles/touch": 40, "catch rate": 25, "kickers FG over expected": 8}


def main():
    first = int(sys.argv[1]) if len(sys.argv) > 1 else 2016
    data = outcomes(load(first))
    for name, df in data.items():
        print(f"\n== {name} ({len(df):,} plays) ==")
        for condition in CONDITIONS:
            s = splits(df, condition, MIN_PLAYS[name])
            if len(s) < 8:
                print(f"  {condition:5s} too few players")
                continue
            r = random_effects(s)
            n, rho, p_half = split_half(df, condition, max(10, MIN_PLAYS[name] // 2))
            print(f"  {condition:5s} {r['players']:4d} players  league {r['league_d']:+.4f}  "
                  f"true spread (tau) {r['tau']:.4f} vs noise per player {r['typical_se']:.4f}  "
                  f"p {r['p']:.3f}  reliability {r['reliability']:.2f}  "
                  f"| even vs odd seasons: r {rho:+.2f} (p {p_half:.2f}, {n} players)")


if __name__ == "__main__":
    main()

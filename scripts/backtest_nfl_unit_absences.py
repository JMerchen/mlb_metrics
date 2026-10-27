"""What does an absence do to a unit's efficiency - by position, and for a
particular player? The NFL page's "What if?" panel defaults.

A unit's efficiency in a game is its yards per play over what was expected
(its own previous 8 games times the opponent's previous 8 allowed, over the
league's): run offense (designed runs), pass offense (dropbacks, sacks
included), and the run and pass defense facing them. Regulars come from
snap counts (half or more of their side's snaps over the team's previous 4
games); a regular with no snaps is out.

1. Position defaults: efficiency ~ the snap shares of the regulars out,
   by position group, for each unit the group touches (play-weighted) - how
   much a unit loses per regular out.
2. Player-specific: each player's own with/without - the unit's efficiency
   in his team's earlier games he missed, minus the games he played - pulled
   toward his position default with K pseudo-games. Does it predict a unit's
   next game better than the position default? (Earlier games only.)
3. Conversions for the panel: points per yard (league), and win log-odds
   per point of margin (Vegas: closing moneyline against the spread).

RESULT (2026-10-27, 2016-2026; efficiency change per regular out, p):

                 run offense    pass offense   run defense    pass defense
    RB          -.040 (.23)    -.022 (.41)
    TE          -.031 (.22)    -.056 (.01)
    WR          -.014 (.34)    -.002 (.90)
    OL           .000 (1.0)    -.000 (.98)
    DL                                        +.025 (.13)    +.021 (.14)
    LB                                        -.003 (.79)    +.029 (.006)
    CB                                        +.004 (.73)    +.023 (.015)
    S                                         -.007 (.59)    -.002 (.85)

Measured on yards per play, single absences barely register - even the
line, whose absences cost the most in results (backtest_nfl_injury_impact).
A player's own with/without predicted his unit's next game WORSE than his
position at every K (run offense MSE: K=3 .146, K=24 .127, position .126,
nothing .126) - a handful of games is mostly noise. So the panel's
defaults are the position's measured cost in results, turned into the
unit drop that costs as much; a player's record is shown for reference.
Conversions: .0696 points per yard, .137 win log-odds per point of spread.

Usage:
    python scripts/backtest_nfl_unit_absences.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import statsmodels.api as sm

from mlb_metrics import nfl_data, nfl_totals, nfl_whatif

SEASONS = list(range(2016, 2027))
K_GRID = [0, 3, 6, 12, 24, None]  # None: position default only


def regulars_by_game(snaps: pd.DataFrame) -> pd.DataFrame:
    """One row per regular per team-game: [game_id, season, week, team,
    pfr_id, group, share (his previous-4-game share), out]."""
    snaps = snaps[snaps["game_type"] == "REG"].copy()
    snaps["team"] = nfl_totals._canonical(snaps["team"])
    snaps["group"] = snaps["position"].map(nfl_whatif._group_of)
    snaps = snaps[snaps["group"].notna()]
    games = snaps[["season", "week", "game_id", "team"]].drop_duplicates().sort_values(["season", "week"])
    out = []
    for group, (side, _) in nfl_whatif.GROUPS.items():
        rows = snaps[snaps["group"] == group]
        grid = rows.pivot_table(index=["team", "game_id"], columns="pfr_player_id", values=f"{side}_pct", aggfunc="max")
        for team, g in grid.groupby(level="team"):
            order = games[games["team"] == team]
            g = g.droplevel("team").reindex(order["game_id"])
            prior = g.fillna(0).shift(1).rolling(nfl_whatif.LOOKBACK_GAMES, min_periods=2).mean()
            regular = prior >= nfl_whatif.REGULAR_SHARE
            played = g.notna() & (g > 0)
            long = pd.DataFrame({"share": prior.where(regular).stack(), "out": (~played).where(regular).stack()})
            long = long.dropna().reset_index().rename(columns={"pfr_player_id": "pfr_id"})
            long["team"], long["group"] = team, group
            out.append(long)
    regs = pd.concat(out, ignore_index=True)
    regs["out"] = regs["out"].astype(bool)
    return regs.merge(games, on=["game_id", "team"])


def design(ratios: pd.DataFrame, regs: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Rows for one play type: efficiency - 1, plays, and the snap share
    out per group on the offense (`off_<G>`) and the defense (`def_<G>`)."""
    yds, n = nfl_whatif.UNIT_PLAYS[kind]
    lost = regs[regs["out"]].groupby(["game_id", "team", "group"])["share"].sum().unstack(fill_value=0)
    d = ratios[["game_id", "season", "team", "opp", n, f"{kind}_ratio"]].rename(columns={n: "plays", f"{kind}_ratio": "ratio"})
    d = d.merge(lost.add_prefix("off_"), left_on=["game_id", "team"], right_index=True, how="left")
    d = d.merge(lost.add_prefix("def_"), left_on=["game_id", "opp"], right_index=True, how="left")
    offense = [g for g, (side, _) in nfl_whatif.GROUPS.items() if side == "offense"]
    keep = [c for c in d.columns if not c.startswith(("off_", "def_"))
            or (c.startswith("off_") and c[4:] in offense) or (c.startswith("def_") and c[4:] not in offense)]
    d = d[keep].dropna(subset=["ratio"])
    d = d[(d["plays"] >= 5) & np.isfinite(d["ratio"])]
    return d.fillna(0)


def position_defaults(designs: dict) -> pd.DataFrame:
    rows = []
    for kind, d in designs.items():
        cols = [c for c in d.columns if c.startswith(("off_", "def_"))]
        fit = sm.WLS(d["ratio"] - 1, sm.add_constant(d[cols]), weights=d["plays"]).fit()
        for c in cols:
            side, group = c.split("_", 1)
            rows.append({"unit": f"{kind}_{'offense' if side == 'off' else 'defense'}", "group": group,
                         "per_regular": fit.params[c], "se": fit.bse[c], "p": fit.pvalues[c]})
    return pd.DataFrame(rows)


def player_test(designs: dict, regs: pd.DataFrame, defaults: pd.DataFrame) -> pd.DataFrame:
    """Each unit's next-game efficiency predicted from the regulars out -
    by position default, or by each player's own earlier with/without
    pulled toward it with K pseudo-games. Weighted MSE, games 2018+ with
    someone out on that unit."""
    out = []
    for kind, d in designs.items():
        for side in ("offense", "defense"):
            unit = f"{kind}_{side}"
            team_col = "team" if side == "offense" else "opp"
            prior = defaults[defaults["unit"] == unit].set_index("group")["per_regular"]
            groups = [g for g in nfl_whatif.UNIT_GROUPS[unit]]
            r = regs[regs["group"].isin(groups)].merge(
                d[["game_id", team_col, "ratio", "plays"]].rename(columns={team_col: "team"}), on=["game_id", "team"])
            r = r.sort_values(["season", "week"])
            # Running with/without from his earlier games only.
            r["w_out"] = r["plays"] * r["out"]
            r["w_in"] = r["plays"] * ~r["out"]
            g = r.groupby("pfr_id")
            for col in ("w_out", "w_in"):
                r[f"{col}_ratio"] = g[col].transform(lambda x: x.shift(1).fillna(0).cumsum())
            r["out_ratio_sum"] = g.apply(lambda x: (x["ratio"] * x["w_out"]).shift(1).fillna(0).cumsum(),
                                         include_groups=False).reset_index(level=0, drop=True)
            r["in_ratio_sum"] = g.apply(lambda x: (x["ratio"] * x["w_in"]).shift(1).fillna(0).cumsum(),
                                        include_groups=False).reset_index(level=0, drop=True)
            r["games_out"] = g["out"].transform(lambda x: x.shift(1).fillna(0).cumsum())
            raw = (r["out_ratio_sum"] / r["w_out_ratio"].replace(0, np.nan)
                   - r["in_ratio_sum"] / r["w_in_ratio"].replace(0, np.nan)).fillna(0)
            pos = r["group"].map(prior).fillna(0) * r["share"]
            absent = r[r["out"]].copy()
            for k in K_GRID:
                if k is None:
                    eff = pos
                else:
                    n_out = r["games_out"]
                    eff = (n_out * raw + k * pos) / (n_out + k).replace(0, np.nan)
                    eff = eff.fillna(pos)
                absent[f"k{k}"] = eff[r["out"]]
            pred = absent.groupby(["game_id", "team"])[[f"k{k}" for k in K_GRID]].sum()
            target = r.drop_duplicates(["game_id", "team"]).set_index(["game_id", "team"])[["ratio", "plays", "season"]]
            t = target.join(pred, how="inner")
            t = t[t["season"] >= 2018]
            row = {"unit": unit, "games": len(t)}
            for k in K_GRID:
                row[f"K={k}" if k is not None else "position only"] = np.average((t["ratio"] - 1 - t[f"k{k}"]) ** 2, weights=t["plays"])
            row["nothing"] = np.average((t["ratio"] - 1) ** 2, weights=t["plays"])
            out.append(row)
    return pd.DataFrame(out).set_index("unit")


def conversions(schedules: pd.DataFrame, yards: pd.DataFrame) -> dict:
    s = schedules[(schedules["game_type"] == "REG") & schedules["home_moneyline"].notna() & schedules["spread_line"].notna()]
    imp = lambda ml: np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))  # noqa: E731
    h, a = imp(s["home_moneyline"]), imp(s["away_moneyline"])
    p = np.clip(h / (h + a), 1e-3, 1 - 1e-3)
    fit = sm.OLS(np.log(p / (1 - p)), s["spread_line"]).fit()  # spread_line: home favored by
    played = schedules[schedules["home_score"].notna()]
    points = played["home_score"].sum() + played["away_score"].sum()
    yds = yards["run_yards"].sum() + yards["dropback_yards"].sum()
    return {"logodds_per_point": round(float(fit.params.iloc[0]), 4), "points_per_yard": round(float(points / yds), 4),
            "points_per_yard_note": "all points over scrimmage yards on runs and dropbacks"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw/nfl")
    args = parser.parse_args()
    snaps = pd.concat([nfl_data.load_persisted_table(args.raw_dir, "snap_counts", s) for s in SEASONS], ignore_index=True)
    pbp_cols = nfl_whatif.YARDS_PBP_COLUMNS
    yards = pd.concat([nfl_whatif.unit_game_yards(nfl_data.load_persisted_table(args.raw_dir, "pbp", s)[pbp_cols])
                       for s in SEASONS], ignore_index=True)
    yards["team"], yards["opp"] = nfl_totals._canonical(yards["team"]), nfl_totals._canonical(yards["opp"])
    schedules = pd.concat([nfl_data.load_persisted_table(args.raw_dir, "schedules", s) for s in SEASONS], ignore_index=True)

    regs = regulars_by_game(snaps)
    ratios = nfl_whatif.unit_ratios(yards)
    designs = {kind: design(ratios, regs, kind) for kind in nfl_whatif.UNIT_PLAYS}
    defaults = position_defaults(designs)
    pd.set_option("display.width", 200)
    print("Efficiency change per regular out (fraction of normal yards per play; defense: + = allows more):\n")
    table = defaults.pivot(index="group", columns="unit", values="per_regular").round(4)
    print(table.to_string())
    print("\np-values:\n")
    print(defaults.pivot(index="group", columns="unit", values="p").round(3).to_string())
    print("\nNext-game unit efficiency, weighted MSE (lower is better), games 2018+ with someone out:\n")
    print(player_test(designs, regs, defaults).round(5).to_string())
    print("\nConversions:", conversions(schedules, yards))


if __name__ == "__main__":
    main()

"""Backtest the Team units tab's four views (nfl_units.VIEWS): which one's
ratings best predict what each unit actually does in its next game?

For every regular-season week 2017 on, each view rates every unit from
the games before that week only, using nfl_units' own functions (the
season view's "current season" is the one being played, so week 1 has
nothing and is skipped). The QB view rates the QB who actually started.
Each rating, in EPA per play, is scored against that game's EPA per play
for the unit - targets for Receiving, designed runs for Rushing,
dropbacks for QB and Pass D, runs allowed for Run D - by play-weighted
correlation, early weeks (2-4) apart from the rest.

RESULT (2026-10-26, 4,636 team-games 2017-2026, weeks 2+; play-weighted
correlation of the rating with the unit's EPA per play that game):

                  QB+   Receiving+  Rushing+  Pass D+  Run D+   average  weeks 2-4
    season       .240     .195       .137      .112     .077     .152      .109
    recency      .277     .191       .142      .120     .078     .162      .170
    season_sos   .239     .196       .141      .116     .080     .154      .109
    recency_sos  .275     .192       .148      .122     .080     .163      .163

recency_sos is best on average and for four of the five units, so it became
nfl_units.DEFAULT_VIEW (it was "season", the weakest - and far the weakest
early in a season, when it has only a few games to go on).

Usage:
    python scripts/backtest_nfl_unit_views.py [--first 2017] [--last 2026]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from mlb_metrics import config, nfl_data, nfl_totals, nfl_units


def load(raw_dir, seasons):
    unit_frames, qb_frames = [], []
    columns = sorted(set(nfl_units.UNIT_PBP_COLUMNS) | set(nfl_totals.QB_PBP_COLUMNS))
    for season in seasons:
        pbp = nfl_data.load_persisted_table(raw_dir, "pbp", season)
        if pbp is None:
            continue
        pbp = pbp[[c for c in columns if c in pbp.columns]]
        unit_frames.append(nfl_units.unit_game_epa(pbp))
        qb_frames.append(nfl_totals.qb_game_epa(pbp))
    return pd.concat(unit_frames, ignore_index=True), pd.concat(qb_frames, ignore_index=True)


def week_ratings(history: pd.DataFrame, qb_history: pd.DataFrame, qb_ids: pd.Series, teams: pd.Index):
    """EPA-per-play ratings per view for `teams` (and the QBs in `qb_ids`,
    indexed like `teams`)."""
    games = history.reset_index(drop=True)
    current_season = games["season"].max()
    out = {}
    for view, (recency, sos) in nfl_units.VIEWS.items():
        w_off = nfl_units._game_weights(games, "team", recency, current_season)
        w_def = nfl_units._game_weights(games, "opp", recency, current_season)
        solved = {pt: nfl_units._solve_play_type(games, num, den, w_off, w_def, sos)
                  for pt, (num, den) in nfl_units._PLAY_TYPES.items()}
        rated = pd.DataFrame(index=teams)
        rated["receiving"] = teams.map(solved["targets"][0])
        rated["rushing"] = teams.map(solved["runs"][0])
        rated["pass_defense"] = teams.map(solved["dropbacks"][1])
        rated["run_defense"] = teams.map(solved["runs"][1])
        rated["qb"] = nfl_units._qb_view(qb_history, qb_ids, recency, current_season,
                                         solved["dropbacks"][1] if sos else None).values
        out[view] = rated
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw/nfl")
    parser.add_argument("--first", type=int, default=2017)
    parser.add_argument("--last", type=int, default=config.NFL_SEASON)
    args = parser.parse_args()

    units, qbs = load(args.raw_dir, range(args.first - 1, args.last + 1))
    # The QB who started: most dropbacks for his team that game.
    starters = qbs.sort_values("dropbacks").drop_duplicates(["game_id", "team"], keep="last")
    rows = []
    for season in range(args.first, args.last + 1):
        for week in sorted(units.loc[units["season"] == season, "week"].unique()):
            if week < 2:
                continue
            prior = (units["season"] == season - 1) | ((units["season"] == season) & (units["week"] < week))
            history, qb_history = units[prior], qbs[((qbs["season"] == season - 1)) | ((qbs["season"] == season) & (qbs["week"] < week))]
            today = units[(units["season"] == season) & (units["week"] == week)]
            if today.empty:
                continue
            today = today.merge(starters[["game_id", "team", "qb"]], on=["game_id", "team"], how="left")
            teams = pd.Index(today["team"])
            rated = week_ratings(history, qb_history, today["qb"].reset_index(drop=True).set_axis(teams), teams)
            opp_index = pd.Index(today["opp"])
            for view, r in rated.items():
                # Defense ratings belong to the opponent the offense faced.
                rows.append(pd.DataFrame({
                    "season": season, "week": week, "view": view,
                    "receiving": r["receiving"].values, "rushing": r["rushing"].values, "qb": r["qb"].values,
                    "pass_defense": rated[view]["pass_defense"].reindex(opp_index).values,
                    "run_defense": rated[view]["run_defense"].reindex(opp_index).values,
                    "act_target": (today["target_epa"] / today["targets"]).values, "n_target": today["targets"].values,
                    "act_run": (today["run_epa"] / today["runs"]).values, "n_run": today["runs"].values,
                    "act_db": (today["dropback_epa"] / today["dropbacks"]).values, "n_db": today["dropbacks"].values,
                }))
    scored = pd.concat(rows, ignore_index=True)

    def wcorr(x, y, w):
        ok = x.notna() & y.notna() & (w > 0)
        x, y, w = x[ok], y[ok], w[ok]
        mx, my = np.average(x, weights=w), np.average(y, weights=w)
        return np.average((x - mx) * (y - my), weights=w) / np.sqrt(np.average((x - mx) ** 2, weights=w) * np.average((y - my) ** 2, weights=w))

    pairs = [("qb", "act_db", "n_db"), ("receiving", "act_target", "n_target"), ("rushing", "act_run", "n_run"),
             ("pass_defense", "act_db", "n_db"), ("run_defense", "act_run", "n_run")]
    table = []
    for view, g in scored.groupby("view", sort=False):
        early = g["week"] <= 4
        row = {"view": view}
        for unit, actual, n in pairs:
            row[unit] = wcorr(g[unit], g[actual], g[n])
            row[f"{unit}_wk2_4"] = wcorr(g[unit][early], g[actual][early], g[n][early])
        table.append(row)
    table = pd.DataFrame(table).set_index("view")
    pd.set_option("display.width", 200)
    print(f"{len(scored) // len(nfl_units.VIEWS)} team-games, {args.first}-{args.last}, weeks 2+")
    print(table[[u for u, _, _ in pairs]].round(4).to_string())
    print("\nWeeks 2-4 only:")
    print(table[[f"{u}_wk2_4" for u, _, _ in pairs]].round(4).to_string())


if __name__ == "__main__":
    main()

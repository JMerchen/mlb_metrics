"""Ranges with a stated hit rate for the NFL pages: writes
docs/data/nfl_ranges.json (src/mlb_metrics/conformal.py has the method,
scripts/research_conformal_ranges.py the test of it).

  total, margin   the past misses of our point total and of our margin,
                  from data/backtest/nfl_game_misses_replay.csv (every
                  game replayed with only what was known before it;
                  --replay --cache DIR rebuilds it). Thousands of games,
                  so one more week does not move them.
  props           the past misses of each player projection, replayed
                  here every run from the weekly player files on disk
                  (this season and the two before it), in 5 groups by the
                  size of the projection and scaled to it.

The page turns these into "half the time" and "4 times in 5" ranges, and
into the chance a player clears a line.

Usage:
  python scripts/build_nfl_ranges.py
  python scripts/build_nfl_ranges.py --replay --cache DIR
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlb_metrics import config, conformal  # noqa: E402

MISSES_FILE = os.path.join("data", "backtest", "nfl_game_misses_replay.csv")
LEVELS = [0.5, 0.8]
PROP_GROUPS = 5
PROP_FIRST_WEEK, PROP_MIN_GAMES = 4, 3
# The lowest a stat can go: a back can finish with negative rushing yards.
PROP_FLOOR = {"Receptions": 0.0, "Receiving Yards": 0.0, "Passing Yards": 0.0, "Rushing Yards": None}


def game_misses(cache: str) -> pd.DataFrame:
    """One row per game: our total and margin as predicted before it, the
    closing lines and what happened. Totals from 2018 (the totals model's
    replay, rain unknown as when posted), margins from 2002."""
    import statsmodels.api as sm

    import backtest_nfl_totals as totals_replay
    import research_nfl_spread_edge as spread_replay
    from mlb_metrics import nfl_totals

    games, weekly = spread_replay.load(cache)
    d = spread_replay.replay(games, weekly)[["game_id", "season", "week", "gameday", "our_margin", "margin", "spread_line"]]
    team_games, qb_games, schedules = totals_replay.load()
    columns = nfl_totals.TOTALS_FEATURE_COLUMNS
    features = totals_replay.weekly_features(team_games, qb_games, schedules).dropna(subset=columns + ["points"])
    scored = []
    for season in totals_replay.SCORED_SEASONS:
        model = totals_replay.fit(features[features["season"] < season])
        test = features[features["season"] == season].copy()
        test["ours"] = model.predict(sm.add_constant(test.assign(wx_rain=0.0)[columns], has_constant="add"))
        scored.append(test)
    t = pd.concat(scored).groupby("game_id").agg(our_total=("ours", "sum"), total=("points", "sum"), sides=("points", "size")).reset_index()
    t = t[t["sides"] == 2].merge(schedules[["game_id", "total_line"]], on="game_id")[["game_id", "our_total", "total", "total_line"]]
    out = d.merge(t, on="game_id", how="outer").sort_values(["gameday", "game_id"])
    out["season"] = out["season"].fillna(out["game_id"].str[:4].astype(float)).astype(int)
    return out.round(3)


def prop_misses(seasons) -> pd.DataFrame:
    """Every player projection the model would have made in `seasons`
    (from week PROP_FIRST_WEEK of each), next to what the player did."""
    import backtest_nfl_prop_projections as replay

    weekly, schedules = replay.load_weekly(seasons), replay.load_schedules(seasons)
    frames = []
    for season in sorted(weekly["season"].unique())[1:]:
        history = weekly[weekly["season"].isin([season - 1, season])]
        for week in sorted(history.loc[history["season"] == season, "week"].unique()):
            if week >= PROP_FIRST_WEEK:
                rows = replay.replay_week(history, season, week, PROP_MIN_GAMES, schedules)
                if len(rows):
                    frames.append(rows)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build(misses: pd.DataFrame | None, props: pd.DataFrame) -> dict:
    out = {"levels": LEVELS, "props": {}}
    if misses is not None and not misses.empty:
        totals = misses.dropna(subset=["our_total", "total"])
        margins = misses.dropna(subset=["our_margin", "margin"])
        out["total"] = conformal.calibrate(totals["our_total"], totals["total"])
        out["margin"] = conformal.calibrate(margins["our_margin"], margins["margin"])
    if not props.empty:
        out["through"] = {"season": int(props["season"].max()),
                          "week": int(props.loc[props["season"] == props["season"].max(), "week"].max())}
        for category, rows in props.groupby("category"):
            cal = conformal.calibrate(rows["projection"], rows["actual"], groups=PROP_GROUPS, scaled=True)
            cal["floor"] = PROP_FLOOR.get(category, 0.0)
            out["props"][category] = cal
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--season", type=int, default=config.NFL_SEASON)
    parser.add_argument("--replay", action="store_true", help="Rebuild the game misses file first (needs --cache).")
    parser.add_argument("--cache", help="Folder for the downloaded games file the margin replay uses.")
    args = parser.parse_args()
    if args.replay:
        if not args.cache:
            parser.error("--replay needs --cache")
        os.makedirs(os.path.dirname(MISSES_FILE), exist_ok=True)
        game_misses(args.cache).to_csv(MISSES_FILE, index=False)
    misses = pd.read_csv(MISSES_FILE) if os.path.exists(MISSES_FILE) else None
    out = build(misses, prop_misses([args.season - 2, args.season - 1, args.season]))
    with open(os.path.join(args.data_dir, "nfl_ranges.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    for name in ("total", "margin"):
        if name in out:
            bands = ", ".join(f"{int(level * 100)}%: {conformal.range_for(out[name], 0.0, level)[0]:+.1f} to "
                              f"{conformal.range_for(out[name], 0.0, level)[1]:+.1f}" for level in LEVELS)
            print(f"{name}: {out[name]['n']} games; {bands}")
    for category, cal in out["props"].items():
        print(f"{category}: {cal['n']} player-games in {len(cal['groups'])} groups")


if __name__ == "__main__":
    main()

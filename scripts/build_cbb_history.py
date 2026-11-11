"""Builds the College Hoops "Time Machine" file: every Division I team's
final rating in every season since 2009 (docs/data/cbb_history.json).

For fun, not for picks: the page lets any team from any year play any
other, and seeds the best teams of the era into one bracket. A team's
rating is how far above ITS OWN season's average team it was, so teams
from different years are compared on that scale; nothing here knows how
the game itself has changed.

Each season is rated on all its games, the NCAA tournament included
(cbb_ratings.fit_ratings, the season before as the prior), and each
team carries how its March ended (0 = no tournament, 1 = lost in the
First Four ... 6 = lost in the Final Four, 7 = lost the title game,
8 = champion).

A static file: run it once a year, after the title game.

Usage:
    PYTHONPATH=src python3 scripts/build_cbb_history.py [--cache DIR]
(--cache holds the seasons' schedule and box-score files; missing ones
are downloaded into it. Defaults to data/raw/cbb.)
"""

import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mlb_metrics import cbb_data, cbb_ratings, config  # noqa: E402

OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "data", "cbb_history.json")
# How far a team got: the round it lost in, by ROUND_NAMES order.
LOST_IN = {name: index + 1 for index, name in enumerate(cbb_ratings.ROUND_NAMES)}
CHAMPION = len(cbb_ratings.ROUND_NAMES) + 1


def march_results(games: pd.DataFrame) -> dict:
    """{team_id: code} for the teams in the NCAA tournament: the round each
    lost in (LOST_IN), or CHAMPION."""
    out = {}
    tournament = games[games["tournament"]].sort_values("date")
    for _, g in tournament.iterrows():
        home_won = g["home_score"] > g["away_score"]
        winner, loser = (g["home_id"], g["away_id"]) if home_won else (g["away_id"], g["home_id"])
        out[int(loser)] = LOST_IN[g["round"]]
        if g["round"] == "Championship":
            out[int(winner)] = CHAMPION
        else:
            out.setdefault(int(winner), 0)
    return {team: code for team, code in out.items() if code}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw", "cbb"))
    args = parser.parse_args()
    names, colors, seasons, prior = {}, {}, {}, None
    for season in range(config.CBB_FIRST_SEASON, config.CBB_SEASON):
        schedules = cbb_data.load_table("schedules", season, args.cache)
        games = cbb_ratings.prepare_games(schedules, cbb_data.load_table("team_box", season, args.cache))
        ratings, league = cbb_ratings.fit_ratings(games, prior=prior, ranked=cbb_ratings.preseason_ranked(schedules))
        prior = ratings
        teams = cbb_ratings.team_names(schedules).set_index("team_id")
        march = march_results(games)
        rows = []
        for _, r in ratings[ratings["team_id"].isin(teams.index) & (ratings["games"] >= 10)].iterrows():
            team = int(r["team_id"])
            names[team] = teams.at[team, "team"]
            color = teams.at[team, "color"]
            if isinstance(color, str) and len(color) == 6:
                colors[team] = color
            rows.append([team, round(float(r["offense"]), 1), round(float(r["defense"]), 1), round(float(r["tempo"]), 1),
                         int(r["wins"]), int(r["losses"]), march.get(team, 0)])
        rows.sort(key=lambda row: -(row[1] + row[2]))
        champion = next((names[row[0]] for row in rows if row[6] == CHAMPION), None)
        if season == config.CBB_FIRST_SEASON:
            continue      # the first season on file only seeds the next one's ratings (its tournament notes are incomplete)
        seasons[str(season)] = {"efficiency": round(league["efficiency"], 1), "tempo": round(league["tempo"], 1),
                                "tournament": bool(march), "teams": rows}
        best = rows[0]
        print(f"{season}: {len(rows)} teams, {len(march)} in the tournament, champion {champion};"
              f" best {names[best[0]]} {best[1] + best[2]:+.1f} ({best[4]}-{best[5]})")
    payload = {"columns": ["team_id", "offense", "defense", "tempo", "wins", "losses", "march"],
               "sigma": config.CBB_MARGIN_SIGMA, "names": {str(k): v for k, v in names.items()},
               "colors": {str(k): v for k, v in colors.items()}, "seasons": seasons}
    with open(OUTPUT, "w") as fh:
        json.dump(payload, fh, separators=(",", ":"), allow_nan=False)
    print(f"wrote {OUTPUT}: {os.path.getsize(OUTPUT) / 1024:.0f} KB")


if __name__ == "__main__":
    main()

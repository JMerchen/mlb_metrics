"""Where the NFL win model (src/mlb_metrics/nfl_qb_elo.py) is behind Vegas,
and what three additions do about it - on 25 seasons, not the 8 the model
was first scored on.

Why (2026-11-15; the user's "only number 3 helps the nfl model"): of the
leads chased that day only one aimed at the model itself, and it added
nothing. This goes after the model directly. The rating needs only scores,
quarterback box scores and past closing lines, so it can be replayed back
to 1999 (scripts/research_nfl_spread_edge.py loads the data) - enough
games to see what its misses have in common.

RESULT (2026-11-15; 6,636 games 2002-2026, each predicted from ratings as
they stood before it):
  Before the additions the model's log loss was 0.6190 against Vegas's
  0.6106. Where the distance was:
    week 1                        behind by 0.0187 a game  (13% of the whole)
    weeks 2-8                     0.0050  (23%)
    weeks 9 to the second-last    0.0082  (44%)
    the last week                 0.0266  (18% of the whole, from 6% of games)
    a regular starter sitting: 0.0168 a game (half the whole, from a
    quarter of the games); otherwise 0.0056
  The last week: a team with 10+ wins sitting its regular quarterback was
  expected to win 56.1% and won 32.3% (62 teams, 5.8 points worse than
  expected) - the other starters sit too. With fewer wins, as expected
  (37.0% and 37.1%). Any earlier week, a backup in for the season's
  regular starter: expected 40.4%, won 37.4%, 1.1 points worse (1,582).
  A screen of 28 other things the misses might share (one at a time, as an
  offset logistic regression; exploratory, not kept): only rest was
  consistent - +0.033 log-odds a day of extra rest, z 3.1, positive in
  every era. Not consistent or nothing: travel distance and time zones
  (the sign flips by era), division games, dome teams outdoors in the
  cold, wind, turnover / fumble-recovery / return-touchdown luck over the
  last four games, last game's surprise, a quarterback's career starts,
  rookies, first-year coaches, night games, playoff games, early or late
  weeks widening or narrowing the gap.

  The three additions (config.NFL_QB_ELO_REST_PER_DAY and the settings
  after it), each on its own and together:
                                  log loss   better by   t     by era
    as it was                      0.6190
    + rest, 3 Elo a day            0.6184    +0.0006    2.4   + + +
    + backup penalty, 25           0.6183    +0.0007    2.2   + + +
    + both penalties, 25 and 140   0.6172    +0.0018    3.1   + + +
    all three                      0.6166    +0.0024    3.8   +0.0023 +0.0013 +0.0036
  Better in 21 of 25 seasons; over a quarter of the distance to Vegas.
  Twelve neighbouring settings all land within 0.0005. On the site's own
  replay (1,999 games 2019-2026) 0.6197 became 0.6162 against Vegas's
  0.6094. Against the closing spread the same model's 3.5+ point
  disagreements went from 54.6% of 885 to 57.3% of 848
  (research_nfl_spread_edge.py).

  Found on the way (2026-11-15): the schedule feed's listed starters for
  recent seasons are sometimes wrong for games already played (2026: Drew
  Lock listed for Seattle in weeks 3-6, Sam Darnold threw every pass), so
  the real starter's rating never moved and the starts were miscounted.
  nfl_qb_elo.rate now takes the top passer when the listed starter threw
  no pass. It changes 560 predictions from 2022 on; the numbers above
  include it (without it "all three" was 0.6167).

  Tried here and left out: a home edge that drifts with recent results
  (+0.0005, but worse since 2019); a smaller home edge in division games
  (nothing); a flat penalty for any quarterback new to the lineup (the
  backup rule does it better); learning faster from the line in the first
  weeks (+0.0003, t 1.5 at best); and the model's other settings (K,
  market pull, quarterback weight and memory, season carryover), none of
  which moved it more than 0.0005 either way.

Usage:
    PYTHONPATH=src python3 scripts/research_nfl_model_gaps.py --cache DIR
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlb_metrics import config, nfl_qb_elo, nfl_totals  # noqa: E402
import research_nfl_spread_edge as spread  # noqa: E402

ERAS = spread.ERAS
OFF = {"NFL_QB_ELO_REST_PER_DAY": 0.0, "NFL_QB_ELO_BACKUP_PENALTY": 0.0, "NFL_QB_ELO_SITTING_PENALTY": 0.0}
ON = {name: getattr(config, name) for name in OFF}


def loss(d: pd.DataFrame, gap: str = "qb_elo_gap") -> pd.Series:
    p = np.clip(nfl_qb_elo.gap_to_probability(d[gap]), 1e-6, 1 - 1e-6)
    y = np.where(d["margin"] > 0, 1.0, np.where(d["margin"] < 0, 0.0, 0.5))
    return pd.Series(-(y * np.log(p) + (1 - y) * np.log(1 - p)), index=d.index)


def situations(games: pd.DataFrame) -> pd.DataFrame:
    """For each game and side, as known before kickoff: the team's wins so
    far this season and whether its regular starter (most starts this
    season, at least config.NFL_QB_ELO_STARTER_MIN_STARTS) is sitting."""
    alias, wins, starts, season, rows = nfl_totals.NFL_TEAM_ALIASES, {}, {}, None, []
    for g in games.sort_values(["season", "week", "gameday", "game_id"]).itertuples(index=False):
        if g.season != season:
            wins, starts, season = {}, {}, g.season
        row = {"game_id": g.game_id}
        for side, team, starter in (("home", alias.get(g.home_team, g.home_team), g.home_qb_id),
                                    ("away", alias.get(g.away_team, g.away_team), g.away_qb_id)):
            counts = starts.get(team, {})
            regular = max(counts, key=counts.get) if counts else None
            established = regular is not None and counts[regular] >= config.NFL_QB_ELO_STARTER_MIN_STARTS
            row[f"{side}_wins"] = wins.get(team, 0)
            row[f"{side}_sitting"] = bool(established and starter != regular)
        rows.append(row)
        if g.game_type == "REG":
            margin = g.home_score - g.away_score
            for team, won, starter in ((alias.get(g.home_team, g.home_team), margin > 0, g.home_qb_id),
                                       (alias.get(g.away_team, g.away_team), margin < 0, g.away_qb_id)):
                wins[team] = wins.get(team, 0) + int(won)
                if isinstance(starter, str):
                    starts.setdefault(team, {})[starter] = starts.get(team, {}).get(starter, 0) + 1
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    games, weekly = spread.load(args.cache)
    before = spread.replay(games, weekly, **OFF)
    before = before.merge(situations(games), on="game_id")
    before["ours"] = loss(before)
    before["vegas"] = loss(before.assign(line_gap=before["spread_line"] * config.NFL_QB_ELO_POINTS), "line_gap")
    behind = before["ours"] - before["vegas"]
    print(f"{len(before)} games 2002-2026. Before the additions: ours {before['ours'].mean():.4f}, Vegas {before['vegas'].mean():.4f}")

    def part(label, use):
        print(f"  {label:38s} {int(use.sum()):5d} games: behind by {behind[use].mean():+.4f} a game "
              f"({behind[use].sum() / behind.sum():.0%} of the whole)")

    print("\nWhere the distance to Vegas was")
    regular = before["game_type"] == "REG"
    final = regular & (before["week"] == before["season"].map(nfl_qb_elo.final_regular_week))
    part("week 1", regular & (before["week"] == 1))
    part("weeks 2-8", regular & before["week"].between(2, 8))
    part("weeks 9 to the second-last", regular & (before["week"] >= 9) & ~final)
    part("the last week", final)
    part("playoffs", ~regular)
    sitting = before["home_sitting"] | before["away_sitting"]
    part("a regular starter sitting", sitting)
    part("both regular starters (or none yet)", ~sitting)

    print("\nA team whose regular quarterback is sitting: what the model expected and what happened")
    chance = nfl_qb_elo.gap_to_probability(before["qb_elo_gap"])
    sides = pd.concat([
        pd.DataFrame({"expected": chance, "won": before["margin"] > 0, "sitting": before["home_sitting"], "wins": before["home_wins"],
                      "miss": before["margin"] - before["qb_elo_gap"] / config.NFL_QB_ELO_POINTS, "final": final}),
        pd.DataFrame({"expected": 1 - chance, "won": before["margin"] < 0, "sitting": before["away_sitting"], "wins": before["away_wins"],
                      "miss": -(before["margin"] - before["qb_elo_gap"] / config.NFL_QB_ELO_POINTS), "final": final})])
    sides = sides[sides["sitting"]]
    good = sides["wins"] >= config.NFL_QB_ELO_SITTING_MIN_WINS
    for label, use in (("the last week, 10+ wins", sides["final"] & good), ("the last week, fewer wins", sides["final"] & ~good),
                       ("any earlier week", ~sides["final"])):
        s = sides[use]
        print(f"  {label:28s} {len(s):5d} teams: expected to win {s['expected'].mean():.1%}, won {s['won'].mean():.1%}, "
              f"margin {s['miss'].mean():+.1f} points against ours")

    print("\nThe additions")
    for label, settings in (("+ rest", {**OFF, "NFL_QB_ELO_REST_PER_DAY": ON["NFL_QB_ELO_REST_PER_DAY"]}),
                            ("+ backup penalty", {**OFF, "NFL_QB_ELO_BACKUP_PENALTY": ON["NFL_QB_ELO_BACKUP_PENALTY"]}),
                            ("+ resting starters (with the backup penalty)", {**ON, "NFL_QB_ELO_REST_PER_DAY": 0.0}),
                            ("all three (the model)", ON)):
        after = spread.replay(games, weekly, **settings)
        gain = before["ours"].to_numpy() - loss(after).to_numpy()
        gain = pd.Series(gain, index=before.index)
        by_season = gain.groupby(before["season"]).mean()
        eras = " ".join(f"{gain[before['season'].between(a, b)].mean():+.4f}" for a, b in ERAS)
        print(f"  {label:46s} log loss {loss(after).mean():.4f}, better by {gain.mean():+.4f} "
              f"(t {gain.mean() / gain.std() * np.sqrt(len(gain)):.1f}); by era {eras}; "
              f"better in {int((by_season > 0).sum())} of {len(by_season)} seasons")


if __name__ == "__main__":
    main()

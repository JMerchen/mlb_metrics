"""The checks behind our NFL wins above replacement
(src/mlb_metrics/nfl_player_value.py), 2016-2025.

1. REPLACEMENT GAPS. With each season's starter and reserve tiers left
   free, how far below a starter is a reserve, a play? One season's answer
   is mostly noise, so the module uses the ten-season pooled gaps.
2. CARRY-OVER. Does a player's points-a-play carry to his next season,
   against plain EPA a play, for players who stayed and players who moved?
3. TEAMS. Do a team's players add up to its offense?
4. DEFENDERS. Starter-to-reserve gaps, how much to pull a rate toward the
   position's average, and how playmaking relates to team defense.
5. KICKERS, and points per win.
6. TEAMS. How many games does a replacement team win, and is what a
   team's rated players do not explain luck?

RESULT (2026-11-23):

  Replacement gaps, points a play (pooled, weighted by reserves' plays; se
  from the ten seasons): quarterback throw 0.123 (.008), no-target 0.025
  (.005), after-catch 0.035 (.010), designed run 0.199 (.073); running
  back run 0.055 (.006), throw 0.061 (.018), after catch 0.046 (.022);
  wide receiver throw 0.122 (.016), after catch 0.029 (.009); tight end
  throw 0.051 (.012), after catch 0.007 (.014). Season by season the wide
  receivers' throw gap ran from 0.03 (2025) to 0.20 (2024) - which is why
  they are constants: with 2025's own gap wide receivers summed to 10
  wins league-wide and running backs to 41; with the pooled gaps, 31 and
  25.

  Carry-over, this season's points a play to next season's (correlation;
  our number / plain EPA a play; pairs in brackets):
    stayed with his team   QB 0.32 / 0.41 (198)   WR 0.19 / 0.17 (427)
                           TE 0.34 / 0.36 (186)   RB catching 0.11 / 0.08
                           (235)   RB running 0.12 / 0.20 (243)
    changed teams          QB 0.00 / -0.08 (39)   WR 0.02 / 0.05 (87)
                           TE -0.08 / -0.02 (38)   RB catching 0.22 / 0.17
                           (43)   RB running 0.01 / -0.04 (60)
  A season of per-play efficiency is a weak guide to the next one for
  everybody, and close to none for a player who changes teams; splitting
  the credit does not fix that, and for quarterbacks who stay our number
  carries over a little LESS than plain EPA (it has taken the team's part
  out, and the team mostly stays too). Points above replacement, which
  also counts how much a player plays, carries at 0.34 to 0.43 for
  players who stay (0.15 for running backs' rushing) and about 0.3 for
  quarterbacks and wide receivers who move. So: read WAR as what a player
  was worth this season, not as a forecast.

  Teams: a team's players' points above average track its offensive EPA
  at r 0.91 (320 team-seasons); each point of theirs goes with 1.8 of the
  team's (the pull toward average, and the team effects, hold the rest).
  An average team's offense is 155 points, 4.4 wins, above replacement.

  Defenders: playmaking points a snap, starter less reserve: DT 0.0064,
  ED 0.0074, LB 0.0029, CB 0.0021, S -0.0012 (so 0). A defender's rate
  carries over better than any offensive efficiency: DT 0.42, ED 0.37, LB
  0.29, S 0.28, CB 0.20 (300+ snaps both seasons). Pulling it toward the
  position's average with k snaps, the error predicting next season: k=0
  0.0176, 100 0.0124, 300 0.0113, 500 0.0109, 800 0.0107, 1500 0.0106 ->
  600. Team by team, the players' playmaking above their positions'
  averages tracks defensive EPA saved at r 0.83, slope 1.71
  (DEFENSE_SCALE); 81% of the value of plays that lost the offense points
  names a defender.

  Kickers: points over expected a kick carry over at r 0.09 (203 pairs of
  40+ kick seasons); error predicting next season with k kicks of pull:
  0 0.132, 50 0.111, 200 0.1045, 400 0.1042 -> 200. A team's fill-in
  kickers were 0.134 points a kick below its main one (1,846 kicks).

  Points per win: wins on point differential, 320 team-seasons: 0.0283
  wins a point (r 0.89), 35.3 points a win.

  Teams (320 team-seasons). A REPLACEMENT TEAM wins 2.7 of 17 (.157;
  2.2 to 3.1 by season) built from the parts - an average team with the
  average team's points above replacement taken off both sides, through
  the Pythagorean curve; 2.07 by the books (the league's 8.25 wins less
  the average team's 6.18 players' WAR); and the five real teams nearest
  it won 3, 2, 3, 6 and 2. Wins on players' WAR added up: slope 1.35
  (+-0.06), r 0.78 - the rated players' sum is narrower than real
  records (spread 1.62 wins against 3.16), because linemen and coverage
  are not in it and every player is pulled toward average. What is left
  splits in two: NOT CREDITED (real points through the curve, less
  expected wins; spread 1.34) and CLOSE GAMES (wins less real points
  through the curve; 1.45). To the next season (288 pairs): wins carry
  at 0.37, expected wins 0.37, not credited 0.31, close games 0.04. Next
  season's wins on this season's three parts: 0.56 (+-0.12), 0.47
  (+-0.14), 0.03 (+-0.12). So close games are luck, and not credited is
  skill the play-by-play cannot hand to a player.

Usage:
  python scripts/research_nfl_player_value.py [--first 2016] [--last 2025] [--only-teams]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd

from mlb_metrics import config, nfl_player_value as v  # noqa: E402

ROLES = [("pass", "QB", 200), ("catch", "WR", 50), ("catch", "TE", 35), ("catch", "RB", 25), ("rush", "RB", 80)]


def corr(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")


def offense_fits(seasons, raw_dir, gaps):
    """(effects with season, plays with season) over the seasons."""
    effects, plays = [], []
    for season in seasons:
        pbp = v.load_table("pbp", season, raw_dir, v.PLAY_COLUMNS)
        roster = v.roster_facts(v.load_table("rosters_weekly", season, raw_dir))
        p = v.offense_plays(pbp)
        effects.append(v.fit_offense(p, roster["position"], gaps=gaps).assign(season=season))
        plays.append(p)
        print(f"  fit {season}{' (free tiers)' if gaps is None else ''}", file=sys.stderr)
    return pd.concat(effects, ignore_index=True), pd.concat(plays, ignore_index=True)


def replacement_gaps(seasons, raw_dir):
    effects, _ = offense_fits(seasons, raw_dir, None)
    effects["position"] = effects["tier"].map(v._tier_position)
    effects["starter"] = effects["tier"].str.endswith("1")
    print("Replacement gaps: a starter less a reserve, points a play (tiers free, each season; pooled by the reserves' plays)")
    for (part, role, position), d in effects[effects["position"] != "OTH"].groupby(["part", "role", "position"]):
        gaps, weights = [], []
        for _, season in d.groupby("season"):
            a, b = season[season["starter"]], season[~season["starter"]]
            if len(a) and len(b) and b["plays"].sum():
                gaps.append(np.average(a["effect"], weights=a["plays"]) - np.average(b["effect"], weights=b["plays"]))
                weights.append(b["plays"].sum())
        gaps = np.array(gaps)
        print(f"  {part:11} {role:5} {position}: {np.average(gaps, weights=weights):+.3f} (se {gaps.std(ddof=1) / np.sqrt(len(gaps)):.3f}; "
              f"seasons {gaps.min():+.2f} to {gaps.max():+.2f}; in use {v.REPLACEMENT_GAP.get((part, role, position), 0.0):.3f})")


def carry_over(effects):
    effects = effects.assign(position=effects["tier"].map(v._tier_position))

    def roll(d):
        n = d["plays"].max()
        return pd.Series({"plays": n, "ours": ((d["effect"] - d["average"]) * d["plays"]).sum() / n, "plain": (d["raw"] * d["plays"]).sum() / n,
                          "par": ((d["effect"] - d["replacement"]) * d["plays"]).sum()})
    rows = effects.groupby(["season", "player_id", "role", "position", "team"]).apply(roll).reset_index()
    rows = rows.sort_values("plays", ascending=False).groupby(["season", "player_id", "role", "position"]).first().reset_index()
    later = rows.assign(season=rows["season"] - 1)
    pairs = rows.merge(later, on=["season", "player_id", "role", "position"], suffixes=("", "_next"))
    pairs["moved"] = pairs["team"] != pairs["team_next"]
    print("Carry-over to his next season (correlation), players with enough plays in both")
    for role, position, least in ROLES:
        for moved in (False, True):
            d = pairs[(pairs["role"] == role) & (pairs["position"] == position) & (pairs["plays"] >= least)
                      & (pairs["plays_next"] >= least) & (pairs["moved"] == moved)]
            print(f"  {role:5} {position} {'moved ' if moved else 'stayed'} n={len(d):4d}  ours {corr(d['ours'], d['ours_next']):+.2f}  "
                  f"plain EPA a play {corr(d['plain'], d['plain_next']):+.2f}  points above replacement {corr(d['par'], d['par_next']):+.2f}")


def team_totals(effects, plays):
    points = effects.assign(paa=effects["effect"] * effects["plays"], par=(effects["effect"] - effects["replacement"]) * effects["plays"])
    team = points.groupby(["season", "team"])[["paa", "par"]].sum()
    team["epa"] = plays.groupby(["season", "posteam"])["epa"].sum().rename_axis(["season", "team"])
    print(f"Teams ({len(team)} team-seasons): players' points above average against the team's offensive EPA: r {corr(team['paa'], team['epa']):.2f}, "
          f"slope {np.polyfit(team['paa'], team['epa'], 1)[0]:.2f}; an average offense is {team['par'].mean():.0f} points "
          f"({team['par'].mean() / v.POINTS_PER_WIN:.1f} wins) above replacement")


def defenders(seasons, raw_dir):
    tables, teams = [], []
    for season in seasons:
        pbp = v.load_table("pbp", season, raw_dir, v.PLAY_COLUMNS + v.DEFENSE_COLUMNS)
        rosters = v.load_table("rosters_weekly", season, raw_dir)
        roster = v.roster_facts(rosters)
        depth = rosters[rosters["gsis_id"].notna()].groupby("gsis_id")["depth_chart_position"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
        credits = v.defense_credits(pbp)
        table = v.defense_table(credits, v.defense_snaps(v.load_table("snap_counts", season, raw_dir), roster), roster, depth)
        tables.append(table.assign(season=season, team=table["team"].replace(v.TEAM_ALIASES)))
        p = pbp[(pbp["season_type"] == "REG") & pbp["play_type"].isin(["run", "pass"]) & pbp["epa"].notna() & pbp["down"].notna()
                & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)]
        t = p.groupby("defteam").agg(epa=("epa", "sum"), lost=("epa", lambda s: -s[s < 0].sum())).rename_axis("team").reset_index()
        t["credited"] = t["team"].map(credits.groupby("team")["points"].sum())
        teams.append(t.assign(season=season, team=t["team"].replace(v.TEAM_ALIASES)))
    d, teams = pd.concat(tables, ignore_index=True), pd.concat(teams, ignore_index=True)
    d["rate"] = d["points"] / d["snaps"]
    by = d.groupby(["position", "starter"])[["points", "snaps"]].sum()
    rate = (by["points"] / by["snaps"]).unstack("starter")
    print("Defenders: playmaking points a snap, starters and reserves")
    for position in rate.index:
        print(f"  {position}: {rate.loc[position, True]:.4f} and {rate.loc[position, False]:.4f}, gap {rate.loc[position, True] - rate.loc[position, False]:+.4f} "
              f"(in use {v.DEFENSE_REPLACEMENT_GAP[position]:.4f})")
    average = d.groupby(["season", "position"])[["points", "snaps"]].sum()
    d = d.join((average["points"] / average["snaps"]).rename("average"), on=["season", "position"])
    later = d[["season", "player_id", "position", "rate", "snaps"]].assign(season=d["season"] - 1)
    pairs = d.merge(later, on=["season", "player_id", "position"], suffixes=("", "_next"))
    for position in rate.index:
        both = pairs[(pairs["position"] == position) & (pairs["snaps"] >= 300) & (pairs["snaps_next"] >= 300)]
        print(f"  {position} rate to next season's (300+ snaps both): r {corr(both['rate'], both['rate_next']):.2f} ({len(both)})")
    pairs = pairs[pairs["snaps_next"] >= 300]
    print("  error predicting next season's rate, by snaps of pull toward the position's average: " + ", ".join(
        f"{k} {np.sqrt(np.average((pairs['rate_next'] - (pairs['points'] + k * pairs['average']) / (pairs['snaps'] + k)) ** 2, weights=pairs['snaps_next'])):.4f}"
        for k in (0, 100, 300, 500, 800, 1500)))
    d["above"] = d["points"] - d["average"] * d["snaps"]
    teams = teams.join(d.groupby(["season", "team"])["above"].sum(), on=["season", "team"]).dropna(subset=["above"])
    teams["saved"] = -(teams["epa"] - teams.groupby("season")["epa"].transform("mean"))
    print(f"  teams ({len(teams)}): playmaking above the positions' averages against defensive EPA saved: r {corr(teams['above'], teams['saved']):.2f}, "
          f"slope {np.polyfit(teams['above'], teams['saved'], 1)[0]:.2f} (in use {v.DEFENSE_SCALE}); "
          f"{(teams['credited'] / teams['lost']).mean():.0%} of the value of plays that lost the offense points names a defender")


def kickers(seasons, raw_dir):
    rows = []
    for season in seasons:
        pbp = v.load_table("pbp", season, raw_dir, ["season_type", "play_type", "posteam", "kicker_player_id", "field_goal_result", "extra_point_result", "fg_prob"])
        p = pbp[(pbp["season_type"] == "REG") & pbp["kicker_player_id"].notna() & pbp["play_type"].isin(["field_goal", "extra_point"])]
        fg = p["play_type"] == "field_goal"
        made = np.where(fg, p["field_goal_result"] == "made", p["extra_point_result"] == "good").astype(float)
        expected = np.where(fg, p["fg_prob"], made[~fg.to_numpy()].mean())
        rows.append(pd.DataFrame({"season": season, "team": p["posteam"].to_numpy(), "kicker": p["kicker_player_id"].to_numpy(),
                                  "points": np.where(fg, 3.0, 1.0) * (made - expected)}))
    d = pd.concat(rows).dropna(subset=["points"])
    by_team = d.groupby(["season", "team", "kicker"])["points"].agg(["sum", "size"]).reset_index()
    by_team["main"] = by_team.groupby(["season", "team"])["size"].rank(ascending=False, method="first") == 1
    main, rest = by_team[by_team["main"]], by_team[~by_team["main"]]
    by = d.groupby(["season", "kicker"])["points"].agg(["sum", "size"]).reset_index()
    by["rate"] = by["sum"] / by["size"]
    pairs = by.merge(by.assign(season=by["season"] - 1), on=["season", "kicker"], suffixes=("", "_next"))
    pairs = pairs[(pairs["size"] >= 40) & (pairs["size_next"] >= 40)]
    print(f"Kickers: main {main['sum'].sum() / main['size'].sum():+.3f} a kick, fill-ins {rest['sum'].sum() / rest['size'].sum():+.3f} ({rest['size'].sum()} kicks); "
          f"carry-over r {corr(pairs['rate'], pairs['rate_next']):.2f} ({len(pairs)}); error by kicks of pull: " + ", ".join(
              f"{k} {np.sqrt(np.mean((pairs['rate_next'] - pairs['sum'] / (pairs['size'] + k)) ** 2)):.4f}" for k in (0, 50, 200, 400)))


def points_per_win(seasons, raw_dir):
    rows = []
    for season in seasons:
        g = v.load_table("schedules", season, raw_dir)
        g = g[(g["game_type"] == "REG") & g["home_score"].notna()]
        for side, other in (("home", "away"), ("away", "home")):
            margin = g[f"{side}_score"] - g[f"{other}_score"]
            rows.append(pd.DataFrame({"season": season, "team": g[f"{side}_team"], "margin": margin, "win": (margin > 0) + 0.5 * (margin == 0)}))
    t = pd.concat(rows).groupby(["season", "team"])[["margin", "win"]].sum()
    slope = np.polyfit(t["margin"], t["win"], 1)[0]
    print(f"Points per win ({len(t)} team-seasons): {slope:.4f} wins a point (r {corr(t['margin'], t['win']):.2f}), {1 / slope:.1f} points a win")


def team_checks(seasons, raw_dir):
    """The replacement team, and whether what a team's rated players do
    not explain is luck: does it carry to the next season, and does it
    predict it?"""
    rows = []
    for season in seasons:
        d = v.build(season, raw_dir)
        players = pd.DataFrame(d["players"]).groupby("team")["war"].sum()
        rows.append(pd.DataFrame(d["teams"]).assign(season=season, rate=d["replacement_win_rate"]).join(players.rename("players"), on="team"))
        print(f"  teams {season}", file=sys.stderr)
    t = pd.concat(rows, ignore_index=True)
    print(f"Teams ({len(t)} team-seasons): a replacement team wins {17 * t['rate'].mean():.1f} of 17 through the curve ({t['rate'].mean():.3f}; "
          f"seasons {17 * t['rate'].min():.1f} to {17 * t['rate'].max():.1f}); by the books, the league's {t['wins'].mean():.2f} wins less the average team's "
          f"{t['players'].mean():.2f} players' WAR = {t['wins'].mean() - t['players'].mean():.2f}")
    x = np.column_stack([np.ones(len(t)), t["players"]])
    b = np.linalg.lstsq(x, t["wins"], rcond=None)[0]
    resid = t["wins"] - x @ b
    se = np.sqrt(np.diag(np.linalg.inv(x.T @ x)) * resid.var(ddof=2))
    print(f"  wins on players' WAR added up: intercept {b[0]:.2f} (+-{se[0]:.2f}), slope {b[1]:.2f} (+-{se[1]:.2f}), r {corr(t['players'], t['wins']):.2f}")
    print(f"  spread of real wins {t['wins'].std():.2f}; of expected wins {t['expected'].std():.2f} (missing wins by {np.sqrt(((t['expected'] - t['wins']) ** 2).mean()):.2f}; "
          f"real points through the curve miss by {np.sqrt((t['close'] ** 2).mean()):.2f}); leftovers: not credited {t['uncredited'].std():.2f}, close games {t['close'].std():.2f}")
    worst = t.sort_values("war").head(5)
    print("  the nearest things to a replacement team: " + "; ".join(f"{r.team} {r.season} WAR {r.war:.1f}, won {r.wins:g}" for r in worst.itertuples()))
    later = t[["season", "team", "wins", "expected", "uncredited", "close"]].assign(season=t["season"] - 1)
    m = t.merge(later, on=["season", "team"], suffixes=("", "_next"))
    print(f"  to the next season ({len(m)} pairs): wins r {corr(m['wins'], m['wins_next']):.2f}; expected wins {corr(m['expected'], m['expected_next']):.2f}; "
          f"not credited {corr(m['uncredited'], m['uncredited_next']):.2f}; close games {corr(m['close'], m['close_next']):.2f}")
    x = np.column_stack([np.ones(len(m)), m["expected"] - m["expected"].mean(), m["uncredited"], m["close"]])
    b = np.linalg.lstsq(x, m["wins_next"], rcond=None)[0]
    resid = m["wins_next"] - x @ b
    se = np.sqrt(np.diag(np.linalg.inv(x.T @ x)) * resid.var(ddof=4))
    print(f"  next season's wins on this season's expected wins, not credited and close games: {b[1]:.2f} (+-{se[1]:.2f}), {b[2]:.2f} (+-{se[2]:.2f}), {b[3]:.2f} (+-{se[3]:.2f})")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=config.NFL_RAW_DATA_DIR)
    parser.add_argument("--first", type=int, default=2016)
    parser.add_argument("--last", type=int, default=2025)
    parser.add_argument("--only-teams", action="store_true", help="Just the team checks (section 6).")
    args = parser.parse_args()
    seasons = list(range(args.first, args.last + 1))
    if args.only_teams:
        team_checks(seasons, args.raw_dir)
        return
    replacement_gaps(seasons, args.raw_dir)
    effects, plays = offense_fits(seasons, args.raw_dir, v.REPLACEMENT_GAP)
    carry_over(effects)
    team_totals(effects, plays)
    defenders(seasons, args.raw_dir)
    kickers(seasons, args.raw_dir)
    points_per_win(seasons, args.raw_dir)
    team_checks(seasons, args.raw_dir)


if __name__ == "__main__":
    main()

"""The checks behind our MLB wins above replacement
(src/mlb_metrics/mlb_war.py), on the seasons with pitches on disk.

1. THE OUT MODEL. How well does it say which balls in play become outs,
   and is a "70%" ball caught 70% of the time?
2. REPLACEMENT. How far below the average of their kind are the players
   past a team's roster?
3. RUNS PER WIN, from wins on run differential.
4. TEAMS. Do a team's players add up to its wins and its run
   differential?
5. CARRY-OVER. Does a player's number carry to his next season?
6. BASERUNNING. Is it already in Statcast's run value? How big is it?
7. TEAMS. How many games does a replacement team win; do players' WAR
   add up to a team's record; and are the leftovers luck?

RESULT (2026-11-23; 2025 and 2026):

  Out model (every ball scored by a model fit on other games): log loss
  0.353 and 0.351 against 0.61 for the league's rate alone. Calibration,
  2026: balls given 0-10% were outs 2%, 10-30% 15%, 30-50% 40%, 50-70%
  63%, 70-90% 83%, 90-100% 97%. An out is worth over a hit: fly ball 0.86
  runs, line drive 0.77, ground ball 0.69, popup 0.68. Fielders range
  from about -16 to +16 runs a season (+21 the best of 2025); the spread
  at second base and shortstop (3.7 runs) is nearly twice that at first
  base and left field (2.1).

  Replacement, runs a plate appearance below the average of their kind
  (2025, 2026): hitters past a team's 13th 0.042, 0.035 (about 20,000
  plate appearances a season) -> 0.038, 23 runs over 600; starters past
  its 6th 0.015, 0.013 -> 0.014; relievers past its 8th 0.023, 0.026 ->
  0.025.

  Runs per win: 11.3 on our 60 team-seasons (r 0.91, give or take 0.6).
  The usual rule at 4.5 runs a team a game gives 9.7, which is what is
  used.

  Teams: a team's WAR tracks its wins at r 0.89 (2025) and 0.90 (2026);
  its players' runs above average (by the team each was playing for)
  track its run differential at r 0.95 and 0.94, slope 1.02 and 1.09 -
  a run of ours is a run. The league adds
  to 1,069 and 1,078 WAR (the conventional scale is about 1,000), two
  thirds of it hitters' (conventionally 57%).

  The average hitter at each position, runs over 600 plate appearances
  against the league (2025, 2026): DH +12.5, +8.8; 1B +5.2, +10.2; RF
  +2.6, -0.4; 3B -1.2, +4.2; LF -0.6, -2.1; SS -0.9, -3.3; C -1.7, -2.7;
  2B -5.0, -7.0; CF -8.8, -5.2.

  Baserunning: not in Statcast's number. A single with a man on first
  and nobody out is worth 0.45 runs to it when he takes third and 0.43
  when he stops at second (2026; 0.47 and 0.46 in 2025); by our own
  run-expectancy table the two are 0.89 and 0.65. And 5,598 pitches in
  2026 moved a runner in the middle of a plate appearance (5,001
  advances worth +0.19 each, 597 outs worth -0.55), where Statcast's
  value for those pitches averaged +0.016 - the count's. Runners' runs:
  best +10.7 and +9.8, worst -5.9 and -8.5, spread 1.9; steals and
  advances each sum to zero. Carry-over a time on base r 0.34 (257
  runners with 120+ both years; steals 0.34, advances 0.21).

  Teams (60 team-seasons). A REPLACEMENT TEAM wins 44.2 of 162 (.273)
  built from the parts (an average team scoring 491 and allowing 838);
  45.2 by the books (the league's 81.0 wins less the average team's 35.8
  players' WAR); and the 2025 Rockies, at 1.5 wins above one, were
  expected to win 45.7 and won 43. Regressing wins on players' WAR added
  up gives an intercept of 53.2 (+-2.0) and a slope of 0.78 (+-0.05):
  adding up overstates the extremes (spread 12.9 against 11.3 for real
  wins; misses by 5.92). Players' runs through the Pythagorean curve:
  spread 10.3, misses by 5.40. Leftovers: not credited spread 3.9 wins,
  close games 4.7. First half to second half (60 team-halves): expected
  wins carry at 0.51, not credited 0.21, close games 0.23; predicting the
  second half's record: first-half record 0.48, real runs through the
  curve 0.43, expected wins 0.40. 2025 to 2026: 0.68, 0.22, -0.04.

  Carry-over 2025 to 2026: batting runs a plate appearance r 0.54 (194
  hitters with 300+ both years), pitching 0.40 (172 with 250+ batters
  faced), fielding a chance 0.39 (197 with 150+), WAR 0.60 (1,124
  players in both). Far steadier than the NFL's.

Usage:
  python scripts/research_mlb_war.py [--seasons 2025 2026]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import numpy as np
import pandas as pd

from mlb_metrics import data, mlb_war as w  # noqa: E402


def corr(a, b) -> float:
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 2 else float("nan")


def season_pieces(season, raw_dir):
    pitches = data.load_persisted_statcast(raw_dir, season, w.PITCH_COLUMNS)
    pitches = pitches[pitches["game_type"] == "R"]
    p = w.evaluate(pitches)
    p["pitches"] = pitches
    return p


def running_checks(season, p):
    """Baserunning: is it in Statcast's number already, how big is it, does it add up."""
    pitches = p["pitches"].sort_values(["game_pk", "at_bat_number", "pitch_number"])
    ended = pitches[pitches["events"].notna()].copy()
    first_only = (ended["events"] == "single") & ended["on_1b"].notna() & ended["on_2b"].isna() & ended["on_3b"].isna() & (ended["outs_when_up"] == 0)
    half = ["game_pk", "inning", "inning_topbot"]
    after_third = pitches.groupby(half)["on_3b"].shift(-1).notna().loc[ended.index]
    singles = ended[first_only]
    took_third = after_third[first_only]
    r = p["running"].groupby("player_id")[["steals", "advances", "running", "times_on"]].sum()
    print(f"{season} baserunning: a single with a man on first and nobody out is worth {singles.loc[took_third, 'delta_run_exp'].mean():.2f} runs to Statcast when he "
          f"takes third and {singles.loc[~took_third, 'delta_run_exp'].mean():.2f} when he stops at second ({int(took_third.sum())} and {int((~took_third).sum())} singles)")
    print(f"  runners' runs: best {r['running'].max():+.1f}, worst {r['running'].min():+.1f}, spread {r['running'].std():.2f}; "
          f"steals sum {r['steals'].sum():+.1f}, advances sum {r['advances'].sum():+.1f}")
    return r


def team_checks(pieces, raw_dir):
    """The replacement team three ways, adding up against the curve, and whether the leftovers carry."""
    full = pd.concat([p["teams"].assign(season=season, players=p["table"].groupby("team")["war"].sum()) for season, p in pieces.items()])
    print(f"Teams ({len(full)} team-seasons): a replacement team wins {full['baseline'].mean():.1f} of 162 through the curve "
          f"({full['baseline'].mean() / 162:.3f}); by the books, the league's {full['wins'].mean():.1f} wins less the average team's "
          f"{full['players'].mean():.1f} players' WAR = {full['wins'].mean() - full['players'].mean():.1f}")
    x = np.column_stack([np.ones(len(full)), full["players"]])
    b = np.linalg.lstsq(x, full["wins"], rcond=None)[0]
    resid = full["wins"] - x @ b
    se = np.sqrt(np.diag(np.linalg.inv(x.T @ x)) * resid.var(ddof=2))
    print(f"  wins on players' WAR added up: intercept {b[0]:.1f} (+-{se[0]:.1f}), slope {b[1]:.2f} (+-{se[1]:.2f})")
    added = full["wins"].mean() - full["players"].mean() + full["players"]
    print(f"  spread of real wins {full['wins'].std():.1f}; of players' WAR added up {added.std():.1f} (missing wins by {np.sqrt(((added - full['wins']) ** 2).mean()):.2f}); "
          f"of expected wins through the curve {full['expected'].std():.1f} (missing by {np.sqrt(((full['expected'] - full['wins']) ** 2).mean()):.2f})")
    worst = full.sort_values("war").iloc[0]
    print(f"  the nearest thing to a replacement team: {worst.name} {int(worst['season'])}, team WAR {worst['war']:.1f}, expected {worst['expected']:.1f}, won {int(worst['wins'])}")
    print(f"  leftovers, wins: not credited spread {full['uncredited'].std():.1f}, close games {full['close'].std():.1f}")
    halves = []
    for season, p in pieces.items():
        pitches = p["pitches"]
        dates = np.sort(pitches["game_date"].unique())
        middle = dates[len(dates) // 2]
        first, second = w.evaluate(pitches[pitches["game_date"] < middle])["teams"], w.evaluate(pitches[pitches["game_date"] >= middle])["teams"]
        halves.append(first.join(second, lsuffix="_1", rsuffix="_2").assign(season=season))
        print(f"  halves of {season} done", file=sys.stderr)
    h = pd.concat(halves)
    rate = {k: h[f"{k}_1"] / h["games_1"] for k in ("wins", "expected", "uncredited", "close")}
    later = {k: h[f"{k}_2"] / h["games_2"] for k in ("wins", "expected", "uncredited", "close")}
    actual_runs = (h["wins_1"] - h["close_1"]) / h["games_1"]
    print(f"  first half to second half ({len(h)} team-halves): expected wins carry at r {corr(rate['expected'], later['expected']):.2f}, "
          f"not credited {corr(rate['uncredited'], later['uncredited']):.2f}, close games {corr(rate['close'], later['close']):.2f}")
    print(f"  predicting the second half's record from the first half's: record r {corr(rate['wins'], later['wins']):.2f}, "
          f"real runs through the curve {corr(actual_runs, later['wins']):.2f}, expected wins {corr(rate['expected'], later['wins']):.2f}")
    seasons = sorted(pieces)
    for a, b2 in zip(seasons, seasons[1:]):
        m = full[full["season"] == a].join(full[full["season"] == b2], lsuffix="_a", rsuffix="_b")
        print(f"  {a} to {b2}: expected wins r {corr(m['expected_a'], m['expected_b']):.2f}, not credited {corr(m['uncredited_a'], m['uncredited_b']):.2f}, "
              f"close games {corr(m['close_a'], m['close_b']):.2f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2025, 2026])
    args = parser.parse_args()
    pieces = {}
    for season in args.seasons:
        pieces[season] = season_pieces(season, args.raw_dir)
        print(f"  fit {season}", file=sys.stderr)

    runners = {}
    for season, p in pieces.items():
        runners[season] = running_checks(season, p)
        balls = p["balls"]
        out, chance = balls["out"].astype(int), balls["chance"].clip(1e-6, 1 - 1e-6)
        base = out.mean()
        print(f"{season} out model: log loss {-(out * np.log(chance) + (1 - out) * np.log(1 - chance)).mean():.3f} "
              f"(the league's rate alone {-(base * np.log(base) + (1 - base) * np.log(1 - base)):.3f}); given -> were outs: "
              + ", ".join(f"{interval.left:.0%}-{interval.right:.0%} {rate:.0%}" for interval, rate in
                          balls.groupby(pd.cut(balls["chance"], [0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]), observed=True)["out"].mean().items()))
        print(f"  an out over a hit, runs: {balls.groupby('bb_type')['gap'].first().round(2).to_dict()}")
        f = p["credits"].groupby(["player_id", "position"])["runs"].sum().reset_index()
        print(f"  fielders' runs, spread by position: {f.groupby('position')['runs'].std().round(1).to_dict()}; best {f['runs'].max():.1f}, worst {f['runs'].min():.1f}")
        b, pt = p["fit"]["batters"], p["fit"]["pitchers"]
        spare = b[b["tier"] == "replacement"]
        gaps = [f"hitters {b['runs'].sum() / b['pa'].sum() - spare['runs'].sum() / spare['pa'].sum():.4f} ({spare['pa'].sum()} plate appearances)"]
        for role in ("SP", "RP"):
            every, spare = pt[pt["tier"].str.startswith(role)], pt[pt["tier"] == f"{role} replacement"]
            gaps.append(f"{role} {every['runs'].sum() / every['pa'].sum() - spare['runs'].sum() / spare['pa'].sum():.4f} ({spare['pa'].sum()} batters)")
        print(f"  replacement below the average of their kind, runs a plate appearance: {'; '.join(gaps)} (in use {w.REPLACEMENT})")
        t = p["table"]
        team = t.groupby("team")[["war", "raa"]].sum().join(p["records"])
        team["raa"] = p["teams"]["offense"] + p["teams"]["defense"]          # by the team each was playing for at the time
        margin = team["scored"] - team["allowed"]
        hitters = t[t["pa"] >= t["bf"]]
        print(f"  teams: WAR against wins r {corr(team['war'], team['wins']):.2f}; runs above average against run differential r {corr(team['raa'], margin):.2f}, "
              f"slope {np.polyfit(team['raa'], margin, 1)[0]:.2f}; league {t['war'].sum():.0f} WAR, {hitters['war'].sum() / t['war'].sum():.0%} hitters; "
              f"{team['scored'].sum() / team['games'].sum():.2f} runs a team a game")
        print("  average hitter by position, runs over 600 plate appearances against the league: "
              + str(hitters.groupby("position").apply(lambda d: 600 * d["batting_league"].sum() / max(d["pa"].sum(), 1)).round(1).to_dict()))

    records = pd.concat([p["records"] for p in pieces.values()])
    margin = records["scored"] - records["allowed"]
    slope = np.polyfit(margin, records["wins"], 1)[0]
    print(f"Runs per win ({len(records)} team-seasons): {1 / slope:.1f} (r {corr(margin, records['wins']):.2f}; in use {w.RUNS_PER_WIN})")

    seasons = sorted(pieces)
    for first, second in zip(seasons, seasons[1:]):
        m = pieces[first]["table"].set_index("player_id").join(pieces[second]["table"].set_index("player_id"), lsuffix="_a", rsuffix="_b", how="inner")
        h = m[(m["pa_a"] >= 300) & (m["pa_b"] >= 300)]
        p = m[(m["bf_a"] >= 250) & (m["bf_b"] >= 250)]
        f = m[(m["chances_a"] >= 150) & (m["chances_b"] >= 150)]
        ra, rb = runners[first], runners[second]
        on = ra.join(rb, lsuffix="_a", rsuffix="_b", how="inner")
        on = on[(on["times_on_a"] >= 120) & (on["times_on_b"] >= 120)]
        print(f"  running a time on base r {corr(on['running_a'] / on['times_on_a'], on['running_b'] / on['times_on_b']):.2f} ({len(on)}: "
              f"steals {corr(on['steals_a'] / on['times_on_a'], on['steals_b'] / on['times_on_b']):.2f}, advances {corr(on['advances_a'] / on['times_on_a'], on['advances_b'] / on['times_on_b']):.2f})")
        print(f"Carry-over {first} to {second}: batting runs a plate appearance r {corr(h['batting_league_a'] / h['pa_a'], h['batting_league_b'] / h['pa_b']):.2f} ({len(h)}), "
              f"pitching r {corr(p['pitching_a'] / p['bf_a'], p['pitching_b'] / p['bf_b']):.2f} ({len(p)}), "
              f"fielding a chance r {corr(f['fielding_a'] / f['chances_a'], f['fielding_b'] / f['chances_b']):.2f} ({len(f)}), WAR r {corr(m['war_a'], m['war_b']):.2f} ({len(m)})")
    team_checks(pieces, args.raw_dir)


if __name__ == "__main__":
    main()

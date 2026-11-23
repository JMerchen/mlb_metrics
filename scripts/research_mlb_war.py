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

  Teams: a team's WAR tracks its wins at r 0.88 (2025) and 0.89 (2026);
  its players' runs above average track its run differential at r 0.92
  in both, slope 0.96 and 1.09 - a run of ours is a run. The league adds
  to 1,069 and 1,078 WAR (the conventional scale is about 1,000), two
  thirds of it hitters' (conventionally 57%).

  The average hitter at each position, runs over 600 plate appearances
  against the league (2025, 2026): DH +12.5, +8.8; 1B +5.2, +10.2; RF
  +2.6, -0.4; 3B -1.2, +4.2; LF -0.6, -2.1; SS -0.9, -3.3; C -1.7, -2.7;
  2B -5.0, -7.0; CF -8.8, -5.2.

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
    pa = w.plate_appearances(pitches)
    balls = w.fielding_chances(pa)
    credits = w.fielding_credits(balls)
    fit = w.fit_plate_appearances(pa, w.neutral_value(pa, balls))
    return {"balls": balls, "credits": credits, "fit": fit, "table": w.player_table(pa, fit, credits), "records": w.team_records(pitches)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--seasons", type=int, nargs="+", default=[2025, 2026])
    args = parser.parse_args()
    pieces = {}
    for season in args.seasons:
        pieces[season] = season_pieces(season, args.raw_dir)
        print(f"  fit {season}", file=sys.stderr)

    for season, p in pieces.items():
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
        print(f"Carry-over {first} to {second}: batting runs a plate appearance r {corr(h['batting_league_a'] / h['pa_a'], h['batting_league_b'] / h['pa_b']):.2f} ({len(h)}), "
              f"pitching r {corr(p['pitching_a'] / p['bf_a'], p['pitching_b'] / p['bf_b']):.2f} ({len(p)}), "
              f"fielding a chance r {corr(f['fielding_a'] / f['chances_a'], f['fielding_b'] / f['chances_b']):.2f} ({len(f)}), WAR r {corr(m['war_a'], m['war_b']):.2f} ({len(m)})")


if __name__ == "__main__":
    main()

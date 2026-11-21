"""Are play-calling habits real? The checks behind the play-calling tree
(src/mlb_metrics/nfl_play_sequencing.py; they decided how its offense and
defense sides are described).

Three questions, on every regular-season run and pass of 2021-2026:

1. Does what came before change the call, league-wide? For each situation
   the pass rate is compared with nflfastR's expected pass rate for the
   same plays (down, distance, field, score, clock - but not the play
   before).
2. Is a team's lean in a situation a habit or luck? Its pass rate over
   typical in the odd weeks of a season against the even weeks. The same
   for what a defense allows.
3. Does knowing the habits predict calls? Each team's leans from weeks
   1-9, pulled toward the league as the map does, predict its calls from
   week 10 on, against the typical call alone (log loss per play).

RESULT (2026-11-20):
1. What came before changes the call. On 2nd & long a team that just
   threw incomplete passes 68% of the time and one that was just stuffed
   on a run 72% - while the situation alone says the first should pass
   MORE (74% against 70%). A 7-point swing toward doing the other thing
   after a failure. (The expected pass rate runs about 5 points high in
   every situation in these seasons; the differences between rows are the
   sequence.)
2. An offense's lean in a situation is a habit: odd weeks against even
   weeks it correlates 0.25 to 0.57 situation by situation (0.72 over all
   its plays). What a defense allows in a situation is not: -0.06 to
   +0.17, on 5 to 40 plays a half - noise. Even over a whole half-season a
   defense's success allowed against the pass carries over at only 0.26
   and against the run at 0.15. Hence the map: offense habits per
   situation, the defense read through its overall pass and run numbers,
   pulled hard toward the league.
3. Habits predict calls. Weeks 1-9 predicting weeks 10 on: log loss 0.5812
   from the situation alone, 0.5672 adding what came before league-wide,
   0.5645 adding the team's habits pulled toward the league as the map
   does - and 0.5746, WORSE, with the team's raw rates. The pulling is
   what makes the habits useful.

  185,998 plays, 2021-2026

  1. and 2. By situation: the pass rate, what the situation alone predicts, and how well a team's lean and a defense's results carry over
                                             plays   pass expected   lean | offense lean r (n) | defense success allowed r: v pass (n), v run (n)
    To open a drive                         31,582  47.8%    51.4%  -3.6% | 0.50 (82)          | +0.13 (39), -0.02 (42)
    After a run moved the chains            17,469  45.3%    48.8%  -3.5% | 0.33 (45)          | -0.06 (20), -0.04 (25)
    After a pass moved the chains           29,201  50.1%    55.3%  -5.2% | 0.50 (75)          | +0.05 (37), +0.07 (37)
    2nd & long, after a pass that failed    16,031  68.3%    73.9%  -5.6% | 0.55 (41)          | +0.12 (28), +0.15 (13)
    2nd & long, after a run that failed     15,502  71.7%    70.0%  +1.7% | 0.44 (40)          | +0.14 (29), -0.00 (11)
    2nd & medium, after a run               14,361  53.0%    57.8%  -4.8% | 0.43 (37)          | -0.02 (19), +0.08 (17)
    2nd & medium, after a pass               5,878  53.6%    65.1% -11.5% | 0.25 (15)          | +0.05 (8), -0.03 (7)
    2nd & short                             10,048  35.0%    39.8%  -4.8% | 0.28 (26)          | +0.09 (9), -0.02 (17)
    3rd & short                              8,009  37.6%    47.9% -10.3% | 0.37 (20)          | +0.10 (7), +0.00 (13)
    3rd & medium                            12,682  82.6%    91.4%  -8.9% | 0.44 (32)          | +0.03 (27), +0.17 (5)
    3rd & long                              17,755  86.9%    93.4%  -6.6% | 0.25 (46)          | +0.03 (40), +0.09 (6)
    Right after a gain of 15+               16,720  46.0%    53.2%  -7.2% | 0.43 (43)          | +0.12 (20), +0.01 (23)
    Right after a sack or a loss            11,238  77.7%    82.8%  -5.1% | 0.29 (29)          | +0.08 (22), +0.03 (6)
    Right after an incompletion             23,314  74.9%    80.7%  -5.8% | 0.48 (60)          | +0.03 (45), +0.05 (15)
    Right after a run held to 2 or less     28,784  67.5%    68.8%  -1.3% | 0.45 (75)          | +0.14 (50), +0.03 (24)
    After two runs in a row                 27,465  52.9%    58.9%  -6.0% | 0.46 (71)          | +0.09 (38), +0.02 (33)
    After two passes in a row               36,766  66.7%    72.0%  -5.3% | 0.40 (95)          | +0.12 (63), +0.14 (31)
    Inside the 20                           28,233  51.0%    58.0%  -7.0% | 0.57 (73)          | +0.10 (37), -0.13 (36)
    Whole season, every play:
      offense lean                     r 0.72 on 484 plays a half -> a prior of about 190 plays
      offense success passing          r 0.60 on 278 plays a half -> a prior of about 184 plays
      offense success running          r 0.44 on 205 plays a half -> a prior of about 258 plays
      defense success allowed v pass   r 0.26 on 278 plays a half -> a prior of about 787 plays
      defense success allowed v run    r 0.15 on 205 plays a half -> a prior of about 1127 plays

  3. Weeks 1-9 habits predicting weeks 10 on (log loss per play; lower is better)
    the situation alone (expected pass rate)             0.5812   call guessed right 68.9%
    ... plus what came before, league-wide               0.5672   call guessed right 69.5%
    ... plus the team's habits, as the map pulls them    0.5645   call guessed right 69.7%
    ... plus the team's raw rates, not pulled            0.5746   call guessed right 69.3%
    (80,349 plays, 2021-2025)

Usage:
  python scripts/research_nfl_play_sequencing.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mlb_metrics import nfl_play_sequencing as seq  # noqa: E402

SEASONS = range(2021, 2027)
LABELS = {key: label for key, label, _ in seq.STATES} | dict(seq.HABITS)


def plays() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        p = seq.scrimmage(seq.load_plays(season=season))
        p["situation"] = seq.situations(p)
        frames.append(pd.concat([p, seq.habit_flags(p)], axis=1))
    out = pd.concat(frames, ignore_index=True)
    out["lean"] = out["is_pass"] - out["xpass"]
    out["half"] = out["week"] % 2
    return out


def carry_over(rows: pd.DataFrame, team: str, value: str) -> tuple:
    """(r, plays a half): a team's value over the league's in the odd
    weeks of a season against the even weeks."""
    cell = rows.groupby(["season", team, "half"])[value].agg(["mean", "count"]).unstack("half").dropna()
    league = rows.groupby("season")[value].mean()
    a = cell[("mean", 0)] - cell.index.get_level_values("season").map(league)
    b = cell[("mean", 1)] - cell.index.get_level_values("season").map(league)
    return float(np.corrcoef(a, b)[0, 1]), int(cell[("count", 0)].mean())


def main():
    p = plays()
    groups = [(key, p[p["situation"] == key]) for key, _, _ in seq.STATES] + [(key, p[p[key]]) for key, _ in seq.HABITS]
    print(f"{len(p):,} plays, {SEASONS[0]}-{SEASONS[-1]}")
    print("\n1. and 2. By situation: the pass rate, what the situation alone predicts, and how well a team's lean and a defense's results carry over")
    print(f"  {'':38s} {'plays':>7s} {'pass':>6s} {'expected':>8s} {'lean':>6s} | offense lean r (n) | defense success allowed r: v pass (n), v run (n)")
    for key, rows in groups:
        off = carry_over(rows, "posteam", "lean")
        dp = carry_over(rows[rows["is_pass"] == 1], "defteam", "success")
        dr = carry_over(rows[rows["is_pass"] == 0], "defteam", "success")
        print(f"  {LABELS[key]:38s} {len(rows):7,} {rows['is_pass'].mean():6.1%} {rows['xpass'].mean():8.1%} {rows['lean'].mean():+6.1%} | "
              f"{off[0]:.2f} ({off[1]})          | {dp[0]:+.2f} ({dp[1]}), {dr[0]:+.2f} ({dr[1]})")
    print("  Whole season, every play:")
    for label, rows, team, value in (
            ("offense lean", p, "posteam", "lean"),
            ("offense success passing", p[p["is_pass"] == 1], "posteam", "success"), ("offense success running", p[p["is_pass"] == 0], "posteam", "success"),
            ("defense success allowed v pass", p[p["is_pass"] == 1], "defteam", "success"),
            ("defense success allowed v run", p[p["is_pass"] == 0], "defteam", "success")):
        r, n = carry_over(rows, team, value)
        print(f"    {label:32s} r {r:.2f} on {n} plays a half -> a prior of about {n * (1 - r) / r:.0f} plays")

    print("\n3. Weeks 1-9 habits predicting weeks 10 on (log loss per play; lower is better)")
    rows = []
    for season in SEASONS[:-1]:
        s = p[(p["season"] == season) & (p["situation"] != "")]
        early, late = s[s["week"] <= 9], s[s["week"] >= 10].copy()
        league = early.groupby("situation")["lean"].mean()
        late["typical"] = np.clip(late["xpass"] + late["situation"].map(league), 0.02, 0.98)
        cells = early.groupby(["posteam", "situation"]).agg(n=("is_pass", "size"), passes=("is_pass", "sum"), xpass=("xpass", "mean")).reset_index()
        cells["typical"] = np.clip(cells["xpass"] + cells["situation"].map(league), 0.02, 0.98)
        cells["team_lean"] = (cells["passes"] + seq.LEAN_PRIOR * cells["typical"]) / (cells["n"] + seq.LEAN_PRIOR) - cells["typical"]
        cells["raw_lean"] = cells["passes"] / cells["n"] - cells["typical"]
        late = late.merge(cells[["posteam", "situation", "team_lean", "raw_lean"]], on=["posteam", "situation"], how="left").fillna({"team_lean": 0.0, "raw_lean": 0.0})
        rows.append(late)
    late = pd.concat(rows)
    loss = lambda prob: float(-np.mean(np.where(late["is_pass"] == 1, np.log(prob), np.log(1 - prob))))
    clip = lambda x: np.clip(x, 0.02, 0.98)
    for label, prob in (("the situation alone (expected pass rate)", clip(late["xpass"])),
                        ("... plus what came before, league-wide", late["typical"]),
                        ("... plus the team's habits, as the map pulls them", clip(late["typical"] + late["team_lean"])),
                        ("... plus the team's raw rates, not pulled", clip(late["typical"] + late["raw_lean"]))):
        right = np.mean((prob >= 0.5) == (late["is_pass"] == 1))
        print(f"  {label:52s} {loss(prob):.4f}   call guessed right {right:.1%}")
    print(f"  ({len(late):,} plays, {SEASONS[0]}-{SEASONS[-2]})")


if __name__ == "__main__":
    main()

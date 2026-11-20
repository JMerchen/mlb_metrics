"""Do ranges built from the model's own past misses (conformal prediction,
src/mlb_metrics/conformal.py) hit as often as they say - for the point
total, the margin and a player's projected stat line?

Every range is made only from games before the one it is for: totals and
margins season by season (calibrated on all earlier seasons), player
projections week by week (calibrated on every earlier week).

RESULT (2026-11-20):
  Point total (ours with rain unknown, 2019-2025, 1,871 games):
    "half the time"  hit 50.3%, 17.8 points wide (the site's hand-set
                     plus or minus 9 hit 50.8% - it was already right)
    "4 times in 5"   hit 80.1%, 34.1 wide (79-81% by season)
    The same ranges around Vegas's total: 52.3% and 82.0%. Vegas's misses
    are a little smaller than ours, as its record says.
  Margin (our rating's, 2006-2026, 5,581 games):
    half the time 51.4%, 16.9 wide; 4 in 5 80.5%, 34.6 wide. Around the
    closing spread: 52.0% and 81.2% at the same widths.
    A scale by the size of the prediction changes nothing for either.
  Player projections (2024 week 4 to 2026 week 10, updated weekly; "4 in
  5" ranges: hit rate overall, then for the smallest / middle / largest
  third of projections):
                       one range for all       5 groups + a scale (used)
    Receiving yards    78.9% (92 / 85 / 59)    79.6% (81 / 79 / 80)
    Receptions         80.5% (93 / 84 / 64)    80.1% (81 / 80 / 79)
    Rushing yards      79.0% (97 / 83 / 57)    79.7% (84 / 75 / 80)
    Passing yards      80.3% (72 / 86 / 84)    80.0% (78 / 83 / 79)
    One range for everyone is right on average and wrong for almost
    everyone: far too wide for small projections, far too narrow for big
    ones. Grouping by size and scaling fixes it. "Half the time" ranges:
    50.0%, 49.1%, 51.1%, 49.8%.
  The chance of going over a line (the line taken as the player's own
  recent average, rounded to a half): said 27% -> happened 27%; 36 -> 37;
  42 -> 45; 47 -> 46; 52 -> 53; 57 -> 57; 64 -> 64; 80 -> 79 (20,855
  player-games). Log loss against always saying the base rate: receiving
  yards 0.669 v 0.691, receptions 0.650 v 0.693, rushing 0.658 v 0.693,
  passing 0.679 v 0.692.
  CAUTION: that is against a line set at his average. A sportsbook's line
  already knows the matchup and the injuries, and we have no past prop
  lines to test against, so against a real line treat the chance as an
  upper bound on how sure to be.

Usage:
  python scripts/research_conformal_ranges.py --cache DIR
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlb_metrics import conformal  # noqa: E402

LEVELS = (0.5, 0.8)


def by_season(frame, predicted, actual, first=None, scaled=False) -> pd.DataFrame:
    """Each game's ranges from a calibration on all earlier seasons."""
    out = []
    for season in sorted(frame["season"].unique()):
        past, now = frame[frame["season"] < season], frame[frame["season"] == season]
        if len(past) < 200 or (first and season < first):
            continue
        cal = conformal.calibrate(past[predicted], past[actual], scaled=scaled)
        out.append(_score(cal, now, predicted, actual, None))
    return pd.concat(out, ignore_index=True)


def by_week(frame, groups, scaled, lower, start=(2024, 4)) -> pd.DataFrame:
    """Each player-game's ranges from a calibration on every earlier week."""
    when = frame["season"] * 100 + frame["week"]
    out = []
    for t in sorted(when.unique()):
        if t < start[0] * 100 + start[1]:
            continue
        cal = conformal.calibrate(frame.loc[when < t, "projection"], frame.loc[when < t, "actual"], groups=groups, scaled=scaled)
        out.append(_score(cal, frame[when == t], "projection", "actual", lower))
    return pd.concat(out, ignore_index=True)


def _score(cal, rows, predicted, actual, lower) -> pd.DataFrame:
    out = pd.DataFrame({"season": rows["season"].to_numpy(), "predicted": rows[predicted].to_numpy(), "actual": rows[actual].to_numpy()})
    for level in LEVELS:
        ranges = [conformal.range_for(cal, p, level, lower) for p in out["predicted"]]
        low, high = np.array([r[0] for r in ranges]), np.array([r[1] for r in ranges])
        out[f"in{int(level * 100)}"] = (out["actual"] >= low) & (out["actual"] <= high)
        out[f"wide{int(level * 100)}"] = high - low
    if "naive" in rows.columns:
        line = np.floor(rows["naive"].to_numpy()) + 0.5
        out["said"] = [conformal.chance_over_line(cal, p, x) for p, x in zip(out["predicted"], line)]
        out["over"] = out["actual"].to_numpy() > line
    return out


def report(label, scored) -> None:
    thirds = pd.qcut(scored["predicted"], 3, labels=False, duplicates="drop")
    parts = []
    for level in LEVELS:
        hit, wide = scored[f"in{int(level * 100)}"], scored[f"wide{int(level * 100)}"]
        seasons = hit.groupby(scored["season"]).mean()
        parts.append(f"{int(level * 100)}%: hit {hit.mean():.1%} ({seasons.min():.0%}-{seasons.max():.0%} by season), "
                     f"{wide.mean():.1f} wide, by size {' / '.join(f'{v:.0%}' for v in hit.groupby(thirds).mean())}")
    print(f"  {label:34s} n {len(scored):5d} | " + " | ".join(parts))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True, help="Folder for the downloaded games file (as research_nfl_spread_edge.py).")
    args = parser.parse_args()
    import build_nfl_ranges

    games = build_nfl_ranges.game_misses(args.cache)
    totals = games.dropna(subset=["our_total", "total"])
    print("Point total")
    report("ours", by_season(totals, "our_total", "total"))
    report("ours, scaled", by_season(totals, "our_total", "total", scaled=True))
    fixed = (totals["total"] - totals["our_total"]).abs() <= 9
    print(f"  the hand-set +-9 hit {fixed[totals['season'] >= 2019].mean():.1%}")
    report("around Vegas's total", by_season(totals.dropna(subset=["total_line"]), "total_line", "total"))
    margins = games.dropna(subset=["our_margin", "margin"])
    print("Margin")
    report("ours", by_season(margins, "our_margin", "margin", first=2006))
    report("ours, scaled", by_season(margins, "our_margin", "margin", first=2006, scaled=True))
    report("around the closing spread", by_season(margins.dropna(subset=["spread_line"]), "spread_line", "margin", first=2006))

    props = build_nfl_ranges.prop_misses([2022, 2023, 2024, 2025, 2026])
    print("Player projections, updated weekly")
    said = []
    for category, rows in props.groupby("category"):
        lower = build_nfl_ranges.PROP_FLOOR.get(category, 0.0)
        report(f"{category}: one range for all", by_week(rows, 1, False, lower))
        used = by_week(rows, build_nfl_ranges.PROP_GROUPS, True, lower)
        report(f"{category}: grouped and scaled", used)
        said.append(used.assign(category=category))
    said = pd.concat(said)
    print("The chance of going over a line set at the player's own average")
    bins = pd.cut(said["said"], [0, .3, .4, .45, .5, .55, .6, .7, 1])
    print(said.groupby(bins, observed=True).agg(n=("said", "size"), said=("said", "mean"), happened=("over", "mean")).round(3).to_string())
    for category, rows in said.groupby("category"):
        ours = -np.mean(np.where(rows["over"], np.log(rows["said"]), np.log(1 - rows["said"])))
        base = rows["over"].mean()
        flat = -np.mean(np.where(rows["over"], np.log(base), np.log(1 - base)))
        print(f"  {category}: log loss {ours:.4f} against {flat:.4f} for always saying {base:.0%}")


if __name__ == "__main__":
    main()

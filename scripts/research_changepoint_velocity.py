"""A series where real changes do happen: a pitcher's fastball speed.

Hot streaks at the plate turned out to be noise
(scripts/research_changepoint_hot_cold.py). A pitcher's velocity is the
opposite kind of number: steady from game to game, and when it moves - an
injury, a move to the bullpen, a mechanical fix - it stays moved. This is
the test of whether changepoint detection (changepoint.level_runs) earns
its keep where there is something to find.

Test: every pitcher's average four-seam fastball speed per game (games
with 8 or more of them), 2025 and 2026. Before each game from his fourth
on, his speed is predicted from his earlier games that season; the miss is
the average absolute error in mph. The hazard is chosen on 2025 and scored
on 2026.

RESULT (2026-11-20): here it wins. It beats every fixed window and the
season average, most where it flags a change (about one game in eight).
The gain over the best window is small, about a hundredth of a mph a
game, because velocity is steady most of the time.

  2025, by hazard: 0.03: 0.5627, 0.1: 0.5546, 0.2: 0.5539, 0.3: 0.5555, 0.5: 0.5611 -> cp_0.2
  scored on 2026 (7,390 pitcher-games, 514 pitchers), average miss in mph:
    season       0.6068
    last game    0.6490
    last 3       0.5701
    last 5       0.5642
    changepoint  0.5539
    changepoint beats the best window (last 5) by 0.0104 mph a game (t 4.6 across pitchers)
    a likely change (50%+) before 12.2% of pitcher-games; there the misses are: season 0.640, last 5 0.584, changepoint 0.561

Usage:
  python scripts/research_changepoint_velocity.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mlb_metrics import changepoint  # noqa: E402

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
MIN_PITCHES, FIRST_GAME = 8, 3
HAZARDS = (0.03, 0.1, 0.2, 0.3, 0.5)


def pitcher_games() -> pd.DataFrame:
    frames = []
    for path in sorted(glob.glob(os.path.join(RAW, "statcast_20*.parquet"))):
        d = pd.read_parquet(path, columns=["game_date", "game_pk", "pitcher", "pitch_type", "release_speed", "game_type"])
        d = d[(d["game_type"] == "R") & (d["pitch_type"] == "FF") & d["release_speed"].notna()]
        frames.append(d.groupby(["game_date", "game_pk", "pitcher"]).agg(n=("release_speed", "size"), mph=("release_speed", "mean")).reset_index())
    games = pd.concat(frames)
    games = games[games["n"] >= MIN_PITCHES]
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["season"] = games["game_date"].dt.year
    return games.sort_values(["pitcher", "game_date", "game_pk"]).reset_index(drop=True)


def predict(games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pitcher, g in games.groupby("pitcher", sort=False):
        x = g["mph"].to_numpy()
        if len(x) <= FIRST_GAME:
            continue
        before = lambda values: np.concatenate([[np.nan], values[:-1]])
        out = {"mph": x, "season": before(np.cumsum(x) / np.arange(1, len(x) + 1)),
               "last game": before(x), "last 3": before(pd.Series(x).rolling(3, min_periods=1).mean().to_numpy()),
               "last 5": before(pd.Series(x).rolling(5, min_periods=1).mean().to_numpy())}
        for hz in HAZARDS:
            runs = changepoint.level_runs(x, hazard=hz)
            out[f"cp_{hz}"] = before(runs["mean"])
            if hz == 0.1:
                out["changed"] = before(runs["changed"])
        rows.append(pd.DataFrame(out).assign(pitcher=pitcher).iloc[FIRST_GAME:])
    return pd.concat(rows, ignore_index=True)


def main():
    games = pitcher_games()
    scored = {season: predict(games[games["season"] == season]) for season in (2025, 2026)}
    miss = lambda frame, column: float((frame[column] - frame["mph"]).abs().mean())
    best = min((f"cp_{hz}" for hz in HAZARDS), key=lambda c: miss(scored[2025], c))
    print("2025, by hazard: " + ", ".join(f"{hz}: {miss(scored[2025], f'cp_{hz}'):.4f}" for hz in HAZARDS) + f" -> {best}")
    test = scored[2026]
    print(f"scored on 2026 ({len(test):,} pitcher-games, {test['pitcher'].nunique()} pitchers), average miss in mph:")
    for column in ("season", "last game", "last 3", "last 5", best):
        print(f"  {'changepoint' if column == best else column:12s} {miss(test, column):.4f}")
    diff = (test["last 5"] - test["mph"]).abs() - (test[best] - test["mph"]).abs()
    by = diff.groupby(test["pitcher"]).mean()
    print(f"  changepoint beats the best window (last 5) by {diff.mean():.4f} mph a game (t {by.mean() / (by.std(ddof=1) / np.sqrt(len(by))):.1f} across pitchers)")
    flagged = test["changed"] >= 0.5
    print(f"  a likely change (50%+) before {flagged.mean():.1%} of pitcher-games; there the misses are: "
          + ", ".join(f"{'changepoint' if c == best else c} {miss(test[flagged], c):.3f}" for c in ("season", "last 5", best)))


if __name__ == "__main__":
    main()

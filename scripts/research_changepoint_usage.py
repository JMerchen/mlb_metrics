"""Does changepoint detection (src/mlb_metrics/changepoint.py) predict who
gets the ball next game better than the fixed windows the simulator uses?

The simulator's shares (nfl_sim.RECENT_WEIGHT) put 70% of the weight on a
player's last four active games and 30% on his season for carries, 50/50
for targets. Changepoint detection has no window: it weighs every possible
"his current role began g games ago" by how well it explains his games.

Test: 2016-2026, each team's games from its fifth of a season on. Every
active player's share of the game's carries (or targets) is predicted from
his earlier active games that season; predictions are scaled to sum to 1
over the active players, and the miss is half the summed absolute error
(0 = perfect). The changepoint settings (hazard: the prior chance a role
changes in any game; temper: how many independent chances one carry is
worth) are chosen on 2016-2020 and scored on 2021-2026.

RESULT (2026-11-20): no better than the blend, and a touch worse for
carries. The settings that looked best on 2016-2020 (0.2321 against the
blend's 0.2399) did not hold up on 2021-2026 (0.2181 against 0.2155). A
back's role does change, but four games already catch it about as fast as
the data allows. In the games where the detector saw a change coming the
blend missed by more (carries 0.226 against 0.205 elsewhere) - and so did
the detector. The simulator keeps its blend.


  carries: chosen on 2016-2020: hazard 0.05, temper 1.0 (miss 0.2321; the grid ran 0.2321-0.2448)
    2016-20 grid (rows hazard, columns temper):
             0.2     0.4     0.7     1.0     1.5
    0.05  0.2448  0.2361  0.2323  0.2321  0.2340
    0.10  0.2421  0.2347  0.2322  0.2326  0.2343
    0.20  0.2408  0.2348  0.2334  0.2339  0.2350
    0.30  0.2416  0.2366  0.2353  0.2353  0.2360
    (the simulator's blend on the same seasons: 0.2399)
    scored on 2021-2026 (2239 team-games):
      the simulator's blend  miss 0.2155; with a player back or new 0.2264; everyone settled 0.1948; seasons 0.200-0.238
      season to date         miss 0.2195; with a player back or new 0.2300; everyone settled 0.1998; seasons 0.205-0.240
      changepoint            miss 0.2181; with a player back or new 0.2290; everyone settled 0.1974; seasons 0.200-0.239
      changepoint better by -0.0026 a team-game (t -1.9), better in 51% of team-games

  targets: chosen on 2016-2020: hazard 0.05, temper 0.7 (miss 0.2717; the grid ran 0.2717-0.2791)
    2016-20 grid (rows hazard, columns temper):
             0.2     0.4     0.7     1.0     1.5
    0.05  0.2763  0.2732  0.2717  0.2729  0.2768
    0.10  0.2747  0.2721  0.2719  0.2738  0.2782
    0.20  0.2738  0.2722  0.2731  0.2753  0.2791
    0.30  0.2743  0.2734  0.2743  0.2761  0.2789
    (the simulator's blend on the same seasons: 0.2750)
    scored on 2021-2026 (2250 team-games):
      the simulator's blend  miss 0.2564; with a player back or new 0.2586; everyone settled 0.2387; seasons 0.248-0.263
      season to date         miss 0.2579; with a player back or new 0.2601; everyone settled 0.2406; seasons 0.249-0.263
      changepoint            miss 0.2565; with a player back or new 0.2587; everyone settled 0.2389; seasons 0.249-0.261
      changepoint better by -0.0001 a team-game (t -0.3), better in 51% of team-games

Usage:
  python scripts/research_changepoint_usage.py
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mlb_metrics import changepoint, config, nfl_sim  # noqa: E402

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", config.NFL_RAW_DATA_DIR)
SEASONS = range(2016, 2027)
TUNE_THROUGH = 2020
HAZARDS = (0.05, 0.1, 0.2, 0.3)
TEMPERS = (0.2, 0.4, 0.7, 1.0, 1.5)
FIRST_GAME = 4          # predict from a team's fifth game on


def team_series(season: int):
    """Yields (stat, team, weeks, counts[week x player], active[week x player])."""
    pbp = pd.read_parquet(os.path.join(RAW, f"pbp_{season}.parquet"), columns=[
        "week", "season_type", "posteam", "play_type", "sack", "rusher_player_id", "receiver_player_id", "passer_player_id"])
    pbp = pbp[(pbp["season_type"] == "REG") & pbp["posteam"].notna()]
    rosters = pd.read_parquet(os.path.join(RAW, f"rosters_weekly_{season}.parquet"), columns=["week", "team", "gsis_id", "status"])
    listed = rosters[rosters["status"] == "ACT"]
    on_list = set(zip(listed["team"], listed["week"], listed["gsis_id"]))
    passers = set(pbp["passer_player_id"].dropna())
    runs = pbp[(pbp["play_type"] == "run") & pbp["rusher_player_id"].notna() & ~pbp["rusher_player_id"].isin(passers)]
    targets = pbp[(pbp["play_type"] == "pass") & (pbp["sack"] != 1) & pbp["receiver_player_id"].notna()]
    for stat, plays, column in (("carries", runs, "rusher_player_id"), ("targets", targets, "receiver_player_id")):
        for team, rows in plays.groupby("posteam"):
            counts = rows.pivot_table(index="week", columns=column, values="play_type", aggfunc="count", fill_value=0).sort_index()
            weeks = list(counts.index)
            active = np.array([[(team, w, p) in on_list or counts.at[w, p] > 0 for p in counts.columns] for w in weeks])
            yield stat, team, weeks, counts.to_numpy(dtype=float), active


def predictions(stat: str, counts: np.ndarray, active: np.ndarray) -> dict:
    """{method: array[week x player]} - each player's predicted share of a
    week's touches, made from his earlier active games (NaN before any)."""
    games, players = counts.shape
    totals = counts.sum(axis=1)
    recent_weight = nfl_sim.RECENT_WEIGHT[stat]
    out = {"blend": np.full(counts.shape, np.nan), "season": np.full(counts.shape, np.nan)}
    out.update({(h, t): np.full(counts.shape, np.nan) for h in HAZARDS for t in TEMPERS})
    for p in range(players):
        mine = np.flatnonzero(active[:, p])
        if mine.size < 2:
            continue
        k, n = counts[mine, p], totals[mine]
        season = np.cumsum(k) / np.maximum(1, np.cumsum(n))
        last = np.array([k[max(0, i - nfl_sim.PLAYER_GAMES + 1): i + 1].sum() / max(1, n[max(0, i - nfl_sim.PLAYER_GAMES + 1): i + 1].sum())
                         for i in range(mine.size)])
        after = {"blend": recent_weight * last + (1 - recent_weight) * season, "season": season}
        for h in HAZARDS:
            for t in TEMPERS:
                after[(h, t)] = changepoint.rate_runs(k, n, hazard=h, temper=t, prior_mean=0.1)["mean"]
        # The estimate after his i-th active game is the prediction for his next.
        for method, values in after.items():
            out[method][mine[1:], p] = values[:-1]
    return out


def score(season: int) -> list:
    rows = []
    for stat, team, weeks, counts, active in team_series(season):
        made = predictions(stat, counts, active)
        totals = counts.sum(axis=1)
        for g in range(FIRST_GAME, len(weeks)):
            playing = active[g]
            if totals[g] == 0 or counts[g, playing].sum() == 0:
                continue
            actual = counts[g, playing] / counts[g, playing].sum()
            # A player back from missed games, or new to the role: someone
            # active now who was not in every one of the team's last four.
            returning = bool((active[g] & ~active[max(0, g - 4): g].all(axis=0) & (counts[:g].sum(axis=0) > 0)).any())
            for method, values in made.items():
                share = np.nan_to_num(values[g, playing], nan=0.0)
                total = share.sum()
                share = share / total if total > 0 else share
                rows.append((season, stat, method, returning, float(np.abs(share - actual).sum() / 2)))
    return rows


def main():
    rows = []
    for season in SEASONS:
        rows += score(season)
        print(season, file=sys.stderr)
    d = pd.DataFrame(rows, columns=["season", "stat", "method", "returning", "miss"])
    for stat, s in d.groupby("stat"):
        tune, test = s[s["season"] <= TUNE_THROUGH], s[s["season"] > TUNE_THROUGH]
        grid = tune[tune["method"].map(lambda m: isinstance(m, tuple))].groupby("method")["miss"].mean()
        best = grid.idxmin()
        print(f"\n{stat}: chosen on 2016-{TUNE_THROUGH}: hazard {best[0]}, temper {best[1]} "
              f"(miss {grid.min():.4f}; the grid ran {grid.min():.4f}-{grid.max():.4f})")
        print("  2016-20 grid (rows hazard, columns temper):")
        table = grid.copy()
        table.index = pd.MultiIndex.from_tuples(table.index)
        print("  " + table.unstack().round(4).to_string().replace("\n", "\n  "))
        blend_tune = tune.loc[tune["method"] == "blend", "miss"].mean()
        print(f"  (the simulator's blend on the same seasons: {blend_tune:.4f})")
        print(f"  scored on {TUNE_THROUGH + 1}-2026 ({test[test['method'] == 'blend'].shape[0]} team-games):")
        for label, method in (("the simulator's blend", "blend"), ("season to date", "season"), ("changepoint", best)):
            m = test[test["method"].map(lambda x: x == method)]
            by = m.groupby("season")["miss"].mean()
            print(f"    {label:22s} miss {m['miss'].mean():.4f}; with a player back or new {m.loc[m['returning'], 'miss'].mean():.4f}; "
                  f"everyone settled {m.loc[~m['returning'], 'miss'].mean():.4f}; seasons {by.min():.3f}-{by.max():.3f}")
        a = test[test["method"].map(lambda x: x == best)]["miss"].to_numpy()
        b = test[test["method"] == "blend"]["miss"].to_numpy()
        diff = b - a
        print(f"    changepoint better by {diff.mean():.4f} a team-game (t {diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))):.1f}), "
              f"better in {np.mean(diff > 0):.0%} of team-games")


if __name__ == "__main__":
    main()

"""Walk-forward backtest of the nfl_prop_projections volume x efficiency
model against the two baselines it would replace.

The question this answers is narrow and falsifiable: does projecting a
player's stat line from usage share and shrunk per-play efficiency
predict what he actually does better than the board's current approach?
Three methods are replayed over the same weeks, the same players, and
the same no-lookahead history:

  naive      - the player's own recency-blended per-game rate, with no
               opponent adjustment at all. The honest floor: any
               matchup model that cannot beat this is adding noise.
  v3_ratio   - what the live board does today. The same per-game rate,
               multiplied by the clipped per-GAME opponent ratio from
               nfl_player_props.compute_position_defense_rolling_rates.
               Called through the real functions, not reimplemented.
  projection - nfl_prop_projections.project_player_stats: usage share x
               team volume x shrunk per-play efficiency x per-PLAY
               opponent multiplier.

Scoring is MAE and RMSE against the actual stat, plus the directional
hit rate that the board's Over/Under call actually turns on: taking the
player's own blended rate as a stand-in for the betting line (there are
no book lines in this repo - see nfl_prop_projections' docstring), how
often does each method call the right side of it? Directional accuracy
is the metric that matters for a prop; MAE is reported because a method
can be directionally lucky while being numerically useless.

No-lookahead is enforced the same way nfl_game_picks_backtest.py does
it: for target week W of season S, history is every regular-season week
strictly before (S, W), including all of S-1. A player's actual result
in week W is read only to score him, never to build a feature.

Usage:
    PYTHONPATH=src python3 scripts/backtest_nfl_prop_projections.py
    PYTHONPATH=src python3 scripts/backtest_nfl_prop_projections.py --sweep
"""

import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, nfl_matchup, nfl_player_props, nfl_prop_projections, nfl_rush_rec

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nfl")

# (category, actual weekly column, projection column, attempt column used
# to decide whether the player was actually involved that week).
CATEGORIES = [
    ("Receptions", "receptions", "projected_receptions", "targets"),
    ("Receiving Yards", "receiving_yards", "projected_receiving_yards", "targets"),
    ("Rushing Yards", "rushing_yards", "projected_rushing_yards", "carries"),
]

V3_STAT_COLUMNS = {
    "Receptions": ("receptions", "receptions_per_game"),
    "Receiving Yards": ("receiving_yards", "receiving_yards_per_game"),
    "Rushing Yards": ("rushing_yards", "rushing_yards_per_game"),
}


def load_weekly(seasons):
    frames = []
    for season in seasons:
        path = os.path.join(RAW_DIR, f"weekly_{season}.parquet")
        if os.path.exists(path):
            frames.append(pd.read_parquet(path))
    weekly = pd.concat(frames, ignore_index=True)
    return weekly[weekly["season_type"] == "REG"]


def baseline_projections(history: pd.DataFrame, opponents: pd.DataFrame) -> pd.DataFrame:
    """`naive` and `v3_ratio` for every skill player, via the real
    production functions so the comparison indicts the shipped code
    rather than a paraphrase of it."""
    skill = nfl_rush_rec.compute_skill_rolling_stats(history)
    ordered = history.sort_values(["season", "week"], ascending=False)
    latest_team = ordered.groupby("player_id", as_index=False).first()[["player_id", "team"]]
    players = skill.merge(latest_team, on="player_id", how="left").merge(opponents, on="team", how="left")

    frames = []
    for category, (stat_col, rate_col) in V3_STAT_COLUMNS.items():
        rows = players.rename(columns={rate_col: "naive"}).copy()
        rows["category"] = category

        defense = nfl_player_props.compute_position_defense_rolling_rates(history, stat_col)
        defense = defense.rename(columns={"team": "opponent", "allowed_per_game": "opponent_allowed_rate"})
        rows = rows.merge(
            defense[["opponent", "position", "opponent_allowed_rate"]], on=["opponent", "position"], how="left"
        )
        league_rate = defense.groupby("position")["opponent_allowed_rate"].mean()
        rows["league_rate"] = rows["position"].map(league_rate)
        rows["opponent_allowed_rate"] = rows["opponent_allowed_rate"].fillna(rows["league_rate"])
        ratio = nfl_matchup.compute_opponent_adjustment_ratio(
            rows["opponent_allowed_rate"], rows["league_rate"],
            config.NFL_PROP_MATCHUP_WEIGHT, clip=config.NFL_PROP_MATCHUP_CLIP,
        )
        rows["v3_ratio"] = rows["naive"] * ratio
        frames.append(rows[["player_id", "position", "team", "opponent", "category", "naive", "v3_ratio", "games"]])

    return pd.concat(frames, ignore_index=True)


def replay_week(weekly: pd.DataFrame, season: int, week: int, min_games: int) -> pd.DataFrame:
    """One row per (player, category) scored for this week, carrying all
    three methods' projections and the actual result."""
    is_before = (weekly["season"] < season) | ((weekly["season"] == season) & (weekly["week"] < week))
    history = weekly[is_before]
    actual_rows = weekly[(weekly["season"] == season) & (weekly["week"] == week)]
    if history.empty or actual_rows.empty:
        return pd.DataFrame()

    # The real matchups that week come from the actual rows' own
    # opponent_team, so no schedule file is needed and a bye is simply
    # an absent row.
    opponents = actual_rows[["team", "opponent_team"]].drop_duplicates().rename(
        columns={"opponent_team": "opponent"}
    )

    projections = nfl_prop_projections.project_player_stats(history, opponents)
    if projections.empty:
        return pd.DataFrame()
    baselines = baseline_projections(history, opponents)

    frames = []
    for category, actual_col, projection_col, attempt_col in CATEGORIES:
        actual = actual_rows[["player_id", actual_col, attempt_col]].rename(
            columns={actual_col: "actual", attempt_col: "attempts"}
        )
        merged = baselines[baselines["category"] == category].merge(
            projections[["player_id", projection_col]], on="player_id", how="inner"
        ).rename(columns={projection_col: "projection"}).merge(actual, on="player_id", how="inner")
        merged["season"] = season
        merged["week"] = week
        # A player who did not take a single target/carry that week was
        # almost always hurt or inactive; scoring a projection against
        # his forced 0 measures availability, not the model. The live
        # board has its own snap-share filter for the same reason.
        merged = merged[(merged["attempts"] > 0) & (merged["games"] >= min_games)]
        frames.append(merged)

    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def score(results: pd.DataFrame) -> pd.DataFrame:
    """MAE/RMSE per method, plus the directional hit rate against the
    player's own blended rate standing in for a betting line."""
    rows = []
    for category, group in results.groupby("category"):
        for method in ("naive", "v3_ratio", "projection"):
            error = group[method] - group["actual"]
            # Directional scoring only makes sense where the method
            # actually takes a side: if a method equals the line to the
            # cent it is abstaining, and pushes (actual == line) are
            # excluded the same way a sportsbook voids them.
            line = group["naive"]
            takes_side = (group[method] - line).abs() > 1e-9
            no_push = (group["actual"] - line).abs() > 1e-9
            live = takes_side & no_push
            if live.any():
                called_over = group.loc[live, method] > line[live]
                went_over = group.loc[live, "actual"] > line[live]
                hit_rate = (called_over == went_over).mean()
            else:
                hit_rate = np.nan
            rows.append({
                "category": category, "method": method, "n": len(group),
                "mae": error.abs().mean(), "rmse": np.sqrt((error ** 2).mean()),
                "n_directional": int(live.sum()), "hit_rate": hit_rate,
            })
    return pd.DataFrame(rows)


def paired_bootstrap(results: pd.DataFrame, challenger: str, incumbent: str, draws: int = 2000, seed: int = 0):
    """Bootstrap CI for the hit-rate and MAE difference between two
    methods, resampling PLAYERS rather than player-weeks.

    The naive per-row treatment overstates precision badly: the same
    receiver contributes ~15 rows across a season and his rows share
    whatever the model gets right or wrong about him, so they are
    nowhere near independent. Resampling whole players keeps each
    player's rows together and lets the CI reflect the roughly 300
    independent units actually present rather than the ~3,400 rows.
    Pairing (both methods scored on the same resampled players) removes
    the shared week-to-week variance that would otherwise swamp the
    difference being measured.
    """
    rng = np.random.default_rng(seed)
    line = results["naive"]
    frame = pd.DataFrame({
        "player_id": results["player_id"],
        "went_over": results["actual"] > line,
        "push": (results["actual"] - line).abs() <= 1e-9,
        "line": line,
        challenger: results[challenger],
        incumbent: results[incumbent],
        "actual": results["actual"],
    })

    def statistics(sample: pd.DataFrame):
        out = {}
        for method in (challenger, incumbent):
            live = (~sample["push"]) & ((sample[method] - sample["line"]).abs() > 1e-9)
            subset = sample[live]
            out[f"hit_{method}"] = (
                (subset[method] > subset["line"]) == subset["went_over"]
            ).mean() if len(subset) else np.nan
            out[f"mae_{method}"] = (sample[method] - sample["actual"]).abs().mean()
        return out

    players = frame["player_id"].unique()
    by_player = {pid: group for pid, group in frame.groupby("player_id")}
    observed = statistics(frame)

    hit_deltas, mae_deltas = [], []
    for _ in range(draws):
        picked = rng.choice(players, size=len(players), replace=True)
        sample = pd.concat([by_player[pid] for pid in picked], ignore_index=True)
        stats = statistics(sample)
        hit_deltas.append(stats[f"hit_{challenger}"] - stats[f"hit_{incumbent}"])
        mae_deltas.append(stats[f"mae_{challenger}"] - stats[f"mae_{incumbent}"])

    return {
        "n_players": len(players),
        "hit_delta": observed[f"hit_{challenger}"] - observed[f"hit_{incumbent}"],
        "hit_ci": tuple(np.percentile(hit_deltas, [2.5, 97.5])),
        "mae_delta": observed[f"mae_{challenger}"] - observed[f"mae_{incumbent}"],
        "mae_ci": tuple(np.percentile(mae_deltas, [2.5, 97.5])),
    }


def run(seasons, target_season, weeks, min_games):
    weekly = load_weekly(seasons)
    frames = [replay_week(weekly, target_season, week, min_games) for week in weeks]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise SystemExit("no weeks replayed - check the seasons/weeks requested")
    return pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", default=[2024, 2025])
    parser.add_argument("--target-season", type=int, default=2025)
    parser.add_argument("--first-week", type=int, default=4)
    parser.add_argument("--last-week", type=int, default=18)
    parser.add_argument("--min-games", type=int, default=config.NFL_PROP_MIN_GAMES)
    parser.add_argument("--sweep", action="store_true", help="grid-search the shrinkage priors and EPA blend")
    args = parser.parse_args()

    weeks = list(range(args.first_week, args.last_week + 1))

    if not args.sweep:
        results = run(args.seasons, args.target_season, weeks, args.min_games)
        summary = score(results)
        print(f"\nreplayed {args.target_season} weeks {weeks[0]}-{weeks[-1]}  "
              f"(efficiency_prior={config.NFL_PROP_EFFICIENCY_PRIOR_TARGETS}, "
              f"defense_prior={config.NFL_PROP_DEFENSE_PRIOR_TARGETS}, "
              f"epa_blend={config.NFL_PROP_DEFENSE_EPA_BLEND})\n")
        print(summary.round(4).to_string(index=False))
        return

    grid = list(itertools.product([20, 40, 60, 100], [30, 60, 120], [0.0, 0.35, 0.7]))
    rows = []
    for efficiency_prior, defense_prior, epa_blend in grid:
        config.NFL_PROP_EFFICIENCY_PRIOR_TARGETS = efficiency_prior
        config.NFL_PROP_DEFENSE_PRIOR_TARGETS = defense_prior
        config.NFL_PROP_DEFENSE_EPA_BLEND = epa_blend
        results = run(args.seasons, args.target_season, weeks, args.min_games)
        summary = score(results)
        projection_only = summary[summary["method"] == "projection"]
        rows.append({
            "efficiency_prior": efficiency_prior, "defense_prior": defense_prior, "epa_blend": epa_blend,
            "mae": projection_only["mae"].mean(), "hit_rate": projection_only["hit_rate"].mean(),
        })
        print(f"  eff={efficiency_prior:>3} def={defense_prior:>3} epa={epa_blend:<4} "
              f"mae={rows[-1]['mae']:.4f} hit={rows[-1]['hit_rate']:.4f}")

    sweep = pd.DataFrame(rows).sort_values("hit_rate", ascending=False)
    print("\nbest by mean directional hit rate across categories:")
    print(sweep.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

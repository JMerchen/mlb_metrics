"""Grid search the settings that BUILD the NFL game-pick model's features,
all at once (the model's own settings - C, tree depth, learning rate,
iterations - are already grid-searched every time
scripts/train_nfl_game_pick_model.py runs). Several of these were set by
hand and are labelled "placeholder, not yet backtested" in config.py:

    NFL_TEAM_STRENGTH_WINDOWS   recency windows (games back) and weights
    NFL_PYTHAGOREAN_EXPONENT    points-to-wins exponent
    NFL_CONFIDENCE_SOS_WEIGHT   strength of schedule's share of Confidence
    NFL_QB_WINDOWS              the QB rating's recency windows
    NFL_QB_CONTINUITY_WEIGHT    how much a starter change moves the QB term

Every combination of every value in OPTIONS is tried (6 x 4 x 4 x 3 x 4 =
1,152), because the settings interact - the best schedule weight can depend
on the recency windows. Each combination rebuilds the 2016-2025 history with
no lookahead (nfl_game_picks_backtest.build_multi_season_history, ~4
minutes), assembles the live feature set (nfl_game_picks.build_game_
features_disaggregated) and fits the live model type (standardized
LogisticRegression, its C re-picked from C_GRID for every combination)
walk-forward: each validation season 2018-2024 is predicted by a model
trained on the seasons before it. Each combination's 2025 predictions are
stored too, but only the validation winner's are compared with today's,
once, with a paired test - the guard against a combination that won
validation by luck out of 1,152.

Too long for one machine (~75 hours), so it runs as shards in parallel on
GitHub Actions (.github/workflows/nfl_grid_search.yml):

  shard      run every num_shards-th combination starting at shard,
             appending a row per combination to --out (rerunning skips the
             combinations already there)
  aggregate  combine the shard CSVs: ranking, each setting's effect, the
             strongest interactions, the 2025 check, and "can we train in
             epochs?" (a gradient-boosted model's trees are its epochs, so
             the validation log loss is traced against the number of trees)

Usage:
    python scripts/grid_search_nfl_game_picks.py --mode shard --shard 0 --num-shards 24 --out grid/shard_0.csv
    python scripts/grid_search_nfl_game_picks.py --mode aggregate --results-dir grid --out-dir data/backtest
"""

import argparse
import copy
import csv
import io
import itertools
import json
import os
import sys
import time
from contextlib import redirect_stdout
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from mlb_metrics import config, nfl_game_picks, nfl_game_picks_backtest as backtest

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "nfl")
SEASONS = list(range(2016, 2026))
VALIDATION_SEASONS = list(range(2018, 2025))
HOLDOUT_SEASON = 2025
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
# The history only needs play-by-play for points per drive; loading every
# column runs workers out of memory (with only these, one peaks near 4 GB).
PBP_COLUMNS = ["game_id", "play_id", "season", "week", "season_type", "posteam", "fixed_drive",
               "posteam_score", "posteam_score_post"]

BASELINE = {
    "NFL_TEAM_STRENGTH_WINDOWS": copy.deepcopy(config.NFL_TEAM_STRENGTH_WINDOWS),
    "NFL_PYTHAGOREAN_EXPONENT": config.NFL_PYTHAGOREAN_EXPONENT,
    "NFL_CONFIDENCE_SOS_WEIGHT": config.NFL_CONFIDENCE_SOS_WEIGHT,
    "NFL_QB_WINDOWS": copy.deepcopy(config.NFL_QB_WINDOWS),
    "NFL_QB_CONTINUITY_WEIGHT": config.NFL_QB_CONTINUITY_WEIGHT,
}

# Today's value first, so combination 0 is today's settings.
OPTIONS = {
    "NFL_TEAM_STRENGTH_WINDOWS": [
        BASELINE["NFL_TEAM_STRENGTH_WINDOWS"],
        [(2, 0.50), (6, 0.30), (None, 0.20)],
        [(4, 0.50), (8, 0.30), (None, 0.20)],
        [(3, 0.35), (7, 0.35), (None, 0.30)],
        [(3, 0.65), (7, 0.25), (None, 0.10)],
        [(5, 0.40), (10, 0.30), (None, 0.30)],
    ],
    "NFL_PYTHAGOREAN_EXPONENT": [BASELINE["NFL_PYTHAGOREAN_EXPONENT"], 2.0, 2.7, 3.0],
    "NFL_CONFIDENCE_SOS_WEIGHT": [BASELINE["NFL_CONFIDENCE_SOS_WEIGHT"], 0.0, 0.15, 0.5],
    "NFL_QB_WINDOWS": [
        BASELINE["NFL_QB_WINDOWS"],
        [(None, 0.40), (8, 0.30), (4, 0.30)],
        [(None, 0.10), (8, 0.30), (4, 0.60)],
    ],
    "NFL_QB_CONTINUITY_WEIGHT": [BASELINE["NFL_QB_CONTINUITY_WEIGHT"], 0.25, 0.75, 1.0],
}

RESULT_FIELDS = ["combo", *OPTIONS, "C", "val_log_loss", "val_brier", "val_accuracy", "val_games",
                 "holdout_log_loss", "holdout_accuracy", "holdout_predictions", "seconds"]

_DATA = {}


def all_combinations() -> list[dict]:
    """Every combination of OPTIONS, in a fixed order (0 = today's)."""
    names = list(OPTIONS)
    return [dict(zip(names, values)) for values in itertools.product(*OPTIONS.values())]


def _label(value) -> str:
    return json.dumps(value, default=str)


def _load(table: str) -> pd.DataFrame:
    columns = PBP_COLUMNS if table == "pbp" else None
    return pd.concat([pd.read_parquet(os.path.join(RAW_DIR, f"{table}_{season}.parquet"), columns=columns)
                      for season in SEASONS], ignore_index=True)


def _init():
    for table in ("schedules", "team_stats", "weekly", "snap_counts", "rosters_weekly", "pbp"):
        _DATA[table] = _load(table)


def build_rows(settings: dict) -> pd.DataFrame:
    """Rebuild the history under `settings` and return the feature rows."""
    for name, value in settings.items():
        setattr(config, name, value)
    with redirect_stdout(io.StringIO()):
        snapshots = backtest.build_multi_season_history(
            _DATA["schedules"], _DATA["team_stats"], _DATA["weekly"], _DATA["snap_counts"],
            _DATA["rosters_weekly"], _DATA["pbp"], SEASONS)
        return backtest.assemble_nfl_game_pick_log(snapshots, nfl_game_picks.build_game_features_disaggregated)


def _model(c):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))


def _xy(rows):
    X = nfl_game_picks.game_feature_matrix(rows, nfl_game_picks.DISAGGREGATED_FEATURE_COLUMNS)
    return X, rows["home_won"].astype(int)


def walk_forward(rows, c, seasons=VALIDATION_SEASONS) -> pd.DataFrame:
    """Per-game predictions for `seasons`, each from a model trained on the
    seasons before it."""
    out = []
    for season in seasons:
        train, test = rows[rows["season"] < season], rows[rows["season"] == season]
        if train.empty or test.empty:
            continue
        model = _model(c).fit(*_xy(train))
        X, y = _xy(test)
        out.append(pd.DataFrame({"game_id": test["game_id"].values, "season": season,
                                 "y": y.values, "p": model.predict_proba(X)[:, 1]}))
    return pd.concat(out, ignore_index=True)


def _ll(y, p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def evaluate(rows) -> dict:
    """Best C by validation log loss, its validation scores, and its 2025
    scores and per-game predictions."""
    best = None
    for c in C_GRID:
        pred = walk_forward(rows, c)
        ll = _ll(pred["y"], pred["p"]).mean()
        if best is None or ll < best["val_log_loss"]:
            best = {"C": c, "val_log_loss": ll, "val_brier": ((pred["p"] - pred["y"]) ** 2).mean(),
                    "val_accuracy": ((pred["p"] > 0.5) == (pred["y"] == 1)).mean(), "val_games": len(pred)}
    hold = walk_forward(rows, best["C"], [HOLDOUT_SEASON])
    best["holdout_log_loss"] = _ll(hold["y"], hold["p"]).mean()
    best["holdout_accuracy"] = ((hold["p"] > 0.5) == (hold["y"] == 1)).mean()
    best["holdout_predictions"] = json.dumps(
        {g: [int(y), round(float(p), 5)] for g, y, p in zip(hold["game_id"], hold["y"], hold["p"])})
    return best


def run_one(job) -> dict:
    index, settings = job
    start = time.time()
    result = evaluate(build_rows(settings))
    return {"combo": index, **{k: _label(v) for k, v in settings.items()}, **result,
            "seconds": round(time.time() - start)}


def run_shard(args):
    combos = list(enumerate(all_combinations()))[args.shard::args.num_shards]
    done = set()
    if os.path.exists(args.out):
        done = set(pd.read_csv(args.out, usecols=["combo"])["combo"])
    todo = [job for job in combos if job[0] not in done]
    print(f"Shard {args.shard}/{args.num_shards}: {len(combos)} combinations, {len(todo)} to run, "
          f"{args.workers} at a time", flush=True)
    if not todo:
        return
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new_file = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh, Pool(args.workers, initializer=_init) as pool:
        writer = csv.DictWriter(fh, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        for n, row in enumerate(pool.imap_unordered(run_one, todo), start=1):
            writer.writerow(row)
            fh.flush()
            print(f"  [{n}/{len(todo)}] combo {row['combo']}: validation log loss {row['val_log_loss']:.4f} "
                  f"(C={row['C']}, {row['seconds']}s)", flush=True)


def epochs_curve(rows) -> pd.DataFrame:
    """Validation log loss against the number of boosting iterations."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    stops = [10, 25, 50, 100, 200, 400]
    table = {}
    for lr in (0.03, 0.1):
        losses = {n: [] for n in stops}
        for season in VALIDATION_SEASONS:
            train, test = rows[rows["season"] < season], rows[rows["season"] == season]
            model = HistGradientBoostingClassifier(learning_rate=lr, max_iter=max(stops), max_depth=2,
                                                   min_samples_leaf=50, early_stopping=False).fit(*_xy(train))
            X, y = _xy(test)
            for i, proba in enumerate(model.staged_predict_proba(X), start=1):
                if i in losses:
                    losses[i].extend(_ll(y.values, proba[:, 1]))
        table[f"learning rate {lr}"] = {n: np.mean(v) for n, v in losses.items()}
    return pd.DataFrame(table).rename_axis("trees")


def _short(name: str) -> str:
    return name.removeprefix("NFL_").lower()


def aggregate(args):
    from scipy.stats import ttest_rel

    paths = [os.path.join(args.results_dir, f) for f in sorted(os.listdir(args.results_dir)) if f.endswith(".csv")]
    table = pd.concat([pd.read_csv(p, dtype={n: str for n in OPTIONS}) for p in paths], ignore_index=True)
    table = table.drop_duplicates("combo", keep="last").sort_values("val_log_loss").reset_index(drop=True)
    total = len(all_combinations())
    baseline = table[table["combo"] == 0]
    lines = ["# NFL game-pick settings grid search", "",
             f"{len(table)} of {total} combinations scored (validation: walk-forward {VALIDATION_SEASONS[0]}-"
             f"{VALIDATION_SEASONS[-1]}, lower log loss is better; C re-picked per combination).", ""]
    if baseline.empty:
        lines += ["Today's settings (combination 0) are missing from the results.", ""]
        base_ll = np.nan
    else:
        base_ll = baseline["val_log_loss"].iloc[0]
        rank = baseline.index[0] + 1
        lines += [f"Today's settings: validation log loss {base_ll:.4f}, rank {rank} of {len(table)}.", ""]

    lines += ["## Top 15", "", "| rank | combo | " + " | ".join(_short(n) for n in OPTIONS)
              + " | C | log loss | vs today | accuracy |",
              "|" + "---|" * (len(OPTIONS) + 6)]
    for i, r in table.head(15).iterrows():
        lines.append(f"| {i + 1} | {r['combo']} | " + " | ".join(str(r[n]) for n in OPTIONS)
                     + f" | {r['C']} | {r['val_log_loss']:.4f} | {r['val_log_loss'] - base_ll:+.4f} | {r['val_accuracy']:.3f} |")
    lines.append("")

    # Each setting's effect: its mean validation log loss by value, across
    # every combination of the others (a balanced grid, so these are fair).
    lines += ["## Each setting, averaged over every combination of the others", ""]
    grand = table["val_log_loss"].mean()
    for name in OPTIONS:
        by_value = table.groupby(name)["val_log_loss"].agg(["mean", "min", "size"]).sort_values("mean")
        lines += [f"**{name}** (today: `{_label(BASELINE[name])}`)", "", "| value | mean | best | vs grand mean | combos |",
                  "|---|---|---|---|---|"]
        for value, row in by_value.iterrows():
            lines.append(f"| `{value}` | {row['mean']:.4f} | {row['min']:.4f} | {row['mean'] - grand:+.4f} | {int(row['size'])} |")
        lines.append("")

    # Interactions: how much each pair's cell means depart from the sum of
    # the two settings' own effects (root mean square, in log loss).
    effects = {n: table.groupby(n)["val_log_loss"].mean() - grand for n in OPTIONS}
    pairs = []
    for a, b in itertools.combinations(OPTIONS, 2):
        cells = table.groupby([a, b])["val_log_loss"].mean()
        resid = [cells[(va, vb)] - grand - effects[a][va] - effects[b][vb] for va, vb in cells.index]
        pairs.append((np.sqrt(np.mean(np.square(resid))), a, b))
    lines += ["## Interactions (RMS departure of each pair's cells from additive, log loss)", "",
              "| pair | RMS |", "|---|---|"]
    for rms, a, b in sorted(pairs, reverse=True):
        lines.append(f"| {_short(a)} x {_short(b)} | {rms:.5f} |")
    lines.append("")

    # The untouched 2025 season: the validation winner against today's,
    # once, from the predictions each stored.
    best = table.iloc[0]
    lines += ["## 2025 holdout", ""]
    if baseline.empty:
        lines.append("Skipped: today's settings are missing.")
    elif best["combo"] == 0:
        lines.append("Today's settings won validation - nothing to check on 2025.")
    else:
        hb = json.loads(baseline["holdout_predictions"].iloc[0])
        hc = json.loads(best["holdout_predictions"])
        games = sorted(set(hb) & set(hc))
        yb, pb = np.array([hb[g][0] for g in games]), np.array([hb[g][1] for g in games])
        pc = np.array([hc[g][1] for g in games])
        lb, lc = _ll(yb, pb), _ll(yb, pc)
        changed = {n: best[n] for n in OPTIONS if best[n] != _label(BASELINE[n])}
        lines += [f"Best combination {best['combo']} changes: " + ", ".join(f"{n} -> `{v}`" for n, v in changed.items()), "",
                  f"{len(games)} games: log loss today {lb.mean():.4f}, best {lc.mean():.4f} "
                  f"({lc.mean() - lb.mean():+.4f}); accuracy {((pb > .5) == (yb == 1)).mean():.3f} vs "
                  f"{((pc > .5) == (yb == 1)).mean():.3f}; paired t-test p = {ttest_rel(lc, lb).pvalue:.3f}.", "",
                  "Ship it only if 2025 improves too (lower log loss, ideally p < 0.05)."]
    lines.append("")

    if not args.no_epochs:
        print("Rebuilding today's history for the epochs curve...", flush=True)
        _init()
        rows = build_rows(BASELINE)
        pred = walk_forward(rows, float(baseline["C"].iloc[0]) if not baseline.empty else 0.1)
        lr_ll = _ll(pred["y"], pred["p"]).mean()
        curve = epochs_curve(rows)
        lines += ["## Epochs: gradient boosting's validation log loss by number of trees (today's settings)", "",
                  "| trees | " + " | ".join(curve.columns) + " |", "|---|" + "---|" * len(curve.columns)]
        for n, r in curve.iterrows():
            lines.append(f"| {n} | " + " | ".join(f"{v:.4f}" for v in r) + " |")
        lines += ["", f"Logistic regression (today's model type) on the same seasons: {lr_ll:.4f}.", ""]

    os.makedirs(args.out_dir, exist_ok=True)
    table.drop(columns=["holdout_predictions"]).to_csv(os.path.join(args.out_dir, "nfl_grid_search_results.csv"), index=False)
    summary = "\n".join(lines)
    with open(os.path.join(args.out_dir, "nfl_grid_search_summary.md"), "w") as fh:
        fh.write(summary)
    print(summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["shard", "aggregate"], required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--out", default="grid/shard_0.csv")
    parser.add_argument("--results-dir", default="grid")
    parser.add_argument("--out-dir", default="data/backtest")
    parser.add_argument("--no-epochs", action="store_true")
    args = parser.parse_args()
    run_shard(args) if args.mode == "shard" else aggregate(args)


if __name__ == "__main__":
    main()

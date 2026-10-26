"""Grid search the settings that BUILD the NFL game-pick model's features
(the model's own settings - C, tree depth, learning rate, iterations - are
already grid-searched every time scripts/train_nfl_game_pick_model.py
runs). Several of these were set by hand and are labelled "placeholder,
not yet backtested" in config.py:

    NFL_TEAM_STRENGTH_WINDOWS   recency windows (games back) and weights
    NFL_PYTHAGOREAN_EXPONENT    points-to-wins exponent
    NFL_CONFIDENCE_SOS_WEIGHT   strength of schedule's share of Confidence
    NFL_QB_WINDOWS              the QB rating's recency windows
    NFL_QB_CONTINUITY_WEIGHT    how much a starter change moves the QB term

Each candidate value rebuilds the 2016-2025 history with no lookahead
(nfl_game_picks_backtest.build_multi_season_history, ~6.5 minutes each),
assembles the live feature set (nfl_game_picks.build_game_features_
disaggregated) and fits the live model type (standardized
LogisticRegression, with its own C picked from the grid) walk-forward:
each validation season 2018-2024 is predicted by a model trained on the
seasons before it. One setting moves at a time from today's values; then
every setting that beat today's values is combined and checked once on
the untouched 2025 season, with a paired test.

Also answers "can we train in epochs?": a gradient-boosted model's trees
are its epochs, so the validation log loss is traced against the number of
trees for two learning rates.

Usage:
    python scripts/grid_search_nfl_game_picks.py [--workers 3]
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from mlb_metrics import config, nfl_game_picks, nfl_game_picks_backtest as backtest

TRAINER_PATH = os.path.join(os.path.dirname(__file__), "train_nfl_game_pick_model.py")
VALIDATION_SEASONS = list(range(2018, 2025))
HOLDOUT_SEASON = 2025
C_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]

BASELINE = {
    "NFL_TEAM_STRENGTH_WINDOWS": copy.deepcopy(config.NFL_TEAM_STRENGTH_WINDOWS),
    "NFL_PYTHAGOREAN_EXPONENT": config.NFL_PYTHAGOREAN_EXPONENT,
    "NFL_CONFIDENCE_SOS_WEIGHT": config.NFL_CONFIDENCE_SOS_WEIGHT,
    "NFL_QB_WINDOWS": copy.deepcopy(config.NFL_QB_WINDOWS),
    "NFL_QB_CONTINUITY_WEIGHT": config.NFL_QB_CONTINUITY_WEIGHT,
}

CANDIDATES = {
    "NFL_TEAM_STRENGTH_WINDOWS": [
        [(2, 0.50), (6, 0.30), (None, 0.20)],
        [(4, 0.50), (8, 0.30), (None, 0.20)],
        [(3, 0.35), (7, 0.35), (None, 0.30)],
        [(3, 0.65), (7, 0.25), (None, 0.10)],
        [(5, 0.40), (10, 0.30), (None, 0.30)],
    ],
    "NFL_PYTHAGOREAN_EXPONENT": [2.0, 2.7, 3.0],
    "NFL_CONFIDENCE_SOS_WEIGHT": [0.0, 0.15, 0.5],
    "NFL_QB_WINDOWS": [
        [(None, 0.40), (8, 0.30), (4, 0.30)],
        [(None, 0.10), (8, 0.30), (4, 0.60)],
    ],
    "NFL_QB_CONTINUITY_WEIGHT": [0.25, 0.75, 1.0],
}

_DATA = {}


def _key(settings):
    return hashlib.md5(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()[:10]


def _init():
    spec = importlib.util.spec_from_file_location("trainer", TRAINER_PATH)
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    for table in ("schedules", "team_stats", "weekly", "snap_counts", "rosters_weekly", "pbp"):
        _DATA[table] = trainer._load_all(table)
    _DATA["seasons"] = trainer.SEASONS


def build_rows(job):
    """Rebuild the history under `settings` and return the feature rows
    (cached to `cache_dir`)."""
    settings, cache_dir = job
    path = os.path.join(cache_dir, f"{_key(settings)}.parquet")
    if os.path.exists(path):
        return path
    for name, value in settings.items():
        setattr(config, name, value)
    snapshots = backtest.build_multi_season_history(
        _DATA["schedules"], _DATA["team_stats"], _DATA["weekly"], _DATA["snap_counts"],
        _DATA["rosters_weekly"], _DATA["pbp"], _DATA["seasons"])
    rows = backtest.assemble_nfl_game_pick_log(snapshots, nfl_game_picks.build_game_features_disaggregated)
    rows.to_parquet(path)
    return path


def _model(c):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))


def _xy(rows):
    X = nfl_game_picks.game_feature_matrix(rows, nfl_game_picks.DISAGGREGATED_FEATURE_COLUMNS)
    return X, rows["home_won"].astype(int)


def walk_forward(rows, c, seasons=VALIDATION_SEASONS):
    """Per-game predictions for `seasons`, each from a model trained on the
    seasons before it."""
    from sklearn.metrics import log_loss  # noqa: F401

    out = []
    for season in seasons:
        train, test = rows[rows["season"] < season], rows[rows["season"] == season]
        if train.empty or test.empty:
            continue
        model = _model(c).fit(*_xy(train))
        p = model.predict_proba(_xy(test)[0])[:, 1]
        out.append(pd.DataFrame({"game_id": test["game_id"].values, "season": season, "y": _xy(test)[1].values, "p": p}))
    return pd.concat(out, ignore_index=True)


def _ll(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def score(rows):
    """Best C and its validation log loss / Brier / accuracy."""
    best = None
    for c in C_GRID:
        pred = walk_forward(rows, c)
        ll = _ll(pred["y"], pred["p"]).mean()
        if best is None or ll < best["log_loss"]:
            best = {"C": c, "log_loss": ll, "brier": ((pred["p"] - pred["y"]) ** 2).mean(),
                    "accuracy": ((pred["p"] > 0.5) == (pred["y"] == 1)).mean(), "games": len(pred)}
    return best


def epochs_curve(rows):
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
    return pd.DataFrame(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cache-dir", default="data/backtest/nfl_grid_cache")
    args = parser.parse_args()
    os.makedirs(args.cache_dir, exist_ok=True)

    configs = [("baseline", BASELINE)]
    for name, values in CANDIDATES.items():
        for value in values:
            settings = copy.deepcopy(BASELINE)
            settings[name] = value
            configs.append((f"{name}={value}", settings))
    print(f"Round 1: {len(configs)} settings, {args.workers} at a time...", flush=True)
    with Pool(args.workers, initializer=_init) as pool:
        paths = pool.map(build_rows, [(s, args.cache_dir) for _, s in configs])

    results = []
    for (label, settings), path in zip(configs, paths):
        r = score(pd.read_parquet(path))
        results.append({"setting": label, **r})
        print(f"  {label}: log loss {r['log_loss']:.4f} (C={r['C']})", flush=True)
    table = pd.DataFrame(results).set_index("setting")
    base = table.loc["baseline", "log_loss"]
    pd.set_option("display.width", 200)
    print("\nRound 1 (validation 2018-2024, lower log loss is better):")
    print(table.assign(vs_baseline=table["log_loss"] - base).round(4).to_string())

    # Round 2: every setting's best value, where it beat today's.
    combined = copy.deepcopy(BASELINE)
    for name, values in CANDIDATES.items():
        options = [(table.loc[f"{name}={v}", "log_loss"], v) for v in values]
        best_ll, best_value = min(options, key=lambda o: o[0])
        if best_ll < base:
            combined[name] = best_value
    changed = {k: v for k, v in combined.items() if v != BASELINE[k]}
    print(f"\nRound 2: combining {changed or 'nothing (no setting beat today)'}")
    if not changed:
        return
    with Pool(1, initializer=_init) as pool:
        combined_path = pool.map(build_rows, [(combined, args.cache_dir)])[0]
    combo = score(pd.read_parquet(combined_path))
    print(f"  combined: validation log loss {combo['log_loss']:.4f} vs baseline {base:.4f}")

    # Untouched 2025: train on everything before it.
    from scipy.stats import ttest_rel

    base_rows, combo_rows = pd.read_parquet(paths[0]), pd.read_parquet(combined_path)
    hb = walk_forward(base_rows, table.loc["baseline", "C"], [HOLDOUT_SEASON])
    hc = walk_forward(combo_rows, combo["C"], [HOLDOUT_SEASON]).set_index("game_id").loc[hb["game_id"]].reset_index()
    lb, lc = _ll(hb["y"], hb["p"]), _ll(hc["y"], hc["p"])
    print(f"\n2025 holdout ({len(hb)} games): log loss baseline {lb.mean():.4f}, combined {lc.mean():.4f}; "
          f"accuracy {((hb.p > .5) == (hb.y == 1)).mean():.3f} vs {((hc.p > .5) == (hc.y == 1)).mean():.3f}; "
          f"paired p = {ttest_rel(lc, lb).pvalue:.3f}")

    print("\nEpochs: validation log loss by number of boosting trees (today's settings):")
    print(epochs_curve(base_rows).round(4).to_string())
    print(f"(logistic regression, today's model type: {base:.4f})")


if __name__ == "__main__":
    main()

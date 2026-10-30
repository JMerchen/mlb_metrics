"""Backtest: do Stuff+ and Decision+ improve the hit-probability model and
the MLB game model?

Adds two point-in-time features to the hitter hit log
(data/predictions/hitter_hit_log.csv, one row per hitter per game day with
whether he got a hit):

    Decision_RV       the hitter's decision runs per 100 pitches through the
                      day before (2025 and 2026 so far), pulled toward 0
                      with pitch_metrics.DECISION_SHRINK pitches
    Opp_Stuff_RV      the opposing starter's average Stuff run value per
                      pitch through the day before, pulled toward the
                      league with pitch_metrics.STUFF_SHRINK pitches
                      (lower = better stuff)

Nothing after the day leaks in: the league models (Stuff per pitch group,
the ball-in-play value, Decision's swing and run-value models) are fitted
on 2025 alone and applied to 2026 pitches (2025 pitches are graded by
models cross-fitted by pitcher within 2025), and each day's grades use only
earlier pitches.

Then it runs scripts/train_hitter_hit_model.py's own model selection
(walk-forward grid search over logistic regression and gradient boosting,
the same untouched final holdout) twice - the production feature list, and
the list plus the two new features - and prints both holdout scores and
each new feature's univariate significance.

Second, the game model (mlb_game_model: run differential, starter K-BB,
bullpen xwOBA): each game gets the starters' Stuff difference and the two
lineups' average Decision (the nine who started, each through the day
before), and the model's own monthly walk-forward (fit on everything
before the month, history from 2021 - earlier games carry 0, no
information) predicts every 2026 month with and without them.

Usage:
    python scripts/backtest_pitch_metrics.py [--skip-hits] [--skip-games]
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, dfs_ml, ml_models  # noqa: E402
from mlb_metrics import pitch_metrics as pm  # noqa: E402

HERE = os.path.dirname(__file__)
NEW_FEATURES = ["Decision_RV", "Opp_Stuff_RV"]
EXTRA_COLUMNS = ["home_team", "away_team", "inning_topbot", "at_bat_number", "pitch_number"]


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load(raw_dir: str) -> pd.DataFrame:
    builder = _load_script("build_pitch_metrics")
    builder.COLUMNS = builder.COLUMNS + [c for c in EXTRA_COLUMNS if c not in builder.COLUMNS]
    return builder.load_pitches(raw_dir, 2026)


def stuff_point_in_time(p: pd.DataFrame) -> np.ndarray:
    """Each pitch's Stuff run value from models that never saw its season's
    future: 2025 pitches cross-fitted by pitcher within 2025, 2026 pitches
    from models fitted on all of 2025."""
    f = pm.stuff_features(p)
    is25 = (p["game_date"].dt.year == 2025).to_numpy()
    y = np.full(len(p), np.nan)
    y[is25] = pm.expected_run_value(p[is25])
    pred = np.full(len(p), np.nan)
    pred[is25] = pm.stuff_predictions(p[is25].assign(delta_run_exp=y[is25]), f[is25])
    group = p["pitch_type"].map(pm.PITCH_GROUP).to_numpy()
    X = f[pm.STUFF_FEATURES].to_numpy(dtype=float)
    for g in ("FB", "BR", "OS"):
        train = is25 & (group == g) & np.isfinite(y)
        test = ~is25 & (group == g)
        if train.sum() and test.sum():
            pred[test] = pm._model("regressor").fit(X[train], y[train]).predict(X[test])
    return pred


def decision_point_in_time(p: pd.DataFrame) -> pd.DataFrame:
    frame = pm.decision_frame(p)
    models = pm.decision_models(frame[frame["game_date"].dt.year == 2025])
    return pm.decision_values(frame, models)


def _running_before(daily: pd.DataFrame, key: str) -> pd.DataFrame:
    """daily [key, date, total, n] -> the same with each day's totals
    replaced by everything strictly before that day."""
    daily = daily.sort_values([key, "date"])
    for col in ("total", "n"):
        daily[col] = daily.groupby(key)[col].cumsum() - daily[col]
    return daily


def opposing_starters(p: pd.DataFrame) -> pd.DataFrame:
    """[date, team, opp_starter]: each team's opponent's starting pitcher
    that day (the first game of a doubleheader)."""
    first = p.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    top = first["inning_topbot"] == "Top"
    first = first.assign(field_team=np.where(top, first["home_team"], first["away_team"]),
                         bat_team=np.where(top, first["away_team"], first["home_team"]))
    starts = first.drop_duplicates(["game_pk", "field_team"])
    out = starts.assign(date=starts["game_date"].dt.strftime("%Y-%m-%d"))[["date", "game_pk", "bat_team", "pitcher"]]
    out = out.sort_values("game_pk").drop_duplicates(["date", "bat_team"])
    return out.rename(columns={"bat_team": "team", "pitcher": "opp_starter"})[["date", "team", "opp_starter"]]


def add_features(log: pd.DataFrame, p: pd.DataFrame, stuff_pred: np.ndarray, values: pd.DataFrame) -> pd.DataFrame:
    log = log.copy()
    # Decision runs per 100 pitches through the day before.
    v = values.assign(date=values["game_date"].dt.strftime("%Y-%m-%d"))
    daily = v.groupby(["batter", "date"]).agg(total=("value", "sum"), n=("value", "size")).reset_index()
    before = _running_before(daily, "batter")
    before["Decision_RV"] = 100 * before["total"] / (before["n"] + pm.DECISION_SHRINK)
    log["key_mlbam"] = log["key_mlbam"].astype(int)
    log = log.merge(before.rename(columns={"batter": "key_mlbam"})[["key_mlbam", "date", "Decision_RV"]],
                    on=["key_mlbam", "date"], how="left")
    # The opposing starter's Stuff through the day before.
    q = p.assign(pred=stuff_pred, date=p["game_date"].dt.strftime("%Y-%m-%d"))
    q = q[np.isfinite(q["pred"])]
    league = q.loc[q["game_date"].dt.year == 2025, "pred"].mean()
    daily = q.groupby(["pitcher", "date"]).agg(total=("pred", "sum"), n=("pred", "size")).reset_index()
    before = _running_before(daily, "pitcher")
    before["Opp_Stuff_RV"] = (before["total"] + pm.STUFF_SHRINK * league) / (before["n"] + pm.STUFF_SHRINK) - league
    log = log.merge(opposing_starters(p), on=["date", "team"], how="left")
    log = log.merge(before.rename(columns={"pitcher": "opp_starter"})[["opp_starter", "date", "Opp_Stuff_RV"]],
                    on=["opp_starter", "date"], how="left")
    return log


def evaluate(trainer, rows: pd.DataFrame, features: list[str], label: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """The production model's spec (the trainer's last pick: gradient
    boosting, depth 3, learning rate 0.03, 50-row leaves, isotonic) fit on
    the train pool and scored on the trainer's final holdout dates - the
    full grid search runs for hours with the new columns."""
    from sklearn.ensemble import HistGradientBoostingClassifier

    dfs_ml.HITTER_FEATURE_COLUMNS = features
    train_pool, holdout, _ = trainer._split_holdout(rows, config.ML_FINAL_HOLDOUT_DATES)
    X_train = dfs_ml.hitter_feature_matrix(train_pool)
    model = ml_models.fit_calibrated(
        HistGradientBoostingClassifier(learning_rate=0.03, max_depth=3, min_samples_leaf=50, random_state=0),
        X_train, train_pool["Got_Hit"], train_pool["date"], "isotonic",
        config.ML_WALK_FORWARD_MIN_TRAIN_DATES_HITTER, config.ML_WALK_FORWARD_TEST_BLOCK_DATES_HITTER,
    )
    probs = model.predict_proba(dfs_ml.hitter_feature_matrix(holdout))[:, 1]
    result = ml_models.evaluate_classifier_predictions(holdout["Got_Hit"], probs)
    print(f"  {label}: holdout log_loss={result['log_loss']:.5f}, brier={result['brier_score']:.5f}, "
          f"roc_auc={result['roc_auc']:.4f} (n={result['n']})")
    return result, holdout["Got_Hit"].to_numpy(dtype=float), probs


def lineups(p: pd.DataFrame) -> pd.DataFrame:
    """[game_pk, side (home/away), batter]: the nine each side started."""
    first = p.sort_values(["game_pk", "at_bat_number", "pitch_number"]).drop_duplicates(["game_pk", "batter"])
    first = first.assign(side=np.where(first["inning_topbot"] == "Top", "away", "home"))
    return first.groupby(["game_pk", "side"]).head(9)[["game_pk", "side", "batter", "game_date"]]


def game_model_backtest(p: pd.DataFrame, stuff_pred: np.ndarray, values: pd.DataFrame) -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import brier_score_loss, log_loss

    from mlb_metrics import data, mlb_game_model

    games, pa = mlb_game_model.prepare_history(data.load_game_model_history("data/raw", tuple(range(2021, 2027))))
    feats = mlb_game_model.build_features(games, games, pa)
    feats["home_win"] = games["home_win"].to_numpy()
    feats["game_pk"] = games["game_pk"].to_numpy()

    # Starters' Stuff through the day before (lower run value = better).
    q = p.assign(pred=stuff_pred, date=p["game_date"].dt.normalize())
    q = q[np.isfinite(q["pred"])]
    league = q.loc[q["game_date"].dt.year == 2025, "pred"].mean()
    daily = q.groupby(["pitcher", "date"]).agg(total=("pred", "sum"), n=("pred", "size")).reset_index()
    sp = _running_before(daily, "pitcher")
    sp["stuff"] = (sp["total"] + pm.STUFF_SHRINK * league) / (sp["n"] + pm.STUFF_SHRINK) - league
    for side in ("home", "away"):
        feats = feats.merge(sp.rename(columns={"pitcher": f"{side}_sp", "date": "game_date", "stuff": f"{side}_stuff"})
                            [[f"{side}_sp", "game_date", f"{side}_stuff"]], on=[f"{side}_sp", "game_date"], how="left")
    feats["sp_stuff_diff"] = (feats["away_stuff"] - feats["home_stuff"]).fillna(0) * 1000

    # Lineups' average Decision through the day before.
    v = values.assign(date=values["game_date"].dt.normalize())
    daily = v.groupby(["batter", "date"]).agg(total=("value", "sum"), n=("value", "size")).reset_index()
    before = _running_before(daily, "batter")
    before["decision"] = 100 * before["total"] / (before["n"] + pm.DECISION_SHRINK)
    lu = lineups(p).assign(date=lambda x: x["game_date"].dt.normalize())
    lu = lu.merge(before[["batter", "date", "decision"]], on=["batter", "date"], how="left")
    team = lu.groupby(["game_pk", "side"])["decision"].mean().unstack("side")
    feats = feats.merge(team.rename(columns={"home": "home_decision", "away": "away_decision"}), left_on="game_pk",
                        right_index=True, how="left")
    feats["lineup_decision_diff"] = (feats["home_decision"] - feats["away_decision"]).fillna(0)
    covered = (feats["game_date"].dt.year == 2026)
    print(f"\n=== Game model (2026 games {covered.sum():,}; with both new features "
          f"{(covered & feats['home_stuff'].notna() & feats['home_decision'].notna()).sum():,}) ===")

    feats["month"] = feats["game_date"].dt.to_period("M")
    sets = {
        "base (rd, sp K-BB, pen)": mlb_game_model.FEATURE_COLUMNS,
        "+ starter Stuff": mlb_game_model.FEATURE_COLUMNS + ["sp_stuff_diff"],
        "+ lineup Decision": mlb_game_model.FEATURE_COLUMNS + ["lineup_decision_diff"],
        "+ both": mlb_game_model.FEATURE_COLUMNS + ["sp_stuff_diff", "lineup_decision_diff"],
    }
    preds = {name: [] for name in sets}
    ys = []
    for month in sorted(feats.loc[covered, "month"].unique()):
        train = feats[feats["game_date"] < month.start_time]
        test = feats[feats["month"] == month]
        ys.append(test["home_win"].to_numpy())
        for name, cols in sets.items():
            model = LogisticRegression(C=1.0).fit(train[cols], train["home_win"])
            preds[name].append(model.predict_proba(test[cols])[:, 1])
    y = np.concatenate(ys)
    base = np.concatenate(preds["base (rd, sp K-BB, pen)"])
    rng = np.random.default_rng(0)
    for name in sets:
        pr = np.concatenate(preds[name])
        diffs = []
        for _ in range(500):
            i = rng.integers(0, len(y), len(y))
            diffs.append(brier_score_loss(y[i], pr[i]) - brier_score_loss(y[i], base[i]))
        lo, hi = np.percentile(diffs, [2.5, 97.5])
        print(f"  {name:26s} Brier {brier_score_loss(y, pr):.5f}, log loss {log_loss(y, pr):.5f}; "
              f"vs base {brier_score_loss(y, pr) - brier_score_loss(y, base):+.5f} (95% CI {lo:+.5f} to {hi:+.5f})")
    final = LogisticRegression(C=1.0).fit(feats[sets["+ both"]], feats["home_win"])
    print("  coefficients (+ both):", dict(zip(sets["+ both"], np.round(final.coef_[0], 4))))


def main():
    raw_dir = "data/raw"
    skip_hits = "--skip-hits" in sys.argv
    skip_games = "--skip-games" in sys.argv
    p = load(raw_dir)
    print(f"{len(p):,} pitches")
    stuff_pred = stuff_point_in_time(p)
    values = decision_point_in_time(p)
    if not skip_games:
        game_model_backtest(p, stuff_pred, values)
    if skip_hits:
        return
    log = pd.read_csv("data/predictions/hitter_hit_log.csv")
    rows = add_features(log, p, stuff_pred, values)
    covered = rows[NEW_FEATURES].notna().mean()
    print(f"hit log rows {len(rows):,}; coverage: " + ", ".join(f"{k} {v:.1%}" for k, v in covered.items()))
    trainer = _load_script("train_hitter_hit_model")
    print("\n=== Univariate significance of the new features (standardized Logit) ===")
    import statsmodels.api as sm

    for feature in NEW_FEATURES:
        x = rows[feature].fillna(0).astype(float)
        z = (x - x.mean()) / x.std()
        fit = sm.Logit(rows["Got_Hit"].astype(float), sm.add_constant(z)).fit(disp=0)
        print(f"  {feature}: coef {fit.params.iloc[1]:+.4f} per SD, p = {fit.pvalues.iloc[1]:.2g}")
    base = list(dfs_ml.HITTER_FEATURE_COLUMNS)
    print("\n=== Holdout comparison (the production model's spec, final holdout dates) ===")
    before, y, p0 = evaluate(trainer, rows, base, "production features")
    after, _, p1 = evaluate(trainer, rows, base + NEW_FEATURES, "plus Decision_RV, Opp_Stuff_RV")
    dfs_ml.HITTER_FEATURE_COLUMNS = base
    rng = np.random.default_rng(0)
    diffs = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        diffs.append(np.mean((p1[i] - y[i]) ** 2) - np.mean((p0[i] - y[i]) ** 2))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"\nRESULT: log loss {before['log_loss']:.5f} -> {after['log_loss']:.5f} "
          f"({after['log_loss'] - before['log_loss']:+.5f}); brier {before['brier_score']:.5f} -> "
          f"{after['brier_score']:.5f} ({after['brier_score'] - before['brier_score']:+.5f}, 95% CI {lo:+.5f} to {hi:+.5f})")


if __name__ == "__main__":
    main()

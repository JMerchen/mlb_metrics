"""Does the MLB game-pick model actually DISCRIMINATE between games, and
can its famously narrow probability spread be widened honestly?

This exists because of a real, specific user question (2026-09-17): the
live model's own logged probabilities have a spread roughly a third of
the real market's (std 0.042 vs 0.130), which mechanically manufactures
apparent "edge" on whichever side the market prices as unlikely - the
direct cause of a real, logged -16.8% ROI across 99 real advised bets,
96% of them underdogs (see config.GAME_PICK_MARKET_DISAGREEMENT_THRESHOLD's
own comment block). The obvious-looking fix is to stretch the
probabilities. This backtest exists to check whether that is actually
the right fix, honestly, either way.

The key distinction it measures, which accuracy alone cannot show, is
the standard Murphy decomposition of a real Brier score:

    Brier = Reliability - Resolution + Uncertainty

- RESOLUTION (higher is better) is real discrimination: does the model
  actually separate games it gets right from games it gets wrong?
- RELIABILITY (lower is better) is real calibration: when it says 55%,
  does 55% really happen?

A probability spread can only honestly be as wide as the model's own
real resolution supports. A model with little resolution that hugs the
base rate is CORRECTLY narrow - stretching it converts its one real
strength (calibration) into overconfidence, and produces exactly the
false-edge bets above. So "widen the spread" is only a real fix if there
is real, unexpressed resolution to widen INTO.

Replay is no-lookahead by construction, reusing
game_picks_backtest.derive_historical_schedule_games and
pipeline.compute_outputs: each replayed date's metrics are computed from
persisted Statcast strictly BEFORE that date, and graded against that
date's real final scores. This is the same "what would the code I'd ship
TODAY have produced" question
game_picks_backtest.reconstruct_historical_game_picks_from_persisted
answers, but it keeps each game's raw FEATURES rather than only the
final probability, so candidate models can be evaluated offline without
paying for the replay again (it takes ~30-60 real minutes for two real
seasons - use --cache).

Requires network on first run (pipeline.compute_outputs looks up the
Chadwick player register, same as the live pipeline), so run it from a
GitHub Actions runner or any unrestricted machine; afterwards --cache
makes re-analysis instant and offline.

Usage:
    python scripts/backtest_game_pick_discrimination.py --seasons 2025 2026
    python scripts/backtest_game_pick_discrimination.py --cache /tmp/gp.parquet
"""

import argparse
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, "src")

from mlb_metrics import config, game_picks, game_picks_backtest, matchup, ml_models, pipeline  # noqa: E402

# A real model needs some real history before a walk-forward fit means
# anything - this many real games must precede the first real prediction.
MIN_TRAIN_GAMES = 800


def extract_features(seasons, raw_dir: str = "data/raw") -> pd.DataFrame:
    """Replay each real date and keep per-game FEATURES + the real
    outcome. No lookahead: every date's metrics come from persisted
    Statcast strictly before it."""
    from mlb_metrics import data

    frames = []
    for season in seasons:
        persisted = data.load_persisted_statcast(raw_dir, season)
        if persisted is None:
            print(f"  {season}: no persisted Statcast - skipped")
            continue
        schedule_games = game_picks_backtest.derive_historical_schedule_games(persisted)
        dates = sorted(schedule_games["date"].unique())
        print(f"  {season}: replaying {len(dates)} real dates ({len(schedule_games)} real games)...")

        t0 = time.time()
        for i, date in enumerate(dates):
            history = persisted[persisted["game_date"] < date]
            todays = schedule_games[schedule_games["date"] == date]
            if history.empty or todays.empty:
                continue
            try:
                outputs = pipeline.compute_outputs(history)
                feats = game_picks.build_game_features(outputs["confidence"], outputs["pave"], todays)
                probs = game_picks.compute_game_win_probabilities(
                    outputs["confidence"], outputs["pave"], todays
                )
            except Exception as exc:
                print(f"    {date}: skipped ({exc})")
                continue
            merged = feats.merge(probs[["game_pk", "home_win_probability"]], on="game_pk", how="left").merge(
                todays[["game_pk", "home_score", "away_score"]], on="game_pk", how="left"
            )
            merged["season"] = season
            frames.append(merged)
            if (i + 1) % 25 == 0:
                print(f"    {i+1}/{len(dates)} dates, {time.time()-t0:.0f}s", flush=True)

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["home_won"] = (out["home_score"] > out["away_score"]).astype(int)
    return out.sort_values(["season", "date"]).reset_index(drop=True)


def _ratings(df: pd.DataFrame):
    """Rebuild the exact home/away ratings compute_game_win_probabilities
    forms internally, from the stored real features - confirmed to
    reproduce its own real output exactly (max abs diff 0.0), so a
    candidate can be compared against the real live model rather than an
    approximation of it."""
    home_faces = game_picks._blend_pitching_quality(
        matchup.clip_and_blend_pitching_quality(df["away_starter_pave_plus"], df["away_bullpen_pave_plus"]),
        matchup.clip_and_blend_pitching_quality(df["away_starter_power_a_plus"], df["away_bullpen_power_a_plus"]),
    )
    away_faces = game_picks._blend_pitching_quality(
        matchup.clip_and_blend_pitching_quality(df["home_starter_pave_plus"], df["home_bullpen_pave_plus"]),
        matchup.clip_and_blend_pitching_quality(df["home_starter_power_a_plus"], df["home_bullpen_power_a_plus"]),
    )
    return (df["home_composite"] * home_faces).to_numpy(), (df["away_composite"] * away_faces).to_numpy()


def probability_with_home_field(df: pd.DataFrame, home_field_weight: float) -> np.ndarray:
    """The real ratio formula with an additive home-field term - the MLB
    analog of config.NFL_HOME_FIELD_ADVANTAGE_WEIGHT, which MLB has never
    had."""
    home, away = _ratings(df)
    floor = config.GAME_PICK_RATING_FLOOR
    home = np.clip(home + home_field_weight, floor, None)
    away = np.clip(away, floor, None)
    return home / (home + away)


def murphy_decomposition(p: np.ndarray, y: np.ndarray, bins: int = 10) -> dict:
    """Brier = Reliability - Resolution + Uncertainty, over `bins` real
    equal-count bins. See this module's own docstring for why resolution
    is the number that actually decides whether a wider spread is honest."""
    binned = pd.qcut(pd.Series(p), bins, duplicates="drop")
    base = y.mean()
    reliability = resolution = 0.0
    for b in binned.unique():
        m = (binned == b).to_numpy()
        # A real degenerate forecast (every game given the same real
        # probability) leaves empty bins behind - taking a mean over one
        # yields NaN and silently poisons the whole decomposition, which
        # is exactly the "constant forecast" case this function most needs
        # to report honestly (resolution 0, not NaN).
        if not m.any():
            continue
        reliability += m.sum() * (p[m].mean() - y[m].mean()) ** 2
        resolution += m.sum() * (y[m].mean() - base) ** 2
    return {
        "reliability": reliability / len(y),
        "resolution": resolution / len(y),
        "uncertainty": base * (1 - base),
    }


def score(p: np.ndarray, y: np.ndarray, label: str, rng=None) -> dict:
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    auc = roc_auc_score(y, p)
    row = {
        "model": label,
        "AUC": auc,
        "Brier": brier_score_loss(y, p),
        "logloss": log_loss(y, np.clip(p, 1e-6, 1 - 1e-6)),
        "mean_p": p.mean(),
        "bias": p.mean() - y.mean(),
        "std": p.std(),
        **murphy_decomposition(p, y),
    }
    if rng is not None:
        boot = []
        for _ in range(1500):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() == y[i].max():
                continue
            boot.append(roc_auc_score(y[i], p[i]))
        row["AUC_lo"], row["AUC_hi"] = np.percentile(boot, [2.5, 97.5])
    return row


def walk_forward_logistic(df: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Expanding-window logistic fit on the same real features the live
    heuristic already uses - the "let a real model weigh them itself"
    candidate, the same one that worked for NFL
    (scripts/train_nfl_game_pick_model.py). Predicts each real date using
    only games strictly before it."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = df.reindex(columns=game_picks.GAME_PICK_FEATURE_COLUMNS).fillna(0).to_numpy()
    dates = df["date"].to_numpy()
    out = np.full(len(df), np.nan)
    for d in sorted(pd.unique(dates)):
        test, train = dates == d, dates < d
        if train.sum() < MIN_TRAIN_GAMES:
            continue
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        model.fit(X[train], y[train])
        out[test] = model.predict_proba(X[test])[:, 1]
    return out


def walk_forward_calibration(p: np.ndarray, y: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """Refit ml_models.fit_probability_calibration walk-forward - the
    honest way to judge a change that sits UPSTREAM of calibration, since
    the live calibrator would retrain against it rather than stay frozen
    on the old regime."""
    out = np.full(len(p), np.nan)
    for d in sorted(pd.unique(dates)):
        test, train = dates == d, dates < d
        if train.sum() < MIN_TRAIN_GAMES:
            continue
        model = ml_models.fit_probability_calibration(p[train], y[train])
        out[test] = np.clip(model.predict(p[test]), 1e-6, 1 - 1e-6)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2025, 2026])
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--cache", default=None, help="parquet path to reuse/write the replayed features")
    args = parser.parse_args()

    import os

    if args.cache and os.path.exists(args.cache):
        print(f"Loading cached real replay from {args.cache}")
        df = pd.read_parquet(args.cache)
    else:
        print(f"Replaying real seasons {args.seasons} (no lookahead)...")
        df = extract_features(args.seasons, args.raw_dir)
        if df.empty:
            print("No real replayed games - nothing to evaluate.")
            return
        if args.cache:
            df.to_parquet(args.cache, index=False)
            print(f"Cached real replay -> {args.cache}")

    df = df[df["home_win_probability"].notna()].copy().sort_values(["season", "date"]).reset_index(drop=True)
    y = df["home_won"].to_numpy()
    dates = df["date"].to_numpy()
    rng = np.random.default_rng(0)
    print(f"\n{len(df)} real games | real home win rate {y.mean():.4f}\n")

    rows = [score(df["home_win_probability"].to_numpy(), y, "live heuristic (raw, uncalibrated)", rng)]

    print("=== Real home-field-advantage sweep (MLB has no such term today) ===")
    print(f'{"HFA":>6} {"AUC":>8} {"Brier":>8} {"logloss":>9} {"bias":>9}')
    for hfa in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        p = probability_with_home_field(df, hfa)
        s = score(p, y, f"HFA={hfa:.2f}")
        print(f'{hfa:6.2f} {s["AUC"]:8.4f} {s["Brier"]:8.4f} {s["logloss"]:9.5f} {s["bias"]:+9.4f}')
        if hfa > 0:
            rows.append(s)

    print("\n=== Real candidate models (walk-forward, no lookahead) ===")
    logistic = walk_forward_logistic(df, y)
    ok = ~np.isnan(logistic)
    rows.append(score(logistic[ok], y[ok], "walk-forward logistic (same features)", rng))
    rows.append(score(df["home_win_probability"].to_numpy()[ok], y[ok], "live heuristic (same games as above)", rng))

    for hfa in (0.0, 0.15):
        cal = walk_forward_calibration(probability_with_home_field(df, hfa), y, dates)
        ok2 = ~np.isnan(cal)
        rows.append(score(cal[ok2], y[ok2], f"HFA={hfa:.2f} + walk-forward refit calibration", rng))

    out = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print()
    print(out.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    print("\n=== What this really means ===")
    base = out.iloc[0]
    print(
        f"Real resolution {base['resolution']:.5f} against real uncertainty {base['uncertainty']:.5f} - the live "
        f"model explains about {100*base['resolution']/base['uncertainty']:.1f}% of the real outcome variance it "
        f"could. A real probability spread is only honest up to the real resolution behind it, so widening this "
        f"model's own spread without first raising that number would manufacture confidence it has not earned."
    )


if __name__ == "__main__":
    main()

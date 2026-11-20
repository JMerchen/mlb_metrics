"""Should WAVE stop chasing hot and cold streaks? The full pick replay.

Until 2026-11-20 WAVE (hitters.compute_wave) blended a hitter's at-bat hit
rate over the season and his last 81, 30 and 10 days, with 60% of the
weight on the last 30 days (config.WAVE_WINDOWS). Changepoint detection
found no real changes in hitters' levels, and on its own that WAVE
predicted at-bats worse than a plain season average (scripts/research_changepoint_hot_cold.py). But WAVE
is not the pick: it is one input to the hit model, next to the pitcher,
the lineup spot and more. This replays the picks themselves.

Same replay as scripts/backtest_wave_at_bats.py: for every date each
variant's features are rebuilt from games strictly before it
(dfs_backtest._compute_date_outputs) and joined to whether each hitter got
a hit; walk-forward by month, the hit model (the shipped settings) is refit
on each variant's earlier months and scored on the next.

Variants (only WAVE changes unless noted):
  old            season / 81 / 30 / 10 days at 15 / 25 / 27.5 / 32.5%,
                 each window pulled to the league with 50 at-bats
  season 50      his season only, pulled with 50 at-bats
  season 200     his season only, pulled with 200
  season 400     his season only, pulled with 400
  season + 81    60% season, 40% last 81 days, pulled with 200
  all season     season 200, and every other number built on the same
                 windows (CAVE's rates among them) made season-only too

RESULT (2026-11-20): WAVE moved to "season 200" (config.WAVE_RATE_WINDOWS,
WAVE_SHRINKAGE_STRENGTH). It was better on the hit model's log loss in
both seasons (clearly in 2025, where the whole 95% range is below zero)
and ahead on seven of the eight pick measures; "season 400" looked best in 2026 and did not
repeat in 2025. The differences are small next to how much one fit of the
hit model differs from the next, which is why every number is an average
of 10 fits. Columns: model_top1/top2 = the hit model's top pick got a hit /
both of its top two did; live_top1/top2 = the same for the live rule's
picks; wave_top1 = the hitter with the day's highest WAVE got a hit.

  2026: 86 test days, 24,397 hitter-games; every number the average of 10 fits of the hit model
               log_loss   brier     auc  wave_auc  model_top1  model_top2  live_top1  live_top2  wave_top1  wave_top5_share
  variant
  old            0.6723  0.2397  0.5857    0.5500      0.7186      0.5326     0.6674     0.4884     0.6628           0.6279
  season 50      0.6723  0.2397  0.5861    0.5546      0.7209      0.5244     0.7140     0.4872     0.5814           0.6488
  season 200     0.6722  0.2397  0.5866    0.5535      0.7372      0.5384     0.7256     0.5081     0.6395           0.6744
  season 400     0.6720  0.2396  0.5873    0.5528      0.7477      0.5709     0.7279     0.5116     0.6512           0.6767
  season + 81    0.6722  0.2397  0.5867    0.5537      0.7233      0.5349     0.6884     0.5012     0.6163           0.6744
  all season     0.6722  0.2397  0.5864    0.5535      0.7419      0.5500     0.7326     0.5221     0.6395           0.6744
  How much one fit differs from another (standard deviation across fits), old WAVE:
    log_loss 0.0004, model_top1 0.0566, model_top2 0.0387, live_top1 0.0360, live_top2 0.0338
  The hit model's log loss against the old WAVE's, same hitter-games (negative is better; 95% by resampling days):
    season 50    -0.00002 (-0.00056 to +0.00046)
    season 200   -0.00022 (-0.00071 to +0.00028)
    season 400   -0.00038 (-0.00093 to +0.00012)
    season + 81  -0.00004 (-0.00052 to +0.00045)
    all season   -0.00021 (-0.00078 to +0.00035)
  Pick measures against the old WAVE, paired by day (difference in the share of days hit; t across days):
    season 50    model_top1 +0.002 (t +0.1), model_top2 -0.008 (t -0.4), live_top1 +0.047 (t +1.3), live_top2 -0.001 (t -0.0)
    season 200   model_top1 +0.019 (t +0.8), model_top2 +0.006 (t +0.2), live_top1 +0.058 (t +1.7), live_top2 +0.020 (t +0.7)
    season 400   model_top1 +0.029 (t +1.2), model_top2 +0.038 (t +1.5), live_top1 +0.060 (t +1.6), live_top2 +0.023 (t +0.8)
    season + 81  model_top1 +0.005 (t +0.2), model_top2 +0.002 (t +0.1), live_top1 +0.021 (t +0.7), live_top2 +0.013 (t +0.5)
    all season   model_top1 +0.023 (t +1.0), model_top2 +0.017 (t +0.7), live_top1 +0.065 (t +1.9), live_top2 +0.034 (t +1.1)

  2025: 85 test days, 23,734 hitter-games; every number the average of 10 fits of the hit model
              log_loss   brier     auc  wave_auc  model_top1  model_top2  live_top1  live_top2  wave_top1  wave_top5_share
  variant
  old           0.6712  0.2392  0.5828    0.5479      0.6824      0.4953     0.6976     0.5118     0.6000           0.6635
  season 200    0.6708  0.2390  0.5853    0.5553      0.7424      0.5329     0.7035     0.4941     0.6588           0.6471
  season 400    0.6710  0.2390  0.5845    0.5541      0.6941      0.5106     0.6800     0.4847     0.6706           0.6541
  How much one fit differs from another (standard deviation across fits), old WAVE:
    log_loss 0.0004, model_top1 0.0392, model_top2 0.0269, live_top1 0.0304, live_top2 0.0373
  The hit model's log loss against the old WAVE's, same hitter-games (negative is better; 95% by resampling days):
    season 200   -0.00078 (-0.00153 to -0.00004)
    season 400   -0.00043 (-0.00103 to +0.00023)
  Pick measures against the old WAVE, paired by day (difference in the share of days hit; t across days):
    season 200   model_top1 +0.060 (t +2.6), model_top2 +0.038 (t +1.4), live_top1 +0.006 (t +0.2), live_top2 -0.018 (t -0.5)
    season 400   model_top1 +0.012 (t +0.4), model_top2 +0.015 (t +0.5), live_top1 -0.018 (t -0.5), live_top2 -0.027 (t -0.9)

Usage:
  python scripts/backtest_wave_windows.py [--start 2026-05-01] [--end 2026-09-27] [--workers 3]
"""

import argparse
import os
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from mlb_metrics import config, data, dfs_backtest, dfs_ml, hitters, matchup

import backtest_wave_at_bats as replay

SEASON_ONLY = [(None, 1.0)]
# WAVE as it was until 2026-11-20: the recency windows, each pulled to the
# league with 50 at-bats.
OLD_WINDOWS = list(config.WAVE_WINDOWS)
# name -> (WAVE's windows, WAVE's shrinkage, also every other windowed number?)
VARIANTS = {
    "old": (OLD_WINDOWS, 50.0, False),
    "season 50": (SEASON_ONLY, 50.0, False),
    "season 200": (SEASON_ONLY, 200.0, False),
    "season 400": (SEASON_ONLY, 400.0, False),
    "season + 81": ([(None, 0.6), (81, 0.4)], 200.0, False),
    "all season": (SEASON_ONLY, 200.0, True),
}
_STATE = {}
_COMPUTE_WAVE = hitters.compute_wave


def _init(raw_dir, season):
    persisted = data.load_persisted_statcast(raw_dir, season)
    persisted = persisted[persisted["game_type"] == "R"]
    _STATE["persisted"] = persisted
    _STATE["schedule"] = dfs_backtest.derive_historical_team_schedule(persisted)
    # Names only label rows; the real register is a download this replay
    # does not need.
    ids = pd.unique(pd.concat([persisted["batter"], persisted["pitcher"]]).dropna()).astype(int)
    data._name_register_cache = pd.DataFrame({"key_mlbam": ids, "name_first": "", "name_last": ids.astype(str)})


def _use(windows, strength, everything):
    """Point WAVE (or everything built on the recency windows) at a variant."""
    config.WAVE_WINDOWS = windows if everything else OLD_WINDOWS

    def compute_wave(dt, *args, **kwargs):
        saved = config.WAVE_RATE_WINDOWS
        config.WAVE_RATE_WINDOWS = windows
        try:
            return _COMPUTE_WAVE(dt, *args, shrinkage_strength=kwargs.pop("shrinkage_strength", strength), **kwargs)
        finally:
            config.WAVE_RATE_WINDOWS = saved

    hitters.compute_wave = compute_wave


def _date_rows(date):
    persisted, schedule = _STATE["persisted"], _STATE["schedule"]
    day_events = persisted[persisted["game_date"] == date]
    got_hit = dfs_backtest.compute_actual_hitter_got_hit(data.completed_events(day_events, ["game_date", "batter", "events"]))
    frames = []
    for name, (windows, strength, everything) in VARIANTS.items():
        _use(windows, strength, everything)
        day = dfs_backtest._compute_date_outputs(persisted, schedule, date)
        if day is None:
            continue
        features = dfs_ml.build_hitter_features(
            day["outputs"]["wave"], day["outputs"]["pave"], day["outputs"]["confidence"],
            day["todays_schedule"], day["matchup_probability"],
        )
        scored = features.merge(got_hit, on="key_mlbam", how="inner")
        scored["date"] = date
        scored["variant"] = name
        frames.append(scored)
    return pd.concat(frames, ignore_index=True) if frames else None


def build_log(dates, raw_dir, season, workers):
    with Pool(workers, initializer=_init, initargs=(raw_dir, season)) as pool:
        parts = [part for part in pool.imap(_date_rows, dates) if part is not None]
    return pd.concat(parts, ignore_index=True)


MEASURES = ["model_top1", "model_top2", "live_top1", "live_top2", "wave_top1", "wave_top5_share"]


FITS = 10


def evaluate(log: pd.DataFrame):
    """Each variant's scores over the test months, averaged over FITS fits
    of the hit model.

    Why averaged: the shipped model stops early on a random tenth of its
    training rows, so two fits on the same data differ - and "did the top
    pick get a hit" turns on which of several near-equal hitters comes
    first. Scored twice on the same log, the old WAVE's top pick hit
    81.4% and then 76.7% of the 86 days. One fit cannot compare variants;
    the average over fits can."""
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    log = log.copy()
    log["date"] = pd.to_datetime(log["date"])
    log["month"] = log["date"].dt.to_period("M")
    months = sorted(log["month"].unique())
    rows, daily, losses, spread = [], {}, {}, {}
    for variant in VARIANTS:
        frame = log[log["variant"] == variant]
        if frame.empty:
            continue
        fits, day_fits, loss_fits = [], [], []
        for seed in range(FITS):
            tested = []
            for month in months[2:]:
                train, test = frame[frame["month"] < month], frame[frame["month"] == month].copy()
                if train.empty or test.empty:
                    continue
                model = replay._model().set_params(random_state=seed).fit(dfs_ml.hitter_feature_matrix(train), train["Got_Hit"].astype(int))
                test["Model_Hit_Probability"] = model.predict_proba(dfs_ml.hitter_feature_matrix(test))[:, 1]
                tested.append(test)
            tested = pd.concat(tested, ignore_index=True)
            tested["Matchup_Approach"] = matchup.compute_matchup_approach(tested["Approach"], tested["Matchup_Hit_Probability"])
            y, p = tested["Got_Hit"].astype(int), tested["Model_Hit_Probability"]
            by_day = tested.groupby("date")
            days = pd.DataFrame({
                "model_top1": by_day.apply(lambda d: replay._top_hits(d, "Model_Hit_Probability", 1)),
                "model_top2": by_day.apply(lambda d: replay._top_hits(d, "Model_Hit_Probability", 2)),
                "live_top1": by_day.apply(lambda d: replay._live_rule_hits(d, 1)),
                "live_top2": by_day.apply(lambda d: replay._live_rule_hits(d, 2)),
                # WAVE on its own: its favorite, and the share of its top five who hit.
                "wave_top1": by_day.apply(lambda d: replay._top_hits(d, "WAVE", 1)),
                "wave_top5_share": by_day.apply(lambda d: d.nlargest(5, "WAVE")["Got_Hit"].astype(bool).mean()),
            }).astype(float)
            day_fits.append(days)
            loss_fits.append(pd.Series((-(y * np.log(p) + (1 - y) * np.log(1 - p))).to_numpy(), index=pd.MultiIndex.from_frame(tested[["date", "key_mlbam"]])))
            fits.append({"log_loss": log_loss(y, p), "brier": brier_score_loss(y, p), "auc": roc_auc_score(y, p),
                         "wave_auc": roc_auc_score(y, tested["WAVE"]), **days.mean().to_dict()})
        fits = pd.DataFrame(fits)
        daily[variant] = sum(day_fits) / len(day_fits)              # each day: the share of fits whose pick hit
        mean_loss = sum(loss_fits) / len(loss_fits)
        losses[variant] = mean_loss.rename("loss").reset_index()
        spread[variant] = fits.std()
        rows.append({"variant": variant, "hitter_games": len(tested), "days": len(daily[variant]), **fits.mean().to_dict()})
    return pd.DataFrame(rows).set_index("variant"), daily, losses, pd.DataFrame(spread).T


def loss_against_old(losses: dict, variant: str, draws: int = 2000, seed: int = 0) -> tuple:
    """(mean, low, high): the hit model's log loss with `variant` minus
    with the old WAVE over the same hitter-games (negative = better),
    with a 95% interval from resampling days."""
    both = losses[variant].merge(losses["old"], on=["date", "key_mlbam"], suffixes=("_new", "_old"))
    by_day = (both["loss_new"] - both["loss_old"]).groupby(both["date"]).agg(["sum", "count"])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(by_day), (draws, len(by_day)))
    boot = by_day["sum"].to_numpy()[picks].sum(axis=1) / by_day["count"].to_numpy()[picks].sum(axis=1)
    return by_day["sum"].sum() / by_day["count"].sum(), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--season", type=int, default=config.SEASON_START.year)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--log", default="data/backtest/wave_windows_log.csv")
    parser.add_argument("--reuse-log", action="store_true", help="Score an existing --log instead of rebuilding it.")
    parser.add_argument("--variants", nargs="+", help="Only these variants (old is always kept).")
    args = parser.parse_args()
    if args.variants:
        for name in [n for n in VARIANTS if n != "old" and n not in args.variants]:
            del VARIANTS[name]

    if not args.reuse_log:
        _init(args.raw_dir, args.season)
        dates = sorted(_STATE["schedule"]["date"].unique())
        start = pd.Timestamp(args.start) if args.start else pd.Timestamp(f"{args.season}-05-01")
        end = pd.Timestamp(args.end) if args.end else pd.Timestamp(config.REGULAR_SEASON_END).replace(year=args.season)
        dates = [d for d in dates if start <= pd.Timestamp(d) <= end]
        _STATE.clear()
        print(f"Rebuilding features for {len(dates)} dates, {len(VARIANTS)} variants...", flush=True)
        log = build_log(dates, args.raw_dir, args.season, args.workers)
        os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
        log.to_csv(args.log, index=False)
    log = pd.read_csv(args.log)

    summary, daily, losses, spread = evaluate(log)
    pd.set_option("display.width", 200)
    print(f"{args.season}: {int(summary['days'].iloc[0])} test days, {int(summary['hitter_games'].iloc[0]):,} hitter-games; "
          f"every number the average of {FITS} fits of the hit model")
    print(summary.drop(columns=["hitter_games", "days"]).round(4).to_string())
    print("How much one fit differs from another (standard deviation across fits), old WAVE:")
    print("  " + ", ".join(f"{name} {value:.4f}" for name, value in spread.loc["old"].items() if name in ("log_loss", "model_top1", "model_top2", "live_top1", "live_top2")))
    print("The hit model's log loss against the old WAVE's, same hitter-games (negative is better; 95% by resampling days):")
    for variant in losses:
        if variant != "old":
            mean, low, high = loss_against_old(losses, variant)
            print(f"  {variant:12s} {mean:+.5f} ({low:+.5f} to {high:+.5f})")
    print("Pick measures against the old WAVE, paired by day (difference in the share of days hit; t across days):")
    for variant, days in daily.items():
        if variant == "old":
            continue
        both = days.join(daily["old"], lsuffix="_new", rsuffix="_old", how="inner")
        parts = []
        for column in ("model_top1", "model_top2", "live_top1", "live_top2"):
            diff = both[f"{column}_new"] - both[f"{column}_old"]
            t = diff.mean() / (diff.std(ddof=1) / np.sqrt(len(diff))) if diff.std(ddof=1) > 0 else 0.0
            parts.append(f"{column} {diff.mean():+.3f} (t {t:+.1f})")
        print(f"  {variant:12s} " + ", ".join(parts))

if __name__ == "__main__":
    main()

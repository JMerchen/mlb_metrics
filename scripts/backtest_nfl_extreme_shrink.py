"""Backtest: should an extreme NFL unit be pulled toward average less?

User suggestion (2026-11-01, after the Cowboys - the league's worst
defense - lost to the Cardinals in a shootout): "the further a team is away
from the middle, the less we help them. Pulling the Cowboys defense by .05
is too far. We're in week 8, that should be plenty of time without looking
back to last season."

The 0.05 is the predicted-totals rating (nfl_totals.wave_ratings: last 4,
8 and 17 games - crossing into last season - each pulled toward the league
by config.NFL_TOTALS_SHRINK_PLAYS = 300 pseudo-plays), shown on the picks
page as each team's defensive EPA: Dallas allowed +0.138 EPA a play
through week 7, shown as +0.080. The win-pick model's defensive_edge is a
recency blend z-scored across teams, so a pull applied to every team alike
changes nothing there - only pulling extremes less (stretching the tails)
or dropping last season can. Both models are tested:

Win picks (scripts/backtest_nfl_defense_carryover.py's replay, the live
logistic regression refit on earlier seasons, 2019-2026):
    live           as now
    season_only    this season's plain average from week 4 on (the live
                   blend before that)
    stretch_d_1.3  defense z-scores stretched: sign(z) |z|^1.3, rescaled
    stretch_d_1.6  the same, ^1.6
    stretch_od_1.3 offense and defense both, ^1.3

Predicted totals (scripts/backtest_nfl_totals.py's replay, refit on
earlier seasons, 2018-2025), for offense and defense EPA:
    live           300 pseudo-plays per window
    k150, k75      lighter pulls
    fade_0.05      the pull fades with distance from the league:
                   k / (1 + (gap / 0.05)^2) - at a 0.05 gap, half the pull
    fade_0.03      faster fade
    season_only    this season's games only once there are 4

RESULT (2026-11-01):
Win picks (1,972 games 2019-2026): no change. This season only is worse
(log loss +0.0011; extreme-defense games 0.6439 against 0.6426) - last
season's games still help at week 8. Stretching the extremes moves log
loss by 0.0003 or less (stretch_od_1.3 -0.00027, 95% CI -0.00094 to
+0.00036): within noise.
Predicted totals (2,127 games 2018-2025): yes. Teams scored 0.57 points
more than predicted against a defense 1.5+ sd below average - the pull
toward the league was too strong for the worst units. Fading it with
distance fixes most of that and lowers the miss on the total:
    live        MAE 10.645   bad-defense miss +0.57
    k75         -0.031 (95% CI -0.050 to -0.010), better in 6/8 seasons
    fade_0.05   -0.035 (-0.060 to -0.011), 6/8,     bad-defense miss +0.24
    fade_0.03   -0.038 (-0.064 to -0.012), 6/8,     bad-defense miss +0.21
    season_only -0.003                              bad-defense miss +0.70
(Paired bootstrap over games; same refit-on-earlier-seasons regression.)

Usage:
    python scripts/backtest_nfl_extreme_shrink.py
"""

import importlib.util
import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, ml_models, nfl_game_picks, nfl_team_strength, nfl_totals  # noqa: E402


def _load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(os.path.dirname(__file__), f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


carry = _load_script("backtest_nfl_defense_carryover")
totals_bt = _load_script("backtest_nfl_totals")


def _paired_ci(y, p, base, score, rng):
    diffs = []
    for _ in range(1000):
        i = rng.integers(0, len(y), len(y))
        diffs.append(score(y[i], p[i]) - score(y[i], base[i]))
    return np.percentile(diffs, [2.5, 97.5])


# --- Win picks ---

LIVE_SEASON_BLEND = nfl_team_strength._season_aware_blend


def _season_only_blend(stats_df, group_col, value_col, current_season, **kwargs):
    live = LIVE_SEASON_BLEND
    current = stats_df[stats_df["season"] == current_season]
    if value_col != "epa_allowed_adjusted" or current.groupby(group_col).size().min() < 3:
        return live(stats_df, group_col, value_col, current_season, **kwargs)
    out = current.groupby(group_col)[value_col].mean()
    out.index.name = group_col
    return out


def _stretch(master, cols, power):
    master = master.copy()
    for col in cols:
        z = (master[col] - master[col].mean()) / master[col].std()
        s = np.sign(z) * z.abs() ** power
        master[col] = 1 + (s - s.mean()) / s.std() * config.NFL_NORMALIZATION_Z_SCALE
    return master


def win_rows(snaps, variant):
    if variant == "live":
        return carry.feature_rows(snaps, None)
    if variant == "season_only":
        live = carry._blend
        carry._blend = lambda kind, k: _season_only_blend
        try:
            return carry.feature_rows(snaps, ("season_only", 0))
        finally:
            carry._blend = live
    cols = ["defensive_edge"] if variant.startswith("stretch_d_") else ["offensive_edge", "defensive_edge"]
    power = float(variant.rsplit("_", 1)[1])
    stretched = [{**s, "master": _stretch(s["master"], cols, power)} for s in snaps]
    return carry.feature_rows(stretched, None)


def win_picks():
    snaps = carry._snapshots()
    artifact = ml_models.load_model(config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH)
    c = artifact["model"].get_params()["C"] if artifact else 1.0
    variants = ["live", "season_only", "stretch_d_1.3", "stretch_d_1.6", "stretch_od_1.3"]
    live_rows = carry.feature_rows(snaps, None)
    z = config.NFL_NORMALIZATION_Z_SCALE
    extreme = set(live_rows.loc[((live_rows["home_defensive_edge"] - 1).abs() >= 1.5 * z)
                                | ((live_rows["away_defensive_edge"] - 1).abs() >= 1.5 * z), "game_id"])
    results = {v: carry.walk_forward(win_rows(snaps, v), c).set_index("game_id") for v in variants}
    base = results["live"]
    rng = np.random.default_rng(0)
    print(f"\n=== Win picks: walk-forward 2019-2026, {len(base)} games ({len(extreme & set(base.index))} with a defense 1.5+ sd out) ===")
    for v, r in results.items():
        r = r.loc[base.index]
        y, p, b = r["home_won"].to_numpy(), r["p"].to_numpy(), base["p"].to_numpy()
        lo, hi = _paired_ci(y, p, b, log_loss, rng)
        e = r.loc[r.index.isin(extreme)]
        print(f"  {v:15s} log loss {log_loss(y, p):.5f} ({log_loss(y, p) - log_loss(y, b):+.5f}, 95% CI {lo:+.5f} to {hi:+.5f})"
              f"  Brier {brier_score_loss(y, p):.5f} | extreme-defense games log loss {log_loss(e['home_won'], e['p']):.4f}")


# --- Predicted totals ---

def _totals_blend(kind, k, fade):
    def blend(rows, value, count, prior, pseudo, windows):
        if value != "epa_sum":
            return live_blend(rows, value, count, prior, pseudo, windows)
        if kind == "season_only" and "season" in rows:
            this = rows[rows["season"] == rows["season"].max()]
            if len(this) >= 4:
                rows = this
        total = 0.0
        for n_games, weight in windows:
            window = rows if n_games is None else rows.tail(n_games)
            n = window[count].sum()
            rate = window[value].sum() / n if n else prior
            kk = k if fade is None else k / (1 + ((rate - prior) / fade) ** 2)
            total += weight * (window[value].sum() + kk * prior) / (n + kk)
        return total
    return blend


live_blend = nfl_totals._blend
TOTALS_VARIANTS = {
    "live": ("plain", 300.0, None), "k150": ("plain", 150.0, None), "k75": ("plain", 75.0, None),
    "fade_0.05": ("plain", 300.0, 0.05), "fade_0.03": ("plain", 300.0, 0.03), "season_only": ("season_only", 300.0, None),
}


def totals():
    team_games, qb_games, schedules = totals_bt.load()
    out = {}
    for name, (kind, k, fade) in TOTALS_VARIANTS.items():
        nfl_totals._blend = _totals_blend(kind, k, fade)
        try:
            features = totals_bt.weekly_features(team_games, qb_games, schedules).dropna(
                subset=nfl_totals.TOTALS_FEATURE_COLUMNS + ["points"])
        finally:
            nfl_totals._blend = live_blend
        scored = []
        for season in totals_bt.SCORED_SEASONS:
            model = totals_bt.fit(features[features["season"] < season])
            test = features[features["season"] == season].copy()
            test["predicted"] = model.predict(sm.add_constant(test[nfl_totals.TOTALS_FEATURE_COLUMNS], has_constant="add"))
            scored.append(test)
        out[name] = pd.concat(scored)
    # Extreme defenses: by each team's season-to-date raw EPA allowed a play
    # (no pull), 1.5+ sd from that week's mean.
    base = out["live"]
    print(f"\n=== Predicted totals: refit season by season, 2018-2025 ===")
    for name, s in out.items():
        g = s.groupby("game_id").agg(pred=("predicted", "sum"), total=("points", "sum"), sides=("points", "size"))
        g = g[g["sides"] == 2]
        # Points scored against a defense rated far from average (its
        # opponent's row carries it as opp_def_epa).
        z = s.groupby(["season"])["opp_def_epa"].transform(lambda v: (v - v.mean()) / v.std())
        far = s[z.abs() >= 1.5]
        bias = (far["points"] - far["predicted"]).groupby(np.sign(z[z.abs() >= 1.5])).mean()
        print(f"  {name:12s} total MAE {(g['pred'] - g['total']).abs().mean():.3f}  team-points MAE {(s['points'] - s['predicted']).abs().mean():.3f}"
              f" | vs a defense 1.5+ sd out: scored minus predicted - good D {bias.get(-1.0, np.nan):+.2f}, bad D {bias.get(1.0, np.nan):+.2f} (n={len(far)})")


def main():
    win_picks()
    totals()


if __name__ == "__main__":
    main()

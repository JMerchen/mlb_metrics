"""Are a hitter's hot and cold streaks real changes, or noise?

The site's batting number, WAVE, believes in them: it puts 32.5% of its
weight on a hitter's last 10 days and 27.5% on his last 30
(config.WAVE_WINDOWS). Changepoint detection
(src/mlb_metrics/changepoint.py) asks the data instead: after every game
it weighs "his level changed g games ago" against "nothing changed", and
only moves off his season when a change is the better explanation.

Test: every hitter's at-bats, game by game, 2025 and 2026 (Statcast,
regular season). Before each game his chance of a hit in an at-bat is
predicted from his earlier games that season, four ways, and scored on the
at-bats he then had (log loss per at-bat, lower is better; games where he
already had 50 at-bats that season). Settings are chosen on 2025 and
scored on 2026.

  league       every hitter gets the league's rate
  season       his season so far, pulled toward the league
  WAVE         the site's windows (season, 81, 30, 10 days), as weighted
  changepoint  his level since the last change the data supports

RESULT (2026-11-20): the detector found no changes at all - it saw a
likely one before 0.00% of hitter-games - and its predictions are his
season average to four decimals. Hot and cold streaks, at the level of a
hit in an at-bat, are noise. WAVE, which chases them, predicts WORSE than
giving every hitter the league average, and a hitter WAVE had 67 points
over his season hit 3 points under it. As a daily pick on its own, WAVE's
favorite got a hit less often than an average hitter-game. (The live picks
are not WAVE alone - a model reads it with the pitcher, the lineup spot
and more - so this says WAVE is a poor input, not that the picks are that
bad. Whether a season-based number improves the live picks needs the
full replay, scripts/backtest_wave_at_bats.py.)

  chosen on 2025: season_400; cp_0.1_800
  changepoint grid on 2025 (rows hazard, columns prior strength):
             50       100      200      400      800
  0.002  0.55724  0.55658  0.55613  0.55593  0.55591
  0.010  0.55728  0.55658  0.55612  0.55593  0.55591
  0.030  0.55735  0.55657  0.55611  0.55592  0.55591
  0.100  0.55734  0.55651  0.55608  0.55591  0.55591
  scored on 2026 (39,443 hitter-games, 136,258 at-bats), log loss per at-bat:
    league rate for everyone           0.55368  (+0.00% against the league rate; plus is better)
    his season, pulled to the league   0.55340  (+0.05% against the league rate; plus is better)
    WAVE's windows                     0.55972  (-1.09% against the league rate; plus is better)
    changepoint                        0.55341  (+0.05% against the league rate; plus is better)
    his season beats WAVE: by 0.00632 an at-bat (t 24.0 across 522 hitters)
    changepoint beats his season: by -0.00001 an at-bat (t -0.5 across 522 hitters)
    WAVE 40+ points over his season (hot)   7,231 games: WAVE said 0.318, his season said 0.251, he hit 0.248
    within 15 points                       10,535 games: WAVE said 0.243, his season said 0.243, he hit 0.245
    WAVE 40+ points under (cold)            7,403 games: WAVE said 0.167, his season said 0.234, he hit 0.231
    daily picks by each number alone, 2026 (170 days; 58.3% of all hitter-games had a hit):
      WAVE         top 1 got a hit 58.2%, top 2 got a hit 56.0%, top 5 got a hit 57.6%
      his season   top 1 got a hit 64.1%, top 2 got a hit 66.1%, top 5 got a hit 66.9%
    the detector saw a likely change (50%+) before 0.00% of hitter-games

Usage:
  python scripts/research_changepoint_hot_cold.py
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mlb_metrics import changepoint, config, helpers  # noqa: E402

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
MIN_AT_BATS = 50
STRENGTHS = (50, 100, 200, 400, 800)
HAZARDS = (0.002, 0.01, 0.03, 0.1)


def batter_games() -> pd.DataFrame:
    """One row per hitter per game: at-bats and hits, oldest first."""
    frames = []
    for path in sorted(glob.glob(os.path.join(RAW, "statcast_20*.parquet"))):
        d = pd.read_parquet(path, columns=["game_date", "game_pk", "batter", "events", "game_type"])
        d = d[(d["game_type"] == "R") & d["events"].notna() & ~d["events"].isin(helpers.NON_AT_BAT_EVENTS)]
        d = d.assign(hit=helpers.is_hit(d["events"]).astype(int))
        frames.append(d.groupby(["game_date", "game_pk", "batter"]).agg(ab=("hit", "size"), h=("hit", "sum")).reset_index())
    games = pd.concat(frames)
    games["game_date"] = pd.to_datetime(games["game_date"])
    games["season"] = games["game_date"].dt.year
    return games.sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)


def wave_before(dates: np.ndarray, ab: np.ndarray, h: np.ndarray) -> np.ndarray:
    """WAVE's blend before each game: each window's raw rate over the days
    before it, weighted as config.WAVE_WINDOWS (a window with no at-bats
    drops out and the rest are reweighted)."""
    out = np.full(len(ab), np.nan)
    day = dates.astype("datetime64[D]").astype(int)
    for i in range(len(ab)):
        total = weight_sum = 0.0
        for days, weight in config.WAVE_WINDOWS:
            use = (day < day[i]) if days is None else (day < day[i]) & (day >= day[i] - days)
            n = ab[use].sum()
            if n > 0:
                total += weight * h[use].sum() / n
                weight_sum += weight
        if weight_sum > 0:
            out[i] = total / weight_sum
    return out


def predict(games: pd.DataFrame, league: float) -> pd.DataFrame:
    """Every method's prediction before each hitter-game of one season."""
    rows = []
    for batter, g in games.groupby("batter", sort=False):
        ab, h = g["ab"].to_numpy(dtype=float), g["h"].to_numpy(dtype=float)
        before_ab, before_h = np.cumsum(ab) - ab, np.cumsum(h) - h
        out = {"ab": ab, "h": h, "before_ab": before_ab, "league": np.full(len(ab), league),
               "wave": wave_before(g["game_date"].to_numpy(), ab, h)}
        for s in STRENGTHS:
            out[f"season_{s}"] = (before_h + s * league) / (before_ab + s)
            for hz in HAZARDS:
                runs = changepoint.rate_runs(h, ab, hazard=hz, prior_mean=league, prior_strength=s)
                out[f"cp_{hz}_{s}"] = np.concatenate([[league], runs["mean"][:-1]])
                if (hz, s) == (0.01, 200):
                    out["changed"] = np.concatenate([[0.0], runs["changed"][:-1]])
        rows.append(pd.DataFrame(out).assign(batter=batter, date=g["game_date"].to_numpy()))
    return pd.concat(rows, ignore_index=True)


def loss(frame: pd.DataFrame, column: str) -> float:
    """Log loss per at-bat."""
    p = frame[column].clip(0.01, 0.99)
    return float(-(frame["h"] * np.log(p) + (frame["ab"] - frame["h"]) * np.log(1 - p)).sum() / frame["ab"].sum())


def main():
    games = batter_games()
    scored = {}
    for season in (2025, 2026):
        g = games[games["season"] == season]
        earlier = games[games["season"] < season]
        league = earlier["h"].sum() / earlier["ab"].sum() if len(earlier) else g["h"].sum() / g["ab"].sum()
        made = predict(g, league)
        scored[season] = made[(made["before_ab"] >= MIN_AT_BATS) & made["wave"].notna()]
        print(f"{season}: {len(scored[season]):,} hitter-games, {int(scored[season]['ab'].sum()):,} at-bats", file=sys.stderr)
    tune, test = scored[2025], scored[2026]
    best_season = min((f"season_{s}" for s in STRENGTHS), key=lambda c: loss(tune, c))
    best_cp = min((f"cp_{hz}_{s}" for hz in HAZARDS for s in STRENGTHS), key=lambda c: loss(tune, c))
    print(f"chosen on 2025: {best_season}; {best_cp}")
    print("changepoint grid on 2025 (rows hazard, columns prior strength):")
    print(pd.DataFrame([[loss(tune, f"cp_{hz}_{s}") for s in STRENGTHS] for hz in HAZARDS], index=HAZARDS, columns=STRENGTHS).round(5).to_string())
    print(f"scored on 2026 ({len(test):,} hitter-games, {int(test['ab'].sum()):,} at-bats), log loss per at-bat:")
    base = loss(test, "league")
    for label, column in (("league rate for everyone", "league"), ("his season, pulled to the league", best_season),
                          ("WAVE's windows", "wave"), ("changepoint", best_cp)):
        value = loss(test, column)
        print(f"  {label:34s} {value:.5f}  ({100 * (base - value) / base:+.2f}% against the league rate; plus is better)")
    # Paired, by hitter-game: WAVE against his season.
    def game_loss(column):
        p = test[column].clip(0.01, 0.99)
        return -(test["h"] * np.log(p) + (test["ab"] - test["h"]) * np.log(1 - p))
    for label, a, b in (("his season beats WAVE", "wave", best_season), ("changepoint beats his season", best_season, best_cp)):
        diff = game_loss(a) - game_loss(b)
        by_batter = diff.groupby(test["batter"]).sum()
        t = by_batter.mean() / (by_batter.std(ddof=1) / np.sqrt(len(by_batter)))
        print(f"  {label}: by {diff.sum() / test['ab'].sum():.5f} an at-bat (t {t:.1f} across {len(by_batter)} hitters)")
    # How hot is "hot"? After WAVE is far above his season, what does he hit?
    gap = test["wave"] - test[best_season]
    for label, use in (("WAVE 40+ points over his season (hot)", gap >= 0.040), ("within 15 points", gap.abs() < 0.015),
                       ("WAVE 40+ points under (cold)", gap <= -0.040)):
        x = test[use]
        print(f"  {label:38s} {len(x):6,} games: WAVE said {np.average(x['wave'], weights=x['ab']):.3f}, his season said "
              f"{np.average(x[best_season], weights=x['ab']):.3f}, he hit {x['h'].sum() / x['ab'].sum():.3f}")
    # What a daily pick turns on: of the hitters playing each day, take the
    # one (or five) the number likes best. How often did he get a hit?
    print(f"  daily picks by each number alone, 2026 ({test['date'].nunique()} days; {(test['h'] > 0).mean():.1%} of all hitter-games had a hit):")
    for label, column in (("WAVE", "wave"), ("his season", best_season)):
        parts = []
        for count in (1, 2, 5):
            top = test.sort_values(column, ascending=False).groupby("date").head(count)
            parts.append(f"top {count} got a hit {(top['h'] > 0).mean():.1%}")
        print(f"    {label:12s} " + ", ".join(parts))
    flagged = test["changed"] >= 0.5
    print(f"  the detector saw a likely change (50%+) before {flagged.mean():.2%} of hitter-games")


if __name__ == "__main__":
    main()

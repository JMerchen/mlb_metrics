"""Grades nfl_totals.py's predicted point totals against real scores and
the market's closing total, season by season.

For every regular-season week of 2017-2025, each team's WAVE-style ratings
are built by nfl_totals.wave_ratings from the prior season plus the current
season's earlier weeks - the same history the live pipeline has - and
turned into features for that week's games. The points regression is then
refit season by season on earlier seasons only (2017 onward) and scored on
2018-2025 against nflverse's closing `total_line`.

Reports: mean absolute error of the predicted total vs the market's (and a
flat league-average total), a regression of the actual total on both (does
the prediction add anything the market's number doesn't?), and how often
the over/under side the prediction leans to wins, by size of disagreement
(52.4% breaks even at -110). Ends with the coefficients fit on every
season, for config.NFL_TOTALS_COEFFICIENTS.

RESULT (2026-10-13, 2,127 games 2018-2025 with a closing total):
    predicted total   MAE 10.64 points
    market total      MAE 10.39
    flat average      MAE 11.00
The first version (team EPA and pace only) missed by 10.80. What closed
part of the gap: the listed starting QB's own EPA per dropback (a backup
starting is invisible to team ratings built on the starter's games) and
whether the game is under a roof. Tested and NOT helping: points
scored/allowed, success rate, early-down EPA, separate pass/rush EPA,
neutral-situation pass rate, drives per game, red-zone TD rate,
turnovers, a season-to-date league scoring level, gradient boosting on
all of them. Temperature and wind help on history (10.57 with the QB
term) but weren't published before kickoff - see the weather note below.
Two seasons of QB history instead of one: 10.63, not worth it.
Given the market's total the prediction still adds nothing (coefficient
0.01, p = 0.90); over/under leans are right 50-51% overall, 55.5% on the
236 games where it disagrees with the market by 5+ points - within noise
of break-even. Shown next to Vegas's number, never turned into bets.

With the pull toward the league fading for units far from it
(config.NFL_TOTALS_SHRINK_FADE, 2026-11-01): MAE 10.61 (market 10.39),
leans right 58.3% on the 216 games 5+ points from the market, and teams
score 0.21 points more than predicted against the worst defenses rather
than 0.57 (scripts/backtest_nfl_extreme_shrink.py).

With the game's weather (nfl_weather.py, 2026-11-04: nflverse's readings
for past games, Open-Meteo's forecast live) - wind over 5 mph, degrees
below 40F and rain: MAE 10.47 (market 10.39), and for the first time the
prediction adds to the market's total (coefficient 0.26, p = 0.004); leans
right 62.0% on the 237 games 5+ points from the market. Rain alone is -2.8
points a team. Tried and not kept: wind from 0/8/10/12 mph (5 best), cold
below 32/50 (40 best), snow as its own term or counted as rain (22 games,
nothing). Fit on game-time readings; a day-before forecast is less sure,
so live the gain will be somewhat smaller.

Shown as a range (2026-11-06, config.NFL_TOTAL_BAND_HALF_WIDTH): the real
total landed within 6/9/11 points of ours in 36/52/60% of games (Vegas's
total: 37/53/62%); the median miss is 8.7 (7.6 on the lowest third of
predicted totals, 9.4 on the middle). The site uses +/-9: "about half
the time".

Usage:
    python scripts/backtest_nfl_totals.py
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import nfl_totals, nfl_weather  # noqa: E402

RAW_DIR = "data/raw/nfl"
SEASONS = range(2016, 2026)
FIRST_FIT_SEASON = 2017
SCORED_SEASONS = range(2018, 2026)


def load():
    schedules = pd.concat([pd.read_parquet(os.path.join(RAW_DIR, f"schedules_{s}.parquet")) for s in SEASONS])
    schedules = schedules[schedules["game_type"] == "REG"]
    team_games, qb_games, weather = [], [], []
    for season in SEASONS:
        pbp = pd.read_parquet(
            os.path.join(RAW_DIR, f"pbp_{season}.parquet"),
            columns=sorted(set(nfl_totals.PBP_COLUMNS) | set(nfl_totals.QB_PBP_COLUMNS) | {"weather"}),
        )
        weather.append(nfl_weather.historical_weather(schedules[schedules["season"] == season], pbp))
        team_games.append(nfl_totals.team_game_epa(pbp, schedules[schedules["season"] == season]))
        qb_games.append(nfl_totals.qb_game_epa(pbp))
    schedules = schedules.merge(pd.concat(weather, ignore_index=True), on="game_id", how="left")
    return pd.concat(team_games, ignore_index=True), pd.concat(qb_games, ignore_index=True), schedules


def weekly_features(team_games: pd.DataFrame, qb_games: pd.DataFrame, schedules: pd.DataFrame) -> pd.DataFrame:
    """Features for every team-game from FIRST_FIT_SEASON on, each built
    only from the prior season and the current season's earlier weeks -
    the history the live pipeline has. Starting QBs are the ones the
    schedule lists, as before kickoff."""
    def before(df, season, week):
        return df[(df["season"] == season - 1) | ((df["season"] == season) & (df["week"] < week))]

    frames = []
    for (season, week), games in schedules[schedules["season"] >= FIRST_FIT_SEASON].groupby(["season", "week"]):
        features = nfl_totals.matchup_features(
            nfl_totals.wave_ratings(before(team_games, season, week)), games,
            before(qb_games, season, week), before(schedules, season, week),
        )
        features["season"] = season
        frames.append(features)
    features = pd.concat(frames, ignore_index=True)
    return features.merge(team_games[["game_id", "team", "points"]], on=["game_id", "team"], how="inner")


def fit(rows: pd.DataFrame):
    return sm.OLS(rows["points"], sm.add_constant(rows[nfl_totals.TOTALS_FEATURE_COLUMNS])).fit()


def main():
    team_games, qb_games, schedules = load()
    features = weekly_features(team_games, qb_games, schedules).dropna(subset=nfl_totals.TOTALS_FEATURE_COLUMNS + ["points"])

    scored = []
    for season in SCORED_SEASONS:
        model = fit(features[features["season"] < season])
        test = features[features["season"] == season].copy()
        test["predicted"] = model.predict(sm.add_constant(test[nfl_totals.TOTALS_FEATURE_COLUMNS], has_constant="add"))
        scored.append(test)
    scored = pd.concat(scored)
    games = (
        scored.groupby("game_id")
        .agg(predicted_total=("predicted", "sum"), total=("points", "sum"), sides=("points", "size"))
        .reset_index()
    )
    games = games[games["sides"] == 2].merge(schedules[["game_id", "season", "total_line"]], on="game_id")
    games = games.dropna(subset=["total_line"])

    flat = games["total"].mean()
    print(f"{len(games)} games {SCORED_SEASONS.start}-{SCORED_SEASONS.stop - 1} with a closing total")
    print(f"MAE  predicted {(games['predicted_total'] - games['total']).abs().mean():.2f}"
          f"  market {(games['total_line'] - games['total']).abs().mean():.2f}"
          f"  flat average {(games['total'] - flat).abs().mean():.2f}")
    both = sm.OLS(games["total"], sm.add_constant(games[["total_line", "predicted_total"]])).fit()
    print("actual total ~ market + predicted:", both.params.round(3).to_dict(), "p", both.pvalues.round(4).to_dict())

    games["lean"] = games["predicted_total"] - games["total_line"]
    decided = games[games["total"] != games["total_line"]]
    for gap in (0, 2, 3, 4, 5):
        lean = decided[decided["lean"].abs() >= gap]
        right = ((lean["lean"] > 0) == (lean["total"] > lean["total_line"])).mean()
        print(f"  disagree by {gap}+ points: {len(lean)} games, over/under lean right {right:.3f}")
    by_season = games.groupby("season").apply(lambda d: pd.Series({
        "predicted": (d["predicted_total"] - d["total"]).abs().mean(),
        "market": (d["total_line"] - d["total"]).abs().mean(),
    }), include_groups=False)
    print(by_season.round(2).to_string())

    outdoor = features[features["dome"] == 0]
    known = outdoor.merge(schedules[["game_id", *nfl_weather.WEATHER_INPUT_COLUMNS]], on="game_id")
    print("\nconfig.NFL_WEATHER_OUTDOOR_MEANS (outdoor games with a reading):")
    print({c: round(float(known.loc[known[i].notna(), c].mean()), 3) for c, i in
           zip(nfl_weather.WEATHER_FEATURE_COLUMNS, ["weather_wind", "weather_temp", "weather_rain"])})

    # How far off a game's total is, for the band shown on the site
    # (config.NFL_TOTAL_BAND_HALF_WIDTH).
    miss = (games["predicted_total"] - games["total"]).abs()
    print("\nmiss quantiles (points):", {q: round(float(miss.quantile(q)), 1) for q in (0.5, 0.6, 2 / 3, 0.75, 0.8)})
    for half in (6, 7, 8, 9, 10, 11):
        print(f"  within +/-{half}: {(miss <= half).mean():.3f}"
              f"  (market's total: {((games['total_line'] - games['total']).abs() <= half).mean():.3f})")
    games["third"] = pd.qcut(games["predicted_total"], 3, labels=["low", "mid", "high"])
    print("  median miss by predicted total:", games.groupby("third", observed=True)
          .apply(lambda d: round(float((d["predicted_total"] - d["total"]).abs().median()), 1), include_groups=False).to_dict())

    final = fit(features)
    print("\nconfig.NFL_TOTALS_COEFFICIENTS (fit on every season):")
    print({name: round(float(value), 3) for name, value in final.params.items()})
    print("RMSE predicted %.2f market %.2f" % (
        np.sqrt(((games["predicted_total"] - games["total"]) ** 2).mean()),
        np.sqrt(((games["total_line"] - games["total"]) ** 2).mean()),
    ))


if __name__ == "__main__":
    main()

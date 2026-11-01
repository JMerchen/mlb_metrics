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
term) but aren't published before kickoff, so they can't be used live.
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

Usage:
    python scripts/backtest_nfl_totals.py
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import nfl_totals  # noqa: E402

RAW_DIR = "data/raw/nfl"
SEASONS = range(2016, 2026)
FIRST_FIT_SEASON = 2017
SCORED_SEASONS = range(2018, 2026)


def load():
    schedules = pd.concat([pd.read_parquet(os.path.join(RAW_DIR, f"schedules_{s}.parquet")) for s in SEASONS])
    schedules = schedules[schedules["game_type"] == "REG"]
    team_games, qb_games = [], []
    for season in SEASONS:
        pbp = pd.read_parquet(
            os.path.join(RAW_DIR, f"pbp_{season}.parquet"),
            columns=sorted(set(nfl_totals.PBP_COLUMNS) | set(nfl_totals.QB_PBP_COLUMNS)),
        )
        team_games.append(nfl_totals.team_game_epa(pbp, schedules[schedules["season"] == season]))
        qb_games.append(nfl_totals.qb_game_epa(pbp))
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

    final = fit(features)
    print("\nconfig.NFL_TOTALS_COEFFICIENTS (fit on every season):")
    print({name: round(float(value), 3) for name, value in final.params.items()})
    print("RMSE predicted %.2f market %.2f" % (
        np.sqrt(((games["predicted_total"] - games["total"]) ** 2).mean()),
        np.sqrt(((games["total_line"] - games["total"]) ** 2).mean()),
    ))


if __name__ == "__main__":
    main()

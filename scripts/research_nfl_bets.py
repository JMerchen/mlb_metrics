"""Which NFL bets has our model actually beaten the market on?

User request (2026-11-14): "The point of the site is to act as a quant
level site to make educated bets ... The accuracy of our bets is more
important than the accuracy of our model ... If there are better models
out there for this goal, leverage what they have that we don't."

Every test here is walk-forward (a season is predicted by a model fit on
earlier seasons only) and graded at real prices where the data has them.
Two outside files make the difference:
  * nfelo's game file (github.com/greerreNFL/nfelo, output_data/
    nfelo_games.csv): every game's OPENING and closing spread, opening
    totals from 2024, and that public model's own ratings.
  * nflverse's games.csv (github.com/nflverse/nfldata): closing totals,
    spreads, wind and roof back to 1999.

RESULTS (2026-11-15; our model's games are 2019-2026, 1,766 with lines):

1. Sides against the CLOSING spread: nothing. Our side won 49-51% at every
   gap (533 bets at 3+ points: 49.3%), and betting our winner at the
   closing moneyline lost the bookmaker's cut whether we agreed with Vegas
   (1,691 bets, -53.2 units) or not (308 bets, 42.2% right, needed 44.0%,
   -10.7 units). Betting any side whose price our percentage beat by 5+
   points: 875 bets, -64.8 units.

2. Sides against the OPENING spread look good and are not to be trusted.
   Our side won 54.2% of 483 at 3+ points (55.5% of 263 at 4+), and the
   line then moved toward us (4+: our way 49%, against 25%). It is not
   hindsight about quarterbacks: with the same starters as the week
   before it is 54.7% of 373. But the openers move with the PREVIOUS
   week's results (0.135 points per 10 points a team beat its spread by,
   t 6.5) and by 1.13 points on average, so many were posted before those
   games - numbers nobody could bet once our model had run. At the line
   halfway between open and close the edge is gone (48.7% of 483 at 3+).
   nfelo's own ratings show the same shape, only bigger (61.7% of 418 at
   3+ against the open, 52.9% of 873 at 2+ against the close), and where
   both models agreed against the close by 3+ it was 43.6% of 101.
   Verdict: no spread bet. What it does say is that beating the close is
   the whole game: having the opener on the side the line later moved to
   by a point or more won 58.4% of 1,203.

3. Over/unders: the one thing that holds, by less than we had claimed.
   The site's 62% (237 leans) knew whether it rained: past games' rain is
   read from the play-by-play, and rain games carried the record (83% of
   23). With rain unknown for the game being predicted, at the real odds:
       gap 5+ : 174 bets, 58.6% (51.3-65.9%), +21.9 units, winning
                record in 6 of 7 seasons, p 0.058 against 52.4%
       gap 4+ : 349, 54.2%, +14.7     gap 3+ : 591, 52.8%, +10.3
       overs 59.2% of 125, unders 57.1% of 49
   Adding a wind term for what the line misses made it worse (54.6% of
   207 against 59.2% of 152 on the same seasons), so the lean is the plain
   gap. Against opening totals (2024-25 only) a 3+ point gap saw the line
   move our way 63% of the time and against 17%: take them early.

4. Wind on its own: unders in outdoor games with 10+ mph at kickoff won
   54.7% of 1,782 since 1999 (a winning record in 18 of 28 seasons; 58.2% of 395
   in 2019-25), against 48-49% in calmer games. Real but thin, already in
   our total through its wind term, and it needs the kickoff reading.

5. Teasers (6 points, through 3 and 7: favorites of 7.5-8.5 down,
   underdogs of 1.5-2.5 up): 75.3% of 643 legs in 2016-25, 73.9% of 322 in
   2006-15. A two-leg teaser needs 73.9% a leg at -120 and 75.2% at -130,
   so this is break-even at today's prices, not a recommendation.

6. Player props can't be judged: ours are graded against the player's own
   average, not a sportsbook's line, and we have no history of prop lines.

WHAT THE SITE DOES WITH IT: recommends only the over/under lean, on the
plain gap with the forecast's rain taken out of our total, with its
corrected record; shows the worst line and price worth taking and says to
bet early; reports units, not just wins, for picks with and against
Vegas; and calls winners and spread sides picks, not bets.

Usage:
    PYTHONPATH=src python3 scripts/research_nfl_bets.py --cache DIR [--totals]
(--cache holds the two downloaded files; --totals also refits the points
model season by season for part 3, a few minutes.)
"""

import argparse
import glob
import os
import sys
import urllib.request

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import norm

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlb_metrics import config  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SOURCES = {
    "nfelo_games.csv": "https://raw.githubusercontent.com/greerreNFL/nfelo/main/output_data/nfelo_games.csv",
    "games.csv": "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv",
}


def cached(cache: str, name: str) -> pd.DataFrame:
    path = os.path.join(cache, name)
    if not os.path.exists(path):
        os.makedirs(cache, exist_ok=True)
        urllib.request.urlretrieve(SOURCES[name], path)
    return pd.read_csv(path)


def profit(won, price):
    price = pd.to_numeric(pd.Series(price), errors="coerce").fillna(-110.0).to_numpy()
    pay = np.where(price > 0, price / 100.0, 100.0 / np.abs(price))
    won = np.asarray(won, dtype=float)
    return np.where(np.isnan(won), 0.0, np.where(won == 1, pay, -1.0))


def report(label, won, price=None, season=None):
    won = pd.Series(np.asarray(won, dtype=float))
    graded = won.dropna()
    if graded.empty:
        print(f"{label}: none")
        return
    units = pd.Series(profit(won.to_numpy(), np.full(len(won), -110.0) if price is None else np.asarray(price)))
    rate, n = graded.mean(), len(graded)
    line = f"{label}: {n} bets, {rate:.1%} (±{1.96 * np.sqrt(rate * (1 - rate) / n):.1%}), {units.sum():+.1f} units"
    if season is not None:
        by = units.groupby(np.asarray(season)).sum()
        line += f", winning seasons {int((by > 0).sum())}/{len(by)}"
    print(line)


def load_games(cache: str) -> pd.DataFrame:
    schedules = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(REPO, "data/raw/nfl/schedules_20*.parquet")))])
    games = schedules[schedules["home_score"].notna()].copy()
    games["margin"] = games["home_score"] - games["away_score"]
    games = games.merge(cached(cache, "nfelo_games.csv").drop(columns=["Unnamed: 0"], errors="ignore"), on="game_id", how="left")
    ours = pd.read_csv(os.path.join(REPO, "data/backtest/nfl_win_model_v4_walk_forward.csv"))[["game_id", "market", "new"]]
    games = games.merge(ours, on="game_id", how="left")
    games["our_margin"] = config.NFL_WIN_MARGIN_SIGMA * norm.ppf(games["new"].clip(0.001, 0.999))
    games["nfelo_margin"] = games["nfelo_dif_base"] / 25.0
    games["line_halfway"] = (games["home_line_open"] + games["home_line_close"]) / 2
    return games


def against_the_spread(games, margin, line, low, label):
    gap = games[margin] + games[line]
    use = gap.abs() >= low
    result = games["margin"] + games[line]
    won = np.where(result == 0, np.nan, np.where(gap > 0, result > 0, result < 0).astype(float))
    report(label, won[use], None, games["season"][use])


def sides(games):
    ours = games[games["new"].notna() & games["home_line_open"].notna()]
    print(f"\n== 1-2. Sides ({len(ours)} games with our model and both lines) ==")
    for line, name in (("home_line_close", "closing"), ("home_line_open", "opening"), ("line_halfway", "halfway")):
        for low in (3, 4):
            against_the_spread(ours, "our_margin", line, low, f"our side vs the {name} spread, gap {low}+")
    everyone = games[games["home_line_open"].notna() & games["nfelo_margin"].notna()]
    against_the_spread(everyone, "nfelo_margin", "home_line_open", 3, "nfelo's ratings vs the opening spread, gap 3+ (2016-26)")
    against_the_spread(everyone, "nfelo_margin", "home_line_close", 2, "nfelo's ratings vs the closing spread, gap 2+ (2016-26)")
    gap = ours["our_margin"] + ours["home_line_open"]
    toward = np.sign(gap) * (ours["home_line_open"] - ours["home_line_close"])
    big = gap.abs() >= 4
    print(f"line movement after a 4+ point gap at the open: our way {np.mean(toward[big] > 0):.1%}, against {np.mean(toward[big] < 0):.1%}")
    # Do the openers already know last week's results?
    lined = games[games["home_line_close"].notna()].copy()
    lined["cover"] = lined["margin"] + lined["home_line_close"]
    team = pd.concat([
        lined[["season", "week", "game_id", "home_team", "cover"]].set_axis(["season", "week", "game_id", "team", "ats"], axis=1),
        lined.assign(cover=-lined["cover"])[["season", "week", "game_id", "away_team", "cover"]].set_axis(
            ["season", "week", "game_id", "team", "ats"], axis=1),
    ]).sort_values(["season", "team", "week"])
    team["previous"] = team.groupby(["season", "team"])["ats"].shift(1)
    previous = team.set_index(["game_id", "team"])["previous"]
    lined["surprise"] = [previous.get((g, h), np.nan) - previous.get((g, a), np.nan)
                         for g, h, a in zip(lined["game_id"], lined["home_team"], lined["away_team"])]
    lined = lined.dropna(subset=["surprise", "home_line_open"])
    move = lined["home_line_open"] - lined["home_line_close"]
    fit = sm.OLS(move, sm.add_constant(lined[["surprise"]])).fit()
    print(f"open-to-close move: {move.abs().mean():.2f} points on average; per 10 points of last week's surprise "
          f"{10 * fit.params['surprise']:+.3f} (t {fit.tvalues['surprise']:.1f})")
    # Moneylines at the close.
    priced = games[games["new"].notna() & games["home_moneyline"].notna()]
    home = priced["new"] > 0.5
    line = np.where(home, priced["home_moneyline"], priced["away_moneyline"])
    won = np.where(home, priced["margin"] > 0, priced["margin"] < 0).astype(float)
    agree = home == (priced["market"] > 0.5)
    report("our winner at the closing moneyline, agreeing with Vegas", won[agree], line[agree], priced["season"][agree])
    report("our winner at the closing moneyline, against Vegas", won[~agree], line[~agree], priced["season"][~agree])


def totals(cache):
    import backtest_nfl_totals as replay
    from mlb_metrics import nfl_totals
    print("\n== 3. Over/unders with rain unknown (refitting season by season) ==")
    team_games, qb_games, schedules = replay.load()
    columns = nfl_totals.TOTALS_FEATURE_COLUMNS
    features = replay.weekly_features(team_games, qb_games, schedules).dropna(subset=columns + ["points"])
    scored = []
    for season in replay.SCORED_SEASONS:
        model = replay.fit(features[features["season"] < season])
        test = features[features["season"] == season].copy()
        test["predicted"] = model.predict(sm.add_constant(test[columns], has_constant="add"))
        test["blind"] = model.predict(sm.add_constant(test.assign(wx_rain=0.0)[columns], has_constant="add"))
        scored.append(test)
    rows = pd.concat(scored)
    games = rows.groupby("game_id").agg(predicted=("predicted", "sum"), blind=("blind", "sum"), total=("points", "sum"),
                                        sides=("points", "size"), wind=("wx_wind", "first")).reset_index()
    games = games[games["sides"] == 2].merge(
        schedules[["game_id", "season", "total_line", "over_odds", "under_odds"]], on="game_id").dropna(subset=["total_line"])
    games = games[games["season"] >= 2019]
    games["miss"] = games["total"] - games["total_line"]

    def grade(frame, edge, label):
        over = frame[edge] > 0
        won = np.where(frame["miss"] == 0, np.nan, (over == (frame["miss"] > 0)).astype(float))
        report(label, won, np.where(over, frame["over_odds"], frame["under_odds"]), frame["season"])

    games["gap"], games["gap_blind"] = games["predicted"] - games["total_line"], games["blind"] - games["total_line"]
    grade(games[games["gap"].abs() >= 5], "gap", "knowing the rain (as the site had it), gap 5+")
    for low in (3, 4, 5):
        grade(games[games["gap_blind"].abs() >= low], "gap_blind", f"rain unknown, gap {low}+")
    opened = games.merge(cached(cache, "nfelo_games.csv")[["game_id", "total_line_open", "total_line_close"]], on="game_id")
    opened = opened.dropna(subset=["total_line_open"])
    gap = opened["blind"] - opened["total_line_open"]
    toward = np.sign(gap) * (opened["total_line_close"] - opened["total_line_open"])
    big = gap.abs() >= 3
    print(f"opening totals (2024-25), gap 3+: line moved our way {np.mean(toward[big] > 0):.1%}, against {np.mean(toward[big] < 0):.1%}")


def wind_and_teasers(cache):
    games = cached(cache, "games.csv")
    games = games[games["home_score"].notna() & games["total_line"].notna() & (games["game_type"] == "REG")].copy()
    games["miss"] = games["home_score"] + games["away_score"] - games["total_line"]
    games["margin"] = games["home_score"] - games["away_score"]
    print("\n== 4. Unders by wind at kickoff, outdoor games since 1999 ==")
    outdoors = games[games["roof"].isin(["outdoors", "open"]) & games["wind"].notna()]
    for low, high in ((0, 10), (10, 99)):
        band = outdoors[(outdoors["wind"] >= low) & (outdoors["wind"] < high)]
        report(f"unders, wind {low}-{high} mph", np.where(band["miss"] == 0, np.nan, (band["miss"] < 0).astype(float)), None, band["season"])
    print("\n== 5. Six-point teaser legs through 3 and 7 ==")
    games["home_line"] = -games["spread_line"]
    for first, last in ((2006, 2015), (2016, 2025)):
        era = games[(games["season"] >= first) & (games["season"] <= last)]
        legs = []
        for frame, home in ((era[era["home_line"].between(-8.5, -7.5)], True), (era[era["home_line"].between(7.5, 8.5)], False),
                            (era[era["home_line"].between(1.5, 2.5)], True), (era[era["home_line"].between(-2.5, -1.5)], False)):
            side = 1 if home else -1
            legs.append(((side * frame["margin"] + side * frame["home_line"] + 6) > 0).astype(float))
        legs = pd.concat(legs)
        print(f"{first}-{last}: {len(legs)} legs, {legs.mean():.1%} (a two-leg teaser needs 73.9% a leg at -120, 75.2% at -130)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--totals", action="store_true")
    args = parser.parse_args()
    sides(load_games(args.cache))
    if args.totals:
        totals(args.cache)
    wind_and_teasers(args.cache)


if __name__ == "__main__":
    main()

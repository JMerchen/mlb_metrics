"""Tests per-category, opponent-adjusted NFL team ratings against the market.

The idea (user, 2026-10-11): rate each team's passing, rushing and scoring
on offense and the same on defense, weighting recent games more and
crediting production by who it came against - 350 passing yards against a
bad defense can be worth the same as 200 against a good one - and feed
both teams' ratings to the model.

The live model already has recency-weighted offense/defense EPA and a
one-pass opponent adjustment (each game netted against the opponent's own
average), which tested flat on 2,333 games and ships switched off. This
tries the two pieces that hadn't been: separate categories, and a proper
opponent adjustment - every team's offense and defense rating solved
together in one weighted ridge regression, so a weak schedule is
accounted for all the way through rather than one step deep.

For each category c (pass EPA per dropback, rush EPA per carry, points per
game), as of the start of every week, using only earlier games:
    y[team, game] = mean + offense[team] + defense[opponent] + home * hfa
fit by ridge regression with exponentially decaying weights on older
games (and a further discount on earlier seasons). A game's feature is
the home side's expected edge: (off_home + def_away) - (off_away + def_home).

Variants, each a logistic regression refit season by season on earlier
seasons only, scored on 2019-2025 (2016-2017 warm up the ratings):
    A  points only (adjusted)
    B  one combined EPA per play (adjusted) - closest to today's model
    C  pass and rush EPA, separately (adjusted)
    D  pass, rush and points (adjusted)
    E  pass, rush and points, NOT opponent-adjusted (recency means)
Each is compared with the de-vigged closing moneyline on the same games,
plus a logistic regression of home wins on both log-odds (does the model
add anything the market doesn't know?).

RESULT (2026-10-11, 1,865 games 2019-2025 with a moneyline):
    market                              Brier 0.2105, picks right 66.5%
    A points only (adjusted)            Brier 0.2245, 64.5%
    B one combined EPA (adjusted)       Brier 0.2358, 58.8%
    C pass + rush EPA (adjusted)        Brier 0.2314, 60.7%
    D pass + rush + points (adjusted)   Brier 0.2246, 64.3%
    E pass + rush + points (unadjusted) Brier 0.2247, 64.1%
Splitting into categories adds nothing over points alone, and solving the
opponent adjustment jointly changes almost nothing (D vs E). Every
variant trails the market, and in the logistic regression on both
log-odds the model's coefficient is NEGATIVE (-0.26 for D, -0.65 for C):
once the market's price is known, these ratings point the wrong way -
they overreact to noise the market already discounts. A sweep of
half-life (6-40 weeks), prior-season discount (0.5-0.7) and ridge
strength (5-80) moved the best Brier between 0.2245 and 0.2321 and never
changed the sign. Not shipped; the idea is sound, but the market already
prices it in.

Usage:
    python scripts/backtest_nfl_category_ratings.py
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import brier_score_loss, log_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import historical_odds  # noqa: E402

RAW_DIR = "data/raw/nfl"
SEASONS = range(2016, 2026)
SCORED_SEASONS = range(2019, 2026)
HALF_LIFE_WEEKS = 10.0
PRIOR_SEASON_DISCOUNT = 0.5
RIDGE_ALPHA = 20.0
CATEGORIES = ["pass_epa", "rush_epa", "points", "all_epa"]


def load_team_games() -> pd.DataFrame:
    """One row per (game, team): EPA per pass play, per rush play, per play,
    and points scored, regular season only."""
    frames = []
    for season in SEASONS:
        pbp = pd.read_parquet(
            os.path.join(RAW_DIR, f"pbp_{season}.parquet"),
            columns=["game_id", "season", "week", "season_type", "posteam", "defteam",
                     "home_team", "epa", "pass", "rush", "qb_kneel", "qb_spike"],
        )
        pbp = pbp[(pbp["season_type"] == "REG") & pbp["epa"].notna() & pbp["posteam"].notna()]
        pbp = pbp[(pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)]
        pbp = pbp[(pbp["pass"] == 1) | (pbp["rush"] == 1)]
        grouped = pbp.groupby(["game_id", "season", "week", "posteam", "defteam", "home_team"])
        agg = grouped.apply(lambda g: pd.Series({
            "pass_epa": g.loc[g["pass"] == 1, "epa"].mean(),
            "rush_epa": g.loc[g["rush"] == 1, "epa"].mean(),
            "all_epa": g["epa"].mean(),
            "plays": len(g),
        }), include_groups=False).reset_index()
        frames.append(agg)
    team_games = pd.concat(frames, ignore_index=True).rename(columns={"posteam": "team", "defteam": "opp"})
    team_games["home"] = (team_games["team"] == team_games["home_team"]).astype(float)

    sched = pd.concat([pd.read_parquet(os.path.join(RAW_DIR, f"schedules_{s}.parquet")) for s in SEASONS])
    sched = sched[sched["game_type"] == "REG"]
    points = pd.concat([
        sched[["game_id", "home_team", "home_score"]].rename(columns={"home_team": "team", "home_score": "points"}),
        sched[["game_id", "away_team", "away_score"]].rename(columns={"away_team": "team", "away_score": "points"}),
    ])
    team_games = team_games.merge(points, on=["game_id", "team"], how="left")
    team_games["t"] = team_games["season"] * 20 + team_games["week"]
    return team_games, sched


def ratings_as_of(history: pd.DataFrame, season: int, week: int, category: str, adjusted: bool) -> pd.DataFrame:
    """Offense and defense rating per team for `category`, from games before
    (season, week). Adjusted: ridge with team offense + opponent defense +
    home terms. Unadjusted: recency-weighted means of produced/allowed."""
    now = season * 20 + week
    h = history[(history["t"] < now) & history[category].notna()]
    if h.empty:
        return pd.DataFrame(columns=["team", "off", "def"])
    weights = 0.5 ** ((now - h["t"]) / HALF_LIFE_WEEKS)
    weights = weights * np.where(h["season"] < season, PRIOR_SEASON_DISCOUNT, 1.0)
    teams = sorted(set(h["team"]) | set(h["opp"]))
    if not adjusted:
        mean = np.average(h[category], weights=weights)
        off = h.assign(w=weights).groupby("team").apply(
            lambda g: np.average(g[category], weights=g["w"]) - mean, include_groups=False)
        deff = h.assign(w=weights).groupby("opp").apply(
            lambda g: np.average(g[category], weights=g["w"]) - mean, include_groups=False)
        return pd.DataFrame({"team": teams, "off": off.reindex(teams).fillna(0).to_numpy(),
                             "def": deff.reindex(teams).fillna(0).to_numpy()})
    index = {team: i for i, team in enumerate(teams)}
    n = len(teams)
    X = np.zeros((len(h), 2 * n + 1))
    rows = np.arange(len(h))
    X[rows, h["team"].map(index).to_numpy()] = 1.0
    X[rows, n + h["opp"].map(index).to_numpy()] = 1.0
    X[:, -1] = h["home"].to_numpy()
    model = Ridge(alpha=RIDGE_ALPHA).fit(X, h[category].to_numpy(), sample_weight=weights.to_numpy())
    return pd.DataFrame({"team": teams, "off": model.coef_[:n], "def": model.coef_[n:2 * n]})


def build_features(team_games: pd.DataFrame, sched: pd.DataFrame) -> pd.DataFrame:
    games = sched[["game_id", "season", "week", "home_team", "away_team", "home_score", "away_score",
                   "home_moneyline", "away_moneyline"]].copy()
    games = games[games["home_score"].notna() & (games["home_score"] != games["away_score"])]
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
    rows = []
    for (season, week), slate in games.groupby(["season", "week"]):
        if season < 2017:
            continue
        feats = {gid: {} for gid in slate["game_id"]}
        for adjusted in (True, False):
            for category in CATEGORIES:
                r = ratings_as_of(team_games, season, week, category, adjusted).set_index("team")
                for g in slate.itertuples():
                    get = lambda team, col: r[col].get(team, 0.0) if not r.empty else 0.0  # noqa: E731
                    edge = (get(g.home_team, "off") + get(g.away_team, "def")) - (
                        get(g.away_team, "off") + get(g.home_team, "def"))
                    feats[g.game_id][f"{category}_{'adj' if adjusted else 'raw'}"] = edge
        for g in slate.itertuples():
            rows.append({"game_id": g.game_id, "season": season, "week": week, "home_win": g.home_win,
                         "home_moneyline": g.home_moneyline, "away_moneyline": g.away_moneyline,
                         **feats[g.game_id]})
    df = pd.DataFrame(rows)
    hi = historical_odds.implied(df["home_moneyline"])
    ai = historical_odds.implied(df["away_moneyline"])
    df["market_p"] = hi / (hi + ai)
    return df


VARIANTS = {
    "A points only (adjusted)": ["points_adj"],
    "B one combined EPA (adjusted)": ["all_epa_adj"],
    "C pass + rush EPA (adjusted)": ["pass_epa_adj", "rush_epa_adj"],
    "D pass + rush + points (adjusted)": ["pass_epa_adj", "rush_epa_adj", "points_adj"],
    "E pass + rush + points (NOT adjusted)": ["pass_epa_raw", "rush_epa_raw", "points_raw"],
}


def walk_forward(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    out = []
    for season in SCORED_SEASONS:
        train = df[df["season"] < season]
        test = df[df["season"] == season]
        model = LogisticRegression(C=1.0).fit(train[columns], train["home_win"])
        out.append(test.assign(p=model.predict_proba(test[columns])[:, 1]))
    return pd.concat(out, ignore_index=True)


def main() -> None:
    team_games, sched = load_team_games()
    df = build_features(team_games, sched)
    scored = df[df["season"].isin(SCORED_SEASONS) & df["market_p"].notna()]
    y = scored["home_win"]
    print(f"Scored games 2019-2025 with a moneyline: {len(scored)}")
    print(f"  market (de-vigged moneyline): Brier {brier_score_loss(y, scored['market_p']):.4f}, "
          f"log loss {log_loss(y, scored['market_p']):.4f}, picks right {((scored['market_p'] > .5) == y).mean():.3f}")
    logit = lambda x: np.log(x / (1 - x))  # noqa: E731
    for name, columns in VARIANTS.items():
        pred = walk_forward(df[df["market_p"].notna()], columns)
        pred = pred[pred["season"].isin(SCORED_SEASONS)]
        yp = pred["home_win"]
        fit = sm.Logit(yp, sm.add_constant(pd.DataFrame({"market": logit(pred["market_p"]), "model": logit(pred["p"])}))).fit(disp=0)
        print(f"  {name:40s} Brier {brier_score_loss(yp, pred['p']):.4f}, log loss {log_loss(yp, pred['p']):.4f}, "
              f"picks right {((pred['p'] > .5) == yp).mean():.3f}; beyond market p = {fit.pvalues['model']:.3f}")


if __name__ == "__main__":
    main()

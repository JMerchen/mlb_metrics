"""MLB game win probability, trained on results.

Replaces the hand-built composite ratio in game_picks.compute_game_win_probabilities
as the source of each game's home win probability. That ratio was never fit to
outcomes, had no home-field term, and was compressed toward 50% (SD 0.050
against the market's 0.087 on the same games), which rated every underdog
about 4.8 points above the market - 95% of its advised bets were underdogs,
and they lost 21% of the money staked. See scripts/backtest_market_shrinkage.py.

Everything here is built from persisted Statcast, so the same code produces
training rows for every past game and features for today's slate. The model
is a three-feature logistic regression refit on every run - it is cheap
(~5,000 rows) and never goes stale the way a saved artifact would.

PREGAME ONLY. Every feature for a game on date d uses games strictly before
d, so a doubleheader's second game does not see the first.

RECENCY WITHOUT SEASON BOUNDARIES. Each rate is an exponentially weighted sum
over all prior days. Last season fades on its own across the winter instead
of being dropped on Opening Day or carried at full weight, so early-season
games still have informative features. Each rate is then shrunk toward the
league rate by a pseudo-count, so a starter with three career starts is not
rated on three starts.

FEATURES, each home minus away so "positive favors home" throughout, plus an
intercept that is the home-field advantage:
  - rd_diff:     run differential per game
  - sp_kbb_diff: the probable starter's strikeouts minus walks per batter
  - pen_diff:    expected wOBA allowed by the bullpen (everyone but the
                 starter; xwOBA on contact, actual value otherwise)

HOW THEY WERE CHOSEN (scripts/backtest_mlb_game_model.py, 2026-10-04). Each
month from June 2025 on was predicted by a model fit only on earlier games.
Recency and shrinkage settings were tuned on games before 2026-08-01 only;
August 2026 on - the 488 games with logged market odds - was held back as
the test. On the tuning games, this set tied the best alternative (team
offense wOBA and starter xwOBA instead of K-BB; Brier 0.2468 both) with
fewer inputs, and on the held-back games it was better (0.2363 vs 0.2386).
That later period was therefore used as a tiebreak, which is worth knowing
when reading its numbers.

On the held-back games: Brier 0.2363 for this model, 0.2455 for the old
ratio, 0.2345 for the de-vigged market. It is a better forecaster than the
model it replaces, and still slightly worse than the market.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from mlb_metrics import data

LN2 = np.log(2.0)

# Statcast and the live schedule spell two teams differently; features are
# keyed on one spelling.
TEAM_ALIASES = {"AZ": "ARI", "ATH": "OAK"}

FEATURE_COLUMNS = ["rd_diff", "sp_kbb_diff", "pen_diff"]

# Half-lives in days; k is the shrinkage pseudo-count, in games for run
# differential and in batters faced for the pitching rates. Tuned by
# coordinate search on pre-August-2026 games (see module docstring).
DEFAULT_PARAMS = {
    "team_half_life": 240.0,
    "sp_half_life": 60.0,
    "pen_half_life": 180.0,
    "rd_k": 5.0,
    "sp_k": 600.0,
    "pen_k": 150.0,
}

STATCAST_COLUMNS = data.GAME_MODEL_STATCAST_COLUMNS

_ORIGIN = pd.Timestamp("2015-01-01")


def normalize_team(series: pd.Series) -> pd.Series:
    return series.replace(TEAM_ALIASES)


def regular_season(statcast: pd.DataFrame) -> pd.DataFrame:
    """Regular-season pitches with normalized team codes and each pitch's
    pitching and batting team."""
    df = statcast
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"]).dt.normalize()
    df["home_team"] = normalize_team(df["home_team"])
    df["away_team"] = normalize_team(df["away_team"])
    # The home side pitches in the top of the inning.
    top = df["inning_topbot"] == "Top"
    df["pitching_team"] = np.where(top, df["home_team"], df["away_team"])
    df["batting_team"] = np.where(top, df["away_team"], df["home_team"])
    return df


def _starters(df: pd.DataFrame) -> pd.DataFrame:
    """The first pitcher each side used in each game."""
    first = df.sort_values(["at_bat_number", "pitch_number"]).groupby(["game_pk", "pitching_team"]).head(1)
    return first[["game_pk", "pitching_team", "pitcher"]]


def build_game_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per completed regular-season game: date, teams, final score,
    both starting pitchers and whether the home team won."""
    if df.empty:
        return pd.DataFrame(columns=["game_pk", "game_date", "home_team", "away_team", "home_score",
                                     "away_score", "home_sp", "away_sp", "home_win"])
    finals = (
        df.assign(total=df["post_home_score"] + df["post_away_score"])
        .sort_values(["game_pk", "total", "at_bat_number", "pitch_number"])
        .groupby("game_pk")
        .tail(1)[["game_pk", "game_date", "home_team", "away_team", "post_home_score", "post_away_score"]]
        .rename(columns={"post_home_score": "home_score", "post_away_score": "away_score"})
    )
    starters = _starters(df)
    games = finals.merge(
        starters.rename(columns={"pitching_team": "home_team", "pitcher": "home_sp"}),
        on=["game_pk", "home_team"], how="left",
    ).merge(
        starters.rename(columns={"pitching_team": "away_team", "pitcher": "away_sp"}),
        on=["game_pk", "away_team"], how="left",
    )
    # A tie in the data is a game Statcast did not record to the end.
    games = games[games["home_score"] != games["away_score"]].copy()
    games["home_win"] = (games["home_score"] > games["away_score"]).astype(int)
    return games.sort_values(["game_date", "game_pk"]).reset_index(drop=True)


def build_pa_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per plate appearance: expected wOBA value, strikeout-minus-
    walk value, and whether the pitcher was his side's starter."""
    columns = ["game_pk", "game_date", "pitcher", "is_starter", "pitching_team",
               "xwoba_value", "woba_denom", "k_minus_bb"]
    pa = df[df["events"].notna() & (df["woba_denom"].fillna(0) > 0)].copy()
    if pa.empty:
        return pd.DataFrame(columns=columns)
    pa = pa.merge(_starters(df).assign(is_starter=True), on=["game_pk", "pitching_team", "pitcher"], how="left")
    pa["is_starter"] = pa["is_starter"].eq(True)
    # Statcast's xwOBA where the ball was put in play; walks, strikeouts and
    # hit batters keep their actual value, having no luck to strip out.
    pa["xwoba_value"] = pa["estimated_woba_using_speedangle"].where(
        pa["estimated_woba_using_speedangle"].notna(), pa["woba_value"]
    )
    pa["k_minus_bb"] = (
        pa["events"].isin(["strikeout", "strikeout_double_play"]).astype(float)
        - pa["events"].isin(["walk", "intent_walk"]).astype(float)
    )
    return pa[columns]


def _ew_asof(daily: pd.DataFrame, key: str, value_cols: list, half_life: float,
             query_keys: pd.Series, query_dates: pd.Series) -> pd.DataFrame:
    """Exponentially weighted sums of `value_cols` for each (key, date)
    query, over days strictly before that date.

    With t in days since _ORIGIN and tau = half_life / ln 2, the weighted
    sum at d is exp(-d/tau) * sum_{t<d} x_t exp(t/tau), so one cumulative
    sum per key answers every query date. A key with no history (a debut
    starter, an unknown probable pitcher) comes back as zeros, which the
    shrinkage then turns into the league rate."""
    tau = half_life / LN2
    q = pd.DataFrame({"_key": query_keys.to_numpy(), "game_date": pd.to_datetime(query_dates).to_numpy()})
    q["_row"] = np.arange(len(q))
    zeros = pd.DataFrame(0.0, index=range(len(q)), columns=value_cols)
    if daily.empty or q.empty:
        return zeros

    d = daily.rename(columns={key: "_key"})[["_key", "game_date"] + value_cols].copy()
    scale = np.exp((d["game_date"] - _ORIGIN).dt.days.to_numpy(dtype=float) / tau)
    for col in value_cols:
        d[col] = d[col].to_numpy(dtype=float) * scale
    d = d.sort_values("game_date")
    d[value_cols] = d.groupby("_key")[value_cols].cumsum()

    known = q["_key"].notna()
    found = q[known].copy()
    found["_key"] = found["_key"].astype(d["_key"].dtype)
    merged = pd.merge_asof(
        found.sort_values("game_date"), d, on="game_date", by="_key",
        allow_exact_matches=False, direction="backward",
    )
    decay = np.exp(-(merged["game_date"] - _ORIGIN).dt.days.to_numpy(dtype=float) / tau)
    for col in value_cols:
        merged[col] = merged[col].fillna(0.0).to_numpy() * decay
    result = zeros.copy()
    result.loc[merged["_row"].to_numpy(), value_cols] = merged[value_cols].to_numpy()
    return result


def _shrunk(num, den, prior, k):
    return (num + k * prior) / (den + k)


def build_features(games: pd.DataFrame, history_games: pd.DataFrame, pa: pd.DataFrame,
                   params: dict = None) -> pd.DataFrame:
    """Pregame features for each row of `games` (game_date, home_team,
    away_team, home_sp, away_sp). Rows may be past games (training) or
    today's slate (prediction); either way only `history_games` and `pa`
    strictly before each game's date are used."""
    params = params or DEFAULT_PARAMS
    out = games[["game_date", "home_team", "away_team", "home_sp", "away_sp"]].copy().reset_index(drop=True)
    out["game_date"] = pd.to_datetime(out["game_date"]).dt.normalize()

    team_days = pd.concat([
        history_games.assign(team=history_games["home_team"],
                             rd=history_games["home_score"] - history_games["away_score"]),
        history_games.assign(team=history_games["away_team"],
                             rd=history_games["away_score"] - history_games["home_score"]),
    ])
    team_days = team_days.assign(g=1.0).groupby(["team", "game_date"], as_index=False)[["rd", "g"]].sum()

    starter_pa = pa[pa["is_starter"]]
    sp_days = starter_pa.groupby(["pitcher", "game_date"], as_index=False)[["k_minus_bb", "woba_denom"]].sum()
    pen_days = pa[~pa["is_starter"]].groupby(["pitching_team", "game_date"], as_index=False)[
        ["xwoba_value", "woba_denom"]
    ].sum()
    denom = pa["woba_denom"].sum()
    lg_xwoba = pa["xwoba_value"].sum() / denom if denom else 0.0
    starter_denom = starter_pa["woba_denom"].sum()
    lg_sp_kbb = starter_pa["k_minus_bb"].sum() / starter_denom if starter_denom else 0.0

    for side in ("home", "away"):
        rd = _ew_asof(team_days, "team", ["rd", "g"], params["team_half_life"], out[f"{side}_team"], out["game_date"])
        out[f"{side}_rd"] = _shrunk(rd["rd"], rd["g"], 0.0, params["rd_k"])

        sp = _ew_asof(sp_days, "pitcher", ["k_minus_bb", "woba_denom"], params["sp_half_life"],
                      out[f"{side}_sp"], out["game_date"])
        out[f"{side}_sp_kbb"] = _shrunk(sp["k_minus_bb"], sp["woba_denom"], lg_sp_kbb, params["sp_k"])

        pen = _ew_asof(pen_days, "pitching_team", ["xwoba_value", "woba_denom"], params["pen_half_life"],
                       out[f"{side}_team"], out["game_date"])
        out[f"{side}_pen_xwoba"] = _shrunk(pen["xwoba_value"], pen["woba_denom"], lg_xwoba, params["pen_k"])

    out["rd_diff"] = out["home_rd"] - out["away_rd"]
    out["sp_kbb_diff"] = (out["home_sp_kbb"] - out["away_sp_kbb"]) * 100
    # Lower xwOBA allowed is better pitching, so away minus home.
    out["pen_diff"] = (out["away_pen_xwoba"] - out["home_pen_xwoba"]) * 100
    return out


def prepare_history(statcast: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(games, plate appearances) from raw Statcast."""
    df = regular_season(statcast)
    return build_game_table(df), build_pa_table(df)


def fit_model(games: pd.DataFrame, pa: pd.DataFrame, params: dict = None) -> LogisticRegression | None:
    """Logistic regression on every completed game's pregame features.
    None when there are too few games of both outcomes to fit."""
    if len(games) < 100 or games["home_win"].nunique() < 2:
        return None
    features = build_features(games, games, pa, params)
    return LogisticRegression(C=1.0).fit(features[FEATURE_COLUMNS], games["home_win"])


def compute_game_win_probabilities(statcast: pd.DataFrame, schedule_games_df: pd.DataFrame,
                                   params: dict = None) -> pd.DataFrame | None:
    """[game_pk, date, home_team, away_team, home_win_probability] for each
    scheduled game, the drop-in replacement for
    game_picks.compute_game_win_probabilities. `statcast` is every
    persisted pitch before today (this season and last); team codes in the
    output are the schedule's own. None when there is not enough history
    to fit, so the caller can fall back rather than log invented numbers."""
    games, pa = prepare_history(statcast)
    model = fit_model(games, pa, params)
    if model is None:
        return None

    slate = pd.DataFrame({
        "game_date": pd.to_datetime(schedule_games_df["date"]),
        "home_team": normalize_team(schedule_games_df["home_team"]),
        "away_team": normalize_team(schedule_games_df["away_team"]),
        "home_sp": pd.to_numeric(schedule_games_df["home_probable_pitcher_key_mlbam"], errors="coerce"),
        "away_sp": pd.to_numeric(schedule_games_df["away_probable_pitcher_key_mlbam"], errors="coerce"),
    })
    features = build_features(slate, games, pa, params)
    result = schedule_games_df[["game_pk", "date", "home_team", "away_team"]].copy().reset_index(drop=True)
    result["home_win_probability"] = model.predict_proba(features[FEATURE_COLUMNS])[:, 1]
    return result

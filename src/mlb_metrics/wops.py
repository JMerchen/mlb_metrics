"""WOPS: a gradient-boosted expected OPS for a hitter's next game, from his
recent form and the starting pitcher he faces.

OPS = OBP + SLG, so WOPS is two models, both scikit-learn
HistGradientBoostingRegressors over the same features:
  - on-base per plate appearance ((H + BB + HBP) / PA), weighted by PA
  - total bases per at-bat, weighted by AB
and WOPS = predicted OBP + predicted SLG.

Features, all from games strictly before the one predicted:
  - the hitter's OBP and SLG over his last 7, 15 and 30 games and the
    season so far, with the plate appearances behind each
  - the opposing starter's OBP and SLG allowed over his last 5 starts and
    the season so far, with the batters faced behind each
  - whether the hitter has the platoon edge (opposite hands) and is at home

Rates are shrunk toward the league rate with pseudo-PA (WOPS_SHRINK_PA) so
a 3-for-3 week doesn't read as a 2.000 OPS. A starter with no history (or
no announced starter) gets the league rate.

Also builds `games_with_hit`: in each hitter's last N games, how many had
at least one hit (a 1-for-4 and a 4-for-4 both count as one game).
"""

import numpy as np
import pandas as pd

from . import config

STATCAST_COLUMNS = [
    "game_pk", "game_date", "game_type", "home_team", "away_team", "inning_topbot",
    "at_bat_number", "batter", "pitcher", "stand", "p_throws", "events",
]

HIT_EVENTS = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
WALK_EVENTS = {"walk", "intent_walk"}
NOT_AT_BAT = {"walk", "intent_walk", "hit_by_pitch", "sac_fly", "sac_bunt", "sac_fly_double_play",
              "sac_bunt_double_play", "catcher_interf", "truncated_pa"}

HITTER_WINDOWS = (7, 15, 30)
PITCHER_WINDOWS = (5,)

FEATURE_COLUMNS = (
    [f"hit_{stat}_{w}" for w in HITTER_WINDOWS + ("season",) for stat in ("obp", "slg", "pa")]
    + [f"sp_{stat}_{w}" for w in PITCHER_WINDOWS + ("season",) for stat in ("obp", "slg", "bf")]
    + ["platoon_edge", "is_home"]
)


def plate_appearances(statcast: pd.DataFrame) -> pd.DataFrame:
    """One row per completed regular-season plate appearance, with the
    batting team, the pitcher's team and the counting stats for OBP/SLG."""
    df = statcast[statcast["events"].notna() & (statcast.get("game_type", "R") == "R")].copy()
    df = df[df["events"] != "truncated_pa"]
    top = df["inning_topbot"] == "Top"
    df["bat_team"] = np.where(top, df["away_team"], df["home_team"])
    df["fld_team"] = np.where(top, df["home_team"], df["away_team"])
    df["is_home"] = (~top).astype(int)
    df["H"] = df["events"].map(HIT_EVENTS).notna().astype(int)
    df["TB"] = df["events"].map(HIT_EVENTS).fillna(0).astype(int)
    df["OB"] = (df["H"] + df["events"].isin(WALK_EVENTS) + (df["events"] == "hit_by_pitch")).astype(int)
    df["AB"] = (~df["events"].isin(NOT_AT_BAT)).astype(int)
    df["PA"] = 1
    return df


def game_starters(pas: pd.DataFrame) -> pd.DataFrame:
    """Each team's starting pitcher per game: the pitcher of its first PA in
    the field. [game_pk, fld_team, starter, starter_throws]."""
    first = pas.sort_values("at_bat_number").groupby(["game_pk", "fld_team"], as_index=False).first()
    return first[["game_pk", "fld_team", "pitcher", "p_throws"]].rename(columns={"pitcher": "starter", "p_throws": "starter_throws"})


def hitter_games(pas: pd.DataFrame) -> pd.DataFrame:
    """One row per hitter per game: date, team, opponent, home, his batting
    side against the opposing starter, and PA/AB/H/TB/OB."""
    starters = game_starters(pas)
    lines = pas.groupby(["batter", "game_pk", "game_date", "bat_team", "fld_team", "is_home"], as_index=False)[
        ["PA", "AB", "H", "TB", "OB"]].sum()
    # His side of the plate in his first PA of the game.
    side = pas.sort_values("at_bat_number").groupby(["batter", "game_pk"], as_index=False)["stand"].first()
    lines = lines.merge(side, on=["batter", "game_pk"], how="left")
    lines = lines.merge(starters, on=["game_pk", "fld_team"], how="left")
    return lines.sort_values(["batter", "game_date", "game_pk"]).reset_index(drop=True)


def starter_games(pas: pd.DataFrame) -> pd.DataFrame:
    """Each starter's line in each game he started (all hitters he faced):
    [pitcher, game_pk, game_date, BF, AB, TB, OB]."""
    starters = game_starters(pas)
    faced = pas.merge(starters, on=["game_pk", "fld_team"])
    faced = faced[faced["pitcher"] == faced["starter"]]
    lines = faced.groupby(["pitcher", "game_pk", "game_date"], as_index=False)[["PA", "AB", "TB", "OB"]].sum()
    return lines.rename(columns={"PA": "BF"}).sort_values(["pitcher", "game_date", "game_pk"]).reset_index(drop=True)


def league_rates(pas: pd.DataFrame) -> tuple[float, float]:
    """League OBP (per PA) and SLG (TB per AB)."""
    return pas["OB"].sum() / max(1, pas["PA"].sum()), pas["TB"].sum() / max(1, pas["AB"].sum())


def _shrunk(num, den, league, pseudo):
    return (num + league * pseudo) / (den + pseudo)


def _prior_window_sums(lines: pd.DataFrame, key: str, cols: list, window) -> pd.DataFrame:
    """Sums of `cols` over each row's previous `window` games for the same
    `key` (all previous games when window is "season"), excluding the row."""
    grouped = lines.groupby(key)[cols]
    if window == "season":
        sums = grouped.cumsum() - lines[cols]
    else:
        sums = grouped.transform(lambda s: s.shift(1).rolling(window, min_periods=1).sum()).fillna(0)
    return sums


def hitter_form(lines: pd.DataFrame, league_obp: float, league_slg: float, include_current: bool = False) -> pd.DataFrame:
    """Per hitter-game: shrunk OBP/SLG and PA over each window before that
    game (or through it, with include_current, for "as of now" features)."""
    out = pd.DataFrame(index=lines.index)
    pseudo = config.WOPS_SHRINK_PA
    cols = ["PA", "AB", "TB", "OB"]
    for window in HITTER_WINDOWS + ("season",):
        if include_current:
            grouped = lines.groupby("batter")[cols]
            sums = grouped.cumsum() if window == "season" else grouped.transform(lambda s: s.rolling(window, min_periods=1).sum())
        else:
            sums = _prior_window_sums(lines, "batter", cols, window)
        out[f"hit_obp_{window}"] = _shrunk(sums["OB"], sums["PA"], league_obp, pseudo)
        out[f"hit_slg_{window}"] = _shrunk(sums["TB"], sums["AB"], league_slg, pseudo)
        out[f"hit_pa_{window}"] = sums["PA"]
    return out


def starter_form(starts: pd.DataFrame, league_obp: float, league_slg: float, include_current: bool = False) -> pd.DataFrame:
    """Per starter-game: shrunk OBP/SLG allowed and batters faced over his
    previous starts (or through this one, with include_current)."""
    out = starts[["pitcher", "game_pk"]].copy()
    pseudo = config.WOPS_SHRINK_PA
    cols = ["BF", "AB", "TB", "OB"]
    for window in PITCHER_WINDOWS + ("season",):
        if include_current:
            grouped = starts.groupby("pitcher")[cols]
            sums = grouped.cumsum() if window == "season" else grouped.transform(lambda s: s.rolling(window, min_periods=1).sum())
        else:
            sums = _prior_window_sums(starts, "pitcher", cols, window)
        out[f"sp_obp_{window}"] = _shrunk(sums["OB"], sums["BF"], league_obp, pseudo)
        out[f"sp_slg_{window}"] = _shrunk(sums["TB"], sums["AB"], league_slg, pseudo)
        out[f"sp_bf_{window}"] = sums["BF"]
    return out


def _league_starter_features(league_obp, league_slg) -> dict:
    feats = {}
    for window in PITCHER_WINDOWS + ("season",):
        feats[f"sp_obp_{window}"] = league_obp
        feats[f"sp_slg_{window}"] = league_slg
        feats[f"sp_bf_{window}"] = 0
    return feats


def training_frame(statcast: pd.DataFrame) -> pd.DataFrame:
    """One row per hitter-game with FEATURE_COLUMNS (all pre-game), the
    targets' counts (PA, OB, AB, TB) and the date."""
    pas = plate_appearances(statcast)
    league_obp, league_slg = league_rates(pas)
    lines = hitter_games(pas)
    frame = pd.concat([lines, hitter_form(lines, league_obp, league_slg)], axis=1)
    starts = starter_games(pas)
    sp = starter_form(starts, league_obp, league_slg).rename(columns={"pitcher": "starter"})
    frame = frame.merge(sp, on=["starter", "game_pk"], how="left")
    for column, value in _league_starter_features(league_obp, league_slg).items():
        frame[column] = frame[column].fillna(value)
    frame["platoon_edge"] = (frame["stand"] != frame["starter_throws"]).astype(int)
    return frame


def fit(frame: pd.DataFrame):
    """(obp_model, slg_model) fit on `frame`'s hitter-games."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    params = dict(max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=200, l2_regularization=1.0)
    X = frame[FEATURE_COLUMNS]
    obp = HistGradientBoostingRegressor(**params).fit(X, frame["OB"] / frame["PA"], sample_weight=frame["PA"])
    has_ab = frame["AB"] > 0
    slg = HistGradientBoostingRegressor(**params).fit(
        X[has_ab], frame.loc[has_ab, "TB"] / frame.loc[has_ab, "AB"], sample_weight=frame.loc[has_ab, "AB"])
    return obp, slg


def predict(models, features: pd.DataFrame) -> pd.Series:
    obp_model, slg_model = models
    X = features[FEATURE_COLUMNS]
    return pd.Series(obp_model.predict(X) + slg_model.predict(X), index=features.index)


def realized_ops(frame: pd.DataFrame) -> pd.Series:
    """OPS of a set of hitter-games pooled: OB/PA + TB/AB."""
    return frame["OB"].sum() / frame["PA"].sum() + frame["TB"].sum() / max(1, frame["AB"].sum())


def current_features(statcast: pd.DataFrame, probables: pd.DataFrame | None = None) -> pd.DataFrame:
    """Each hitter's features for his next game, as of the end of
    `statcast`: his form through his latest game, against his team's
    probable starter today when `probables` (probable_pitchers.csv:
    team, opponent, is_home, key_mlbam, Throws) names one, else a
    league-average starter. Adds `opponent` and `starter` (blank when
    there's no game)."""
    pas = plate_appearances(statcast)
    league_obp, league_slg = league_rates(pas)
    lines = hitter_games(pas)
    form = pd.concat([lines, hitter_form(lines, league_obp, league_slg, include_current=True)], axis=1)
    latest = form.groupby("batter").tail(1).reset_index(drop=True)
    latest["team"] = latest["bat_team"]

    starts = starter_games(pas)
    sp_now = starter_form(starts, league_obp, league_slg, include_current=True).groupby("pitcher").tail(1)
    league_sp = _league_starter_features(league_obp, league_slg)

    latest["opponent"] = ""
    latest["starter"] = np.nan
    latest["starter_throws"] = ""
    latest["is_home"] = 0
    if probables is not None and len(probables):
        today = probables.rename(columns={"key_mlbam": "starter_id"})
        opp_starter = today[["team", "starter_id", "Throws"]].rename(
            columns={"team": "opponent", "starter_id": "starter", "Throws": "starter_throws"})
        games = today[["team", "opponent", "is_home"]].copy()
        games["is_home"] = games["is_home"].astype(str).str.lower().eq("true").astype(int)
        latest = latest.drop(columns=["opponent", "starter", "starter_throws", "is_home"]).merge(games, on="team", how="left")
        latest = latest.merge(opp_starter, on="opponent", how="left")
        latest["opponent"] = latest["opponent"].fillna("")
        latest["is_home"] = latest["is_home"].fillna(0).astype(int)
    latest = latest.merge(sp_now.rename(columns={"pitcher": "starter"}).drop(columns="game_pk"), on="starter", how="left")
    for column, value in league_sp.items():
        latest[column] = latest[column].fillna(value)
    latest["platoon_edge"] = np.where(latest["starter_throws"].fillna("").eq(""), 0,
                                      (latest["stand"] != latest["starter_throws"]).astype(int))
    return latest


def games_with_hit(statcast: pd.DataFrame, last_n: int) -> pd.DataFrame:
    """[key_mlbam, games_with_hit, games_recent]: of each hitter's last
    `last_n` games, how many had at least one hit."""
    lines = hitter_games(plate_appearances(statcast))
    recent = lines.groupby("batter").tail(last_n)
    out = recent.groupby("batter").agg(games_with_hit=("H", lambda h: int((h > 0).sum())), games_recent=("H", "size"))
    return out.reset_index().rename(columns={"batter": "key_mlbam"})

"""Stuff+ and Decision+, from the stored Statcast pitches.

Stuff+ grades a pitch by its physical traits alone - how hard, how it
moves, how it spins, where and how it's released, its approach angle, and
how it differs from the pitcher's own fastball - never where it went or what
happened to it. A gradient-boosted model per pitch group (fastballs,
breaking balls, offspeed) learns what each pitch is worth in runs
(Statcast's delta_run_exp, the change in run expectancy on the pitch) from
those traits - with each ball in play valued at what its exit velocity and
launch angle are usually worth, not what the defense happened to make of it
(that sharpens what the model learns: 2025 Stuff+ predicts 2026 whiff rate
at r = 0.36, 0.30 when trained on raw results). It's cross-fitted by pitcher: each pitch is graded by a model
that never saw that pitcher, so it can't memorize him. A pitcher's grade
for a pitch type is the average grade of his pitches of that type; 100 is
league average and every 10 points is one standard deviation between
pitchers (lower expected runs = higher Stuff+).

Decision+ grades a hitter's swing decisions. Every pitch he saw is placed by
its location relative to his own strike zone, the count, the outs, the
runners and the pitch type. League models give, for that spot, how often
hitters swing and what swinging and taking are each worth in runs (league
average outcomes, so his own contact skill doesn't count). Each pitch scores
what his choice was worth over the league's usual mix of choices there:
swing p and take 1 - p. Summed over his pitches (per 100), 100 is league
average and every 10 points is one standard deviation between hitters.
"""

import numpy as np
import pandas as pd

PITCH_GROUP = {"FF": "FB", "SI": "FB", "FC": "FB", "FA": "FB", "SL": "BR", "ST": "BR", "CU": "BR", "KC": "BR",
               "SV": "BR", "CS": "BR", "EP": "BR", "CH": "OS", "FS": "OS", "FO": "OS", "SC": "OS", "KN": "OS"}
FASTBALLS = ["FF", "SI", "FC"]
STUFF_FEATURES = ["velo", "hb", "ivb", "spin", "axis", "ext", "rel_x", "rel_z", "arm_angle", "vaa",
                  "velo_diff", "hb_diff", "ivb_diff"]
SWINGS = {"swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "hit_into_play", "foul_bunt",
          "missed_bunt", "bunt_foul_tip"}
TAKES = {"ball", "called_strike", "blocked_ball", "hit_by_pitch"}
DECISION_FEATURES = ["xb", "zn", "balls", "strikes", "outs", "bases", "group_code", "same_hand"]
GROUP_CODE = {"FB": 0, "BR": 1, "OS": 2}
FOLDS = 5
MIN_PITCHES_TYPE = 100
MIN_PITCHES_PITCHER = 300
MIN_PITCHES_BATTER = 400
SCALE = 10.0
# Pseudo-pitches of league average in each grade, from how well it holds
# up between halves of a season (Stuff+ r = 0.91, Decision+ 0.78, at about
# 1,000 pitches a half).
STUFF_SHRINK = 100
DECISION_SHRINK = 300


def _approach_angle(df: pd.DataFrame) -> np.ndarray:
    """Vertical approach angle at the plate, in degrees (flatter = closer
    to 0), from the pitch's release velocity and acceleration."""
    vy_f = -np.sqrt(np.clip(df["vy0"] ** 2 - 2 * df["ay"] * (50 - 17 / 12), 0, None))
    t = (vy_f - df["vy0"]) / df["ay"]
    vz_f = df["vz0"] + df["az"] * t
    return -np.degrees(np.arctan(vz_f / vy_f)).to_numpy()


def stuff_features(p: pd.DataFrame) -> pd.DataFrame:
    """The pitch's physical traits, mirrored for lefties (so arm side is
    always positive), plus its differences from the pitcher's main fastball
    that season (his most-thrown of four-seam, sinker, cutter)."""
    lefty = (p["p_throws"] == "L").to_numpy()
    flip = np.where(lefty, -1.0, 1.0)
    f = pd.DataFrame(index=p.index)
    f["velo"] = p["release_speed"]
    f["hb"] = p["pfx_x"] * 12 * flip
    f["ivb"] = p["pfx_z"] * 12
    f["spin"] = p["release_spin_rate"]
    f["axis"] = np.where(lefty, 360 - p["spin_axis"], p["spin_axis"])
    f["ext"] = p["release_extension"]
    f["rel_x"] = p["release_pos_x"] * flip
    f["rel_z"] = p["release_pos_z"]
    f["arm_angle"] = p["arm_angle"]
    f["vaa"] = _approach_angle(p)
    season = p["game_date"].dt.year
    fb = p[p["pitch_type"].isin(FASTBALLS)].assign(season=season)
    main = fb.groupby(["pitcher", "season", "pitch_type"]).size().rename("n").reset_index()
    main = main.sort_values("n").drop_duplicates(["pitcher", "season"], keep="last")[["pitcher", "season", "pitch_type"]]
    ref = fb.merge(main, on=["pitcher", "season", "pitch_type"])
    ref = ref.assign(hb=ref["pfx_x"] * 12 * np.where(ref["p_throws"] == "L", -1.0, 1.0), ivb=ref["pfx_z"] * 12)
    ref = ref.groupby(["pitcher", "season"]).agg(fb_velo=("release_speed", "mean"), fb_hb=("hb", "mean"), fb_ivb=("ivb", "mean"))
    keys = pd.MultiIndex.from_arrays([p["pitcher"], season])
    joined = ref.reindex(keys)
    f["velo_diff"] = f["velo"].to_numpy() - joined["fb_velo"].to_numpy()
    f["hb_diff"] = f["hb"].to_numpy() - joined["fb_hb"].to_numpy()
    f["ivb_diff"] = f["ivb"].to_numpy() - joined["fb_ivb"].to_numpy()
    return f


def _model(kind: str, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

    common = dict(max_iter=300, learning_rate=0.08, max_leaf_nodes=48, min_samples_leaf=400, l2_regularization=1.0,
                  random_state=seed)
    return HistGradientBoostingClassifier(**common) if kind == "classifier" else HistGradientBoostingRegressor(**common)


def expected_run_value(p: pd.DataFrame) -> np.ndarray:
    """Each pitch's run value, with a ball in play's replaced by what the
    league's balls in play with that exit velocity and launch angle, in that
    count and base-out state, were worth."""
    y = p["delta_run_exp"].to_numpy(dtype=float).copy()
    if not {"launch_speed", "launch_angle"} <= set(p.columns):
        return y
    bip = ((p["description"] == "hit_into_play") & p["launch_speed"].notna() & p["launch_angle"].notna()
           & p["delta_run_exp"].notna()).to_numpy()
    if bip.sum() < 1000:
        return y
    bases = p["on_1b"].notna().astype(int) + 2 * p["on_2b"].notna().astype(int) + 4 * p["on_3b"].notna().astype(int)
    X = np.column_stack([p["launch_speed"], p["launch_angle"], p["balls"], p["strikes"], p["outs_when_up"], bases])
    model = _model("regressor").fit(X[bip], y[bip])
    y[bip] = model.predict(X[bip])
    return y


def stuff_predictions(p: pd.DataFrame, features: pd.DataFrame, folds: int = FOLDS) -> np.ndarray:
    """Each pitch's expected run value from its traits, cross-fitted by
    pitcher within each pitch group (NaN for pitches without a group)."""
    pred = np.full(len(p), np.nan)
    group = p["pitch_type"].map(PITCH_GROUP).to_numpy()
    y = expected_run_value(p)
    pitcher = p["pitcher"].to_numpy()
    fold_of = pd.Series(pitcher).map(lambda x: hash(int(x)) % folds).to_numpy()
    X = features[STUFF_FEATURES].to_numpy(dtype=float)
    for g in ("FB", "BR", "OS"):
        mine = (group == g) & np.isfinite(y)
        for k in range(folds):
            train = mine & (fold_of != k)
            test = mine & (fold_of == k)
            if test.sum() == 0 or train.sum() < 200:
                continue
            model = _model("regressor", seed=k).fit(X[train], y[train])
            pred[test] = model.predict(X[test])
    return pred


def _plus(values: pd.Series, weights: pd.Series, lower_is_better: bool, shrink: float = 0.0) -> pd.Series:
    """100 + 10 per weighted standard deviation from the weighted mean,
    each value first pulled toward the mean with `shrink` pseudo-pitches."""
    w = weights.to_numpy(dtype=float)
    v = values.to_numpy(dtype=float)
    mean = np.average(v, weights=w)
    v = mean + (v - mean) * w / (w + shrink)
    sd = np.sqrt(np.average((v - mean) ** 2, weights=w))
    z = (v - mean) / sd if sd > 0 else np.zeros_like(v)
    return pd.Series(100 + SCALE * (-z if lower_is_better else z), index=values.index)


def stuff_plus(p: pd.DataFrame, pred: np.ndarray, season: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(by pitch type: pitcher, pitch_type, pitches, stuff_rv, stuff_plus;
    by pitcher: pitcher, pitches, stuff_rv, stuff_plus), over `season` if
    given. Pitch types need MIN_PITCHES_TYPE pitches, pitchers
    MIN_PITCHES_PITCHER; each is scaled against those that qualify."""
    q = p.assign(pred=pred)
    q = q[np.isfinite(q["pred"])]
    if season is not None:
        q = q[q["game_date"].dt.year == season]
    by_type = q.groupby(["pitcher", "pitch_type"]).agg(pitches=("pred", "size"), stuff_rv=("pred", "mean")).reset_index()
    by_type = by_type[by_type["pitches"] >= MIN_PITCHES_TYPE]
    by_type["stuff_plus"] = np.nan
    for t, idx in by_type.groupby("pitch_type").groups.items():
        rows = by_type.loc[idx]
        if len(rows) >= 5:
            by_type.loc[idx, "stuff_plus"] = _plus(rows["stuff_rv"], rows["pitches"], lower_is_better=True, shrink=STUFF_SHRINK)
    by_pitcher = q.groupby("pitcher").agg(pitches=("pred", "size"), stuff_rv=("pred", "mean")).reset_index()
    by_pitcher = by_pitcher[by_pitcher["pitches"] >= MIN_PITCHES_PITCHER]
    by_pitcher["stuff_plus"] = _plus(by_pitcher["stuff_rv"], by_pitcher["pitches"], lower_is_better=True, shrink=STUFF_SHRINK)
    return by_type, by_pitcher


def decision_frame(p: pd.DataFrame) -> pd.DataFrame:
    """Pitches a hitter decided on (a swing or a take, no bunts or
    pitchouts), with the decision features and run value."""
    desc = p["description"]
    keep = desc.isin(SWINGS | TAKES) & ~desc.isin({"foul_bunt", "missed_bunt", "bunt_foul_tip"})
    q = p[keep & p["plate_x"].notna() & p["plate_z"].notna() & (p["sz_top"] > p["sz_bot"]) & p["delta_run_exp"].notna()]
    group = q["pitch_type"].map(PITCH_GROUP)
    q = q[group.notna()]
    f = pd.DataFrame(index=q.index)
    f["xb"] = np.where(q["stand"] == "R", -q["plate_x"], q["plate_x"])
    f["zn"] = (q["plate_z"] - q["sz_bot"]) / (q["sz_top"] - q["sz_bot"])
    f["balls"] = q["balls"].astype(float)
    f["strikes"] = q["strikes"].astype(float)
    f["outs"] = q["outs_when_up"].astype(float)
    f["bases"] = (q["on_1b"].notna().astype(int) + 2 * q["on_2b"].notna().astype(int) + 4 * q["on_3b"].notna().astype(int)).astype(float)
    f["group_code"] = q["pitch_type"].map(PITCH_GROUP).map(GROUP_CODE).astype(float)
    f["same_hand"] = (q["stand"] == q["p_throws"]).astype(float)
    f["swing"] = q["description"].isin(SWINGS).astype(int)
    f["rv"] = q["delta_run_exp"].astype(float)
    f["batter"] = q["batter"]
    f["game_date"] = q["game_date"]
    return f


def decision_models(f: pd.DataFrame):
    """(swing chance, run value of a swing, run value of a take) models,
    fitted on the league."""
    X = f[DECISION_FEATURES].to_numpy(dtype=float)
    swing = f["swing"].to_numpy()
    p_swing = _model("classifier").fit(X, swing)
    rv_swing = _model("regressor").fit(X[swing == 1], f["rv"].to_numpy()[swing == 1])
    rv_take = _model("regressor").fit(X[swing == 0], f["rv"].to_numpy()[swing == 0])
    return p_swing, rv_swing, rv_take


def decision_values(f: pd.DataFrame, models) -> pd.DataFrame:
    """Each pitch's decision value: what the hitter's choice was worth in
    league-average runs over the league's usual mix of choices there."""
    p_swing, rv_swing, rv_take = models
    X = f[DECISION_FEATURES].to_numpy(dtype=float)
    ps = p_swing.predict_proba(X)[:, 1]
    es = rv_swing.predict(X)
    et = rv_take.predict(X)
    chosen = np.where(f["swing"].to_numpy() == 1, es, et)
    return f.assign(p_swing=ps, ev_swing=es, ev_take=et, value=chosen - (ps * es + (1 - ps) * et),
                    right=(np.where(es > et, 1, 0) == f["swing"].to_numpy()).astype(int))


def decision_plus(values: pd.DataFrame, season: int | None = None) -> pd.DataFrame:
    """By hitter: pitches, decision runs per 100 pitches, how often he made
    the choice worth more, swing rate, and Decision+ (hitters with
    MIN_PITCHES_BATTER pitches)."""
    v = values if season is None else values[values["game_date"].dt.year == season]
    out = v.groupby("batter").agg(pitches=("value", "size"), runs=("value", "sum"), right=("right", "mean"),
                                  swing=("swing", "mean")).reset_index()
    zone = v.assign(out_zone=(v["xb"].abs() > 0.83) | (v["zn"] < 0) | (v["zn"] > 1))
    chase = zone[zone["out_zone"]].groupby("batter")["swing"].mean()
    in_zone = zone[~zone["out_zone"]].groupby("batter")["swing"].mean()
    out["chase"] = out["batter"].map(chase)
    out["zone_swing"] = out["batter"].map(in_zone)
    out["runs_per_100"] = 100 * out["runs"] / out["pitches"]
    out = out[out["pitches"] >= MIN_PITCHES_BATTER].copy()
    out["decision_plus"] = _plus(out["runs_per_100"], out["pitches"], lower_is_better=False, shrink=DECISION_SHRINK)
    return out

"""Features for the Beat the Streak hit-probability model
(config.HITTER_HIT_PROBABILITY_MODEL_PATH, trained by
scripts/train_hitter_hit_model.py): build_hitter_features assembles each
batter's raw ingredients - his own form (WAVE and splits), the opposing
starter's and bullpen's PAVE, the venue's Park_Factor, is_home, Statcast
contact quality (Exit_Velo/Barrel_Rate/xBA/xwOBA), pitch-family rates
against the starter's pitch mix, and plate discipline (Whiff_Rate/
Chase_Rate) - and hitter_feature_matrix turns them into the model's input,
every not-yet-computable value filled with 0. The same no-lookahead
discipline as dfs_backtest.assemble_ml_training_rows: each row's features
come only from data strictly before that row's game date.

**Retraining note**: whenever HITTER_FEATURE_COLUMNS changes, a model
trained on the older schema is incompatible and must be retrained or
deleted - predict_hitter_hit_probability's predict_proba() call isn't
wrapped in a try/except, so a stale artifact would crash live daily picks
rather than fall back.

(The module once also held DFS fantasy-point models; the DFS pages and
their models are gone.)
"""

import pandas as pd

from mlb_metrics import config, ml_models

HITTER_FEATURE_COLUMNS = [
    "WAVE", "WAVE_L", "WAVE_R", "PA_L", "PA_R", "probability",
    "Game_Hit_Probability", "Consistency", "Approach", "Expected_Bases",
    "Expected_BB", "Expected_HBP", "Expected_RBI",
    "Exit_Velo", "Barrel_Rate", "xBA", "xwOBA",
    "Fastball_WAVE", "Breaking_WAVE", "Offspeed_WAVE",
    "Whiff_Rate", "Chase_Rate",
    "WAVE_Home", "WAVE_Away",
    "starter_PAVE", "Bullpen_PAVE", "Park_Factor", "is_home",
    "starter_fastball_rate", "starter_breaking_rate", "starter_offspeed_rate",
    "Matchup_Hit_Probability",
]

PITCHER_FEATURE_COLUMNS = [
    "starts", "K9", "BB9", "HR9", "IP_per_start", "Expected_IP",
    "Expected_K", "Expected_H_Allowed", "FIP_Windowed",
]


def build_hitter_features(
    wave: pd.DataFrame,
    pave: pd.DataFrame,
    confidence: pd.DataFrame,
    schedule_df: pd.DataFrame,
    matchup_probability: pd.DataFrame,
) -> pd.DataFrame:
    """One row per hitter with a game today (schedule_df) AND a resolvable
    Matchup_Hit_Probability - the same qualifying join
    dfs.compute_hitter_dk_points uses - carrying key_mlbam plus every
    column in HITTER_FEATURE_COLUMNS. Used identically at training-table
    assembly time (dfs_backtest.assemble_ml_training_rows) and at live
    prediction time (apply_ml_overrides), so training and serving can
    never drift apart."""
    schedule_columns = [c for c in ("team", "opponent", "probable_pitcher_key_mlbam", "is_home") if c in schedule_df.columns]
    df = wave.merge(schedule_df[schedule_columns], on="team", how="inner")

    # Falls back to the hand-blended overall WAVE when WAVE_L/WAVE_R aren't
    # present at all - an old wave.csv snapshot predating platoon-awareness,
    # same known scenario matchup.py's _platoon_wave already guards against.
    for column in ("WAVE_L", "WAVE_R"):
        if column not in df.columns:
            df[column] = df["WAVE"]

    # A batter's own pitch-family rates (an even older wave.csv snapshot
    # predating hitters.compute_pitch_family_rates) degrade to null rather
    # than a crash - hitter_feature_matrix's blanket fillna(0) then reads
    # this exactly like "no family data", the same convention every other
    # optional feature column here already uses.
    for column in ("Fastball_WAVE", "Breaking_WAVE", "Offspeed_WAVE"):
        if column not in df.columns:
            df[column] = pd.NA

    # A batter's own home/road split (hitters.compute_home_road_split, an
    # even older wave.csv snapshot predating it) degrades to null the same
    # way - same convention as the pitch-family columns just above.
    for column in ("WAVE_Home", "WAVE_Away"):
        if column not in df.columns:
            df[column] = pd.NA

    # A batter's own Whiff_Rate/Chase_Rate (hitters.compute_plate_discipline,
    # an OPTIONAL assemble_hitters input - absent whenever the caller didn't
    # pass all_pitches) degrade to null the same way - never a KeyError just
    # because the live pipeline's own optional wiring wasn't exercised.
    for column in ("Whiff_Rate", "Chase_Rate"):
        if column not in df.columns:
            df[column] = pd.NA

    pave_columns = [
        c for c in ("key_mlbam", "PAVE", "Fastball_Rate", "Breaking_Rate", "Offspeed_Rate") if c in pave.columns
    ]
    starter = pave[pave_columns].rename(
        columns={
            "key_mlbam": "probable_pitcher_key_mlbam", "PAVE": "starter_PAVE",
            "Fastball_Rate": "starter_fastball_rate", "Breaking_Rate": "starter_breaking_rate",
            "Offspeed_Rate": "starter_offspeed_rate",
        }
    )
    df = df.merge(starter, on="probable_pitcher_key_mlbam", how="left")
    for column in ("starter_fastball_rate", "starter_breaking_rate", "starter_offspeed_rate"):
        if column not in df.columns:
            df[column] = pd.NA

    if "Bullpen_PAVE" in confidence.columns:
        bullpen = confidence[["team", "Bullpen_PAVE"]].rename(columns={"team": "opponent"})
        df = df.merge(bullpen, on="opponent", how="left")
    else:
        df["Bullpen_PAVE"] = pd.NA

    if "Park_Factor" in confidence.columns and "is_home" in df.columns:
        df["venue_team"] = df["team"].where(df["is_home"], df["opponent"])
        park = confidence[["team", "Park_Factor"]].rename(columns={"team": "venue_team"})
        df = df.merge(park, on="venue_team", how="left")
    else:
        df["Park_Factor"] = pd.NA

    df = df.merge(matchup_probability[["key_mlbam", "Matchup_Hit_Probability"]], on="key_mlbam", how="inner")
    return df[["key_mlbam"] + HITTER_FEATURE_COLUMNS]


def hitter_feature_matrix(features_df: pd.DataFrame) -> pd.DataFrame:
    """The numeric X matrix a hitter model trains/predicts on - is_home
    coerced to float (0/1), everything else NaN-filled to 0 (an
    unannounced/unmatched opposing starter's starter_PAVE, e.g.) rather
    than dropped, so a row is never silently excluded from prediction just
    because one matchup ingredient is missing."""
    X = features_df.reindex(columns=HITTER_FEATURE_COLUMNS).copy()
    X["is_home"] = X["is_home"].fillna(False).astype(float)
    return X.fillna(0)


def predict_hitter_hit_probability(hitter_features: pd.DataFrame) -> pd.DataFrame:
    """[key_mlbam, Model_Hit_Probability] - the trained hit-probability
    classifier's (config.HITTER_HIT_PROBABILITY_MODEL_PATH,
    scripts/train_hitter_hit_model.py) predicted probability for each row
    in `hitter_features` (build_hitter_features's own output - same
    train/serve feature parity apply_ml_overrides relies on). Returns an
    EMPTY frame (never raises) if the model artifact is missing/corrupt/a
    version mismatch - same ml_models.load_model fallback discipline as
    apply_ml_overrides, so a caller can print/skip rather than crash. This
    model is a classifier, so `.predict_proba(X)[:, 1]` (probability of the
    positive class), not apply_ml_overrides's regression-style `.predict()`."""
    model = ml_models.load_model(config.HITTER_HIT_PROBABILITY_MODEL_PATH)
    if model is None or hitter_features.empty:
        return pd.DataFrame(columns=["key_mlbam", "Model_Hit_Probability"])

    X = hitter_feature_matrix(hitter_features)
    predicted = model.predict_proba(X)[:, 1]
    return pd.DataFrame({"key_mlbam": hitter_features["key_mlbam"].values, "Model_Hit_Probability": predicted})

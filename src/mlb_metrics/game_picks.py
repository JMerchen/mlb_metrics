"""Automated Game Picks: predicts a winner for each of today's games from
team-level metrics - not hitter picks (see predictions.py/matchup.py for
those). Uses exactly the signal set requested: each team's Pythagorean
strength, Pythagorean confidence, suppression resistance, and true power
(config.GAME_PICK_COMPOSITE_WEIGHTS), adjusted by the specific pitching
(probable starter + bullpen PAVE, and - see below - Power_A_PLUS) they're
projected to face today.

suppression_resistance is deliberately weighted both directly and inside
true_power (which already averages it with offensive_edge) - both signals
were named explicitly, so the overlap is intentional, not a bug.

This is a first-pass, unvalidated blend, same spirit as matchup.py: meant
to be logged and tracked (see game_predictions.py/game_evaluation.py) and
compared against reality before ever being trusted. There is no way to
backtest this against past dates - schedule/game data has never been
persisted to git history the way wave.csv is (see git_backtest.py) - so it
can only accumulate a resolved dataset forward from the day it ships.

Pitching quality blends TWO complementary signals: PAVE_PLUS (hit-rate
against) and Power_A_PLUS (total-bases-allowed rate against - see
pitchers.py's module docstring for why PAVE_PLUS alone misses a pitcher's
run-prevention/ERA-like quality). config.GAME_PICK_SUSCEPTIBILITY_WEIGHT
controls how much Power_A_PLUS contributes to the blend - validated (not
just theorized) via a 35-day backtest reconstructing confidence.csv/pave.csv
directly from persisted Statcast (safe now that data.assign_game_ids no
longer fragments games - see its docstring) and replaying
compute_game_win_probabilities against real final scores at several
candidate weights: an equal 0.5/0.5 PAVE_PLUS/Power_A_PLUS blend won on both
accuracy (56.3% vs 54.3% for PAVE_PLUS alone, n=70-80 picks over ~31 days)
and Brier score (0.2493 vs 0.2517), beating both a lighter 0.25 blend and a
pure Power_A_PLUS (w=1.0) alternative. This is the direct fix for a real
gap the equal-weighted PAVE_PLUS-only version had: a start with an elite
run-prevention profile (few extra-base hits allowed) but only a middling
hit-rate barely moved the old pick.
"""

import pandas as pd

from mlb_metrics import config, matchup, ml_models

GAME_PICK_FEATURE_COLUMNS = [
    "home_composite",
    "away_composite",
    "home_bullpen_pave_plus",
    "home_bullpen_power_a_plus",
    "away_bullpen_pave_plus",
    "away_bullpen_power_a_plus",
    "home_starter_pave_plus",
    "home_starter_power_a_plus",
    "away_starter_pave_plus",
    "away_starter_power_a_plus",
]


def _team_composite(confidence: pd.DataFrame) -> pd.DataFrame:
    """One row per team: [team, composite] - the equal-weighted blend of
    config.GAME_PICK_COMPOSITE_WEIGHTS' four columns. All four inputs are
    already z-normalized to mean 1.0 (config.NORMALIZATION_Z_SCALE), so a
    straight weighted sum needs no further rescaling."""
    composite = sum(confidence[col] * weight for col, weight in config.GAME_PICK_COMPOSITE_WEIGHTS)
    return pd.DataFrame({"team": confidence["team"], "composite": composite})


def _safe_column(df: pd.DataFrame, key_col: str, value_col: str, new_name: str) -> pd.DataFrame:
    """Returns a 2-column frame [key_col, new_name], defaulting to all-NA
    when `value_col` is missing from `df` entirely (not just sparse) -
    handles confidence.csv/pave.csv snapshots from before a column existed
    (e.g. replayed via game_picks_backtest.py). A missing column degrades to
    the same neutral-per-row handling matchup.clip_and_blend_pitching_quality
    already does for a missing/null value."""
    if value_col in df.columns:
        return df[[key_col, value_col]].rename(columns={value_col: new_name})
    return pd.DataFrame({key_col: df[key_col], new_name: pd.NA})


def _blend_pitching_quality(pave_quality: pd.Series, power_a_quality: pd.Series) -> pd.Series:
    """Combines the hit-rate-based (PAVE_PLUS) and damage-rate-based
    (Power_A_PLUS) pitching-quality multipliers via
    config.GAME_PICK_SUSCEPTIBILITY_WEIGHT."""
    weight = config.GAME_PICK_SUSCEPTIBILITY_WEIGHT
    return (1 - weight) * pave_quality + weight * power_a_quality


def build_game_features(
    confidence: pd.DataFrame,
    pave: pd.DataFrame,
    schedule_games_df: pd.DataFrame,
) -> pd.DataFrame:
    """Returns [game_pk, date, home_team, away_team] + GAME_PICK_FEATURE_COLUMNS
    - the raw, unblended per-team/matchup ingredients
    compute_game_win_probabilities combines into home_win_probability,
    exposed separately so a model can learn its own combination instead of
    inheriting the ratio assumption (same reasoning as
    dfs_ml.build_hitter_features)."""
    composite = _team_composite(confidence)
    # Bullpen_PAVE_PLUS/Bullpen_Power_A_PLUS were added to confidence.csv at
    # different points in this project's history (see matchup.py,
    # pitchers.py) - older snapshots (e.g. replayed via
    # game_picks_backtest.py) can lack a column entirely, not just have
    # missing values in it, so _safe_column degrades that to the same
    # neutral-per-row handling clip_and_blend_pitching_quality already does
    # for a missing value.
    bullpen = _safe_column(confidence, "team", "Bullpen_PAVE_PLUS", "Bullpen_PAVE_PLUS").merge(
        _safe_column(confidence, "team", "Bullpen_Power_A_PLUS", "Bullpen_Power_A_PLUS"), on="team"
    )
    starter_pave = _safe_column(pave, "key_mlbam", "PAVE_PLUS", "starter_pave_plus").merge(
        _safe_column(pave, "key_mlbam", "Power_A_PLUS", "starter_power_a_plus"), on="key_mlbam"
    )

    games = schedule_games_df.merge(
        composite.rename(columns={"team": "home_team", "composite": "home_composite"}), on="home_team", how="left"
    ).merge(
        composite.rename(columns={"team": "away_team", "composite": "away_composite"}), on="away_team", how="left"
    )

    games = games.merge(
        bullpen.rename(columns={
            "team": "home_team",
            "Bullpen_PAVE_PLUS": "home_bullpen_pave_plus",
            "Bullpen_Power_A_PLUS": "home_bullpen_power_a_plus",
        }),
        on="home_team", how="left",
    ).merge(
        bullpen.rename(columns={
            "team": "away_team",
            "Bullpen_PAVE_PLUS": "away_bullpen_pave_plus",
            "Bullpen_Power_A_PLUS": "away_bullpen_power_a_plus",
        }),
        on="away_team", how="left",
    )

    games = games.merge(
        starter_pave.rename(columns={
            "key_mlbam": "home_probable_pitcher_key_mlbam",
            "starter_pave_plus": "home_starter_pave_plus",
            "starter_power_a_plus": "home_starter_power_a_plus",
        }),
        on="home_probable_pitcher_key_mlbam", how="left",
    ).merge(
        starter_pave.rename(columns={
            "key_mlbam": "away_probable_pitcher_key_mlbam",
            "starter_pave_plus": "away_starter_pave_plus",
            "starter_power_a_plus": "away_starter_power_a_plus",
        }),
        on="away_probable_pitcher_key_mlbam", how="left",
    )

    return games[["game_pk", "date", "home_team", "away_team"] + GAME_PICK_FEATURE_COLUMNS]


def game_feature_matrix(features_df: pd.DataFrame) -> pd.DataFrame:
    """Numeric X matrix, NaN-filled to 0 - mirrors dfs_ml.hitter_feature_matrix."""
    return features_df.reindex(columns=GAME_PICK_FEATURE_COLUMNS).copy().fillna(0)


def compute_game_win_probabilities(
    confidence: pd.DataFrame,
    pave: pd.DataFrame,
    schedule_games_df: pd.DataFrame,
) -> pd.DataFrame:
    """Returns [game_pk, date, home_team, away_team, home_win_probability] -
    one row per game in `schedule_games_df` (see schedule.normalize_schedule_games).

    Each team's rating is its own offensive composite multiplied by the
    OPPOSING team's pitching-weakness multiplier (probable starter's
    PAVE_PLUS blended with that team's Bullpen_PAVE_PLUS, via
    matchup.clip_and_blend_pitching_quality - the same reasoning as the
    batter-level matchup blend, applied at the team level: a higher
    opposing PAVE_PLUS means easier pitching to score against, so it boosts
    this team's rating). home_win_probability is a simple ratio of the two
    teams' ratings, not a log5-style formula - these composites aren't
    calibrated win percentages, so a ratio is the honestly-explainable
    choice rather than borrowing false precision from a formula built for a
    different kind of input. Both ratings are floored to
    config.GAME_PICK_RATING_FLOOR before dividing, purely as a
    degenerate-input guard.
    """
    games = build_game_features(confidence, pave, schedule_games_df)

    # Home team faces the AWAY team's pitching, and vice versa.
    home_pave_quality = matchup.clip_and_blend_pitching_quality(
        games["away_starter_pave_plus"], games["away_bullpen_pave_plus"]
    )
    home_power_a_quality = matchup.clip_and_blend_pitching_quality(
        games["away_starter_power_a_plus"], games["away_bullpen_power_a_plus"]
    )
    home_pitching_quality_faced = _blend_pitching_quality(home_pave_quality, home_power_a_quality)

    away_pave_quality = matchup.clip_and_blend_pitching_quality(
        games["home_starter_pave_plus"], games["home_bullpen_pave_plus"]
    )
    away_power_a_quality = matchup.clip_and_blend_pitching_quality(
        games["home_starter_power_a_plus"], games["home_bullpen_power_a_plus"]
    )
    away_pitching_quality_faced = _blend_pitching_quality(away_pave_quality, away_power_a_quality)

    home_rating = (games["home_composite"] * home_pitching_quality_faced).clip(lower=config.GAME_PICK_RATING_FLOOR)
    away_rating = (games["away_composite"] * away_pitching_quality_faced).clip(lower=config.GAME_PICK_RATING_FLOOR)

    games["home_win_probability"] = home_rating / (home_rating + away_rating)

    return games[["game_pk", "date", "home_team", "away_team", "home_win_probability"]]


def apply_calibration(win_probabilities: pd.DataFrame) -> pd.DataFrame:
    """Rescales `home_win_probability` through the saved recalibration at
    config.GAME_PICK_CALIBRATION_MODEL_PATH (ml_models.fit_probability_calibration,
    fit by scripts/train_game_pick_calibration.py) - quant-analytics
    follow-up "dig into calibration": the raw ratio compute_game_win_probabilities
    returns is explicitly NOT a calibrated probability (see that function's
    own docstring), and real data confirmed it - its spread (std 0.035) is
    far narrower than the real market's on the same games (std 0.059),
    which was the direct mechanical cause of a real bet-advice false-edge
    bug (see README's "Real quant sanity-check" section). This nudges the
    reported probability toward what real outcomes actually support,
    without changing which side is favored (`.predict()` for both isotonic
    and sigmoid methods is a real, monotonic transform of the input, never
    flips a > b to a real calibrated_a < calibrated_b - see
    ml_models.fit_probability_calibration's own docstring).

    Same graceful-degradation contract as dfs_ml.apply_ml_overrides: when
    no artifact exists yet (hasn't been trained, or the last training run
    didn't clear its own real-holdout bar - see the training script's own
    docstring), `win_probabilities` is returned completely UNCHANGED, same
    as before this function existed at all - never a crash, never a
    fabricated recalibration. Preserves every other column and row order;
    only `home_win_probability` itself is ever touched.

    A real, genuine edge case, not hypothetical: compute_game_win_probabilities
    can return NaN for a game whose home/away composite rating is itself
    NaN (e.g. a team missing from today's confidence.csv snapshot - clip()
    preserves NaN rather than flooring it) - both IsotonicRegression and
    the Platt/sigmoid wrapper's `.predict()` raise ValueError on a NaN
    input (confirmed directly against real sklearn, not assumed), and this
    function is called unconditionally in pipeline.run() (not inside the
    market-fetch try/except), so an unguarded call here would crash the
    ENTIRE daily pipeline run over one game's missing data - not just
    silently skip that game. Only the real, finite rows are ever passed to
    `.predict()`; a NaN row stays NaN, the same honest "no real calibrated
    output for no real input" contract the training script's own NaN-drop
    uses."""
    model = ml_models.load_model(config.GAME_PICK_CALIBRATION_MODEL_PATH)
    if model is None:
        return win_probabilities

    calibrated = win_probabilities.copy()
    real_valued = calibrated["home_win_probability"].notna()
    if real_valued.any():
        calibrated.loc[real_valued, "home_win_probability"] = model.predict(
            calibrated.loc[real_valued, "home_win_probability"].to_numpy()
        )
    return calibrated


def apply_kelly_uncertainty(win_probabilities: pd.DataFrame, confidence: pd.DataFrame) -> pd.DataFrame:
    """Adds home_win_probability_pessimistic/away_win_probability_pessimistic
    to `win_probabilities` - real bet-sizing follow-up (2026-08-25 -
    "we need the units risked to not be arbitrary"): supplements
    config.KELLY_FRACTION_MULTIPLIER's old flat, unconditional 0.5
    shrinkage (applied identically to every bet regardless of how solid
    the underlying estimate actually is) with a real, PER-GAME conservative
    probability grounded in each team's own real season-to-date win-rate
    Wilson confidence interval (teams.compute_team_win_rate_ci, carried on
    `confidence` via teams.assemble_team_metrics). Not a replacement for
    that flat multiplier (2026-08-26 - "the short-priced-favorite blowup
    risk": a team's season-long win-rate CI can stay genuinely tight even
    though a single game still carries real matchup-level uncertainty the
    season-level CI can't see) - game_predictions.advise_bets applies
    BOTH, stacked.

    Each team's CI half-width ((CI_High - CI_Low) / 2) is a real measure of
    how much a team's true quality could still deviate from its observed
    record - wide early in the season (few real games played), narrow
    once real games accumulate. The two teams' half-widths are combined
    via standard root-sum-square error propagation (the two records are
    independent real samples) into one real per-game uncertainty, then
    subtracted from the raw win_probability for whichever side is being
    considered - game_predictions.advise_bets sizes off THIS pessimistic
    probability instead of the raw point estimate (still scaled by
    kelly_fraction_multiplier on top), so kelly.kelly_fraction naturally
    computes a SMALLER (or zero) edge whenever the underlying team records
    are too thin to trust, with no new tunable constant beyond the
    already-established Wilson formula and this standard propagation
    math.

    A team missing from `confidence` (a real data gap, or an early-season
    date before any games are recorded) gets the same maximal real
    degenerate half-width helpers.wilson_ci's own n=0 case already uses
    ((1.0 - 0.0) / 2 = 0.5) - a genuinely unknown team is treated as
    maximally uncertain, not silently ordinary."""
    ci = confidence[["team", "win_rate_CI_Low", "win_rate_CI_High"]].copy()
    ci["ci_half_width"] = (ci["win_rate_CI_High"] - ci["win_rate_CI_Low"]) / 2

    result = win_probabilities.copy()
    result = result.merge(
        ci[["team", "ci_half_width"]].rename(columns={"team": "home_team", "ci_half_width": "home_ci_half_width"}),
        on="home_team", how="left",
    )
    result = result.merge(
        ci[["team", "ci_half_width"]].rename(columns={"team": "away_team", "ci_half_width": "away_ci_half_width"}),
        on="away_team", how="left",
    )
    result["home_ci_half_width"] = result["home_ci_half_width"].fillna(0.5)
    result["away_ci_half_width"] = result["away_ci_half_width"].fillna(0.5)

    combined_uncertainty = (result["home_ci_half_width"] ** 2 + result["away_ci_half_width"] ** 2) ** 0.5
    result["home_win_probability_pessimistic"] = (result["home_win_probability"] - combined_uncertainty).clip(0, 1)
    result["away_win_probability_pessimistic"] = (
        (1 - result["home_win_probability"]) - combined_uncertainty
    ).clip(0, 1)
    return result.drop(columns=["home_ci_half_width", "away_ci_half_width"])

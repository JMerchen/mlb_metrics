"""NFL Automated Game Picks: predicts a winner for each game in a real
schedule slate (upcoming, for live use, or historical, for
nfl_game_picks_backtest.py's no-lookahead replay) from
nfl_team_strength.py's team-level composite metrics - a direct structural
port of game_picks.py (see that module's own docstring for the original),
NOT a reimplementation. Uses config.NFL_GAME_PICK_COMPOSITE_WEIGHTS' equal
blend of pyth_Strength/pyth_Confidence/defensive_edge/true_power, adjusted
by the real QB-continuity signal for the SPECIFIC confirmed starter each
team faces that game - the honest NFL analog of MLB's probable-starter +
bullpen pitching-quality adjustment (see nfl_team_strength.py's own
docstring for why the underlying "who's actually been playing" detection
lives there while the game-specific COMPARISON lives here, mirroring how
MLB's own starter/bullpen adjustment lives in game_picks.py, not teams.py).

QB-continuity adjustment mechanics: nfl_team_strength.compute_qb_continuity_adjustment
already carries each team's own real recent-primary-QB `recent_primary_qb_epa`
(via nfl_passing.compute_qb_rolling_stats' real `passing_epa_per_game` -
the SAME source this module looks the game's own CONFIRMED starter up
against, via `schedule_games_df`'s real `home_qb_id`/`away_qb_id` -
confirmed live to already be gsis ids, the same id space
nfl_passing.compute_qb_rolling_stats keys on, so no crosswalk is needed on
this side). Both the confirmed starter's own epa and the team's
recent-primary-QB's own epa are z-normalized against the SAME population
(every team's own recent-primary QB that week - config.NFL_NORMALIZATION_Z_SCALE,
the same scale every other signal in this pipeline uses) before
differencing - so when the confirmed starter IS the team's own
recent-primary QB (the ordinary, no-injury case), the two epa lookups
resolve to the exact same real number and the adjustment is naturally
exactly 0. No separate "did the starter change" branch is needed; the math
already collapses to a no-op on its own. config.NFL_QB_CONTINUITY_WEIGHT
controls how much this shifts a team's composite rating - a real starting
point, pending nfl_game_picks_backtest.py's own real 2025 weeks-8-18
validation, same honest status as every other constant in this pipeline.

This is a first-pass, unvalidated blend, same spirit as MLB's own
game_picks.py before its own backtest existed: meant to be logged and
tracked (nfl_game_predictions.py/nfl_game_evaluation.py) and compared
against reality before ever being trusted.
"""

import pandas as pd

from mlb_metrics import config, ml_models, nfl_passing

GAME_PICK_FEATURE_COLUMNS = [
    "home_composite",
    "away_composite",
    "home_qb_adjustment",
    "away_qb_adjustment",
]


def _team_composite(master: pd.DataFrame) -> pd.DataFrame:
    """One row per team: [team, composite] - the equal-weighted blend of
    config.NFL_GAME_PICK_COMPOSITE_WEIGHTS' four columns. All four inputs
    are already z-normalized to mean 1.0 (config.NFL_NORMALIZATION_Z_SCALE),
    so a straight weighted sum needs no further rescaling - direct mirror
    of game_picks._team_composite."""
    composite = sum(master[col] * weight for col, weight in config.NFL_GAME_PICK_COMPOSITE_WEIGHTS)
    return pd.DataFrame({"team": master["team"], "composite": composite})


def _z_normalize(series: pd.Series, mean: float, std: float) -> pd.Series:
    """Real z-score against a caller-supplied (mean, std) - scaled by
    config.NFL_NORMALIZATION_Z_SCALE, same convention every other signal
    in this pipeline uses. Degrades to an all-0 (neutral) series on a
    degenerate zero-std population (e.g. a tiny test fixture, or every
    real recent-primary QB genuinely tied) rather than dividing by zero."""
    if not std or pd.isna(std):
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std * config.NFL_NORMALIZATION_Z_SCALE


def build_game_features(
    master: pd.DataFrame,
    qb_continuity: pd.DataFrame,
    weekly_df: pd.DataFrame,
    schedule_games_df: pd.DataFrame,
) -> pd.DataFrame:
    """Returns [game_id, season, week, home_team, away_team] + GAME_PICK_FEATURE_COLUMNS
    - the raw, unblended per-team/matchup ingredients compute_game_win_probabilities
    combines into home_win_probability, exposed separately (mirrors
    game_picks.build_game_features's own reasoning).

    `master` is nfl_team_strength.assemble_team_metrics' own output.
    `qb_continuity` is nfl_team_strength.compute_qb_continuity_adjustment's
    own output. `weekly_df` is the real per-player weekly stats
    (nfl_data.fetch_weekly_stats) used to look up the CONFIRMED starter's
    own rolling passing_epa_per_game. `schedule_games_df` needs
    [game_id, season, week, home_team, away_team, home_qb_id, away_qb_id]
    - a real schedules_*.parquet slice (see nfl_team_strength.build_team_record
    for the same REG-season/real-score filtering convention; this function
    does NOT filter on game_type or completion status itself, since an
    UPCOMING game has no score yet by definition - the caller decides which
    games to feature)."""
    composite = _team_composite(master)

    games = schedule_games_df.merge(
        composite.rename(columns={"team": "home_team", "composite": "home_composite"}), on="home_team", how="left"
    ).merge(
        composite.rename(columns={"team": "away_team", "composite": "away_composite"}), on="away_team", how="left"
    )

    recent_epa = qb_continuity.set_index("team")["recent_primary_qb_epa"]
    mean, std = qb_continuity["recent_primary_qb_epa"].mean(), qb_continuity["recent_primary_qb_epa"].std()

    confirmed_quality = nfl_passing.compute_qb_rolling_stats(weekly_df).set_index("player_id")[
        "passing_epa_per_game"
    ]

    def _adjustment(team_col: str, qb_id_col: str) -> pd.Series:
        recent_epa_for_team = games[team_col].map(recent_epa).fillna(0.0)
        confirmed_epa_for_qb = games[qb_id_col].map(confirmed_quality).fillna(0.0)
        return _z_normalize(confirmed_epa_for_qb, mean, std) - _z_normalize(recent_epa_for_team, mean, std)

    games["home_qb_adjustment"] = _adjustment("home_team", "home_qb_id")
    games["away_qb_adjustment"] = _adjustment("away_team", "away_qb_id")

    return games[["game_id", "season", "week", "home_team", "away_team"] + GAME_PICK_FEATURE_COLUMNS]


def game_feature_matrix(features_df: pd.DataFrame) -> pd.DataFrame:
    """Numeric X matrix, NaN-filled to 0 - mirrors game_picks.game_feature_matrix."""
    return features_df.reindex(columns=GAME_PICK_FEATURE_COLUMNS).copy().fillna(0)


def compute_game_win_probabilities(
    master: pd.DataFrame,
    qb_continuity: pd.DataFrame,
    weekly_df: pd.DataFrame,
    schedule_games_df: pd.DataFrame,
) -> pd.DataFrame:
    """Returns [game_id, season, week, home_team, away_team, home_win_probability]
    - one row per game in `schedule_games_df`.

    Each team's rating is its own composite PLUS its real
    config.NFL_QB_CONTINUITY_WEIGHT-weighted QB-continuity adjustment for
    THIS specific game (see module docstring). home_win_probability is a
    simple ratio of the two teams' ratings, not a log5-style formula -
    these composites aren't calibrated win percentages, so a ratio is the
    honestly-explainable choice, same reasoning as game_picks.compute_game_win_probabilities.
    Both ratings are floored to config.NFL_GAME_PICK_RATING_FLOOR before
    dividing, purely as a degenerate-input guard."""
    games = build_game_features(master, qb_continuity, weekly_df, schedule_games_df)

    home_rating = (
        games["home_composite"] + config.NFL_QB_CONTINUITY_WEIGHT * games["home_qb_adjustment"]
    ).clip(lower=config.NFL_GAME_PICK_RATING_FLOOR)
    away_rating = (
        games["away_composite"] + config.NFL_QB_CONTINUITY_WEIGHT * games["away_qb_adjustment"]
    ).clip(lower=config.NFL_GAME_PICK_RATING_FLOOR)

    games["home_win_probability"] = home_rating / (home_rating + away_rating)

    return games[["game_id", "season", "week", "home_team", "away_team", "home_win_probability"]]


def apply_calibration(win_probabilities: pd.DataFrame) -> pd.DataFrame:
    """Rescales `home_win_probability` through the saved recalibration at
    config.NFL_GAME_PICK_CALIBRATION_MODEL_PATH (ml_models.fit_probability_calibration,
    fit by scripts/train_nfl_game_pick_calibration.py) - direct mirror of
    game_picks.apply_calibration (see that function's own docstring for the
    full reasoning: graceful no-op when no artifact exists yet or the last
    training run didn't clear its own real-holdout bar, NaN-safe for a game
    missing a team's composite)."""
    model = ml_models.load_model(config.NFL_GAME_PICK_CALIBRATION_MODEL_PATH)
    if model is None:
        return win_probabilities

    calibrated = win_probabilities.copy()
    real_valued = calibrated["home_win_probability"].notna()
    if real_valued.any():
        calibrated.loc[real_valued, "home_win_probability"] = model.predict(
            calibrated.loc[real_valued, "home_win_probability"].to_numpy()
        )
    return calibrated

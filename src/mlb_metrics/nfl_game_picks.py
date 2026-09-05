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

Real follow-up (2026-09-04 - "a little push or pull from home/away"):
`compute_game_win_probabilities` now adds a single, real, GLOBAL
`config.NFL_HOME_FIELD_ADVANTAGE_WEIGHT` to the home side's own rating
before the win-probability ratio (there was previously NO home/away term
anywhere in this pipeline - confirmed live, a real gap, not a deliberate
omission). `_team_composite`/`build_game_features`/
`compute_game_win_probabilities` also gained explicit
`composite_weights`/`home_field_weight` override parameters (default
None -> the two config constants above) - see
scripts/backtest_nfl_season_carryover.py, which sweeps both alongside
nfl_team_strength.py's own season-carryover shrinkage.
"""

import pandas as pd

from mlb_metrics import config, ml_models, nfl_passing

GAME_PICK_FEATURE_COLUMNS = [
    "home_composite",
    "away_composite",
    "home_qb_adjustment",
    "away_qb_adjustment",
]


def _team_composite(master: pd.DataFrame, weights=None) -> pd.DataFrame:
    """One row per team: [team, composite] - the weighted blend of
    `weights` (defaults to config.NFL_GAME_PICK_COMPOSITE_WEIGHTS). All
    inputs are already z-normalized to mean 1.0
    (config.NFL_NORMALIZATION_Z_SCALE), so a straight weighted sum needs
    no further rescaling - direct mirror of game_picks._team_composite.
    `weights` is an explicit override parameter (same
    "config default, backtest-sweepable override" pattern as
    decision_score.py's own compute_zone_ops/compute_decision_advice) so
    scripts/backtest_nfl_season_carryover.py can sweep candidate weight
    sets (config.NFL_GAME_PICK_COMPOSITE_WEIGHTS_CORE_ONLY/_CORE_HEAVY)
    without monkeypatching the config module."""
    weights = config.NFL_GAME_PICK_COMPOSITE_WEIGHTS if weights is None else weights
    composite = sum(master[col] * weight for col, weight in weights)
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


def _add_qb_adjustments(games: pd.DataFrame, qb_continuity: pd.DataFrame, weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Adds `home_qb_adjustment`/`away_qb_adjustment` columns to `games` in
    place (returns it too, for chaining) - factored out of
    `build_game_features` (real follow-up, 2026-09-04 - "we don't capture
    real blowout confidence") so `build_game_features_disaggregated` can
    reuse the exact same real QB-continuity math without duplicating it.
    See module docstring for the full reasoning."""
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
    return games


def build_game_features(
    master: pd.DataFrame,
    qb_continuity: pd.DataFrame,
    weekly_df: pd.DataFrame,
    schedule_games_df: pd.DataFrame,
    composite_weights=None,
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
    games to feature). `composite_weights` is passed straight through to
    `_team_composite` (see that function's own docstring for the
    override-parameter reasoning)."""
    composite = _team_composite(master, composite_weights)

    games = schedule_games_df.merge(
        composite.rename(columns={"team": "home_team", "composite": "home_composite"}), on="home_team", how="left"
    ).merge(
        composite.rename(columns={"team": "away_team", "composite": "away_composite"}), on="away_team", how="left"
    )
    games = _add_qb_adjustments(games, qb_continuity, weekly_df)

    return games[["game_id", "season", "week", "home_team", "away_team"] + GAME_PICK_FEATURE_COLUMNS]


# Real follow-up (2026-09-04 - "we don't capture real blowout confidence"):
# a real, confirmed structural ceiling - home_win_probability's own ratio
# formula can never exceed ~59% for ANY real matchup, since both ratings
# are z-normalized composites clustered near 1.0 with a real std of only
# ~0.075 (see config.py's own NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH
# comments for the full diagnosis). DISAGGREGATED_SIGNAL_COLUMNS exposes
# each individual real signal `nfl_team_strength.assemble_team_metrics`
# already computes SEPARATELY for home/away, rather than only ever letting
# a caller see them pre-blended through config.NFL_GAME_PICK_COMPOSITE_WEIGHTS'
# hand-picked equal weights - a real, richer candidate feature set for
# scripts/train_nfl_game_pick_model.py's own real logistic/gradient-
# boosting fit to weigh (and scale) itself, swept honestly against the
# minimal GAME_PICK_FEATURE_COLUMNS candidate, not asserted better.
DISAGGREGATED_SIGNAL_COLUMNS = [
    "pyth_Strength", "pyth_Confidence", "offensive_edge", "defensive_edge", "turnover_margin", "points_per_drive",
]
DISAGGREGATED_FEATURE_COLUMNS = (
    [f"home_{c}" for c in DISAGGREGATED_SIGNAL_COLUMNS]
    + [f"away_{c}" for c in DISAGGREGATED_SIGNAL_COLUMNS]
    + ["home_qb_adjustment", "away_qb_adjustment"]
)


def build_game_features_disaggregated(
    master: pd.DataFrame, qb_continuity: pd.DataFrame, weekly_df: pd.DataFrame, schedule_games_df: pd.DataFrame,
) -> pd.DataFrame:
    """Returns [game_id, season, week, home_team, away_team] + DISAGGREGATED_FEATURE_COLUMNS
    - same real inputs/shape as `build_game_features`, but exposing each
    individual real signal for home/away separately instead of collapsing
    them into one hand-weighted composite first (see
    DISAGGREGATED_SIGNAL_COLUMNS' own comment for the full reasoning).
    Reuses `_add_qb_adjustments` unchanged - the QB-continuity math is
    identical either way, only the team-strength side differs."""
    home_master = master[["team"] + DISAGGREGATED_SIGNAL_COLUMNS].rename(
        columns={"team": "home_team", **{c: f"home_{c}" for c in DISAGGREGATED_SIGNAL_COLUMNS}}
    )
    away_master = master[["team"] + DISAGGREGATED_SIGNAL_COLUMNS].rename(
        columns={"team": "away_team", **{c: f"away_{c}" for c in DISAGGREGATED_SIGNAL_COLUMNS}}
    )
    games = schedule_games_df.merge(home_master, on="home_team", how="left").merge(
        away_master, on="away_team", how="left"
    )
    games = _add_qb_adjustments(games, qb_continuity, weekly_df)

    return games[["game_id", "season", "week", "home_team", "away_team"] + DISAGGREGATED_FEATURE_COLUMNS]


def game_feature_matrix(features_df: pd.DataFrame, feature_columns=None) -> pd.DataFrame:
    """Numeric X matrix, NaN-filled to 0 - mirrors game_picks.game_feature_matrix.
    `feature_columns` defaults to GAME_PICK_FEATURE_COLUMNS; pass
    DISAGGREGATED_FEATURE_COLUMNS for the richer candidate."""
    feature_columns = GAME_PICK_FEATURE_COLUMNS if feature_columns is None else feature_columns
    return features_df.reindex(columns=feature_columns).copy().fillna(0)


def compute_game_win_probabilities(
    master: pd.DataFrame,
    qb_continuity: pd.DataFrame,
    weekly_df: pd.DataFrame,
    schedule_games_df: pd.DataFrame,
    composite_weights=None,
    home_field_weight: float = None,
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
    dividing, purely as a degenerate-input guard.

    `composite_weights`/`home_field_weight` are explicit override
    parameters (default None -> config.NFL_GAME_PICK_COMPOSITE_WEIGHTS/
    config.NFL_HOME_FIELD_ADVANTAGE_WEIGHT), same "config default,
    backtest-sweepable override" pattern as decision_score.py's own
    compute_decision_advice - lets
    scripts/backtest_nfl_season_carryover.py sweep candidates without
    monkeypatching the config module."""
    home_field_weight = config.NFL_HOME_FIELD_ADVANTAGE_WEIGHT if home_field_weight is None else home_field_weight
    games = build_game_features(master, qb_continuity, weekly_df, schedule_games_df, composite_weights)

    # Real follow-up (2026-09-04 - "a little push or pull from home/
    # away"): a single, real, GLOBAL additive home-field term (see
    # config.NFL_HOME_FIELD_ADVANTAGE_WEIGHT's own docstring for why this
    # is a league-wide constant, not a per-team fit) - added to the HOME
    # side only, same z-normalized rating units as the composite.
    home_rating = (
        games["home_composite"] + config.NFL_QB_CONTINUITY_WEIGHT * games["home_qb_adjustment"]
        + home_field_weight
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


def apply_ml_model(
    win_probabilities: pd.DataFrame,
    master: pd.DataFrame,
    qb_continuity: pd.DataFrame,
    weekly_df: pd.DataFrame,
    schedule_games_df: pd.DataFrame,
) -> pd.DataFrame:
    """Overwrites `home_win_probability` with a real, walk-forward-validated
    ML model's own prediction (scripts/train_nfl_game_pick_model.py) when
    one exists at config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH - direct
    structural mirror of `apply_calibration`'s own graceful-degradation
    contract (a real no-op, unchanged heuristic, when no artifact exists),
    except this REPLACES the whole ratio-based probability rather than
    rescaling it - the real fix for the structural "can't exceed ~59%"
    ceiling `compute_game_win_probabilities`'s own ratio formula has (see
    config.py's own NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH comments for
    the full diagnosis: both ratings are z-normalized composites clustered
    within a real std of ~0.075 of 1.0, so the ratio has no free scale
    parameter and can never express a real blowout's true confidence, no
    matter how much real historical data exists).

    The saved artifact is a real `{"model", "feature_columns"}` dict (not
    a bare estimator) so this function knows whether to rebuild features
    via `build_game_features` or `build_game_features_disaggregated` -
    whichever candidate scripts/train_nfl_game_pick_model.py's own real
    backtest validated - without hardcoding one choice here. A game with
    a genuinely incomplete feature row (e.g. a team missing from `master`
    entirely - not expected in practice, but a real degenerate-input
    guard) keeps its ORIGINAL heuristic probability rather than a
    fabricated ML prediction from filled-in zeros - a per-row fallback,
    not an all-or-nothing one."""
    artifact = ml_models.load_model(config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH)
    if artifact is None:
        return win_probabilities

    model, feature_columns = artifact["model"], artifact["feature_columns"]
    builder = build_game_features_disaggregated if feature_columns == DISAGGREGATED_FEATURE_COLUMNS else build_game_features
    features = builder(master, qb_continuity, weekly_df, schedule_games_df).set_index("game_id")

    complete = features[feature_columns].notna().all(axis=1)
    X = game_feature_matrix(features, feature_columns)
    proba = pd.Series(model.predict_proba(X)[:, 1], index=X.index)

    result = win_probabilities.copy()
    real_rows = result["game_id"].map(complete).fillna(False)
    result.loc[real_rows, "home_win_probability"] = result.loc[real_rows, "game_id"].map(proba)
    return result


def apply_market_tiebreak(
    win_probabilities: pd.DataFrame,
    market_probabilities: pd.DataFrame,
    disagreement_threshold: float = None,
) -> pd.DataFrame:
    """Defers to the real market's own (devigged) probability when our
    own prediction disagrees with it by a lot - see config.py's
    NFL_GAME_PICK_MARKET_DISAGREEMENT_THRESHOLD comments for the full
    real backtest (1,482 real games, all of 2016-2025, proper walk-
    forward, no lookahead): at every disagreement magnitude tested, the
    market was more accurate than our own model in that specific zone,
    and deferring ONLY there (not overall) raised real overall accuracy
    with zero risk to the rest of the season.

    `market_probabilities` needs real [home_team, away_team,
    market_home_win_probability] - the same shape
    nfl_pipeline._build_market_probabilities already produces (matches
    `nfl_game_predictions.select_game_picks`'s own `market_probabilities`
    contract). Matched on (home_team, away_team) - safe for a single
    real week's slate (no team plays itself twice in one week), the same
    scope every caller already uses this at.

    A real, useful side effect, not something requiring special-casing
    elsewhere: `market_home_win_probability` is DEVIGGED, always <= the
    RAW vigged implied probability `nfl_game_predictions.advise_bets`
    compares against for the same side (market_odds.devig's own
    docstring) - so a deferred game's real edge is <= 0 by construction,
    naturally suppressing bet advice on exactly the bucket this backtest
    proved unreliable.

    Graceful degradation: a game with no real market probability
    (missing moneylines, or absent from `market_probabilities` entirely)
    keeps its original probability unchanged - disagreement can't be
    measured, so no defer decision is made."""
    threshold = (
        config.NFL_GAME_PICK_MARKET_DISAGREEMENT_THRESHOLD if disagreement_threshold is None else disagreement_threshold
    )
    merged = win_probabilities.merge(
        market_probabilities[["home_team", "away_team", "market_home_win_probability"]],
        on=["home_team", "away_team"], how="left",
    )
    disagreement = (merged["home_win_probability"] - merged["market_home_win_probability"]).abs()
    should_defer = (disagreement >= threshold).fillna(False).to_numpy()

    result = win_probabilities.copy()
    result.loc[should_defer, "home_win_probability"] = merged.loc[should_defer, "market_home_win_probability"].to_numpy()
    return result

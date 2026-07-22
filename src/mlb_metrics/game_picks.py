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

from mlb_metrics import config, matchup


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

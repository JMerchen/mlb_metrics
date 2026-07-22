"""Automated Game Picks: predicts a winner for each of today's games from
team-level metrics - not hitter picks (see predictions.py/matchup.py for
those). Uses exactly the signal set requested: each team's Pythagorean
strength, Pythagorean confidence, suppression resistance, and true power
(config.GAME_PICK_COMPOSITE_WEIGHTS), adjusted by the specific pitching
(probable starter + bullpen PAVE) they're projected to face today.

suppression_resistance is deliberately weighted both directly and inside
true_power (which already averages it with offensive_edge) - both signals
were named explicitly, so the overlap is intentional, not a bug.

This is a first-pass, unvalidated blend, same spirit as matchup.py: meant
to be logged and tracked (see game_predictions.py/game_evaluation.py) and
compared against reality before ever being trusted. There is no way to
backtest this against past dates - schedule/game data has never been
persisted to git history the way wave.csv is (see git_backtest.py) - so it
can only accumulate a resolved dataset forward from the day it ships.
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
    bullpen = confidence[["team", "Bullpen_PAVE_PLUS"]]
    starter_pave = pave[["key_mlbam", "PAVE_PLUS"]].rename(columns={"PAVE_PLUS": "starter_pave_plus"})

    games = schedule_games_df.merge(
        composite.rename(columns={"team": "home_team", "composite": "home_composite"}), on="home_team", how="left"
    ).merge(
        composite.rename(columns={"team": "away_team", "composite": "away_composite"}), on="away_team", how="left"
    )

    games = games.merge(
        bullpen.rename(columns={"team": "home_team", "Bullpen_PAVE_PLUS": "home_bullpen_pave_plus"}),
        on="home_team", how="left",
    ).merge(
        bullpen.rename(columns={"team": "away_team", "Bullpen_PAVE_PLUS": "away_bullpen_pave_plus"}),
        on="away_team", how="left",
    )

    games = games.merge(
        starter_pave.rename(
            columns={"key_mlbam": "home_probable_pitcher_key_mlbam", "starter_pave_plus": "home_starter_pave_plus"}
        ),
        on="home_probable_pitcher_key_mlbam", how="left",
    ).merge(
        starter_pave.rename(
            columns={"key_mlbam": "away_probable_pitcher_key_mlbam", "starter_pave_plus": "away_starter_pave_plus"}
        ),
        on="away_probable_pitcher_key_mlbam", how="left",
    )

    # Home team faces the AWAY team's pitching, and vice versa.
    home_pitching_quality_faced = matchup.clip_and_blend_pitching_quality(
        games["away_starter_pave_plus"], games["away_bullpen_pave_plus"]
    )
    away_pitching_quality_faced = matchup.clip_and_blend_pitching_quality(
        games["home_starter_pave_plus"], games["home_bullpen_pave_plus"]
    )

    home_rating = (games["home_composite"] * home_pitching_quality_faced).clip(lower=config.GAME_PICK_RATING_FLOOR)
    away_rating = (games["away_composite"] * away_pitching_quality_faced).clip(lower=config.GAME_PICK_RATING_FLOOR)

    games["home_win_probability"] = home_rating / (home_rating + away_rating)

    return games[["game_pk", "date", "home_team", "away_team", "home_win_probability"]]

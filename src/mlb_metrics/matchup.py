"""Matchup-adjusted hit probability: blends a batter's own Game_Hit_Probability
with the specific pitching they're projected to face today - their team's
opponent's probable starter (most of their at-bats) and that opponent's
bullpen (the rest, via pitchers.compute_bullpen_pave, already built).

This is a first-pass, unvalidated blend. Per the project's own finding that
Game_Hit_Probability is already overconfident, Matchup_Hit_Probability is
meant to be logged alongside it (both as separate `metric` values in
predictions.csv - see pipeline.py) and backtested before ever being trusted
as the "recommended" metric, not switched to blindly.

Known first-pass simplifications, not fixed here: modern bullpen-game usage
(an "opener" starting 1-2 innings before the real workhorse) breaks the
assumed starter/bullpen at-bat-share split; schedule.py's v1 only carries a
team's first game of a doubleheader.
"""

import pandas as pd

from mlb_metrics import config


def clip_and_blend_pitching_quality(starter_pave_plus: pd.Series, bullpen_pave_plus: pd.Series) -> pd.Series:
    """A single opposing-pitching-quality multiplier from a probable
    starter's PAVE_PLUS and their team's Bullpen_PAVE_PLUS, weighted by
    assumed at-bat share (config.MATCHUP_STARTER_AB_SHARE/BULLPEN_AB_SHARE).
    Missing values (unannounced starter, not found in pave.csv) default to a
    neutral 1.0. Both inputs are clipped to config.MATCHUP_PAVE_PLUS_CLIP
    BEFORE blending, not just clipping the final result - a small-sample
    outlier PAVE_PLUS would otherwise corrupt the blend (see config.py).
    Shared by compute_matchup_hit_probability (batter-level) and
    game_picks.compute_game_win_probabilities (team-level) - same physical
    reasoning applies at both levels."""
    lo, hi = config.MATCHUP_PAVE_PLUS_CLIP
    starter = starter_pave_plus.fillna(1.0).clip(lo, hi)
    bullpen = bullpen_pave_plus.fillna(1.0).clip(lo, hi)
    return config.MATCHUP_STARTER_AB_SHARE * starter + config.MATCHUP_BULLPEN_AB_SHARE * bullpen


def compute_matchup_hit_probability(
    wave: pd.DataFrame,
    pave: pd.DataFrame,
    confidence: pd.DataFrame,
    schedule_df: pd.DataFrame,
) -> pd.DataFrame:
    """Returns [key_mlbam, Matchup_Hit_Probability] - one row per batter
    whose team has a game in `schedule_df` today (a batter with no game
    today has no matchup and is left out entirely, not given a neutral
    value). A probable starter not yet announced, or not found in `pave`,
    contributes a neutral 1.0 rather than dropping the batter."""
    matchup = wave[["key_mlbam", "team", "Game_Hit_Probability"]].merge(
        schedule_df[["team", "opponent", "probable_pitcher_key_mlbam"]], on="team", how="inner"
    )

    starter_pave = pave[["key_mlbam", "PAVE_PLUS"]].rename(
        columns={"key_mlbam": "probable_pitcher_key_mlbam", "PAVE_PLUS": "starter_pave_plus"}
    )
    matchup = matchup.merge(starter_pave, on="probable_pitcher_key_mlbam", how="left")

    bullpen_pave = confidence[["team", "Bullpen_PAVE_PLUS"]].rename(columns={"team": "opponent"})
    matchup = matchup.merge(bullpen_pave, on="opponent", how="left")

    opponent_quality = clip_and_blend_pitching_quality(matchup["starter_pave_plus"], matchup["Bullpen_PAVE_PLUS"])
    matchup["Matchup_Hit_Probability"] = (matchup["Game_Hit_Probability"] * opponent_quality).clip(0, 1)

    return matchup[["key_mlbam", "Matchup_Hit_Probability"]]

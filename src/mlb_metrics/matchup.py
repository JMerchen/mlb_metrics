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
    lo, hi = config.MATCHUP_PAVE_PLUS_CLIP

    matchup = wave[["key_mlbam", "team", "Game_Hit_Probability"]].merge(
        schedule_df[["team", "opponent", "probable_pitcher_key_mlbam"]], on="team", how="inner"
    )

    starter_pave = pave[["key_mlbam", "PAVE_PLUS"]].rename(
        columns={"key_mlbam": "probable_pitcher_key_mlbam", "PAVE_PLUS": "starter_pave_plus"}
    )
    matchup = matchup.merge(starter_pave, on="probable_pitcher_key_mlbam", how="left")

    bullpen_pave = confidence[["team", "Bullpen_PAVE_PLUS"]].rename(columns={"team": "opponent"})
    matchup = matchup.merge(bullpen_pave, on="opponent", how="left")

    # Clip BEFORE blending, not just the final probability - a small-sample
    # outlier PAVE_PLUS would otherwise corrupt the blend (see config.py).
    matchup["starter_pave_plus"] = matchup["starter_pave_plus"].fillna(1.0).clip(lo, hi)
    matchup["Bullpen_PAVE_PLUS"] = matchup["Bullpen_PAVE_PLUS"].fillna(1.0).clip(lo, hi)

    opponent_quality = (
        config.MATCHUP_STARTER_AB_SHARE * matchup["starter_pave_plus"]
        + config.MATCHUP_BULLPEN_AB_SHARE * matchup["Bullpen_PAVE_PLUS"]
    )
    matchup["Matchup_Hit_Probability"] = (matchup["Game_Hit_Probability"] * opponent_quality).clip(0, 1)

    return matchup[["key_mlbam", "Matchup_Hit_Probability"]]

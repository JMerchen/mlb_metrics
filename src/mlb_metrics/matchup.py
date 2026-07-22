"""Matchup-adjusted hit probability: combines a batter's own AB-level hit
rate (WAVE) with the specific pitching they're projected to face today -
their team's opponent's probable starter (most of their at-bats) and that
opponent's bullpen (the rest, via pitchers.compute_bullpen_pave, already
built) - using each side's raw PAVE (a real batting-average-against-scale
rate), not PAVE_PLUS (a league-normalized ratio centered on 1.0).

The combination is log5 (the odds-ratio method, a generalization of Bill
James' log5 winning-percentage formula): given a batter's rate, an
opposing-pitching rate, and the league-average rate both are measured
against, log5 returns the rate expected specifically in that matchup. This
needs real rates on a shared 0-1 scale (WAVE and PAVE both are), which is
why this module uses raw PAVE instead of the PAVE_PLUS ratio
game_picks.py's team-level model uses (see clip_and_blend_pitching_quality
there) - multiplying a probability by a ~1.0-centered ratio, the old
approach here, isn't the same thing as a real matchup-specific rate.

The resulting matchup AB-level rate is converted to a per-game probability
via the same binomial-trials formula hitters.compute_wave already uses
(config.WAVE_TRIALS_PER_GAME), so Matchup_Hit_Probability lands on the same
0-1 per-game-probability scale as probability/Game_Hit_Probability and can
be thresholded/ranked alongside them directly (see predictions.select_picks).

Known first-pass simplifications, not fixed here: modern bullpen-game usage
(an "opener" starting 1-2 innings before the real workhorse) breaks the
assumed starter/bullpen at-bat-share split; schedule.py's v1 only carries a
team's first game of a doubleheader.

Two further adjustments beyond the base batter-vs-starter blend: platoon
(uses the batter's WAVE_L/WAVE_R against the probable starter's actual
throwing hand instead of a hand-blended overall WAVE - see _platoon_wave)
and park (scales the AB rate by the game's actual venue's Park_Factor - see
_park_factor_multiplier and teams.compute_park_factors). Both were
empirically validated together via backtest before shipping on by default -
see config.MATCHUP_PARK_FACTOR_WEIGHT's docstring for the numbers.

Validation status: backtested on 30 days of real history reconstructed
directly from persisted Statcast (git-history CSV replay wasn't available -
raw PAVE/WAVE were never persisted before this). Replaying the actual
predictions.select_picks logic (PA + probability/Game_Hit_Probability/
Matchup_Hit_Probability all >=0.7, ranked by Matchup_Approach) against the
"recommended" subset that actually counts toward Beat the Streak
(rank<=2, Game_Hit_Probability>=0.80) raised the hit rate from 65.8%
(n=38, no matchup) to 75.0% (n=32, with matchup) - a real, meaningful
improvement, not just a directional one. This was only possible after
fixing a pre-existing bug in data.assign_game_ids (see below) that was
silently corrupting any from-scratch Game_Hit_Probability reconstruction;
an earlier pass at this same backtest, before the fix, could only validate
a weaker signal (a probability-only-qualified pool, no Game_Hit_Probability
gate) and found a smaller, less certain edge.

That same investigation surfaced the assign_game_ids bug and it's now
fixed (see data.assign_game_ids' docstring): it used to reconstruct game
boundaries from scratch via an at_bat_number-reset counter that silently
assumed one real game's rows were contiguous in the table, which nothing
guaranteed - it now groups directly by Statcast's own game_pk instead,
which needs no such reconstruction. Confirmed against a real historical
wave.csv commit: reconstructed Game_Hit_Probability now matches the
committed values exactly (previously it was deflated by roughly 40%).
"""

import numpy as np
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
    Used by game_picks.compute_game_win_probabilities (team-level) - its
    composites are already league-normalized ~1.0 ratios, so a ~1.0-centered
    multiplier is the right fit there. compute_matchup_hit_probability below
    uses clip_and_blend_pitching_pave instead (raw PAVE, not this)."""
    lo, hi = config.MATCHUP_PAVE_PLUS_CLIP
    starter = starter_pave_plus.fillna(1.0).clip(lo, hi)
    bullpen = bullpen_pave_plus.fillna(1.0).clip(lo, hi)
    return config.MATCHUP_STARTER_AB_SHARE * starter + config.MATCHUP_BULLPEN_AB_SHARE * bullpen


def clip_and_blend_pitching_pave(starter_pave: pd.Series, bullpen_pave: pd.Series, league_pave: float) -> pd.Series:
    """A single opposing-pitching AB-level hit rate from a probable
    starter's raw PAVE and their team's raw Bullpen_PAVE, weighted by
    assumed at-bat share. Missing values (unannounced starter, not found in
    pave.csv) default to `league_pave` - a neutral matchup, the same role
    PAVE_PLUS's neutral 1.0 plays for clip_and_blend_pitching_quality above.
    Both inputs are clipped to [lo, hi] * league_pave (config.
    MATCHUP_PAVE_CLIP_MULTIPLIER) BEFORE blending, not just clipping the
    final result - a small-sample outlier PAVE would otherwise corrupt the
    blend, same reasoning as the PAVE_PLUS version."""
    lo_mult, hi_mult = config.MATCHUP_PAVE_CLIP_MULTIPLIER
    lo, hi = lo_mult * league_pave, hi_mult * league_pave
    starter = starter_pave.fillna(league_pave).clip(lo, hi)
    bullpen = bullpen_pave.fillna(league_pave).clip(lo, hi)
    return config.MATCHUP_STARTER_AB_SHARE * starter + config.MATCHUP_BULLPEN_AB_SHARE * bullpen


def _log5(rate_a: pd.Series, rate_b: pd.Series, league_rate: float) -> pd.Series:
    """Odds-ratio (log5) combination of two rates measured against a shared
    league baseline into the rate expected specifically between them - the
    standard sabermetric technique for e.g. a hitter's rate vs. a pitcher's
    allowed rate (Bill James' log5 formula, generalized from win percentage
    to any rate stat)."""
    numerator = rate_a * rate_b / league_rate
    denominator = numerator + (1 - rate_a) * (1 - rate_b) / (1 - league_rate)
    return numerator / denominator


def _league_pave(pave: pd.DataFrame) -> float:
    """The league-average AB-level hit rate implied by `pave` - PAVE_PLUS is
    PAVE divided by exactly this value for every row (see
    pitchers.assemble_pitchers), so it can be recovered directly rather than
    needing its own persisted column. Falls back to
    config.MATCHUP_LEAGUE_PAVE_FALLBACK when `pave` can't yield a real value
    (empty, or missing the raw PAVE column - e.g. a test fixture)."""
    if pave.empty or "PAVE" not in pave.columns or "PAVE_PLUS" not in pave.columns:
        return config.MATCHUP_LEAGUE_PAVE_FALLBACK
    league_pave = (pave["PAVE"] / pave["PAVE_PLUS"]).mean()
    return league_pave if pd.notna(league_pave) else config.MATCHUP_LEAGUE_PAVE_FALLBACK


def _platoon_wave(matchup: pd.DataFrame) -> pd.Series:
    """Picks WAVE_L/WAVE_R based on the probable starter's own Throws
    (pitchers.compute_pitcher_throws) instead of the batter's handedness-
    blended overall WAVE - facing a lefty and a righty aren't the same
    matchup for a batter with any real platoon split. Falls back to WAVE
    when Throws is unknown (unannounced starter, not found in `pave`) or
    when WAVE_L/WAVE_R/Throws aren't present at all (old wave.csv/pave.csv
    snapshots predating this - see git_backtest.py, which replays those
    directly)."""
    if not {"WAVE_L", "WAVE_R", "starter_throws"}.issubset(matchup.columns):
        return matchup["WAVE"]
    platoon = np.where(
        matchup["starter_throws"] == "L", matchup["WAVE_L"],
        np.where(matchup["starter_throws"] == "R", matchup["WAVE_R"], matchup["WAVE"]),
    )
    return pd.Series(platoon, index=matchup.index)


def _park_factor_multiplier(park_factor: pd.Series) -> pd.Series:
    """1.0 (no-op) when Park_Factor is missing (old confidence.csv
    snapshots, or a test fixture - see teams.compute_park_factors);
    otherwise the clipped park effect scaled by
    config.MATCHUP_PARK_FACTOR_WEIGHT (0 = fully off, 1 = full effect)."""
    lo, hi = config.MATCHUP_PARK_FACTOR_CLIP
    clipped = park_factor.fillna(1.0).clip(lo, hi)
    return 1 + config.MATCHUP_PARK_FACTOR_WEIGHT * (clipped - 1)


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
    contributes a neutral (league-average) matchup rather than dropping the
    batter.

    Two adjustments beyond the base batter-vs-pitcher log5 blend:
    - Platoon: uses the batter's WAVE_L/WAVE_R against the probable
      starter's actual throwing hand (see _platoon_wave) instead of their
      hand-blended overall WAVE.
    - Park: scales the matchup AB rate by the game's actual venue's
      Park_Factor (see teams.compute_park_factors) - the home team's park,
      looked up via `schedule_df`'s `is_home`, since neither side's own
      Park_Factor is relevant when they're the road team today.
    """
    wave_columns = [c for c in ("key_mlbam", "team", "WAVE", "WAVE_L", "WAVE_R") if c in wave.columns]
    schedule_columns = [
        c for c in ("team", "opponent", "probable_pitcher_key_mlbam", "is_home") if c in schedule_df.columns
    ]
    matchup = wave[wave_columns].merge(schedule_df[schedule_columns], on="team", how="inner")

    league_pave = _league_pave(pave)

    pave_columns = [c for c in ("key_mlbam", "PAVE", "Throws") if c in pave.columns]
    starter_pave = pave[pave_columns].rename(
        columns={"key_mlbam": "probable_pitcher_key_mlbam", "PAVE": "starter_pave", "Throws": "starter_throws"}
    )
    matchup = matchup.merge(starter_pave, on="probable_pitcher_key_mlbam", how="left")

    bullpen_pave = confidence[["team", "Bullpen_PAVE"]].rename(columns={"team": "opponent"})
    matchup = matchup.merge(bullpen_pave, on="opponent", how="left")

    if "Park_Factor" in confidence.columns and "is_home" in matchup.columns:
        matchup["venue_team"] = matchup["team"].where(matchup["is_home"], matchup["opponent"])
        park_factor = confidence[["team", "Park_Factor"]].rename(columns={"team": "venue_team"})
        matchup = matchup.merge(park_factor, on="venue_team", how="left")
    else:
        matchup["Park_Factor"] = pd.NA

    opponent_rate = clip_and_blend_pitching_pave(matchup["starter_pave"], matchup["Bullpen_PAVE"], league_pave)
    batter_rate = _platoon_wave(matchup)
    matchup_ab_rate = _log5(batter_rate, opponent_rate, league_pave) * _park_factor_multiplier(matchup["Park_Factor"])
    matchup["Matchup_Hit_Probability"] = (
        1 - (1 - matchup_ab_rate) ** config.WAVE_TRIALS_PER_GAME
    ).clip(0, 1)

    return matchup[["key_mlbam", "Matchup_Hit_Probability"]]

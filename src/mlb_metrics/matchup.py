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

Validation status: WAVE/PAVE/Bullpen_PAVE are all AB-level rates, computed
independently of data.assign_game_ids - unlike Game_Hit_Probability, they're
reliably reproducible from persisted Statcast at any past as-of-date, which
made this formula backtestable on 30 days of real history even though a
git-history-CSV replay wasn't available (raw PAVE/WAVE were never persisted
before this). Ranking a probability>=0.7-qualified pool by
Matchup_Hit_Probability scored modestly better than ranking by probability
alone (Brier 0.270 vs 0.277, hit rate 62.5% vs 61.5%, n=104 resolved) - a
real but not statistically decisive edge at this sample size, unlike the
probability/Game_Hit_Probability joint-threshold work (see
config.HITTER_MIN_PROBABILITY), which showed a much larger, clearer effect.
Treat this as directionally validated, not proven - the same "watch it
accumulate real results" posture the project has always applied to a new
signal (Automated Game Picks, this metric's own original first-pass launch).

That same investigation also surfaced a real, pre-existing bug worth flagging
separately: data.assign_game_ids can badly fragment real games (one calendar
date's game split into 2-3 different game_id values) when replayed against a
large multi-month reconstructed dataset, which is why this backtest avoided
Game_Hit_Probability-based qualifiers entirely rather than risk validating
against corrupted numbers. Not fixed here - out of scope for this change,
and touching the game-id algorithm needs its own careful validation given
how much (including the live daily Game_Hit_Probability) depends on it.
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
    batter."""
    matchup = wave[["key_mlbam", "team", "WAVE"]].merge(
        schedule_df[["team", "opponent", "probable_pitcher_key_mlbam"]], on="team", how="inner"
    )

    league_pave = _league_pave(pave)

    starter_pave = pave[["key_mlbam", "PAVE"]].rename(
        columns={"key_mlbam": "probable_pitcher_key_mlbam", "PAVE": "starter_pave"}
    )
    matchup = matchup.merge(starter_pave, on="probable_pitcher_key_mlbam", how="left")

    bullpen_pave = confidence[["team", "Bullpen_PAVE"]].rename(columns={"team": "opponent"})
    matchup = matchup.merge(bullpen_pave, on="opponent", how="left")

    opponent_rate = clip_and_blend_pitching_pave(matchup["starter_pave"], matchup["Bullpen_PAVE"], league_pave)
    matchup_ab_rate = _log5(matchup["WAVE"], opponent_rate, league_pave)
    matchup["Matchup_Hit_Probability"] = (
        1 - (1 - matchup_ab_rate) ** config.WAVE_TRIALS_PER_GAME
    ).clip(0, 1)

    return matchup[["key_mlbam", "Matchup_Hit_Probability"]]

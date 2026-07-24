"""Estimated DraftKings Classic MLB fantasy points for today's hitters and
probable starting pitchers - a ranked list of good plays for
docs/dfs.html, NOT a salary-cap lineup optimizer (no salary data is
ingested anywhere in this project - see config.py's DFS section for why).

DraftKings' scoring rules (config.DFS_DK_*) were confirmed live via web
search (not from memory) - see config.py's DFS section for sources.

## Hitters

DK pays non-linearly for hit type (a double isn't 2x a single's value:
5 != 2*3), but this project only computes a linear Expected_Bases signal
(hitters.compute_wtb) - no per-player 1B/2B/3B/HR rate breakdown exists.
DK_Points_Hitter approximates the non-linear scoring with a single
calibrated "DK points per expected total base" constant
(config.DFS_DK_POINTS_PER_TOTAL_BASE, computed from real Lahman 2015-2025
hit-type shares - see its docstring), applied to Expected_Bases after
scaling it for TODAY's specific matchup via compute_matchup_adjustment - a
ratio of Matchup_Hit_Probability (today's actual opposing pitcher,
platoon+park adjusted) to the batter's own blended Game_Hit_Probability.
That ratio is a real heuristic, not a rigorous transform: it's derived
from hit-PROBABILITY signals but applied multiplicatively to a
TOTAL-BASES signal - the two are correlated but not the same thing (a
good matchup for making contact isn't necessarily proportionally as good
for extra-base power). This is the single highest-risk modeling choice
here - see scripts/backtest_dfs_rankings.py for how well it actually
holds up.

**Backtest result (config.py's DFS section has the full numbers): this
heuristic does NOT hold up.** DK_Points_Hitter's real backtest correlation
against actual next-game hit-type points is -0.004 (essentially zero) -
reported honestly, not hidden. Treat DK_Points_Hitter as unvalidated for
now; dropping compute_matchup_adjustment entirely (using raw
Expected_Bases) or a different adjustment approach is a needed follow-up,
not a nice-to-have.

**Explicitly excluded from v1** (documented, not silently rounded away):
BB, HBP, runs scored, RBI, and stolen bases. This project has no
walk-rate, lineup-run-context (runners on base when a player bats depends
on teammates' OBP, not just this player), or stolen-base signal at all
today - faking precision on those would be worse than leaving them out.
DK_Points_Hitter is a hit-scoring-only estimate, not the full DK score.

## Pitchers

Needs one new signal this project didn't have: pitcher_form.py's
recency-windowed K9/BB9/HR9/IP-per-start (mirroring pitchers.compute_pave's
windowing pattern applied to different underlying stats). DK_Points_Pitcher
combines:
- Expected_IP (= pitcher_form's blended IP_per_start) at
  config.DFS_DK_PITCHER_IP_POINTS/inning.
- Expected_K = K9 * Expected_IP / 9, at config.DFS_DK_PITCHER_K_POINTS/K.
- Expected_BB = BB9 * Expected_IP / 9, at config.DFS_DK_PITCHER_BB_POINTS/BB.
- Expected_H_Allowed = PAVE (pitchers.compute_pave's AB-level hit rate,
  already computed for the live pipeline) * an estimated batters-faced
  count (Expected_IP * config.DFS_BATTERS_FACED_PER_INNING, computed from
  this project's own persisted Statcast - see that constant's docstring),
  at config.DFS_DK_PITCHER_H_POINTS/hit.
- Expected_ER, estimated via FIP rather than a real earned-run signal -
  this project has a consistent, repeated house principle against
  modeling ERA/earned runs directly (defense/sequencing-dependent, not
  just the pitcher's own stuff - the same reasoning pitchers.py's
  Power_A_PLUS and age_curve.py's pitcher metrics both already use FIP
  over ERA for). FIP_Windowed is computed from K9/BB9/HR9 the same
  formula age_curve.py/traditional_stats.py use from raw counts (the
  count-based and rate-based forms are algebraically identical - IP
  cancels out), reusing config.FIP_CONSTANT. Expected_ER =
  FIP_Windowed * Expected_IP / 9, clipped at 0 (a small-sample-windowed
  FIP can go slightly negative before the additive constant fully
  corrects it). Deliberately excludes HBP allowed (pitcher_form.py
  doesn't track it - a documented simplification, consistent with
  age_curve.py's own pitcher FIP already being HBP-inclusive only where
  Lahman happens to carry it).

**Explicitly excluded from v1**: Win (needs a win-probability estimate -
too dependent on the pitcher's own team's offense that day to reasonably
approximate), and the rare discrete bonuses (complete game, complete game
shutout, no-hitter). Relief pitchers are entirely out of scope - IP per
appearance is only a meaningful "per start" number for a starter, not a
bullpen arm's variable-length outing.

## Qualifiers

Hitters need config.BACKTEST_MIN_PLATE_APPEARANCES (the same guard
predictions.select_picks uses) and a game today. Pitchers need
config.DFS_PITCHER_MIN_STARTS recorded starts (a 1-2 start sample makes
HR9 especially pure noise) and to be a team's announced probable starter
today - no neutral fallback for an unannounced pitcher, unlike
matchup.py's hitter-side blends, since there's no reasonable "neutral
pitcher" to rank.
"""

import pandas as pd

from mlb_metrics import config

HITTER_DFS_COLUMNS = [
    "key_mlbam", "name_first", "name_last", "team", "opponent", "is_home",
    "PA_L", "PA_R", "Expected_Bases", "Game_Hit_Probability",
    "Matchup_Hit_Probability", "Matchup_Ratio", "Adjusted_Expected_Bases",
    "DK_Points_Hitter",
]

PITCHER_DFS_COLUMNS = [
    "key_mlbam", "name_first", "name_last", "team", "opponent", "is_home",
    "starts", "K9", "BB9", "HR9", "IP_per_start", "Expected_IP",
    "Expected_K", "Expected_BB", "Expected_H_Allowed", "FIP_Windowed",
    "Expected_ER", "DK_Points_Pitcher",
]


def compute_matchup_adjustment(matchup_hit_probability: pd.Series, game_hit_probability: pd.Series) -> pd.Series:
    """Ratio of today's matchup-specific hit probability to the batter's
    own blended baseline - >1 means today's matchup is more favorable than
    their average, <1 less favorable. `game_hit_probability` is floored at
    config.DFS_MATCHUP_RATIO_MIN_DENOM before dividing (a batter with an
    almost-zero blended probability would otherwise blow the ratio up
    arbitrarily on noise), and the final ratio is clipped to
    config.DFS_MATCHUP_RATIO_CLIP - clipping the INPUT denominator and the
    OUTPUT ratio separately guards against two different failure modes,
    same clip-before-and-after-blending spirit as
    matchup.clip_and_blend_pitching_pave."""
    denom = game_hit_probability.clip(lower=config.DFS_MATCHUP_RATIO_MIN_DENOM)
    ratio = matchup_hit_probability / denom
    lo, hi = config.DFS_MATCHUP_RATIO_CLIP
    return ratio.clip(lo, hi)


def compute_hitter_dk_points(
    wave: pd.DataFrame,
    matchup_probability: pd.DataFrame,
    schedule_df: pd.DataFrame,
    min_plate_appearances: int = config.BACKTEST_MIN_PLATE_APPEARANCES,
) -> pd.DataFrame:
    """One row per hitter clearing `min_plate_appearances` (PA_L + PA_R,
    same guard predictions.select_picks uses) whose team has a game in
    `schedule_df` today AND who has a resolvable Matchup_Hit_Probability
    (matchup.compute_matchup_hit_probability's own output - already
    restricted to today's games) - a batter with no game today, or not
    found in `matchup_probability`, is excluded, not defaulted. Sorted by
    DK_Points_Hitter descending."""
    qualified = wave[(wave["PA_L"] + wave["PA_R"]) >= min_plate_appearances].copy()

    schedule_columns = [c for c in ("team", "opponent", "is_home") if c in schedule_df.columns]
    scheduled = qualified.merge(schedule_df[schedule_columns], on="team", how="inner")
    scheduled = scheduled.merge(
        matchup_probability[["key_mlbam", "Matchup_Hit_Probability"]], on="key_mlbam", how="inner"
    )

    scheduled["Matchup_Ratio"] = compute_matchup_adjustment(
        scheduled["Matchup_Hit_Probability"], scheduled["Game_Hit_Probability"]
    )
    scheduled["Adjusted_Expected_Bases"] = scheduled["Expected_Bases"] * scheduled["Matchup_Ratio"]
    scheduled["DK_Points_Hitter"] = scheduled["Adjusted_Expected_Bases"] * config.DFS_DK_POINTS_PER_TOTAL_BASE

    return scheduled[HITTER_DFS_COLUMNS].sort_values("DK_Points_Hitter", ascending=False)


def compute_pitcher_dk_points(
    pave: pd.DataFrame,
    pitcher_form: pd.DataFrame,
    schedule_df: pd.DataFrame,
    min_starts: int = config.DFS_PITCHER_MIN_STARTS,
) -> pd.DataFrame:
    """One row per team's announced probable starter today
    (schedule_df's probable_pitcher_key_mlbam) who also clears
    `min_starts` in `pitcher_form`. No neutral fallback for an
    unannounced/unmatched/under-min-starts pitcher - simply excluded, not
    given a league-average placeholder. Relief pitchers are out of scope
    entirely (not just filtered by min_starts) - `pitcher_form` only ever
    carries starts, and only a team's probable STARTER is looked up here.
    Sorted by DK_Points_Pitcher descending."""
    starters = schedule_df.dropna(subset=["probable_pitcher_key_mlbam"]).copy()
    starters["key_mlbam"] = starters["probable_pitcher_key_mlbam"].astype(int)

    merged = starters.merge(pitcher_form, on="key_mlbam", how="inner")
    merged = merged[merged["starts"] >= min_starts].copy()

    pave_columns = [c for c in ("key_mlbam", "name_first", "name_last", "PAVE") if c in pave.columns]
    merged = merged.merge(pave[pave_columns], on="key_mlbam", how="left")
    merged["PAVE"] = merged["PAVE"].fillna(config.MATCHUP_LEAGUE_PAVE_FALLBACK)

    merged["Expected_IP"] = merged["IP_per_start"]
    merged["Expected_K"] = merged["K9"] * merged["Expected_IP"] / 9
    merged["Expected_BB"] = merged["BB9"] * merged["Expected_IP"] / 9
    expected_batters_faced = merged["Expected_IP"] * config.DFS_BATTERS_FACED_PER_INNING
    merged["Expected_H_Allowed"] = merged["PAVE"] * expected_batters_faced

    merged["FIP_Windowed"] = (13 * merged["HR9"] + 3 * merged["BB9"] - 2 * merged["K9"]) / 9 + config.FIP_CONSTANT
    merged["Expected_ER"] = (merged["FIP_Windowed"] * merged["Expected_IP"] / 9).clip(lower=0)

    merged["DK_Points_Pitcher"] = (
        merged["Expected_IP"] * config.DFS_DK_PITCHER_IP_POINTS
        + merged["Expected_K"] * config.DFS_DK_PITCHER_K_POINTS
        + merged["Expected_BB"] * config.DFS_DK_PITCHER_BB_POINTS
        + merged["Expected_H_Allowed"] * config.DFS_DK_PITCHER_H_POINTS
        + merged["Expected_ER"] * config.DFS_DK_PITCHER_ER_POINTS
    )

    return merged[PITCHER_DFS_COLUMNS].sort_values("DK_Points_Pitcher", ascending=False)

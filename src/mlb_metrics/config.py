"""Named constants for the window lengths and blend weights used across the
metrics pipeline. Every value here was previously a hardcoded literal spread
across scripts/wave.py; consolidating them here is what lets Phase B (the
backtesting framework) grid-search alternative weightings without editing
metric logic.

Each *_WINDOWS list is a sequence of (days_back, weight) pairs, applied to a
per-window rate/stat via a straight weighted sum. days_back=None means "full
season to date". The order and values below reproduce the original script's
formulas exactly; note that windows/weights are NOT consistent between
metrics (e.g. WAVE weights recent data highest, PAVE weights it lowest) -
this reflects the original hand-tuning, not a design decision made here.
"""

from datetime import date

# 2026 MLB season boundaries used for the full-season Statcast pull.
SEASON_START = date(2026, 3, 25)
SEASON_END = date(2026, 10, 10)

# WAVE (hitter batting-average-based hit probability): full/81d/30d/10d.
WAVE_WINDOWS = [
    (None, 0.150),
    (81, 0.250),
    (30, 0.275),
    (10, 0.325),
]
# Trials-per-game used to convert an at-bat hit rate into a game hit
# probability: probability = 1 - (1 - rate) ** WAVE_TRIALS_PER_GAME.
WAVE_TRIALS_PER_GAME = 3.5

# PAVE (pitcher hits-allowed rate, adjusted for K/BB/HBP rate): full/30d/81d/15d.
PAVE_WINDOWS = [
    (None, 0.300),
    (30, 0.265),
    (81, 0.230),
    (15, 0.205),
]
# A pitcher must have thrown at least this fraction of the max at-bats faced
# by any pitcher in the pool to count toward the PAVE_PLUS league-average baseline.
PAVE_QUALIFIED_AB_FRACTION = 0.75
# Plate appearances assumed per start, used for Expected_Hits/Bases/HRs.
PAVE_PA_PER_START = 22

# Game_Hit_Probability (rate of games with >=1 hit, not at-bat rate): full/81d/30d/10d.
GAME_HIT_PROB_WINDOWS = [
    (None, 0.175),
    (81, 0.225),
    (30, 0.275),
    (10, 0.325),
]

# WHOPS/RC and WTB share the same window scheme: full/7d/30d/15d.
WHOPS_WTB_WINDOWS = [
    (None, 0.175),
    (7, 0.225),
    (30, 0.275),
    (15, 0.325),
]
# Trials-per-game used for WTB's Expected_Bases (same role as WAVE_TRIALS_PER_GAME).
WTB_TRIALS_PER_GAME = 3.5

# --- Team-level metrics ---

# strength / current_strength / pyth_strength blend: 10g/30g/81g/full.
TEAM_STRENGTH_WINDOWS = [
    (10, 0.5),
    (30, 0.3),
    (81, 0.1),
    (None, 0.1),
]

# offensive_edge (bases scored/allowed per game) blend: 10g/30g/81g/full.
OFFENSIVE_EDGE_WINDOWS = [
    (10, 0.3),
    (30, 0.3),
    (81, 0.2),
    (None, 0.2),
]

# suppression_resistance (1 - pct of games scoring under 3 runs) blend: 162g/81g/30g/10g.
SUPPRESSION_WINDOWS = [
    (162, 0.2),
    (81, 0.2),
    (30, 0.3),
    (10, 0.3),
]

# Pythagorean win-expectation exponent (custom-tuned, not the classic 2).
PYTHAGOREAN_EXPONENT = 1.83

# Every z-scored team metric (Strength, SOS, current_strength, pyth_strength,
# pyth_SOS, Confidence, pyth_Confidence, offensive_edge, suppression_resistance)
# is renormalized as 1 + (z_score * NORMALIZATION_Z_SCALE).
NORMALIZATION_Z_SCALE = 0.15

# Confidence = Strength + SOS * CONFIDENCE_SOS_WEIGHT (and same for pyth_*).
CONFIDENCE_SOS_WEIGHT = 0.3

# --- Backtesting (Phase B) ---

# Minimum combined full-season plate appearances (PA_L + PA_R) for a batter
# to be eligible as a daily "pick" - without this, a batter with a handful
# of at-bats and a lucky hit can show a probability of 1.0 and dominate the
# top of the rankings on pure sample-size noise. This matches a filter the
# original script computed but never actually applied to its output
# (`WAVE[(WAVE['l_at_bat'] + WAVE['r_at_bat']) > 30]`, wave.py:129).
BACKTEST_MIN_PLATE_APPEARANCES = 30

# Default number of top-ranked picks to log/evaluate per day.
BACKTEST_TOP_N = 5

# A hitter only qualifies as a pick if BOTH probability (the binomial-model
# estimate from at-bat-level WAVE) AND Game_Hit_Probability (the directly
# observed rate of games with >=1 hit) clear this bar. Either one alone is
# misleading: high Game_Hit_Probability with low probability is a batter who
# barely gets a hit most games (a lot of 1-for-4/5s); high probability with
# low Game_Hit_Probability is boom-or-bust (multi-hit games mixed with
# 0-fors) - neither is the "reliable to get a hit today" signal the pick is
# supposed to represent. Empirically validated via a 42-day git-history
# replay backtest (see git_backtest.py): requiring both >=0.7 and ranking the
# qualified pool by Approach (Game_Hit_Probability * probability) raised the
# resolved hit rate of the picks that actually clear DAILY_PICK_MIN_PROBABILITY
# from 55% to 65% and cut the Brier score from 0.284 to 0.260, without
# reducing pick coverage (still >=1 pick on all 42 backtested days).
HITTER_MIN_PROBABILITY = 0.7

# Stamped onto every hitter pick logged (predictions.select_picks) so
# evaluation.py/the dashboard can segment stats by which selection-logic
# version actually produced a given row, instead of silently blending old
# and new logic into one number - this is what let the qualifier/ranking
# change above (and any future recalibration) actually show up in "did it
# work" stats, rather than being diluted by history logged under old logic
# forever. Bump this string whenever select_picks' qualifier or ranking
# logic meaningfully changes.
HITTER_MODEL_VERSION = "v2-matchup-qualifier"

# Beat the Streak Tracker (dashboard): a batter is only "recommended" if
# evaluation._combined_probability (the mean of whichever of
# predicted_probability/Game_Hit_Probability, probability, and
# Matchup_Hit_Probability are available for that pick - see that function's
# docstring) clears this bar - on a day with no good matchups, that's zero
# picks; on a strong day, up to DAILY_PICK_MAX picks.
#
# Previously this gated on predicted_probability (Game_Hit_Probability)
# alone at 0.80 - a single-signal bar that ignored `probability` and
# `Matchup_Hit_Probability` entirely, and produced 0-pick days whenever GHP
# landed just under 0.80 even with a strong matchup (e.g. a real 0.79 GHP/
# strong-matchup day that should have surfaced a pick, didn't). Blending in
# the other two signals is the fix, but also means the same nominal
# threshold is no longer directly comparable to before - a mean of two or
# three probabilities runs lower than GHP alone whenever `probability`
# trails GHP (common - see JOINT_PROBABILITY_GATE_COLUMNS's docstring),
# so keeping 0.80 with the new blend would have been STRICTER, not looser
# (empirically: 3/42 zero-pick days at 0.80 in the backtest below, vs 1/42
# under the old GHP-only gate).
#
# 0.77 is empirically validated via a 42-day git-history replay backtest
# (git_backtest.reconstruct_historical_picks, resolved against persisted
# Statcast - Matchup_Hit_Probability is NaN throughout that replay window,
# since it's never persisted to git history, so this validates the
# GHP+probability blend specifically): the highest threshold that still
# achieves full day coverage (0/42 zero-pick days) while matching the best
# resolved hit rate/Brier score plateau seen across the whole 0.70-0.77
# range (62.5% hit rate, 0.2894 Brier, n=56 resolved picks) - thresholds
# above 0.77 both start producing zero-pick days again AND stop improving
# (sometimes worsening) hit rate/Brier on this sample. Revisit once more
# live picks accumulate carrying real (non-NaN) Matchup_Hit_Probability.
DAILY_PICK_MAX = 2
DAILY_PICK_MIN_PROBABILITY = 0.77

# --- Lineup awareness ---
#
# Backtesting found ~30% of logged top-5 picks had zero at-bats on the day
# they were picked - the batter wasn't even in the lineup. These qualifiers
# (see lineup.py, predictions.select_picks) use each batter's *historical*
# batting-order slot - derived from Statcast data already persisted, via
# data.assign_batting_order - as a proxy for "reliably plays, and plays high
# enough in the order to get real at-bats", without needing same-day lineup
# confirmation (which isn't available until a few hours before first pitch -
# see schedule.py).

# Rolling window (team games, not days) used to compute batting-order
# consistency. A game-count window, unlike the day-count *_WINDOWS lists
# above, since roster usage doesn't follow a calendar cadence.
LINEUP_WINDOW_GAMES = 20

# A batter's average batting-order slot over the window must be strictly
# above this to count as "top half" - the 9-man order's true median is 5,
# so 4.5 means strictly better than a 50/50 split, matching "more at-bats
# than the bottom of the order" precisely (a mean of exactly 5.0 is not
# "top half").
LINEUP_TOP_HALF_MAX_SLOT = 4.5

# Fraction of the team's games in the window (or fewer, early in the season)
# a batter must have actually started in to count as a regular, not a bench
# player on a hot week or a recent call-up riding an unsustainable streak.
# A rate, not an absolute game count, so it doesn't hard-disqualify every
# batter before a team has played LINEUP_WINDOW_GAMES games.
LINEUP_MIN_START_RATE = 0.6

# --- Matchup awareness (Part B) ---
#
# Blends a batter's own AB-level hit rate (WAVE) with the specific pitching
# they're projected to face today: their team's opponent's probable starter
# (most of their at-bats) and that opponent's bullpen (the rest), via a log5
# (odds-ratio) combination against the league-average PAVE - the standard
# sabermetric technique for combining two rate stats measured against a
# shared baseline (see matchup.py). Uses raw PAVE/Bullpen_PAVE (an actual
# batting-average-against-scale rate), not PAVE_PLUS/Bullpen_PAVE_PLUS (a
# league-normalized ratio) - log5 needs real rates on the same 0-1 scale as
# WAVE, not a ratio centered on 1.0. game_picks.py's team-level model is
# unaffected and keeps using PAVE_PLUS/clip_and_blend_pitching_quality (a
# different, still-appropriate use for its already-normalized composites).

# Assumed share of a batter's at-bats against the opposing starter vs. the
# opposing bullpen (roughly 2-3 of a 3-5 AB game). Sums to 1.0 so the blend
# has no systematic bias vs. the league-average PAVE it's measured against.
MATCHUP_STARTER_AB_SHARE = 0.6
MATCHUP_BULLPEN_AB_SHARE = 0.4

# PAVE_PLUS/Bullpen_PAVE_PLUS are clipped to this range before blending (not
# just clipping the final probability) - assemble_pitchers has no upper
# bound on individual PAVE_PLUS, so a small-sample outlier (an opener's short
# outing, early season) could otherwise multiply a good Game_Hit_Probability
# past 1.0 and erase real distinctions at the ceiling clip. Used by
# game_picks.py's clip_and_blend_pitching_quality (team-level model).
MATCHUP_PAVE_PLUS_CLIP = (0.5, 1.75)

# Same outlier protection as MATCHUP_PAVE_PLUS_CLIP, but expressed as a
# multiplier of that day's league-average PAVE (since raw PAVE isn't
# pre-normalized to a mean of 1.0 the way PAVE_PLUS is) - a starter/bullpen
# PAVE is clipped to [lo*league_pave, hi*league_pave] before blending. Used
# by matchup.py's clip_and_blend_pitching_pave (hitter-level model).
MATCHUP_PAVE_CLIP_MULTIPLIER = (0.5, 1.75)

# Fallback league-average PAVE (roughly historical MLB batting-average-against)
# used only when the day's `pave` table can't yield a real value (e.g. no
# qualified pitchers yet, or a test fixture) - matchup.py's log5 blend needs
# some league baseline to divide by and this should never bind in practice
# once the season is underway.
MATCHUP_LEAGUE_PAVE_FALLBACK = 0.245

# Whether a batter faces a lefty or righty starter today changes which of
# their platoon-split rates (WAVE_L/WAVE_R - see hitters.compute_wave) is
# the relevant one; blending both together (the old behavior) understates a
# real platoon-driven matchup either way. matchup.py picks WAVE_L/WAVE_R
# based on the probable starter's own Throws (pitchers.compute_pitcher_throws)
# and falls back to the handedness-blended WAVE when Throws is unknown
# (unannounced starter, not found in pave.csv) - same missing-data
# degradation pattern used throughout this module. Always applied (no
# separate weight knob) when the data is available - see
# MATCHUP_PARK_FACTOR_WEIGHT's docstring for the backtest that validated
# turning it on, which validated platoon and park together.

# Statcast's own venue proxy: each team's home games are treated as being
# at one park (see teams.compute_park_factors) - Park_Factor is the
# combined runs/game at that venue relative to the across-all-parks
# average, clipped to this range before blending into the matchup AB rate
# so a small-sample outlier park doesn't swing a probability further than
# any real park effect would (MLB parks realistically span roughly this
# range - Coors-like hitter's parks near the top, extreme pitcher's parks
# near the bottom).
MATCHUP_PARK_FACTOR_CLIP = (0.85, 1.15)

# How much of the (clipped) park effect gets applied to the matchup AB
# rate: effective_multiplier = 1 + MATCHUP_PARK_FACTOR_WEIGHT * (Park_Factor - 1).
# 0.0 is a no-op (fully off); 1.0 applies the park effect in full.
#
# Empirically validated via a 15-date persisted-Statcast backtest (recomputes
# wave/pave/confidence fresh per date via pipeline.compute_outputs, same
# technique game_picks_backtest.py's persisted variant uses - platoon/park
# columns don't exist in old git-committed snapshots, so a git-history
# replay can't see them), comparing four variants on the full top-N
# qualified-candidate pool (n=43-52 resolved picks - the more reliable
# sample; the further-filtered "recommended" rank<=2 subset only had
# n=11-18, too small to trust on its own and noisier in the opposite
# direction on this window):
#   baseline (neither):        58% hit rate, 0.286 Brier
#   platoon only:               62% hit rate, 0.261 Brier
#   park only:                  57% hit rate, 0.293 Brier (alone, roughly a wash)
#   both (this default):        65% hit rate, 0.244 Brier - best of all four
# Platoon and park together beat either alone or neither on the larger
# sample, so both ship on by default; revisit (especially park alone, whose
# solo read was weak here) once more dates accumulate.
MATCHUP_PARK_FACTOR_WEIGHT = 1.0

# --- Automated Game Picks ---
#
# Predicts a winner for each of today's games from team-level metrics (not
# hitter picks) - see game_picks.py. Unvalidated first pass, same as Matchup
# awareness above: logged and tracked before ever being trusted.

# Equal-weighted blend of the four team-level signals the user asked for by
# name into one offensive composite rating. All four inputs are already
# z-normalized to mean 1.0 (NORMALIZATION_Z_SCALE), so a straight weighted
# average needs no further rescaling. suppression_resistance is deliberately
# counted both directly and inside true_power (which already averages it
# with offensive_edge) - the user asked for both signals by name.
GAME_PICK_COMPOSITE_WEIGHTS = [
    ("pyth_Strength", 0.25),
    ("pyth_Confidence", 0.25),
    ("suppression_resistance", 0.25),
    ("true_power", 0.25),
]

# A game is only "picked" if the favored side's win probability clears this
# bar - a day can surface 0 or more picks depending on how much separation
# the model sees, not a forced pick every game. Much lower than
# DAILY_PICK_MIN_PROBABILITY (0.77): single-game MLB win probabilities are
# compressed near 50/50 even for real favorites, so reusing the hitter-pick
# bar would produce picks on almost no days. First-pass default, meant to be
# recalibrated once real data accumulates.
GAME_PICK_MIN_PROBABILITY = 0.58

# Floors each team's matchup-adjusted rating before computing
# home_rating / (home_rating + away_rating), purely as a degenerate-input
# guard - composites center around 1.0, so this shouldn't bind in practice.
GAME_PICK_RATING_FLOOR = 0.05

# How much Power_A_PLUS (total-bases-allowed rate against, a run-prevention/
# ERA-like signal - see pitchers.py's module docstring) contributes to the
# opposing-pitching-quality multiplier, blended against PAVE_PLUS (hit-rate
# against): final = (1-w)*PAVE_PLUS_quality + w*Power_A_PLUS_quality.
# Empirically validated via a 30+ day persisted-Statcast backtest (see
# game_picks.py's module docstring): an equal 0.5/0.5 blend beat pure
# PAVE_PLUS (w=0.0) on both accuracy (56.3% vs 54.3%) and Brier score
# (0.2493 vs 0.2517), and beat other weights tried (0.25, 1.0 i.e.
# Power_A_PLUS alone). Revisit once more resolved games accumulate.
GAME_PICK_SUSCEPTIBILITY_WEIGHT = 0.5

# Same purpose as HITTER_MODEL_VERSION above, for game_predictions.py -
# bump whenever compute_game_win_probabilities'/select_game_picks' logic
# meaningfully changes (e.g. GAME_PICK_SUSCEPTIBILITY_WEIGHT's introduction).
GAME_PICK_MODEL_VERSION = "v1"

# --- Age Curves (exploratory, separate page - not part of the daily pick pipeline) ---
#
# Given a current hitter's age and season stat line (traditional_stats.py,
# not WAVE/PAVE - Lahman's historical seasons have no Statcast-derived
# signal to compare against), finds the K nearest historical hitter-seasons
# at the same age (age_curve.py) and projects next season from what those
# comparables actually did next. v1 scope: hitters only, no era/park
# adjustment - see age_curve.py's module docstring.

# A separate curve/projection/comparable-list per metric, not one blended
# number - AVG/OBP/SLG age differently (power typically peaks earlier and
# declines faster than plate discipline; contact rate tends to be more
# stable), so a single composite would hide that. Same "metric" convention
# already used by predictions.csv/game_predictions.csv elsewhere in this
# project (a metric name plus generic value columns).
AGE_CURVE_METRICS = ["AVG", "OBP", "SLG", "OPS"]

# Minimum at-bats for a season (current or historical) to be eligible at
# all - a tiny-sample season can otherwise show an OPS of 0 or 3.000+ on
# pure noise, same reasoning as BACKTEST_MIN_PLATE_APPEARANCES above.
AGE_CURVE_MIN_AB = 200

# How many nearest same-age comparables (by whichever metric is in use) to
# use per projection.
AGE_CURVE_K_NEIGHBORS = 25

# Comparables are drawn from ages within this many years of the current
# player's age (0 = exact age match only). A small window trades strict
# age-comparability for a larger, less noisy comparable pool.
AGE_CURVE_AGE_WINDOW = 1

# Empirically validated (scripts/backtest_age_curve.py) against 500 real
# sampled player-seasons from 2010-2019 (29,692 qualified historical
# hitter-seasons total, 1871-present, from Lahman's actual database - not
# synthetic data), using only comparable data available at or before each
# test season's own year (no lookahead). Every metric beat the naive
# "always guess the sample's mean" baseline, with a real (not noise-level)
# positive correlation between projected and actual next-season value:
#   AVG: MAE 0.0246 vs. 0.0254 baseline, correlation 0.348 (n=355 scored)
#   OBP: MAE 0.0274 vs. 0.0284 baseline, correlation 0.430 (n=355 scored)
#   SLG: MAE 0.0535 vs. 0.0582 baseline, correlation 0.474 (n=355 scored)
#   OPS: MAE 0.0752 vs. 0.0789 baseline, correlation 0.445 (n=355 scored)
# The remaining 145/500 sampled seasons had no comparable with a resolvable
# next season and couldn't be scored (see project_next_season's docstring -
# reported, not hidden). SLG/OPS show the strongest signal; AVG/OBP beat
# the baseline only modestly, consistent with contact rate/plate discipline
# being harder to project from a same-age same-value comparable pool alone
# than power is. Re-run this backtest (and update these numbers) after any
# change to AGE_CURVE_K_NEIGHBORS/AGE_CURVE_AGE_WINDOW/AGE_CURVE_MIN_AB.

# Statcast plate-appearance outcome values that count as a "completed" event
# (used to filter pitch-by-pitch data down to one row per at-bat outcome).
COUNTED_EVENTS = [
    "field_out",
    "force_out",
    "single",
    "double",
    "strikeout",
    "home_run",
    "grounded_into_double_play",
    "triple",
    "fielders_choice_out",
    "double_play",
    "field_error",
    "fielders_choice",
    "strikeout_double_play",
    "walk",
    "hit_by_pitch",
]

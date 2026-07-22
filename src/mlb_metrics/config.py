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

# Beat the Streak Tracker (dashboard): a batter is only "recommended" if
# their predicted_probability clears this bar - on a day with no good
# matchups, that's zero picks; on a strong day, up to DAILY_PICK_MAX picks.
# 0.80 is close to the real historical rank-3/4 probability range (see
# evaluation.py docstring for the actual accuracy at that band), so most
# days still surface 2 picks and 0-pick days are the exception, not the rule.
DAILY_PICK_MAX = 2
DAILY_PICK_MIN_PROBABILITY = 0.80

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
# Blends a batter's own hit probability with the specific pitching they're
# projected to face today: their team's opponent's probable starter (most of
# their at-bats) and that opponent's bullpen (the rest). Unvalidated first
# pass - see matchup.py - meant to be logged alongside the unblended metric
# and backtested before ever becoming the "recommended" metric.

# Assumed share of a batter's at-bats against the opposing starter vs. the
# opposing bullpen (roughly 2-3 of a 3-5 AB game). Sums to 1.0 so the blend
# has no systematic bias vs. Game_Hit_Probability's own average, since
# PAVE_PLUS/Bullpen_PAVE_PLUS are both normalized to a league mean of 1.0.
MATCHUP_STARTER_AB_SHARE = 0.6
MATCHUP_BULLPEN_AB_SHARE = 0.4

# PAVE_PLUS/Bullpen_PAVE_PLUS are clipped to this range before blending (not
# just clipping the final probability) - assemble_pitchers has no upper
# bound on individual PAVE_PLUS, so a small-sample outlier (an opener's short
# outing, early season) could otherwise multiply a good Game_Hit_Probability
# past 1.0 and erase real distinctions at the ceiling clip.
MATCHUP_PAVE_PLUS_CLIP = (0.5, 1.75)

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
# DAILY_PICK_MIN_PROBABILITY (0.80): single-game MLB win probabilities are
# compressed near 50/50 even for real favorites, so reusing the hitter-pick
# bar would produce picks on almost no days. First-pass default, meant to be
# recalibrated once real data accumulates.
GAME_PICK_MIN_PROBABILITY = 0.58

# Floors each team's matchup-adjusted rating before computing
# home_rating / (home_rating + away_rating), purely as a degenerate-input
# guard - composites center around 1.0, so this shouldn't bind in practice.
GAME_PICK_RATING_FLOOR = 0.05

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

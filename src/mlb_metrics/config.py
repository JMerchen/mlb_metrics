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

# Beat the Streak Tracker (dashboard): a batter is only "recommended" if
# their predicted_probability clears this bar - on a day with no good
# matchups, that's zero picks; on a strong day, up to DAILY_PICK_MAX picks.
# 0.80 is close to the real historical rank-3/4 probability range (see
# evaluation.py docstring for the actual accuracy at that band), so most
# days still surface 2 picks and 0-pick days are the exception, not the rule.
DAILY_PICK_MAX = 2
DAILY_PICK_MIN_PROBABILITY = 0.80

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

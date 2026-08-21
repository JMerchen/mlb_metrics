"""Decision theory for Beat the Streak's actual sequential-decision
structure (quant-analytics item #4, slice 1 of "decision theory for the
actual game structure"): every prior signal in this project answers
"how likely is a hit today" - none of them answer the actually different
question "given the CURRENT streak and how many days remain, is today's
pick good enough to risk it, or is sitting out worth more?"

`predictions.select_picks`/`config.DAILY_PICK_MIN_PROBABILITY` apply the
same fixed 0.77 bar every day, with zero awareness of streak state. This
module solves the real problem underneath that: a finite-horizon optimal-
stopping / dynamic-programming problem, the same "reservation threshold"
shape as the classic asset-selling/secretary problem.

State: `s` = current streak length. Each of `horizon` remaining days has
an available pick, probability `p` of extending the streak (drawn from a
real empirical sample, not a fitted distribution - see
`solve_reservation_thresholds`). Decision: PLAY (succeed w.p. `p` ->
`s + gain`; fail -> `s` resets to 0) or SIT (`s` unchanged). Terminal
utility (season end) = final streak length itself (a deliberately simple,
monotonic utility - no discrete MLB.com prize-threshold modeling this
slice, per explicit user choice).

Bellman recursion (`r` = days remaining, `r=0` is terminal):
    V_0(s) = s
    V_r(s) = E_p[ max( V_{r-1}(s),                              # SIT
                       p * V_{r-1}(s+gain) + (1-p) * V_{r-1}(0)  # PLAY
                     ) ]
Since V_{r-1} is already known by induction, the optimal action at
(r, s, p) is a closed-form threshold:
    PLAY iff p > (V_{r-1}(s) - V_{r-1}(0)) / (V_{r-1}(s+gain) - V_{r-1}(0))
- provably non-decreasing in `s` (a longer streak needs a better pick to
risk it) and non-increasing as `r` shrinks (less time left to recover a
reset lowers the opportunity cost of sitting out today - a real,
slightly counterintuitive DP insight, not glossed over).

Explicit non-goals (see README's "Decision theory" section): no
correlation/joint modeling between two same-day picks (item #4's other,
separate sub-problem) - `p` is taken as-is, one real number per day,
exactly as `evaluation.streak_progression` already treats it. No "1 pick
vs. 2 picks" decision - only binary PLAY(today's picks as selected by the
existing rule)/SIT. Days are treated as i.i.d. draws from the same real
historical distribution - a real, stated simplification (serial
correlation across days, e.g. a league-wide hot/cold week, isn't
modeled).
"""

import pandas as pd


def solve_reservation_thresholds(p_samples: list[float], horizon: int, gain: float) -> dict:
    """Backward-induction DP over the real empirical `p_samples` (each a
    real historical day's own win probability - the expectation over `p`
    is a plain average over this finite sample, not a fitted/assumed
    continuous distribution).

    Streak state is tracked in "gain units" (`k` successes since the last
    reset, actual streak length = `k * gain`) - state grid k=0..horizon+1
    (one extra buffer state beyond the queryable range) so that computing
    V at the top of the QUERYABLE range (k=horizon) never needs to
    silently cap/approximate `k+1` against the grid edge. Valid, exact
    queries are k=0..horizon (streak values up to `horizon * gain`) -
    comfortably above any realistic real Beat the Streak streak length
    for a `horizon` sized to a real backtest window (dozens to low
    hundreds of dates).

    Returns {"V": V, "thresholds": thresholds, "horizon": horizon,
    "gain": gain} where V[r][k]/thresholds[r][k] are 2D lists indexed by
    days-remaining `r` (0..horizon) and streak-state `k` (0..horizon+1).
    `thresholds[r][k]` is `None` when SIT is optimal regardless of `p`
    (playing offers no real upside even at p=1 - only possible at the
    unqueryable top buffer state, included for completeness)."""
    if not p_samples:
        raise ValueError("p_samples must be non-empty - the DP needs a real empirical sample to average over")
    if horizon < 0:
        raise ValueError("horizon must be >= 0")
    if gain <= 0:
        raise ValueError("gain must be > 0 - a play that can't ever increase the streak isn't a real decision")

    n_states = horizon + 2  # k = 0..horizon+1 (see docstring: one buffer state past the queryable range)
    V = [[0.0] * n_states for _ in range(horizon + 1)]
    thresholds = [[None] * n_states for _ in range(horizon + 1)]

    for k in range(n_states):
        V[0][k] = k * gain

    for r in range(1, horizon + 1):
        for k in range(n_states):
            sit_value = V[r - 1][k]
            k_next = k + 1 if k + 1 < n_states else k  # only ever hit at the unqueryable top buffer state
            play_success_value = V[r - 1][k_next]
            play_fail_value = V[r - 1][0]

            denom = play_success_value - play_fail_value
            thresholds[r][k] = (sit_value - play_fail_value) / denom if denom > 0 else None

            total = 0.0
            for p in p_samples:
                play_value = p * play_success_value + (1 - p) * play_fail_value
                total += max(sit_value, play_value)
            V[r][k] = total / len(p_samples)

    return {"V": V, "thresholds": thresholds, "horizon": horizon, "gain": gain}


def should_play(streak: float, days_remaining: int, todays_probability: float, solved: dict) -> bool:
    """PLAY iff `todays_probability` clears the closed-form reservation
    threshold `solved` computed at (`days_remaining`, `streak`) -
    `solve_reservation_thresholds`'s own decision rule. `streak` is
    converted to the nearest gain-unit `k` (real streaks are integer
    picks-that-hit; `k = round(streak / solved["gain"])`). `days_remaining`
    is clipped to `solved`'s own solved horizon (querying beyond it reuses
    the longest-horizon solution rather than raising - the reservation
    threshold's real horizon-sensitivity is already captured within the
    solved range; a horizon further out than what was solved doesn't
    change the qualitative "which is better" comparison enough to justify
    refusing to answer)."""
    horizon = solved["horizon"]
    gain = solved["gain"]
    n_states = horizon + 2
    r = max(0, min(days_remaining, horizon))
    k = round(streak / gain)
    if k >= n_states - 1:
        raise ValueError(
            f"streak={streak} (k={k}) is at/beyond the unqueryable top buffer state "
            f"(solved for k=0..{horizon}) - re-solve with a larger horizon to query this streak"
        )

    threshold = solved["thresholds"][r][k]
    if threshold is None:
        return False
    return todays_probability > threshold


def estimate_gain(streak_progression: pd.DataFrame) -> float:
    """Real historical average picks-that-hit on a non-reset day
    (`evaluation.streak_progression`'s own output - a `hit` day's streak
    increment) - the empirical `gain` parameter for
    `solve_reservation_thresholds`, not an assumed constant
    `config.DAILY_PICK_MAX`. A day's own increment is `streak - previous
    streak` on any non-reset row; reset rows contribute nothing (matches
    `streak_progression`'s own "no game" contributes nothing precedent).
    Returns 0.0 (an honest, non-crashing "no information") if there's no
    real non-reset history to average over."""
    df = streak_progression.sort_values("date").reset_index(drop=True)
    increments = []
    previous = 0
    for _, row in df.iterrows():
        if not row["reset"]:
            increments.append(row["streak"] - previous)
        previous = row["streak"]
    return float(sum(increments) / len(increments)) if increments else 0.0

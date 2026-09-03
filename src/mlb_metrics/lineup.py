"""Batting-order consistency: is a batter actually a regular, playing high
enough in the order to get real at-bats - as opposed to a bench player
having a hot week or a recent call-up riding an unsustainable streak.

Derived entirely from data.assign_batting_order's output, itself derived
from Statcast data already being persisted (no external dependency, and it
works retroactively for any date the raw data already covers). This is a
game-count window (see config.LINEUP_WINDOW_GAMES), not the day-count
windows the rest of the pipeline's metrics use - roster usage doesn't follow
a calendar cadence the way rolling batting stats do, so it doesn't fit
hitters.py's `_blend_windows` machinery and instead follows teams.py's
existing rolling-by-game_id pattern.

compute_expected_plate_appearances turns a batter's own recent
batting-order slot into a real, empirically-derived Expected_PA (real
at-bats per game observed at that slot, league-wide, from already-
persisted history) - see its own docstring for why this replaced the old
LINEUP_TOP_HALF_MAX_SLOT hard cutoff (REMOVED, config.py).
"""

import numpy as np
import pandas as pd

from mlb_metrics import config

STARTER_MAX_BATTING_ORDER = 9


def compute_lineup_consistency(
    batting_order: pd.DataFrame, latest_team: pd.DataFrame, window: int | None = None
) -> pd.DataFrame:
    """Per batter, over their CURRENT team's last `window`
    (default config.LINEUP_WINDOW_GAMES) games (via `latest_team` - a
    batter traded mid-season has their window reset to the new team only,
    not blended with games played for the old one): `avg_batting_order`
    (mean slot across games they started - null if they never started for
    this team in the window) and `start_rate` (starts / min(team games in
    window, `window`)).

    batting_order: data.assign_batting_order's output.
    latest_team: [key_mlbam, team] - e.g. data.latest_team_for_batters's output.
    `window` defaults to None (looked up at call time, not def time, so
    tests can monkeypatch config.LINEUP_WINDOW_GAMES) rather than being a
    plain default argument - compute_expected_plate_appearances passes its
    own, shorter config.LINEUP_RECENT_WINDOW_GAMES instead.
    """
    window = window if window is not None else config.LINEUP_WINDOW_GAMES

    team_games = (
        batting_order[["team", "game_id"]]
        .drop_duplicates()
        .sort_values(["team", "game_id"], ascending=[True, False])
    )
    team_games["recency_rank"] = team_games.groupby("team").cumcount()
    recent_team_games = team_games[team_games["recency_rank"] < window]

    team_games_in_window = (
        recent_team_games.groupby("team", as_index=False)
        .size()
        .rename(columns={"size": "team_games_in_window"})
    )

    starts = batting_order[batting_order["batting_order"] <= STARTER_MAX_BATTING_ORDER].merge(
        recent_team_games[["team", "game_id"]], on=["team", "game_id"]
    )
    per_batter = starts.groupby(["team", "batter"], as_index=False).agg(
        avg_batting_order=("batting_order", "mean"),
        games_started=("game_id", "count"),
    )
    per_batter = per_batter.merge(team_games_in_window, on="team", how="left")
    per_batter["start_rate"] = per_batter["games_started"] / per_batter["team_games_in_window"].clip(lower=1)

    result = latest_team.rename(columns={"key_mlbam": "batter"}).merge(
        per_batter[["team", "batter", "avg_batting_order", "start_rate"]], on=["team", "batter"], how="left"
    )
    result["start_rate"] = result["start_rate"].fillna(0)
    # avg_batting_order is intentionally left null for a batter who never
    # started for their current team in the window - there's no slot to
    # average, and compute_expected_plate_appearances below correctly
    # falls back to the league-average Expected_PA for a null rather than
    # treating it as slot 0 (the best possible spot).

    return result.rename(columns={"batter": "key_mlbam"})[["key_mlbam", "avg_batting_order", "start_rate"]]


def compute_expected_plate_appearances(
    data_with_game_id: pd.DataFrame,
    batting_order: pd.DataFrame,
    latest_team: pd.DataFrame,
    window: int | None = None,
    shrinkage: float | None = None,
) -> pd.DataFrame:
    """[key_mlbam, Recent_Avg_Batting_Order, Expected_PA] - a real,
    empirically-derived estimate of how many at-bats a batter is likely to
    get TODAY, driven by where they've actually been hitting in the order
    recently - SHRUNK toward the league-flat config.WAVE_TRIALS_PER_GAME
    by `shrinkage` (default config.EXPECTED_PA_SHRINKAGE - see its own
    docstring for the real calibration regression that using the raw,
    unshrunk estimate at full strength was found to cause, and why: a
    real batter's actual PA in a given game is variable - early exits,
    pinch-hits, extra innings - and plugging a single point ESTIMATE of
    it into 1-(1-p)**n (concave in n) systematically overpredicts the
    true expected hit probability, worse the further the estimate sits
    from the flat default). `Expected_PA` returned here is already the
    shrunk value - hitters.assemble_hitters/
    matchup.compute_matchup_hit_probability use it as-is, the same
    "the exposed value IS the regularized one" convention compute_wave
    already uses for WAVE itself (helpers.shrink_rate).

    Replaces the old LINEUP_TOP_HALF_MAX_SLOT hard cutoff (REMOVED - see
    its own docstring in config.py for the real full-season backtest that
    found it excluding batters who hit BETTER than the ones it let
    through). Batting order still matters - a leadoff hitter really does
    get more real at-bats per game than a #9 hitter, which is real extra
    opportunity for a hit at an identical per-AB rate - but instead of a
    pass/fail gate, this turns it into a continuous per-batter trials
    count that hitters.assemble_hitters/matchup.compute_matchup_hit_probability
    use in place of the league-flat config.WAVE_TRIALS_PER_GAME.

    Two real, no-lookahead ingredients, both derived from data already
    persisted for this run - nothing hardcoded/assumed:

    1. Recent_Avg_Batting_Order: compute_lineup_consistency's own
       avg_batting_order, but over a SHORTER window (`window`, default
       config.LINEUP_RECENT_WINDOW_GAMES=7, not LINEUP_WINDOW_GAMES=20) -
       "how many at-bats will they get today" is answered by where a
       batter has been hitting lately, not a slower-moving longer-run
       average. Null (no starts for their current team in the window)
       the same way avg_batting_order already can be.

    2. A real, EMPIRICAL batting-order-slot -> plate-appearances table:
       for every real (game_id, batter) row in `data_with_game_id` with a
       real starting batting_order (1-9), how many distinct at-bats
       (at_bat_number) did that batter actually get in that game, averaged
       by slot, league-wide - not an assumed/hardcoded table, so it's
       exactly as reliable as the amount of real season data behind it
       (same "improves as more of the season is observed" property every
       other rate in this project already has).

    Each batter's Recent_Avg_Batting_Order (typically fractional, e.g.
    4.2) is linearly interpolated against that real per-slot table -
    interpolating rather than rounding to the nearest whole slot uses the
    real fractional precision instead of discarding it. A batter with no
    recent starts for their current team (null Recent_Avg_Batting_Order),
    or on a date too early in the season for the per-slot table to exist
    at all, falls back to the real league-wide average PA/game across all
    real starters that history - the same "no information, use the
    average" default every other missing-signal fallback in this project
    uses, and never worse-informed than the flat WAVE_TRIALS_PER_GAME
    constant it replaces. Shrinkage (below) applies uniformly to this
    fallback too, not just the per-batter interpolated estimate - simpler
    than a special case, and league_avg_pa is itself already just a
    different point estimate of the same underlying quantity
    WAVE_TRIALS_PER_GAME approximates, so it warrants the same real-vs-
    variable-outcome caution.
    """
    window = window if window is not None else config.LINEUP_RECENT_WINDOW_GAMES
    shrinkage = shrinkage if shrinkage is not None else config.EXPECTED_PA_SHRINKAGE

    pa_per_game = (
        data_with_game_id.groupby(["game_id", "batter"])["at_bat_number"]
        .nunique()
        .rename("PA")
        .reset_index()
    )
    starters_with_pa = batting_order[batting_order["batting_order"] <= STARTER_MAX_BATTING_ORDER].merge(
        pa_per_game, on=["game_id", "batter"], how="inner"
    )

    slot_pa = starters_with_pa.groupby("batting_order")["PA"].mean()
    league_avg_pa = starters_with_pa["PA"].mean() if len(starters_with_pa) else config.WAVE_TRIALS_PER_GAME

    recent = compute_lineup_consistency(batting_order, latest_team, window=window)
    recent = recent.rename(columns={"avg_batting_order": "Recent_Avg_Batting_Order"})

    expected_pa = pd.Series(league_avg_pa, index=recent.index, dtype=float)
    known_slot = recent["Recent_Avg_Batting_Order"].notna()
    if not slot_pa.empty and known_slot.any():
        slots = slot_pa.index.to_numpy(dtype=float)
        values = slot_pa.to_numpy(dtype=float)
        order = np.argsort(slots)
        slots, values = slots[order], values[order]
        expected_pa[known_slot] = np.interp(
            recent.loc[known_slot, "Recent_Avg_Batting_Order"], slots, values, left=values[0], right=values[-1]
        )

    # Shrink the raw estimate toward the flat league constant (see
    # config.EXPECTED_PA_SHRINKAGE's docstring for the real calibration
    # evidence) - shrinkage=1.0 keeps the raw estimate unchanged,
    # shrinkage=0.0 collapses back to the flat constant for everyone.
    recent["Expected_PA"] = config.WAVE_TRIALS_PER_GAME + (expected_pa - config.WAVE_TRIALS_PER_GAME) * shrinkage

    return recent[["key_mlbam", "Recent_Avg_Batting_Order", "Expected_PA"]]

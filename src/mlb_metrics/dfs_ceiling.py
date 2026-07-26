"""Ceiling/volatility signal for DFS Player Rankings (docs/dfs.html) - a
new, ADDITIONAL informational column alongside the existing mean
projection (DK_Points_Hitter/DK_Points_Pitcher), not a replacement for it.

## Why this exists

GPP (tournament) DFS lineups are won by boom/spike-game players, not
players who reliably score near their own mean - a real, well-known DFS
strategy concept (the user's own framing that motivated this: "the
winner won't have players getting 5 points across their lineup, they're
more likely to have lucked into players averaging 15 or so"). The
existing DK_Points_Hitter/DK_Points_Pitcher are both mean-style
projections; nothing in this project measured a player's UPSIDE before
this module.

`Ceiling_DK_Points` is the `config.DFS_CEILING_PERCENTILE`-th percentile
(default 90th) of a player's own REAL historical modeled DK points -
`dfs_backtest.compute_actual_hitter_dk_points`/
`compute_actual_pitcher_dk_points` applied per real game date they
played, not a projected/modeled ceiling. This is deliberately an
OUTCOME-history signal, not a new projection model - the existing
DK_Points_Hitter/Pitcher already cover the "project from ingredients"
approach; this instead asks "how big has this player's real game-to-game
swing actually been."

## Small-sample fallback

A player with fewer than `config.DFS_CEILING_MIN_GAMES` real scored games
has too little history for a meaningful per-player percentile (a
rookie's 3rd game could literally BE their whole "ceiling" sample, not a
real percentile) - falls back to the GROUP-WIDE (all hitters', or all
pitchers', pooled) percentile at the same level instead of computing a
degenerate one, the same small-sample philosophy
`MATCHUP_LEAGUE_PAVE_FALLBACK` already uses elsewhere in this project.
`Ceiling_Source` ("player"/"group_fallback") makes this visible rather
than silently blending it in.

## Why informational-only, not the optimizer's default objective

`backtest_ceiling_signal` below checks, with the same no-lookahead
discipline as every other backtest in this project, whether ranking
players by `Ceiling_DK_Points` actually enriches for real boom-game days
better than ranking by the existing mean projection. Until that's
genuinely validated, defaulting the lineup optimizer to maximize ceiling
instead of mean points would be shipping an unvalidated assumption as if
it were proven - `scripts/build_optimal_lineup.py` instead exposes an
explicit `--objective {mean,ceiling}` flag (default `mean`, preserving
all existing behavior) so a user can opt in and see the tradeoff, rather
than having the objective silently changed under them. See README for
the actual backtest numbers."""

import numpy as np
import pandas as pd

from mlb_metrics import config, data, dfs_backtest


def compute_player_dk_points_history(persisted: pd.DataFrame, as_of_date=None) -> dict[str, pd.DataFrame]:
    """{"hitters": df[key_mlbam, game_date, Actual_DK_Points_Modeled],
    "pitchers": df[...]} - one row per player per real game date they had a
    scored plate appearance (hitters) or recorded start (pitchers), via
    dfs_backtest.compute_actual_hitter_dk_points/
    compute_actual_pitcher_dk_points applied date-by-date, so the later
    percentile computation sees the real game-to-game distribution rather
    than one already-blended number.

    `as_of_date`, if given, excludes any game_date >= as_of_date (the same
    no-lookahead discipline every other backtest in this project uses) -
    live callers building today's ranking naturally only ever have
    persisted history up through yesterday anyway, but
    backtest_ceiling_signal below needs this explicit cutoff to honestly
    replay a past date."""
    empty = pd.DataFrame(columns=["key_mlbam", "game_date", "Actual_DK_Points_Modeled"])
    if persisted is None or persisted.empty:
        return {"hitters": empty.copy(), "pitchers": empty.copy()}

    if as_of_date is not None:
        persisted = persisted[persisted["game_date"] < as_of_date]
    if persisted.empty:
        return {"hitters": empty.copy(), "pitchers": empty.copy()}

    hitter_events = data.completed_events(persisted, ["game_date", "batter", "events", "bat_score", "post_bat_score"])
    hitter_rows = []
    for date, day in hitter_events.groupby("game_date"):
        day_points = dfs_backtest.compute_actual_hitter_dk_points(day)
        day_points["game_date"] = date
        hitter_rows.append(day_points)
    hitters = pd.concat(hitter_rows, ignore_index=True) if hitter_rows else empty.copy()

    pitcher_events = data.completed_events(persisted, ["game_date", "pitcher", "events"])
    pitcher_rows = []
    for date, day in pitcher_events.groupby("game_date"):
        day_points = dfs_backtest.compute_actual_pitcher_dk_points(day)[["key_mlbam", "Actual_DK_Points_Modeled"]]
        day_points["game_date"] = date
        pitcher_rows.append(day_points)
    pitchers = pd.concat(pitcher_rows, ignore_index=True) if pitcher_rows else empty.copy()

    return {"hitters": hitters, "pitchers": pitchers}


def compute_ceiling_percentiles(
    history: pd.DataFrame,
    percentile: float = config.DFS_CEILING_PERCENTILE,
    min_games: int = config.DFS_CEILING_MIN_GAMES,
) -> pd.DataFrame:
    """[key_mlbam, Ceiling_DK_Points, n_games, Ceiling_Source] - per-player
    `percentile`-th percentile of their own Actual_DK_Points_Modeled
    history (`history` is one of compute_player_dk_points_history's two
    output frames). A player with fewer than `min_games` real scored games
    falls back to the GROUP-WIDE (every row in `history`) percentile at the
    same level instead of a noisy small-sample per-player number -
    `Ceiling_Source` distinguishes "player" from "group_fallback" rather
    than hiding which applies."""
    columns = ["key_mlbam", "Ceiling_DK_Points", "n_games", "Ceiling_Source"]
    if history.empty:
        return pd.DataFrame(columns=columns)

    # compute_actual_pitcher_dk_points's ip_safe = Actual_IP.replace(0, pd.NA)
    # can upcast Actual_DK_Points_Modeled to an object dtype further downstream
    # (harmless for the element-wise arithmetic dfs_backtest.py itself does,
    # but pandas' groupby quantile requires numeric dtype) - coerce here
    # rather than touching that already-validated function.
    history = history.copy()
    history["Actual_DK_Points_Modeled"] = pd.to_numeric(history["Actual_DK_Points_Modeled"])

    q = percentile / 100
    group_ceiling = history["Actual_DK_Points_Modeled"].quantile(q)

    grouped = history.groupby("key_mlbam")["Actual_DK_Points_Modeled"]
    result = pd.DataFrame({
        "key_mlbam": grouped.size().index,
        "Ceiling_DK_Points": grouped.quantile(q).to_numpy(),
        "n_games": grouped.size().to_numpy(),
    })
    result["Ceiling_Source"] = "player"

    small_sample = result["n_games"] < min_games
    result.loc[small_sample, "Ceiling_DK_Points"] = group_ceiling
    result.loc[small_sample, "Ceiling_Source"] = "group_fallback"

    return result[columns]


def backtest_ceiling_signal(raw_dir: str = "data/raw", season: int | None = None, days: int = 20) -> dict:
    """No-lookahead validation: for each of the last `days` real game dates
    (the same sample dfs_backtest.backtest_dfs_projections uses),
    Ceiling_DK_Points is computed from ONLY history strictly before that
    date, then checked against two questions per player type:

    - `ceiling_correlation`: does a higher Ceiling_DK_Points predict a
      better real day at all (same MAE/correlation-style check every other
      signal in this project gets)?
    - `ceiling_capture_rate` vs `mean_projection_capture_rate`: of the
      player-days that ACTUALLY landed in that date's real top decile
      (top `100 - DFS_CEILING_PERCENTILE` percent by Actual_DK_Points_Modeled),
      what fraction were ALSO in the top decile by Ceiling_DK_Points going
      in, vs. the same fraction using the existing mean heuristic
      projection (DK_Points_Hitter/DK_Points_Pitcher) instead - the
      DFS-relevant question of whether ranking by ceiling actually
      surfaces spike-game players better than ranking by mean.

    Returns {"hitters": {...}, "pitchers": {...}}, each either the metrics
    above or {"n": 0} when there's not enough scored data for that player
    type to report anything meaningful."""
    season = season or config.SEASON_START.year
    persisted = data.load_persisted_statcast(raw_dir, season)
    if persisted is None or persisted.empty:
        return {"hitters": {"n": 0}, "pitchers": {"n": 0}}

    heuristic = dfs_backtest.backtest_dfs_projections(raw_dir, season, days)

    results = {}
    for player_type, projection_col in (("hitters", "DK_Points_Hitter"), ("pitchers", "DK_Points_Pitcher")):
        df = heuristic[player_type]
        if df.empty:
            results[player_type] = {"n": 0}
            continue

        rows = []
        for date, day_df in df.groupby("date"):
            history = compute_player_dk_points_history(persisted, as_of_date=date)
            ceiling = compute_ceiling_percentiles(history[player_type])
            if ceiling.empty:
                continue
            merged = day_df.merge(ceiling[["key_mlbam", "Ceiling_DK_Points"]], on="key_mlbam", how="inner")
            if merged.empty:
                continue
            rows.append(merged)

        if not rows:
            results[player_type] = {"n": 0}
            continue

        combined = pd.concat(rows, ignore_index=True)
        n = len(combined)
        if n < 2:
            results[player_type] = {"n": n}
            continue

        # See compute_ceiling_percentiles's comment: compute_actual_pitcher_
        # dk_points's FIP-safe-division step can leave Actual_DK_Points_Modeled
        # (and anything merged alongside it) as an object-dtype column, which
        # np.corrcoef/quantile both reject.
        for column in ("Ceiling_DK_Points", "Actual_DK_Points_Modeled", projection_col):
            combined[column] = pd.to_numeric(combined[column])

        ceiling_correlation = float(np.corrcoef(combined["Ceiling_DK_Points"], combined["Actual_DK_Points_Modeled"])[0, 1])

        top_fraction = 1 - config.DFS_CEILING_PERCENTILE / 100
        actual_cutoff = combined["Actual_DK_Points_Modeled"].quantile(1 - top_fraction)
        actual_top = combined[combined["Actual_DK_Points_Modeled"] >= actual_cutoff]

        if actual_top.empty:
            results[player_type] = {"n": n, "ceiling_correlation": ceiling_correlation, "n_actual_top_decile_days": 0}
            continue

        ceiling_cutoff = combined["Ceiling_DK_Points"].quantile(1 - top_fraction)
        mean_cutoff = combined[projection_col].quantile(1 - top_fraction)

        results[player_type] = {
            "n": n,
            "ceiling_correlation": ceiling_correlation,
            "n_actual_top_decile_days": len(actual_top),
            "ceiling_capture_rate": float((actual_top["Ceiling_DK_Points"] >= ceiling_cutoff).mean()),
            "mean_projection_capture_rate": float((actual_top[projection_col] >= mean_cutoff).mean()),
        }

    return results

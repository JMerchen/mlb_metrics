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
explicit `--objective {mean,ceiling,boom}` flag (default `mean`,
preserving all existing behavior) so a user can opt in and see the
tradeoff, rather than having the objective silently changed under them.
See README for the actual backtest numbers.

## Boom_Adjusted_DK_Points: neither pure mean nor pure ceiling

Pure ceiling (a single 90th-percentile game) rewards raw upside
regardless of how OFTEN a player actually reaches it - a true boom-or-bust
signal. The user explicitly wants neither that nor the plain mean: a
player with real, frequent upside swings (sometimes 20, sometimes 5,
sometimes 0, averaging 4.8) should rank above a metronomic player
averaging 5 who never deviates - credit for genuine volatility, not just
a max, and not blind to the mean either.

`compute_upside_deviation` computes an UPSIDE-ONLY semi-deviation of a
player's real historical DK points: `sqrt(mean((x - historical_mean)
clipped at 0) ** 2)` over ALL their games (not just the boom ones) - a
below-average game contributes exactly 0, so this only measures spread
above the mean, and dividing by the FULL game count (not just the boom
subset) means a player who booms rarely scores lower than one who booms
often at similar magnitude, matching real DFS value (boom FREQUENCY
matters, not just size). This is deliberately a semi-deviation, not plain
standard deviation - plain stdev would reward bust-heavy inconsistency
exactly as much as boom-heavy inconsistency, which is the opposite of
what "I'd rather have the volatile player" actually means.

`compute_boom_adjusted_score(mean_points, upside_deviation, k)` is
`mean_points + k * upside_deviation` - a risk-SEEKING blend, literally the
mirror image of a Sharpe ratio (which subtracts deviation to reward
consistency). `k=0` reduces to the plain mean; `backtest_boom_adjusted_signal`
below grid-searches `k` against real data (same no-lookahead, same
top-decile "capture rate" methodology as backtest_ceiling_signal) to pick
a value that's actually validated to help, not guessed - see
config.DFS_BOOM_ADJUSTED_K's docstring for the chosen value and why.

## Matchup_Boom_Score: WHICH matchups make a boom more likely today

Both `Ceiling_DK_Points` and `Boom_Adjusted_DK_Points` are matchup-blind
on their volatility side - they measure how much a player has swung in
the PAST, with no read on whether TODAY's specific opponent makes a boom
more or less likely. The user's own framing: real winning lineups need "a
couple players to boom... we need to predict most likely booms for the
right price based on matchup." That's a genuinely different question -
P(boom TODAY | this matchup) - not just "has this player ever boomed."

`compute_boom_threshold(history, percentile)` is a single FIXED,
GROUP-WIDE point value (the percentile pooled across every player's real
game, not each player's own percentile) - deliberately NOT a per-player
threshold. A per-player threshold would be tautological here: by
construction, roughly `100 - percentile` percent of any player's OWN
games clear their OWN percentile, so every player's "boom rate" against
their own bar would converge to about the same number regardless of how
boom-prone they really are. A single shared bar lets `compute_boom_rate`
measure a real, meaningfully different quantity per player: how often
THEY, specifically, clear a bar that's hard for the average player to
clear.

`compute_matchup_boom_score(boom_rate, matchup_ratio)` is
`boom_rate * matchup_ratio` - `dfs.compute_matchup_adjustment`'s existing
Matchup_Ratio (today's real opposing pitcher/park-adjusted signal,
already computed as part of `DK_Points_Hitter`) scales the player's own
historical boom rate up or down for today's specific matchup. This is
explicitly a SCORE, not a calibrated probability, despite the name
"boom" in it - it is not clipped to [0, 1] and has not been fit/validated
as a real probability, only as a RANKING signal (see the backtest below
for what was actually validated: relative ordering, not absolute
calibration).

**Hitters only.** Pitchers have no equivalent "Matchup_Ratio" signal in
this project (their own mean projection has no separate matchup-ratio
multiplier the way `compute_hitter_dk_points` does), and the earlier
Ceiling/Boom-Adjusted backtests already found no real upside-signal edge
for pitchers - not worth inventing a new pitcher-side signal on a
foundation that's already shown no signal.

**Small sample size is a real risk here, checked honestly in the
backtest below, not glossed over**: booms are inherently rare (~10% of
games by construction at the default percentile), and this signal
further conditions on matchup context - the real backtest sample sizes
are reported alongside the results specifically so a reader can judge
whether they're large enough to trust."""

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


def compute_upside_deviation(
    history: pd.DataFrame,
    min_games: int = config.DFS_CEILING_MIN_GAMES,
) -> pd.DataFrame:
    """[key_mlbam, Upside_Deviation, n_games, Upside_Deviation_Source] -
    per-player UPSIDE-ONLY semi-deviation of their own real historical
    Actual_DK_Points_Modeled (`history` is one of
    compute_player_dk_points_history's two output frames):
    sqrt(mean((x - their_own_historical_mean).clip(lower=0) ** 2)) over
    ALL their games. Below-average games contribute exactly 0 (this is a
    semi-deviation, not plain standard deviation - plain stdev would
    reward bust-heavy inconsistency exactly as much as boom-heavy
    inconsistency, the opposite of what this signal is for), and dividing
    by the FULL game count rather than just the boom subset means a
    player who booms rarely scores lower than one who booms often at a
    similar magnitude - boom FREQUENCY matters, not just size.

    Small-sample fallback mirrors compute_ceiling_percentiles: a player
    with fewer than `min_games` real scored games falls back to the
    GROUP-WIDE upside deviation (computed the same way, pooling every row
    in `history`) instead of a noisy per-player number -
    `Upside_Deviation_Source` marks which applies."""
    columns = ["key_mlbam", "Upside_Deviation", "n_games", "Upside_Deviation_Source"]
    if history.empty:
        return pd.DataFrame(columns=columns)

    history = history.copy()
    history["Actual_DK_Points_Modeled"] = pd.to_numeric(history["Actual_DK_Points_Modeled"])

    def _upside_semi_deviation(values: pd.Series) -> float:
        deviations = (values - values.mean()).clip(lower=0)
        return float(((deviations ** 2).sum() / len(values)) ** 0.5)

    group_deviation = _upside_semi_deviation(history["Actual_DK_Points_Modeled"])

    grouped = history.groupby("key_mlbam")["Actual_DK_Points_Modeled"]
    result = pd.DataFrame({
        "key_mlbam": grouped.size().index,
        "Upside_Deviation": grouped.apply(_upside_semi_deviation).to_numpy(),
        "n_games": grouped.size().to_numpy(),
    })
    result["Upside_Deviation_Source"] = "player"

    small_sample = result["n_games"] < min_games
    result.loc[small_sample, "Upside_Deviation"] = group_deviation
    result.loc[small_sample, "Upside_Deviation_Source"] = "group_fallback"

    return result[columns]


def compute_boom_adjusted_score(mean_points: pd.Series, upside_deviation: pd.Series, k: float) -> pd.Series:
    """mean_points + k * upside_deviation - a risk-SEEKING blend (the
    mirror of a Sharpe ratio, which subtracts deviation to reward
    consistency instead). k=0 reduces to the plain mean projection; see
    backtest_boom_adjusted_signal for how k is chosen from real data."""
    return mean_points + k * upside_deviation


def backtest_boom_adjusted_signal(
    raw_dir: str = "data/raw",
    season: int | None = None,
    days: int = 20,
    k_grid: list[float] | None = None,
) -> dict:
    """No-lookahead grid search (same 20-date sample and top-decile
    "capture rate" methodology as backtest_ceiling_signal) of
    compute_boom_adjusted_score across `k_grid` (default
    config.DFS_BOOM_ADJUSTED_K_GRID), to pick a k that's actually
    validated on real data rather than guessed. Upside_Deviation is
    computed from ONLY history strictly before each test date.

    Returns {"hitters": {k: {...same metric shape as
    backtest_ceiling_signal...}, ...}, "pitchers": {...}} - one result per
    k per player type, so the best k can be chosen by inspection (see
    config.DFS_BOOM_ADJUSTED_K's docstring for the value ultimately
    picked and why)."""
    if k_grid is None:
        k_grid = config.DFS_BOOM_ADJUSTED_K_GRID
    season = season or config.SEASON_START.year
    persisted = data.load_persisted_statcast(raw_dir, season)
    if persisted is None or persisted.empty:
        return {"hitters": {k: {"n": 0} for k in k_grid}, "pitchers": {k: {"n": 0} for k in k_grid}}

    heuristic = dfs_backtest.backtest_dfs_projections(raw_dir, season, days)

    results = {"hitters": {}, "pitchers": {}}
    for player_type, projection_col in (("hitters", "DK_Points_Hitter"), ("pitchers", "DK_Points_Pitcher")):
        df = heuristic[player_type]
        if df.empty:
            for k in k_grid:
                results[player_type][k] = {"n": 0}
            continue

        # Compute the per-date (mean projection, upside deviation, actual)
        # rows ONCE (no-lookahead history per date), then score every k
        # against the SAME rows, instead of recomputing history per k.
        rows = []
        for date, day_df in df.groupby("date"):
            history = compute_player_dk_points_history(persisted, as_of_date=date)
            deviation = compute_upside_deviation(history[player_type])
            if deviation.empty:
                continue
            merged = day_df.merge(deviation[["key_mlbam", "Upside_Deviation"]], on="key_mlbam", how="inner")
            if merged.empty:
                continue
            rows.append(merged)

        if not rows:
            for k in k_grid:
                results[player_type][k] = {"n": 0}
            continue

        combined = pd.concat(rows, ignore_index=True)
        for column in ("Upside_Deviation", "Actual_DK_Points_Modeled", projection_col):
            combined[column] = pd.to_numeric(combined[column])

        n = len(combined)
        if n < 2:
            for k in k_grid:
                results[player_type][k] = {"n": n}
            continue

        top_fraction = 1 - config.DFS_CEILING_PERCENTILE / 100
        actual_cutoff = combined["Actual_DK_Points_Modeled"].quantile(1 - top_fraction)
        actual_top = combined[combined["Actual_DK_Points_Modeled"] >= actual_cutoff]

        for k in k_grid:
            score = compute_boom_adjusted_score(combined[projection_col], combined["Upside_Deviation"], k)
            correlation = float(np.corrcoef(score, combined["Actual_DK_Points_Modeled"])[0, 1])

            if actual_top.empty:
                results[player_type][k] = {"n": n, "correlation": correlation, "n_actual_top_decile_days": 0}
                continue

            score_cutoff = score.quantile(1 - top_fraction)
            capture_rate = float((score.loc[actual_top.index] >= score_cutoff).mean())

            results[player_type][k] = {
                "n": n,
                "correlation": correlation,
                "n_actual_top_decile_days": len(actual_top),
                "capture_rate": capture_rate,
            }

    return results


def compute_boom_threshold(history: pd.DataFrame, percentile: float = config.DFS_CEILING_PERCENTILE) -> float:
    """A single FIXED, GROUP-WIDE `percentile`-th percentile of every real
    game in `history` (`history` is one of compute_player_dk_points_history's
    output frames) - deliberately NOT a per-player threshold (see module
    docstring for why a per-player bar would be tautological here).
    Returns float('nan') for empty history."""
    if history.empty:
        return float("nan")
    points = pd.to_numeric(history["Actual_DK_Points_Modeled"])
    return float(points.quantile(percentile / 100))


def compute_boom_rate(
    history: pd.DataFrame,
    threshold: float,
    min_games: int = config.DFS_CEILING_MIN_GAMES,
) -> pd.DataFrame:
    """[key_mlbam, Boom_Rate, n_games, Boom_Rate_Source] - per-player
    fraction of their real historical games that cleared the FIXED
    `threshold` (from compute_boom_threshold, shared across every player -
    NOT each player's own percentile). A player with fewer than
    `min_games` real scored games falls back to the GROUP-WIDE clear-rate
    (every row in `history`, same threshold) instead of a noisy
    small-sample per-player rate - `Boom_Rate_Source` marks which
    applies."""
    columns = ["key_mlbam", "Boom_Rate", "n_games", "Boom_Rate_Source"]
    if history.empty or pd.isna(threshold):
        return pd.DataFrame(columns=columns)

    history = history.copy()
    history["Actual_DK_Points_Modeled"] = pd.to_numeric(history["Actual_DK_Points_Modeled"])
    history["_boomed"] = (history["Actual_DK_Points_Modeled"] >= threshold).astype(float)

    group_rate = history["_boomed"].mean()

    grouped = history.groupby("key_mlbam")["_boomed"]
    result = pd.DataFrame({
        "key_mlbam": grouped.size().index,
        "Boom_Rate": grouped.mean().to_numpy(),
        "n_games": grouped.size().to_numpy(),
    })
    result["Boom_Rate_Source"] = "player"

    small_sample = result["n_games"] < min_games
    result.loc[small_sample, "Boom_Rate"] = group_rate
    result.loc[small_sample, "Boom_Rate_Source"] = "group_fallback"

    return result[columns]


def compute_matchup_boom_score(boom_rate: pd.Series, matchup_ratio: pd.Series) -> pd.Series:
    """boom_rate * matchup_ratio - a player's own real historical boom
    FREQUENCY (against a fixed, group-wide bar - see compute_boom_rate),
    scaled by today's real matchup favorability
    (dfs.compute_matchup_adjustment's Matchup_Ratio). A ranking/score, NOT
    a calibrated probability - see module docstring."""
    return boom_rate * matchup_ratio


def backtest_matchup_boom_signal(
    raw_dir: str = "data/raw",
    season: int | None = None,
    days: int = 20,
    percentile: float = config.DFS_CEILING_PERCENTILE,
) -> dict:
    """No-lookahead backtest (same 20-date sample as the other DFS
    backtests, HITTERS ONLY - see module docstring for why), checking
    whether Matchup_Boom_Score actually flags real boom days better than
    Boom_Rate alone (context-free) or the plain mean projection alone
    (DK_Points_Hitter). For each test date, compute_boom_threshold and
    compute_boom_rate are computed from ONLY history strictly before that
    date; "did they boom" is that date's REAL Actual_DK_Points_Modeled
    clearing that date's (no-lookahead) threshold. Capture rate: of the
    hitter-days that ACTUALLY boomed, what fraction were in the (pooled
    across all test dates, same convention backtest_ceiling_signal uses)
    top decile by each of the three signals.

    Returns a flat metrics dict (not nested by player type - hitters
    only), or {"n": 0}/{"n": n, "n_actual_booms": 0, ...} when there isn't
    enough scored data to report anything meaningful."""
    season = season or config.SEASON_START.year
    persisted = data.load_persisted_statcast(raw_dir, season)
    if persisted is None or persisted.empty:
        return {"n": 0}

    heuristic = dfs_backtest.backtest_dfs_projections(raw_dir, season, days)
    df = heuristic["hitters"]
    if df.empty:
        return {"n": 0}

    rows = []
    thresholds = []
    for date, day_df in df.groupby("date"):
        history = compute_player_dk_points_history(persisted, as_of_date=date)["hitters"]
        threshold = compute_boom_threshold(history, percentile)
        if pd.isna(threshold):
            continue
        thresholds.append(threshold)

        boom_rate = compute_boom_rate(history, threshold)
        if boom_rate.empty:
            continue
        merged = day_df.merge(boom_rate[["key_mlbam", "Boom_Rate"]], on="key_mlbam", how="inner")
        if merged.empty:
            continue
        merged = merged.copy()
        merged["_threshold"] = threshold
        rows.append(merged)

    if not rows:
        return {"n": 0}

    combined = pd.concat(rows, ignore_index=True)
    for column in ("DK_Points_Hitter", "Matchup_Ratio", "Actual_DK_Points_Modeled", "Boom_Rate", "_threshold"):
        combined[column] = pd.to_numeric(combined[column])

    n = len(combined)
    if n < 2:
        return {"n": n}

    combined["Matchup_Boom_Score"] = compute_matchup_boom_score(combined["Boom_Rate"], combined["Matchup_Ratio"])
    combined["_actual_boomed"] = combined["Actual_DK_Points_Modeled"] >= combined["_threshold"]

    avg_threshold = float(sum(thresholds) / len(thresholds))
    actual_boomed = combined[combined["_actual_boomed"]]
    n_actual_booms = len(actual_boomed)
    if n_actual_booms == 0:
        return {"n": n, "n_actual_booms": 0, "avg_threshold": avg_threshold}

    top_fraction = 1 - percentile / 100

    def _capture_rate(column: str) -> float:
        cutoff = combined[column].quantile(1 - top_fraction)
        return float((actual_boomed[column] >= cutoff).mean())

    return {
        "n": n,
        "n_actual_booms": n_actual_booms,
        "avg_threshold": avg_threshold,
        "matchup_boom_capture_rate": _capture_rate("Matchup_Boom_Score"),
        "boom_rate_only_capture_rate": _capture_rate("Boom_Rate"),
        "mean_projection_capture_rate": _capture_rate("DK_Points_Hitter"),
    }

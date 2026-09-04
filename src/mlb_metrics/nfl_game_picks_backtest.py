"""Real, no-lookahead week-by-week replay of nfl_game_picks.py against a
real historical NFL season (`schedules_*.parquet`'s own real completed
games, scores, and - unlike MLB, which needed market_odds.py's own ESPN
scraper - real historical `home_moneyline`/`away_moneyline` odds already
persisted for every REG game). This is what actually validates every "real
starting point pending backtest" constant nfl_team_strength.py/
nfl_game_picks.py's own docstrings flag (config.py's "NFL Game
Predictions" section docstring is the single source of truth for the full
list) - reports honestly either way, same discipline this project already
applies to every MLB backtest (see README's "Real quant sanity-check"
sections).

Methodology (per the approved plan, "we're in the preseason, so use up to
week 7 from last year to build and then the rest of the year to test and
calibrate models"): TRAIN_MAX_WEEK=7 splits a real season into weeks 3-7
("train" - available for fitting/sweeping candidate constants) and weeks
8-18 ("test" - the real held-out validation). Weeks 1 AND 2 are both
excluded from every replay by default (see `replay_season`'s own default
`weeks` - real, structural, not arbitrary):
- Week 1 has zero real prior-game history within the season to build
  nfl_team_strength.py's ratings from (the same no-lookahead
  "current_strength as of a team's most recent game" design used
  throughout this pipeline), so there is no honest prediction to make.
- Week 2's history is exactly one real game per team - confirmed live
  against the real 2025 season: with only one game each, EVERY team's
  `current_strength`/`strength`/`pyth_strength` collapses to the exact
  same real 0 (no games strictly before a team's own single game to draw
  a rolling window from), so the whole real population has zero variance
  and z-normalization (mean-centered, divided by std) is a real 0/0 -
  `assemble_team_metrics` returns real, honest NaN ratings for every team
  that week, not a fabricated number. This is a genuine cold-start
  property of a single-season backtest (a live multi-season pipeline
  carries real prior-season history into early-season windows instead -
  see nfl_pipeline.py - so this specific degeneracy is a backtest-only
  artifact of deliberately NOT carrying prior-season data into this
  validation, not a live-pipeline concern).
Playoff games (WC/DIV/CON/SB) are excluded entirely (elimination-game
dynamics, rest advantages, and small sample don't fit the same
statistical treatment as the regular season - explicit non-goal in the
approved plan).

This module deliberately does NOT write anything into
data/predictions/nfl_game_predictions.csv - like the MLB live log, that
log accumulates forward from when the live pipeline ships (see
nfl_game_predictions.py), it is never backfilled from this backtest. This
module's whole job is producing the honest report that decides whether/how
the live constants get adjusted before that log starts.
"""

import pandas as pd

from mlb_metrics import config, evaluation, market_odds, nfl_game_picks, nfl_team_strength

TRAIN_MAX_WEEK = 7

REPLAY_COLUMNS = [
    "game_id", "season", "week", "home_team", "away_team",
    "home_win_probability", "home_score", "away_score", "home_won",
    "home_moneyline", "away_moneyline", "market_home_win_probability",
]


def replay_season(
    schedules_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    snap_counts_df: pd.DataFrame,
    rosters_df: pd.DataFrame,
    pbp_df: pd.DataFrame,
    season: int,
    weeks: list[int] | None = None,
) -> pd.DataFrame:
    """Returns REPLAY_COLUMNS - one row per real completed REG game across
    `weeks` (default: every real REG week from 3 through the season's max,
    see module docstring for why weeks 1 and 2 are excluded). For each replayed
    week, nfl_team_strength.assemble_team_metrics/compute_qb_continuity_adjustment
    are rebuilt from ONLY the real games/stats STRICTLY BEFORE that week
    (no lookahead - the same "current_strength as of a team's most recent
    game" contract every function here already carries), so this genuinely
    reconstructs what nfl_game_picks.compute_game_win_probabilities would
    have predicted in real time, not a full-season-hindsight number.
    `pbp_df` is `nfl_data.fetch_pbp`'s own real play-by-play output,
    feeding `compute_team_points_per_drive` the same no-lookahead way."""
    season_sched = schedules_df[schedules_df["season"] == season]
    reg = season_sched[season_sched["game_type"] == "REG"]
    reg_stats = team_stats_df[(team_stats_df["season"] == season) & (team_stats_df["season_type"] == "REG")]
    reg_snaps = snap_counts_df[(snap_counts_df["season"] == season) & (snap_counts_df["game_type"] == "REG")]
    reg_weekly = weekly_df[(weekly_df["season"] == season) & (weekly_df["season_type"] == "REG")]
    reg_pbp = pbp_df[(pbp_df["season"] == season) & (pbp_df["season_type"] == "REG")]

    if weeks is None:
        weeks = sorted(w for w in reg["week"].unique() if w > 2)

    all_rows = []
    for week in weeks:
        history_sched = reg[reg["week"] < week]
        if history_sched.empty:
            continue

        this_week_games = reg[
            (reg["week"] == week)
            & reg["home_score"].notna()
            & reg["away_score"].notna()
        ][[
            "game_id", "season", "week", "home_team", "away_team", "home_qb_id", "away_qb_id",
            "home_score", "away_score", "home_moneyline", "away_moneyline",
        ]]
        if this_week_games.empty:
            continue

        history_stats = reg_stats[reg_stats["week"] < week]
        history_snaps = reg_snaps[reg_snaps["week"] < week]
        history_weekly = reg_weekly[reg_weekly["week"] < week]
        history_pbp = reg_pbp[reg_pbp["week"] < week]

        master = nfl_team_strength.assemble_team_metrics(history_sched, history_stats, history_pbp, current_season=season)
        qb_continuity = nfl_team_strength.compute_qb_continuity_adjustment(history_snaps, history_weekly, rosters_df)

        probs = nfl_game_picks.compute_game_win_probabilities(
            master, qb_continuity, history_weekly, this_week_games
        )
        rows = probs.merge(
            this_week_games[["game_id", "home_score", "away_score", "home_moneyline", "away_moneyline"]],
            on="game_id", how="left",
        )
        all_rows.append(rows)

    if not all_rows:
        return pd.DataFrame(columns=REPLAY_COLUMNS)

    result = pd.concat(all_rows, ignore_index=True)
    result["home_won"] = (result["home_score"] > result["away_score"]).astype(float)

    home_implied = result["home_moneyline"].apply(market_odds.moneyline_to_implied_probability)
    away_implied = result["away_moneyline"].apply(market_odds.moneyline_to_implied_probability)
    result["market_home_win_probability"] = market_odds.devig(home_implied, away_implied)

    return result[REPLAY_COLUMNS]


def score_predictions(replay: pd.DataFrame, prob_col: str = "home_win_probability") -> dict:
    """Real accuracy/Brier/log-loss for `prob_col` against `home_won`,
    reusing evaluation.py's generic scoring functions directly (not
    reimplemented - the same reuse pattern game_evaluation.py already
    established for MLB) by reshaping into evaluation.py's own
    [predicted_probability, outcome_col] contract. n=0 (an empty replay,
    e.g. a season/week slate with no resolved games yet) returns real NaNs
    for every rate, not a fabricated number. A row with a real but NaN
    `prob_col` (e.g. a real degenerate zero-variance week - see module
    docstring) is likewise excluded from `n`/accuracy rather than
    silently coerced by a `>= 0.5` comparison, which would treat an
    honestly-unknown NaN prediction as a fabricated "predicted away"."""
    frame = replay[[prob_col, "home_won"]].rename(columns={prob_col: "predicted_probability"})
    resolved = evaluation.resolved_only(frame, outcome_col="home_won")
    resolved = resolved[resolved["predicted_probability"].notna()]
    n = len(resolved)
    if n == 0:
        return {
            "n": 0, "accuracy": float("nan"), "accuracy_ci_low": float("nan"), "accuracy_ci_high": float("nan"),
            "brier_score": float("nan"), "log_loss": float("nan"),
        }

    predicted_home = resolved["predicted_probability"] >= 0.5
    actual_home = resolved["home_won"] == 1.0
    n_correct = int((predicted_home == actual_home).sum())
    ci_low, ci_high = evaluation.wilson_confidence_interval(n_correct, n)

    return {
        "n": n,
        "accuracy": float(n_correct / n),
        "accuracy_ci_low": ci_low,
        "accuracy_ci_high": ci_high,
        "brier_score": evaluation.brier_score(frame, outcome_col="home_won"),
        "log_loss": evaluation.log_loss(frame, outcome_col="home_won"),
    }


def beat_closing_line_rate(replay: pd.DataFrame) -> dict:
    """The real "did this beat the market's own closing line" question -
    direct mirror of game_evaluation._beat_closing_line_rate's own
    reasoning (see that function's own docstring), simplified here since
    `replay` already carries `home_won`/`home_win_probability` directly
    (no predicted_winner/actual_winner string-matching indirection
    needed). Puts both sides on the SAME basis (probability the HOME team
    wins) and compares each side's squared error against the real
    outcome, per game - reports the fraction of real market-available,
    resolved games where the model's squared error is strictly lower
    (ties excluded from both numerator and denominator, real n reported
    separately so a rate can never hide a tiny sample), plus a real
    Wilson CI and a real binomial test p-value against a 0.5 null (a
    genuinely well-posed null here - see evaluation.binomial_significance's
    own docstring for why). Also requires a real, non-NaN
    `home_win_probability` (see score_predictions' own docstring for why
    a NaN model prediction - a real degenerate zero-variance week, not a
    missing-data gap - must never be silently compared as if it were a
    real number)."""
    scoped = replay[
        replay["market_home_win_probability"].notna()
        & replay["home_won"].notna()
        & replay["home_win_probability"].notna()
    ].copy()
    if scoped.empty:
        return {"n_compared": 0, "rate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}

    model_error = (scoped["home_win_probability"] - scoped["home_won"]) ** 2
    market_error = (scoped["market_home_win_probability"] - scoped["home_won"]) ** 2

    compared = model_error != market_error
    n_compared = int(compared.sum())
    if n_compared == 0:
        return {"n_compared": 0, "rate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}

    n_beat = int(((model_error < market_error) & compared).sum())
    ci_low, ci_high = evaluation.wilson_confidence_interval(n_beat, n_compared)
    p_value = evaluation.binomial_significance(n_beat, n_compared, null_probability=0.5)
    return {
        "n_compared": n_compared,
        "rate": float(n_beat / n_compared),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_value": p_value,
    }


def build_backtest_report(replay: pd.DataFrame) -> pd.DataFrame:
    """One row per (split, source) - train (weeks 3-TRAIN_MAX_WEEK) and
    test (weeks TRAIN_MAX_WEEK+1 onward) - both the model's own
    (uncalibrated) heuristic and the real market's own closing line, plus
    a beat_closing_line row per split. `replay` is replay_season's own
    output; splitting happens here (not inside replay_season) so a caller
    can re-slice the same real replayed data by a different week boundary
    without re-running the (comparatively expensive) week-by-week
    rebuild."""
    splits = {
        "train (wk3-%d)" % TRAIN_MAX_WEEK: replay[replay["week"] <= TRAIN_MAX_WEEK],
        "test (wk%d-18)" % (TRAIN_MAX_WEEK + 1): replay[replay["week"] > TRAIN_MAX_WEEK],
    }

    rows = []
    for split_name, split_df in splits.items():
        model_metrics = score_predictions(split_df, "home_win_probability")
        market_metrics = score_predictions(split_df, "market_home_win_probability")
        closing_line = beat_closing_line_rate(split_df)
        rows.append({"split": split_name, "source": "model", **model_metrics})
        rows.append({"split": split_name, "source": "market", **market_metrics})
        rows.append({
            "split": split_name, "source": "beat_closing_line",
            "n": closing_line["n_compared"], "accuracy": closing_line["rate"],
            "accuracy_ci_low": closing_line["ci_low"], "accuracy_ci_high": closing_line["ci_high"],
            "brier_score": float("nan"), "log_loss": float("nan"),
        })

    return pd.DataFrame(rows)


# --- Multi-season replay (real cross-season carryover validation, 2026-09-04) ---


def build_multi_season_history(
    schedules_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    snap_counts_df: pd.DataFrame,
    rosters_df: pd.DataFrame,
    pbp_df: pd.DataFrame,
    seasons: list[int],
    carryover_regression: float = None,
    carryover_prior_strength: float = None,
    season_aware: bool = False,
) -> list[dict]:
    """The expensive, no-lookahead per-week team-strength assembly a real
    cross-season replay needs - split out from `replay_multi_season` so a
    caller sweeping `composite_weights`/`home_field_weight` candidates
    (which only affect the CHEAP nfl_game_picks.compute_game_win_probabilities
    step, not team-strength assembly itself) can rebuild this ONCE per
    (carryover_regression, carryover_prior_strength) pair and re-score
    every composite/home-field candidate against the same real history
    via `score_multi_season_snapshots`, instead of re-running
    nfl_team_strength.assemble_team_metrics for every combination in a
    full grid sweep - same "don't re-run the comparatively expensive
    rebuild" reasoning `build_backtest_report`'s own docstring already
    establishes for its train/test re-slicing.

    Unlike `replay_season` (which only ever replays ONE season in
    isolation, with no real prior-season data fed in at all - it cannot
    exercise `nfl_team_strength._season_aware_blend`'s real cross-season
    carryover mechanism), this builds `history` for each season AFTER the
    first in `seasons` exactly the way `nfl_pipeline.py`'s own `run()`
    does: the immediately PRIOR season's full real REG data plus the
    current season's own real data STRICTLY BEFORE the week being
    replayed, concatenated (rosters are not week-scoped - the whole
    current season's real rosters plus last season's are always fair
    game, same as `nfl_pipeline.py`). Real weeks 1 and 2 of the current
    season ARE replayed here (unlike `replay_season`'s own week 1/2
    exclusion) - a real prior season's history is available even at week
    1, so there is a real prediction to make. The FIRST season in
    `seasons` is never replayed itself - there is no real prior season
    anywhere in the supplied data to build ITS OWN history from (the
    exact real cold-start `replay_season`'s own module docstring already
    documents, not something this function can fix by definition).

    Returns one dict per real replayed (season, week) - {master,
    qb_continuity, weekly, this_week_games}."""
    snapshots = []
    for season in seasons[1:]:
        prior_season = season - 1
        prior_sched = schedules_df[
            (schedules_df["season"] == prior_season) & (schedules_df["game_type"] == "REG")
            & schedules_df["home_score"].notna() & schedules_df["away_score"].notna()
        ]
        prior_stats = team_stats_df[(team_stats_df["season"] == prior_season) & (team_stats_df["season_type"] == "REG")]
        prior_weekly = weekly_df[(weekly_df["season"] == prior_season) & (weekly_df["season_type"] == "REG")]
        prior_snaps = snap_counts_df[(snap_counts_df["season"] == prior_season) & (snap_counts_df["game_type"] == "REG")]
        prior_pbp = pbp_df[(pbp_df["season"] == prior_season) & (pbp_df["season_type"] == "REG")]
        prior_rosters = rosters_df[rosters_df["season"] == prior_season]

        season_all = schedules_df[schedules_df["season"] == season]
        reg = season_all[season_all["game_type"] == "REG"]
        reg_stats = team_stats_df[(team_stats_df["season"] == season) & (team_stats_df["season_type"] == "REG")]
        reg_weekly = weekly_df[(weekly_df["season"] == season) & (weekly_df["season_type"] == "REG")]
        reg_snaps = snap_counts_df[(snap_counts_df["season"] == season) & (snap_counts_df["game_type"] == "REG")]
        reg_pbp = pbp_df[(pbp_df["season"] == season) & (pbp_df["season_type"] == "REG")]
        season_rosters = rosters_df[rosters_df["season"] == season]

        for week in sorted(reg["week"].unique()):
            this_week_games = reg[
                (reg["week"] == week) & reg["home_score"].notna() & reg["away_score"].notna()
            ][[
                "game_id", "season", "week", "home_team", "away_team", "home_qb_id", "away_qb_id",
                "home_score", "away_score", "home_moneyline", "away_moneyline",
            ]]
            if this_week_games.empty:
                continue

            history_sched = pd.concat([prior_sched, reg[reg["week"] < week]], ignore_index=True)
            history_stats = pd.concat([prior_stats, reg_stats[reg_stats["week"] < week]], ignore_index=True)
            history_weekly = pd.concat([prior_weekly, reg_weekly[reg_weekly["week"] < week]], ignore_index=True)
            history_snaps = pd.concat([prior_snaps, reg_snaps[reg_snaps["week"] < week]], ignore_index=True)
            history_pbp = pd.concat([prior_pbp, reg_pbp[reg_pbp["week"] < week]], ignore_index=True)
            history_rosters = pd.concat([prior_rosters, season_rosters], ignore_index=True)

            master = nfl_team_strength.assemble_team_metrics(
                history_sched, history_stats, history_pbp, current_season=season,
                carryover_regression=carryover_regression, carryover_prior_strength=carryover_prior_strength,
                season_aware=season_aware,
            )
            qb_continuity = nfl_team_strength.compute_qb_continuity_adjustment(
                history_snaps, history_weekly, history_rosters
            )
            snapshots.append({
                "master": master, "qb_continuity": qb_continuity,
                "weekly": history_weekly, "this_week_games": this_week_games,
            })
    return snapshots


def score_multi_season_snapshots(
    snapshots: list[dict], composite_weights=None, home_field_weight: float = None
) -> pd.DataFrame:
    """Cheap re-scoring step (nfl_game_picks.compute_game_win_probabilities
    only) against already-built `build_multi_season_history` snapshots -
    the real performance payoff for a composite_weights/home_field_weight
    grid sweep (see that function's own docstring). Returns REPLAY_COLUMNS,
    same real shape as `replay_season`/`replay_multi_season`."""
    if not snapshots:
        return pd.DataFrame(columns=REPLAY_COLUMNS)

    all_rows = []
    for snap in snapshots:
        probs = nfl_game_picks.compute_game_win_probabilities(
            snap["master"], snap["qb_continuity"], snap["weekly"], snap["this_week_games"],
            composite_weights=composite_weights, home_field_weight=home_field_weight,
        )
        rows = probs.merge(
            snap["this_week_games"][["game_id", "home_score", "away_score", "home_moneyline", "away_moneyline"]],
            on="game_id", how="left",
        )
        all_rows.append(rows)

    result = pd.concat(all_rows, ignore_index=True)
    result["home_won"] = (result["home_score"] > result["away_score"]).astype(float)

    home_implied = result["home_moneyline"].apply(market_odds.moneyline_to_implied_probability)
    away_implied = result["away_moneyline"].apply(market_odds.moneyline_to_implied_probability)
    result["market_home_win_probability"] = market_odds.devig(home_implied, away_implied)

    return result[REPLAY_COLUMNS]


def replay_multi_season(
    schedules_df: pd.DataFrame,
    team_stats_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    snap_counts_df: pd.DataFrame,
    rosters_df: pd.DataFrame,
    pbp_df: pd.DataFrame,
    seasons: list[int],
    composite_weights=None,
    home_field_weight: float = None,
    carryover_regression: float = None,
    carryover_prior_strength: float = None,
    season_aware: bool = False,
) -> pd.DataFrame:
    """Convenience one-call wrapper: `build_multi_season_history` +
    `score_multi_season_snapshots` (see both functions' own docstrings for
    the full reasoning). Use the two-step form directly when sweeping
    several `composite_weights`/`home_field_weight` candidates against the
    SAME `carryover_regression`/`carryover_prior_strength` pair, to avoid
    rebuilding team-strength assembly redundantly."""
    snapshots = build_multi_season_history(
        schedules_df, team_stats_df, weekly_df, snap_counts_df, rosters_df, pbp_df, seasons,
        carryover_regression, carryover_prior_strength, season_aware,
    )
    return score_multi_season_snapshots(snapshots, composite_weights, home_field_weight)


# --- Real feature+outcome training log (2026-09-04 - fixing the ML win-probability ceiling) ---


def assemble_nfl_game_pick_log(snapshots: list[dict], feature_fn=nfl_game_picks.build_game_features) -> pd.DataFrame:
    """Real feature+outcome training log for
    scripts/train_nfl_game_pick_model.py, built from
    `build_multi_season_history`'s own already-built snapshots (real,
    no-lookahead `master`/`qb_continuity`/`weekly`/`this_week_games` per
    replayed week, spanning every real cached season) - zero new data
    fetching, zero new team-strength code, purely wiring together what
    the season-carryover work already built.

    `feature_fn` defaults to `nfl_game_picks.build_game_features` (the
    minimal, MLB-precedented candidate - home/away composite + QB
    adjustments); pass `nfl_game_picks.build_game_features_disaggregated`
    for the richer candidate (see that function's own docstring). Returns
    one row per real snapshot game with that feature set's own columns
    plus a real `home_won` outcome column, derived from `this_week_games`'s
    real `home_score`/`away_score` - a genuine completed game, never an
    upcoming/unplayed one (every `build_multi_season_history` snapshot's
    `this_week_games` is already filtered to real completed scores)."""
    rows = []
    for snap in snapshots:
        features = feature_fn(snap["master"], snap["qb_continuity"], snap["weekly"], snap["this_week_games"])
        outcomes = snap["this_week_games"][["game_id", "home_score", "away_score"]]
        merged = features.merge(outcomes, on="game_id", how="left")
        merged["home_won"] = (merged["home_score"] > merged["away_score"]).astype(float)
        rows.append(merged.drop(columns=["home_score", "away_score"]))

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)

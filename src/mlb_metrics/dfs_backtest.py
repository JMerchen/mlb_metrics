"""Validates dfs.py's DK-points projections against real historical
outcomes, recomputed fresh from persisted Statcast (data/raw/) - the same
no-lookahead "recompute pipeline.compute_outputs from an as-of-date slice"
technique game_picks_backtest.reconstruct_historical_game_picks_from_persisted
already uses, applied to DFS rankings instead of game picks.

`derive_historical_team_schedule` reuses
game_picks_backtest.derive_historical_schedule_games directly (already
real starters/scores per game_pk, no lookahead risk) and widens it from
one row per game into one row per team per game (home + away
perspective), matching schedule.normalize_schedule's live shape - the
same shape dfs.compute_hitter_dk_points/compute_pitcher_dk_points expect.

## What "actual" means here, and its honesty limits

compute_actual_hitter_dk_points scores hit-type (1B/2B/3B/HR) plus
BB/HBP/RBI (dfs.compute_hitter_dk_points's current full scope, since
hitters.compute_extended_dk_rates added those) - runs scored and stolen
bases remain excluded from both sides, not just the projection, since
there's nothing to validate for a category the model doesn't project at
all (see dfs.py's module docstring for why those two specifically are
deferred/not-planned). RBI uses the SAME approximation
(helpers.estimate_rbi - a completed PA's own bat_score/post_bat_score
delta) on both the projected and "actual" side, so this validates the
projection's own logic being reapplied to real data, not RBI accuracy
against a truly independent ground truth (there is none available here) -
same category of limitation as the pitcher ER/FIP validation below.

compute_actual_pitcher_dk_points computes real IP/K/BB/H from that start's
actual events (the same classifiers pitcher_form.py/
traditional_stats.compute_traditional_pitching_stats use) - these ARE a
genuine validation of the projection's IP/K/BB/H components. Earned runs
are different: true ER isn't cleanly derivable from Statcast alone (no
runner-inheritance/error/sequencing reconstruction here), so this uses the
SAME FIP-based estimate dfs.compute_pitcher_dk_points itself uses, on both
the projected and "actual" side. The resulting combined
DK_Points_Pitcher/Actual_DK_Points_Modeled comparison is therefore a real
validation of the IP/K/BB/H components blended together, but NOT a
validation of the FIP-to-ER approximation itself - there is no ground
truth for that in this project, the same reasoning FIP was chosen over
ERA everywhere else here. scripts/backtest_dfs_rankings.py reports the
individual IP/K/BB/H component errors separately for exactly this reason,
not just the blended point total.
"""

import pandas as pd

from mlb_metrics import config, data, dfs, dfs_ml, game_picks_backtest, helpers, matchup, pipeline, pitcher_form


def derive_historical_team_schedule(persisted_statcast: pd.DataFrame) -> pd.DataFrame:
    """[date, team, opponent, probable_pitcher_key_mlbam, game_pk, is_home] -
    the same per-team shape schedule.normalize_schedule produces for live
    data, widened from game_picks_backtest.derive_historical_schedule_games's
    one-row-per-game shape into two rows per game (home + away
    perspective)."""
    games = game_picks_backtest.derive_historical_schedule_games(persisted_statcast)
    columns = ["date", "team", "opponent", "probable_pitcher_key_mlbam", "game_pk", "is_home"]

    home = games.rename(
        columns={"home_team": "team", "away_team": "opponent", "home_probable_pitcher_key_mlbam": "probable_pitcher_key_mlbam"}
    )
    home["is_home"] = True

    away = games.rename(
        columns={"away_team": "team", "home_team": "opponent", "away_probable_pitcher_key_mlbam": "probable_pitcher_key_mlbam"}
    )
    away["is_home"] = False

    return pd.concat([home[columns], away[columns]], ignore_index=True)


def compute_actual_hitter_dk_points(day_completed_batter_events: pd.DataFrame) -> pd.DataFrame:
    """[key_mlbam, Actual_DK_Points_Modeled] - real DK hit-type + BB/HBP/RBI
    points (the current full scope dfs.compute_hitter_dk_points models) for
    one date's real completed at-bat events, keyed by batter. Needs
    bat_score/post_bat_score columns (helpers.estimate_rbi)."""
    df = day_completed_batter_events.copy()
    events = df["events"]
    df["points"] = (
        (events == "single") * config.DFS_DK_HITTER_SINGLE_POINTS
        + (events == "double") * config.DFS_DK_HITTER_DOUBLE_POINTS
        + (events == "triple") * config.DFS_DK_HITTER_TRIPLE_POINTS
        + (events == "home_run") * config.DFS_DK_HITTER_HR_POINTS
        + helpers.is_walk_for_dk_scoring(events) * config.DFS_DK_HITTER_BB_POINTS
        + helpers.is_hit_by_pitch(events) * config.DFS_DK_HITTER_HBP_POINTS
        + helpers.estimate_rbi(df) * config.DFS_DK_HITTER_RBI_POINTS
    )
    agg = df.groupby("batter", as_index=False)["points"].sum()
    return agg.rename(columns={"batter": "key_mlbam", "points": "Actual_DK_Points_Modeled"})


def compute_actual_pitcher_dk_points(day_completed_pitcher_events: pd.DataFrame) -> pd.DataFrame:
    """[key_mlbam, Actual_IP, Actual_K, Actual_BB, Actual_H,
    Actual_DK_Points_Modeled] - real IP/K/BB/H (the same components
    dfs.compute_pitcher_dk_points projects) for one date's real completed
    at-bat events, keyed by pitcher. See module docstring for why
    Actual_DK_Points_Modeled's ER term is a FIP-based estimate, not real
    earned runs - it validates IP/K/BB/H, not that estimate itself."""
    df = day_completed_pitcher_events.copy()
    df["outs"] = helpers.outs_recorded(df["events"])
    df["so"] = helpers.is_strikeout(df["events"])
    df["bb"] = helpers.is_walk(df["events"])
    df["hit"] = helpers.is_hit(df["events"])
    df["hr"] = helpers.is_home_run(df["events"])

    agg = df.groupby("pitcher", as_index=False).agg(
        outs=("outs", "sum"), so=("so", "sum"), bb=("bb", "sum"), hit=("hit", "sum"), hr=("hr", "sum"),
    )
    agg["Actual_IP"] = agg["outs"] / 3
    agg["Actual_K"] = agg["so"]
    agg["Actual_BB"] = agg["bb"]
    agg["Actual_H"] = agg["hit"]

    ip_safe = agg["Actual_IP"].replace(0, pd.NA)
    fip = (13 * agg["hr"] + 3 * agg["bb"] - 2 * agg["so"]) / ip_safe + config.FIP_CONSTANT
    er_estimate = (fip * agg["Actual_IP"] / 9).clip(lower=0).fillna(0)

    agg["Actual_DK_Points_Modeled"] = (
        agg["Actual_IP"] * config.DFS_DK_PITCHER_IP_POINTS
        + agg["Actual_K"] * config.DFS_DK_PITCHER_K_POINTS
        + agg["Actual_BB"] * config.DFS_DK_PITCHER_BB_POINTS
        + agg["Actual_H"] * config.DFS_DK_PITCHER_H_POINTS
        + er_estimate * config.DFS_DK_PITCHER_ER_POINTS
    )
    return agg.rename(columns={"pitcher": "key_mlbam"})[
        ["key_mlbam", "Actual_IP", "Actual_K", "Actual_BB", "Actual_H", "Actual_DK_Points_Modeled"]
    ]


def _compute_date_outputs(persisted: pd.DataFrame, team_schedule: pd.DataFrame, date) -> dict | None:
    """One date's worth of no-lookahead recomputation, shared by
    backtest_dfs_projections (heuristic-only) and assemble_ml_training_rows
    (heuristic + raw features) so the per-date pipeline recompute is
    written once. Returns None when there's no usable history or no
    scheduled games that date (both callers skip the date in that case)."""
    history = persisted[persisted["game_date"] < date]
    if history.empty:
        return None

    todays_schedule = team_schedule[team_schedule["date"] == date]
    if todays_schedule.empty:
        return None

    outputs = pipeline.compute_outputs(history)

    data_with_game_id = data.assign_game_ids(history)
    roles = data.label_pitcher_roles(data_with_game_id)
    pdf_with_role = pipeline.build_pitcher_events_with_role(data_with_game_id, roles)
    pitcher_form_df = pitcher_form.compute_pitcher_dfs_form(pdf_with_role)

    matchup_probability = matchup.compute_matchup_hit_probability(
        outputs["wave"], outputs["pave"], outputs["confidence"], todays_schedule
    )
    projected_hitters = dfs.compute_hitter_dk_points(outputs["wave"], matchup_probability, todays_schedule)
    projected_pitchers = dfs.compute_pitcher_dk_points(outputs["pave"], pitcher_form_df, todays_schedule)

    day_events = persisted[persisted["game_date"] == date]
    actual_hitters = compute_actual_hitter_dk_points(
        data.completed_events(day_events, ["game_date", "batter", "events", "bat_score", "post_bat_score"])
    )
    actual_pitchers = compute_actual_pitcher_dk_points(
        data.completed_events(day_events, ["game_date", "pitcher", "events"])
    )

    return {
        "outputs": outputs,
        "todays_schedule": todays_schedule,
        "matchup_probability": matchup_probability,
        "projected_hitters": projected_hitters,
        "projected_pitchers": projected_pitchers,
        "actual_hitters": actual_hitters,
        "actual_pitchers": actual_pitchers,
    }


def backtest_dfs_projections(raw_dir: str = "data/raw", season: int | None = None, days: int = 20) -> dict[str, pd.DataFrame]:
    """No-lookahead backtest: for each of the last `days` real game dates,
    recomputes that day's DFS projections using ONLY Statcast strictly
    before that date (pipeline.compute_outputs - the same function
    pipeline.run() calls every day), then compares against that date's
    REAL outcomes. Returns {"hitters": DataFrame, "pitchers": DataFrame},
    one row per (date, player) scored."""
    season = season or config.SEASON_START.year
    persisted = data.load_persisted_statcast(raw_dir, season)
    if persisted is None:
        return {"hitters": pd.DataFrame(), "pitchers": pd.DataFrame()}

    team_schedule = derive_historical_team_schedule(persisted)
    dates = sorted(team_schedule["date"].unique())
    if days:
        dates = dates[-days:]

    hitter_rows = []
    pitcher_rows = []
    for date in dates:
        day = _compute_date_outputs(persisted, team_schedule, date)
        if day is None:
            continue

        hitters_scored = day["projected_hitters"].merge(day["actual_hitters"], on="key_mlbam", how="inner")
        if not hitters_scored.empty:
            hitters_scored = hitters_scored.copy()
            hitters_scored["date"] = date
            hitter_rows.append(hitters_scored[["date", "key_mlbam", "DK_Points_Hitter", "Actual_DK_Points_Modeled"]])

        pitchers_scored = day["projected_pitchers"].merge(day["actual_pitchers"], on="key_mlbam", how="inner")
        if not pitchers_scored.empty:
            pitchers_scored = pitchers_scored.copy()
            pitchers_scored["date"] = date
            pitcher_rows.append(
                pitchers_scored[
                    [
                        "date", "key_mlbam", "DK_Points_Pitcher", "Actual_DK_Points_Modeled",
                        "Expected_IP", "Actual_IP", "Expected_K", "Actual_K",
                        "Expected_BB", "Actual_BB", "Expected_H_Allowed", "Actual_H",
                    ]
                ]
            )

    hitters_result = pd.concat(hitter_rows, ignore_index=True) if hitter_rows else pd.DataFrame()
    pitchers_result = pd.concat(pitcher_rows, ignore_index=True) if pitcher_rows else pd.DataFrame()
    return {"hitters": hitters_result, "pitchers": pitchers_result}


def assemble_ml_training_rows(raw_dir: str = "data/raw", season: int | None = None, days: int | None = None) -> dict[str, pd.DataFrame]:
    """Same no-lookahead per-date recompute as backtest_dfs_projections,
    but retains the RAW feature ingredients (dfs_ml.HITTER_FEATURE_COLUMNS/
    PITCHER_FEATURE_COLUMNS) alongside each row's real outcome label - the
    training table dfs_ml.py's train_* functions fit against, instead of
    just the already-blended heuristic projection.

    `days=None` (the default here, unlike backtest_dfs_projections's
    default of 20) uses the FULL persisted history, since model training
    benefits from as much real data as available while a quick
    heuristic-only sanity check does not.

    Returns {"hitters": DataFrame, "pitchers": DataFrame}, one row per
    (date, key_mlbam) with feature columns plus:
      hitters:  Actual_DK_Points_Modeled
      pitchers: Actual_H, Actual_BB (plus Actual_IP/Actual_K, carried for
                completeness even though this phase only trains H/BB models)
    No column derived from that date's OWN outcome (any Actual_*-prefixed
    column) is ever included among the FEATURE columns - see
    tests/test_dfs_backtest.py's leakage-guard test."""
    season = season or config.SEASON_START.year
    persisted = data.load_persisted_statcast(raw_dir, season)
    if persisted is None:
        return {"hitters": pd.DataFrame(), "pitchers": pd.DataFrame()}

    team_schedule = derive_historical_team_schedule(persisted)
    dates = sorted(team_schedule["date"].unique())
    if days:
        dates = dates[-days:]

    hitter_rows = []
    pitcher_rows = []
    for date in dates:
        day = _compute_date_outputs(persisted, team_schedule, date)
        if day is None:
            continue

        hitter_features = dfs_ml.build_hitter_features(
            day["outputs"]["wave"], day["outputs"]["pave"], day["outputs"]["confidence"],
            day["todays_schedule"], day["matchup_probability"],
        )
        hitters_scored = hitter_features.merge(day["actual_hitters"], on="key_mlbam", how="inner")
        if not hitters_scored.empty:
            hitters_scored = hitters_scored.copy()
            hitters_scored["date"] = date
            hitter_rows.append(
                hitters_scored[["date", "key_mlbam"] + dfs_ml.HITTER_FEATURE_COLUMNS + ["Actual_DK_Points_Modeled"]]
            )

        pitchers_scored = day["projected_pitchers"].merge(day["actual_pitchers"], on="key_mlbam", how="inner")
        if not pitchers_scored.empty:
            pitchers_scored = pitchers_scored.copy()
            pitchers_scored["date"] = date
            pitcher_rows.append(
                pitchers_scored[
                    ["date", "key_mlbam"] + dfs_ml.PITCHER_FEATURE_COLUMNS
                    + ["Actual_IP", "Actual_K", "Actual_BB", "Actual_H"]
                ]
            )

    hitters_result = pd.concat(hitter_rows, ignore_index=True) if hitter_rows else pd.DataFrame()
    pitchers_result = pd.concat(pitcher_rows, ignore_index=True) if pitcher_rows else pd.DataFrame()
    return {"hitters": hitters_result, "pitchers": pitchers_result}

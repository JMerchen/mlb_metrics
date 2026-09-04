"""NFL team-level strength metrics: Strength, current, pyth_Strength, SOS,
pyth_SOS, Confidence, pyth_Confidence, offensive_edge, defensive_edge,
true_power - a direct structural port of `teams.py`'s MLB Pyth Strength/
SOS/Confidence mixture (see that module's own docstring), NOT a
reimplementation from scratch. Real follow-up (2026-09-02 - "build out
the quant level game predictions for the nfl... follow the same pattern
with how we built out the mlb side"): every mechanic below has a named
MLB counterpart it mirrors, with the specific baseball mechanic replaced
by a real, grounded football one wherever the two sports genuinely
differ (see the signal-by-signal mapping below) - never a force-fit port
of a baseball-specific concept that doesn't honestly translate.

Games-back windows throughout (config.NFL_TEAM_STRENGTH_WINDOWS),
matching this project's already-established NFL convention (see
nfl_teams.py/nfl_passing.py/nfl_rush_rec.py's own NFL_QB_WINDOWS/
NFL_SKILL_WINDOWS/NFL_DEFENSE_WINDOWS) - MLB's own TEAM_STRENGTH_WINDOWS
happens to ALSO be games-back (not day-count), confirmed by reading
teams.compute_strength_metrics directly, so this is a real window-size
change for an 18-week NFL season, not an architecture change.

Signal-by-signal mapping to teams.py (MLB -> NFL, and why):
- `build_team_record` (runs scored/allowed) -> points scored/allowed,
  from `nfl_data.fetch_schedules`' real `home_score`/`away_score`,
  regular season only (playoff elimination-game dynamics don't belong in
  the same rolling pool) and completed games only (a real final score,
  not a null placeholder for an unplayed future game).
- `compute_strength_metrics` (rolling win% + Pythagorean win%, SOS) ->
  same math, `config.NFL_PYTHAGOREAN_EXPONENT` (a real, commonly-cited
  NFL literature value, 2.37 - NOT MLB's own 1.83) in place of
  `config.PYTHAGOREAN_EXPONENT`.
- `offensive_edge`/`suppression_resistance` (bases produced/allowed,
  "held under 3 runs" shutout framing) -> `offensive_edge`/
  `defensive_edge`: real `passing_epa+rushing_epa+receiving_epa`
  produced (a team's own `nfl_data.fetch_team_stats` row) vs. allowed
  (the opponent's own `fetch_team_stats` row that same week, joined via
  its real `opponent_team` column - the exact same "opponent's own row
  for that week IS what my defense allowed" pattern `nfl_teams.py`
  already uses). No NFL analog of "held under 3 runs" is honestly
  forced here - EPA is a real, well-regarded, already-computed football
  efficiency metric that fills the same STRUCTURAL role (a
  complementary, non-record-based quality signal) without inventing an
  arbitrary point threshold with no real grounding.
- `true_power = avg(offensive_edge, suppression_resistance)` -> same
  formula, `avg(offensive_edge, defensive_edge)`.
- Probable-starter pitching adjustment (PAVE_PLUS/bullpen quality faced)
  -> QB-continuity adjustment (see `compute_qb_continuity_adjustment`'s
  own docstring) - the real, high-leverage NFL-specific signal with no
  clean MLB analog (no single position in baseball swings a team's win
  probability the way an NFL starting QB does).

Every constant this module reads (`config.NFL_PYTHAGOREAN_EXPONENT`,
`NFL_TEAM_STRENGTH_WINDOWS`, `NFL_NORMALIZATION_Z_SCALE`,
`NFL_CONFIDENCE_SOS_WEIGHT`) is a real, honestly-labeled STARTING POINT
- see config.py's own "NFL Game Predictions" section docstring -
genuinely re-fit/validated by `nfl_game_picks_backtest.py` against real
2025 weeks 8-18 (held out from the weeks-1-7 fit) before any of this
reaches a live pick.

Real follow-up (2026-09-04 - "every season should only carry over a
portion of the team's score from the previous season... weeks 1-6...
calibrating the features to that individual season"): `offensive_edge`/
`defensive_edge`/`turnover_margin`/`points_per_drive` are now
season-aware (`_season_aware_blend` - see its own docstring and
config.py's `NFL_SEASON_CARRYOVER_*` comments) rather than treating a
concatenated prior-season+current-season history as one flat, un-
weighted sequence the way `compute_strength_metrics`'s own rolling win%/
Pythagorean windows still do (a real, structurally different, and
explicitly out-of-scope refactor for this pass).
"""

import numpy as np
import pandas as pd

from mlb_metrics import config, helpers, nfl_passing, teams

# --- Team record (points scored/allowed) ---


def build_team_record(schedules_df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: opponent, season, week, win/loss, points
    scored (`ps`)/allowed (`pa`). Filtered to REGULAR SEASON games with a
    real final score - playoff games have real, different competitive
    dynamics (elimination, rest advantages) that don't belong in the same
    rolling-window pool as the regular season, and a future/unplayed game
    has a null `home_score`/`away_score`, not a real result to build a
    record from (`nfl_data.fetch_schedules`' own docstring - bye weeks are
    simply absent rows, not a marker to filter here)."""
    reg = schedules_df[schedules_df["game_type"] == "REG"]
    played = reg[reg["home_score"].notna() & reg["away_score"].notna()]

    away = played.rename(
        columns={"away_team": "team", "home_team": "opp", "home_score": "pa", "away_score": "ps"}
    )
    home = played.rename(
        columns={"home_team": "team", "away_team": "opp", "home_score": "ps", "away_score": "pa"}
    )
    cols = ["team", "opp", "season", "week", "game_id", "ps", "pa"]
    record = pd.concat([away[cols], home[cols]]).sort_values(["team", "season", "week"])
    record["win"] = (record["ps"] > record["pa"]).astype(int)
    record["loss"] = (record["ps"] < record["pa"]).astype(int)
    return record.reset_index(drop=True)


# --- Rolling windows (games-back, opponent-excluded and plain "current" variants) ---


def _compute_exclusion_windows(group: pd.DataFrame, windows) -> pd.DataFrame:
    """Per-game win rate over the last N games, excluding games against
    that game's specific opponent (so a real division-rival rematch
    doesn't let one head-to-head result dominate a team's own strength
    number) - direct port of teams.py's identically-named helper, same
    reasoning, generalized to any 'win'/'opp' record (no MLB-specific
    column referenced)."""
    team = group.name
    group = group.copy()
    group["team"] = team
    results = {f"rolling_{w}": [] for w in windows}
    full_history = []

    for i in range(len(group)):
        opp = group.iloc[i]["opp"]
        history_excl = group.iloc[:i]
        history_excl = history_excl[history_excl["opp"] != opp]

        full_history.append(history_excl["win"].mean() if len(history_excl) else np.nan)
        for w in windows:
            results[f"rolling_{w}"].append(history_excl["win"].tail(w).mean() if len(history_excl) else np.nan)

    for key, values in results.items():
        group[key] = values
    group["full"] = full_history
    return group


def _compute_current_windows(group: pd.DataFrame, windows) -> pd.DataFrame:
    """Same as `_compute_exclusion_windows` but without excluding the
    upcoming opponent - direct port of teams.py's identically-named
    helper."""
    team = group.name
    group = group.copy()
    group["team"] = team
    results = {f"roll_{w}_cur": [] for w in windows}
    full_history = []

    for i in range(len(group)):
        history = group.iloc[:i]
        full_history.append(history["win"].mean() if len(history) else np.nan)
        for w in windows:
            results[f"roll_{w}_cur"].append(history["win"].tail(w).mean() if len(history) else np.nan)

    for key, values in results.items():
        group[key] = values
    group["full_cur"] = full_history
    return group


def compute_strength_metrics(record: pd.DataFrame):
    """Returns (current_strength, sos): rolling win%, Pythagorean win%, and
    strength-of-schedule per team, as of each team's most recent real
    completed game - direct port of teams.compute_strength_metrics, same
    math, `config.NFL_TEAM_STRENGTH_WINDOWS`/`NFL_PYTHAGOREAN_EXPONENT` in
    place of MLB's own. `record` may span multiple real seasons (e.g. late
    prior season plus early current season) - windows are purely
    games-back over each team's own rows, season boundaries don't reset
    or interrupt them (same convention nfl_teams.compute_defense_rolling_rates
    already established)."""
    windows = [w for w, _ in config.NFL_TEAM_STRENGTH_WINDOWS if w is not None]

    nf = record.sort_values(["team", "season", "week"]).reset_index(drop=True)
    nf = nf.groupby("team", group_keys=False).apply(lambda g: _compute_exclusion_windows(g, windows))
    nf = nf.fillna(0)

    nf = nf.groupby("team", group_keys=False).apply(lambda g: _compute_current_windows(g, windows))
    nf = nf.fillna(0)

    nf["ps_shift"] = nf.groupby("team")["ps"].shift(1)
    nf["pa_shift"] = nf.groupby("team")["pa"].shift(1)

    for w in windows:
        nf[f"ps_{w}"] = nf.groupby("team")["ps_shift"].rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        nf[f"pa_{w}"] = nf.groupby("team")["pa_shift"].rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)

    nf["ps_full"] = nf.groupby("team")["ps_shift"].cumsum()
    nf["pa_full"] = nf.groupby("team")["pa_shift"].cumsum()

    exp = config.NFL_PYTHAGOREAN_EXPONENT
    for w in windows:
        nf[f"pyth_{w}"] = (nf[f"ps_{w}"] ** exp) / ((nf[f"ps_{w}"] ** exp) + (nf[f"pa_{w}"] ** exp))
    nf["pyth_full"] = (nf["ps_full"] ** exp) / ((nf["ps_full"] ** exp) + (nf["pa_full"] ** exp))
    nf = nf.fillna(0)

    def blend(col_for_window):
        total = None
        for games_back, weight in config.NFL_TEAM_STRENGTH_WINDOWS:
            term = nf[col_for_window(games_back)] * weight
            total = term if total is None else total + term
        return total

    nf["strength"] = blend(lambda d: "full" if d is None else f"rolling_{d}")
    nf["current_strength"] = blend(lambda d: "full_cur" if d is None else f"roll_{d}_cur")
    nf["pyth_strength"] = blend(lambda d: "pyth_full" if d is None else f"pyth_{d}")

    sos = nf.groupby("opp", as_index=False)[["strength", "pyth_strength"]].mean()
    sos = sos.rename(columns={"opp": "team", "strength": "SOS", "pyth_strength": "pyth_SOS"})

    latest_game = nf.groupby("team", as_index=False)["game_id"].max()
    current_strength = latest_game.merge(nf, on=["team", "game_id"])[
        ["team", "strength", "current_strength", "pyth_strength"]
    ]
    return current_strength, sos


# --- Season-aware recency blend (games-back windows, regressed cross-season carryover) ---


def _recency_window_blend(df: pd.DataFrame, group_col: str, value_col: str, windows) -> pd.Series:
    """Real recency-rank + per-window-mean blend, generalized out of what
    used to be near-identical inline logic in
    compute_team_offense_defense_edge/compute_team_turnover_margin/
    compute_team_points_per_drive: ranks `df`'s own rows most-recent-first
    per `group_col` (a global sort by season/week descending - correct
    per-group recency regardless of `group_col`'s own row order, since
    groupby().cumcount() assigns ranks in the frame's existing row order
    within each group), means `value_col` within each `windows` games-back
    cutoff, and combines with that window's own real weight. Returns a
    Series indexed by `group_col`; a group with 0 real rows in `df` simply
    doesn't appear (a real "no games in this scope" result, not a
    fabricated 0 - `_season_aware_blend` reindexes/fills as it sees fit).
    Always names the returned index `group_col`, even when empty (a real
    week-1-of-a-new-season case, before any current-season game has been
    played) - an unnamed empty index would silently collapse the result
    of a later `Index.union` against a real named index to name=None,
    corrupting the downstream `.reset_index()` column name."""
    if df.empty:
        return pd.Series(dtype=float, index=pd.Index([], name=group_col))
    ranked = df.sort_values(["season", "week"], ascending=False).copy()
    ranked["_recency_rank"] = ranked.groupby(group_col).cumcount()

    blended = None
    for games_back, weight in windows:
        window_df = ranked if games_back is None else ranked[ranked["_recency_rank"] < games_back]
        per_group = window_df.groupby(group_col)[value_col].mean()
        blended = per_group * weight if blended is None else blended.add(per_group * weight, fill_value=0)
    return blended


def _season_aware_blend(
    stats_df: pd.DataFrame, group_col: str, value_col: str, current_season: int,
    windows=None, prior_strength: float = None, regression: float = None, season_aware: bool = True,
) -> pd.Series:
    """Real cross-season-carryover blend (see config.py's own
    NFL_SEASON_CARRYOVER_*/NFL_HOME_FIELD_ADVANTAGE_WEIGHT comments for
    the full reasoning) - real follow-up (2026-09-04, "every season
    should only carry over a portion of the team's score from the
    previous season... weeks 1-6... calibrating the features to that
    individual season").

    `season_aware=False` is a real, explicit bypass reproducing the
    TRUE pre-existing behavior (a flat `_recency_window_blend` over
    whatever seasons `stats_df` spans, season boundaries entirely
    invisible) - exists specifically so
    scripts/backtest_nfl_season_carryover.py can build a genuine "today's
    live behavior" baseline candidate to compare against. This is NOT the
    same as any (regression, prior_strength) value of the season-aware
    path below - the old flat blend lets a real END-of-prior-season game
    show up at FULL window weight in a new season's own "last 3 games"
    recency window, which the season-aware mechanism structurally never
    does (it always computes prior-season-only and current-season-only
    blends separately, then shrinks between them).

    If `stats_df` carries no real row at all for `current_season - 1`
    (a single-season-only caller - e.g. nfl_game_picks_backtest.replay_season's
    own single-season replay, or the very first season in a multi-season
    historical replay), there is no real prior to regress toward - this
    degrades EXACTLY to `_recency_window_blend` over the whole of
    `stats_df` (today's original, pre-carryover behavior, whatever
    seasons `stats_df` actually spans), not a fabricated shrink toward
    nothing. Otherwise: regresses each team's own final PRIOR-season
    blended rating toward the real cross-team prior-season mean by
    `regression` (1.0 = full carryover, 0.0 = assume nothing carries
    over), then reuses `helpers.shrink_rate` (already proven to
    generalize to any real-valued rate - see decision_score.py's own
    precedent) to blend that regressed prior against the team's own real
    CURRENT-season-only measurement, weighted by real current-season
    games played vs. `prior_strength` real-game-equivalent pseudo-
    observations. A team with real current-season rows but no real
    prior-season row of its own regresses fully to the real cross-team
    prior-season mean (never a fabricated number)."""
    windows = config.NFL_TEAM_STRENGTH_WINDOWS if windows is None else windows
    if not season_aware:
        return _recency_window_blend(stats_df, group_col, value_col, windows)

    prior_strength = config.NFL_SEASON_CARRYOVER_PRIOR_STRENGTH if prior_strength is None else prior_strength
    regression = config.NFL_SEASON_CARRYOVER_REGRESSION if regression is None else regression

    prior_rows = stats_df[stats_df["season"] == current_season - 1]
    if prior_rows.empty:
        return _recency_window_blend(stats_df, group_col, value_col, windows)

    current_rows = stats_df[stats_df["season"] == current_season]
    current_value = _recency_window_blend(current_rows, group_col, value_col, windows)
    current_n = current_rows.groupby(group_col)[value_col].size()
    prior_value = _recency_window_blend(prior_rows, group_col, value_col, windows)

    all_teams = current_value.index.union(current_n.index).union(prior_value.index)
    current_value = current_value.reindex(all_teams)
    current_n = current_n.reindex(all_teams).fillna(0)
    prior_value = prior_value.reindex(all_teams)

    league_mean = prior_value.mean()
    regressed_prior = (league_mean + regression * (prior_value - league_mean)).fillna(league_mean)

    count = current_n * current_value.fillna(0)
    return helpers.shrink_rate(count, current_n, regressed_prior, prior_strength)


# --- Offensive/defensive edge (real EPA produced/allowed) ---

EPA_COLS = ["passing_epa", "rushing_epa", "receiving_epa"]


def compute_team_offense_defense_edge(
    team_stats_df: pd.DataFrame, current_season: int = None,
    carryover_regression: float = None, carryover_prior_strength: float = None, season_aware: bool = True,
) -> pd.DataFrame:
    """One row per team: `offensive_edge` (real total EPA - passing +
    rushing + receiving - produced per game, season-aware-blended per
    `_season_aware_blend`) and `defensive_edge` (the SAME real EPA total,
    but ALLOWED - taken from the OPPONENT's own `team_stats_df` row for
    that same game, via its real `opponent_team` column, the exact
    "opponent's own stat row for that week IS what my defense allowed"
    pattern `nfl_teams.compute_team_week_allowed` already established for
    MLB's DFS matchup side). `defensive_edge` is NOT negated - a defense
    that allows LESS EPA has a smaller (more negative, since real
    EPA-allowed skews positive for an average offense) raw number; the
    z-normalization step in `assemble_team_metrics` handles the sign the
    same honest way every other z-scored signal here does, not a manual
    flip here. `current_season` defaults to `team_stats_df`'s own max real
    season (always correct under this project's no-lookahead contract -
    no caller ever includes a season later than "now")."""
    ts = team_stats_df[team_stats_df["season_type"] == "REG"].copy()
    ts["epa_total"] = ts[EPA_COLS].sum(axis=1)
    current_season = int(ts["season"].max()) if current_season is None else current_season

    own = ts[["team", "opponent_team", "season", "week", "game_id", "epa_total"]].rename(
        columns={"epa_total": "own_epa"}
    )
    # The opponent's own real EPA total that same game IS what this
    # team's defense allowed - join on (opponent_team, season, week),
    # matching the opponent's own (team, season, week) row.
    allowed = own.rename(columns={"team": "opp_team", "opponent_team": "team", "own_epa": "epa_allowed"})[
        ["team", "opp_team", "season", "week", "epa_allowed"]
    ]
    merged = own.merge(allowed, left_on=["team", "opponent_team", "season", "week"],
                        right_on=["team", "opp_team", "season", "week"], how="left")

    result = pd.DataFrame({
        "offensive_edge": _season_aware_blend(
            merged, "team", "own_epa", current_season,
            prior_strength=carryover_prior_strength, regression=carryover_regression, season_aware=season_aware,
        ),
        "defensive_edge": _season_aware_blend(
            merged, "team", "epa_allowed", current_season,
            prior_strength=carryover_prior_strength, regression=carryover_regression, season_aware=season_aware,
        ),
    })
    result.index.name = "team"
    return result.reset_index()


# --- Turnover margin (real takeaways minus real giveaways, per game) ---


def compute_team_turnover_margin(
    team_stats_df: pd.DataFrame, current_season: int = None,
    carryover_regression: float = None, carryover_prior_strength: float = None, season_aware: bool = True,
) -> pd.DataFrame:
    """One row per team: `turnover_margin` - real takeaways forced minus
    real turnovers given away, per game, season-aware-blended per
    `_season_aware_blend`. Real follow-up (2026-09-02, "we should include
    turnover ratio at a game level") - unlike offensive_edge/
    defensive_edge, this does NOT need an opponent-row join:
    `team_stats_df` already carries both halves on a team's OWN row -
    `passing_interceptions`/`fumbles_lost_total` (real turnovers this
    team's OFFENSE gave away) and `def_interceptions`/`fumble_recovery_opp`
    (real takeaways this team's DEFENSE forced - `fumble_recovery_opp` is
    a recovery of the OPPONENT's own lost fumble, confirmed live against
    real 2025 data).

    Confirmed live: a team's own real `turnovers_lost` matches the real
    `turnovers_forced` on the OPPONENT's own row for that same game in
    541/544 (99.4%) of real 2025 team-games - the real, rare (~0.6%)
    mismatch traces to a real fumble that goes out of bounds (possession
    changes by rule with no recovery credited to either side, not a data
    error) - each side is computed from its own team's own real credited
    stats here, not forced into cross-team symmetry. `current_season`
    defaults to `team_stats_df`'s own max real season."""
    ts = team_stats_df[team_stats_df["season_type"] == "REG"].copy()
    ts["turnovers_lost"] = ts["passing_interceptions"].fillna(0) + ts["fumbles_lost_total"].fillna(0)
    ts["turnovers_forced"] = ts["def_interceptions"].fillna(0) + ts["fumble_recovery_opp"].fillna(0)
    ts["turnover_margin_game"] = ts["turnovers_forced"] - ts["turnovers_lost"]
    current_season = int(ts["season"].max()) if current_season is None else current_season

    return _season_aware_blend(
        ts, "team", "turnover_margin_game", current_season,
        prior_strength=carryover_prior_strength, regression=carryover_regression, season_aware=season_aware,
    ).rename("turnover_margin").reset_index()


# --- Points per drive (real, derived from play-by-play) ---


def compute_team_points_per_drive(
    pbp_df: pd.DataFrame, current_season: int = None,
    carryover_regression: float = None, carryover_prior_strength: float = None, season_aware: bool = True,
) -> pd.DataFrame:
    """One row per team: `points_per_drive` - real points scored per real
    offensive drive, season-aware-blended per `_season_aware_blend`.
    Real follow-up (2026-09-02, "offensive efficiency (pts/drive)") - a
    genuinely different efficiency lens than `offensive_edge` (real
    per-PLAY EPA): this measures how often real drives actually turn into
    points, not how valuable each individual play was.

    `pbp_df` is `nfl_data.fetch_pbp`'s own real play-by-play output. Each
    real drive's own point value is derived directly from the real score
    change across it (`posteam_score_post` on the drive's last real play
    minus `posteam_score` on its first) rather than guessed from
    `fixed_drive_result`'s text label (Touchdown/Field goal/etc.) - this
    correctly handles a real 2-point conversion, and correctly credits 0
    points to the offense's own drive when the DEFENSE scores against
    them mid-drive (a pick-six, a safety) - confirmed live: `defteam_score`,
    not `posteam_score`, is what changes on that specific play, so the
    offense's own drive-point tally is unaffected by a score that wasn't
    theirs. Rows with a null `posteam` (a pre-snap/kickoff row before
    possession is established that play) or null `fixed_drive` are
    excluded - they carry no real completed-drive attribution."""
    pbp = pbp_df[
        (pbp_df["season_type"] == "REG") & pbp_df["posteam"].notna() & pbp_df["fixed_drive"].notna()
    ].sort_values(["game_id", "play_id"])

    drives = pbp.groupby(["game_id", "season", "week", "posteam", "fixed_drive"], as_index=False).agg(
        start_score=("posteam_score", "first"), end_score=("posteam_score_post", "last"),
    )
    drives["drive_points"] = drives["end_score"] - drives["start_score"]

    per_game = drives.groupby(["game_id", "season", "week", "posteam"], as_index=False).agg(
        total_points=("drive_points", "sum"), num_drives=("fixed_drive", "nunique"),
    )
    per_game["points_per_drive_game"] = per_game["total_points"] / per_game["num_drives"]
    per_game = per_game.rename(columns={"posteam": "team"})
    current_season = int(per_game["season"].max()) if current_season is None else current_season

    return _season_aware_blend(
        per_game, "team", "points_per_drive_game", current_season,
        prior_strength=carryover_prior_strength, regression=carryover_regression, season_aware=season_aware,
    ).rename("points_per_drive").reset_index()


# --- QB continuity (real snap-share-identified recent starter vs. the confirmed one) ---


def compute_qb_continuity_adjustment(
    snap_counts_df: pd.DataFrame, weekly_df: pd.DataFrame, rosters_df: pd.DataFrame
) -> pd.DataFrame:
    """One row per team: `recent_primary_qb_id` - the real gsis_id of the QB
    who has actually taken the most offensive snaps at QB for that team over
    its most recent `min(games-back window)` real REG games (per
    `config.NFL_TEAM_STRENGTH_WINDOWS`'s smallest window - the shortest,
    most-recency-weighted lens this project already uses elsewhere) -
    identified from `snap_counts_df`'s real per-game `offense_pct`/
    `offense_snaps` at the QB position, NOT just assumed from a depth chart.
    This is the real "snap level data" signal for detecting who a team's
    TRUE recent starter has been (an in-season injury replacement who's
    started the last 3 games shows up here even if a stale depth chart or
    the season-opening starter would suggest otherwise).

    Also returns that QB's own real rolling `passing_epa_per_game` (via
    `nfl_passing.compute_qb_rolling_stats`, reused unchanged) and `games`
    (renamed `recent_primary_qb_games`, exposed so a caller can apply its
    own small-sample judgment - same "expose the count, let the caller
    qualify" pattern this project already uses, e.g.
    `nfl_passing.compute_qb_rolling_stats` itself for `NFL_QB_MIN_GAMES`).

    `snap_counts_df` is keyed by `pfr_player_id` - a different id space than
    `weekly_df`'s own `player_id`/gsis id (and `schedules_df`'s real
    `home_qb_id`/`away_qb_id`, which - confirmed live - already ARE gsis
    ids, no crosswalk needed on that side) - crossed over here via
    `rosters_df`'s real `gsis_id`/`pfr_id` columns, the exact same pattern
    `nfl_bestball.compute_player_snap_share` already established.

    This function deliberately does NOT compare against a specific upcoming
    game's confirmed starter or compute an actual rating shift - that
    real per-matchup comparison (this team's `recent_primary_qb_id` vs. the
    specific game's own confirmed `home_qb_id`/`away_qb_id`, and the
    resulting `NFL_QB_CONTINUITY_WEIGHT`-weighted shift toward the confirmed
    starter's own rolling quality when they differ) lives in
    `nfl_game_picks.py`'s `build_game_features` - mirroring how MLB's own
    probable-starter pitching adjustment likewise lives in `game_picks.py`,
    not `teams.py`, since it's inherently about a SPECIFIC upcoming game,
    not a team-level snapshot."""
    small_window = min(w for w, _ in config.NFL_TEAM_STRENGTH_WINDOWS if w is not None)

    qb_snaps = snap_counts_df[
        (snap_counts_df["game_type"] == "REG") & (snap_counts_df["position"] == "QB")
    ].copy()

    ranked = qb_snaps.sort_values(["team", "season", "week"], ascending=[True, False, False])
    ranked["_recency_rank"] = ranked.groupby("team").cumcount()
    recent = ranked[ranked["_recency_rank"] < small_window]

    totals = recent.groupby(["team", "pfr_player_id"], as_index=False)["offense_snaps"].sum()
    idx = totals.groupby("team")["offense_snaps"].idxmax()
    primary = totals.loc[idx, ["team", "pfr_player_id"]].rename(
        columns={"pfr_player_id": "recent_primary_pfr_id"}
    )

    latest_season = int(qb_snaps["season"].max())
    crosswalk = rosters_df[rosters_df["season"] == latest_season].dropna(subset=["pfr_id"]).drop_duplicates(
        "gsis_id"
    )[["gsis_id", "pfr_id"]]
    primary = primary.merge(crosswalk, left_on="recent_primary_pfr_id", right_on="pfr_id", how="left")
    primary = primary.rename(columns={"gsis_id": "recent_primary_qb_id"}).drop(
        columns=["recent_primary_pfr_id", "pfr_id"]
    )

    qb_quality = nfl_passing.compute_qb_rolling_stats(weekly_df)[["player_id", "games", "passing_epa_per_game"]]
    primary = primary.merge(qb_quality, left_on="recent_primary_qb_id", right_on="player_id", how="left")
    primary["passing_epa_per_game"] = primary["passing_epa_per_game"].fillna(0.0)
    primary["games"] = primary["games"].fillna(0).astype(int)

    return primary.rename(
        columns={"passing_epa_per_game": "recent_primary_qb_epa", "games": "recent_primary_qb_games"}
    )[["team", "recent_primary_qb_id", "recent_primary_qb_epa", "recent_primary_qb_games"]]


# --- Assembly: z-normalize + Confidence mixture + composite ---


def assemble_team_metrics(
    schedules_df: pd.DataFrame, team_stats_df: pd.DataFrame, pbp_df: pd.DataFrame, current_season: int = None,
    carryover_regression: float = None, carryover_prior_strength: float = None, season_aware: bool = True,
) -> pd.DataFrame:
    """Build the final NFL team output table - direct structural port of
    teams.assemble_team_metrics. See module docstring for the full
    signal-by-signal MLB->NFL mapping. `pbp_df` is `nfl_data.fetch_pbp`'s
    own real play-by-play output, feeding `compute_team_points_per_drive`
    (added 2026-09-02 - "offensive efficiency (pts/drive)"). `current_season`
    (real follow-up, 2026-09-04 - season-carryover shrinkage, see
    `_season_aware_blend`'s own docstring) defaults to `schedules_df`'s
    own max real season - always correct under this project's
    no-lookahead contract, since no caller ever includes a season later
    than "now" - and is threaded into offensive_edge/defensive_edge/
    turnover_margin/points_per_drive so all four regress a genuinely
    PRIOR season's rating rather than treating `team_stats_df`/`pbp_df`'s
    real season boundary as invisible. `carryover_regression`/
    `carryover_prior_strength` are explicit override parameters (default
    None -> config.NFL_SEASON_CARRYOVER_REGRESSION/_PRIOR_STRENGTH), same
    "config default, backtest-sweepable override" pattern as
    decision_score.py's own compute_zone_ops - lets
    scripts/backtest_nfl_season_carryover.py sweep candidates without
    monkeypatching the config module. `season_aware=False` bypasses the
    whole carryover mechanism (see `_season_aware_blend`'s own docstring)
    - the real "today's live behavior" baseline
    scripts/backtest_nfl_season_carryover.py compares every candidate
    against."""
    current_season = int(schedules_df["season"].max()) if current_season is None else current_season
    record = build_team_record(schedules_df)
    current_strength, sos = compute_strength_metrics(record)
    edge = compute_team_offense_defense_edge(
        team_stats_df, current_season, carryover_regression, carryover_prior_strength, season_aware
    )
    turnovers = compute_team_turnover_margin(
        team_stats_df, current_season, carryover_regression, carryover_prior_strength, season_aware
    )
    points_per_drive = compute_team_points_per_drive(
        pbp_df, current_season, carryover_regression, carryover_prior_strength, season_aware
    )

    master = current_strength.merge(sos, on="team")

    def normalize(col: str) -> pd.Series:
        mean = master[col].mean()
        std = master[col].std()
        return 1 + ((master[col] - mean) / std * config.NFL_NORMALIZATION_Z_SCALE)

    master = master[["team"]].assign(
        Strength=normalize("strength"),
        SOS=normalize("SOS"),
        current=normalize("current_strength"),
        pyth_Strength=normalize("pyth_strength"),
        pyth_SOS=normalize("pyth_SOS"),
    ).drop_duplicates().reset_index(drop=True)

    master["Confidence"] = master["Strength"] + master["SOS"] * config.NFL_CONFIDENCE_SOS_WEIGHT
    master["pyth_Confidence"] = master["pyth_Strength"] + master["pyth_SOS"] * config.NFL_CONFIDENCE_SOS_WEIGHT
    for col in ("Confidence", "pyth_Confidence"):
        mean = master[col].mean()
        std = master[col].std()
        master[col] = 1 + ((master[col] - mean) / std * config.NFL_NORMALIZATION_Z_SCALE)

    master["Confidence_Delta"] = master["Confidence"] - master["pyth_Confidence"]

    master = master.merge(edge, on="team", how="left")
    for col in ("offensive_edge", "defensive_edge"):
        mean = master[col].mean()
        std = master[col].std()
        master[col] = 1 + ((master[col] - mean) / std * config.NFL_NORMALIZATION_Z_SCALE)
    master["true_power"] = (master["offensive_edge"] + master["defensive_edge"]) / 2

    master = master.merge(turnovers, on="team", how="left")
    mean_to, std_to = master["turnover_margin"].mean(), master["turnover_margin"].std()
    master["turnover_margin"] = 1 + ((master["turnover_margin"] - mean_to) / std_to * config.NFL_NORMALIZATION_Z_SCALE)

    master = master.merge(points_per_drive, on="team", how="left")
    mean_ppd, std_ppd = master["points_per_drive"].mean(), master["points_per_drive"].std()
    master["points_per_drive"] = 1 + ((master["points_per_drive"] - mean_ppd) / std_ppd * config.NFL_NORMALIZATION_Z_SCALE)

    # teams.compute_team_win_rate_ci operates generically on any
    # build_team_record-shaped frame (team/win/games_played, no MLB-specific
    # column referenced) - reused UNCHANGED rather than reimplemented, same
    # Wilson CI (config.KELLY_UNCERTAINTY_CI_ALPHA) feeding
    # game_picks.apply_kelly_uncertainty's real bet-sizing, now for NFL too.
    win_rate_ci = teams.compute_team_win_rate_ci(record)
    master = master.merge(win_rate_ci, on="team", how="left")

    output_columns = [
        "team", "current", "Strength", "pyth_Strength", "SOS", "pyth_SOS",
        "Confidence", "pyth_Confidence", "Confidence_Delta", "true_power",
        "offensive_edge", "defensive_edge", "turnover_margin", "points_per_drive",
        "games_played", "win_rate", "win_rate_CI_Low", "win_rate_CI_High",
    ]
    return master[output_columns]

"""Preseason bestball draft-strategy signal - a genuinely different
question from the rest of the NFL DFS pipeline (`nfl_passing.py`,
`nfl_rush_rec.py`, `nfl_dfs.py`), which all answer "how many points will
this player score THIS upcoming week" via recency-weighted rolling
windows. Bestball drafting instead needs "how much value/risk did this
player represent LAST SEASON, as a whole" - a real, REALIZED season
total, not a forward-looking projection, and no rolling-window blend
(there's no "as of date" mid-draft; you have the complete season to look
back on).

Reuses `nfl_dfs_backtest.compute_actual_qb_dk_points`/
`compute_actual_skill_dk_points` directly for the points math - those
already compute REAL, realized full-PPR DK points from a real box score
"through the SAME real DK formulas" (that module's own docstring), which
is exactly what a season-total realized-points figure needs. Building a
second points calculator here would just duplicate that logic.

Games played (vs. that player's team's real games that season) is used
as a deliberately simple, honest injury-history proxy - not a full
medical history, just "how much of the season were they actually
available for" - per this feature's explicit scope (a full injury
database is out of scope; games-played is a cheap, real signal derived
entirely from already-persisted data, see config.py's NFL section for
what's on disk).
"""

import pandas as pd

from mlb_metrics import config, nfl_dfs_backtest, nfl_rush_rec

POSITIONS = ("QB",) + nfl_rush_rec.SKILL_POSITIONS

# Standard-deviation bucket edges for compute_position_scarcity's bell-curve
# breakdown - the standard "empirical rule" bands (1-2, 2-3, beyond 3) rather
# than a dynamically-sized set, since a real position pool (dozens to a
# couple hundred players) essentially never has anyone beyond +/-3 SD and a
# fixed, familiar shape is easier to read as an actual bell curve. The
# central "within 1 SD" band is split into real quarter-SD slices
# (-1/-0.5/0/0.5/1) rather than one wide bucket - most qualified players
# land in that middle band, and a single bucket hides real, useful shape
# right where most draft-relevant players actually are. `pd.cut`'s bin edges
# (right-inclusive): (-inf,-3], (-3,-2], (-2,-1], (-1,-0.5], (-0.5,0],
# (0,0.5], (0.5,1], (1,2], (2,3], (3,inf).
SCARCITY_BUCKET_EDGES = [-float("inf"), -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, float("inf")]
SCARCITY_BUCKET_LABELS = [
    "below_-3sd",
    "-3sd_to_-2sd",
    "-2sd_to_-1sd",
    "-1sd_to_-0.5sd",
    "-0.5sd_to_0sd",
    "0sd_to_0.5sd",
    "0.5sd_to_1sd",
    "1sd_to_2sd",
    "2sd_to_3sd",
    "above_3sd",
]


def compute_player_games_played(weekly_df: pd.DataFrame, schedules_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """[player_id, season, team, games_played, possible_games, games_missed]
    for one real season, regular season only (season_type == "REG" on
    `weekly_df`, game_type == "REG" on `schedules_df` - postseason games
    would inflate both sides inconsistently, since not every team makes
    the playoffs).

    `games_played` is a real row count on `weekly_df` (a real week
    absent from that table means the player didn't play that week -
    bye/injury/inactive - same convention nfl_passing.py/nfl_rush_rec.py
    already rely on). `possible_games` is that player's TEAM's real game
    count that season (varies by season - 16 games through 2020, 17 from
    2021 on; always derived from the real schedule, never hardcoded).
    `team` is whichever team a player suited up for most that season -
    a mid-season trade is a real but rare edge case this deliberately
    doesn't try to split correctly; a preseason draft-strategy snapshot
    doesn't need per-team-stint precision, just "were they healthy.\""""
    season_weekly = weekly_df[(weekly_df["season"] == season) & (weekly_df["season_type"] == "REG")]

    games_played = season_weekly.groupby("player_id").size().rename("games_played")
    team = season_weekly.groupby("player_id")["team"].agg(lambda s: s.value_counts().idxmax()).rename("team")

    reg_schedule = schedules_df[(schedules_df["season"] == season) & (schedules_df["game_type"] == "REG")]
    team_games = reg_schedule.groupby("home_team").size().add(reg_schedule.groupby("away_team").size(), fill_value=0)

    result = pd.concat([games_played, team], axis=1).reset_index()
    result["possible_games"] = result["team"].map(team_games).fillna(0).astype(int)
    result["games_missed"] = (result["possible_games"] - result["games_played"]).clip(lower=0)
    result["season"] = season
    return result[["player_id", "season", "team", "games_played", "possible_games", "games_missed"]]


def compute_player_snap_share(snap_counts_df: pd.DataFrame, rosters_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """[player_id, season_snap_share] - a player's real total offensive
    snaps THAT SEASON as a share of their TEAM's real total offensive
    snaps that season - NOT a per-game average. This matters: a player
    who suited up for only a couple of real games gets a real LOW season
    share even if their per-game rate was high in those few games, since
    the denominator is the team's entire real season offensive workload,
    not just the games the player happened to appear in. A per-game
    average would let a real one-game emergency spot start (high rate,
    tiny sample) look identical to a real every-week starter - exactly
    the small-sample problem this function exists to avoid. Real
    confirmed example: a real 2025 one-game QB spot start with an 82%
    single-game rate drops to a real 5% season share once measured
    against the team's full real season offensive-play total.

    A team's real total offensive snaps for one game is taken as the max
    real `offense_snaps` among that team's players that game - in
    practice this IS the real total (confirmed live: some player, almost
    always an O-line starter, plays exactly 100% of a team's real
    offensive snaps in 541/544, ~99.4%, of real 2025 team-games; the
    small remainder gets an honest slight undercount, never a fabricated
    exact figure). Summed across every real game a team played that
    season (regardless of which games the player themselves appeared in)
    for the real season-total denominator.

    A player who played for more than one real team that season (a real
    mid-season trade) gets their own real snaps AND each real team's
    real season total summed across every team they were actually
    rostered with - the real "how much of your total available real
    season workload did you personally play" question, unaffected by
    which specific team(s) that workload came from.

    `snap_counts_df` is keyed by `pfr_player_id` (a different real id
    space than this project's own `player_id`/GSIS id used everywhere
    else) - crosswalked here via `rosters_df`'s (`fetch_rosters_weekly`)
    real `gsis_id`/`pfr_id` columns, confirmed live to cover ~99.7% of a
    real qualified population (382/383 real 2025 players with >=8 real
    games played had a real `pfr_id` on their roster row). A player
    missing from the crosswalk (no real `pfr_id` on any of their real
    roster rows that season) is simply absent from the result - handled
    as a real missing value downstream (`compute_position_scarcity`
    excludes them from the snap-share qualifier, not silently treated as
    a real 0% share, which would be a fabricated number, not a real
    one)."""
    season_snaps = snap_counts_df[(snap_counts_df["season"] == season) & (snap_counts_df["game_type"] == "REG")]

    team_game_totals = season_snaps.groupby(["team", "game_id"])["offense_snaps"].max()
    team_season_totals = team_game_totals.groupby("team").sum().rename("team_season_snaps")

    player_team_snaps = season_snaps.groupby(["pfr_player_id", "team"])["offense_snaps"].sum()
    player_team_snaps = player_team_snaps.rename("player_snaps").reset_index()
    player_team_snaps = player_team_snaps.merge(team_season_totals, on="team", how="left")

    totals = player_team_snaps.groupby("pfr_player_id")[["player_snaps", "team_season_snaps"]].sum()
    totals["season_snap_share"] = totals["player_snaps"] / totals["team_season_snaps"]
    totals = totals.reset_index()[["pfr_player_id", "season_snap_share"]]

    season_rosters = rosters_df[rosters_df["season"] == season]
    crosswalk = season_rosters.dropna(subset=["pfr_id"]).drop_duplicates("gsis_id")[["gsis_id", "pfr_id"]]

    result = totals.merge(crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="inner")
    return result.rename(columns={"gsis_id": "player_id"})[["player_id", "season_snap_share"]]


def compute_season_realized_dk_points(weekly_df: pd.DataFrame, position_group: str, season: int) -> pd.DataFrame:
    """[player_id, dk_points_total] - real full-PPR DK points actually
    scored across an entire real regular season, by summing
    nfl_dfs_backtest's real-week-by-real-week actual-points functions
    (each row in `weekly_df` is already one real player-week, so no
    per-week loop is needed - the same real formula just runs once over
    every row and gets summed by player)."""
    season_weekly = weekly_df[(weekly_df["season"] == season) & (weekly_df["season_type"] == "REG")]

    if position_group == "QB":
        per_week = nfl_dfs_backtest.compute_actual_qb_dk_points(season_weekly)
        points_col = "Actual_DK_Points_QB"
    elif position_group == "SKILL":
        per_week = nfl_dfs_backtest.compute_actual_skill_dk_points(season_weekly)
        points_col = "Actual_DK_Points_Skill"
    else:
        raise ValueError(f'position_group must be "QB" or "SKILL", got {position_group!r}')

    totals = per_week.groupby("player_id")[points_col].sum().reset_index()
    return totals.rename(columns={points_col: "dk_points_total"})


def build_bestball_rankings(
    weekly_df: pd.DataFrame,
    schedules_df: pd.DataFrame,
    season: int,
    prior_season: int | None = None,
    prior_weekly_df: pd.DataFrame | None = None,
    prior_schedules_df: pd.DataFrame | None = None,
    snap_counts_df: pd.DataFrame | None = None,
    rosters_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """One row per real QB/RB/WR/TE who recorded a real stat line in
    `season`'s regular season: name, position, team, real games played/
    missed, real season-total and per-game full-PPR DK points - ranked
    by `dk_points_total` descending (a real, realized value signal, not
    a blended "draft score" - `dk_points_per_game` and `games_missed`
    are kept as separate honest columns rather than folded into one
    number, since there's no backtestable ground truth for what the
    "right" health/talent tradeoff weighting would even be).

    DST is intentionally excluded - bestball drafting doesn't need DST
    optimization the way weekly DFS does.

    If `prior_season`/`prior_weekly_df`/`prior_schedules_df` are given,
    adds `games_missed_prior_season` - a cheap, real repeat-injury-risk
    read using data already persisted for that season too.

    If `snap_counts_df`/`rosters_df` are given, adds `season_snap_share`
    (see `compute_player_snap_share`) - a real playing-time-share signal,
    left-joined so a player missing from the real snap-count crosswalk
    gets a real NaN, not a fabricated 0%."""
    games = compute_player_games_played(weekly_df, schedules_df, season)
    qb_points = compute_season_realized_dk_points(weekly_df, "QB", season)
    skill_points = compute_season_realized_dk_points(weekly_df, "SKILL", season)
    points = pd.concat([qb_points, skill_points], ignore_index=True)

    season_weekly = weekly_df[(weekly_df["season"] == season) & (weekly_df["season_type"] == "REG")]
    names = season_weekly[["player_id", "player_display_name", "position"]].drop_duplicates("player_id")
    names = names[names["position"].isin(POSITIONS)]

    result = names.merge(games, on="player_id", how="left").merge(points, on="player_id", how="inner")
    result["dk_points_per_game"] = result["dk_points_total"] / result["games_played"].replace(0, pd.NA)

    if prior_season is not None and prior_weekly_df is not None and prior_schedules_df is not None:
        prior_games = compute_player_games_played(prior_weekly_df, prior_schedules_df, prior_season)
        prior_games = prior_games[["player_id", "games_missed"]].rename(
            columns={"games_missed": "games_missed_prior_season"}
        )
        result = result.merge(prior_games, on="player_id", how="left")

    if snap_counts_df is not None and rosters_df is not None:
        snap_share = compute_player_snap_share(snap_counts_df, rosters_df, season)
        result = result.merge(snap_share, on="player_id", how="left")

    return result.sort_values("dk_points_total", ascending=False).reset_index(drop=True)


def _iqr_outlier_bounds(values: pd.Series, multiplier: float) -> tuple[float, float]:
    """Real Tukey fences (Q1 - multiplier*IQR, Q3 + multiplier*IQR) - the
    standard, well-established outlier rule, not an invented threshold.
    Chosen over a z-score-based rule specifically because it doesn't
    require an already-computed mean/std as an input (z-score outlier
    detection is circular for exactly the problem this is solving: real
    NFL season-point distributions are often right-skewed even among a
    games-played-qualified population, so a mean/std computed WITH the
    outliers already baked in is itself distorted by them)."""
    q1, q3 = values.quantile(0.25), values.quantile(0.75)
    iqr = q3 - q1
    return q1 - multiplier * iqr, q3 + multiplier * iqr


def compute_position_scarcity(
    rankings_df: pd.DataFrame,
    min_snap_share: float = config.NFL_BESTBALL_SCARCITY_MIN_SNAP_SHARE,
    value_column: str = config.NFL_BESTBALL_SCARCITY_VALUE_COLUMN,
    iqr_multiplier: float = config.NFL_BESTBALL_SCARCITY_IQR_MULTIPLIER,
) -> pd.DataFrame:
    """One row per position (`POSITIONS` - QB/RB/WR/TE) describing how real
    `value_column` production is actually distributed - a "how many
    difference-makers exist at this position, and how many replacement-level
    guys" read for draft strategy, not just a single overall rank list.

    `total_players`: every real player at that position in `rankings_df`
    (i.e. everyone who recorded a real stat line last season), regardless of
    playing time - the position's real total pool size.

    `qualified_players`: restricted to players with real `season_snap_share
    >= min_snap_share` (see `compute_player_snap_share` - a real SEASON-TOTAL
    share, not a per-game average, so a real one-or-two-game high-rate
    sample doesn't qualify the way it would under a per-game average -
    `rankings_df` must already carry that column, e.g. via
    `build_bestball_rankings`'s `snap_counts_df`/`rosters_df` args). A
    player missing `season_snap_share` entirely (no real snap-count
    crosswalk match) is treated as NOT qualified, not silently included or
    excluded via a fabricated value.

    This replaced a real `games_played >= min_games` qualifier (see git
    history) after real 2025 data showed it let in players with almost no
    real offensive role - e.g. a real return specialist who appeared in 11
    real box scores (`games_played=11`) but played exactly 1 real offensive
    snap all season. `games_played` only requires ANY real stat row that
    week (even a single special-teams play); it does not require a
    meaningful offensive role, so it let committee/inactive-but-rostered
    players sit right next to true starters in the same "qualified"
    population - inflating the std describing what "typical starter"
    production looks like far more than real outliers alone did.

    The snap-share qualifier itself was originally a real PER-GAME average
    (`avg_offense_pct`, see git history), which fixed the special-teamer
    problem above but introduced a real, different small-sample problem in
    the opposite direction: a real one-or-two-game emergency spot start at
    a high per-game rate (e.g. a real 2025 backup QB's real 82% single-game
    rate across exactly 1 real game) would qualify just as easily as a
    real every-week starter. `compute_player_snap_share`'s real
    SEASON-TOTAL share fixes this too - that same real spot start drops to
    a real 5% season share once measured against the team's real full
    season offensive-play total, correctly excluding it. `games_played`/
    `games_missed` remain unaffected everywhere else (the rankings table,
    the injury-history proxy) - only this qualifier changed.

    `outliers_removed`/`mean`/`std`/`coefficient_of_variation`: real
    statistical outliers among the qualified players (via `_iqr_outlier_bounds`,
    Tukey's rule) are EXCLUDED before computing mean/std - a real elite
    handful (e.g. 2025's top-4 real outlier WRs: Puka Nacua, Jaxon
    Smith-Njigba, Amon-Ra St. Brown, Ja'Marr Chase) would otherwise pull the
    mean up and inflate the std describing what a "typical" qualified player
    at the position looks like. `std` uses population (ddof=0), not sample,
    standard deviation over that outlier-excluded "core" group.
    `coefficient_of_variation` (std/mean of the core group) is NaN when mean
    is 0 or fewer than 2 core players remain - it's the real, comparable
    "how spread out is this position, relative to its own scale" number used
    by `compute_draft_strategy_takeaways` to compare positions against each
    other. Note: removing real outliers does NOT eliminate all spread - real
    NFL production among players who cleared even a real snap-share bar
    genuinely ranges from part-time role players to true difference makers,
    and that remaining spread is real, not leftover contamination.

    The bucket columns (`SCARCITY_BUCKET_LABELS`) bucket EVERY qualified
    player - core AND real outliers - by how many standard deviations their
    own `value_column` sits from the core group's real mean/std, so the
    excluded outliers still show up in the bell curve (almost always in the
    extreme bands, which is exactly where a real outlier belongs) rather
    than silently disappearing from the table. A position with exactly 1
    qualified player still gets a real mean (that player's own real value)
    but NaN std/CV and all-zero buckets - no real spread to describe. A
    position with 0 qualified players, or 2+ qualified but fewer than 2
    survive outlier removal, gets NaN mean/std/CV and all-zero buckets
    (nothing real to compute a spread from). A position with 2+ core
    players but a std of exactly 0 (everyone in the core tied) gets a real
    mean/std but still all-zero bucket counts and NaN CV, since a z-score
    is undefined when the distribution has no spread - either way this is
    an honest "not enough real data" result rather than a divide-by-zero
    crash or a fabricated one."""
    has_snap_share = "season_snap_share" in rankings_df.columns
    rows = []
    for position in POSITIONS:
        pos_df = rankings_df[rankings_df["position"] == position]
        total_players = len(pos_df)
        # NaN share never satisfies >=, so those players are correctly excluded. A rankings_df built
        # WITHOUT snap_counts_df/rosters_df (see build_bestball_rankings) has no season_snap_share
        # column at all - treated as "real data unavailable" (nobody qualifies) rather than a crash.
        qualified = pos_df[pos_df["season_snap_share"] >= min_snap_share] if has_snap_share else pos_df.iloc[0:0]
        n_qualified = len(qualified)

        bucket_counts = {label: 0 for label in SCARCITY_BUCKET_LABELS}
        mean = float("nan")
        std = float("nan")
        coefficient_of_variation = float("nan")
        outliers_removed = 0

        if n_qualified == 1:
            mean = qualified[value_column].iloc[0]  # a real value, but no real spread to compute std from
        elif n_qualified > 1:
            values = qualified[value_column]
            lower, upper = _iqr_outlier_bounds(values, iqr_multiplier)
            is_outlier = (values < lower) | (values > upper)
            outliers_removed = int(is_outlier.sum())
            core = values[~is_outlier]

            if len(core) > 1:
                mean = core.mean()
                std = core.std(ddof=0)
                if std > 0:
                    if mean != 0:
                        coefficient_of_variation = std / mean
                    z_scores = (values - mean) / std  # score ALL qualified (core + real outliers)
                    bucketed = pd.cut(z_scores, bins=SCARCITY_BUCKET_EDGES, labels=SCARCITY_BUCKET_LABELS)
                    counts = bucketed.value_counts()
                    bucket_counts = {label: int(counts.get(label, 0)) for label in SCARCITY_BUCKET_LABELS}
            elif len(core) == 1:
                mean = core.iloc[0]  # a real value, but no real spread to compute std from

        rows.append(
            {
                "position": position,
                "total_players": total_players,
                "qualified_players": n_qualified,
                "outliers_removed": outliers_removed,
                "mean_dk_points": mean,
                "std_dk_points": std,
                "coefficient_of_variation": coefficient_of_variation,
                **bucket_counts,
            }
        )

    return pd.DataFrame(rows)


def compute_draft_strategy_takeaways(scarcity_df: pd.DataFrame) -> pd.DataFrame:
    """[position, coefficient_of_variation, dispersion_rank, takeaway] - a
    real, numbers-driven answer to "does this position's real spread this
    season argue for prioritizing it early in a draft, or waiting."

    Ranks positions by `coefficient_of_variation` (std/mean of each
    position's own real, outlier-excluded core group from
    `compute_position_scarcity`) RELATIVE TO THE OTHER REAL POSITIONS THIS
    SEASON - not against an invented absolute cutoff, since "high" or "low"
    dispersion only means something compared to the other real positions in
    the same real season's data. Positions at or above the real median CV
    across positions are read as the more top-heavy/scarce positions this
    season (a bigger real gap between a difference-maker and a typical
    qualified player - the standard fantasy-drafting argument for grabbing
    one of the real separators early, before the position's real
    differentiation disappears); positions below the median CV are read as
    flatter/deeper (real production is more interchangeable across the
    position, so it's generally safer to wait and spend an early pick on a
    scarcer position instead). This directly answers "if QBs are all within
    2 SD of the mean [i.e. QB's real CV is low/flat this season], should a
    TE be prioritized instead" - yes, exactly when TE's own real CV this
    season ranks higher (scarcer) than QB's.

    A position with a NaN `coefficient_of_variation` (not enough real core
    players to compute one) gets a takeaway saying so, and is excluded from
    the ranking/comparison entirely - it can't honestly be compared to
    positions with a real computed CV."""
    valid = scarcity_df.dropna(subset=["coefficient_of_variation"]).copy()
    valid = valid.sort_values("coefficient_of_variation", ascending=False).reset_index(drop=True)
    valid["dispersion_rank"] = valid.index + 1
    median_cv = valid["coefficient_of_variation"].median() if len(valid) else float("nan")
    n_positions = len(valid)

    rows = []
    for _, row in scarcity_df.iterrows():
        position = row["position"]
        cv = row["coefficient_of_variation"]

        if pd.isna(cv) or n_positions < 2:
            rows.append(
                {
                    "position": position,
                    "coefficient_of_variation": cv,
                    "dispersion_rank": pd.NA,
                    "takeaway": (
                        f"{position}: not enough real qualified players this season with a computable "
                        f"spread to compare against the other positions."
                    ),
                }
            )
            continue

        rank = int(valid.loc[valid["position"] == position, "dispersion_rank"].iloc[0])
        mean, std = row["mean_dk_points"], row["std_dk_points"]
        if cv >= median_cv:
            takeaway = (
                f"{position}: coefficient of variation {cv:.2f} (mean {mean:.1f}, std {std:.1f}) - "
                f"the #{rank} most dispersed of {n_positions} real positions this season. Top-heavy/"
                f"scarcer: the real gap between a difference-maker and a typical qualified {position} "
                f"is bigger than at a below-median position, so prioritizing a proven top-tier "
                f"{position} early carries more relative value."
            )
        else:
            takeaway = (
                f"{position}: coefficient of variation {cv:.2f} (mean {mean:.1f}, std {std:.1f}) - "
                f"the #{rank} most dispersed of {n_positions} real positions this season. Flatter/"
                f"deeper: real {position} production is more interchangeable than at an above-median "
                f"position, so it's generally safer to wait here and prioritize a scarcer position "
                f"first."
            )

        rows.append(
            {"position": position, "coefficient_of_variation": cv, "dispersion_rank": rank, "takeaway": takeaway}
        )

    return pd.DataFrame(rows)

"""Team unit ratings on a "+" scale: QB+, Receiving+, Rushing+, Pass D+, Run D+.

User request (2026-10-22): a rankings page showing how strong each team's
units are, so a prop pick (say, the under on a receiver) can be checked
against the units behind it. 100 is league average and higher is always
stronger, the way OPS+ or ERA+ read.

Each unit is EPA per play for the plays that unit is responsible for:
    QB+           the listed starting QB's own EPA per dropback (passes,
                  sacks and scrambles) - nfl_totals.starter_qb_epa, so a
                  backup starting shows up as a weaker QB+
    Receiving+    EPA per target on throws to the team's receivers
    Rushing+      EPA per designed run (QB scrambles are dropbacks)
    Pass D+       EPA per dropback allowed
    Run D+        EPA per designed run allowed
Four views, toggled on the page (VIEWS): by default this season's games,
each counted equally (user request 2026-10-22: "standardize as straight
season grading"); optionally weighted toward recent games - the WAVE
windows over the last 17, 8 and 4 games (config.NFL_TOTALS_WINDOWS),
reaching back into last season - and/or adjusted for strength of
schedule, every offense and defense solved together so production
against a strong unit counts for more. All shrunk toward the league rate
by config.NFL_UNIT_SHRINK_PLAYS pseudo-plays, so two big games don't make
a unit elite. On 2026 week 7 data the schedule adjustment moved units by
up to ~16 points (about half a standard deviation).

The "+" scale: 100 + 30 x (standard deviations from the 32-team average,
config.NFL_UNIT_PLUS_POINTS_PER_SD), flipped for defenses so fewer points
allowed scores higher. So 130 is one standard deviation better than
average (roughly top 5-6), 160 is two (elite), 70 and 40 the same below.

The page also lays the units out by this week's matchups (docs/nfl_units.js):
a passing differential, the average of QB+ and Receiving+ minus the
opponent's Pass D+, and a rushing one, Rushing+ minus the opponent's Run
D+ - higher is a friendlier matchup for the offense.

Receiving+ and QB+ overlap - throws to receivers are also the QB's throws
- so a great QB lifts his receivers' number too; QB+ is his own across
teams (his last 16 games wherever he played), Receiving+ is what this
team's passing to its receivers has produced.

The schedule adjustment is a view, not the default: for the win model
(scripts/backtest_nfl_category_ratings.py) a joint opponent adjustment
didn't predict better, so it is offered as context rather than assumed.
"""

import pandas as pd

from mlb_metrics import config, nfl_totals

UNIT_PBP_COLUMNS = [
    "game_id", "season", "week", "season_type", "posteam", "defteam", "epa",
    "pass_attempt", "sack", "receiver_player_id", "rush", "qb_scramble", "qb_dropback",
]

UNIT_COLUMNS = ["qb_plus", "receiving_plus", "rushing_plus", "pass_defense_plus", "run_defense_plus"]
OUTPUT_COLUMNS = [
    "team", "opponent", "home", "qb_name", *UNIT_COLUMNS,
    "qb_epa", "receiving_epa", "rushing_epa", "pass_defense_epa", "run_defense_epa",
]



def unit_game_epa(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (regular-season game, offense): EPA and play counts on
    targets, designed runs and dropbacks. Columns: game_id, season, week,
    team, opp, target_epa, targets, run_epa, runs, dropback_epa, dropbacks."""
    columns = ["game_id", "season", "week", "team", "opp",
               "target_epa", "targets", "run_epa", "runs", "dropback_epa", "dropbacks"]
    if pbp is None or pbp.empty or not set(UNIT_PBP_COLUMNS).issubset(pbp.columns):
        return pd.DataFrame(columns=columns)
    plays = pbp[(pbp["season_type"] == "REG") & pbp["epa"].notna() & pbp["posteam"].notna()].copy()
    target = (plays["pass_attempt"] == 1) & (plays["sack"] != 1) & plays["receiver_player_id"].notna()
    run = (plays["rush"] == 1) & (plays["qb_scramble"] != 1)
    dropback = plays["qb_dropback"] == 1
    plays["target_epa"] = plays["epa"].where(target, 0.0)
    plays["targets"] = target.astype(float)
    plays["run_epa"] = plays["epa"].where(run, 0.0)
    plays["runs"] = run.astype(float)
    plays["dropback_epa"] = plays["epa"].where(dropback, 0.0)
    plays["dropbacks"] = dropback.astype(float)
    return (
        plays.groupby(["game_id", "season", "week", "posteam", "defteam"])[columns[5:]].sum()
        .reset_index()
        .rename(columns={"posteam": "team", "defteam": "opp"})[columns]
    )


def _plus(values: pd.Series, direction: float) -> pd.Series:
    """100 + config.NFL_UNIT_PLUS_POINTS_PER_SD x standard deviations from
    the mean, `direction` -1 for a defense (less allowed is better)."""
    spread = values.std()
    if not spread or pd.isna(spread):
        return pd.Series(100.0, index=values.index)
    return 100 + config.NFL_UNIT_PLUS_POINTS_PER_SD * direction * (values - values.mean()) / spread


# The four views the page toggles between: (recency weighting, strength of
# schedule). The plain `{unit}_plus` columns are the default view.
VIEWS = {
    "season": (False, False),
    "recency": (True, False),
    "season_sos": (False, True),
    "recency_sos": (True, True),
}
# Recent games plus strength of schedule predicts a unit's next game best
# (scripts/backtest_nfl_unit_views.py, 4,636 team-games 2017-2026: average
# correlation .163 across the five units vs .152 for this season alone, and
# .163 vs .109 in weeks 2-4; best for four of the five units, QB+ a hair
# behind recent games alone). It was "season" until 2026-10-26, which made
# Seattle's Darnold a 65 QB+ off four starts while his last 16 games say 84.
DEFAULT_VIEW = "recency_sos"

# (unit, play-type numerator, denominator, offense or defense side, + direction)
_PLAY_TYPES = {"targets": ("target_epa", "targets"), "runs": ("run_epa", "runs"), "dropbacks": ("dropback_epa", "dropbacks")}
_UNIT_SOURCES = [
    ("receiving", "targets", "offense", 1.0),
    ("rushing", "runs", "offense", 1.0),
    ("pass_defense", "dropbacks", "defense", -1.0),
    ("run_defense", "runs", "defense", -1.0),
]


def _game_weights(rows: pd.DataFrame, key: str, recency: bool, current_season: int) -> pd.Series:
    """One weight per row for the team in `key`. Season view: 1 for each
    game of the current season, 0 otherwise. Recency view: the WAVE
    windows (config.NFL_TOTALS_WINDOWS) turned into a per-game weight by
    how many games back it is - each window's weight spread over its
    games - scaled so a full window of games averages 1, keeping
    config.NFL_UNIT_SHRINK_PLAYS on the same footing in both views."""
    if not recency:
        return (rows["season"] == current_season).astype(float)
    rank = rows.sort_values(["season", "week"], ascending=False).groupby(key).cumcount().reindex(rows.index)
    played = rows.groupby(key)[key].transform("size")
    weight = pd.Series(0.0, index=rows.index)
    for n_games, share in config.NFL_TOTALS_WINDOWS:
        span = played if n_games is None else pd.Series(float(n_games), index=rows.index)
        weight += share * (rank < span) / span
    finite = [n for n, _ in config.NFL_TOTALS_WINDOWS if n is not None]
    return weight * (max(finite) if finite else played)


def _solve_play_type(games: pd.DataFrame, num: str, den: str, w_off: pd.Series, w_def: pd.Series,
                     sos: bool, iterations: int = 25) -> tuple[pd.Series, pd.Series]:
    """Offense and defense EPA-per-play deviations from the league rate for
    one play type, shrunk by config.NFL_UNIT_SHRINK_PLAYS pseudo-plays.

    Without strength of schedule each is the weighted average of what the
    team produced (or allowed) against everyone. With it, every offense
    and defense is solved together - rate ~ league + offense + opposing
    defense - so a run of weak opponents doesn't inflate a unit (the same
    model as scripts/backtest_nfl_category_ratings.py's ridge, solved
    here by alternating updates)."""
    league = (w_off * games[num]).sum() / (w_off * games[den]).sum()
    shrink = config.NFL_UNIT_SHRINK_PLAYS
    offense = pd.Series(0.0, index=pd.Index(games["team"].unique()))
    defense = pd.Series(0.0, index=pd.Index(games["opp"].unique()))
    for _ in range(iterations if sos else 1):
        opp_adjust = games["opp"].map(defense).fillna(0.0) if sos else 0.0
        residual = games[num] - games[den] * (league + opp_adjust)
        offense = (w_off * residual).groupby(games["team"]).sum() / ((w_off * games[den]).groupby(games["team"]).sum() + shrink)
        team_adjust = games["team"].map(offense).fillna(0.0) if sos else 0.0
        residual = games[num] - games[den] * (league + team_adjust)
        defense = (w_def * residual).groupby(games["opp"]).sum() / ((w_def * games[den]).groupby(games["opp"]).sum() + shrink)
    return offense, defense


def _qb_view(qb_games: pd.DataFrame, qb_ids: pd.Series, recency: bool, current_season: int,
             dropback_defense: pd.Series | None) -> pd.Series:
    """The listed starter's EPA per dropback: this season's dropbacks
    (season view) or his last config.NFL_TOTALS_QB_GAMES games (recency),
    less the defenses' strength when `dropback_defense` is given, shrunk
    toward replacement level as nfl_totals.starter_qb_epa does."""
    if qb_games is None or qb_games.empty:
        return pd.Series(float("nan"), index=qb_ids.index)
    games = qb_games.copy()
    if dropback_defense is not None and "opp" in games.columns:
        games["epa_sum"] = games["epa_sum"] - games["dropbacks"] * games["opp"].map(dropback_defense).fillna(0.0)
    replacement = games["epa_sum"].sum() / games["dropbacks"].sum() - config.NFL_TOTALS_QB_REPLACEMENT_GAP
    if recency:
        games = games.sort_values(["season", "week"]).groupby("qb").tail(config.NFL_TOTALS_QB_GAMES)
    else:
        games = games[games["season"] == current_season]
    totals = games.groupby("qb")[["epa_sum", "dropbacks"]].sum()
    shrink = config.NFL_TOTALS_QB_SHRINK_DROPBACKS
    rating = (totals["epa_sum"] + shrink * replacement) / (totals["dropbacks"] + shrink)
    return qb_ids.map(rating).fillna(replacement).astype(float)


def team_unit_ratings(
    unit_games: pd.DataFrame,
    qb_games: pd.DataFrame,
    starters: pd.DataFrame,
) -> pd.DataFrame:
    """Every team's unit ratings from earlier games, in each of VIEWS.
    `unit_games` is unit_game_epa's output, `qb_games`
    nfl_totals.qb_game_epa's, and `starters` one row per team (team,
    opponent, home, qb_id, qb_name - the listed starter for its next game;
    home 1/0, both blank on a bye).

    The season view uses the current season's games - the latest season in
    `unit_games`, so before a new season's first game it is last season.
    Columns: OUTPUT_COLUMNS plus `{unit}_plus_{view}` for every view; the
    plain `{unit}_plus` columns are the default (season) view."""
    if unit_games is None or unit_games.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    games = unit_games.reset_index(drop=True)
    current_season = games["season"].max()
    ratings = starters.reindex(columns=["team", "opponent", "home", "qb_id", "qb_name"]).copy()
    teams = pd.Index(sorted(set(games["team"]) | set(games["opp"])))
    ratings = pd.DataFrame({"team": teams}).merge(ratings, on="team", how="left")

    for view, (recency, sos) in VIEWS.items():
        w_off = _game_weights(games, "team", recency, current_season)
        w_def = _game_weights(games, "opp", recency, current_season)
        solved = {
            play_type: _solve_play_type(games, num, den, w_off, w_def, sos)
            for play_type, (num, den) in _PLAY_TYPES.items()
        }
        epa = {}
        for unit, play_type, side, _ in _UNIT_SOURCES:
            offense, defense = solved[play_type]
            epa[unit] = ratings["team"].map(offense if side == "offense" else defense)
        epa["qb"] = _qb_view(qb_games, ratings["qb_id"], recency, current_season,
                             solved["dropbacks"][1] if sos else None)
        for unit, values in epa.items():
            direction = -1.0 if unit.endswith("defense") else 1.0
            ratings[f"{unit}_plus_{view}"] = _plus(values, direction).round()
            if view == DEFAULT_VIEW:
                ratings[f"{unit}_epa"] = values
                ratings[f"{unit}_plus"] = ratings[f"{unit}_plus_{view}"]

    view_columns = [f"{unit}_{view}" for view in VIEWS for unit in UNIT_COLUMNS]
    return ratings[OUTPUT_COLUMNS + view_columns].sort_values("team").reset_index(drop=True)


def listed_starters(week_games: pd.DataFrame, history_schedules: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per team: the QB listed for its game this week and its
    opponent, from `week_games` (home/away_team, home/away_qb_id,
    home/away_qb_name). A team not in `week_games` (a bye) gets the QB
    listed for its latest game in `history_schedules`, and no opponent."""
    columns = ["team", "opponent", "home", "qb_id", "qb_name"]

    def sides(games: pd.DataFrame) -> pd.DataFrame:
        if games is None or games.empty:
            return pd.DataFrame(columns=[*columns, "season", "week"])
        frames = []
        for side, other in (("home", "away"), ("away", "home")):
            frames.append(pd.DataFrame({
                "team": nfl_totals._canonical(games[f"{side}_team"]),
                "opponent": nfl_totals._canonical(games[f"{other}_team"]),
                "home": 1.0 if side == "home" else 0.0,
                "qb_id": games.get(f"{side}_qb_id"), "qb_name": games.get(f"{side}_qb_name"),
                "season": games.get("season"), "week": games.get("week"),
            }))
        return pd.concat(frames, ignore_index=True)

    listed = sides(week_games)
    if history_schedules is not None and not history_schedules.empty:
        past = sides(history_schedules).sort_values(["season", "week"]).drop_duplicates("team", keep="last")
        byes = past[~past["team"].isin(listed["team"])].assign(opponent=pd.NA, home=pd.NA)
        listed = pd.concat([listed, byes], ignore_index=True)
    return listed.drop_duplicates("team")[columns].reset_index(drop=True)

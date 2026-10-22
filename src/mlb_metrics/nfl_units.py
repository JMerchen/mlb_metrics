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
Team units are blended over each team's last 17, 8 and 4 games
(config.NFL_TOTALS_WINDOWS, the same WAVE recipe the point totals use)
and shrunk toward the league rate by config.NFL_UNIT_SHRINK_PLAYS
pseudo-plays, so two big games don't make a unit elite.

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

Not opponent-adjusted, same as the EPA on the game cards: tested for the
win model (scripts/backtest_nfl_category_ratings.py), a joint opponent
adjustment didn't predict better.
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

# (unit, numerator, denominator, team key, + scale direction)
_TEAM_UNITS = [
    ("receiving", "target_epa", "targets", "team", 1.0),
    ("rushing", "run_epa", "runs", "team", 1.0),
    ("pass_defense", "dropback_epa", "dropbacks", "opp", -1.0),
    ("run_defense", "run_epa", "runs", "opp", -1.0),
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


def team_unit_ratings(
    unit_games: pd.DataFrame,
    qb_games: pd.DataFrame,
    starters: pd.DataFrame,
) -> pd.DataFrame:
    """Every team's unit ratings from earlier games. `unit_games` is
    unit_game_epa's output, `qb_games` nfl_totals.qb_game_epa's, and
    `starters` one row per team (team, opponent, home, qb_id, qb_name -
    the listed starter for its next game; home 1/0, both blank on a bye). Columns:
    OUTPUT_COLUMNS, best Receiving+ first is left to the page."""
    if unit_games is None or unit_games.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    games = unit_games.sort_values(["season", "week"])
    rows = {}
    for unit, num, den, key, _ in _TEAM_UNITS:
        prior = games[num].sum() / games[den].sum()
        for team, own in games.groupby(key):
            rows.setdefault(team, {"team": team})[f"{unit}_epa"] = nfl_totals._blend(
                own, num, den, prior, config.NFL_UNIT_SHRINK_PLAYS, config.NFL_TOTALS_WINDOWS
            )
    ratings = pd.DataFrame(list(rows.values()))
    ratings = ratings.merge(starters.reindex(columns=["team", "opponent", "home", "qb_id", "qb_name"]), on="team", how="left")
    ratings["qb_epa"] = nfl_totals.starter_qb_epa(qb_games, ratings["qb_id"])
    ratings["qb_plus"] = _plus(ratings["qb_epa"], 1.0)
    for unit, _, _, _, direction in _TEAM_UNITS:
        ratings[f"{unit}_plus"] = _plus(ratings[f"{unit}_epa"], direction)
    ratings[UNIT_COLUMNS] = ratings[UNIT_COLUMNS].round()
    return ratings[OUTPUT_COLUMNS].sort_values("team").reset_index(drop=True)


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

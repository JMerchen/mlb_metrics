"""A quarterback-adjusted Elo that also learns from the betting line: the
NFL win model (v5).

Why (2026-11-15; the user's "if there are better models out there ...
leverage what they have that we don't", then "could we recreate their
inputs instead of just tailing or combining?"): the public nfelo model
scored a log loss of 0.6197 on the 1,766 games our v4 model scored 0.6305
on. Taking nfelo apart, its team ratings alone were no better than ours
(0.6300); the gap was its quarterback adjustment - the one FiveThirtyEight
designed - and its use of the market's lines. This module rebuilds both
from our own data. It reads nothing from nfelo.

How it works:
  * Every quarterback game gets a VALUE from the box score
    (FiveThirtyEight's weights: completions, yards and touchdowns for,
    attempts, interceptions and sacks against, plus his rushing), less how
    much more than average the defense he faced usually allows.
  * A quarterback's rating is a running average of his values (a tenth of
    the way to each new game); a team's is the same over whoever played
    (a twentieth); a defense's is what it has allowed.
  * A team's Elo moves with each result (config.NFL_QB_ELO_K points at
    stake, scaled by the margin, a third of the way back to average
    between seasons), and each game's expectation includes the two
    starters' adjustments: 3.3 Elo points for every point of value the
    listed starter is above or below his team's running level. So the
    team rating is the team with its usual quarterback play, and a backup
    starting, or a starter back, moves the game and not the rating.
  * After each game the two teams' gap also moves a quarter of the way
    (config.NFL_QB_ELO_MARKET_PULL) to what that game's closing point
    spread said it was. A past line is public before the next game, and it
    knows things a final score doesn't: who was hurt, who rested starters.
    The line of the game being predicted is never used.
  * Three terms from a 25-season replay of the rating's own misses (see
    config.NFL_QB_ELO_REST_PER_DAY): extra days of rest; a point off a
    backup standing in for the season's regular starter; and, in the last
    week of the regular season, a large penalty for a 10-win team sitting
    that starter, because the rest of its starters are sitting too.

Walk-forward by construction (each game is predicted from the ratings as
they stood before it; nothing is fit), 2019-2026, 1,999 games:
                                  log loss   winners
    v4 (the model this replaced)   0.6307     1,284
    QB adjustment, no lines        0.6242     1,299
    + past closing lines           0.6197     1,303
    + rest, backups, resting       0.6162     1,304   <- this model
    this model and v4 averaged     0.6198     1,313
    Vegas                          0.6094     1,332
It beats v4 by 0.0145 (t 3.8) and in all 8 seasons - two thirds of the
distance v4 was from Vegas - and on nfelo's own 1,766 games the version
before the last three terms scored 0.6177 to nfelo's 0.6197. Over 25
seasons (2002-2026, scripts/research_nfl_model_gaps.py): 0.6166 to Vegas's
0.6106. Averaging v4 back in made it worse at every
weight tried, so v4 is retired. Two more of nfelo's inputs were tried and
left out: an EPA-based margin in place of the score (worse: 0.6251 at
half weight) and a bonus after a bye (within noise). The settings were
chosen where both halves of the history agreed (2019-22 and 2023-26), and
nine neighbouring settings all landed within 0.001.
scripts/train_nfl_qb_elo.py prints these and saves the model.
"""

import numpy as np
import pandas as pd

from mlb_metrics import config, nfl_totals

# The What if? panel's "Average starter" option (nfl_whatif.AVERAGE_QB_ID).
AVERAGE_QB_ID = "average"

VALUE_COLUMNS = ["attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
                 "sacks_suffered", "carries", "rushing_yards", "rushing_tds"]


def qb_game_values(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team, passer) with at least one pass
    attempt in a regular-season game: his VALUE that day. Columns: season,
    week, team, player_id, value, attempts."""
    columns = ["season", "week", "team", "player_id", "value", "attempts"]
    if weekly_df is None or weekly_df.empty or not {"player_id", "attempts", "season", "week"}.issubset(weekly_df.columns):
        return pd.DataFrame(columns=columns)
    rows = weekly_df
    if "season_type" in rows.columns:
        rows = rows[rows["season_type"] == "REG"]
    rows = rows[pd.to_numeric(rows["attempts"], errors="coerce").fillna(0) >= 1].copy()
    team_column = "team" if "team" in rows.columns else "recent_team"
    stat = {c: pd.to_numeric(rows[c], errors="coerce").fillna(0) if c in rows.columns else 0.0 for c in VALUE_COLUMNS}
    rows["value"] = (-2.2 * stat["attempts"] + 3.7 * stat["completions"] + stat["passing_yards"] / 5
                     + 11.3 * stat["passing_tds"] - 14.1 * stat["passing_interceptions"] - 8 * stat["sacks_suffered"]
                     - 1.1 * stat["carries"] + 0.6 * stat["rushing_yards"] + 15.9 * stat["rushing_tds"])
    rows["team"] = nfl_totals._canonical(rows[team_column])
    rows["attempts"] = stat["attempts"]
    return rows.groupby(["season", "week", "team", "player_id"], as_index=False)[["value", "attempts"]].sum()[columns]


class QbEloState:
    """The ratings after some run of games: team Elo, each team's running
    quarterback value, each defense's value allowed, and each quarterback's
    own."""

    def __init__(self):
        self.elo, self.team_value, self.defense_value, self.qb_value = {}, {}, {}, {}
        # This season so far: each team's wins and each quarterback's starts
        # for it (regular season), for the backup and resting-starters terms.
        self.wins, self.starts = {}, {}
        self.season = None

    def new_qb_value(self) -> float:
        return config.NFL_QB_ELO_LEAGUE_VALUE * config.NFL_QB_ELO_NEW_QB_SHARE

    def regular_starter(self, team: str):
        """The quarterback with the most starts for `team` this season, once
        he has config.NFL_QB_ELO_STARTER_MIN_STARTS of them; None before
        that."""
        starts = self.starts.get(team)
        if not starts:
            return None
        regular = max(starts, key=starts.get)
        return regular if starts[regular] >= config.NFL_QB_ELO_STARTER_MIN_STARTS else None

    def backup_penalty(self, team: str, qb_id, final_week: bool = False) -> float:
        """Elo points off for a quarterback standing in for his team's
        regular starter: config.NFL_QB_ELO_BACKUP_PENALTY, and in the last
        week of the regular season, for a team with
        config.NFL_QB_ELO_SITTING_MIN_WINS wins - one resting its starters
        for the playoffs - config.NFL_QB_ELO_SITTING_PENALTY more."""
        regular = self.regular_starter(team)
        if regular is None or qb_id == regular:
            return 0.0
        penalty = config.NFL_QB_ELO_BACKUP_PENALTY
        if final_week and self.wins.get(team, 0) >= config.NFL_QB_ELO_SITTING_MIN_WINS:
            penalty += config.NFL_QB_ELO_SITTING_PENALTY
        return penalty

    def qb_adjustment(self, team: str, qb_id, final_week: bool = False) -> float:
        """Elo points the listed starter adds to (or takes from) his team:
        his own rating against his team's usual level, less the backup
        penalty when he is standing in for the regular starter. 0 with no
        starter listed; a quarterback not seen before is a new one; the
        What if? panel's average starter is never a backup."""
        team_level = self.team_value.get(team, config.NFL_QB_ELO_LEAGUE_VALUE)
        if not isinstance(qb_id, str) or not qb_id:
            return 0.0
        if qb_id == AVERAGE_QB_ID:
            return self.average_qb_adjustment(team)
        return (config.NFL_QB_ELO_QB_WEIGHT * (self.qb_value.get(qb_id, self.new_qb_value()) - team_level)
                - self.backup_penalty(team, qb_id, final_week))

    def average_qb_adjustment(self, team: str) -> float:
        """The same for a league-average quarterback (the What if? panel's
        "average QB")."""
        return config.NFL_QB_ELO_QB_WEIGHT * (
            config.NFL_QB_ELO_LEAGUE_VALUE - self.team_value.get(team, config.NFL_QB_ELO_LEAGUE_VALUE))

    def gap(self, home: str, away: str, home_qb=None, away_qb=None, neutral: bool = False,
            rest_diff: float = 0.0, final_week: bool = False) -> float:
        """Home minus away in Elo points: the ratings, home field, the
        home team's extra days of rest (`rest_diff`, home minus away) and
        both starters. `final_week` marks the last week of the regular
        season, when a winning team's backup means resting starters."""
        start = config.NFL_ELO_START
        cap = config.NFL_QB_ELO_REST_CAP_DAYS
        rest = 0.0 if rest_diff is None or pd.isna(rest_diff) else float(np.clip(rest_diff, -cap, cap))
        return (self.elo.get(home, start) - self.elo.get(away, start) + (0.0 if neutral else config.NFL_QB_ELO_HOME_EDGE)
                + config.NFL_QB_ELO_REST_PER_DAY * rest
                + self.qb_adjustment(home, home_qb, final_week) - self.qb_adjustment(away, away_qb, final_week))

    def roll_season(self, season) -> None:
        """Between seasons: Elo a third of the way back to average, the
        team and defense quarterback levels a quarter of the way."""
        if self.season is not None and season != self.season:
            start, league = config.NFL_ELO_START, config.NFL_QB_ELO_LEAGUE_VALUE
            self.elo = {t: start + (r - start) * config.NFL_ELO_SEASON_KEEP for t, r in self.elo.items()}
            self.team_value = {t: v + (league - v) / 4 for t, v in self.team_value.items()}
            self.defense_value = {t: v + (league - v) / 4 for t, v in self.defense_value.items()}
            self.wins, self.starts = {}, {}
        self.season = season

    def team_table(self) -> pd.DataFrame:
        """[team, qb_elo, qb_team_value] for every team rated."""
        teams = sorted(set(self.elo) | set(self.team_value))
        return pd.DataFrame({"team": teams,
                             "qb_elo": [self.elo.get(t, config.NFL_ELO_START) for t in teams],
                             "qb_team_value": [self.team_value.get(t, config.NFL_QB_ELO_LEAGUE_VALUE) for t in teams]})


def final_regular_week(season) -> int:
    """The last week of the regular season: 18 since 2021, 17 before."""
    return 18 if int(season) >= 2021 else 17


def is_final_week(season, week, game_type="REG") -> bool:
    """Whether a game is in the last week of the regular season."""
    if pd.isna(season) or pd.isna(week) or (isinstance(game_type, str) and game_type != "REG"):
        return False
    return int(week) == final_regular_week(season)


def gap_to_probability(gap) -> np.ndarray:
    return 1.0 / (1.0 + 10.0 ** (-np.asarray(gap, dtype=float) / 400.0))


def rate(schedules_df: pd.DataFrame, weekly_df: pd.DataFrame, current_season=None, record: bool = False):
    """Runs every completed game in `schedules_df` in order and returns the
    QbEloState after the last one (rolled into `current_season` when that
    is later). `weekly_df` supplies the quarterback values; a game without
    them (a season with no weekly file) still moves the Elo, with no
    quarterback adjustment. With `record`, also returns [game_id,
    qb_elo_gap] as each game stood BEFORE it was played - the walk-forward
    predictions."""
    state, gaps = QbEloState(), []
    games = schedules_df if schedules_df is not None else pd.DataFrame()
    if not games.empty:
        games = games.dropna(subset=["home_score", "away_score"])
        order = [c for c in ("season", "week", "gameday", "game_id") if c in games.columns]
        games = games.sort_values(order)
    values = qb_game_values(weekly_df)
    by_player = values.set_index(["season", "week", "team", "player_id"])["value"].to_dict()
    by_team = values.groupby(["season", "week", "team"])["value"].sum().to_dict()
    # Who threw the most passes for each team in each game: the starter
    # when the schedule's listed one never threw a pass. The feed's
    # starters for the current season can be stale for weeks (2026: Drew
    # Lock listed for Seattle in weeks 3-6, Sam Darnold threw every pass),
    # which left the real starter's rating untouched and miscounted starts.
    top_passer = (values.sort_values("attempts").drop_duplicates(["season", "week", "team"], keep="last")
                  .set_index(["season", "week", "team"])["player_id"].to_dict()) if not values.empty else {}

    def played(listed, season, week, team):
        top = top_passer.get((season, week, team))
        if top is None or (isinstance(listed, str) and (season, week, team, listed) in by_player):
            return listed
        return top
    league = config.NFL_QB_ELO_LEAGUE_VALUE
    has = {c: c in games.columns for c in ("home_qb_id", "away_qb_id", "location", "game_type", "spread_line",
                                           "home_rest", "away_rest")}
    for row in games.itertuples(index=False):
        season, week = row.season, row.week
        home, away = nfl_totals.NFL_TEAM_ALIASES.get(row.home_team, row.home_team), nfl_totals.NFL_TEAM_ALIASES.get(row.away_team, row.away_team)
        state.roll_season(season)
        home_qb = played(row.home_qb_id if has["home_qb_id"] else None, season, week, home)
        away_qb = played(row.away_qb_id if has["away_qb_id"] else None, season, week, away)
        neutral = has["location"] and row.location == "Neutral"
        regular_season = not has["game_type"] or row.game_type == "REG"
        rest_diff = row.home_rest - row.away_rest if has["home_rest"] and has["away_rest"] else 0.0
        gap = state.gap(home, away, home_qb, away_qb, neutral, rest_diff,
                        is_final_week(season, week) and regular_season)
        if record:
            gaps.append((row.game_id, gap))
        margin = row.home_score - row.away_score
        expected = float(gap_to_probability(gap))
        result = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        winner_gap = gap if margin > 0 else -gap
        shift = config.NFL_QB_ELO_K * np.log(abs(margin) + 1) * 2.2 / (winner_gap * 0.001 + 2.2) * (result - expected)
        # What the closing line said about these two teams, which is public
        # before either plays again: the gap moves part of the way to it.
        spread = getattr(row, "spread_line", None) if has["spread_line"] else None
        if spread is not None and not pd.isna(spread):
            shift += config.NFL_QB_ELO_MARKET_PULL * (spread * config.NFL_QB_ELO_POINTS - gap) / 2
        start = config.NFL_ELO_START
        state.elo[home] = state.elo.get(home, start) + shift
        state.elo[away] = state.elo.get(away, start) - shift
        if not regular_season:
            continue
        for team, won, starter in ((home, margin > 0, home_qb), (away, margin < 0, away_qb)):
            state.wins[team] = state.wins.get(team, 0) + int(won)
            if isinstance(starter, str) and starter:
                counts = state.starts.setdefault(team, {})
                counts[starter] = counts.get(starter, 0) + 1
        for team, opponent, starter in ((home, away, home_qb), (away, home, away_qb)):
            raw = by_team.get((season, week, team))
            if raw is None:
                continue
            allowed = state.defense_value.get(opponent, league)
            level = state.team_value.get(team, league)
            state.team_value[team] = level + ((raw - (allowed - league)) - level) / config.NFL_QB_ELO_TEAM_GAMES
            state.defense_value[opponent] = allowed + (raw - allowed) / config.NFL_QB_ELO_TEAM_GAMES
            mine = by_player.get((season, week, team, starter)) if isinstance(starter, str) else None
            if mine is not None:
                rating = state.qb_value.get(starter, state.new_qb_value())
                state.qb_value[starter] = rating + ((mine - (allowed - league)) - rating) / config.NFL_QB_ELO_QB_GAMES
    if current_season is not None:
        state.roll_season(current_season)
    if record:
        return state, pd.DataFrame(gaps, columns=["game_id", "qb_elo_gap"])
    return state

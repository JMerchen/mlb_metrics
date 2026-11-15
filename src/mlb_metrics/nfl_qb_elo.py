"""A quarterback-adjusted Elo: team ratings that know who is under center.

Why (2026-11-15, scripts/research_nfl_bets.py and the user's "if there are
better models out there ... leverage what they have that we don't"): the
public nfelo model scored a log loss of 0.6197 on the 1,766 games our v4
model scored 0.6305 on. Taking nfelo apart, its team ratings alone were no
better than ours (0.6300); the whole gap was its quarterback adjustment,
the one FiveThirtyEight designed. Our own QB+ term never captured it,
because it rates the starter on his own and not against what his team has
been getting at the position. This module is that design, built from our
own data.

How it works:
  * Every quarterback game gets a VALUE from the box score
    (FiveThirtyEight's weights: completions, yards and touchdowns for,
    attempts, interceptions and sacks against, plus his rushing), less how
    much more than average the defense he faced usually allows.
  * A quarterback's rating is a running average of his values (a tenth of
    the way to each new game); a team's is the same over whoever played
    (a twentieth); a defense's is what it has allowed.
  * A team's Elo is the standard one (20 points a game, scaled by the
    margin, a third of the way back to average between seasons), except
    that each game's expectation includes the two starters' adjustments -
    3.3 Elo points for every point of value the listed starter is above
    or below his team's running level. So the team rating is the team
    with its usual quarterback play, and a backup starting, or a starter
    back, moves the game and not the rating.

Walk-forward by construction (each game is predicted from the ratings as
they stood before it; nothing is fit), 2019-2026, 1,999 games:
                             log loss   winners
    our v4 model              0.6307     1,284
    this rating alone         0.6242     1,299
    the two averaged (v5)     0.6242     1,306
    Vegas                     0.6094     1,332
The average beats v4 by 0.0065 (t 3.7) and in all 8 seasons - about a
third of the distance v4 was from Vegas - and its picks against Vegas come
out near break-even at the closing moneyline (219 bets, -3.0 units; v4's
were -12.1 on 260). The settings are FiveThirtyEight's except the home
edge (40 Elo points, not 48: home field has shrunk) and a new
quarterback's starting value (0.4 of the league's); fifteen variations all
landed within 0.002 of each other on both halves of the history, so
nothing here is tuned finely. scripts/train_nfl_qb_elo_blend.py prints
these and saves the model.
"""

import numpy as np
import pandas as pd

from mlb_metrics import config, nfl_totals

VALUE_COLUMNS = ["attempts", "completions", "passing_yards", "passing_tds", "passing_interceptions",
                 "sacks_suffered", "carries", "rushing_yards", "rushing_tds"]


def qb_game_values(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (season, week, team, passer) with at least one pass
    attempt in a regular-season game: his VALUE that day. Columns: season,
    week, team, player_id, value."""
    columns = ["season", "week", "team", "player_id", "value"]
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
    return rows.groupby(["season", "week", "team", "player_id"], as_index=False)["value"].sum()[columns]


class QbEloState:
    """The ratings after some run of games: team Elo, each team's running
    quarterback value, each defense's value allowed, and each quarterback's
    own."""

    def __init__(self):
        self.elo, self.team_value, self.defense_value, self.qb_value = {}, {}, {}, {}
        self.season = None

    def new_qb_value(self) -> float:
        return config.NFL_QB_ELO_LEAGUE_VALUE * config.NFL_QB_ELO_NEW_QB_SHARE

    def qb_adjustment(self, team: str, qb_id) -> float:
        """Elo points the listed starter adds to (or takes from) his team.
        0 with no starter listed; a quarterback not seen before is a new
        one."""
        team_level = self.team_value.get(team, config.NFL_QB_ELO_LEAGUE_VALUE)
        if not isinstance(qb_id, str) or not qb_id:
            return 0.0
        return config.NFL_QB_ELO_QB_WEIGHT * (self.qb_value.get(qb_id, self.new_qb_value()) - team_level)

    def average_qb_adjustment(self, team: str) -> float:
        """The same for a league-average quarterback (the What if? panel's
        "average QB")."""
        return config.NFL_QB_ELO_QB_WEIGHT * (
            config.NFL_QB_ELO_LEAGUE_VALUE - self.team_value.get(team, config.NFL_QB_ELO_LEAGUE_VALUE))

    def gap(self, home: str, away: str, home_qb=None, away_qb=None, neutral: bool = False) -> float:
        """Home minus away in Elo points, home field and both starters included."""
        start = config.NFL_ELO_START
        return (self.elo.get(home, start) - self.elo.get(away, start) + (0.0 if neutral else config.NFL_QB_ELO_HOME_EDGE)
                + self.qb_adjustment(home, home_qb) - self.qb_adjustment(away, away_qb))

    def roll_season(self, season) -> None:
        """Between seasons: Elo a third of the way back to average, the
        team and defense quarterback levels a quarter of the way."""
        if self.season is not None and season != self.season:
            start, league = config.NFL_ELO_START, config.NFL_QB_ELO_LEAGUE_VALUE
            self.elo = {t: start + (r - start) * config.NFL_ELO_SEASON_KEEP for t, r in self.elo.items()}
            self.team_value = {t: v + (league - v) / 4 for t, v in self.team_value.items()}
            self.defense_value = {t: v + (league - v) / 4 for t, v in self.defense_value.items()}
        self.season = season

    def team_table(self) -> pd.DataFrame:
        """[team, qb_elo, qb_team_value] for every team rated."""
        teams = sorted(set(self.elo) | set(self.team_value))
        return pd.DataFrame({"team": teams,
                             "qb_elo": [self.elo.get(t, config.NFL_ELO_START) for t in teams],
                             "qb_team_value": [self.team_value.get(t, config.NFL_QB_ELO_LEAGUE_VALUE) for t in teams]})


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
    league = config.NFL_QB_ELO_LEAGUE_VALUE
    has = {c: c in games.columns for c in ("home_qb_id", "away_qb_id", "location", "game_type")}
    for row in games.itertuples(index=False):
        season, week = row.season, row.week
        home, away = nfl_totals.NFL_TEAM_ALIASES.get(row.home_team, row.home_team), nfl_totals.NFL_TEAM_ALIASES.get(row.away_team, row.away_team)
        state.roll_season(season)
        home_qb = row.home_qb_id if has["home_qb_id"] else None
        away_qb = row.away_qb_id if has["away_qb_id"] else None
        neutral = has["location"] and row.location == "Neutral"
        gap = state.gap(home, away, home_qb, away_qb, neutral)
        if record:
            gaps.append((row.game_id, gap))
        margin = row.home_score - row.away_score
        expected = float(gap_to_probability(gap))
        result = 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
        winner_gap = gap if margin > 0 else -gap
        shift = config.NFL_ELO_K * np.log(abs(margin) + 1) * 2.2 / (winner_gap * 0.001 + 2.2) * (result - expected)
        start = config.NFL_ELO_START
        state.elo[home] = state.elo.get(home, start) + shift
        state.elo[away] = state.elo.get(away, start) - shift
        if has["game_type"] and row.game_type != "REG":
            continue
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

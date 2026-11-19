"""The last few times two NFL teams met in the same building: for each of
this week's games, the most recent completed meetings with the same team
at home, with what Vegas said (the closing spread, total and moneylines)
and how it finished.

Why (2026-11-19, user request): "before the simulation option for this
game, let's show the five most recent games of the same matchup (same home
team as well) with the vegas odds and final score". It is on the Picks
page as background for a bettor, not an input to anything: head-to-head
history was tested and adds nothing to the model (README, "Head-to-head
history: tested, not used").

The games come from every schedule file on disk (2016 on) plus
data/raw/nfl/past_games_1999_2015.csv, the same columns cut from
nflverse's games.csv for the seasons before them (closing spreads and
totals throughout; moneylines from 2006). A franchise that moved keeps
its history (nfl_totals.NFL_TEAM_ALIASES), and a game on a neutral field
is not a home game for either side, so it is left out.
"""

import os

import pandas as pd

from mlb_metrics import config, nfl_totals

OLD_GAMES_FILE = "past_games_1999_2015.csv"
GAME_COLUMNS = ["game_id", "season", "game_type", "week", "gameday", "home_team", "away_team", "home_score",
                "away_score", "location", "spread_line", "total_line", "home_moneyline", "away_moneyline"]
MEETING_COLUMNS = ["game_id", "meeting_game_id", "season", "week", "game_type", "date", "home_team", "away_team",
                   "home_score", "away_score", "spread_line", "total_line", "home_moneyline", "away_moneyline"]
MEETINGS_SHOWN = 5


def load_old_games(raw_dir: str = config.NFL_RAW_DATA_DIR) -> pd.DataFrame:
    """The seasons before the schedule files on disk (GAME_COLUMNS); empty
    when the file isn't there."""
    path = os.path.join(raw_dir, OLD_GAMES_FILE)
    if not os.path.exists(path):
        return pd.DataFrame(columns=GAME_COLUMNS)
    return pd.read_csv(path)


def past_meetings(games: pd.DataFrame, schedules: pd.DataFrame, limit: int = MEETINGS_SHOWN) -> pd.DataFrame:
    """For each game in `games` (game_id, home_team, away_team): its last
    `limit` completed meetings in `schedules` with the same home team,
    latest first, as MEETING_COLUMNS. `game_id` is the upcoming game's and
    `meeting_game_id` the past one's; teams are today's codes; spread_line
    is the points Vegas had the home team winning by (negative = the road
    team was favored). The game itself, a neutral-field game and one with
    no final score are never a meeting."""
    if games is None or games.empty or schedules is None or schedules.empty:
        return pd.DataFrame(columns=MEETING_COLUMNS)
    past = schedules.reindex(columns=GAME_COLUMNS)
    past = past[past["home_score"].notna() & past["away_score"].notna() & (past["location"].fillna("Home") != "Neutral")].copy()
    past["home"], past["away"] = nfl_totals._canonical(past["home_team"]), nfl_totals._canonical(past["away_team"])
    past = past.drop_duplicates("game_id").sort_values(["gameday", "game_id"], ascending=False)
    rows = []
    for game in games.itertuples(index=False):
        home = nfl_totals.NFL_TEAM_ALIASES.get(game.home_team, game.home_team)
        away = nfl_totals.NFL_TEAM_ALIASES.get(game.away_team, game.away_team)
        met = past[(past["home"] == home) & (past["away"] == away) & (past["game_id"] != game.game_id)].head(limit)
        for m in met.itertuples(index=False):
            rows.append({"game_id": game.game_id, "meeting_game_id": m.game_id, "season": int(m.season), "week": int(m.week),
                         "game_type": m.game_type, "date": str(m.gameday)[:10], "home_team": home, "away_team": away,
                         "home_score": int(m.home_score), "away_score": int(m.away_score), "spread_line": m.spread_line,
                         "total_line": m.total_line, "home_moneyline": m.home_moneyline, "away_moneyline": m.away_moneyline})
    return pd.DataFrame(rows, columns=MEETING_COLUMNS)


def write_matchup_history(games: pd.DataFrame, schedules: pd.DataFrame, output_dir: str,
                          raw_dir: str = config.NFL_RAW_DATA_DIR) -> pd.DataFrame:
    """Writes nfl_matchup_history.csv for the Picks page: past_meetings for
    `games` over `schedules` (the seasons on disk) and the older seasons'
    file."""
    every = pd.concat([schedules.reindex(columns=GAME_COLUMNS), load_old_games(raw_dir)], ignore_index=True) \
        if schedules is not None and not schedules.empty else load_old_games(raw_dir)
    meetings = past_meetings(games, every)
    os.makedirs(output_dir, exist_ok=True)
    meetings.to_csv(os.path.join(output_dir, "nfl_matchup_history.csv"), index=False)
    return meetings

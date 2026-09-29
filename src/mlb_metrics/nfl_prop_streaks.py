"""Season-long floors: the most a player has produced in EVERY game of
the season so far, across many stats, for the player props page.

Direct user request (2026-09-29), in two steps. First: "a section of
bets that have hit every week of that season to that point... if a tight
end has had 3+ receptions every week, that might be listed." Then the
clarification that reshaped it: "Let's say a player has had 5+ every week
but then the next week drops to 4, the screen should now show that he's
had 4+ every week... I'd like to consider yards, touchdowns (especially),
sacks, completions, ints, etc. It's just a way to see what players are
doing, I don't care about carry-over success rate as much here. This is
just a history section, not necessarily a prediction section."

So this is DESCRIPTIVE, deliberately. The first version of this module
ranked players by a model probability of clearing a line again and
filtered them to a fair-odds band; that answered a question the user did
not ask. Nothing here projects anything. Every number on the board is
something that already happened.

THE FLOOR is simply the minimum a player has recorded across his games -
if his receptions went 5, 6, 4 the board says "4+ every week", and next
week drops to "3+" the moment he posts a 3. No rounding to betting-line
increments, because this is a record, not a price.

WHAT MAKES A FLOOR WORTH SHOWING. Almost every regular has a floor above
zero in something: with three weeks played, 165 skill players had 1+
reception every week. A list of everyone would be unreadable, so the
board keeps floors that are RARE among a player's position peers -
"only 3 of 119 WRs have scored a touchdown every week". That is still
purely descriptive (it counts what other players did, and predicts
nothing), and it reads the way the request was phrased.

Rarity is measured only within the positions where the stat is part of
the job (`STREAK_STATS`), and this is a correctness point rather than a
tidy-up. Pooling every position together surfaced "Deebo Samuel: 4+
rushing yards every week - only 1 of 119 WRs" on the live board, which is
true and meaningless: it is rare among receivers only because receivers
seldom carry the ball. A floor has to be unusual among players who
actually do that thing.

"EVERY WEEK" means every game his team played, so a bye week is not a
missed week - the team did not play either. A player who sat out a game
his team played is excluded, which is the literal reading of the request
but does thin the board after an injury. A player traded mid-season is
compared against his CURRENT team's schedule and will usually drop out,
because he was not on that team for its early games; that is a known
simplification, not an oversight.

The board also records which game each player has next, so it can be
filtered to one matchup the same way the main props board can. A player
whose team is on bye is still listed - this is history, and his record
stands whether or not he plays this week.
"""

import os

import numpy as np
import pandas as pd

from mlb_metrics import config


# label -> (weekly columns summed per game, position groups where the
# stat is part of the job). Order is display order, touchdowns first
# because the request singled them out ("touchdowns (especially)").
#
# The position groups are what keep the rarity measure honest - see the
# module docstring for the "4+ rushing yards among WRs" case that forced
# them. Rushing counts QBs because a quarterback who runs every week is a
# real, bettable pattern; Rush + Rec is running backs only, because for a
# receiver it duplicates Receiving Yards row for row.
STREAK_STATS = {
    "Anytime TD (Rush + Rec)": (["rushing_tds", "receiving_tds"], ["RB", "WR", "TE", "QB"]),
    "Passing TDs": (["passing_tds"], ["QB"]),
    "Receptions": (["receptions"], ["WR", "TE", "RB"]),
    "Targets": (["targets"], ["WR", "TE", "RB"]),
    "Receiving Yards": (["receiving_yards"], ["WR", "TE", "RB"]),
    "Rushing Yards": (["rushing_yards"], ["RB", "QB"]),
    "Rush Attempts": (["carries"], ["RB", "QB"]),
    "Rush + Rec Yards": (["rushing_yards", "receiving_yards"], ["RB"]),
    "Passing Yards": (["passing_yards"], ["QB"]),
    "Completions": (["completions"], ["QB"]),
    "Pass Attempts": (["attempts"], ["QB"]),
    "Interceptions Thrown": (["passing_interceptions"], ["QB"]),
    "Sacks": (["def_sacks"], ["DL", "LB", "DB"]),
    "Tackles (Solo + Ast)": (["def_tackles_solo", "def_tackle_assists"], ["DL", "LB", "DB"]),
    "QB Hits": (["def_qb_hits"], ["DL", "LB", "DB"]),
    "Interceptions (Defense)": (["def_interceptions"], ["DL", "LB", "DB"]),
    "Passes Defended": (["def_pass_defended"], ["LB", "DB"]),
    # Kickers by POSITION, not group: the SPEC group also holds punters
    # and long snappers, who never attempt a field goal, and pooling them
    # in made "2 of 62" out of what is really "2 of ~30 kickers".
    "Field Goals Made": (["fg_made"], ["K"]),
}

STREAK_COLUMNS = [
    "player_id", "player_name", "position", "position_group", "team", "game",
    "stat", "floor", "games", "season_avg", "game_log",
    "players_at_or_above", "pool_size", "pool_group", "rarity",
]


def _format_number(value) -> str:
    """Whole numbers without a trailing .0 (receptions, yards, TDs), half
    values kept (a split sack is 0.5)."""
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def compute_season_floors(weekly_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """One row per (player, stat) for every player who appeared in every
    game his team has played this season, whatever his floor - including
    zero, because a zero is part of the peer pool rarity is measured
    against.

    `game_log` keeps the actual per-game values in week order, so a
    reader can see "5, 6, 4" rather than taking the floor on trust.
    """
    columns = [
        "player_id", "player_name", "position", "position_group", "team",
        "stat", "floor", "games", "season_avg", "game_log",
    ]
    if weekly_df.empty:
        return pd.DataFrame(columns=columns)

    current = weekly_df[weekly_df["season"] == season]
    if "season_type" in current.columns:
        current = current[current["season_type"] == "REG"]
    if current.empty:
        return pd.DataFrame(columns=columns)

    current = current.sort_values(["week"])
    team_games = current.groupby("team")["week"].nunique()
    latest = current.groupby("player_id").last()
    player_games = current.groupby("player_id")["week"].nunique()

    # Every game his CURRENT team has played - a bye is not a miss, since
    # the team did not play either. See the module docstring on traded
    # players.
    expected = latest["team"].map(team_games)
    eligible = player_games[player_games == expected].index
    scoped = current[current["player_id"].isin(eligible)]
    if scoped.empty:
        return pd.DataFrame(columns=columns)

    name_column = "player_display_name" if "player_display_name" in scoped.columns else "player_name"
    rows = []
    for stat, (stat_columns, _groups) in STREAK_STATS.items():
        present = [c for c in stat_columns if c in scoped.columns]
        if not present:
            continue
        values = scoped[["player_id", "week"]].assign(value=scoped[present].fillna(0).sum(axis=1))
        for player_id, group in values.groupby("player_id"):
            series = group.sort_values("week")["value"]
            info = latest.loc[player_id]
            rows.append({
                "player_id": player_id,
                "player_name": info.get(name_column, player_id),
                "position": info.get("position"),
                "position_group": info.get("position_group"),
                "team": info.get("team"),
                "stat": stat,
                "floor": float(series.min()),
                "games": int(len(series)),
                "season_avg": float(series.mean()),
                "game_log": ", ".join(_format_number(v) for v in series),
            })

    return pd.DataFrame(rows, columns=columns)


def rank_floors_by_rarity(floors_df: pd.DataFrame) -> pd.DataFrame:
    """Adds how many position peers matched or beat each floor, and keeps
    only floors worth showing.

    Rarity is counted within the stat's own position groups
    (`STREAK_STATS`) and is purely descriptive - it says what other
    players have done, not what anyone will do. A floor of zero is never
    shown (it is not a streak), and pools smaller than
    `config.NFL_STREAK_MIN_POOL` are skipped because "1 of 3" carries no
    meaning.
    """
    if floors_df.empty:
        return pd.DataFrame(columns=STREAK_COLUMNS)

    kept = []
    for stat, (_columns, groups) in STREAK_STATS.items():
        for group in groups:
            # A pool key matches a position GROUP (WR, DB) or, where the
            # group is too coarse, an exact POSITION (K) - see "Field
            # Goals Made" in STREAK_STATS.
            in_pool = (floors_df["position_group"] == group) | (floors_df["position"] == group)
            pool = floors_df[(floors_df["stat"] == stat) & in_pool]
            if len(pool) < config.NFL_STREAK_MIN_POOL:
                continue
            floors = pool["floor"].to_numpy()
            candidates = pool[pool["floor"] > 0].copy()
            if candidates.empty:
                continue
            candidates["players_at_or_above"] = [int((floors >= f).sum()) for f in candidates["floor"]]
            candidates["pool_size"] = len(pool)
            candidates["pool_group"] = group
            candidates["rarity"] = candidates["players_at_or_above"] / candidates["pool_size"]
            kept.append(candidates[candidates["rarity"] <= config.NFL_STREAK_MAX_RARITY])

    if not kept:
        return pd.DataFrame(columns=STREAK_COLUMNS)

    ranked = pd.concat(kept, ignore_index=True)
    stat_order = {stat: index for index, stat in enumerate(STREAK_STATS)}
    ranked["_stat_order"] = ranked["stat"].map(stat_order)
    ranked = ranked.sort_values(
        ["_stat_order", "rarity", "floor", "season_avg"], ascending=[True, True, False, False]
    )
    ranked = ranked.groupby("stat", sort=False).head(config.NFL_STREAK_PER_STAT_ROWS)
    return ranked.drop(columns=["_stat_order"]).reset_index(drop=True)


def attach_next_game(board_df: pd.DataFrame, schedule_df: pd.DataFrame) -> pd.DataFrame:
    """Adds the "AWAY @ HOME" matchup each player's team plays next, so
    the board can be filtered to one game. A team on bye gets an empty
    label and stays on the board - this is history, and it stands either
    way."""
    labelled = board_df.copy()
    if (
        labelled.empty
        or schedule_df is None
        or schedule_df.empty
        or not {"home_team", "away_team"}.issubset(schedule_df.columns)
    ):
        labelled["game"] = pd.NA
        return labelled

    games = schedule_df[["home_team", "away_team"]].drop_duplicates()
    games = games.assign(game=games["away_team"] + " @ " + games["home_team"])
    by_team = pd.concat(
        [
            games[["home_team", "game"]].rename(columns={"home_team": "team"}),
            games[["away_team", "game"]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    ).drop_duplicates("team")
    labelled = labelled.drop(columns=["game"], errors="ignore")
    return labelled.merge(by_team, on="team", how="left")


def build_streak_board(weekly_df: pd.DataFrame, season: int, schedule_df: pd.DataFrame = None) -> pd.DataFrame:
    """The whole history board: season floors, kept where they are rare
    among position peers, labelled with each player's next game."""
    ranked = rank_floors_by_rarity(compute_season_floors(weekly_df, season))
    if ranked.empty:
        return pd.DataFrame(columns=STREAK_COLUMNS)
    board = attach_next_game(ranked, schedule_df)
    for column in STREAK_COLUMNS:
        if column not in board.columns:
            board[column] = pd.NA
    return board[STREAK_COLUMNS].reset_index(drop=True)


def write_streak_board_csv(
    weekly_df: pd.DataFrame, season: int, schedule_df: pd.DataFrame, output_path: str
) -> pd.DataFrame:
    """Writes the history board, leaving any prior file in place when
    nothing qualifies - the same "don't let one quiet week erase real
    history" posture the props board already takes."""
    board = build_streak_board(weekly_df, season, schedule_df)
    if board.empty:
        print("No season-long floors qualified this week - writing nothing, leaving the prior CSV in place.")
        return board

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    board.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(board)} rows across {board['stat'].nunique()} stats)")
    return board

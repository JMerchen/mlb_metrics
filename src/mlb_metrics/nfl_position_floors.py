"""Season floors by position: for the 32 most-used players at QB, RB, WR,
TE, DL and DB, the lowest number they have posted in every game this
season, stat by stat - with games they were hurt in left out.

Direct user request (2026-09-29), replacing the rarity-ranked "Season
Floors" list shipped the same day: "a table with selectable tabs by
position with the top 32 at that position by snap count... For each
we'd have their stats as columns with their floor value... For each, a
game where they get injured shouldn't count toward their floor, but
otherwise their floor is shown, even if it's zero."

A history section. Nothing here predicts anything.

THE GAMES COME FROM SNAP COUNTS, NOT FROM THE STATS TABLE, and this is
the difference between a right and a wrong floor. The weekly stats table
only has a row for a player in a game where he recorded something, so a
receiver who played 40 snaps and caught nothing is simply absent from it
that week. Measured on 2025: 6.9% of real game appearances by WRs, TEs
and RBs with 10+ snaps have no stats row at all. Taking the minimum over
the stats table would silently skip exactly those zero games and report
a floor the player never actually held - the opposite of "shown, even if
it's zero". So a game counts whenever the player took snaps on his side
of the ball, and any stat the stats table does not record for that game
is a real zero.

WHAT COUNTS AS AN INJURY GAME, and why it is not just "played few snaps".
The only in-game signal available is snap share, and on its own it does
not identify injuries. Measured over 2025 regulars: a game where a player
took under half his usual share happened 377 times, and only 37% of those
were followed by the player being injured (missing his team's next game,
or appearing on the next injury report). The rest were blowouts, benchings
and rotation changes - and they look the same in the snap data. Players
leaving hurt played a median 24% of their usual snaps; starters pulled in
blowouts played 25%. No threshold separates them: every cut from 50% down
to 20% stays near 40% injuries.

So an injury game here is one where BOTH happened: the player's snap
share collapsed below `config.NFL_FLOOR_INJURY_SNAP_RATIO` of his own
usual share, AND the roster listed him as inactive or on reserve for his
team's next game (`config.NFL_FLOOR_INJURY_STATUSES`). The confirmation
is the roster status, not merely failing to play, and that distinction
was measured rather than assumed. A first version confirmed on "took no
snaps in the next game", which excluded 447 of 2025's 1,290 collapses -
but 201 of those players were still on the active roster that next week
(174), back on the practice squad (22) or released (5). Healthy players,
in other words, and prominently backup quarterbacks: 28 relief
appearances by the likes of Cooper Rush and Davis Mills were being
discarded as injuries, because a backup who mops up one week and does
not play the next looks exactly like an injury if all you check is
snaps. The roster-status rule is a strict subset of that one - it never
excludes a game the older rule kept - and it keeps real exits such as
C.J. Stroud's (inactive the next three weeks) and Daniel Jones's (placed
on reserve). The weekly injury report was also measured as a
confirmation and added only 3 percentage points, so it is not fetched.

A blowout where a starter sat the fourth quarter therefore still counts,
and so does a benching. That is deliberate: the request excluded injuries
specifically, and said everything else should show "even if it's zero".

THE MOST RECENT GAME CANNOT BE CONFIRMED YET - the team's next game has
not been played - and the board runs before the week's injury report is
published. A player who left his latest game early is therefore COUNTED
and flagged, not excluded, because most such games turn out not to be
injuries and excluding them would hide real low games. If he misses the
next game, the following week's run excludes it automatically.

TOP 32 is by season snaps on the player's side of the ball, within the
tab's positions. The tabs follow the request exactly: there is no LB tab,
which means edge rushers listed as outside linebackers - several of the
league's best pass rushers - do not appear under DL.
"""

import os

import numpy as np
import pandas as pd

from mlb_metrics import config


# tab -> (snap-count positions in it, which side's snaps rank and define a
# game, {column key: (label, weekly columns summed per game)}). Stats are
# the request's own lists, in the request's own order.
POSITION_TABS = {
    "QB": (["QB"], "offense", {
        "completions": ("Completions", ["completions"]),
        "passing_tds": ("Pass TDs", ["passing_tds"]),
        "passing_yards": ("Pass Yds", ["passing_yards"]),
        "interceptions_thrown": ("INTs", ["passing_interceptions"]),
        "anytime_td": ("Anytime TD", ["rushing_tds", "receiving_tds"]),
        "rushing_yards": ("Rush Yds", ["rushing_yards"]),
    }),
    "RB": (["RB", "HB", "FB"], "offense", {
        "receptions": ("Receptions", ["receptions"]),
        "receiving_yards": ("Rec Yds", ["receiving_yards"]),
        "rushing_yards": ("Rush Yds", ["rushing_yards"]),
        "anytime_td": ("Anytime TD", ["rushing_tds", "receiving_tds"]),
    }),
    "WR": (["WR"], "offense", {
        "receptions": ("Receptions", ["receptions"]),
        "receiving_yards": ("Rec Yds", ["receiving_yards"]),
        "rushing_yards": ("Rush Yds", ["rushing_yards"]),
        "anytime_td": ("Anytime TD", ["rushing_tds", "receiving_tds"]),
    }),
    "TE": (["TE"], "offense", {
        "receptions": ("Receptions", ["receptions"]),
        "receiving_yards": ("Rec Yds", ["receiving_yards"]),
        "rushing_yards": ("Rush Yds", ["rushing_yards"]),
        "anytime_td": ("Anytime TD", ["rushing_tds", "receiving_tds"]),
    }),
    "DL": (["DE", "DT", "NT", "DL"], "defense", {
        "tackles": ("Tackles", ["def_tackles_solo", "def_tackle_assists"]),
        "sacks": ("Sacks", ["def_sacks"]),
        "def_interceptions": ("INTs", ["def_interceptions"]),
    }),
    "DB": (["CB", "S", "FS", "SS", "DB"], "defense", {
        "tackles": ("Tackles", ["def_tackles_solo", "def_tackle_assists"]),
        "sacks": ("Sacks", ["def_sacks"]),
        "def_interceptions": ("INTs", ["def_interceptions"]),
    }),
}

ALL_STAT_KEYS = list(dict.fromkeys(key for _, _, stats in POSITION_TABS.values() for key in stats))

BASE_COLUMNS = [
    "tab", "rank", "player_id", "player_name", "position", "team", "game",
    "snaps", "games_played", "games_counted", "excluded_weeks", "flagged_week",
]


def _format_value(value) -> str:
    value = float(value)
    return str(int(value)) if value.is_integer() else f"{value:g}"


def player_games(snap_counts_df: pd.DataFrame, rosters_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """One row per (player, game) in which the player took snaps on his
    side of the ball, with his share of that side's snaps and the tab he
    belongs to.

    The game list is built from snap counts on purpose - see the module
    docstring for the 6.9% of real appearances the stats table has no
    row for. `snap_counts` is keyed by pfr id, so it is crossed to the
    stats table's gsis id through the roster's `pfr_id`/`gsis_id`, the
    same crosswalk nfl_player_props.compute_current_season_snap_share
    already uses.
    """
    columns = ["player_id", "player_name", "position", "tab", "side", "team", "week", "snaps", "share"]
    if snap_counts_df is None or snap_counts_df.empty:
        return pd.DataFrame(columns=columns)

    snaps = snap_counts_df[snap_counts_df["season"] == season]
    if "game_type" in snaps.columns:
        snaps = snaps[snaps["game_type"] == "REG"]
    if snaps.empty:
        return pd.DataFrame(columns=columns)

    position_to_tab = {
        position: (tab, side)
        for tab, (positions, side, _stats) in POSITION_TABS.items()
        for position in positions
    }
    snaps = snaps[snaps["position"].isin(position_to_tab)].copy()
    snaps["tab"] = snaps["position"].map(lambda p: position_to_tab[p][0])
    snaps["side"] = snaps["position"].map(lambda p: position_to_tab[p][1])
    snaps["snaps"] = np.where(snaps["side"] == "offense", snaps["offense_snaps"], snaps["defense_snaps"])
    snaps["share"] = np.where(snaps["side"] == "offense", snaps["offense_pct"], snaps["defense_pct"])
    snaps = snaps[snaps["snaps"].fillna(0) > 0]

    season_rosters = rosters_df[rosters_df["season"] == season] if "season" in rosters_df.columns else rosters_df
    crosswalk = season_rosters.dropna(subset=["pfr_id"]).drop_duplicates("pfr_id")[["pfr_id", "gsis_id"]]
    snaps = snaps.merge(crosswalk, left_on="pfr_player_id", right_on="pfr_id", how="inner")

    return snaps.rename(columns={"gsis_id": "player_id", "player": "player_name"})[columns]


def flag_injury_games(games_df: pd.DataFrame, rosters_df: pd.DataFrame = None) -> pd.DataFrame:
    """Adds `excluded` (a confirmed injury exit: snap share collapsed AND
    the roster had him inactive or on reserve for his team's next game)
    and `flagged` (a collapse in his team's most recent game, which cannot
    be confirmed yet and so is counted).

    "His team's next game" is the team's next week with any snaps, so a
    bye is skipped over rather than read as a missed game. Without a
    roster to read statuses from, nothing is excluded - an unconfirmable
    exit is counted, never guessed at.
    """
    if games_df.empty:
        return games_df.assign(collapsed=[], excluded=[], flagged=[])

    games = games_df.copy()
    usual = games.groupby("player_id")["share"].transform("median")
    games["collapsed"] = games["share"] < config.NFL_FLOOR_INJURY_SNAP_RATIO * usual

    team_weeks = games[["team", "week"]].drop_duplicates().sort_values(["team", "week"])
    team_weeks["team_next_week"] = team_weeks.groupby("team")["week"].shift(-1)
    games = games.merge(team_weeks, on=["team", "week"], how="left")

    if rosters_df is not None and not rosters_df.empty and "status" in rosters_df.columns:
        statuses = rosters_df[["gsis_id", "week", "status"]].drop_duplicates(["gsis_id", "week"]).rename(
            columns={"gsis_id": "player_id", "week": "team_next_week", "status": "next_status"}
        )
        games = games.merge(statuses, on=["player_id", "team_next_week"], how="left")
    else:
        games["next_status"] = pd.NA

    has_next = games["team_next_week"].notna()
    games["excluded"] = games["collapsed"] & games["next_status"].isin(config.NFL_FLOOR_INJURY_STATUSES)
    games["flagged"] = games["collapsed"] & ~has_next
    return games


def compute_position_floors(
    snap_counts_df: pd.DataFrame,
    rosters_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    season: int,
    schedule_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """One row per player: the top `config.NFL_FLOOR_TOP_N` at each tab by
    season snaps, with a floor and a game log for every stat in that tab.

    A floor is the minimum over COUNTED games - every game he played
    except confirmed injury exits - and is a real zero whenever he had a
    zero. If every one of his games was an injury exit there is nothing to
    take a floor over, and the stat is left empty rather than invented.
    """
    season_rosters = (
        rosters_df[rosters_df["season"] == season]
        if rosters_df is not None and not rosters_df.empty and "season" in rosters_df.columns
        else rosters_df
    )
    games = flag_injury_games(player_games(snap_counts_df, rosters_df, season), season_rosters)
    output_columns = BASE_COLUMNS + [c for key in ALL_STAT_KEYS for c in (key, f"{key}_log")]
    if games.empty:
        return pd.DataFrame(columns=output_columns)

    weekly = weekly_df[weekly_df["season"] == season] if weekly_df is not None and not weekly_df.empty else pd.DataFrame()
    if not weekly.empty and "season_type" in weekly.columns:
        weekly = weekly[weekly["season_type"] == "REG"]

    stat_columns = sorted({c for _, _, stats in POSITION_TABS.values() for _, cols in stats.values() for c in cols})
    if weekly.empty:
        per_game = pd.DataFrame(columns=["player_id", "week"] + stat_columns)
    else:
        present = [c for c in stat_columns if c in weekly.columns]
        per_game = weekly.groupby(["player_id", "week"], as_index=False)[present].sum()
    games = games.merge(per_game, on=["player_id", "week"], how="left")
    for column in stat_columns:
        # No stats row for a game he played in is a real zero, not a gap.
        games[column] = games[column].fillna(0.0) if column in games.columns else 0.0

    next_game = _next_game_by_team(schedule_df)
    rows = []
    for tab, (_positions, _side, stats) in POSITION_TABS.items():
        tab_games = games[games["tab"] == tab]
        if tab_games.empty:
            continue
        totals = tab_games.groupby("player_id")["snaps"].sum().sort_values(ascending=False)
        for rank, player_id in enumerate(totals.head(config.NFL_FLOOR_TOP_N).index, start=1):
            mine = tab_games[tab_games["player_id"] == player_id].sort_values("week")
            counted = mine[~mine["excluded"]]
            latest = mine.iloc[-1]
            row = {
                "tab": tab,
                "rank": rank,
                "player_id": player_id,
                "player_name": latest["player_name"],
                "position": latest["position"],
                "team": latest["team"],
                "game": next_game.get(latest["team"], pd.NA),
                "snaps": int(totals[player_id]),
                "games_played": int(len(mine)),
                "games_counted": int(len(counted)),
                "excluded_weeks": ", ".join(str(int(w)) for w in mine.loc[mine["excluded"], "week"]),
                "flagged_week": ", ".join(str(int(w)) for w in mine.loc[mine["flagged"], "week"]),
            }
            for key, (_label, columns) in stats.items():
                values = mine[columns].sum(axis=1)
                row[key] = float(values[~mine["excluded"]].min()) if len(counted) else np.nan
                row[f"{key}_log"] = ", ".join(
                    f"[{_format_value(v)}]" if excluded else (f"{_format_value(v)}?" if flagged else _format_value(v))
                    for v, excluded, flagged in zip(values, mine["excluded"], mine["flagged"])
                )
            rows.append(row)

    board = pd.DataFrame(rows)
    for column in output_columns:
        if column not in board.columns:
            board[column] = pd.NA
    return board[output_columns]


def _next_game_by_team(schedule_df: pd.DataFrame) -> dict:
    """team -> "AWAY @ HOME" for the week being previewed; a team on bye
    has no entry."""
    if schedule_df is None or schedule_df.empty or not {"home_team", "away_team"}.issubset(schedule_df.columns):
        return {}
    games = {}
    for home, away in schedule_df[["home_team", "away_team"]].itertuples(index=False):
        label = f"{away} @ {home}"
        games.setdefault(home, label)
        games.setdefault(away, label)
    return games


def write_position_floors_csv(
    snap_counts_df: pd.DataFrame,
    rosters_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    season: int,
    schedule_df: pd.DataFrame,
    output_path: str,
) -> pd.DataFrame:
    """Writes the board, leaving any prior file in place when there is
    nothing to write - the same "don't let one quiet week erase real
    history" posture every other board here takes."""
    board = compute_position_floors(snap_counts_df, rosters_df, weekly_df, season, schedule_df)
    if board.empty:
        print("No position floors to write this week - leaving the prior CSV in place.")
        return board

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    board.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(board)} players across {board['tab'].nunique()} positions)")
    return board

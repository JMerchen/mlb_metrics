"""The NFL Stats tab's data: every team's schedule and results, team
season stats, and players' season views.

Team schedule (team_schedule): one row per team per week of the regular
season - opponent, home or away, final score and result once played,
and a "BYE" row for the week off.

Team stats (team_stats): record, points for and against, and EPA per play
on offense and defense (overall, passing, rushing) plus turnovers, from
play-by-play.

Player views (player_views): per position, a handful of stats blended
across recent games the way WAVE is - the season so far, the last 8 and
the last 4 games (config.NFL_SKILL_WINDOWS weights), a window a player has
no games in carrying his longer one. Each also gets a "vs. defenses faced"
version: in every game, what that defense allowed in the stat to the same
position all season, against the league's average, says how much tougher
or easier his matchup was. Rate stats are scaled by that ratio and EPA
stats shifted by the difference (EPA can be negative, so it can't be
scaled), blended over the same windows and centred so the average player
moves by nothing.
"""

import numpy as np
import pandas as pd

from . import config

# Per position: (column, label, numerator, denominator or None for per
# game, "ratio" or "epa" adjustment).
PLAYER_STATS = {
    "QB": [
        ("pass_ypg", "Passing yards a game", "passing_yards", None, "ratio"),
        ("ypa", "Yards per attempt", "passing_yards", "attempts", "ratio"),
        ("comp_pct", "Completion %", "completions", "attempts", "ratio"),
        ("epa_db", "EPA per dropback", "passing_epa", "dropbacks", "epa"),
        ("td_rate", "TD rate", "passing_tds", "attempts", "ratio"),
        ("int_rate", "INT rate", "passing_interceptions", "attempts", "ratio"),
        ("rush_ypg", "Rushing yards a game", "rushing_yards", None, "ratio"),
    ],
    "RB": [
        ("rush_ypg", "Rushing yards a game", "rushing_yards", None, "ratio"),
        ("carries_pg", "Carries a game", "carries", None, "ratio"),
        ("ypc", "Yards per carry", "rushing_yards", "carries", "ratio"),
        ("epa_carry", "EPA per carry", "rushing_epa", "carries", "epa"),
        ("rec_pg", "Catches a game", "receptions", None, "ratio"),
        ("rec_ypg", "Receiving yards a game", "receiving_yards", None, "ratio"),
    ],
    "WR": [
        ("targets_pg", "Targets a game", "targets", None, "ratio"),
        ("rec_pg", "Catches a game", "receptions", None, "ratio"),
        ("rec_ypg", "Receiving yards a game", "receiving_yards", None, "ratio"),
        ("ypt", "Yards per target", "receiving_yards", "targets", "ratio"),
        ("epa_target", "EPA per target", "receiving_epa", "targets", "epa"),
        ("target_share", "Target share", "target_share", None, "ratio"),
    ],
}
PLAYER_STATS["TE"] = PLAYER_STATS["WR"]

# Who makes the table: at least this many games and this much volume a game.
QUALIFY = {"QB": ("attempts", 15), "RB": ("carries", 6), "WR": ("targets", 3), "TE": ("targets", 2.5)}
MIN_GAMES = config.NFL_SKILL_MIN_GAMES


def team_schedule(schedules: pd.DataFrame) -> pd.DataFrame:
    """One row per team per regular-season week, oldest first: [season,
    week, team, date, opponent, home, points_for, points_against, result,
    margin]. result is W/L/T once played, "" before, and BYE on the bye
    week (no opponent)."""
    games = schedules[schedules["game_type"] == "REG"]
    home = games.assign(team=games["home_team"], opponent=games["away_team"], home=1,
                        points_for=games["home_score"], points_against=games["away_score"])
    away = games.assign(team=games["away_team"], opponent=games["home_team"], home=0,
                        points_for=games["away_score"], points_against=games["home_score"])
    rows = pd.concat([home, away], ignore_index=True)[
        ["season", "week", "team", "gameday", "opponent", "home", "points_for", "points_against"]
    ].rename(columns={"gameday": "date"})
    played = rows["points_for"].notna() & rows["points_against"].notna()
    rows["margin"] = (rows["points_for"] - rows["points_against"]).where(played)
    rows["result"] = np.select([~played, rows["margin"] > 0, rows["margin"] < 0], ["", "W", "L"], default="T")

    weeks = sorted(games["week"].unique())
    byes = []
    for (season, team), team_rows in rows.groupby(["season", "team"]):
        for week in sorted(set(weeks) - set(team_rows["week"])):
            byes.append({"season": season, "week": week, "team": team, "result": "BYE", "opponent": "", "home": np.nan})
    if byes:
        rows = pd.concat([rows, pd.DataFrame(byes)], ignore_index=True)
    return rows.sort_values(["team", "week"]).reset_index(drop=True)


def team_stats(schedule_rows: pd.DataFrame, pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per team: games, wins, losses, ties, points for/against a
    game, point differential a game, and EPA per play on offense and
    defense (overall / pass / rush), turnovers committed and forced a game."""
    played = schedule_rows[schedule_rows["result"].isin(["W", "L", "T"])]
    record = played.groupby("team").agg(
        games=("result", "size"), wins=("result", lambda r: int((r == "W").sum())),
        losses=("result", lambda r: int((r == "L").sum())), ties=("result", lambda r: int((r == "T").sum())),
        points_for=("points_for", "sum"), points_against=("points_against", "sum"),
    )
    record["ppg"] = record["points_for"] / record["games"]
    record["papg"] = record["points_against"] / record["games"]
    record["point_diff_pg"] = record["ppg"] - record["papg"]

    plays = pbp[(pbp["season_type"] == "REG") & pbp["epa"].notna() & (pbp["pass"].eq(1) | pbp["rush"].eq(1))]
    plays = plays.assign(turnover=plays["interception"].fillna(0) + plays["fumble_lost"].fillna(0))

    def side(group_col, prefix):
        g = plays.groupby(group_col)
        out = pd.DataFrame({
            f"{prefix}_epa_play": g["epa"].mean(),
            f"{prefix}_pass_epa": plays[plays["pass"] == 1].groupby(group_col)["epa"].mean(),
            f"{prefix}_rush_epa": plays[plays["rush"] == 1].groupby(group_col)["epa"].mean(),
            f"{prefix}_success": g["success"].mean(),
            f"{prefix}_turnovers": g["turnover"].sum(),
        })
        out.index.name = "team"
        return out

    stats = record.join(side("posteam", "off"), how="left").join(side("defteam", "def"), how="left")
    stats["giveaways_pg"] = stats.pop("off_turnovers") / stats["games"]
    stats["takeaways_pg"] = stats.pop("def_turnovers") / stats["games"]
    stats["net_epa_play"] = stats["off_epa_play"] - stats["def_epa_play"]
    return stats.reset_index()


def _player_games(weekly: pd.DataFrame, position: str) -> pd.DataFrame:
    rows = weekly[(weekly["season_type"] == "REG") & (weekly["position"] == position)].copy()
    rows["dropbacks"] = rows["attempts"].fillna(0) + rows["sacks_suffered"].fillna(0)
    return rows.sort_values(["player_id", "week"])


def _allowed(weekly: pd.DataFrame, position: str) -> tuple[pd.DataFrame, dict]:
    """What each defense allowed to `position` per stat (per game, or per
    unit of the stat's denominator) and the league's average."""
    rows = _player_games(weekly, position)
    per_game = rows.groupby(["opponent_team", "game_id"]).sum(numeric_only=True)
    allowed, league = {}, {}
    for column, _, num, den, _ in PLAYER_STATS[position]:
        if den is None:
            by_def = per_game.groupby("opponent_team")[num].mean()
            league[column] = per_game[num].mean()
        else:
            sums = per_game.groupby("opponent_team")[[num, den]].sum()
            by_def = sums[num] / sums[den].replace(0, np.nan)
            league[column] = per_game[num].sum() / max(per_game[den].sum(), 1e-9)
        allowed[column] = by_def
    return pd.DataFrame(allowed), league


def _blend(rows: pd.DataFrame, value_of) -> pd.Series:
    """Per player: value_of(window rows) per window, blended with the
    window weights (longest first), a window with no games carrying the
    longer one."""
    windows = sorted(config.NFL_SKILL_WINDOWS, key=lambda w: float("-inf") if w[0] is None else -w[0])
    total, previous = None, None
    for games_back, weight in windows:
        window = rows if games_back is None else rows.groupby("player_id").tail(games_back)
        value = value_of(window)
        if previous is not None:
            value = value.reindex(previous.index.union(value.index)).fillna(previous)
        previous = value
        total = value * weight if total is None else total.add(value * weight, fill_value=0)
    return total / sum(weight for _, weight in windows)


def player_views(weekly: pd.DataFrame, rosters: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per qualified QB/RB/WR/TE: id, name, team, position,
    headshot, games, then for each of his position's PLAYER_STATS the
    blended value and its `_adj` "vs. defenses faced" version."""
    frames = []
    for position in PLAYER_STATS:
        rows = _player_games(weekly, position)
        if rows.empty:
            continue
        allowed, league = _allowed(weekly, position)
        rows = rows.join(allowed.add_suffix("__allowed"), on="opponent_team")
        out = rows.groupby("player_id").agg(
            player_name=("player_display_name", "last"), team=("team", "last"),
            headshot_url=("headshot_url", "last"), games=("week", "size"),
        )
        volume_col, minimum = QUALIFY[position]
        out["volume_pg"] = rows.groupby("player_id")[volume_col].mean()
        for column, _, num, den, kind in PLAYER_STATS[position]:
            if den is None:
                out[column] = _blend(rows, lambda w, n=num: w.groupby("player_id")[n].mean())
            else:
                out[column] = _blend(rows, lambda w, n=num, d=den: w.groupby("player_id")[n].sum()
                                     / w.groupby("player_id")[d].sum().replace(0, np.nan))
            weight = rows[den] if den else pd.Series(1.0, index=rows.index)
            lg, opp = league[column], rows[f"{column}__allowed"]
            if kind == "epa":
                rows["_shift"] = (lg - opp) * weight
            else:
                rows["_shift"] = (lg / opp.replace(0, np.nan)) * weight
            rows["_weight"] = weight.where(opp.notna(), 0)
            rows["_shift"] = rows["_shift"].where(opp.notna(), 0)
            factor = _blend(rows, lambda w: w.groupby("player_id")["_shift"].sum()
                            / w.groupby("player_id")["_weight"].sum().replace(0, np.nan))
            default = 0.0 if kind == "epa" else 1.0
            factor = factor.reindex(out.index).fillna(default)
            # Centre on the average player, weighted by his volume.
            vol = out["volume_pg"] * out["games"]
            centre = np.average(factor, weights=vol) if vol.sum() > 0 else default
            if kind == "epa":
                out[f"{column}_adj"] = out[column] + (factor - centre)
            else:
                out[f"{column}_adj"] = out[column] * factor / centre
        out = out[(out["games"] >= MIN_GAMES) & (out["volume_pg"] >= minimum)]
        out["position"] = position
        frames.append(out.reset_index())
    if not frames:
        return pd.DataFrame()
    views = pd.concat(frames, ignore_index=True)
    return views.drop(columns=["volume_pg"])

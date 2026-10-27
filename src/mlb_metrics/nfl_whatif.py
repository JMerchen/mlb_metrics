"""What-if inputs for the NFL page's game picks: swap a team's starting QB
or mark its regulars out, and the page recomputes the win probability.

For every game not yet played this writes the probability shown on the
page (after the market tie-break and calibration), each team's QB choices
(the listed starter, his teammates who have thrown a pass for the team
this season, the QBs on its latest roster and an average starter) rated as
QB+, the model's log-odds for every home/away QB pairing - the page shifts
the shown probability by the change from the listed pairing - and each
team's regulars by position group, with their recent snap shares.

Regulars follow the player, not the team: his snap share over his own last
LOOKBACK_GAMES games on the active roster of whatever team he was on,
listed under the team on his latest weekly roster (nflverse's, fetched by
every weekly run) - so a player traded or signed mid-season shows up with
his new team, and one traded away, cut or placed on injured reserve drops
off.

Absences are in unit terms: each team's run and pass offense and defense
as a percent of normal yards per play. A player out lowers the units his
position touches by a default (GROUP_UNITS' comment), times his snap
share; on the page any player's or unit's percent can be typed over, and
the change in yards becomes points and then win chance. Weather isn't an input: wind, cold and domes moved
neither results nor Vegas's sides beyond the model (it changes how many
points get scored, not who wins).
"""

import json
import math
import os

import pandas as pd

from mlb_metrics import config, ml_models, nfl_game_picks, nfl_totals

# Position groups for snap counts (offense or defense snaps).
GROUPS = {
    "RB": ("offense", {"RB", "FB"}), "WR": ("offense", {"WR"}), "TE": ("offense", {"TE"}),
    "OL": ("offense", {"T", "G", "C", "OT", "OG", "OL"}),
    "DL": ("defense", {"DE", "DT", "NT", "DL"}), "LB": ("defense", {"LB", "OLB", "ILB", "MLB"}),
    "CB": ("defense", {"CB"}), "S": ("defense", {"S", "FS", "SS", "DB"}),
}
# A regular: at least this share of his side's snaps over his last
# LOOKBACK_GAMES games (0 for a game he missed) - what the injury values
# were measured on. The page lists anyone at LISTED_SHARE or more (a
# rotational player or a new arrival still working in), each weighted by
# his share.
LOOKBACK_GAMES = 4
REGULAR_SHARE = 0.5
LISTED_SHARE = 0.3

# Home-win log-odds per regular out (home minus away), times his snap
# share, on top of the model - scripts/backtest_nfl_injury_impact.py,
# 2018-2026. What results showed where they were clear (p < .05: receivers,
# offensive line, cornerbacks); elsewhere results were too noisy to use
# (running backs came out +0.19 +- 0.21) and the value is what Vegas's
# closing line charged for the same absences, measured far more precisely.
# Losing several at once cost about the sum of each - no extra.
INJURY_LOG_ODDS = {"WR": -0.25, "OL": -0.19, "CB": -0.19, "DL": -0.13, "RB": -0.13, "TE": -0.07, "LB": -0.06, "S": -0.04}

# The panel rates each team's units in percent of normal yards per play
# (defense: 90% allows 10% more). A player out lowers the units his group
# touches; the default is the drop that costs what INJURY_LOG_ODDS says an
# absence at his position costs, through POINTS_PER_YARD and
# LOGODDS_PER_POINT (scripts/backtest_nfl_unit_absences.py: all points over
# all scrimmage yards on runs and dropbacks, 2016-2026; Vegas's closing
# moneyline against its spread). Measured on yards directly, single
# absences barely register (0-3%, mostly noise), and a player's own games
# missed predict his unit's next game worse than his position does at any
# blend - so they're shown for reference, not used.
GROUP_UNITS = {
    "RB": ["run_offense"], "WR": ["pass_offense"], "TE": ["pass_offense", "run_offense"],
    "OL": ["pass_offense", "run_offense"], "DL": ["run_defense", "pass_defense"],
    "LB": ["run_defense", "pass_defense"], "CB": ["pass_defense"], "S": ["pass_defense"],
}
POINTS_PER_YARD = 0.0696
LOGODDS_PER_POINT = 0.137
UNIT_PRIOR_GAMES = 8

AVERAGE_QB_ID = "average"

# Units the panel rates in percent of normal yards per play: each play type
# (column of yards, column of plays in unit_game_yards) on offense and on
# the defense facing it, and the position groups that touch each.
UNIT_PLAYS = {"run": ("run_yards", "runs"), "pass": ("dropback_yards", "dropbacks")}
UNIT_GROUPS = {
    "run_offense": ["OL", "RB", "TE", "WR"], "pass_offense": ["OL", "WR", "TE", "RB"],
    "run_defense": ["DL", "LB", "S", "CB"], "pass_defense": ["CB", "S", "LB", "DL"],
}
YARDS_PBP_COLUMNS = ["game_id", "season", "week", "season_type", "posteam", "defteam", "yards_gained",
                     "rush", "qb_scramble", "qb_dropback"]


def unit_game_yards(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (regular-season game, offense): yards and plays on
    designed runs and on dropbacks (sacks and scrambles included).
    Columns: game_id, season, week, team, opp, run_yards, runs,
    dropback_yards, dropbacks."""
    columns = ["game_id", "season", "week", "team", "opp", "run_yards", "runs", "dropback_yards", "dropbacks"]
    if pbp is None or pbp.empty or not set(YARDS_PBP_COLUMNS) <= set(pbp.columns):
        return pd.DataFrame(columns=columns)
    plays = pbp[(pbp["season_type"] == "REG") & pbp["posteam"].notna() & pbp["yards_gained"].notna()].copy()
    run = (plays["rush"] == 1) & (plays["qb_scramble"] != 1)
    dropback = plays["qb_dropback"] == 1
    plays["run_yards"] = plays["yards_gained"].where(run, 0.0)
    plays["runs"] = run.astype(float)
    plays["dropback_yards"] = plays["yards_gained"].where(dropback, 0.0)
    plays["dropbacks"] = dropback.astype(float)
    return (plays.groupby(["game_id", "season", "week", "posteam", "defteam"])[columns[5:]].sum()
            .reset_index().rename(columns={"posteam": "team", "defteam": "opp"})[columns])
def unit_ratios(yards: pd.DataFrame) -> pd.DataFrame:
    """`yards` (unit_game_yards, teams canonical) with each game's run and
    pass efficiency: actual yards per play over expected (the offense's
    previous UNIT_PRIOR_GAMES games times the defense's previous allowed,
    over the league's that season) - `run_ratio`, `pass_ratio`."""
    y = yards.sort_values(["season", "week"]).copy()
    for kind, (yds, n) in UNIT_PLAYS.items():
        league = y.groupby("season")[yds].transform("sum") / y.groupby("season")[n].transform("sum")

        def prior(key):
            g = y.groupby(key)
            total = g[yds].transform(lambda x: x.shift(1).rolling(UNIT_PRIOR_GAMES, min_periods=3).sum())
            count = g[n].transform(lambda x: x.shift(1).rolling(UNIT_PRIOR_GAMES, min_periods=3).sum())
            return total / count

        y[f"{kind}_ratio"] = (y[yds] / y[n].replace(0, float("nan"))) / (prior("team") * prior("opp") / league)
    return y


def team_units(yards: pd.DataFrame) -> dict:
    """{team: {unit: {"ypp", "plays"}}} over each team's last
    UNIT_PRIOR_GAMES games: yards per play and plays a game on offense, and
    allowed on defense."""
    y = yards.sort_values(["season", "week"])
    out = {}
    for team in set(y["team"]) | set(y["opp"]):
        units = {}
        for side, key in (("offense", "team"), ("defense", "opp")):
            rows = y[y[key] == team].tail(UNIT_PRIOR_GAMES)
            for kind, (yds, n) in UNIT_PLAYS.items():
                if rows.empty or rows[n].sum() == 0:
                    continue
                units[f"{kind}_{side}"] = {"ypp": round(float(rows[yds].sum() / rows[n].sum()), 3),
                                           "plays": round(float(rows[n].sum() / len(rows)), 2)}
        out[team] = units
    return out


def group_default_pct(yards: pd.DataFrame) -> dict:
    """{group: unit percent with one every-down regular out} - the drop in
    each unit his group touches (the same for each) that costs
    INJURY_LOG_ODDS at a league-average team's yards a game."""
    games = len(yards)
    per_game = {f"{kind}_{side}": yards[yds].sum() / games
                for kind, (yds, _) in UNIT_PLAYS.items() for side in ("offense", "defense")}
    out = {}
    for group, units in GROUP_UNITS.items():
        yards_at_stake = sum(per_game[u] for u in units)
        drop = abs(INJURY_LOG_ODDS[group]) / (LOGODDS_PER_POINT * POINTS_PER_YARD * yards_at_stake)
        out[group] = round(1 - drop, 3)
    return out


def games_missed_record(snap_counts: pd.DataFrame, yards: pd.DataFrame, regulars: pd.DataFrame) -> dict:
    """{pfr_id: {unit: {"pct", "games"}}} for listed players who missed a
    game for their current team in this history: the unit's efficiency in
    those games over the games he played (from the third he played on) -
    shown for reference only (see GROUP_UNITS' comment)."""
    if regulars.empty or yards.empty or snap_counts is None or snap_counts.empty:
        return {}
    snaps = snap_counts[snap_counts["game_type"] == "REG"].copy()
    snaps["team"] = nfl_totals._canonical(snaps["team"])
    ratios = unit_ratios(yards)
    out = {}
    for row in regulars.itertuples():
        side = GROUPS[row.group][0]
        mine = snaps[(snaps["pfr_player_id"] == row.pfr_player_id) & (snaps["team"] == row.team)]
        played_ids = set(mine.loc[mine[f"{side}_pct"].fillna(0) > 0, "game_id"])
        if mine.empty:
            continue
        first = (mine["season"] * 100 + mine["week"]).min()
        key = "team" if side == "offense" else "opp"
        team_games = ratios[(ratios[key] == row.team) & ((ratios["season"] * 100 + ratios["week"]) >= first)]
        played = team_games["game_id"].isin(played_ids)
        # Two missed games at least - one game's yards are mostly noise.
        if (~played).sum() < 2 or played.sum() < 3:
            continue
        record = {}
        for unit in GROUP_UNITS[row.group]:
            kind = unit.split("_")[0]
            n = UNIT_PLAYS[kind][1]
            ok = team_games[f"{kind}_ratio"].notna()

            def mean(mask):
                rows = team_games[mask & ok]
                return (rows[f"{kind}_ratio"] * rows[n]).sum() / rows[n].sum() if rows[n].sum() else float("nan")

            missed, normal = mean(~played), mean(played)
            if pd.notna(missed) and pd.notna(normal) and normal > 0 and (~played & ok).sum() >= 2:
                pct = missed / normal if side == "offense" else 2 - missed / normal
                record[unit] = {"pct": round(float(pct), 3), "games": int((~played & ok).sum())}
        if record:
            out[row.pfr_player_id] = record
    return out


MAX_QB_OPTIONS = 4


def _group_of(position: str):
    for group, (_, positions) in GROUPS.items():
        if position in positions:
            return group
    return None


def _name_key(name: pd.Series) -> pd.Series:
    key = name.fillna("").str.lower().str.replace(r"[^a-z ]", "", regex=True)
    return key.str.replace(r"\b(jr|sr|ii|iii|iv|v)\b", "", regex=True).str.split().str.join(" ")


def _fill_pfr_ids(roster: pd.DataFrame, snaps: pd.DataFrame) -> pd.DataFrame:
    """Rosters leave pfr_id blank for every offensive lineman: fill it from
    the snap counts by name - on the same team that season, else a name
    that only one player has."""
    roster = roster.copy()
    snap_names = snaps[["season", "team", "player", "pfr_player_id"]].drop_duplicates()
    snap_names["key"] = _name_key(snap_names["player"])
    roster["key"] = _name_key(roster["full_name"]) if "full_name" in roster.columns else ""
    by_team = snap_names.drop_duplicates(["season", "team", "key"]).set_index(["season", "team", "key"])["pfr_player_id"]
    missing = roster["pfr_id"].isna()
    idx = pd.MultiIndex.from_frame(roster.loc[missing, ["season", "team", "key"]])
    roster.loc[missing, "pfr_id"] = by_team.reindex(idx).to_numpy()
    unique = snap_names.drop_duplicates(["key", "pfr_player_id"]).groupby("key")["pfr_player_id"]
    unique = unique.first()[unique.nunique() == 1]
    missing = roster["pfr_id"].isna()
    roster.loc[missing, "pfr_id"] = roster.loc[missing, "key"].map(unique)
    return roster.drop(columns="key")


def regular_players(snap_counts: pd.DataFrame, rosters: pd.DataFrame | None = None,
                    min_share: float = LISTED_SHARE) -> pd.DataFrame:
    """Regulars going into each team's next game: [team, group, player,
    pfr_player_id, share], share = his mean snap share over his last
    LOOKBACK_GAMES games. With weekly `rosters` (pfr_id per team and week)
    a player's games are his own team's games each week, wherever he was,
    and he's listed under his latest roster team if still active there;
    without them, by the team he took snaps for."""
    columns = ["team", "group", "player", "pfr_player_id", "share"]
    needed = {"game_type", "season", "week", "game_id", "team", "pfr_player_id", "position", "offense_pct", "defense_pct"}
    if snap_counts is None or snap_counts.empty or not needed <= set(snap_counts.columns):
        return pd.DataFrame(columns=columns)
    snaps = snap_counts[snap_counts["game_type"] == "REG"].copy()
    if "player" not in snaps.columns:
        snaps["player"] = snaps["pfr_player_id"]
    snaps["team"] = nfl_totals._canonical(snaps["team"])
    snaps = snaps.sort_values(["season", "week"])
    latest_snap = snaps.groupby("pfr_player_id").agg(player=("player", "last"), position=("position", "last"))
    latest_snap["group"] = latest_snap["position"].map(_group_of)
    latest_snap = latest_snap[latest_snap["group"].notna()]
    team_games = snaps[["season", "week", "team", "game_id"]].drop_duplicates()

    # Without a roster, a player's games are every game of each team he
    # took snaps for, and he's listed with the last of them.
    players = snaps.groupby(["team", "pfr_player_id"]).size().reset_index()[["team", "pfr_player_id"]]
    games = players.rename(columns={"pfr_player_id": "pfr_id"}).merge(team_games, on="team")
    current_team = snaps.groupby("pfr_player_id")["team"].last()

    if rosters is not None and {"season", "week", "team", "pfr_id", "status"} <= set(rosters.columns):
        roster = rosters.copy()
        roster["team"] = nfl_totals._canonical(roster["team"])
        roster = _fill_pfr_ids(roster, snaps).dropna(subset=["pfr_id"])
        roster = roster.sort_values(["season", "week"]).drop_duplicates(["season", "week", "pfr_id"], keep="last")
        # With one: his roster team's game each week it played and he was
        # active (a week inactive or on reserve isn't a sign of a smaller
        # role), wherever he was, listed with his latest roster team while
        # still on it.
        on_roster = roster[roster["status"] == "ACT"][["season", "week", "team", "pfr_id"]]
        on_roster = on_roster.merge(team_games, on=["season", "week", "team"])
        latest = roster.groupby("pfr_id").tail(1).set_index("pfr_id")
        last_week = roster.groupby("season")["week"].max()
        active = (latest["status"].isin(["ACT", "INA"]) & (latest["week"] == latest["season"].map(last_week))
                  & (latest["season"] == roster["season"].max()))
        games = pd.concat([games[~games["pfr_id"].isin(latest.index)], on_roster], ignore_index=True)
        current_team = pd.concat([current_team[~current_team.index.isin(latest.index)], latest.loc[active, "team"]])

    games = games[games["pfr_id"].isin(latest_snap.index)]
    pct = snaps[["game_id", "pfr_player_id", "offense_pct", "defense_pct"]].rename(columns={"pfr_player_id": "pfr_id"})
    games = games.merge(pct, on=["game_id", "pfr_id"], how="left")
    games["group"] = games["pfr_id"].map(latest_snap["group"])
    side = games["group"].map({g: s for g, (s, _) in GROUPS.items()})
    games["share"] = games["offense_pct"].where(side == "offense", games["defense_pct"]).fillna(0.0).astype(float)
    games = games.sort_values(["season", "week"]).groupby("pfr_id").tail(LOOKBACK_GAMES)
    share = games.groupby("pfr_id")["share"].mean()
    out = pd.DataFrame({"pfr_player_id": share.index, "share": share.values})
    out["team"] = out["pfr_player_id"].map(current_team)
    out["player"] = out["pfr_player_id"].map(latest_snap["player"])
    out["group"] = out["pfr_player_id"].map(latest_snap["group"])
    out = out[out["team"].notna() & (out["share"] >= min_share)]
    return out[columns].sort_values(["team", "group", "share"], ascending=[True, True, False]).reset_index(drop=True)


def average_regular_shares(regulars: pd.DataFrame) -> dict:
    """{group: league-average regular's snap share} - the weight of the
    page's "an average regular" option."""
    regulars = regulars[regulars["share"] >= REGULAR_SHARE]
    if regulars.empty:
        return {}
    return regulars.groupby("group")["share"].mean().round(3).to_dict()


def qb_options(weekly: pd.DataFrame, rosters: pd.DataFrame, games: pd.DataFrame) -> dict:
    """{team: [{id, name, qb_plus, listed}]} - the listed starter first, then
    up to MAX_QB_OPTIONS - 1 teammates (most dropbacks for the team this
    season, then the latest roster's QBs), each rated as QB+."""
    qb_games = nfl_game_picks.qb_games_from_weekly(weekly)
    season = games["season"].max()
    rosters = rosters if rosters is not None else pd.DataFrame()
    names = {}
    if {"gsis_id", "full_name"} <= set(rosters.columns):
        names.update(rosters.dropna(subset=["gsis_id"]).set_index("gsis_id")["full_name"].to_dict())
    if {"player_id", "player_display_name"} <= set(weekly.columns):
        names.update(weekly.dropna(subset=["player_id"]).set_index("player_id")["player_display_name"].to_dict())
    thrown = pd.Series(dtype=float)
    if "team" in weekly.columns:
        this_season = qb_games.join(weekly["team"])
        this_season = this_season[this_season["season"] == season]
        thrown = this_season.groupby(["team", "qb"])["dropbacks"].sum()
    roster_qbs = {}
    if {"season", "week", "team", "position", "gsis_id", "status"} <= set(rosters.columns):
        latest = rosters[(rosters["season"] == season) & (rosters["position"] == "QB") & rosters["gsis_id"].notna()]
        if not latest.empty:
            latest = latest[latest["week"] == latest.groupby("team")["week"].transform("max")]
            latest = latest[latest["status"].isin(["ACT", "RES", "INA"])]
            roster_qbs = latest.groupby("team")["gsis_id"].apply(list).to_dict()

    options = {}
    for side in ("home", "away"):
        for team, listed, listed_name in zip(games[f"{side}_team"], games[f"{side}_qb_id"],
                                             games.get(f"{side}_qb_name", pd.Series(None, index=games.index))):
            ids = [listed] if isinstance(listed, str) else []
            if isinstance(listed, str) and isinstance(listed_name, str):
                names.setdefault(listed, listed_name)
            if not thrown.empty and team in thrown.index.get_level_values(0):
                ids += [q for q in thrown.loc[team].sort_values(ascending=False).index if q not in ids]
            ids += [q for q in roster_qbs.get(team, []) if q not in ids]
            ids = ids[:MAX_QB_OPTIONS]
            rating = nfl_totals.starter_qb_epa(qb_games, pd.Series(ids, dtype=object))
            options[team] = [{"id": q, "name": names.get(q, q), "qb_plus": None if pd.isna(r) else round(float(r), 4),
                              "listed": q == listed} for q, r in zip(ids, rating)]
    average = average_starter_qb_plus(qb_games, weekly, season)
    if average is not None:
        for team in options:
            options[team].append({"id": AVERAGE_QB_ID, "name": "Average starter", "qb_plus": average, "listed": False})
    return options


def average_starter_qb_plus(qb_games: pd.DataFrame, weekly: pd.DataFrame, season) -> float | None:
    """QB+ of the league's average starter: the mean over every team's
    most-used passer this season (last season's before any games)."""
    if qb_games.empty or "team" not in weekly.columns:
        return None
    games = qb_games.join(weekly["team"])
    this = games[games["season"] == season]
    if this.empty:
        this = games[games["season"] == games["season"].max()]
    starters = this.groupby(["team", "qb"])["dropbacks"].sum().reset_index()
    starters = starters.sort_values("dropbacks").drop_duplicates("team", keep="last")["qb"]
    rating = nfl_totals.starter_qb_epa(qb_games, starters.reset_index(drop=True))
    return None if rating.isna().all() else round(float(rating.mean()), 4)


def _log_odds(p: float) -> float:
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def pairing_log_odds(master, qb_continuity, weekly, games: pd.DataFrame, options: dict) -> dict:
    """{game_id: {"home_id|away_id": model log-odds}} for every pairing of
    the two teams' QB options, as nfl_pipeline computes it before the market
    tie-break and calibration. When the live model rates QBs as QB+, each
    option's rating goes straight into its features (so an average starter
    works too); otherwise the starters are swapped and the features rebuilt."""
    artifact = ml_models.load_model(config.NFL_GAME_PICK_WIN_PROBABILITY_MODEL_PATH)
    if artifact is not None and {"home_qb_plus", "away_qb_plus"} <= set(artifact["feature_columns"]):
        columns = artifact["feature_columns"]
        features = nfl_game_picks.resolve_feature_builder(columns)(master, qb_continuity, weekly, games)
        rows, keys = [], []
        for _, f in features.iterrows():
            if f[columns].drop(["home_qb_plus", "away_qb_plus"]).isna().any():
                continue
            for h in options.get(f["home_team"], []):
                for a in options.get(f["away_team"], []):
                    if h["qb_plus"] is None or a["qb_plus"] is None:
                        continue
                    rows.append({**f.to_dict(), "home_qb_plus": h["qb_plus"], "away_qb_plus": a["qb_plus"]})
                    keys.append((f["game_id"], f"{h['id']}|{a['id']}"))
        if not rows:
            return {}
        proba = artifact["model"].predict_proba(nfl_game_picks.game_feature_matrix(pd.DataFrame(rows), columns))[:, 1]
        out = {}
        for (game_id, key), p in zip(keys, proba):
            out.setdefault(game_id, {})[key] = round(_log_odds(p), 4)
        return out

    rows = []
    for _, g in games.iterrows():
        for h in options.get(g["home_team"], []):
            for a in options.get(g["away_team"], []):
                if AVERAGE_QB_ID in (h["id"], a["id"]):
                    continue
                rows.append({**g.to_dict(), "game_id": f"{g['game_id']}|{h['id']}|{a['id']}",
                             "home_qb_id": h["id"], "away_qb_id": a["id"]})
    if not rows:
        return {}
    pairs = pd.DataFrame(rows)
    probs = nfl_game_picks.compute_game_win_probabilities(master, qb_continuity, weekly, pairs)
    probs = nfl_game_picks.apply_ml_model(probs, master, qb_continuity, weekly, pairs)
    out = {}
    for key, p in zip(probs["game_id"], probs["home_win_probability"]):
        if pd.isna(p):
            continue
        game_id, home_id, away_id = key.split("|")
        out.setdefault(game_id, {})[f"{home_id}|{away_id}"] = round(_log_odds(p), 4)
    return out


def build_whatif(final_probs: pd.DataFrame, master, qb_continuity, weekly, rosters, snap_counts,
                 games: pd.DataFrame, pbp: pd.DataFrame | None = None) -> dict:
    """The page's what-if data for `games` (the ones not yet played)."""
    options = qb_options(weekly, rosters, games)
    pairings = pairing_log_odds(master, qb_continuity, weekly, games, options)
    regulars = regular_players(snap_counts, rosters)
    yards = unit_game_yards(pbp[[c for c in YARDS_PBP_COLUMNS if c in pbp.columns]] if pbp is not None else None)
    if not yards.empty:
        yards["team"], yards["opp"] = nfl_totals._canonical(yards["team"]), nfl_totals._canonical(yards["opp"])
    units = team_units(yards) if not yards.empty else {}
    defaults = group_default_pct(yards) if not yards.empty else {}
    record = games_missed_record(snap_counts, yards, regulars)
    shown = final_probs.set_index("game_id")["home_win_probability"]
    out_games = []
    for _, g in games.iterrows():
        if g["game_id"] not in shown.index or pd.isna(shown[g["game_id"]]) or g["game_id"] not in pairings:
            continue
        teams = {}
        for side in ("home", "away"):
            team = g[f"{side}_team"]
            canonical = nfl_totals._canonical(pd.Series([team])).iloc[0]
            r = regulars[regulars["team"] == canonical]
            # Only QB options the model could price (an average starter
            # needs a QB+ model).
            priced = {key.split("|")[0 if side == "home" else 1] for key in pairings[g["game_id"]]}
            teams[side] = {
                "team": team, "qbs": [q for q in options.get(team, []) if q["id"] in priced],
                "units": units.get(canonical, {}),
                "regulars": [{"id": row.pfr_player_id, "name": row.player, "group": row.group,
                              "share": round(float(row.share), 3), "record": record.get(row.pfr_player_id, {})}
                             for row in r.itertuples()],
            }
        out_games.append({
            "game_id": g["game_id"], "season": int(g["season"]), "week": int(g["week"]),
            "gameday": str(g.get("gameday", "") or ""), "home_win_probability": round(float(shown[g["game_id"]]), 4),
            "home": teams["home"], "away": teams["away"], "pairing_log_odds": pairings[g["game_id"]],
        })
    return {"group_units": GROUP_UNITS, "group_default_pct": defaults, "points_per_yard": POINTS_PER_YARD,
            "logodds_per_point": LOGODDS_PER_POINT, "average_regular_share": average_regular_shares(regulars),
            "games": out_games}


# Injury-report statuses treated as not playing.
REPORT_OUT_STATUSES = {"Out", "Doubtful"}


def unit_pcts(team: dict, out_ids, data: dict) -> dict:
    """{unit: percent} for one side with the players `out_ids` out at their
    defaults - the page's whatifUnitPcts without typed overrides (an
    "average:<G>" id counts the league's average regular at G)."""
    pcts = {unit: 1.0 for kind in UNIT_PLAYS for unit in (f"{kind}_offense", f"{kind}_defense")}
    players = [r for r in team.get("regulars", []) if r["id"] in out_ids]
    for pid in out_ids:
        if str(pid).startswith("average:"):
            group = pid.split(":", 1)[1]
            players.append({"group": group, "share": data.get("average_regular_share", {}).get(group, 0)})
    for player in players:
        full = data.get("group_default_pct", {}).get(player["group"])
        if full is None:
            continue
        pct = 1 - (1 - full) * float(player["share"])
        for unit in data.get("group_units", {}).get(player["group"], []):
            pcts[unit] *= pct
    return pcts


def home_probability(game: dict, data: dict, out: dict | None = None, qbs: dict | None = None) -> float:
    """The page's whatifProbability: the game's pre-report probability moved
    by the model's change to the chosen QBs and by the absences `out`
    ({"home": ids, "away": ids}) at their defaults."""
    out = out or {"home": set(), "away": set()}
    qbs = qbs or {}

    def listed(side):
        options = game[side].get("qbs", [])
        return next((q["id"] for q in options if q.get("listed")), options[0]["id"] if options else None)

    pairs = game.get("pairing_log_odds", {})
    listed_key = f"{listed('home')}|{listed('away')}"
    chosen_key = f"{qbs.get('home') or listed('home')}|{qbs.get('away') or listed('away')}"
    qb_shift = pairs[chosen_key] - pairs[listed_key] if chosen_key in pairs and listed_key in pairs else 0.0

    def net(side):
        team = game[side]
        pcts = unit_pcts(team, out.get(side, set()), data)
        return sum((pct - 1) * team.get("units", {})[unit]["ypp"] * team["units"][unit]["plays"] * data.get("points_per_yard", 0)
                   for unit, pct in pcts.items() if unit in team.get("units", {}))

    margin = net("home") - net("away")
    logit = _log_odds(game["home_win_probability"]) + qb_shift + data.get("logodds_per_point", 0) * margin
    return 1 / (1 + math.exp(-logit))


def apply_injury_report(data: dict, report: pd.DataFrame, rosters: pd.DataFrame | None,
                        snap_counts: pd.DataFrame | None, game_ids) -> dict:
    """For each game in `game_ids` (the ones the report is final for): the
    regulars and starting QBs the injury report lists Out or Doubtful,
    stored on the game as "injury_report" {"out": {side: [ids]}, "qb":
    {side: id}, "note"} with the probability that gives ("posted_probability").
    A listed QB who's out is replaced by his team's next QB option. Returns
    {game_id: {"probability", "note"}} for the games it changed or confirmed."""
    if report is None or report.empty or not {"gsis_id", "report_status"} <= set(report.columns):
        return {}
    out_report = report[report["report_status"].isin(REPORT_OUT_STATUSES)]
    out_gsis = set(out_report["gsis_id"].dropna())
    out_pfr = set()
    if rosters is not None and not rosters.empty and {"gsis_id", "pfr_id"} <= set(rosters.columns):
        roster = rosters.copy()
        if snap_counts is not None and not snap_counts.empty:
            snaps = snap_counts[snap_counts["game_type"] == "REG"].copy()
            snaps["team"] = nfl_totals._canonical(snaps["team"])
            roster["team"] = nfl_totals._canonical(roster["team"])
            roster = _fill_pfr_ids(roster, snaps)
        out_pfr = set(roster.loc[roster["gsis_id"].isin(out_gsis), "pfr_id"].dropna())
    names = out_report.set_index("gsis_id")["full_name"].to_dict() if "full_name" in out_report.columns else {}
    results = {}
    for game in data["games"]:
        if game["game_id"] not in set(game_ids):
            continue
        entry = {"out": {}, "qb": {}, "note": []}
        for side in ("home", "away"):
            team = game[side]
            ids = [r["id"] for r in team.get("regulars", []) if r["id"] in out_pfr]
            entry["out"][side] = ids
            entry["note"] += [r["name"] for r in team.get("regulars", []) if r["id"] in out_pfr]
            options = team.get("qbs", [])
            listed = next((q for q in options if q.get("listed")), None)
            if listed is not None and listed["id"] in out_gsis:
                backup = next((q for q in options if q["id"] not in out_gsis and q["id"] != listed["id"]
                               and q["id"] != AVERAGE_QB_ID), None)
                if backup is not None:
                    entry["qb"][side] = backup["id"]
                    entry["note"].append(f"{names.get(listed['id'], listed['name'])} (QB; {backup['name']} starts)")
        probability = home_probability(game, data, {s: set(v) for s, v in entry["out"].items()}, entry["qb"])
        entry["note"] = ", ".join(entry["note"])
        game["injury_report"] = entry
        game["posted_probability"] = round(probability, 4)
        results[game["game_id"]] = {"probability": probability, "note": entry["note"]}
    return results


def add_results(data: dict, schedules: pd.DataFrame, week_snaps: pd.DataFrame) -> dict:
    """For a replayed week: each game's final score, the QBs who started
    (marked "started"), and each listed player's "played" - whether he took
    a snap on his side of the ball. Returns `data`, changed in place."""
    final = schedules.set_index("game_id")
    played = set()
    if week_snaps is not None and not week_snaps.empty:
        took = week_snaps[(week_snaps["offense_pct"].fillna(0) > 0) | (week_snaps["defense_pct"].fillna(0) > 0)]
        played = set(took["pfr_player_id"])
    for game in data["games"]:
        if game["game_id"] not in final.index:
            continue
        row = final.loc[game["game_id"]]
        game["final"] = {"home_score": int(row["home_score"]), "away_score": int(row["away_score"])}
        for side in ("home", "away"):
            started = row.get(f"{side}_qb_id")
            for qb in game[side]["qbs"]:
                qb["started"] = qb["id"] == started
            for player in game[side]["regulars"]:
                player["played"] = player["id"] in played
    return data


def save_json(path: str, data: dict) -> dict:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    return data


def write_whatif_json(path: str, *args, **kwargs) -> dict:
    return save_json(path, build_whatif(*args, **kwargs))

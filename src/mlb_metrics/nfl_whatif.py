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

Injuries move the log-odds by INJURY_LOG_ODDS per regular marked out,
times his snap share. Weather isn't an input: wind, cold and domes moved
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

AVERAGE_QB_ID = "average"
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
                 games: pd.DataFrame) -> dict:
    """The page's what-if data for `games` (the ones not yet played)."""
    options = qb_options(weekly, rosters, games)
    pairings = pairing_log_odds(master, qb_continuity, weekly, games, options)
    regulars = regular_players(snap_counts, rosters)
    shown = final_probs.set_index("game_id")["home_win_probability"]
    out_games = []
    for _, g in games.iterrows():
        if g["game_id"] not in shown.index or pd.isna(shown[g["game_id"]]) or g["game_id"] not in pairings:
            continue
        teams = {}
        for side in ("home", "away"):
            team = g[f"{side}_team"]
            r = regulars[regulars["team"] == nfl_totals._canonical(pd.Series([team])).iloc[0]]
            # Only QB options the model could price (an average starter
            # needs a QB+ model).
            priced = {key.split("|")[0 if side == "home" else 1] for key in pairings[g["game_id"]]}
            teams[side] = {
                "team": team, "qbs": [q for q in options.get(team, []) if q["id"] in priced],
                "regulars": [{"id": row.pfr_player_id, "name": row.player, "group": row.group,
                              "share": round(float(row.share), 3)} for row in r.itertuples()],
            }
        out_games.append({
            "game_id": g["game_id"], "season": int(g["season"]), "week": int(g["week"]),
            "gameday": str(g.get("gameday", "") or ""), "home_win_probability": round(float(shown[g["game_id"]]), 4),
            "home": teams["home"], "away": teams["away"], "pairing_log_odds": pairings[g["game_id"]],
        })
    return {"injury_log_odds": INJURY_LOG_ODDS, "average_regular_share": average_regular_shares(regulars),
            "games": out_games}


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

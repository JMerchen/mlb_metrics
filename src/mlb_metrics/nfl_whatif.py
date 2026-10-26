"""What-if inputs for the NFL page's game picks: swap a team's starting QB
or mark its regulars out, and the page recomputes the win probability.

For every game not yet played this writes the probability shown on the
page (after the market tie-break and calibration), each team's QB choices
(the listed starter, his teammates who have thrown a pass for the team
this season and the QBs on its latest roster) rated as QB+, the model's
log-odds for every home/away QB pairing - the page shifts the shown
probability by the change from the listed pairing - and each team's
regulars by position group, with their recent snap shares.

Injuries move the log-odds by INJURY_LOG_ODDS per whole group out, scaled
by the snap share of the regulars marked out
(scripts/backtest_nfl_injury_impact.py: what an absence did to the home
team's chance on top of the model, 2018-2026). Weather isn't an input:
wind, cold and domes moved neither results nor Vegas's sides beyond the
model (it changes how many points get scored, not who wins).
"""

import json
import math
import os

import pandas as pd

from mlb_metrics import nfl_game_picks, nfl_totals

# Position groups for snap counts (offense or defense snaps).
GROUPS = {
    "RB": ("offense", {"RB", "FB"}), "WR": ("offense", {"WR"}), "TE": ("offense", {"TE"}),
    "OL": ("offense", {"T", "G", "C", "OT", "OG", "OL"}),
    "DL": ("defense", {"DE", "DT", "NT", "DL"}), "LB": ("defense", {"LB", "OLB", "ILB", "MLB"}),
    "CB": ("defense", {"CB"}), "S": ("defense", {"S", "FS", "SS", "DB"}),
}
# A regular: at least this share of his side's snaps over the team's last
# LOOKBACK_GAMES games (0 for a game he missed).
LOOKBACK_GAMES = 4
REGULAR_SHARE = 0.5

# Home-win log-odds per whole group out (home minus away), on top of the
# model - scripts/backtest_nfl_injury_impact.py. RB's estimate was +0.04
# (noise around zero), kept at 0 rather than rewarding an absence.
INJURY_LOG_ODDS = {"OL": -0.85, "CB": -0.45, "DL": -0.42, "WR": -0.39, "LB": -0.18, "S": -0.08, "TE": -0.05, "RB": 0.0}

MAX_QB_OPTIONS = 4


def regular_players(snap_counts: pd.DataFrame) -> pd.DataFrame:
    """Each team's regulars going into its next game: [team, group, player,
    pfr_player_id, share], share = his mean snap share over the team's last
    LOOKBACK_GAMES regular-season games."""
    columns = ["team", "group", "player", "pfr_player_id", "share"]
    if snap_counts is None or snap_counts.empty:
        return pd.DataFrame(columns=columns)
    snaps = snap_counts[snap_counts["game_type"] == "REG"].copy()
    snaps["team"] = nfl_totals._canonical(snaps["team"])
    games = snaps[["season", "week", "game_id", "team"]].drop_duplicates().sort_values(["season", "week"])
    recent = games.groupby("team").tail(LOOKBACK_GAMES)
    n_games = recent.groupby("team").size()
    snaps = snaps.merge(recent[["game_id", "team"]], on=["game_id", "team"])
    out = []
    for group, (side, positions) in GROUPS.items():
        rows = snaps[snaps["position"].isin(positions)]
        if rows.empty:
            continue
        share = rows.groupby(["team", "pfr_player_id"]).agg(player=("player", "last"), total=(f"{side}_pct", "sum"))
        share = share.reset_index()
        share["share"] = share["total"].astype(float) / share["team"].map(n_games).astype(float)
        share["group"] = group
        out.append(share[share["share"] >= REGULAR_SHARE][columns])
    if not out:
        return pd.DataFrame(columns=columns)
    regulars = pd.concat(out, ignore_index=True)
    return regulars.sort_values(["team", "group", "share"], ascending=[True, True, False]).reset_index(drop=True)


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
    return options


def _log_odds(p: float) -> float:
    p = min(max(float(p), 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def pairing_log_odds(master, qb_continuity, weekly, games: pd.DataFrame, options: dict) -> dict:
    """{game_id: {"home_id|away_id": model log-odds}} for every pairing of
    the two teams' QB options - the live model (or the heuristic when no
    model is saved), as nfl_pipeline computes it before the market
    tie-break and calibration."""
    rows = []
    for _, g in games.iterrows():
        for h in options.get(g["home_team"], []):
            for a in options.get(g["away_team"], []):
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
    regulars = regular_players(snap_counts)
    shown = final_probs.set_index("game_id")["home_win_probability"]
    out_games = []
    for _, g in games.iterrows():
        if g["game_id"] not in shown.index or pd.isna(shown[g["game_id"]]) or g["game_id"] not in pairings:
            continue
        teams = {}
        for side in ("home", "away"):
            team = g[f"{side}_team"]
            r = regulars[regulars["team"] == nfl_totals._canonical(pd.Series([team])).iloc[0]]
            teams[side] = {
                "team": team, "qbs": options.get(team, []),
                "regulars": [{"id": row.pfr_player_id, "name": row.player, "group": row.group,
                              "share": round(float(row.share), 3)} for row in r.itertuples()],
            }
        out_games.append({
            "game_id": g["game_id"], "season": int(g["season"]), "week": int(g["week"]),
            "gameday": str(g.get("gameday", "") or ""), "home_win_probability": round(float(shown[g["game_id"]]), 4),
            "home": teams["home"], "away": teams["away"], "pairing_log_odds": pairings[g["game_id"]],
        })
    return {"injury_log_odds": INJURY_LOG_ODDS, "games": out_games}


def write_whatif_json(path: str, *args, **kwargs) -> dict:
    data = build_whatif(*args, **kwargs)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    return data

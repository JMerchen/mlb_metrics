"""Team play profiles for the NFL page's game simulation (docs/nfl_sim.js):
how each offense plays, from nflverse play-by-play, and the league's kicking
game. Written weekly to docs/data/nfl_sim_profiles.json.

The profiles give a team its style - how often it throws, how its runs and
completions are spread (short and steady, or boom and bust), how often it's
sacked or picked off. They don't set how good a team is: each team's yard
distributions are scaled to the league's average gain, and the simulation
calibrates strength to the posted pick (and the What if? settings) instead,
so a simulated game agrees with the model rather than second-guessing it.

A team's rates and shapes are its current-season plays pulled toward the
league's with SHRINK_PLAYS pseudo-plays, so a few weeks don't swing them.

Each team also lists its players for the simulated box score: who gets the
targets and carries (shares over its last PLAYER_GAMES games, among players
still active on its roster), how each catches and gains next to the team
(shrunk toward the team), its starting QB and its kicker. Players carry
both ids - gsis for the play-by-play, pfr for the What if? panel's
regulars - so a player marked out there sits out the simulation too.
"""

import json
import math

import numpy as np
import pandas as pd

# 2.5% steps, with the very ends trimmed: the simulation draws a value by
# interpolating between neighboring quantiles, and the raw minimum and
# maximum (a 93-yard run, a return touchdown) would stretch the outer bins.
QUANTILES = np.r_[0.003, np.linspace(0.025, 0.975, 39), 0.997]
SHRINK_PLAYS = 150
PLAYER_GAMES = 4
# Pseudo-targets / -carries / -catches toward the team's rate for a player.
SHRINK_TARGETS = 25
SHRINK_CARRIES = 30
SHRINK_CATCHES = 12
PBP_COLUMNS = [
    "game_id", "season", "season_type", "week", "posteam", "defteam", "play_type", "down", "ydstogo", "qtr",
    "score_differential", "yards_gained", "pass_attempt", "sack", "interception", "complete_pass",
    "fumble_lost", "kick_distance", "return_yards", "touchback", "field_goal_result", "extra_point_result",
    "drive", "drive_start_yard_line", "drive_start_transition",
    "passer_player_id", "passer_player_name", "receiver_player_id", "receiver_player_name",
    "rusher_player_id", "rusher_player_name", "kicker_player_id", "kicker_player_name",
]
ROSTER_COLUMNS = ["season", "week", "team", "position", "full_name", "gsis_id", "pfr_id", "status"]


def _quantiles(values) -> list:
    values = np.asarray(values, dtype=float)
    return [round(float(v), 2) for v in np.quantile(values, QUANTILES)] if len(values) else []


def _sampler_mean(q) -> float:
    """The mean of a draw interpolated between neighboring quantiles."""
    q = np.asarray(q, dtype=float)
    w = np.diff(QUANTILES)
    return float(((q[:-1] + q[1:]) / 2 * w).sum() / w.sum())


def _shrunk_rate(hits, n, league_rate, k=SHRINK_PLAYS) -> float:
    return float((hits + k * league_rate) / (n + k))


def _shrunk_shape(team_values, league_q, k=SHRINK_PLAYS) -> list:
    """The team's quantiles blended with the league's by sample size, then
    scaled so the mean gain is the league's (style, not strength)."""
    league_q = np.asarray(league_q)
    if len(team_values) == 0:
        return [round(float(v), 2) for v in league_q]
    n = len(team_values)
    w = n / (n + k)
    q = w * np.asarray(_quantiles(team_values)) + (1 - w) * league_q
    # The draw's mean is linear in the gains, so scaling the gains by k
    # gives mean = losses + k * gains; pick k to land on the league's.
    gains = np.where(q > 0, q, 0.0)
    losses = _sampler_mean(q - gains)
    if _sampler_mean(gains) > 0:
        q = q - gains + gains * (_sampler_mean(league_q) - losses) / _sampler_mean(gains)
    return [round(float(v), 2) for v in q]


def _scrimmage(pbp: pd.DataFrame) -> pd.DataFrame:
    return pbp[(pbp["season_type"] == "REG") & pbp["play_type"].isin(["pass", "run"]) & pbp["posteam"].notna()]


def _offense_rates(plays: pd.DataFrame) -> dict:
    """Raw counts for one group of scrimmage plays."""
    passes = plays[plays["play_type"] == "pass"]
    attempts = passes[passes["sack"] != 1]
    runs = plays[plays["play_type"] == "run"]
    neutral = plays[plays["down"].isin([1, 2]) & (plays["qtr"] <= 3) & (plays["score_differential"].abs() <= 7)]
    completions = attempts[attempts["complete_pass"] == 1]
    return {
        "plays": len(plays), "neutral": len(neutral), "neutral_pass": int((neutral["play_type"] == "pass").sum()),
        "dropbacks": len(passes), "sacks": int((passes["sack"] == 1).sum()),
        "attempts": len(attempts), "completions": len(completions), "ints": int((attempts["interception"] == 1).sum()),
        "runs": len(runs), "run_fumbles": int((runs["fumble_lost"] == 1).sum()),
        "catch_fumbles": int((completions["fumble_lost"] == 1).sum()),
        "rush_yards": runs["yards_gained"].dropna().to_numpy(),
        "completion_yards": completions["yards_gained"].dropna().to_numpy(),
        "sack_yards": passes.loc[passes["sack"] == 1, "yards_gained"].dropna().to_numpy(),
    }


def _drive_starts(pbp: pd.DataFrame, transition: str) -> np.ndarray:
    """Yards from its own goal where the receiving team's drive started
    ("ARI 30" for ARI = 30; for the other team, 100 - 30)."""
    starts = pbp[(pbp["drive_start_transition"] == transition) & pbp["drive_start_yard_line"].notna() & pbp["posteam"].notna()]
    starts = starts.drop_duplicates(["game_id", "drive"])
    out = []
    for side, posteam in zip(starts["drive_start_yard_line"], starts["posteam"]):
        parts = str(side).split()
        if len(parts) == 1 and parts[0] == "50":
            out.append(50)
        elif len(parts) == 2 and parts[1].isdigit():
            yards = int(parts[1])
            out.append(yards if parts[0] == posteam else 100 - yards)
    return np.asarray(out, dtype=float)


def _field_goal_curve(pbp: pd.DataFrame) -> dict:
    """Make probability by distance: 1 / (1 + exp(-(a + b * distance)))."""
    kicks = pbp[(pbp["play_type"] == "field_goal") & pbp["kick_distance"].notna()]
    x = kicks["kick_distance"].to_numpy(dtype=float)
    y = (kicks["field_goal_result"] == "made").to_numpy(dtype=float)
    if len(x) < 50:
        return {"a": 6.0, "b": -0.11}
    import statsmodels.api as sm

    fit = sm.Logit(y, sm.add_constant(x)).fit(disp=0)
    return {"a": round(float(fit.params[0]), 4), "b": round(float(fit.params[1]), 5)}


def _active_roster(rosters: pd.DataFrame | None, season: int) -> pd.DataFrame | None:
    """Each player's latest roster row this season, indexed by gsis id."""
    if rosters is None or rosters.empty or not set(ROSTER_COLUMNS) <= set(rosters.columns):
        return None
    r = rosters[(rosters["season"] == season) & rosters["gsis_id"].notna()]
    return r.sort_values("week").drop_duplicates("gsis_id", keep="last").set_index("gsis_id")


def _ratio(num, den, prior_rate, k) -> float:
    return float((num + k * prior_rate) / (den + k))


def team_players(current: pd.DataFrame, team: str, roster: pd.DataFrame | None) -> dict:
    """{qb, kicker, players: [...]} for one team from this season's plays
    (shares from its last PLAYER_GAMES games). A player: {gsis, pfr, name,
    pos, targets, carries (shares), catch (catch rate / the team's), ypr
    and ypc (yards per catch / carry, over the team's)}. The QB's own runs
    are one entry with role "QB", so they follow whoever starts."""
    plays = _scrimmage(current[current["posteam"] == team])
    if plays.empty:
        return {"qb": None, "kicker": None, "players": []}
    games = plays.drop_duplicates("game_id").sort_values("week")["game_id"].tail(PLAYER_GAMES)
    recent = plays[plays["game_id"].isin(games)]
    passes = plays[(plays["play_type"] == "pass") & (plays["sack"] != 1) & plays["receiver_player_id"].notna()]
    runs = plays[(plays["play_type"] == "run") & plays["rusher_player_id"].notna()]
    team_catch = passes["complete_pass"].mean() if len(passes) else 0.65
    caught = passes[passes["complete_pass"] == 1]
    team_ypr = caught["yards_gained"].mean() if len(caught) else 11.0
    team_ypc = runs["yards_gained"].mean() if len(runs) else 4.4

    dropbacks = recent[recent["play_type"] == "pass"]
    qb_counts = dropbacks["passer_player_id"].value_counts()
    qb_ids = set(plays["passer_player_id"].dropna())
    qb_id = qb_counts.index[0] if len(qb_counts) else None

    def known(gsis):
        if roster is None:
            return True
        return gsis in roster.index and roster.at[gsis, "team"] == team and roster.at[gsis, "status"] == "ACT"

    def info(gsis, fallback):
        if roster is not None and gsis in roster.index:
            row = roster.loc[gsis]
            pfr = row["pfr_id"] if isinstance(row["pfr_id"], str) else None
            return {"name": row["full_name"] or fallback, "pos": row["position"], "pfr": pfr}
        return {"name": fallback, "pos": None, "pfr": None}

    recent_passes = recent[(recent["play_type"] == "pass") & (recent["sack"] != 1) & recent["receiver_player_id"].notna()]
    recent_runs = recent[(recent["play_type"] == "run") & recent["rusher_player_id"].notna()]
    is_qb = recent_runs["rusher_player_id"].isin(qb_ids)
    targets = recent_passes["receiver_player_id"].value_counts()
    carries = recent_runs.loc[~is_qb, "rusher_player_id"].value_counts()
    ids = [g for g in dict.fromkeys(list(targets.index) + list(carries.index)) if known(g)]
    total_targets = sum(targets.get(g, 0) for g in ids)
    total_carries = sum(carries.get(g, 0) for g in ids) + int(is_qb.sum())

    players = []
    for gsis in ids:
        mine_t = passes[passes["receiver_player_id"] == gsis]
        mine_c = caught[caught["receiver_player_id"] == gsis]
        mine_r = runs[runs["rusher_player_id"] == gsis]
        name = (mine_t["receiver_player_name"].dropna().tolist() + mine_r["rusher_player_name"].dropna().tolist() + [gsis])[0]
        players.append({
            "gsis": gsis, **info(gsis, name),
            "targets": round(targets.get(gsis, 0) / max(1, total_targets), 4),
            "carries": round(carries.get(gsis, 0) / max(1, total_carries), 4),
            "catch": round(_ratio(mine_t["complete_pass"].sum(), len(mine_t), team_catch, SHRINK_TARGETS) / team_catch, 3),
            "ypr": round(_ratio(mine_c["yards_gained"].sum(), len(mine_c), team_ypr, SHRINK_CATCHES) / team_ypr, 3),
            "ypc": round(_ratio(mine_r["yards_gained"].sum(), len(mine_r), team_ypc, SHRINK_CARRIES) / team_ypc, 3),
        })
    qb_runs = runs[runs["rusher_player_id"].isin(qb_ids)]
    players.append({
        "role": "QB", "gsis": None, "pfr": None, "name": "QB", "pos": "QB", "targets": 0.0,
        "carries": round(int(is_qb.sum()) / max(1, total_carries), 4), "catch": 1.0, "ypr": 1.0,
        "ypc": round(_ratio(qb_runs["yards_gained"].sum(), len(qb_runs), team_ypc, SHRINK_CARRIES) / team_ypc, 3),
    })
    kicks = current[(current["posteam"] == team) & current["play_type"].isin(["field_goal", "extra_point"])
                    & current["kicker_player_id"].notna()].sort_values("week")
    kicker = None
    if len(kicks):
        gsis = kicks["kicker_player_id"].iloc[-1]
        kicker = {"gsis": gsis, "name": info(gsis, kicks["kicker_player_name"].iloc[-1])["name"]}
    qb = None
    if qb_id is not None:
        qb = {"gsis": qb_id, "name": info(qb_id, dropbacks.loc[dropbacks["passer_player_id"] == qb_id, "passer_player_name"].iloc[0])["name"]}
    return {"qb": qb, "kicker": kicker, "players": players}


def build_profiles(pbp: pd.DataFrame, season: int, rosters: pd.DataFrame | None = None) -> dict:
    """{season, through_week, league: {...}, teams: {code: {...}}} for the
    simulation. League numbers use this and last season (kickoff rules
    changed in 2024-25, so nothing older); team numbers use this season.
    rosters (weekly, nflverse) keep each team's players to those active on
    it now; without them every player who has played for it counts."""
    recent = pbp[pbp["season"].isin([season - 1, season]) & (pbp["season_type"] == "REG")]
    current = recent[recent["season"] == season]
    league = _offense_rates(_scrimmage(recent))
    punts = recent[(recent["play_type"] == "punt") & recent["kick_distance"].notna()]
    punt_net = (punts["kick_distance"] - punts["return_yards"].fillna(0)).to_numpy(dtype=float)
    xps = recent[recent["play_type"] == "extra_point"]
    games = max(1, recent["game_id"].nunique())

    lg = {
        "neutral_pass_rate": round(league["neutral_pass"] / max(1, league["neutral"]), 4),
        "sack_rate": round(league["sacks"] / max(1, league["dropbacks"]), 4),
        "comp_rate": round(league["completions"] / max(1, league["attempts"]), 4),
        "int_rate": round(league["ints"] / max(1, league["attempts"]), 4),
        "run_fumble_rate": round(league["run_fumbles"] / max(1, league["runs"]), 4),
        "catch_fumble_rate": round(league["catch_fumbles"] / max(1, league["completions"]), 4),
        "sack_yards": round(float(np.mean(league["sack_yards"])) if len(league["sack_yards"]) else -7.0, 2),
        "rush_yards": _quantiles(league["rush_yards"]),
        "completion_yards": _quantiles(league["completion_yards"]),
        "kickoff_start": _quantiles(_drive_starts(recent, "KICKOFF")),
        "punt_net": _quantiles(punt_net),
        "punt_touchback_rate": round(float(punts["touchback"].fillna(0).mean()) if len(punts) else 0.07, 4),
        "field_goal": _field_goal_curve(recent),
        "xp_rate": round(float((xps["extra_point_result"] == "good").mean()) if len(xps) else 0.95, 4),
        "scrimmage_plays_per_team_game": round(league["plays"] / games / 2, 1),
        "quantile_probs": [round(float(p), 4) for p in QUANTILES],
    }

    teams = {}
    roster = _active_roster(rosters, season)
    scrimmage = _scrimmage(current)
    for team, plays in scrimmage.groupby("posteam"):
        r = _offense_rates(plays)
        teams[team] = {
            "plays": r["plays"],
            "neutral_pass_rate": round(_shrunk_rate(r["neutral_pass"], r["neutral"], lg["neutral_pass_rate"]), 4),
            "sack_rate": round(_shrunk_rate(r["sacks"], r["dropbacks"], lg["sack_rate"]), 4),
            "comp_rate": round(_shrunk_rate(r["completions"], r["attempts"], lg["comp_rate"]), 4),
            "int_rate": round(_shrunk_rate(r["ints"], r["attempts"], lg["int_rate"]), 4),
            "run_fumble_rate": round(_shrunk_rate(r["run_fumbles"], r["runs"], lg["run_fumble_rate"]), 4),
            "catch_fumble_rate": round(_shrunk_rate(r["catch_fumbles"], r["completions"], lg["catch_fumble_rate"]), 4),
            "rush_yards": _shrunk_shape(r["rush_yards"], lg["rush_yards"]),
            "completion_yards": _shrunk_shape(r["completion_yards"], lg["completion_yards"]),
            **(team_players(current, team, roster) if "receiver_player_id" in current.columns else {}),
        }
    weeks = current["week"].dropna()
    return {"season": int(season), "through_week": int(weeks.max()) if len(weeks) else 0, "league": lg, "teams": teams}


def write_profiles(profiles: dict, path: str) -> None:
    def clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    with open(path, "w") as f:
        json.dump(json.loads(json.dumps(profiles, default=clean)), f, separators=(",", ":"))

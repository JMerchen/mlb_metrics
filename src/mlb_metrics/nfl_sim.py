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
PBP_COLUMNS = [
    "game_id", "season", "season_type", "week", "posteam", "defteam", "play_type", "down", "ydstogo", "qtr",
    "score_differential", "yards_gained", "pass_attempt", "sack", "interception", "complete_pass",
    "fumble_lost", "kick_distance", "return_yards", "touchback", "field_goal_result", "extra_point_result",
    "drive", "drive_start_yard_line", "drive_start_transition",
]


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


def build_profiles(pbp: pd.DataFrame, season: int) -> dict:
    """{season, through_week, league: {...}, teams: {code: {...}}} for the
    simulation. League numbers use this and last season (kickoff rules
    changed in 2024-25, so nothing older); team numbers use this season."""
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

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
# A team's flags a snap barely carry over within a season (a half-season's
# rate correlates 0.16 with the other half's), so it takes this many snaps
# of league average to pull it in.
SHRINK_PENALTY_SNAPS = 2700
# Penalties called before the snap (no play, no clock).
PRESNAP_PENALTIES = {
    "False Start", "Delay of Game", "Defensive Offside", "Neutral Zone Infraction", "Encroachment",
    "Defensive Too Many Men on Field", "Offensive Too Many Men on Field", "Offensive Offside",
    "Defensive Delay of Game", "Illegal Substitution",
}
MIN_PENALTY_SHARE = 0.01
PBP_COLUMNS = [
    "game_id", "season", "season_type", "week", "posteam", "defteam", "play_type", "down", "ydstogo", "qtr",
    "score_differential", "yards_gained", "pass_attempt", "sack", "interception", "complete_pass",
    "fumble_lost", "kick_distance", "return_yards", "touchback", "field_goal_result", "extra_point_result",
    "drive", "drive_start_yard_line", "drive_start_transition",
    "passer_player_id", "passer_player_name", "receiver_player_id", "receiver_player_name",
    "rusher_player_id", "rusher_player_name", "kicker_player_id", "kicker_player_name",
    "penalty", "penalty_team", "penalty_type", "penalty_yards", "first_down_penalty", "desc",
    "yardline_100", "punt_fair_catch", "punt_downed", "punt_out_of_bounds", "kickoff_out_of_bounds",
    "touchdown", "qb_kneel", "qb_spike",
]
KICK_SAMPLES = 600
PUNT_SAMPLES_PER_BUCKET = 150
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


# Field zones by yards to the end zone (yardline_100). A team's own red-zone
# or backed-up numbers barely carry over within a season (odd vs even weeks,
# 2023-25: r = 0.10 inside the 20, 0.00 inside its own 10, against 0.65 for
# all its plays), so the zones are the league's, on top of each team's own
# profile.
FIELD_ZONES = {"goal": (0, 10), "red": (11, 20), "backed": (90, 99)}


def _field_zones(plays: pd.DataFrame) -> dict:
    """{zone: {pass, comp, sack}} - each zone's pass rate (at the zone's own
    mix of downs), completion rate and sack rate over the same between the
    20s - plus the zone's real per-play numbers for checking the
    simulation against: completion rate, yards per run, touchdowns per play
    and pass rate."""
    for column in ("qb_kneel", "qb_spike", "touchdown"):
        if column not in plays:
            plays = plays.assign(**{column: 0})
    plays = plays[(plays["qb_kneel"] != 1) & (plays["qb_spike"] != 1)]
    yl = plays["yardline_100"]
    middle = plays[(yl > 20) & (yl < 90)]

    def rates(g):
        passes = g[g["play_type"] == "pass"]
        attempts = passes[passes["sack"] != 1]
        runs = g[g["play_type"] == "run"]
        return {"pass": (g["play_type"] == "pass").mean(), "comp": attempts["complete_pass"].mean(),
                "sack": (passes["sack"] == 1).mean(), "run_yards": runs["yards_gained"].mean(),
                "td": (g["touchdown"] == 1).mean()}

    mid = rates(middle)
    mid_pass_by_down = middle.groupby("down")["play_type"].apply(lambda s: (s == "pass").mean())
    out = {}
    for zone, (lo, hi) in FIELD_ZONES.items():
        g = plays[(yl >= lo) & (yl <= hi)]
        if len(g) < 200:
            continue
        z = rates(g)
        downs = g["down"].value_counts(normalize=True)
        expected_pass = float(sum(share * mid_pass_by_down.get(d, mid["pass"]) for d, share in downs.items()))
        out[zone] = {
            "pass": round(z["pass"] / expected_pass, 3), "comp": round(z["comp"] / mid["comp"], 3),
            "sack": round(z["sack"] / mid["sack"], 3),
            "real": {k: round(float(v), 3) for k, v in z.items()},
        }
    out["middle"] = {"real": {k: round(float(v), 3) for k, v in mid.items()}}
    return out


def _onside_kicks(pbp: pd.DataFrame) -> pd.DataFrame:
    """Onside kickoffs; each row's (game_id, drive) is the drive it started."""
    if "desc" not in pbp.columns:
        return pbp.iloc[0:0]
    kicks = pbp[pbp["play_type"] == "kickoff"]
    return kicks[kicks["desc"].astype("string").fillna("").str.contains("onside", case=False)]


def _drive_starts(pbp: pd.DataFrame, transition: str, skip: pd.DataFrame | None = None) -> np.ndarray:
    """Yards from its own goal where the receiving team's drive started
    ("ARI 30" for ARI = 30; for the other team, 100 - 30), leaving out the
    drives in skip (rows with game_id and drive)."""
    starts = pbp[(pbp["drive_start_transition"] == transition) & pbp["drive_start_yard_line"].notna() & pbp["posteam"].notna()]
    starts = starts.drop_duplicates(["game_id", "drive"])
    if skip is not None and len(skip):
        keys = set(zip(skip["game_id"], skip["drive"]))
        starts = starts[[k not in keys for k in zip(starts["game_id"], starts["drive"])]]
    out = []
    for side, posteam in zip(starts["drive_start_yard_line"], starts["posteam"]):
        parts = str(side).split()
        if len(parts) == 1 and parts[0] == "50":
            out.append(50)
        elif len(parts) == 2 and parts[1].isdigit():
            yards = int(parts[1])
            out.append(yards if parts[0] == posteam else 100 - yards)
    return np.asarray(out, dtype=float)


def _even_rows(rows: list, n: int) -> list:
    """n rows standing for rows (sorted): the middle one of each of n equal
    slices, so a rare extreme (a 95-yard return) is kept only as often as
    it happens (all rows if there are fewer than n)."""
    if len(rows) <= n:
        return rows
    idx = ((np.arange(n) + 0.5) * len(rows) / n).astype(int)
    return [rows[i] for i in idx]


def _kickoff_model(pbp: pd.DataFrame) -> dict | None:
    """Kickoffs (onside ones left out): how often each is a touchback or
    goes out of bounds and where the receiving team then starts, and real
    [caught at, returned] pairs - yards from the receiving team's goal
    (negative: deep in the end zone) and return yards - picked evenly by
    where the return ended, so a draw keeps a deep catch's longer return."""
    if not {"kick_distance", "kickoff_out_of_bounds"} <= set(pbp.columns):
        return None
    kicks = pbp[(pbp["play_type"] == "kickoff") & pbp["kick_distance"].notna()]
    kicks = kicks[~kicks["desc"].astype("string").fillna("").str.contains("onside", case=False)]
    if len(kicks) < 100:
        return None
    touchback = kicks["touchback"].fillna(0) == 1
    out = kicks["kickoff_out_of_bounds"].fillna(0) == 1
    returned = kicks[~touchback & ~out]
    # From the kicking team's 35, 65 yards from the receiver's goal: where
    # it's caught, in receiver yards ("kicks 64 yards from TB 35 to NO 1").
    caught = (65 - returned["kick_distance"]).to_numpy(dtype=float)
    ret = returned["return_yards"].fillna(0).to_numpy(dtype=float)
    pairs = sorted(([round(float(c)), round(float(r))] for c, r in zip(caught, ret)), key=lambda p: (p[0] + p[1], p[0]))
    return {
        "touchback_rate": round(float(touchback.mean()), 4),
        "out_of_bounds_rate": round(float(out.mean()), 4),
        "touchback_spot": 35,
        "out_of_bounds_spot": 40,
        "returns": _even_rows(pairs, KICK_SAMPLES),
    } if len(pairs) >= 50 else None


def _punt_model(pbp: pd.DataFrame) -> dict | None:
    """Real punts, [gross yards, how it ended, return yards], in buckets of
    where they were kicked from (yards from the punting team's goal, by
    tens) - a punt from midfield is aimed short of the end zone, one from
    its own 20 isn't. Endings: r returned, f fair catch, d downed, o out of
    bounds, t touchback."""
    needed = {"kick_distance", "yardline_100", "punt_fair_catch", "punt_downed", "punt_out_of_bounds", "touchback"}
    if not needed <= set(pbp.columns):
        return None
    punts = pbp[(pbp["play_type"] == "punt") & pbp["kick_distance"].notna() & pbp["yardline_100"].notna()]
    if len(punts) < 200:
        return None
    end = np.select(
        [punts["touchback"].fillna(0) == 1, punts["punt_fair_catch"].fillna(0) == 1,
         punts["punt_downed"].fillna(0) == 1, punts["punt_out_of_bounds"].fillna(0) == 1],
        ["t", "f", "d", "o"], "r")
    frm = (100 - punts["yardline_100"]).to_numpy(dtype=float)
    rows = sorted(zip(frm, punts["kick_distance"].to_numpy(dtype=float), end, punts["return_yards"].fillna(0).to_numpy(dtype=float)),
                  key=lambda r: (r[1] - (r[3] if r[2] == "r" else 0), r[1]))
    buckets = {}
    for f, gross, e, r in rows:
        buckets.setdefault(str(int(min(9, max(0, f // 10)))), []).append([round(float(gross)), str(e), round(float(r)) if e == "r" else 0])
    return {"buckets": {k: _even_rows(v, PUNT_SAMPLES_PER_BUCKET) for k, v in sorted(buckets.items())}}


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


def _snaps_with_flags(pbp: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(every scrimmage snap, including ones a penalty wiped out; the
    accepted penalties on them, each with side "O" or "D")."""
    snaps = pbp[(pbp["season_type"] == "REG") & pbp["play_type"].isin(["pass", "run", "no_play"])
                & pbp["down"].notna() & pbp["posteam"].notna()]
    if "penalty_type" not in snaps.columns:
        return snaps, snaps.iloc[0:0].assign(side=pd.Series(dtype=str))
    declined = snaps["desc"].astype("string").fillna("").str.contains("declined|offsetting", case=False)
    flags = snaps[(snaps["penalty"] == 1) & snaps["penalty_yards"].notna() & snaps["penalty_type"].notna() & ~declined]
    side = np.where(flags["penalty_team"] == flags["posteam"], "O", np.where(flags["penalty_team"] == flags["defteam"], "D", ""))
    flags = flags.assign(side=side)
    return snaps, flags[flags["side"] != ""]


def _penalty_league(snaps: pd.DataFrame, flags: pd.DataFrame) -> dict:
    """Rates a snap for each side, and each side's penalties (those at
    MIN_PENALTY_SHARE or more, renormalized): {name, side, share, yards
    (typical), first_down (chance it gives one), presnap (no play), after
    (the play stands and the yards are added - personal fouls)}. DPI is
    a spot foul: dpi_yards holds its quantiles."""
    n = max(1, len(snaps))
    types = []
    for side in ("O", "D"):
        mine = flags[flags["side"] == side]
        if mine.empty:
            continue
        by = mine.groupby("penalty_type").agg(
            n=("penalty_yards", "size"), yards=("penalty_yards", "median"),
            first_down=("first_down_penalty", "mean"), nullified=("play_type", lambda x: (x == "no_play").mean()))
        by = by[by["n"] / by["n"].sum() >= MIN_PENALTY_SHARE]
        for name, row in by.iterrows():
            types.append({
                "name": name, "side": side, "share": round(float(row["n"] / by["n"].sum()), 4),
                "yards": int(round(float(row["yards"]))), # An automatic first down; a pre-snap foul's 5 yards give one
                # only when they reach the line, which the simulation sees.
                # Measured at 0.94-1.0 for the automatic ones (the rest are
                # flags on plays that already made the line): call them 1.
                "first_down": (1.0 if row["first_down"] >= 0.9 else round(float(row["first_down"]), 3))
                if side == "D" and name not in PRESNAP_PENALTIES else 0.0,
                "presnap": name in PRESNAP_PENALTIES,
                "after": bool(side == "D" and row["nullified"] < 0.5),
            })
    dpi = flags.loc[flags["penalty_type"] == "Defensive Pass Interference", "penalty_yards"]
    return {
        "offense_rate": round(float((flags["side"] == "O").sum() / n), 4),
        "defense_rate": round(float((flags["side"] == "D").sum() / n), 4),
        "types": types,
        "dpi_yards": _quantiles(dpi) if len(dpi) >= 20 else [],
    }


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

    snaps, flags = _snaps_with_flags(recent)
    # Onside kicks come only from a team behind late, and most hand the
    # other side the ball near midfield: kept out of the kickoff starts (they
    # were half of all kickoff drives starting past midfield) and modeled on
    # their own.
    onside = _onside_kicks(recent)
    recovered = int(onside["drive_start_transition"].eq("ONSIDE_KICK").sum()) if "drive_start_transition" in onside else 0
    failed = set(zip(onside["game_id"], onside["drive"]))
    after_failed = recent[[k in failed for k in zip(recent["game_id"], recent["drive"])]]
    failed_starts = _drive_starts(after_failed, "KICKOFF")
    lg_onside = {
        # One pseudo-recovery in ten pseudo-tries keeps a short sample honest.
        "recovery_rate": round((recovered + 1) / (len(onside) + 10), 4),
        "attempts": int(len(onside)),
        # Where the receiving team starts after one fails.
        "failed_start": _quantiles(failed_starts) if len(failed_starts) >= 20 else [],
    }
    lg = {
        "onside": lg_onside,
        "kickoffs": _kickoff_model(recent),
        "punts": _punt_model(recent),
        "penalties": _penalty_league(snaps, flags),
        "neutral_pass_rate": round(league["neutral_pass"] / max(1, league["neutral"]), 4),
        "sack_rate": round(league["sacks"] / max(1, league["dropbacks"]), 4),
        "comp_rate": round(league["completions"] / max(1, league["attempts"]), 4),
        "int_rate": round(league["ints"] / max(1, league["attempts"]), 4),
        "run_fumble_rate": round(league["run_fumbles"] / max(1, league["runs"]), 4),
        "catch_fumble_rate": round(league["catch_fumbles"] / max(1, league["completions"]), 4),
        "sack_yards": round(float(np.mean(league["sack_yards"])) if len(league["sack_yards"]) else -7.0, 2),
        "rush_yards": _quantiles(league["rush_yards"]),
        "completion_yards": _quantiles(league["completion_yards"]),
        "field_zones": _field_zones(_scrimmage(recent)),
        "kickoff_start": _quantiles(_drive_starts(recent, "KICKOFF", onside)),
        "punt_net": _quantiles(punt_net),
        "punt_touchback_rate": round(float(punts["touchback"].fillna(0).mean()) if len(punts) else 0.07, 4),
        "field_goal": _field_goal_curve(recent),
        "xp_rate": round(float((xps["extra_point_result"] == "good").mean()) if len(xps) else 0.95, 4),
        "scrimmage_plays_per_team_game": round(league["plays"] / games / 2, 1),
        "quantile_probs": [round(float(p), 4) for p in QUANTILES],
    }

    teams = {}
    roster = _active_roster(rosters, season)
    pen = lg["penalties"]
    cur_snaps = snaps[snaps["season"] == season]
    cur_flags = flags[flags["season"] == season]
    off_snaps, def_snaps = cur_snaps["posteam"].value_counts(), cur_snaps["defteam"].value_counts()
    off_flags = cur_flags[cur_flags["side"] == "O"]["posteam"].value_counts()
    def_flags = cur_flags[cur_flags["side"] == "D"]["defteam"].value_counts()
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
            "offense_penalty_rate": round(_shrunk_rate(off_flags.get(team, 0), off_snaps.get(team, 0), pen["offense_rate"], SHRINK_PENALTY_SNAPS), 4),
            "defense_penalty_rate": round(_shrunk_rate(def_flags.get(team, 0), def_snaps.get(team, 0), pen["defense_rate"], SHRINK_PENALTY_SNAPS), 4),
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

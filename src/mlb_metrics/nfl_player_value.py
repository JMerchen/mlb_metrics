"""Our own wins above replacement for NFL players, from play-by-play.

Why (2026-11-23, user request: "having our own version of war for the nfl
would be cool, especially if it could be in a viewable table that could be
filtered by position or offense/defense or year in the league or
conference/division"): the site rated teams and units, and quarterbacks by
their own EPA, but had no single number for what a player is worth.

The problem a WAR has to solve is CREDIT. A completed pass has one value
(its EPA, expected points added) and at least three people behind it: the
quarterback, the receiver and the defense across from them. Giving the
whole play to each of them - which is what EPA per target or per dropback
does - makes every receiver of a great quarterback look great.

The method follows nflWAR (Yurko, Ventura and Horowitz, 2019), simplified
where our data is simpler:

1. Every dropback's EPA is split into three parts that add back up to it:
     no-target part   the play's EPA when nobody was thrown to (a sack, a
                      scramble, a throwaway), less what an average
                      targeted throw is worth - the throw that never came
     throw part       a targeted throw up to the catch: for a completion,
                      the value of the ball in the air plus the yards
                      after the catch a typical receiver gets from there
                      (nflfastR's air_epa + xyac_epa); for an incompletion
                      or an interception, the whole play
     after-catch part what the receiver added past that (yac_epa -
                      xyac_epa)
   A designed run is one part, its EPA.
2. Each part is a regression with an effect for everyone involved:
     no-target    quarterback, his team (protection), the defense
     throw        quarterback, receiver, the defense
     after catch  receiver, quarterback, the defense
     run          runner, his team (blocking), the defense
   The effects are fit together, so a receiver is measured after allowing
   for who threw to him and whom he played. Each effect is pulled toward
   the average of players like him - same position, starter or reserve
   (TIERS) - by an amount the data chooses (a mixed model fit by EM): a
   player with 30 targets stays near that average, one with 130 earns his
   own number.
3. A player's POINTS ABOVE AVERAGE is his effect times his plays, summed
   over the parts he appears in. POINTS ABOVE REPLACEMENT measures him
   against the reserve tier at his position instead - what a team would
   get from the next man up - and WAR divides by POINTS_PER_WIN.

Defenders are different, and the table says so. The play-by-play names a
defender only when he makes a play - a sack, a tackle, a pass broken up -
so a cornerback nobody throws at is invisible. What can be measured is
PLAYMAKING: the points his credited plays took away from offenses, per
snap, against a reserve at his position, scaled by how much of a team's
defensive results playmaking explains (DEFENSE_SCALE). Offensive linemen
are not rated: nothing in the data says who blocked whom.

scripts/research_nfl_player_value.py has the checks: how well a player's
number carries over to his next season against plain EPA per play, and
whether a team's players add up to its results.
"""

import numpy as np
import pandas as pd

from mlb_metrics import config

PLAY_COLUMNS = [
    "game_id", "play_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "down", "epa",
    "qb_dropback", "qb_scramble", "qb_kneel", "qb_spike", "complete_pass", "sack", "interception",
    "air_epa", "yac_epa", "xyac_epa", "passer_id", "receiver_player_id", "rusher_player_id", "xpass", "wp",
]
# What EPA does not know about a play's setting, allowed for in every
# part: how obvious a pass was (nflfastR's expected pass rate - a run when
# everyone expects a pass gains more, a pass when everyone expects one
# less) and how decided the game was (the offense's win probability).
# Reserves get their plays in different settings from starters - late, in
# games already decided - and without this that reads as talent.
XPASS_EDGES = [0.2, 0.35, 0.5, 0.65, 0.8, 0.9]
WP_EDGES = [0.05, 0.2, 0.8, 0.95]
# How many players at a position on a team count as starters in each role
# (by plays in that role); everyone else there is the reserve tier, which
# is the replacement level. nflWAR's depth-chart definition.
STARTERS = {
    "pass": {"QB": 1},
    "rush": {"RB": 2, "QB": 1},
    "catch": {"WR": 3, "TE": 1, "RB": 2},
}
# Positions folded together for tiers and the table.
POSITION_GROUP = {"QB": "QB", "RB": "RB", "HB": "RB", "FB": "RB", "WR": "WR", "TE": "TE"}
# Points of season point differential per win: wins on point differential
# over the 320 team-seasons of 2016-2025, 0.0283 wins a point (r 0.89).
POINTS_PER_WIN = 35.0
# REPLACEMENT LEVEL: how many points a play a reserve is worth less than a
# starter at his position, by part and role - the gap between the two
# tiers when the tiers are estimated freely (fit_offense(gaps=None)),
# pooled over 2016-2025 and weighted by the reserves' plays
# (scripts/research_nfl_player_value.py; standard errors .005 to .02,
# .07 for a quarterback's designed runs). Constants rather than each
# season's own estimate because one season's is mostly noise: the wide
# receivers' throw gap ran from 0.03 to 0.20 season to season.
REPLACEMENT_GAP = {
    ("no_target", "pass", "QB"): 0.025, ("throw", "pass", "QB"): 0.123, ("after_catch", "pass", "QB"): 0.035,
    ("run", "rush", "RB"): 0.055, ("run", "rush", "QB"): 0.199,
    ("throw", "catch", "WR"): 0.122, ("after_catch", "catch", "WR"): 0.029,
    ("throw", "catch", "TE"): 0.051, ("after_catch", "catch", "TE"): 0.007,
    ("throw", "catch", "RB"): 0.061, ("after_catch", "catch", "RB"): 0.046,
}


def position_group(position) -> str:
    return POSITION_GROUP.get(str(position), "OTH")


def mixed_fit(y: np.ndarray, fixed: list, random: list, iterations: int = 200, tolerance: float = 1e-6) -> dict:
    """A linear mixed model by EM: y = fixed effects + one random effect
    per level of each factor + noise.

    `fixed` and `random` are lists of (name, codes, levels): `codes` gives
    each row's level (an int, or -1 where the factor does not apply to the
    row). Fixed effects are estimated freely; a random factor's effects
    share a variance, which is what pulls a level with few rows toward
    zero. Returns {"fixed": {name: array}, "random": {name: array},
    "variance": {name: float}, "noise": float}."""
    y = np.asarray(y, dtype=float)
    n = len(y)
    blocks, start = [], 0
    for kind, factors in (("fixed", fixed), ("random", random)):
        for name, codes, levels in factors:
            blocks.append((kind, name, np.asarray(codes), int(levels), start))
            start += int(levels)
    p = start
    gram = np.zeros((p, p))
    moment = np.zeros(p)
    for i, (_, _, codes_i, levels_i, start_i) in enumerate(blocks):
        on_i = codes_i >= 0
        moment[start_i:start_i + levels_i] = np.bincount(codes_i[on_i], weights=y[on_i], minlength=levels_i)
        for _, _, codes_j, levels_j, start_j in blocks[i:]:
            both = on_i & (codes_j >= 0)
            cross = np.zeros((levels_i, levels_j))
            np.add.at(cross, (codes_i[both], codes_j[both]), 1.0)
            gram[start_i:start_i + levels_i, start_j:start_j + levels_j] = cross
            gram[start_j:start_j + levels_j, start_i:start_i + levels_i] = cross.T
    total = float(y @ y)
    noise = float(np.var(y)) or 1.0
    variance = {name: noise * 0.01 for kind, name, *_ in blocks if kind == "random"}
    beta = np.zeros(p)
    for _ in range(iterations):
        penalty = np.full(p, 1e-8)
        for kind, name, _, levels, at in blocks:
            if kind == "random":
                penalty[at:at + levels] = noise / max(variance[name], 1e-12)
        inverse = np.linalg.inv(gram + np.diag(penalty))
        beta = inverse @ moment
        spread = np.diag(inverse)
        previous = dict(variance)
        for kind, name, _, levels, at in blocks:
            if kind == "random" and levels:
                variance[name] = float((beta[at:at + levels] @ beta[at:at + levels] + noise * spread[at:at + levels].sum()) / levels)
        residual = total - 2 * beta @ moment + beta @ gram @ beta
        noise = float((residual + noise * np.sum(inverse * gram)) / n)
        if all(abs(variance[k] - previous[k]) <= tolerance * max(previous[k], 1e-12) for k in variance):
            break
    out = {"fixed": {}, "random": {}, "variance": variance, "noise": noise}
    for kind, name, _, levels, at in blocks:
        out[kind][name] = beta[at:at + levels]
    return out


def offense_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Regular-season dropbacks and designed runs, each with its parts
    (no_target, throw, after_catch, run: they add up to the play's EPA
    less, on a dropback with no target, an average throw's value) and who
    did what: qb, receiver, rusher."""
    p = pbp[(pbp["season_type"] == "REG") & pbp["play_type"].isin(["run", "pass"]) & pbp["epa"].notna()
            & pbp["down"].notna() & pbp["posteam"].notna() & pbp["defteam"].notna()
            & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)].copy()
    dropback = (p["qb_dropback"] == 1) & p["passer_id"].notna()
    target = dropback & p["receiver_player_id"].notna()
    complete = target & (p["complete_pass"] == 1)
    run = ~dropback & p["rusher_player_id"].notna()
    p = p[dropback | run].copy()
    dropback, target, complete, run = (s.loc[p.index] for s in (dropback, target, complete, run))
    xyac = p["xyac_epa"].where(complete)
    xyac = xyac.fillna(xyac.mean()).where(complete, 0.0)
    throw = np.where(complete, p["air_epa"].fillna(0.0) + xyac, p["epa"])
    p["throw"] = np.where(target, throw, 0.0)
    p["after_catch"] = np.where(complete, p["yac_epa"].fillna(0.0) - xyac, 0.0)
    average_throw = float(p.loc[target, "throw"].mean()) if target.any() else 0.0
    p["no_target"] = np.where(dropback & ~target, p["epa"] - average_throw, 0.0)
    p["run"] = np.where(run, p["epa"], 0.0)
    p["kind"] = np.where(run, "run", np.where(target, "target", "no_target"))
    p["complete"] = complete
    p["qb"] = p["passer_id"].where(dropback)
    p["receiver"] = p["receiver_player_id"].where(target)
    p["rusher"] = p["rusher_player_id"].where(run)
    p["xpass_bin"] = np.digitize(p["xpass"].fillna(0.5), XPASS_EDGES)
    p["wp_bin"] = np.digitize(p["wp"].fillna(0.5), WP_EDGES)
    return p[["game_id", "play_id", "season", "week", "posteam", "defteam", "kind", "complete", "epa",
              "no_target", "throw", "after_catch", "run", "qb", "receiver", "rusher", "xpass_bin", "wp_bin"]].reset_index(drop=True)


def roster_facts(rosters: pd.DataFrame) -> pd.DataFrame:
    """One row per player (indexed by gsis_id) from a season's weekly
    rosters: name, position, pfr_id, years_exp, and his latest team."""
    r = rosters[rosters["gsis_id"].notna()].sort_values("week")
    last = r.groupby("gsis_id").last()
    position = r.groupby("gsis_id")["position"].agg(lambda s: s.mode().iloc[0] if not s.mode().empty else None)
    return pd.DataFrame({"name": last["full_name"], "position": position, "pfr_id": last["pfr_id"],
                         "years_exp": last["years_exp"], "team": last["team"]})


def _tiers(plays: pd.DataFrame, column: str, role: str, positions: pd.Series) -> pd.Series:
    """Each row's tier in a role - "QB1", "QBx", "WR1", "WRx", ..., "OTH" -
    from the player's position and whether he is among his team's starters
    in that role by plays (STARTERS). "" where the role does not apply."""
    rows = plays[plays[column].notna()]
    group = rows[column].map(positions).map(position_group)
    counts = rows.assign(group=group).groupby(["posteam", "group", column]).size().rename("n").reset_index()
    counts["rank"] = counts.groupby(["posteam", "group"])["n"].rank(method="first", ascending=False)
    limit = counts["group"].map(STARTERS[role])
    counts["tier"] = np.where(limit.isna(), "OTH", counts["group"] + np.where(counts["rank"] <= limit.fillna(0), "1", "x"))
    key = pd.MultiIndex.from_frame(counts[["posteam", column]])
    lookup = pd.Series(counts["tier"].to_numpy(), index=key)
    tier = pd.Series("", index=plays.index, dtype=object)
    tier.loc[rows.index] = lookup.reindex(pd.MultiIndex.from_frame(rows[["posteam", column]])).to_numpy()
    return tier


def _codes(values: pd.Series) -> tuple:
    """(codes, levels, index): -1 where missing or ""."""
    clean = values.where(values.notna() & (values != ""))
    codes, index = pd.factorize(clean)
    return codes, len(index), index


# Each part's model: the rows it is fit on, and who has an effect in it.
# (role column, tier role or None for a team effect)
PARTS = {
    "no_target": {"rows": ["no_target", "target"], "players": [("qb", "pass")], "teams": ["posteam", "defteam"]},
    "throw": {"rows": ["target"], "players": [("qb", "pass"), ("receiver", "catch")], "teams": ["defteam"]},
    "after_catch": {"rows": ["target"], "complete": True, "players": [("receiver", "catch"), ("qb", "pass")], "teams": ["defteam"]},
    "run": {"rows": ["run"], "players": [("rusher", "rush")], "teams": ["posteam", "defteam"]},
}


def _tier_position(tier: str) -> str:
    return tier if tier == "OTH" else tier[:-1]


def fit_offense(plays: pd.DataFrame, positions: pd.Series, context: bool = True, gaps: dict | None = REPLACEMENT_GAP) -> pd.DataFrame:
    """One row per (player, team, part, role) for one season's plays:
    plays, tier, effect (points a play against the average play of that
    part), average (his position's own average, on the same scale),
    replacement (a reserve's effect at his position) and raw (his plays'
    average of the part, unadjusted).

    With `gaps` (the default, REPLACEMENT_GAP) a starter is pulled toward
    his position's level and a reserve toward that level less the gap, and
    replacement is the level less the gap. With gaps=None each tier's
    level is estimated from the season itself - how the gaps were
    measured."""
    plays = plays.copy()
    tier_of = {"pass": _tiers(plays, "qb", "pass", positions), "rush": _tiers(plays, "rusher", "rush", positions),
               "catch": _tiers(plays, "receiver", "catch", positions)}
    out = []
    for part, spec in PARTS.items():
        rows = plays["kind"].isin(spec["rows"])
        if spec.get("complete"):
            rows &= plays["complete"]
        d = plays[rows]
        if len(d) < 50:
            continue
        y = d[part].to_numpy(dtype=float)
        target = y.copy()
        fixed, random, players = [], [], []
        for column, role in spec["players"]:
            tier = tier_of[role].loc[d.index]
            level = tier if gaps is None else tier.map(_tier_position)
            # A reserve's known deficit comes off before the fit and goes
            # back on after, so he is pulled toward a reserve's level.
            deficit = np.zeros(len(d)) if gaps is None else np.where(
                tier.str.endswith("x"), tier.map(_tier_position).map(lambda pos: gaps.get((part, role, pos), 0.0)).fillna(0.0), 0.0)
            target = target + deficit
            level_codes, levels_n, level_index = _codes(level)
            codes, n_players, index = _codes(d[column] + "|" + d["posteam"])
            # The first role's levels stand in for the intercept; each later
            # role's first level is the baseline, or the columns would repeat it.
            fixed.append((f"{column}_level", level_codes - 1, levels_n - 1) if fixed else (f"{column}_level", level_codes, levels_n))
            random.append((column, codes, n_players))
            players.append((column, role, codes, index, level_codes, level_index, tier.to_numpy(), deficit))
        for column in spec["teams"]:
            codes, levels_n, _ = _codes(d[column])
            random.append((f"team_{column}", codes, levels_n))
        if context:
            for column in ("xpass_bin", "wp_bin"):
                # The first bin present is the baseline.
                codes, levels_n, _ = _codes(d[column].astype(str))
                fixed.append((column, codes - 1, max(levels_n - 1, 0)))
        fit = mixed_fit(target, fixed, random)
        for column, role, codes, index, level_codes, level_index, tier, deficit in players:
            level_effect = fit["fixed"][f"{column}_level"]
            if len(level_effect) < len(level_index):
                level_effect = np.concatenate([[0.0], level_effect])
            on = codes >= 0
            n = np.bincount(codes[on], minlength=len(index))
            raw = np.bincount(codes[on], weights=y[on], minlength=len(index)) / np.maximum(n, 1)
            first = pd.Series(np.arange(len(codes))[on]).groupby(codes[on]).first().reindex(range(len(index))).to_numpy()
            own_level, own_tier, own_deficit = level_codes[first], tier[first], deficit[first]
            total = level_effect[own_level] + fit["random"][column] - own_deficit
            # Against the average play of this part, for this role.
            center = float((total * n).sum() / max(n.sum(), 1))
            if gaps is None:
                by_tier = pd.Series(level_effect, index=level_index)
                reserve = [t[:-1] + "x" if t.endswith("1") else t for t in own_tier]
                # A position with no reserves on the field yet: its own level.
                replacement = np.array([by_tier.get(r, by_tier[t]) for r, t in zip(reserve, own_tier)])
            else:
                replacement = level_effect[own_level] - np.array([gaps.get((part, role, _tier_position(t)), 0.0) for t in own_tier])
            # His position's own average play, for "above average".
            position = pd.Series([_tier_position(t) for t in own_tier])
            sums = pd.DataFrame({"position": position, "points": total * n, "n": n}).groupby("position")[["points", "n"]].sum()
            average = position.map(sums["points"] / sums["n"].clip(lower=1)).to_numpy()
            ids = pd.Series(index).str.split("|", expand=True)
            out.append(pd.DataFrame({
                "player_id": ids[0], "team": ids[1], "part": part, "role": role, "tier": own_tier, "plays": n,
                "effect": total - center, "average": average - center, "replacement": replacement - center,
                "raw": raw - float(y.mean()),
            }))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(
        columns=["player_id", "team", "part", "role", "tier", "plays", "effect", "average", "replacement", "raw"])


def offense_value(effects: pd.DataFrame) -> pd.DataFrame:
    """One row per player from fit_offense's rows: points above the
    average at his position and above replacement, split into passing,
    rushing and receiving, his plays in each, and WAR."""
    e = effects.copy()
    e["paa"] = (e["effect"] - e["average"]) * e["plays"]
    e["par"] = (e["effect"] - e["replacement"]) * e["plays"]
    e["area"] = e["role"].map({"pass": "passing", "rush": "rushing", "catch": "receiving"})
    # A role's plays are counted once: its biggest part (a quarterback's
    # dropbacks, a receiver's targets, a runner's carries).
    volume = e.groupby(["player_id", "area", "part"])["plays"].sum().groupby(["player_id", "area"]).max().unstack("area").fillna(0)
    points = e.groupby(["player_id", "area"])[["paa", "par"]].sum().unstack("area").fillna(0.0)
    out = pd.DataFrame(index=points.index)
    for area in ("passing", "rushing", "receiving"):
        out[f"{area}_plays"] = volume[area].astype(int) if area in volume.columns else 0
        out[f"{area}_paa"] = points[("paa", area)] if ("paa", area) in points.columns else 0.0
        out[f"{area}_par"] = points[("par", area)] if ("par", area) in points.columns else 0.0
    out["plays"] = out[[f"{a}_plays" for a in ("passing", "rushing", "receiving")]].sum(axis=1)
    out["paa"] = out[[f"{a}_paa" for a in ("passing", "rushing", "receiving")]].sum(axis=1)
    out["par"] = out[[f"{a}_par" for a in ("passing", "rushing", "receiving")]].sum(axis=1)
    out["war"] = out["par"] / POINTS_PER_WIN
    return out.reset_index()


def load_table(table: str, season: int, raw_dir: str | None = None, columns: list | None = None) -> pd.DataFrame:
    import os

    path = os.path.join(raw_dir or config.NFL_RAW_DATA_DIR, f"{table}_{season}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame(columns=columns or [])
    return pd.read_parquet(path, columns=columns)


# ---- defenders: playmaking ----

DEFENSE_COLUMNS = [
    "sack_player_id", "half_sack_1_player_id", "half_sack_2_player_id", "interception_player_id",
    "pass_defense_1_player_id", "pass_defense_2_player_id", "forced_fumble_player_1_player_id",
    "forced_fumble_player_2_player_id", "tackle_for_loss_1_player_id", "tackle_for_loss_2_player_id",
    "qb_hit_1_player_id", "qb_hit_2_player_id", "solo_tackle_1_player_id", "solo_tackle_2_player_id",
    "assist_tackle_1_player_id", "assist_tackle_2_player_id", "assist_tackle_3_player_id", "assist_tackle_4_player_id",
    "tackle_with_assist_1_player_id", "tackle_with_assist_2_player_id",
]
# What each credit column counts as in the table.
DEFENSE_KIND = {"sack": "sacks", "half_sack": "sacks", "interception": "interceptions", "pass_defense": "passes_defended",
                "forced_fumble": "forced_fumbles", "tackle_for_loss": "tackles_for_loss", "qb_hit": "qb_hits",
                "solo_tackle": "tackles", "assist_tackle": "tackles", "tackle_with_assist": "tackles"}
DEFENSE_GROUP = {"DT": "DT", "NT": "DT", "DE": "ED", "OLB": "LB", "ILB": "LB", "MLB": "LB", "LB": "LB",
                 "CB": "CB", "FS": "S", "SS": "S", "S": "S", "DB": "S"}
DEFENSE_STARTERS = {"DT": 2, "ED": 2, "LB": 2, "CB": 3, "S": 2}
# An outside linebacker whose credited plays are this much sacks and hits
# on the quarterback is an edge rusher, whatever the depth chart calls him.
EDGE_RUSH_SHARE = 0.25


def defense_credits(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per (play, defender) for every regular-season run or pass
    that lost the offense expected points and names a defender: points
    (the play's loss to the offense, split evenly among the defenders
    named on it) and a count per kind of play (1 or 0; a shared sack is
    half)."""
    p = pbp[(pbp["season_type"] == "REG") & pbp["play_type"].isin(["run", "pass"]) & pbp["epa"].notna()
            & pbp["down"].notna() & pbp["defteam"].notna() & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1) & (pbp["epa"] < 0)]
    columns = [c for c in DEFENSE_COLUMNS if c in p.columns]
    long = p[["game_id", "play_id", "defteam", "epa"] + columns].melt(
        id_vars=["game_id", "play_id", "defteam", "epa"], value_vars=columns, var_name="column", value_name="player_id").dropna(subset=["player_id"])
    long["kind"] = long["column"].str.replace(r"(_player)?(_\d)?_player_id$", "", regex=True).map(DEFENSE_KIND)
    kinds = sorted(set(DEFENSE_KIND.values()))
    flags = (pd.crosstab([long["game_id"], long["play_id"], long["player_id"]], long["kind"]).reindex(columns=kinds, fill_value=0) > 0).astype(float)
    half = long[long["column"].str.startswith("half_sack")]
    if len(half):
        flags.loc[pd.MultiIndex.from_frame(half[["game_id", "play_id", "player_id"]]), "sacks"] = 0.5
    out = flags.reset_index()
    first = long.drop_duplicates(["game_id", "play_id", "player_id"]).set_index(["game_id", "play_id", "player_id"])
    key = pd.MultiIndex.from_frame(out[["game_id", "play_id", "player_id"]])
    out["team"] = first["defteam"].reindex(key).to_numpy()
    named = out.groupby(["game_id", "play_id"])["player_id"].transform("size")
    out["points"] = -first["epa"].reindex(key).to_numpy() / named
    return out


def defense_snaps(snaps: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """Regular-season defensive snaps and games per (player_id, team).
    The snap table is keyed by Pro-Football-Reference's id; the roster
    crosses it to the play-by-play's, by id and failing that by name and
    team."""
    s = snaps[(snaps["game_type"] == "REG") & (snaps["defense_snaps"] > 0)]
    by_pfr = roster.reset_index().dropna(subset=["pfr_id"]).drop_duplicates("pfr_id").set_index("pfr_id")["gsis_id"]
    named = roster.reset_index().assign(key=lambda r: r["name"].str.lower().str.replace(r"[^a-z]", "", regex=True))
    by_name = named.drop_duplicates("key", keep=False).set_index("key")["gsis_id"]
    player = s["pfr_player_id"].map(by_pfr)
    player = player.fillna(s["player"].str.lower().str.replace(r"[^a-z]", "", regex=True).map(by_name))
    s = s.assign(player_id=player).dropna(subset=["player_id"])
    return s.groupby(["player_id", "team"]).agg(snaps=("defense_snaps", "sum"), games=("game_id", "nunique")).reset_index()


def defense_table(credits: pd.DataFrame, snaps: pd.DataFrame, roster: pd.DataFrame, depth: pd.Series) -> pd.DataFrame:
    """One row per defender with snaps: his position group, team (the one
    he played most for), snaps, games, starter or reserve, each kind of
    play made, and points (what his credited plays took off offenses)."""
    kinds = sorted(set(DEFENSE_KIND.values()))
    made = credits.groupby("player_id")[kinds + ["points"]].sum()
    by_team = snaps.sort_values("snaps", ascending=False)
    main = by_team.drop_duplicates("player_id").set_index("player_id")["team"]
    total = by_team.groupby("player_id")[["snaps", "games"]].sum()
    out = total.join(made, how="left").fillna(0.0)
    out["team"] = main
    position = depth.reindex(out.index)
    group = position.map(DEFENSE_GROUP)
    rush = (out["sacks"] + out["qb_hits"]) / (out[["sacks", "qb_hits", "tackles", "passes_defended", "interceptions"]].sum(axis=1)).clip(lower=1)
    group = group.mask((position == "OLB") & (rush >= EDGE_RUSH_SHARE), "ED")
    out["position"] = group
    out = out[out["position"].notna()].copy()
    rank = out.groupby(["team", "position"])["snaps"].rank(method="first", ascending=False)
    out["starter"] = rank <= out["position"].map(DEFENSE_STARTERS)
    out["name"] = roster["name"].reindex(out.index)
    return out.reset_index()


# How much of his position's average a defender's rate is pulled toward,
# in snaps: the pull that best predicted his next season's rate, 2016-2025
# (the error stops falling past about 500).
DEFENSE_SHRINK_SNAPS = 600.0
# A reserve's playmaking points a snap below a starter's, pooled 2016-2025
# (safeties: reserves were no lower, so none).
DEFENSE_REPLACEMENT_GAP = {"DT": 0.0064, "ED": 0.0074, "LB": 0.0029, "CB": 0.0021, "S": 0.0}
# Team by team, a point of credited playmaking above the positions'
# averages went with 1.7 points of defensive EPA saved (320 team-seasons,
# r 0.83): a defense that makes a stop has also not given up the gain.
DEFENSE_SCALE = 1.7


def defense_value(table: pd.DataFrame) -> pd.DataFrame:
    """defense_table's rows with paa, par and war: each defender's
    playmaking points a snap, pulled toward his position's average,
    against that average and against a reserve, times his snaps and
    DEFENSE_SCALE."""
    t = table.copy()
    if t.empty:
        return t.assign(paa=[], par=[], war=[])
    sums = t.groupby("position")[["points", "snaps"]].sum()
    average = t["position"].map(sums["points"] / sums["snaps"])
    starters = t[t["starter"]].groupby("position")[["points", "snaps"]].sum()
    starter_rate = t["position"].map(starters["points"] / starters["snaps"]).fillna(average)
    rate = (t["points"] + DEFENSE_SHRINK_SNAPS * average) / (t["snaps"] + DEFENSE_SHRINK_SNAPS)
    replacement = starter_rate - t["position"].map(DEFENSE_REPLACEMENT_GAP)
    t["paa"] = DEFENSE_SCALE * (rate - average) * t["snaps"]
    t["par"] = DEFENSE_SCALE * (rate - replacement) * t["snaps"]
    t["war"] = t["par"] / POINTS_PER_WIN
    return t


# ---- kickers ----

# Kicks of pull toward an average kicker: a kicker's points over expected
# barely carry over (r 0.09 between seasons of 40+ kicks), and the error
# predicting his next season stops falling at about 200.
KICKER_SHRINK_KICKS = 200.0
# Points a kick a team's fill-in kickers were below its main one, 2016-2025
# (1,846 kicks).
KICKER_REPLACEMENT_GAP = 0.134


def kicker_value(pbp: pd.DataFrame) -> pd.DataFrame:
    """One row per kicker: kicks, field goals made and tried, and points
    over expected - 3 x (made - nflfastR's chance of making it) on each
    field goal, and (made - the league's rate) on each extra point -
    pulled toward zero, against average (paa) and a fill-in (par, war)."""
    p = pbp[(pbp["season_type"] == "REG") & pbp["kicker_player_id"].notna() & pbp["play_type"].isin(["field_goal", "extra_point"])]
    if p.empty:
        return pd.DataFrame(columns=["player_id", "team", "kicks", "fg_made", "fg_tried", "paa", "par", "war"])
    fg = p["play_type"] == "field_goal"
    made = np.where(fg, p["field_goal_result"] == "made", p["extra_point_result"] == "good").astype(float)
    extra_rate = float(made[~fg.to_numpy()].mean()) if (~fg).any() else 0.94
    expected = np.where(fg, p["fg_prob"].fillna(made[fg.to_numpy()].mean() if fg.any() else 0.85), extra_rate)
    d = pd.DataFrame({"player_id": p["kicker_player_id"].to_numpy(), "team": p["posteam"].to_numpy(),
                      "points": np.where(fg, 3.0, 1.0) * (made - expected), "fg": fg.to_numpy(), "made": made})
    out = d.groupby("player_id").agg(kicks=("points", "size"), raw=("points", "sum"), fg_tried=("fg", "sum"),
                                     fg_made=("made", lambda m: float((m * d.loc[m.index, "fg"]).sum())))
    out["team"] = d.groupby("player_id")["team"].agg(lambda s: s.mode().iloc[0])
    out["paa"] = out["raw"] * out["kicks"] / (out["kicks"] + KICKER_SHRINK_KICKS)
    out["par"] = out["paa"] + KICKER_REPLACEMENT_GAP * out["kicks"]
    out["war"] = out["par"] / POINTS_PER_WIN
    return out.reset_index()


# ---- the table ----

DIVISIONS = {
    "AFC East": ["BUF", "MIA", "NE", "NYJ"], "AFC North": ["BAL", "CIN", "CLE", "PIT"],
    "AFC South": ["HOU", "IND", "JAX", "TEN"], "AFC West": ["DEN", "KC", "LAC", "LV"],
    "NFC East": ["DAL", "NYG", "PHI", "WAS"], "NFC North": ["CHI", "DET", "GB", "MIN"],
    "NFC South": ["ATL", "CAR", "NO", "TB"], "NFC West": ["ARI", "LA", "SEA", "SF"],
}
TEAM_DIVISION = {team: division for division, teams in DIVISIONS.items() for team in teams}
# Codes other tables use for the same teams.
TEAM_ALIASES = {"LAR": "LA", "OAK": "LV", "SD": "LAC", "STL": "LA", "JAC": "JAX", "WSH": "WAS"}
# Too little to be worth a row.
MIN_OFFENSE_PLAYS, MIN_DEFENSE_SNAPS, MIN_KICKS = 15, 100, 5


# Wins from points scored and allowed: scored^x / (scored^x + allowed^x),
# the exponent long used for the NFL.
PYTHAGOREAN_EXPONENT = 2.37


def pythagorean(scored, allowed, exponent: float = PYTHAGOREAN_EXPONENT):
    """The share of games a team scoring and allowing these points wins."""
    return scored ** exponent / (scored ** exponent + allowed ** exponent)


def team_records(schedules: pd.DataFrame) -> pd.DataFrame:
    """Games, wins (a tie is half), points scored and allowed per team
    from the regular-season games played."""
    g = schedules[(schedules["game_type"] == "REG") & schedules["home_score"].notna()]
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        margin = g[f"{side}_score"] - g[f"{other}_score"]
        rows.append(pd.DataFrame({"team": g[f"{side}_team"].replace(TEAM_ALIASES), "scored": g[f"{side}_score"], "allowed": g[f"{other}_score"],
                                  "wins": (margin > 0) + 0.5 * (margin == 0)}))
    if not rows or not len(g):
        return pd.DataFrame(columns=["games", "wins", "scored", "allowed"])
    t = pd.concat(rows)
    return t.groupby("team").agg(games=("wins", "size"), wins=("wins", "sum"), scored=("scored", "sum"), allowed=("allowed", "sum"))


def team_table(records: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    """Each team's season as its rated players explain it, and what is
    left over. `points` has each team's players' points above replacement
    on offense (kickers with them) and defense. One row per team:

      wins, losses, scored, allowed   what happened
      offense, defense   its rated players' points against the league's
                         average team
      expected           the wins those points make: a league-average
                         team plus them, through the Pythagorean curve
      baseline           the wins of a REPLACEMENT TEAM: an average team
                         with the average team's points above replacement
                         taken away on both sides, through the same curve
      war                expected - baseline: wins above a replacement
                         team. baseline + war = expected.
      uncredited         the wins its real points make, less expected:
                         what the team did that no rated player's ledger
                         holds - the line, coverage, coaching
      close              wins - what its real points make: how its points
                         fell across games

    What the leftovers are was tested on 320 team-seasons
    (scripts/research_nfl_player_value.py): `close` does not carry to
    the next season (r 0.04) and does not predict it - luck. `uncredited`
    carries (0.31) and predicts next season's wins nearly as well per win
    as `expected` does - skill the play-by-play cannot hand to a player.
    """
    t = records.copy()
    games = t["games"]
    league = t["scored"].sum() / games.sum()
    p = points.reindex(t.index).fillna(0.0)
    per_game = {side: p[side].sum() / games.sum() for side in ("offense", "defense")}
    for side in ("offense", "defense"):
        t[side] = p[side] - per_game[side] * games
    t["expected"] = games * pythagorean(league * games + t["offense"], league * games - t["defense"])
    t["baseline"] = games * pythagorean(league - per_game["offense"], league + per_game["defense"])
    t["war"] = t["expected"] - t["baseline"]
    t["losses"] = games - t["wins"]
    actual = games * pythagorean(t["scored"], t["allowed"])
    t["uncredited"] = actual - t["expected"]
    t["close"] = t["wins"] - actual
    return t


def build(season: int, raw_dir: str | None = None) -> dict:
    """The table's data for one season: {season, through_week,
    points_per_win, replacement_win_rate (the share of games a
    replacement team wins), players: [...], teams: [...] (team_table's
    columns)}. Players: one entry per rated player -
    quarterbacks, backs, receivers and tight ends from the offense model,
    defenders from their playmaking, kickers from kicks over expected.

    Every entry: name, pos, side (offense / defense / special), team, div
    ("AFC East"; the conference is its first word), exp (seasons in the
    league before this one; 0 is a rookie), games, volume (plays, snaps or
    kicks), war, paa (points above the average at his position) and par
    (points above replacement). Offense adds passing / rushing / receiving
    (points above replacement in each) with their plays; defense its
    plays made and the points they took away; kickers field goals made
    and tried."""
    pbp = load_table("pbp", season, raw_dir)
    rosters = load_table("rosters_weekly", season, raw_dir)
    if pbp.empty or rosters.empty:
        return {"season": int(season), "through_week": 0, "points_per_win": POINTS_PER_WIN, "players": [], "teams": []}
    pbp = pbp[pbp["season_type"] == "REG"]
    roster = roster_facts(rosters)
    depth = rosters[rosters["gsis_id"].notna()].groupby("gsis_id")["depth_chart_position"].agg(
        lambda s: s.mode().iloc[0] if not s.mode().empty else None)

    def base(player_id, team, position, side):
        team = TEAM_ALIASES.get(team, team)
        division = TEAM_DIVISION.get(team, "")
        facts = roster.loc[player_id] if player_id in roster.index else None
        years = None if facts is None or pd.isna(facts["years_exp"]) else int(facts["years_exp"])
        return {"name": (facts["name"] if facts is not None else None) or player_id, "pos": position, "side": side,
                "team": team, "div": division, "exp": years}

    players = []
    plays = offense_plays(pbp)
    effects = fit_offense(plays, roster["position"])
    offense = offense_value(effects)
    # Each team's points above replacement: offense by the team a player was on for each play.
    points = pd.DataFrame({"offense": ((effects["effect"] - effects["replacement"]) * effects["plays"]).groupby(effects["team"].replace(TEAM_ALIASES)).sum()})
    points["defense"] = 0.0
    main_team = effects.groupby(["player_id", "team"])["plays"].sum().reset_index().sort_values("plays", ascending=False) \
        .drop_duplicates("player_id").set_index("player_id")["team"]
    games = pd.concat([plays[["game_id", c]].rename(columns={c: "player_id"}) for c in ("qb", "receiver", "rusher")]) \
        .dropna().drop_duplicates().groupby("player_id").size()
    for row in offense.itertuples():
        position = position_group(roster["position"].get(row.player_id))
        if position == "OTH" or row.plays < MIN_OFFENSE_PLAYS:
            continue
        entry = base(row.player_id, main_team[row.player_id], position, "offense")
        entry.update({"games": int(games.get(row.player_id, 0)), "volume": int(row.plays), "war": round(row.war, 2),
                      "paa": round(row.paa, 1), "par": round(row.par, 1)})
        for area in ("passing", "rushing", "receiving"):
            if getattr(row, f"{area}_plays"):
                entry[area] = round(getattr(row, f"{area}_par"), 1)
                entry[f"{area}_plays"] = int(getattr(row, f"{area}_plays"))
        players.append(entry)

    snaps = defense_snaps(load_table("snap_counts", season, raw_dir), roster)
    defense = defense_value(defense_table(defense_credits(pbp), snaps, roster, depth)) if not snaps.empty else pd.DataFrame()
    if len(defense):
        points["defense"] = points["defense"].add(defense.groupby(defense["team"].replace(TEAM_ALIASES))["par"].sum(), fill_value=0.0)
    for row in defense.itertuples():
        if row.snaps < MIN_DEFENSE_SNAPS:
            continue
        entry = base(row.player_id, row.team, row.position, "defense")
        entry.update({"games": int(row.games), "volume": int(row.snaps), "war": round(row.war, 2), "paa": round(row.paa, 1),
                      "par": round(row.par, 1), "points": round(row.points, 1), "sacks": round(row.sacks, 1),
                      "interceptions": int(row.interceptions), "passes_defended": int(row.passes_defended),
                      "tackles_for_loss": int(row.tackles_for_loss), "forced_fumbles": int(row.forced_fumbles), "tackles": int(row.tackles)})
        players.append(entry)

    kickers = kicker_value(pbp)
    if len(kickers):
        points["offense"] = points["offense"].add(kickers.groupby(kickers["team"].replace(TEAM_ALIASES))["par"].sum(), fill_value=0.0)
    for row in kickers.itertuples():
        if row.kicks < MIN_KICKS:
            continue
        entry = base(row.player_id, row.team, "K", "special")
        entry.update({"volume": int(row.kicks), "war": round(row.war, 2), "paa": round(row.paa, 1), "par": round(row.par, 1),
                      "fg_made": int(row.fg_made), "fg_tried": int(row.fg_tried)})
        players.append(entry)
    players.sort(key=lambda e: -e["war"])
    schedules = load_table("schedules", season, raw_dir)
    records = team_records(schedules) if not schedules.empty else pd.DataFrame()
    teams, rate = [], None
    if len(records):
        table = team_table(records, points.fillna(0.0))
        rate = round(float((table["baseline"] / table["games"]).mean()), 4)
        teams = [{"team": team, "div": TEAM_DIVISION.get(team, ""), "games": int(row.games), "wins": float(row.wins), "losses": float(row.losses),
                  "scored": int(row.scored), "allowed": int(row.allowed), "war": round(row.war, 1), "expected": round(row.expected, 1),
                  "offense": round(row.offense, 0), "defense": round(row.defense, 0), "uncredited": round(row.uncredited, 1),
                  "close": round(row.close, 1)} for team, row in table.sort_values("expected", ascending=False).iterrows()]
    return {"season": int(season), "through_week": int(pbp["week"].max()), "points_per_win": POINTS_PER_WIN,
            "replacement_win_rate": rate, "players": players, "teams": teams}

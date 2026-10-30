"""Player profiles for the MLB page's pitch-by-pitch game simulation
(docs/mlb_sim.js), from the stored Statcast pitches. Written daily to
docs/data/mlb_sim_profiles.json.

Every pitch has a result - a ball, called strike, swinging strike (foul tips
count: they can be strike three), foul, ball in play or hit batter - and
every ball in play a result: a groundout, flyout, lineout, single, double,
triple or home run. The league's rates for each pitch type in each count
are the base. A pitcher's own rates for each of his pitches, and a hitter's
against everything he's seen, are measured against what the league would
have done with the same pitches in the same counts, and pulled toward 1 with
pseudo-pitches - so a few hundred pitches don't swing them. The simulation
multiplies the three (and the league's platoon and times-through-the-order
effects) and renormalizes.

What happens to the runners comes straight from the league: for each kind
of play, base state and number of outs, how often it ended with each base
state, outs and runs (double plays, sacrifice flies, taking the extra base,
reaching on an error all fall out of it).

Where a player's sample is small, the pull isn't toward league average but
toward what his Stuff+ (each pitch) or Decision+ (each hitter) says he
should be (pitch_metrics.py): each grade moves each pitch result by a slope
fitted across players, and that start fades out as his own pitches pile up
(gone by 600 pitches of a pitch type, 1,500 for a hitter). Scored 2025 to
2026, that cuts the error in next season's rates 0.8% for pitch types
thrown 100-400 times and 2.0% for hitters with 400-1,000 pitches, and
leaves established players as they were.

Rosters are each player's latest team; the default lineup is a team's
regulars; starters and relievers come from who started.
"""

import json
import math

import numpy as np
import pandas as pd

PITCH_RESULTS = ["B", "C", "S", "F", "X", "H"]
CONTACT_RESULTS = ["GO", "FO", "LO", "1B", "2B", "3B", "HR"]
PLAY_KINDS = ["1B", "2B", "3B", "HR", "BB", "K", "GO", "FO", "LO"]
DESCRIPTION_RESULT = {
    "ball": "B", "blocked_ball": "B", "automatic_ball": "B", "pitchout": "B",
    "called_strike": "C", "automatic_strike": "C",
    "swinging_strike": "S", "swinging_strike_blocked": "S", "missed_bunt": "S", "swinging_pitchout": "S",
    "foul_tip": "S", "bunt_foul_tip": "S",
    "foul": "F", "foul_bunt": "F",
    "hit_into_play": "X",
    "hit_by_pitch": "H",
}
PITCH_TYPES = ["FF", "SI", "FC", "SL", "ST", "CU", "KC", "SV", "CH", "FS"]
PITCH_TYPE_ALIASES = {"FA": "FF", "FO": "FS", "CS": "CU", "SC": "CH", "EP": "CU", "KN": "CH"}
PITCH_GROUP = {"FF": "FB", "SI": "FB", "FC": "FB", "SL": "BR", "ST": "BR", "CU": "BR", "KC": "BR", "SV": "BR",
               "CH": "OS", "FS": "OS"}
HIT_EVENTS = {"single": "1B", "double": "2B", "triple": "3B", "home_run": "HR"}
WALK_EVENTS = {"walk", "hit_by_pitch", "intent_walk", "catcher_interf"}
STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}
BATTED_BALL_OUT = {"ground_ball": "GO", "line_drive": "LO", "fly_ball": "FO", "popup": "FO"}
COUNT_BUCKETS = ["first", "ahead", "even", "behind"]
STATCAST_COLUMNS = [
    "game_pk", "game_date", "game_type", "pitch_type", "release_speed", "batter", "pitcher", "player_name",
    "events", "description", "bb_type", "stand", "p_throws", "home_team", "away_team", "inning", "inning_topbot",
    "balls", "strikes", "outs_when_up", "on_1b", "on_2b", "on_3b", "plate_x", "plate_z", "sz_top", "sz_bot",
    "at_bat_number", "pitch_number", "bat_score", "fld_score", "post_bat_score", "n_thruorder_pitcher",
]

# Pseudo-pitches (or balls in play) of league average in each estimate.
SHRINK_PITCHER_PITCHES = 250
SHRINK_BATTER_PITCHES = 400
SHRINK_PITCHER_BIP = 200
SHRINK_BATTER_BIP = 150
SHRINK_MIX = 60
SHRINK_TYPE_CELL = 200
MIN_TYPE_SHARE = 0.03
LOCATION_SAMPLES = 60
TRANSITION_COVER = 0.995
RECENT_DAYS = 45
PRIOR_FADE_PITCHER = 600
PRIOR_FADE_BATTER = 1500
PRIOR_FIT_MIN_PITCHER = 300
PRIOR_FIT_MIN_BATTER = 800


def _bases(on1, on2, on3) -> np.ndarray:
    return (on1.notna().astype(int) + 2 * on2.notna().astype(int) + 4 * on3.notna().astype(int)).to_numpy()


def _bucket(balls, strikes) -> np.ndarray:
    return np.select([(balls == 0) & (strikes == 0), strikes > balls, balls == strikes], COUNT_BUCKETS[:3], "behind")


def prepare(pitches: pd.DataFrame) -> pd.DataFrame:
    """Regular-season pitches with a result, a pitch type in PITCH_TYPES,
    a count key and each side's team."""
    p = pitches[pitches["game_type"] == "R"].copy()
    p["result"] = p["description"].map(DESCRIPTION_RESULT)
    p["ptype"] = p["pitch_type"].replace(PITCH_TYPE_ALIASES)
    p = p[p["result"].notna() & p["ptype"].isin(PITCH_TYPES) & p["balls"].between(0, 3) & p["strikes"].between(0, 2)]
    p["count"] = p["balls"].astype(int).astype(str) + "-" + p["strikes"].astype(int).astype(str)
    p["group"] = p["ptype"].map(PITCH_GROUP)
    top = p["inning_topbot"] == "Top"
    p["bat_team"] = np.where(top, p["away_team"], p["home_team"])
    p["field_team"] = np.where(top, p["home_team"], p["away_team"])
    p["same_hand"] = p["stand"] == p["p_throws"]
    return p


def _rates(df: pd.DataFrame, by: list[str], col: str, labels: list[str]) -> pd.DataFrame:
    counts = df.groupby(by + [col]).size().unstack(col, fill_value=0).reindex(columns=labels, fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0), counts.sum(axis=1)


def league_pitch_table(p: pd.DataFrame) -> dict:
    """{pitch type: {count: [P(result) for PITCH_RESULTS]}}: each type's own
    rates in each count, pulled toward its group's with SHRINK_TYPE_CELL
    pitches where a cell is thin."""
    by_type, n_type = _rates(p, ["ptype", "count"], "result", PITCH_RESULTS)
    by_group, _ = _rates(p, ["group", "count"], "result", PITCH_RESULTS)
    out = {}
    for (ptype, count), row in by_type.iterrows():
        n = n_type[(ptype, count)]
        g = by_group.loc[(PITCH_GROUP[ptype], count)]
        mixed = (row * n + g * SHRINK_TYPE_CELL) / (n + SHRINK_TYPE_CELL)
        out.setdefault(ptype, {})[count] = [round(float(v), 4) for v in mixed]
    return out


def _expected(p: pd.DataFrame, table: dict) -> np.ndarray:
    """Each pitch's league probabilities for its type and count."""
    keys = list(zip(p["ptype"], p["count"]))
    return np.array([table[t][c] for t, c in keys])


def _ratio(observed: np.ndarray, expected: np.ndarray, k: float, prior=None) -> np.ndarray:
    """Observed over expected for each outcome, pulled toward `prior` (1,
    league average, if not given) with k pseudo-observations at the
    expected rate."""
    n = max(1.0, expected.sum())
    rate = expected / n
    prior = np.ones_like(rate) if prior is None else np.asarray(prior, dtype=float)
    return _divide(observed + k * rate * prior, expected + k * rate)


def _fade(n: float, full: float) -> float:
    return max(0.0, 1.0 - n / full)


def fit_prior_slopes(groups: dict, grades: dict, min_n: int) -> np.ndarray | None:
    """The slope of each pitch result's ratio on a grade's z (its distance
    from 100 in tens): ratio - 1 = slope * z, fitted through the origin over
    `groups` ({key: (observed, expected, n)}) with min_n pitches, weighted
    by pitches."""
    rows = [(o / np.maximum(e, 1e-9) - 1, n, grades[k]) for k, (o, e, n) in groups.items() if k in grades and n >= min_n]
    if len(rows) < 10:
        return None
    r = np.array([x[0] for x in rows])
    w = np.array([x[1] for x in rows], dtype=float)
    z = np.array([x[2] for x in rows])
    return (w[:, None] * z[:, None] * r).sum(0) / max(1e-9, (w * z * z).sum())


def _divide(a, b) -> np.ndarray:
    """a / b, with 1 (no change from the league) where b is 0."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.divide(a, b, out=np.ones_like(a), where=b > 0)


def contact_result(events, bb_type) -> np.ndarray:
    hits = events.map(HIT_EVENTS)
    outs = bb_type.map(BATTED_BALL_OUT)
    return hits.fillna(outs).to_numpy()


def platoon_and_order(p: pd.DataFrame, expected: np.ndarray) -> dict:
    """The league's same-hand / opposite-hand ratios for each pitch result,
    and its 1st / 2nd / 3rd+-time-through-the-order ratios (starters'
    batters), each against the pitch-by-pitch expectation."""
    obs = pd.get_dummies(p["result"]).reindex(columns=PITCH_RESULTS, fill_value=0).to_numpy(dtype=float)
    out = {"platoon": {}, "order": {}}
    for key, mask in (("same", p["same_hand"].to_numpy()), ("opposite", ~p["same_hand"].to_numpy())):
        out["platoon"][key] = [round(float(v), 4) for v in _divide(obs[mask].sum(0), expected[mask].sum(0))]
    tto = p["n_thruorder_pitcher"].fillna(1).clip(upper=3).astype(int).to_numpy()
    for t in (1, 2, 3):
        mask = tto == t
        out["order"][str(t)] = [round(float(v), 4) for v in _divide(obs[mask].sum(0), expected[mask].sum(0))]
    return out


def transitions(p: pd.DataFrame) -> dict:
    """{"kind|bases|outs": [[share, bases after, outs added, runs], ...]}
    from every plate appearance's last pitch to the next one's first (or
    the end of the half-inning), the commonest covering TRANSITION_COVER."""
    last = p[p["events"].notna()].sort_values(["game_pk", "at_bat_number", "pitch_number"])
    last = last.drop_duplicates(["game_pk", "at_bat_number"], keep="last").copy()
    kind = pd.Series(contact_result(last["events"], last["bb_type"]), index=last.index)
    kind[last["events"].isin(WALK_EVENTS)] = "BB"
    kind[last["events"].isin(STRIKEOUT_EVENTS)] = "K"
    last["kind"] = kind
    last = last[last["kind"].isin(PLAY_KINDS)]
    last["bases"] = _bases(last["on_1b"], last["on_2b"], last["on_3b"])
    firsts = p.sort_values(["game_pk", "at_bat_number", "pitch_number"]).drop_duplicates(["game_pk", "at_bat_number"])
    firsts = firsts.assign(bases_next=_bases(firsts["on_1b"], firsts["on_2b"], firsts["on_3b"]))
    nxt = firsts[["game_pk", "at_bat_number", "inning", "inning_topbot", "outs_when_up", "bases_next"]].rename(
        columns={"at_bat_number": "next_ab", "inning": "next_inning", "inning_topbot": "next_half", "outs_when_up": "next_outs"})
    last = last.sort_values(["game_pk", "at_bat_number"])
    nxt = nxt.sort_values(["game_pk", "next_ab"])
    merged = pd.merge_asof(last.assign(probe=last["at_bat_number"] + 1).sort_values("probe"),
                           nxt.sort_values("next_ab"), left_on="probe", right_on="next_ab", by="game_pk", direction="forward")
    same_half = (merged["next_inning"] == merged["inning"]) & (merged["next_half"] == merged["inning_topbot"])
    outs_before = merged["outs_when_up"].astype(int)
    outs_after = np.where(same_half, merged["next_outs"].fillna(3), 3).astype(int)
    merged["outs_added"] = np.clip(outs_after - outs_before, 0, 3)
    merged["bases_after"] = np.where(same_half & (outs_after < 3), merged["bases_next"].fillna(0), 0).astype(int)
    merged["runs"] = (merged["post_bat_score"] - merged["bat_score"]).clip(lower=0, upper=4).fillna(0).astype(int)
    # A game that ends on the play (a walk-off) isn't a real 3-out ending.
    walk_off = (~same_half) & (merged["inning_topbot"] == "Bot") & (merged["post_bat_score"] > merged["fld_score"])
    merged = merged[~walk_off]
    out = {}
    for (kind, bases, outs), g in merged.groupby(["kind", "bases", "outs_when_up"]):
        counts = g.groupby(["bases_after", "outs_added", "runs"]).size().sort_values(ascending=False)
        share = counts / counts.sum()
        keep = share[share.cumsum().shift(fill_value=0) < TRANSITION_COVER]
        keep = keep / keep.sum()
        out[f"{kind}|{int(bases)}|{int(outs)}"] = [[round(float(v), 4), int(b), int(o), int(r)] for (b, o, r), v in keep.items()]
    return out


def locations(p: pd.DataFrame, rng: np.random.Generator) -> dict:
    """{result: {group: [[x, z], ...]}}: where real pitches with that result
    crossed the plate - x in feet from the middle, positive toward the
    batter, z from 0 at the bottom of his zone to 1 at the top."""
    q = p[p["plate_x"].notna() & p["plate_z"].notna() & p["sz_top"].notna() & (p["sz_top"] > p["sz_bot"])]
    x = np.where(q["stand"] == "R", -q["plate_x"], q["plate_x"])
    z = (q["plate_z"] - q["sz_bot"]) / (q["sz_top"] - q["sz_bot"])
    q = q.assign(xb=x, zn=z)
    out = {}
    for (result, group), g in q.groupby(["result", "group"]):
        take = g.sample(min(LOCATION_SAMPLES, len(g)), random_state=int(rng.integers(1 << 31)))
        out.setdefault(result, {})[group] = [[round(float(a), 2), round(float(b), 2)] for a, b in zip(take["xb"], take["zn"])]
    return out


def _mix(df: pd.DataFrame, types: list[str], prior: pd.Series, k: float) -> list:
    counts = df["ptype"].value_counts().reindex(types, fill_value=0)
    mixed = (counts + k * prior) / (counts.sum() + k)
    return [round(float(v), 3) for v in mixed]


def _sums(p: pd.DataFrame, expected: np.ndarray, keys) -> dict:
    """{key: (observed result counts, expected, pitches)}."""
    obs = pd.get_dummies(p["result"]).reindex(columns=PITCH_RESULTS, fill_value=0).to_numpy(dtype=float)
    q = p.assign(_row=np.arange(len(p)))
    out = {}
    for k, g in q.groupby(keys):
        rows = g["_row"].to_numpy()
        out[k] = (obs[rows].sum(0), expected[rows].sum(0), len(rows))
    return out


def pitcher_priors(p: pd.DataFrame, expected: np.ndarray, stuff: dict | None) -> tuple[dict, dict]:
    """({(pitcher, type): prior ratios}, {group: slopes}) from Stuff+
    ({(pitcher, type): z})."""
    if not stuff:
        return {}, {}
    sums = _sums(p, expected, ["pitcher", "ptype"])
    slopes = {}
    for g in ("FB", "BR", "OS"):
        mine = {k: v for k, v in sums.items() if PITCH_GROUP.get(k[1]) == g}
        slope = fit_prior_slopes(mine, stuff, PRIOR_FIT_MIN_PITCHER)
        if slope is not None:
            slopes[g] = slope
    priors = {}
    for k, (_, _, n) in sums.items():
        g = PITCH_GROUP.get(k[1])
        if k in stuff and g in slopes:
            priors[k] = 1 + slopes[g] * stuff[k] * _fade(n, PRIOR_FADE_PITCHER)
    return priors, slopes


def batter_priors(p: pd.DataFrame, expected: np.ndarray, decision: dict | None) -> tuple[dict, np.ndarray | None]:
    """({batter: prior ratios}, slopes) from Decision+ ({batter: z})."""
    if not decision:
        return {}, None
    sums = _sums(p, expected, "batter")
    slope = fit_prior_slopes(sums, decision, PRIOR_FIT_MIN_BATTER)
    if slope is None:
        return {}, None
    return {k: 1 + slope * decision[k] * _fade(n, PRIOR_FADE_BATTER) for k, (_, _, n) in sums.items() if k in decision}, slope


def pitcher_profiles(p: pd.DataFrame, expected: np.ndarray, contact_league: pd.Series, priors: dict | None = None) -> dict:
    """{pitcher id: {name, throws, types, velo [[mean, sd]], ratios
    {type: [PITCH_RESULTS]}, mix {stand: {bucket: [types]}}, contact
    [CONTACT_RESULTS]}}."""
    obs = pd.get_dummies(p["result"]).reindex(columns=PITCH_RESULTS, fill_value=0).to_numpy(dtype=float)
    out = {}
    p = p.assign(_row=np.arange(len(p)), bucket=_bucket(p["balls"], p["strikes"]))
    for pid, g in p.groupby("pitcher"):
        share = g["ptype"].value_counts(normalize=True)
        types = [t for t in PITCH_TYPES if share.get(t, 0) >= MIN_TYPE_SHARE] or [share.index[0]]
        g = g[g["ptype"].isin(types)]
        rows = g["_row"].to_numpy()
        ratios, velo = {}, []
        for t in types:
            mine = (g["ptype"] == t).to_numpy()
            r = _ratio(obs[rows[mine]].sum(0), expected[rows[mine]].sum(0), SHRINK_PITCHER_PITCHES, (priors or {}).get((pid, t)))
            ratios[t] = [round(float(v), 3) for v in r]
            speed = g.loc[mine, "release_speed"].dropna()
            velo.append([round(float(speed.mean()), 1) if len(speed) else None, round(float(speed.std()), 1) if len(speed) > 2 else 1.0])
        overall = g["ptype"].value_counts(normalize=True).reindex(types, fill_value=0)
        mix = {}
        for stand in ("L", "R"):
            side = g[g["stand"] == stand]
            side_prior = (side["ptype"].value_counts().reindex(types, fill_value=0) + SHRINK_MIX * overall) / (len(side) + SHRINK_MIX)
            mix[stand] = {b: _mix(side[side["bucket"] == b], types, side_prior, SHRINK_MIX) for b in COUNT_BUCKETS}
        bip = g[g["result"] == "X"]
        contact = contact_result(bip["events"], bip["bb_type"])
        seen = pd.Series(contact).value_counts().reindex(CONTACT_RESULTS, fill_value=0).to_numpy(dtype=float)
        exp = contact_league.reindex(CONTACT_RESULTS).to_numpy() * len(bip)
        out[str(int(pid))] = {
            "name": _flip_name(g["player_name"].iloc[-1]),
            "throws": g["p_throws"].iloc[-1],
            "types": types, "velo": velo, "ratios": ratios, "mix": mix,
            "contact": [round(float(v), 3) for v in _ratio(seen, exp, SHRINK_PITCHER_BIP)],
        }
    return out


def batter_profiles(p: pd.DataFrame, expected: np.ndarray, contact_league: pd.Series, priors: dict | None = None) -> dict:
    """{batter id: {bats {L: stand vs lefties, R: vs righties}, ratios
    [PITCH_RESULTS], contact [CONTACT_RESULTS]}}."""
    obs = pd.get_dummies(p["result"]).reindex(columns=PITCH_RESULTS, fill_value=0).to_numpy(dtype=float)
    p = p.assign(_row=np.arange(len(p)))
    out = {}
    for bid, g in p.groupby("batter"):
        rows = g["_row"].to_numpy()
        bip = g[g["result"] == "X"]
        seen = pd.Series(contact_result(bip["events"], bip["bb_type"])).value_counts().reindex(CONTACT_RESULTS, fill_value=0).to_numpy(dtype=float)
        exp = contact_league.reindex(CONTACT_RESULTS).to_numpy() * len(bip)
        bats = {}
        for hand in ("L", "R"):
            vs = g.loc[g["p_throws"] == hand, "stand"]
            bats[hand] = vs.mode().iloc[0] if len(vs) else g["stand"].mode().iloc[0]
        out[str(int(bid))] = {
            "bats": bats,
            "ratios": [round(float(v), 3) for v in _ratio(obs[rows].sum(0), expected[rows].sum(0), SHRINK_BATTER_PITCHES,
                                                          (priors or {}).get(bid))],
            "contact": [round(float(v), 3) for v in _ratio(seen, exp, SHRINK_BATTER_BIP)],
        }
    return out


def _flip_name(name) -> str:
    """"Skenes, Paul" -> "Paul Skenes"."""
    parts = str(name).split(", ", 1)
    return f"{parts[1]} {parts[0]}" if len(parts) == 2 else str(name)


def _regular_lineup(games: pd.DataFrame, bat_team: pd.Series, team: str) -> list:
    """The team's regulars: the nine still on it with the most plate
    appearances lately, in their usual batting order (the average spot they
    batted in). The latest game alone is a poor default - at a season's end
    it's often the regulars resting."""
    pa = games.drop_duplicates(["game_pk", "at_bat_number"])
    firsts = pa.sort_values(["game_pk", "at_bat_number"]).drop_duplicates(["game_pk", "batter"])
    firsts = firsts.assign(slot=firsts.groupby("game_pk").cumcount() + 1)
    starters = firsts[firsts["slot"] <= 9]
    usage = starters.groupby("batter").agg(games=("slot", "size"), slot=("slot", "mean"))
    usage = usage[[bat_team.get(b) == team for b in usage.index]]
    top = usage.sort_values("games", ascending=False).head(9)
    return top.sort_values("slot").index.tolist()


def team_rosters(p: pd.DataFrame, season: int) -> dict:
    """{team: {lineup [batter ids, latest batting order], hitters [ids],
    starters [{id, last}], bullpen [{id, role score}]}} from each player's
    latest team this season."""
    cur = p[p["game_date"].dt.year == season]
    if cur.empty:
        return {}
    end = cur["game_date"].max()
    recent = cur[cur["game_date"] >= end - pd.Timedelta(days=RECENT_DAYS)]
    bat_team = cur.sort_values("game_date").groupby("batter")["bat_team"].last()
    pit_team = cur.sort_values("game_date").groupby("pitcher")["field_team"].last()
    first = cur.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    # The starter: each team's first pitcher of each game.
    starts = first.drop_duplicates(["game_pk", "field_team"])[["game_pk", "game_date", "field_team", "pitcher"]]
    appearances = first.drop_duplicates(["game_pk", "pitcher"])[["game_pk", "game_date", "field_team", "pitcher", "inning", "bat_score"]]
    appearances = appearances.merge(starts.assign(started=True)[["game_pk", "pitcher", "started"]], on=["game_pk", "pitcher"], how="left")
    relief = appearances[appearances["started"].isna()].copy()
    out = {}
    for team in sorted(set(bat_team.unique()) | set(pit_team.unique())):
        games = cur[cur["bat_team"] == team]
        if games.empty:
            continue
        order = _regular_lineup(games[games["game_date"] >= end - pd.Timedelta(days=RECENT_DAYS)], bat_team, team)
        hitters = [int(b) for b, t in bat_team.items() if t == team and b in set(recent["batter"])]
        team_starts = starts[(starts["field_team"] == team)]
        last_start = team_starts.groupby("pitcher")["game_date"].max()
        n_starts = team_starts.groupby("pitcher").size()
        # Real starters (three starts or more) first, by who started last;
        # openers after.
        starters = [{"id": int(pid), "last": d.strftime("%Y-%m-%d"), "starts": int(n_starts[pid])}
                    for pid, d in last_start.sort_values(ascending=False).items() if pit_team.get(pid) == team]
        starters = sorted(starters, key=lambda s: s["starts"] < 3)[:8]
        mine = relief[(relief["field_team"] == team) & (relief["game_date"] >= end - pd.Timedelta(days=RECENT_DAYS))]
        pen = []
        for pid, g in mine.groupby("pitcher"):
            if pit_team.get(pid) != team or len(g) < 3:
                continue
            late = float((g["inning"] >= 8).mean())
            pen.append({"id": int(pid), "apps": int(len(g)), "late": round(late, 3)})
        pen.sort(key=lambda r: (-r["late"], -r["apps"]))
        out[team] = {"lineup": [int(b) for b in order], "hitters": hitters, "starters": starters, "bullpen": pen}
    return out


def starter_stamina(p: pd.DataFrame) -> dict:
    """{pitcher id: [mean pitches a start, sd]} over his last 10 starts."""
    first = p.sort_values(["game_pk", "at_bat_number", "pitch_number"])
    starters = first.drop_duplicates(["game_pk", "field_team"])[["game_pk", "pitcher"]]
    counts = p.groupby(["game_pk", "pitcher"]).size().rename("pitches").reset_index()
    s = starters.merge(counts, on=["game_pk", "pitcher"]).merge(p.drop_duplicates("game_pk")[["game_pk", "game_date"]], on="game_pk")
    out = {}
    for pid, g in s.sort_values("game_date").groupby("pitcher"):
        last = g["pitches"].tail(10)
        out[str(int(pid))] = [round(float(last.mean()), 1), round(float(last.std()) if len(last) > 2 else 15.0, 1)]
    return out


def calibration(p: pd.DataFrame, expected: np.ndarray, effects: dict, pitchers: dict, batters: dict,
                contact_league: pd.Series) -> tuple[list, list]:
    """Multipliers for each pitch result and each contact result that make
    the model - the league rate times the pitcher's, hitter's, platoon and
    order ratios, renormalized - add up to what really happened over every
    real pitch between profiled players. (The ratios are each measured
    against the league; multiplied together they run a little hot.)"""
    pr_rows, br_rows, keep = [], [], []
    for i, (pid, bid, t) in enumerate(zip(p["pitcher"], p["batter"], p["ptype"])):
        pp, bb = pitchers.get(str(int(pid))), batters.get(str(int(bid)))
        if pp is None or bb is None or t not in pp["ratios"]:
            continue
        keep.append(i)
        pr_rows.append(pp["ratios"][t])
        br_rows.append(bb["ratios"])
    if not keep:
        return [1.0] * len(PITCH_RESULTS), [1.0] * len(CONTACT_RESULTS)
    keep = np.array(keep)
    q = p.iloc[keep]
    plat = np.where(q["same_hand"].to_numpy()[:, None], effects["platoon"]["same"], effects["platoon"]["opposite"])
    tto = q["n_thruorder_pitcher"].fillna(1).clip(1, 3).astype(int).astype(str).to_numpy()
    order = np.array([effects["order"][t] for t in tto])
    pred = expected[keep] * np.array(pr_rows) * np.array(br_rows) * plat * order
    pred = pred / pred.sum(axis=1, keepdims=True)
    obs = pd.get_dummies(q["result"]).reindex(columns=PITCH_RESULTS, fill_value=0).to_numpy(dtype=float)
    pitch_cal = _divide(obs.sum(0), pred.sum(0))
    bip = q[q["result"] == "X"]
    rows = [(pitchers[str(int(a))]["contact"], batters[str(int(b))]["contact"]) for a, b in zip(bip["pitcher"], bip["batter"])]
    cpred = contact_league.reindex(CONTACT_RESULTS).to_numpy() * np.array([a for a, _ in rows]) * np.array([b for _, b in rows])
    cpred = cpred / cpred.sum(axis=1, keepdims=True)
    seen = pd.Series(contact_result(bip["events"], bip["bb_type"])).value_counts().reindex(CONTACT_RESULTS, fill_value=0).to_numpy(dtype=float)
    return [round(float(v), 4) for v in pitch_cal], [round(float(v), 4) for v in _divide(seen, cpred.sum(0))]


def build_profiles(pitches: pd.DataFrame, season: int, names: pd.DataFrame | None = None, seed: int = 7,
                   stuff: pd.DataFrame | None = None, decision: pd.DataFrame | None = None) -> dict:
    """stuff (pitcher, pitch_type, stuff_plus) and decision (key_mlbam,
    decision_plus), pooled over the seasons used, set the priors for small
    samples (see the module docstring)."""
    p = prepare(pitches)
    table = league_pitch_table(p)
    expected = _expected(p, table)
    bip = p[p["result"] == "X"]
    contact = pd.Series(contact_result(bip["events"], bip["bb_type"])).value_counts(normalize=True).reindex(CONTACT_RESULTS, fill_value=0)
    effects = platoon_and_order(p, expected)
    rosters = team_rosters(p, season)
    on_roster = {pid for team in rosters.values() for pid in team["hitters"] + team["lineup"]}
    arms = {s["id"] for team in rosters.values() for s in team["starters"]} | {r["id"] for team in rosters.values() for r in team["bullpen"]}
    stuff_z = None if stuff is None or stuff.empty else {
        (int(r.pitcher), r.pitch_type): (r.stuff_plus - 100) / 10 for r in stuff.itertuples() if pd.notna(r.stuff_plus)}
    decision_z = None if decision is None or decision.empty else {
        int(r.key_mlbam): (r.decision_plus - 100) / 10 for r in decision.itertuples() if pd.notna(r.decision_plus)}
    p_priors, p_slopes = pitcher_priors(p, expected, stuff_z)
    b_priors, b_slope = batter_priors(p, expected, decision_z)
    pitchers = pitcher_profiles(p[p["pitcher"].isin(arms)], expected[p["pitcher"].isin(arms).to_numpy()], contact, p_priors)
    batters = batter_profiles(p[p["batter"].isin(on_roster)], expected[p["batter"].isin(on_roster).to_numpy()], contact, b_priors)
    stamina = starter_stamina(p)
    for pid, prof in pitchers.items():
        if pid in stamina:
            prof["stamina"] = stamina[pid]
    if names is not None and len(names):
        lookup = {int(r.key_mlbam): f"{r.name_first} {r.name_last}" for r in names.itertuples() if pd.notna(r.key_mlbam)}
        for bid, prof in batters.items():
            prof["name"] = lookup.get(int(bid), bid)
    pitch_cal, contact_cal = calibration(p, expected, effects, pitchers, batters, contact)
    rng = np.random.default_rng(seed)
    pa_ends = p[p["events"].notna()]
    league = {
        "results": PITCH_RESULTS, "contact_results": CONTACT_RESULTS, "pitch_groups": PITCH_GROUP,
        "pitch_table": table,
        "contact": [round(float(v), 4) for v in contact],
        "platoon": effects["platoon"], "order": effects["order"],
        "calibration": pitch_cal, "contact_calibration": contact_cal,
        "prior_slopes": {"stuff": {g: [round(float(v), 4) for v in s] for g, s in p_slopes.items()},
                         "decision": [round(float(v), 4) for v in b_slope] if b_slope is not None else None},
        "transitions": transitions(p),
        "locations": locations(p, rng),
        "stamina": [round(float(np.mean([v[0] for v in stamina.values()])), 1), 15.0] if stamina else [90.0, 15.0],
        "runs_per_team_game": round(float(pa_ends.groupby(["game_pk", "bat_team"])["post_bat_score"].max().mean()), 2),
    }
    return {"season": int(season), "through": str(p["game_date"].max().date()), "league": league,
            "teams": rosters, "pitchers": pitchers, "batters": batters}


def write_profiles(profiles: dict, path: str) -> None:
    def clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v

    with open(path, "w") as f:
        json.dump(json.loads(json.dumps(profiles, default=clean), parse_constant=lambda c: None), f, separators=(",", ":"))

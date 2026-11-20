"""What each offense calls given what just happened, and what the defense
across from it gives up: the Play-calling map on the NFL Picks tab.

Why (2026-11-20, user request: "map play call sequencing tendencies per
team with a way of showing how the opposing defense does in those
situations"): the site rated what teams DO (EPA, units) but not what they
CHOOSE. A play call depends on the down and distance, and on what came
before it - a team that threw incomplete on first down faces the same 2nd
and 10 as one that was stuffed on a run, and calls it differently.

Every scrimmage play gets one SITUATION: its down and distance, split by
what the play before it was where that matters (STATES), and may also
carry HABITS that cut across them (after a gain of 15+, after two runs in
a row, inside the 20...). For each, an offense's pass rate is compared
with what a typical team calls in exactly those plays: nflfastR's
expected pass rate for each play (`xpass`: down, distance, field
position, score, clock) plus the league's own lean in that situation
(xpass does not know the previous play).

What holds up and what does not (scripts/research_nfl_play_sequencing.py,
2021-2026): an offense's lean in a situation IS a habit - between the odd
and even weeks of a season it correlates 0.25 to 0.57 situation by
situation on 15 to 80 plays a half - so it is shown per situation, pulled
toward the league with LEAN_PRIOR plays. A defense's success allowed in
one situation is NOT: 0.0 to 0.15. Even its whole-season rate against the
pass (0.26) or the run (0.15) is mostly noise in half a season. So the
defense's number in each situation is the league's for that situation
moved by how the defense has done against that kind of play overall,
pulled hard toward the league (DEFENSE_PRIOR) - and its raw record in the
situation is given beside it, as a record and not a prediction.
"""

import numpy as np
import pandas as pd

from mlb_metrics import config

PLAY_COLUMNS = [
    "game_id", "play_id", "season", "week", "season_type", "posteam", "defteam", "play_type", "down", "ydstogo",
    "yardline_100", "xpass", "success", "yards_gained", "sack", "complete_pass", "first_down", "fixed_drive",
    "qb_kneel", "qb_spike",
]
# (key, label, group) in the order shown: the down-and-distance tree.
STATES = [
    ("open", "To open a drive", "1st down"),
    ("first_after_run", "After a run moved the chains", "1st down"),
    ("first_after_pass", "After a pass moved the chains", "1st down"),
    ("second_long_after_pass", "2nd & long, after a pass that failed", "2nd down"),
    ("second_long_after_run", "2nd & long, after a run that failed", "2nd down"),
    ("second_medium_after_run", "2nd & medium, after a run", "2nd down"),
    ("second_medium_after_pass", "2nd & medium, after a pass", "2nd down"),
    ("second_short", "2nd & short", "2nd down"),
    ("third_short", "3rd & short", "3rd down"),
    ("third_medium", "3rd & medium", "3rd down"),
    ("third_long", "3rd & long", "3rd down"),
]
HABITS = [
    ("after_big_gain", "Right after a gain of 15+"),
    ("after_loss", "Right after a sack or a loss"),
    ("after_incompletion", "Right after an incompletion"),
    ("after_stuffed_run", "Right after a run held to 2 or less"),
    ("after_two_runs", "After two runs in a row"),
    ("after_two_passes", "After two passes in a row"),
    ("red_zone", "Inside the 20"),
]
# Plays of prior toward the league. From the half-season correlations (r
# at n plays a half gives a prior of n (1 - r) / r): an offense's lean in
# one situation about 0.45 at 45; an offense's success passing 0.60 at 278
# and running 0.44 at 205; a defense's success allowed against the pass
# 0.26 at 278 and the run 0.15 at 205.
LEAN_PRIOR = 55.0
OFFENSE_PRIOR = {"pass": 185.0, "run": 260.0}
DEFENSE_PRIOR = {"pass": 790.0, "run": 1160.0}
# Long, medium and short to go.
LONG, SHORT = 8, 3


def scrimmage(pbp: pd.DataFrame) -> pd.DataFrame:
    """Regular-season runs and passes in the order played, each with what
    the play before it in the drive was: is_pass, prev_pass, prev_success,
    prev_yards, prev_first_down, prev_incomplete, prev2_pass (NaN at the
    start of a drive)."""
    p = pbp[(pbp["season_type"] == "REG") & pbp["play_type"].isin(["run", "pass"]) & pbp["posteam"].notna()
            & pbp["down"].notna() & pbp["xpass"].notna() & (pbp["qb_kneel"] != 1) & (pbp["qb_spike"] != 1)].copy()
    p = p.sort_values(["game_id", "play_id"]).reset_index(drop=True)
    p["is_pass"] = (p["play_type"] == "pass").astype(int)
    p["incomplete"] = ((p["is_pass"] == 1) & (p["complete_pass"] != 1) & (p["sack"] != 1)).astype(int)
    drive = p.groupby(["game_id", "posteam", "fixed_drive"], sort=False)
    p["prev_pass"] = drive["is_pass"].shift(1)
    p["prev_success"] = drive["success"].shift(1)
    p["prev_yards"] = drive["yards_gained"].shift(1)
    p["prev_first_down"] = drive["first_down"].shift(1)
    p["prev_incomplete"] = drive["incomplete"].shift(1)
    p["prev2_pass"] = drive["is_pass"].shift(2)
    return p


def situations(p: pd.DataFrame) -> pd.Series:
    """Each play's situation key (one of STATES, or "" for the few that
    fit none: 4th downs, first downs reached by penalty)."""
    down, togo, prev_pass = p["down"], p["ydstogo"], p["prev_pass"]
    first = prev_pass.isna()
    s = pd.Series("", index=p.index, dtype=object)
    s[(down == 1) & first] = "open"
    s[(down == 1) & ~first & (p["prev_first_down"] == 1) & (prev_pass == 0)] = "first_after_run"
    s[(down == 1) & ~first & (p["prev_first_down"] == 1) & (prev_pass == 1)] = "first_after_pass"
    s[(down == 2) & (togo >= LONG) & (prev_pass == 1)] = "second_long_after_pass"
    s[(down == 2) & (togo >= LONG) & (prev_pass == 0)] = "second_long_after_run"
    s[(down == 2) & (togo > SHORT) & (togo < LONG) & (prev_pass == 0)] = "second_medium_after_run"
    s[(down == 2) & (togo > SHORT) & (togo < LONG) & (prev_pass == 1)] = "second_medium_after_pass"
    s[(down == 2) & (togo <= SHORT)] = "second_short"
    s[(down == 3) & (togo <= 2)] = "third_short"
    s[(down == 3) & (togo >= 3) & (togo <= 6)] = "third_medium"
    s[(down == 3) & (togo >= 7)] = "third_long"
    return s


def habit_flags(p: pd.DataFrame) -> pd.DataFrame:
    """One True/False column per habit in HABITS (a play can carry several)."""
    return pd.DataFrame({
        "after_big_gain": p["prev_yards"] >= 15,
        "after_loss": p["prev_yards"] < 0,
        "after_incompletion": p["prev_incomplete"] == 1,
        "after_stuffed_run": (p["prev_pass"] == 0) & (p["prev_yards"] <= 2),
        "after_two_runs": (p["prev_pass"] == 0) & (p["prev2_pass"] == 0),
        "after_two_passes": (p["prev_pass"] == 1) & (p["prev2_pass"] == 1),
        "red_zone": p["yardline_100"] <= 20,
    }, index=p.index).fillna(False)


def _cell(rows: pd.DataFrame, league_lean: float, prior: float = LEAN_PRIOR) -> dict:
    """An offense's calls in one situation: {plays, passes, typical (what
    a typical team throws in exactly these plays), rate (its own rate,
    pulled toward typical), lean (rate minus typical)}."""
    n = len(rows)
    passes = int(rows["is_pass"].sum())
    typical = float(np.clip(rows["xpass"].mean() + league_lean, 0.02, 0.98)) if n else float("nan")
    rate = (passes + prior * typical) / (n + prior) if n else float("nan")
    return {"plays": n, "passes": passes, "typical": round(typical, 4) if n else None,
            "rate": round(rate, 4) if n else None, "lean": round(rate - typical, 4) if n else None}


def _effect(successes: float, n: float, league_rate: float, prior: float) -> float:
    """A team's success rate minus the league's, pulled toward zero."""
    return float((successes - n * league_rate) / (n + prior)) if n + prior > 0 else 0.0


def build(pbp: pd.DataFrame, season: int) -> dict:
    """The map's data for one season:
    {season, through_week, states, habits, league: {key: {...}},
     teams: {code: {offense: {key: cell}, habits: {key: cell},
                    offense_effect, defense_effect, defense: {key: record},
                    readable: {...}}}}."""
    p = scrimmage(pbp[pbp["season"] == season])
    if p.empty:
        return {"season": int(season), "through_week": 0, "teams": {}}
    p["situation"] = situations(p)
    flags = habit_flags(p)
    p["lean"] = p["is_pass"] - p["xpass"]
    kind = np.where(p["is_pass"] == 1, "pass", "run")
    league = {}
    groups = [(key, p[p["situation"] == key]) for key, _, _ in STATES] + [(key, p[flags[key]]) for key, _ in HABITS]
    for key, rows in groups:
        is_pass = rows["is_pass"] == 1
        league[key] = {
            "plays": int(len(rows)), "pass_rate": round(float(rows["is_pass"].mean()), 4), "lean": float(rows["lean"].mean()),
            "pass_success": round(float(rows.loc[is_pass, "success"].mean()), 4),
            "run_success": round(float(rows.loc[~is_pass, "success"].mean()), 4),
        }
    overall = {k: float(p.loc[kind == k, "success"].mean()) for k in ("pass", "run")}
    # What a typical team does on each play, knowing its situation.
    p["typical"] = np.clip(p["xpass"] + p["situation"].map(lambda key: league[key]["lean"] if key else 0.0), 0.02, 0.98)

    teams = {}
    for team in sorted(set(p["posteam"]) | set(p["defteam"])):
        mine, against = p[p["posteam"] == team], p[p["defteam"] == team]
        offense = {key: _cell(mine[mine["situation"] == key], league[key]["lean"]) for key, _, _ in STATES}
        habits = {key: _cell(mine[flags.loc[mine.index, key]], league[key]["lean"]) for key, _ in HABITS}
        defense = {}
        for key, _, _ in STATES:
            rows = against[against["situation"] == key]
            passes, runs = rows[rows["is_pass"] == 1], rows[rows["is_pass"] == 0]
            defense[key] = {"passes": int(len(passes)), "pass_success": int(passes["success"].sum()),
                            "runs": int(len(runs)), "run_success": int(runs["success"].sum())}
        effects = {}
        for side, rows, priors in (("offense_effect", mine, OFFENSE_PRIOR), ("defense_effect", against, DEFENSE_PRIOR)):
            effects[side] = {}
            for k in ("pass", "run"):
                plays = rows[(rows["is_pass"] == 1) == (k == "pass")]
                effects[side][k] = round(_effect(plays["success"].sum(), len(plays), overall[k], priors[k]), 4)
        # How readable its calls are: the share guessed right from the
        # situation alone, and knowing its habits as well.
        own = mine[mine["situation"] != ""]
        lean = own["situation"].map(lambda key: offense[key]["lean"] or 0.0)
        guess_typical = (own["typical"] >= 0.5) == (own["is_pass"] == 1)
        guess_habit = ((own["typical"] + lean) >= 0.5) == (own["is_pass"] == 1)
        teams[team] = {
            "offense": offense, "habits": habits, "defense": defense, **effects,
            "readable": {"plays": int(len(own)), "situation": round(float(guess_typical.mean()), 4) if len(own) else None,
                         "habits": round(float(guess_habit.mean()), 4) if len(own) else None},
        }
    ranked = sorted((t for t in teams if teams[t]["readable"]["habits"] is not None), key=lambda t: -teams[t]["readable"]["habits"])
    for rank, team in enumerate(ranked, start=1):
        teams[team]["readable"]["rank"] = rank
    for key in league:
        league[key]["lean"] = round(league[key]["lean"], 4)
    return {
        "season": int(season), "through_week": int(p["week"].max()),
        "states": [{"key": key, "label": label, "group": group} for key, label, group in STATES],
        "habits": [{"key": key, "label": label} for key, label in HABITS],
        "league": league, "overall_success": {k: round(v, 4) for k, v in overall.items()}, "teams": teams,
    }


def expected_success(data: dict, offense: str, defense: str, key: str) -> dict:
    """{pass, run}: the success rate to expect when `offense` calls each
    in a situation against `defense`: the league's rate there, moved by
    how the offense does with that kind of play and how the defense does
    against it, both overall."""
    out = {}
    for k in ("pass", "run"):
        base = data["league"][key][f"{k}_success"]
        out[k] = round(float(np.clip(base + data["teams"][offense]["offense_effect"][k] + data["teams"][defense]["defense_effect"][k], 0.0, 1.0)), 4)
    return out


def load_plays(raw_dir: str | None = None, season: int | None = None) -> pd.DataFrame:
    import os

    season = config.NFL_SEASON if season is None else season
    path = os.path.join(raw_dir or config.NFL_RAW_DATA_DIR, f"pbp_{season}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame(columns=PLAY_COLUMNS)
    return pd.read_parquet(path, columns=PLAY_COLUMNS)

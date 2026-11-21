"""How each offense strings its calls together: the play-calling tree on
the NFL Stats tab, and the research behind it.

Why (2026-11-20, user request: "map play call sequencing tendencies per
team with a way of showing how the opposing defense does in those
situations"): the site rated what teams DO (EPA, units) but not what they
CHOOSE. A play call depends on the down and distance, and on what came
before it - a team that threw incomplete on first down faces the same 2nd
and 10 as one that was stuffed on a run, and calls it differently.

On the site (2026-11-21, user request: the first version - a box per
situation for both offenses of a game - was "too big, especially for the
picks page" and "doesn't really inform picks"; make it a tree on the
team's stats, the previous play branching to the next, colored by how
typical each sequence is, a layer per play of the drive or per result of
the play before): `build` counts how each team's drives OPEN - the first
TREE_DEPTH runs and passes in order, and whether each worked - for its
own drives and for the drives it faced on defense. The page draws the
tree from those counts.

The research (scripts/research_nfl_play_sequencing.py, 2021-2026) puts
every scrimmage play in one SITUATION - its down and distance, split by
what the play before it was where that matters (STATES) - and HABITS that
cut across them. An offense's pass rate in each is compared with what a
typical team calls in exactly those plays: nflfastR's expected pass rate
(`xpass`) plus the league's own lean in that situation. What it found: an
offense's lean in a situation IS a habit - between the odd and even weeks
of a season it correlates 0.25 to 0.57 on 15 to 80 plays a half
(LEAN_PRIOR is the pull toward the league those imply). A defense's
success allowed in one situation is NOT (0.0 to 0.15), and even its
whole-season rate against the pass (0.26) or the run (0.15) is mostly
noise in half a season. That is why the tree shows a defense's side as
what offenses chose against it and how often it worked, as a record, and
predicts nothing from it.
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
# Plays of prior toward the league for an offense's lean in one situation:
# from the half-season correlation (r at n plays a half gives a prior of
# n (1 - r) / r), about 0.45 at 45 plays.
LEAN_PRIOR = 55.0
# How many plays into a drive the tree goes.
TREE_DEPTH = 4
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


def drive_sequences(p: pd.DataFrame, depth: int = TREE_DEPTH) -> pd.DataFrame:
    """One row per drive (game_id, posteam, defteam, sequence): its first
    `depth` runs and passes in order, each as the call and whether it
    worked - "R" or "P", then "+" (nflfastR's success: the play added
    expected points) or "-". "P-R+R+" is an incompletion, then two runs
    that worked; a drive that ended sooner has a shorter sequence."""
    if p.empty:
        return pd.DataFrame(columns=["game_id", "posteam", "defteam", "sequence"])
    keys = ["game_id", "posteam", "defteam", "fixed_drive"]
    first = p[p.groupby(keys, sort=False).cumcount() < depth]
    token = pd.Series(np.where(first["is_pass"] == 1, "P", "R"), index=first.index) \
        + np.where(first["success"] == 1, "+", "-")
    out = token.groupby([first[k] for k in keys], sort=False).agg("".join).rename("sequence").reset_index()
    return out[["game_id", "posteam", "defteam", "sequence"]]


def _counts(sequences: pd.Series) -> dict:
    return {key: int(n) for key, n in sequences.value_counts().sort_index().items()}


def build(pbp: pd.DataFrame, season: int) -> dict:
    """The tree's data for one season:
    {season, through_week, depth, league: {sequence: drives},
     teams: {code: {offense: {sequence: drives}, defense: {...}}}}
    - how many of a team's drives opened with each sequence (offense), and
    how many of the drives it faced did (defense). The page folds these
    into the tree, so it can branch on the calls alone or on how each one
    went (docs/nfl_play_calling.js)."""
    p = scrimmage(pbp[pbp["season"] == season])
    if p.empty:
        return {"season": int(season), "through_week": 0, "depth": TREE_DEPTH, "league": {}, "teams": {}}
    drives = drive_sequences(p)
    teams = {}
    for team in sorted(set(drives["posteam"]) | set(drives["defteam"])):
        teams[team] = {"offense": _counts(drives.loc[drives["posteam"] == team, "sequence"]),
                       "defense": _counts(drives.loc[drives["defteam"] == team, "sequence"])}
    return {"season": int(season), "through_week": int(p["week"].max()), "depth": TREE_DEPTH,
            "league": _counts(drives["sequence"]), "teams": teams}


def load_plays(raw_dir: str | None = None, season: int | None = None) -> pd.DataFrame:
    import os

    season = config.NFL_SEASON if season is None else season
    path = os.path.join(raw_dir or config.NFL_RAW_DATA_DIR, f"pbp_{season}.parquet")
    if not os.path.exists(path):
        return pd.DataFrame(columns=PLAY_COLUMNS)
    return pd.read_parquet(path, columns=PLAY_COLUMNS)

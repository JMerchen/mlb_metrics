"""Pitches as sequences: the data side of the question "does what came
before a pitch change what it does?" (scripts/research_pitch_sequences.py
trains the models and has the results).

Why (2026-11-20): everything the site does with pitches - Stuff+, the
simulator's pitch outcomes, PAVE - treats each pitch on its own, given the
count. Pitchers and hitters talk about sequencing constantly: a fastball
up to set up the curve, never the same pitch three times. The newer way
to test that is to treat an at-bat like a sentence and let a sequence
model (the family behind language models) read the pitches in order.

A pitch here is one row of Statcast turned into:
  outcome   one of OUTCOMES: ball, called strike, whiff, foul, in play
  what      its type, speed, movement and location, mirrored for
            left-handed pitchers so "arm side" means the same for all
  context   the count, whether pitcher and hitter stand on the same side,
            the pitcher's pitch count in the game, times he has faced
            this hitter today
and a sequence is the pitches of one at-bat, oldest first, up to MAX_LEN
of them ending at the pitch being predicted.
"""

import numpy as np
import pandas as pd

OUTCOMES = ["ball", "called_strike", "whiff", "foul", "in_play"]
OUTCOME_OF = {
    "ball": "ball", "blocked_ball": "ball", "automatic_ball": "ball", "pitchout": "ball", "hit_by_pitch": "ball",
    "called_strike": "called_strike", "automatic_strike": "called_strike",
    "swinging_strike": "whiff", "swinging_strike_blocked": "whiff", "missed_bunt": "whiff", "foul_tip": "whiff",
    "foul": "foul", "foul_bunt": "foul", "bunt_foul_tip": "foul",
    "hit_into_play": "in_play",
}
PITCH_TYPES = ["FF", "SI", "SL", "CH", "ST", "FC", "CU", "FS", "KC", "SV", "other"]
NUMERIC = ["release_speed", "pfx_x", "pfx_z", "plate_x", "plate_z"]
COLUMNS = ["game_date", "game_pk", "at_bat_number", "pitch_number", "pitcher", "batter", "pitch_type", "description",
           "balls", "strikes", "stand", "p_throws", "game_type", *NUMERIC]
MAX_LEN = 8
SWINGS = ("whiff", "foul", "in_play")
# Pitch families, for how a hitter does against each kind.
FAMILY = {"FF": "fast", "SI": "fast", "FC": "fast", "SL": "breaking", "ST": "breaking", "CU": "breaking", "KC": "breaking",
          "SV": "breaking", "CH": "offspeed", "FS": "offspeed"}


def clean(pitches: pd.DataFrame) -> pd.DataFrame:
    """Regular-season pitches with a known outcome, type and measurements,
    in the order thrown, with the model's columns added: outcome (an index
    into OUTCOMES), type (an index into PITCH_TYPES), same_side, the
    numeric columns mirrored for left-handed pitchers, game_pitch (the
    pitcher's count in the game before this one) and times_faced (earlier
    at-bats against this hitter today)."""
    d = pitches[(pitches["game_type"] == "R") & pitches["description"].isin(OUTCOME_OF)].dropna(subset=NUMERIC + ["pitch_type"]).copy()
    d = d.sort_values(["game_pk", "at_bat_number", "pitch_number"], kind="stable").reset_index(drop=True)
    d["outcome"] = d["description"].map(OUTCOME_OF).map(OUTCOMES.index).astype(int)
    d["type"] = d["pitch_type"].where(d["pitch_type"].isin(PITCH_TYPES), "other").map(PITCH_TYPES.index).astype(int)
    left = d["p_throws"] == "L"
    for column in ("pfx_x", "plate_x"):
        d.loc[left, column] = -d.loc[left, column]
    d["same_side"] = (d["stand"] == d["p_throws"]).astype(float)
    d["family"] = d["pitch_type"].map(FAMILY).fillna("other")
    d["game_pitch"] = d.groupby(["game_pk", "pitcher"]).cumcount()
    first = d.drop_duplicates(["game_pk", "at_bat_number"])[["game_pk", "at_bat_number", "pitcher", "batter"]]
    first = first.assign(times_faced=first.groupby(["game_pk", "pitcher", "batter"]).cumcount())
    return d.merge(first[["game_pk", "at_bat_number", "times_faced"]], on=["game_pk", "at_bat_number"], how="left")


def player_rates(train: pd.DataFrame, strength: float = 200.0, type_strength: float = 100.0) -> dict:
    """What the models need to know about who is involved, from the
    training pitches only, each rate pulled toward the league's:

      pitcher, batter   swing rate, whiffs per swing, balls in play per
                        swing (pulled with `strength` pitches)
      pitcher_type      for each of a pitcher's pitch types, how often he
                        throws it and its whiffs per swing
      batter_family     a hitter's whiffs per swing against fastballs,
                        breaking balls and off-speed

    The last two matter for a fair test: a slider thrown after a slider
    comes mostly from pitchers with a great slider, and without them that
    would pass for a sequencing effect. A player (or pitch) not in the
    training pitches gets the league's."""
    swing = train["outcome"].isin([OUTCOMES.index(o) for o in SWINGS])
    whiff = train["outcome"] == OUTCOMES.index("whiff")
    in_play = train["outcome"] == OUTCOMES.index("in_play")
    league = {"swing": float(swing.mean()), "whiff": float(whiff.sum() / swing.sum()), "in_play": float(in_play.sum() / swing.sum())}
    counts = pd.DataFrame({"n": 1, "swing": swing, "whiff": whiff, "in_play": in_play})
    out = {"league": league}
    for who in ("pitcher", "batter"):
        g = counts.groupby(train[who]).sum()
        out[who] = pd.DataFrame({
            f"{who}_swing": (g["swing"] + strength * league["swing"]) / (g["n"] + strength),
            f"{who}_whiff": (g["whiff"] + strength * league["swing"] * league["whiff"]) / (g["swing"] + strength * league["swing"]),
            f"{who}_in_play": (g["in_play"] + strength * league["swing"] * league["in_play"]) / (g["swing"] + strength * league["swing"]),
        })
    # Each pitch type's and family's own league whiff rate is the prior.
    type_league = counts.groupby(train["type"]).sum()
    type_rate = (type_league["whiff"] / type_league["swing"]).to_dict()
    g = counts.groupby([train["pitcher"], train["type"]]).sum()
    prior = g.index.get_level_values("type").map(type_rate).to_numpy()
    thrown = g["n"] / g.groupby(level="pitcher")["n"].transform("sum")
    out["pitcher_type"] = pd.DataFrame({
        "type_usage": thrown, "type_whiff": (g["whiff"] + type_strength * prior) / (g["swing"] + type_strength)})
    out["type_league"] = type_rate
    family_league = counts.groupby(train["family"]).sum()
    family_rate = (family_league["whiff"] / family_league["swing"]).to_dict()
    g = counts.groupby([train["batter"], train["family"]]).sum()
    prior = g.index.get_level_values("family").map(family_rate).to_numpy()
    out["batter_family"] = pd.DataFrame({"family_whiff": (g["whiff"] + type_strength * prior) / (g["swing"] + type_strength)})
    out["family_league"] = family_rate
    return out


def add_player_rates(d: pd.DataFrame, rates: dict) -> pd.DataFrame:
    out = d
    for who in ("pitcher", "batter"):
        out = out.merge(rates[who], left_on=who, right_index=True, how="left")
        for name in ("swing", "whiff", "in_play"):
            out[f"{who}_{name}"] = out[f"{who}_{name}"].fillna(rates["league"][name])
    out = out.merge(rates["pitcher_type"], left_on=["pitcher", "type"], right_index=True, how="left")
    out["type_usage"] = out["type_usage"].fillna(0.0)
    out["type_whiff"] = out["type_whiff"].fillna(out["type"].map(rates["type_league"])).fillna(rates["league"]["whiff"])
    out = out.merge(rates["batter_family"], left_on=["batter", "family"], right_index=True, how="left")
    out["family_whiff"] = out["family_whiff"].fillna(out["family"].map(rates["family_league"])).fillna(rates["league"]["whiff"])
    return out


def add_training_rates(train: pd.DataFrame, folds: int = 5) -> pd.DataFrame:
    """add_player_rates for the training pitches themselves, with each
    game's rates taken from the OTHER games (in `folds` groups of games).

    Why: a rate that includes the pitch it is attached to tells the model
    part of the answer, so it leans on the rate more than it should, and
    on new pitches - where the rate knows nothing of the outcome - it
    overshoots. Measured here before the fix: whiffs on breaking and
    off-speed pitches were over-predicted by 1 to 3 points."""
    games = train["game_pk"].to_numpy()
    codes = pd.factorize(games)[0] % folds
    parts = []
    for fold in range(folds):
        inside = codes == fold
        parts.append(add_player_rates(train[inside], player_rates(train[~inside])))
    return pd.concat(parts).sort_values(["game_pk", "at_bat_number", "pitch_number"], kind="stable").reset_index(drop=True)


STATIC = ["balls", "strikes", "same_side", "game_pitch", "times_faced", "pitcher_swing", "pitcher_whiff", "pitcher_in_play",
          "batter_swing", "batter_whiff", "batter_in_play", "type_usage", "type_whiff", "family_whiff"]


def previous_pitch_features(d: pd.DataFrame) -> pd.DataFrame:
    """Hand-made sequence features for a model that sees one pitch at a
    time: the pitch before this one in the at-bat (its type, speed,
    location and result) and how this one differs from it. -1 / 0 when
    this is the first pitch."""
    at_bat = [d["game_pk"], d["at_bat_number"]]
    prev = d.groupby(at_bat)[["type", "outcome", *NUMERIC]].shift(1)
    first = prev["type"].isna()
    out = pd.DataFrame({
        "prev_type": prev["type"].fillna(-1), "prev_outcome": prev["outcome"].fillna(-1), "first_pitch": first.astype(float),
        "same_type": ((prev["type"] == d["type"]) & ~first).astype(float),
        "speed_change": (d["release_speed"] - prev["release_speed"]).fillna(0.0),
        "moved_x": (d["plate_x"] - prev["plate_x"]).fillna(0.0), "moved_z": (d["plate_z"] - prev["plate_z"]).fillna(0.0),
        "prev_plate_x": prev["plate_x"].fillna(0.0), "prev_plate_z": prev["plate_z"].fillna(0.0),
    })
    return out


def sequences(d: pd.DataFrame, scale: dict, history: str = "results") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(steps, static, outcome) arrays for a sequence model.

    steps[i] is MAX_LEN rows ending at pitch i, earlier pitches of the
    same at-bat before it and zeros before those. Each row: the pitch type
    (one-hot), its scaled numbers, a flag for "this is the pitch being
    predicted", and - for the earlier pitches only - how each ended
    (one-hot). history: "none" keeps only the pitch itself, "pitches" adds
    what was thrown before it, "results" also how each earlier pitch ended.
    scale: {column: (mean, sd)} from the training pitches."""
    n = len(d)
    width = len(PITCH_TYPES) + len(NUMERIC) + 1 + len(OUTCOMES)
    row = np.zeros((n, width), dtype=np.float32)
    row[np.arange(n), d["type"].to_numpy()] = 1.0
    for j, column in enumerate(NUMERIC):
        mean, sd = scale[column]
        row[:, len(PITCH_TYPES) + j] = (d[column].to_numpy() - mean) / sd
    result = np.zeros((n, len(OUTCOMES)), dtype=np.float32)
    result[np.arange(n), d["outcome"].to_numpy()] = 1.0
    position = d.groupby(["game_pk", "at_bat_number"]).cumcount().to_numpy()
    steps = np.zeros((n, MAX_LEN, width), dtype=np.float32)
    steps[:, -1, :] = row
    steps[:, -1, len(PITCH_TYPES) + len(NUMERIC)] = 1.0            # the pitch being predicted; its result stays hidden
    if history != "none":
        index = np.arange(n)
        for back in range(1, MAX_LEN):
            has = position >= back
            source = index[has] - back
            steps[has, MAX_LEN - 1 - back, : len(PITCH_TYPES) + len(NUMERIC)] = row[source, : len(PITCH_TYPES) + len(NUMERIC)]
            if history == "results":
                steps[has, MAX_LEN - 1 - back, -len(OUTCOMES):] = result[source]
    static = np.column_stack([(d[c].to_numpy(dtype=np.float32) - scale[c][0]) / scale[c][1] for c in STATIC]).astype(np.float32)
    return steps, static, d["outcome"].to_numpy()


def fit_scale(train: pd.DataFrame) -> dict:
    return {c: (float(train[c].mean()), float(train[c].std()) or 1.0) for c in NUMERIC + STATIC}

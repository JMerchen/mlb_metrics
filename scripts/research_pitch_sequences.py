"""Does what came before a pitch change what it does? A sequence model
reads each at-bat in order (src/mlb_metrics/pitch_sequences.py has the
data side and the why).

Every pitch of 2025 and 2026 (regular season, Statcast) ends one of five
ways: ball, called strike, whiff, foul, in play. Each model predicts the
chances of the five and is scored by log loss (lower is better) on pitches
it never saw: trained on 2025 through May 2026, tuned on June 2026, scored
on July-October 2026.

  pitch alone        a boosted-tree model given the pitch (type, speed,
                     movement, location), the count, the handedness
                     matchup, the pitcher's pitch count, times through the
                     order, how often this pitcher and this hitter get
                     swings, whiffs and balls in play, how good this
                     pitcher's version of this pitch is and how this
                     hitter does against its kind
  + previous pitch   the same, plus hand-made features of the pitch
                     before it (type, speed and location change, result)
  network: none      a recurrent network (GRU) shown only the pitch itself
  network: pitches   ... shown every earlier pitch of the at-bat
  network: results   ... and how each of those ended

The three networks differ only in what history they are shown, so the
gaps between them are the worth of the sequence and nothing else.

Needs tensorflow (pip install tensorflow-cpu); it is not a dependency of
the site. About 25 minutes on four cores.

RESULT (2026-11-20; rerun the same day with three trainings of each
network, each run until the June pitches stopped improving - 13 to 40
passes - where the first run capped two trainings at 14): the sequence
is real and small. Reading the at-bat cuts the network's log loss by
0.0080 a pitch (95% 0.0074 to 0.0086, from resampling games): 1.4% of what
the pitch itself tells you. (The first run said 0.0093 and 1.6%; trained
to the end, the network shown no history closed some of the gap.) About
60% of that is what was thrown before, 40% how each of those ended. It
grows through the at-bat - nothing on the first pitch, 0.0192 by the
sixth - and it is knowledge both of whether he swings and of what happens
when he does. One pitch back, hand-made, gets a tenth of it (0.0008): the
worth is in the whole at-bat, which is the case for a sequence model. The
network also beats the trees outright by 0.0156.

What the patterns look like (whiffs per swing against what the pitch
alone predicts): REPEATING a kind of pitch gets more whiffs than the
pitch alone says - a fastball after a fastball +1.2 points, off-speed or
breaking after another +0.8 - and SWITCHING gets fewer: off-speed or
breaking right after a fastball -2.0, a pitch 8+ mph slower than the last
-1.0. After a whiff the next swing misses more (+1.4), after a foul less
(-1.4). That is the opposite of "change speeds to keep him off balance",
but read it with care: pitchers choose. They go back to a pitch when the
hitter has looked bad on it, so this describes what happens, not what to
call. The network that read the at-bat gets those rows right to within a
point, all but two (off-speed to open the at-bat, and a pitch 2+ feet
from the last).

A note on method: an earlier run gave each training pitch player rates
that included the pitch itself, and averaged each swing's own whiff ratio.
Both were wrong (pitch_sequences.add_training_rates; expected_rate below);
neither changed the headline, both changed the table.

  970,546 pitches to train on, 115,446 to tune, 337,891 scored (1,148 games, July-October 2026)
  log loss per pitch (always saying the league's rates: 1.5332)
    pitch alone          0.9617
    + previous pitch     0.9608
    network: none        0.9533
    network: pitches     0.9483
    network: results     0.9453
    gain from the previous pitch, hand-made (trees): 0.0008 (95% 0.0003 to 0.0015), 0.1% of what pitch alone knows
    gain from what was thrown before (network): 0.0050 (95% 0.0045 to 0.0055), 0.9% of what network: none knows
    gain from ... and how each ended (network): 0.0030 (95% 0.0027 to 0.0033), 0.5% of what network: pitches knows
    gain from the whole sequence (network): 0.0080 (95% 0.0074 to 0.0086), 1.4% of what network: none knows
    gain from the best network over the best trees: 0.0156 (95% 0.0147 to 0.0164), 2.7% of what + previous pitch knows
  where the sequence's gain comes from (network: none against network: results)
    will he swing?                    0.4390 -> 0.4329 (gain 0.0061)
    if he swings: whiff, foul, in play 0.9402 -> 0.9367 (gain 0.0035 a swing)
    by pitch of the at-bat: 1st -0.0009, 2nd +0.0054, 3rd +0.0087, 4th +0.0142, 5th +0.0172, 6th on +0.0192
  whiffs per swing: what happened, what the pitch alone predicts (trees), and what the network that read the at-bat predicts
    every swing                                    162,253 swings: 0.250 against 0.249 expected (+0.1 points, +-0.2); the network 0.250
    first pitch of the at-bat                       28,237 swings: 0.261 against 0.261 expected (+0.1 points, +-0.5); the network 0.260
    same pitch type as the one before               45,308 swings: 0.256 against 0.247 expected (+0.9 points, +-0.4); the network 0.258
    the same type a third time running              12,684 swings: 0.244 against 0.238 expected (+0.6 points, +-0.7); the network 0.247
    a different type from the one before            88,708 swings: 0.243 against 0.246 expected (-0.3 points, +-0.3); the network 0.242
    off-speed or breaking to open the at-bat        10,313 swings: 0.361 against 0.371 expected (-1.0 points, +-0.9); the network 0.376
    off-speed or breaking right after a fastball    30,520 swings: 0.294 against 0.314 expected (-2.0 points, +-0.5); the network 0.302
    off-speed or breaking after another one         30,096 swings: 0.329 against 0.321 expected (+0.8 points, +-0.5); the network 0.334
    a fastball right after a fastball               42,608 swings: 0.203 against 0.191 expected (+1.2 points, +-0.4); the network 0.201
    a fastball right after off-speed or breaking    30,792 swings: 0.181 against 0.186 expected (-0.5 points, +-0.4); the network 0.177
    8+ mph slower than the pitch before             19,614 swings: 0.298 against 0.308 expected (-1.0 points, +-0.6); the network 0.303
    8+ mph faster than the pitch before             21,922 swings: 0.190 against 0.192 expected (-0.2 points, +-0.5); the network 0.184
    2+ feet from where the last one crossed         26,645 swings: 0.309 against 0.307 expected (+0.2 points, +-0.6); the network 0.298
    within half a foot of the last one              14,054 swings: 0.214 against 0.219 expected (-0.5 points, +-0.7); the network 0.222
    after a whiff                                   13,707 swings: 0.307 against 0.293 expected (+1.4 points, +-0.8); the network 0.308
    after a foul                                    35,723 swings: 0.229 against 0.243 expected (-1.4 points, +-0.4); the network 0.235
    after taking a ball                             58,509 swings: 0.238 against 0.234 expected (+0.4 points, +-0.3); the network 0.237

Usage:
  python scripts/research_pitch_sequences.py --out DIR
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mlb_metrics import pitch_sequences as ps  # noqa: E402

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")
TRAIN_END, VALID_END = "2026-06-01", "2026-07-01"
# Three trainings of each network, averaged. Each runs until the June
# pitches stop improving for three passes (up to MAX_PASSES).
SEEDS = (1, 2, 3)
MAX_PASSES = 40


def load() -> pd.DataFrame:
    raw = pd.concat([pd.read_parquet(f, columns=ps.COLUMNS) for f in sorted(glob.glob(os.path.join(RAW, "statcast_20*.parquet")))])
    d = ps.clean(raw)
    d["game_date"] = pd.to_datetime(d["game_date"])
    d["part"] = np.where(d["game_date"] < TRAIN_END, "train", np.where(d["game_date"] < VALID_END, "valid", "test"))
    # Player rates: later pitches get them from all the training pitches;
    # the training pitches get them from other games (add_training_rates).
    train = d[d["part"] == "train"]
    later = ps.add_player_rates(d[d["part"] != "train"], ps.player_rates(train))
    out = pd.concat([ps.add_training_rates(train), later])
    return out.sort_values(["game_pk", "at_bat_number", "pitch_number"], kind="stable").reset_index(drop=True)


def log_loss(prob: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Each pitch's log loss."""
    return -np.log(np.clip(prob[np.arange(len(y)), y], 1e-6, 1.0))


def tree_models(d: pd.DataFrame) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    base = pd.concat([d[["type", *ps.NUMERIC, *ps.STATIC]], ], axis=1)
    with_previous = pd.concat([base, ps.previous_pitch_features(d)], axis=1)
    train, valid, test = ((d["part"] == part).to_numpy() for part in ("train", "valid", "test"))
    out = {}
    for name, X in (("pitch alone", base), ("+ previous pitch", with_previous)):
        categorical = [c for c in ("type", "prev_type", "prev_outcome") if c in X.columns]
        X = X.assign(**{c: X[c] + 1 for c in categorical if c != "type"})
        # Stopped on June 2026, the same pitches the networks are tuned on.
        model = HistGradientBoostingClassifier(max_iter=2000, learning_rate=0.05, max_leaf_nodes=63, early_stopping=True,
                                               n_iter_no_change=25, random_state=0, categorical_features=categorical)
        model.fit(X[train], d.loc[train, "outcome"], X_val=X[valid], y_val=d.loc[valid, "outcome"])
        out[name] = model.predict_proba(X[test])
        print(f"  trained {name}", file=sys.stderr)
    return out


def network(history: str, d: pd.DataFrame, scale: dict, seed: int) -> np.ndarray:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    parts = {}
    for part in ("train", "valid", "test"):
        steps, static, y = ps.sequences(d[d["part"] == part], scale, history)
        parts[part] = ([steps, static], y)
    steps_in = tf.keras.layers.Input(parts["train"][0][0].shape[1:])
    static_in = tf.keras.layers.Input(parts["train"][0][1].shape[1:])
    read = tf.keras.layers.GRU(96)(steps_in)
    x = tf.keras.layers.Concatenate()([read, static_in])
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    model = tf.keras.Model([steps_in, static_in], tf.keras.layers.Dense(len(ps.OUTCOMES), activation="softmax")(x))
    model.compile(tf.keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy")
    stop = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
    fit = model.fit(*parts["train"], validation_data=parts["valid"], batch_size=2048, epochs=MAX_PASSES, callbacks=[stop], verbose=0)
    print(f"  trained network: {history} (seed {seed}), {len(fit.history['loss'])} passes, tuned loss {min(fit.history['val_loss']):.5f}", file=sys.stderr)
    return model.predict(parts["test"][0], batch_size=8192, verbose=0)


def report(d: pd.DataFrame, probs: dict) -> None:
    test = d[d["part"] == "test"].reset_index(drop=True)
    y = test["outcome"].to_numpy()
    losses = {name: log_loss(p, y) for name, p in probs.items()}
    base_rate = np.bincount(d.loc[d["part"] == "train", "outcome"], minlength=len(ps.OUTCOMES)) / (d["part"] == "train").sum()
    floor = log_loss(np.tile(base_rate, (len(y), 1)), y).mean()
    print(f"{(d['part'] == 'train').sum():,} pitches to train on, {(d['part'] == 'valid').sum():,} to tune, {len(test):,} scored "
          f"({test['game_pk'].nunique():,} games, July-October 2026)")
    print(f"log loss per pitch (always saying the league's rates: {floor:.4f})")
    rng = np.random.default_rng(0)
    games = test["game_pk"].to_numpy()
    codes, index = np.unique(games, return_inverse=True)

    def against(a, b):
        """Mean of (a - b) with a 95% interval from resampling games."""
        diff = losses[a] - losses[b]
        sums = np.bincount(index, weights=diff, minlength=len(codes))
        counts = np.bincount(index, minlength=len(codes))
        draws = rng.integers(0, len(codes), (1000, len(codes)))
        boot = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
        return diff.mean(), np.percentile(boot, 2.5), np.percentile(boot, 97.5)

    for name in probs:
        print(f"  {name:20s} {losses[name].mean():.4f}")
    for label, a, b in (("the previous pitch, hand-made (trees)", "pitch alone", "+ previous pitch"),
                        ("what was thrown before (network)", "network: none", "network: pitches"),
                        ("... and how each ended (network)", "network: pitches", "network: results"),
                        ("the whole sequence (network)", "network: none", "network: results"),
                        ("the best network over the best trees", "+ previous pitch", "network: results")):
        if a in losses and b in losses:
            mean, low, high = against(a, b)
            print(f"  gain from {label}: {mean:.4f} (95% {low:.4f} to {high:.4f}), {100 * mean / (floor - losses[a].mean()):.1f}% of what {a} knows")

    # What kind of knowledge the sequence is: about whether he swings, or
    # about what happens when he does? And how deep into the at-bat?
    if "network: none" in probs and "network: results" in probs:
        swing_columns = [ps.OUTCOMES.index(o) for o in ps.SWINGS]
        swung = np.isin(y, swing_columns)

        def swing_loss(p):
            chance = np.clip(p[:, swing_columns].sum(axis=1), 1e-6, 1 - 1e-6)
            return -np.where(swung, np.log(chance), np.log(1 - chance))

        def after_swing_loss(p):
            given = p[:, swing_columns] / p[:, swing_columns].sum(axis=1, keepdims=True)
            return -np.log(np.clip(given[np.arange(len(y)), np.searchsorted(swing_columns, np.where(swung, y, swing_columns[0]))], 1e-6, 1))[swung]

        none, full = probs["network: none"], probs["network: results"]
        print("where the sequence's gain comes from (network: none against network: results)")
        print(f"  will he swing?                    {swing_loss(none).mean():.4f} -> {swing_loss(full).mean():.4f} (gain {swing_loss(none).mean() - swing_loss(full).mean():.4f})")
        print(f"  if he swings: whiff, foul, in play {after_swing_loss(none).mean():.4f} -> {after_swing_loss(full).mean():.4f} "
              f"(gain {after_swing_loss(none).mean() - after_swing_loss(full).mean():.4f} a swing)")
        number = test.groupby(["game_pk", "at_bat_number"]).cumcount().to_numpy() + 1
        gain = losses["network: none"] - losses["network: results"]
        print("  by pitch of the at-bat: " + ", ".join(
            f"{label} {gain[use].mean():+.4f}" for label, use in (("1st", number == 1), ("2nd", number == 2), ("3rd", number == 3),
                                                                 ("4th", number == 4), ("5th", number == 5), ("6th on", number >= 6))))

    # Where the sequence shows: whiffs per swing next to what the pitch
    # alone leads the trees to expect, by what came just before.
    if "pitch alone" in probs:
        prev = ps.previous_pitch_features(test)
        swing = np.isin(y, [ps.OUTCOMES.index(o) for o in ps.SWINGS])
        whiff = y == ps.OUTCOMES.index("whiff")
        # Expected whiffs per swing for a group: the whiffs the model
        # expects over the swings it expects, summed over ALL the group's
        # pitches. (Averaging each swing's own ratio instead runs high
        # wherever swings are rare - breaking balls off the plate.)
        swing_columns = [ps.OUTCOMES.index(o) for o in ps.SWINGS]

        def expected_rate(p, use):
            return p[use, ps.OUTCOMES.index("whiff")].sum() / p[use][:, swing_columns].sum()

        fast = np.isin(test["type"], [ps.PITCH_TYPES.index(t) for t in ("FF", "SI", "FC")])
        prev_fast = np.isin(prev["prev_type"], [ps.PITCH_TYPES.index(t) for t in ("FF", "SI", "FC")])
        twice = (prev["same_type"] == 1).to_numpy()
        before = test.groupby(["game_pk", "at_bat_number"])["type"].shift(2)
        thrice = twice & (before == test["type"]).to_numpy()
        groups = [
            ("every swing", np.ones(len(y), dtype=bool)),
            ("first pitch of the at-bat", (prev["first_pitch"] == 1).to_numpy()),
            ("same pitch type as the one before", twice),
            ("the same type a third time running", thrice),
            ("a different type from the one before", (prev["first_pitch"] == 0).to_numpy() & ~twice),
            ("off-speed or breaking to open the at-bat", ~fast & (prev["first_pitch"] == 1).to_numpy()),
            ("off-speed or breaking right after a fastball", ~fast & prev_fast),
            ("off-speed or breaking after another one", ~fast & ~prev_fast & (prev["first_pitch"] == 0).to_numpy()),
            ("a fastball right after a fastball", fast & prev_fast),
            ("a fastball right after off-speed or breaking", fast & ~prev_fast & (prev["first_pitch"] == 0).to_numpy()),
            ("8+ mph slower than the pitch before", (prev["speed_change"] <= -8).to_numpy()),
            ("8+ mph faster than the pitch before", (prev["speed_change"] >= 8).to_numpy()),
            ("2+ feet from where the last one crossed", (np.hypot(prev["moved_x"], prev["moved_z"]) >= 2).to_numpy() & (prev["first_pitch"] == 0).to_numpy()),
            ("within half a foot of the last one", (np.hypot(prev["moved_x"], prev["moved_z"]) <= 0.5).to_numpy() & (prev["first_pitch"] == 0).to_numpy()),
            ("after a whiff", (prev["prev_outcome"] == ps.OUTCOMES.index("whiff")).to_numpy()),
            ("after a foul", (prev["prev_outcome"] == ps.OUTCOMES.index("foul")).to_numpy()),
            ("after taking a ball", (prev["prev_outcome"] == ps.OUTCOMES.index("ball")).to_numpy()),
        ]
        print("whiffs per swing: what happened, what the pitch alone predicts (trees), and what the network that read the at-bat predicts")
        for label, use in groups:
            n = int((use & swing).sum())
            actual = whiff[use & swing].mean()
            expected = expected_rate(probs["pitch alone"], use)
            se = np.sqrt(actual * (1 - actual) / n)
            read = f"; the network {expected_rate(probs['network: results'], use):.3f}" if "network: results" in probs else ""
            print(f"  {label:46s} {n:7,} swings: {actual:.3f} against {expected:.3f} expected ({100 * (actual - expected):+.1f} points, +-{196 * se:.1f}){read}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="Folder to keep each model's predictions in (reused if present).")
    parser.add_argument("--only", choices=["trees", "networks", "report"], help="Run one stage.")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    d = load()
    saved = lambda name: os.path.join(args.out, name.replace(" ", "_").replace(":", "").replace("+", "plus") + ".npy")
    if args.only in (None, "trees"):
        for name, prob in tree_models(d).items():
            np.save(saved(name), prob)
    if args.only in (None, "networks"):
        scale = ps.fit_scale(d[d["part"] == "train"])
        for history in ("none", "pitches", "results"):
            runs = [network(history, d, scale, seed) for seed in SEEDS]
            np.save(saved(f"network: {history}"), np.mean(runs, axis=0))
    names = ["pitch alone", "+ previous pitch", "network: none", "network: pitches", "network: results"]
    report(d, {name: np.load(saved(name)) for name in names if os.path.exists(saved(name))})


if __name__ == "__main__":
    main()

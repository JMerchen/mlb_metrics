"""Our own wins above replacement for MLB players, from every pitch.

Why (2026-11-23, user request: "when a hit occurs and in what situation.
Or when an error occurs ... Or when an out occurs and the situation for
both a player batting and the player making the out. We could grade based
on difficulty also, like if an out was expected and made ... composite
these into a singular score and judge against the league average for that
position and a particular situation. Almost like our own version of war").

Every pitch in Statcast carries the change it made to the batting team's
expected runs in that exact situation - inning aside: the outs, the
runners and the count (delta_run_exp). Summed over a plate appearance
that is what the batter did in the spot he was in; a single with the
bases loaded is worth more than one with nobody on. Everything here is
built from that number:

  BATTING   A plate appearance's value belongs to the batter - less the
            fielders' part (below), so a line drive speared by a diving
            shortstop counts as what that ball usually does, not as an
            out. Fit together with the pitcher, the park and which side
            each throws and bats from (a mixed model, nfl_player_value's
            mixed_fit), so a hitter is measured after allowing for whom
            he faced and where.
  PITCHING  The same plate appearances from the other side: the same fit
            gives each pitcher what is left for him.
  FIELDING  For every ball in play, the chance a ball hit that hard, at
            that angle, to that part of the field is turned into an out
            (OUT MODEL), and whose ball it was (WHOSE MODEL: of the outs
            made on balls like it, the share made at each position). The
            fielder who makes an out gets one minus the chance; when it
            falls in, the positions it belonged to share the blame. In
            runs: times what an out is worth over a hit on that kind of
            ball.
A player's runs are compared with the average at his position - a
catcher's bat with catchers', a starter with starters - then with a
reserve's (REPLACEMENT), and RUNS_PER_WIN turns runs into wins.

What it cannot see: where the fielders were standing or how fast they
ran, so range is judged from where the ball went; a catcher's framing and
blocking; and baserunning - Statcast's per-pitch number moves with the
count, not with a steal, and an extra base taken on a hit is on the
batter's ledger. scripts/research_mlb_war.py has the checks.
"""

import numpy as np
import pandas as pd

from mlb_metrics.nfl_player_value import mixed_fit

PITCH_COLUMNS = [
    "game_pk", "game_date", "game_type", "at_bat_number", "pitch_number", "batter", "pitcher", "events", "description",
    "delta_run_exp", "inning", "inning_topbot", "home_team", "away_team", "stand", "p_throws", "hit_location", "bb_type",
    "launch_speed", "launch_angle", "hc_x", "hc_y", "on_1b", "on_2b", "on_3b", "outs_when_up", "balls", "strikes",
    "fielder_2", "fielder_3", "fielder_4", "fielder_5", "fielder_6", "fielder_7", "fielder_8", "fielder_9",
    "post_home_score", "post_away_score", "age_bat", "age_pit", "player_name", "des",
]
# Ball-in-play results where the defense got an out, and where it did not.
OUT_EVENTS = {"field_out", "force_out", "grounded_into_double_play", "double_play", "fielders_choice_out", "sac_fly",
              "sac_bunt", "sac_fly_double_play", "triple_play"}
SAFE_EVENTS = {"single", "double", "triple", "field_error", "fielders_choice"}
POSITIONS = {1: "P", 2: "C", 3: "1B", 4: "2B", 5: "3B", 6: "SS", 7: "LF", 8: "CF", 9: "RF"}


def plate_appearances(pitches: pd.DataFrame) -> pd.DataFrame:
    """One row per regular-season plate appearance: who batted and
    pitched, for and against which team, the park, its value (the summed
    change in expected runs over its pitches), how it ended and, for a
    ball in play, how it was hit and who was in the field."""
    p = pitches[pitches["game_type"] == "R"].sort_values(["game_pk", "at_bat_number", "pitch_number"])
    key = ["game_pk", "at_bat_number"]
    value = p.groupby(key)["delta_run_exp"].sum().rename("value")
    last = p.groupby(key).tail(1).set_index(key)
    pa = last[["game_date", "batter", "pitcher", "events", "inning", "inning_topbot", "home_team", "away_team", "stand", "p_throws",
               "hit_location", "bb_type", "launch_speed", "launch_angle", "hc_x", "hc_y", "outs_when_up", "age_bat", "age_pit"]
              + [f"fielder_{k}" for k in range(2, 10)]].copy()
    pa["value"] = value
    # A plate appearance cut short by a runner's out is nobody's result.
    pa = pa[pa["events"].notna() & (pa["events"] != "truncated_pa")].reset_index()
    top = pa["inning_topbot"] == "Top"
    pa["bat_team"] = np.where(top, pa["away_team"], pa["home_team"])
    pa["field_team"] = np.where(top, pa["home_team"], pa["away_team"])
    pa["fielder_1"] = pa["pitcher"]
    pa["in_play"] = pa["events"].isin(OUT_EVENTS | SAFE_EVENTS) & pa["bb_type"].notna()
    pa["out"] = pa["events"].isin(OUT_EVENTS)
    # Spray angle: 0 is straight up the middle, negative toward left field.
    pa["spray"] = np.degrees(np.arctan2(pa["hc_x"] - 125.42, 198.27 - pa["hc_y"]))
    return pa


# ---- fielding ----

FIELD_FEATURES = ["launch_speed", "launch_angle", "spray", "bats_left", "bunt"]
FIELD_FOLDS = 5


def _field_features(balls: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "launch_speed": balls["launch_speed"], "launch_angle": balls["launch_angle"], "spray": balls["spray"],
        "bats_left": (balls["stand"] == "L").astype(float), "bunt": balls["events"].eq("sac_bunt").astype(float),
    }, index=balls.index)[FIELD_FEATURES]


def fielding_chances(pa: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """The balls in play of `pa` with, for each: chance (the chance a ball
    hit like it is turned into an out), whose_1 .. whose_9 (the share of
    the outs on balls like it made at each position) and gap (runs an out
    is worth over a hit on that kind of ball).

    Both models are gradient-boosted trees on how hard, how high and
    where the ball was hit, which side the batter hits from and whether
    he bunted. Every ball's numbers come from models fit on other games
    (FIELD_FOLDS folds by game), so a fielder is never graded against a
    model that has seen his own play."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold

    balls = pa[pa["in_play"]].copy()
    features = _field_features(balls)
    out = balls["out"].to_numpy().astype(int)
    position = pd.to_numeric(balls["hit_location"], errors="coerce").astype(float)
    made = (out == 1) & position.between(1, 9).to_numpy(dtype=bool)
    chance = np.zeros(len(balls))
    whose = np.zeros((len(balls), 9))
    for train, test in GroupKFold(n_splits=FIELD_FOLDS).split(features, out, balls["game_pk"]):
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)
        model.fit(features.iloc[train], out[train])
        chance[test] = model.predict_proba(features.iloc[test])[:, 1]
        rows = train[made[train]]
        who = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)
        who.fit(features.iloc[rows], position.iloc[rows].astype(int))
        shares = who.predict_proba(features.iloc[test])
        for column, k in enumerate(who.classes_):
            whose[test, int(k) - 1] = shares[:, column]
    balls["chance"] = chance
    for k in range(1, 10):
        balls[f"whose_{k}"] = whose[:, k - 1]
    by_type = balls.groupby("bb_type").apply(lambda g: g.loc[~g["out"], "value"].mean() - g.loc[g["out"], "value"].mean())
    balls["gap"] = balls["bb_type"].map(by_type).fillna(float(by_type.mean()))
    return balls


def fielding_credits(balls: pd.DataFrame) -> pd.DataFrame:
    """One row per (ball in play, fielder with a stake in it): position,
    plays (his share of the ball: 1 when he made the out or the error,
    whose_k when it fell in), outs_above (what he got: 1 - chance for an
    out made; -chance x his share when it was not) and runs (x gap)."""
    rows = []
    position = pd.to_numeric(balls["hit_location"], errors="coerce")
    error = balls["events"] == "field_error"
    for k in range(1, 10):
        fielder = balls[f"fielder_{k}"]
        own = position == k
        made = balls["out"] & own
        charged = error & own
        # A hit, or an error nobody is named on, is shared by whose ball it was.
        shared = ~balls["out"] & ~(error & position.between(1, 9))
        share = np.where(made | charged, 1.0, np.where(shared, balls[f"whose_{k}"], 0.0))
        above = np.where(made, 1.0 - balls["chance"], np.where(charged, -balls["chance"], np.where(shared, -balls["chance"] * balls[f"whose_{k}"], 0.0)))
        keep = (share > 0) & fielder.notna()
        rows.append(pd.DataFrame({
            "game_pk": balls.loc[keep, "game_pk"], "player_id": fielder[keep].astype(int), "team": balls.loc[keep, "field_team"],
            "position": POSITIONS[k], "plays": share[keep], "outs_above": above[keep], "runs": above[keep] * balls.loc[keep, "gap"],
        }))
    return pd.concat(rows, ignore_index=True)


# ---- batting and pitching ----

# REPLACEMENT LEVEL is the players a team reaches for when its roster
# runs out: its hitters past the 13 with the most plate appearances (a
# roster carries 13 position players), its starters past the 6 who faced
# the most batters and its relievers past the 8. Their level is fit from
# the season itself, next to the level of everyone inside those cuts.
ROSTER = {"bat": 13, "SP": 6, "RP": 8}
# A pitcher is a starter when he started at least this share of his games.
STARTER_SHARE = 0.5


def neutral_value(pa: pd.DataFrame, balls: pd.DataFrame) -> pd.Series:
    """Each plate appearance's value with the fielders' part taken out: a
    ball in play counts as what a ball hit like it usually does, in runs
    (an out made on a 30% chance is 70% of an out-over-a-hit worse for
    the pitcher than the scorebook says)."""
    luck = pd.Series(0.0, index=pa.index)
    luck.loc[balls.index] = (balls["out"].astype(float) - balls["chance"]) * balls["gap"]
    return pa["value"] + luck


def pitcher_roles(pa: pd.DataFrame) -> pd.Series:
    """"SP" or "RP" per pitcher: a starter threw his team's first pitch of
    the game in at least STARTER_SHARE of his games."""
    first = pa.sort_values(["game_pk", "at_bat_number"]).groupby(["game_pk", "field_team"])["pitcher"].first()
    starts = first.value_counts()
    games = pa.groupby("pitcher")["game_pk"].nunique()
    return pd.Series(np.where(starts.reindex(games.index).fillna(0) / games >= STARTER_SHARE, "SP", "RP"), index=games.index)


def _regular(player: pd.Series, team: pd.Series, limit: int) -> pd.Series:
    """True for each row whose player is among his team's `limit` busiest."""
    counts = pd.DataFrame({"player": player, "team": team}).groupby(["team", "player"]).size().rename("n").reset_index()
    counts["rank"] = counts.groupby("team")["n"].rank(method="first", ascending=False)
    regular = set(map(tuple, counts.loc[counts["rank"] <= limit, ["team", "player"]].to_numpy()))
    return pd.Series([pair in regular for pair in zip(team, player)], index=player.index)


def fit_plate_appearances(pa: pd.DataFrame, neutral: pd.Series) -> dict:
    """Batters and pitchers fit together on the fielding-neutral value of
    every plate appearance, with the park, whether batter and pitcher are
    same-handed, and each side's level (on the roster or past it; starter
    or reliever) as fixed effects.

    Returns {"batters": frame, "pitchers": frame}, one row per player: pa
    (plate appearances or batters faced), tier, runs (see below), level
    (his tier's runs a plate appearance, against the league's average),
    talent (the fit's own estimate of him, pulled toward his tier's
    level; runs a plate appearance) and raw (his plate appearances' plain
    average value).

    `runs` is what he PRODUCED, graded for difficulty: every one of his
    plate appearances' neutral value, less the park's and the handedness
    matchup's share and less what the fit says the man across from him is
    worth, summed and set against the league's average. Nothing of his own
    is pulled toward the average - a season is what it was - but the
    pitchers a hitter faced (and the hitters a pitcher faced) are the
    fit's estimates, so one hot opponent does not make a schedule hard.
    For a pitcher the sign is flipped: runs he saved."""
    roles = pa["pitcher"].map(pitcher_roles(pa))
    bat_tier = np.where(_regular(pa["batter"], pa["bat_team"], ROSTER["bat"]), "roster", "replacement")
    on_roster = pd.Series(False, index=pa.index)
    for role in ("SP", "RP"):
        rows = roles == role
        on_roster[rows] = _regular(pa.loc[rows, "pitcher"], pa.loc[rows, "field_team"], ROSTER[role])
    pitch_tier = roles + np.where(on_roster, "", " replacement")
    y = neutral.to_numpy(dtype=float)

    def codes(values):
        c, index = pd.factorize(pd.Series(values), sort=True)
        return c, len(index), index
    park, parks, _ = codes(pa["home_team"].to_numpy())
    bt, bt_n, bt_index = codes(bat_tier)
    pt, pt_n, pt_index = codes(pitch_tier.to_numpy())
    same = (pa["stand"] == pa["p_throws"]).to_numpy().astype(int)
    b, b_n, b_index = codes(pa["batter"].to_numpy())
    p, p_n, p_index = codes(pa["pitcher"].to_numpy())
    # The parks stand in for the intercept; each other factor's first level is its baseline.
    fit = mixed_fit(y, [("park", park, parks), ("bat_tier", bt - 1, bt_n - 1), ("pitch_tier", pt - 1, pt_n - 1), ("same_hand", same - 1, 1)],
                    [("batter", b, b_n), ("pitcher", p, p_n)], iterations=60)
    bat_levels = np.concatenate([[0.0], fit["fixed"]["bat_tier"]])
    pitch_levels = np.concatenate([[0.0], fit["fixed"]["pitch_tier"]])
    setting = fit["fixed"]["park"][park] + fit["fixed"]["same_hand"][0] * same
    batter_total = bat_levels[bt] + fit["random"]["batter"][b]
    pitcher_total = pitch_levels[pt] + fit["random"]["pitcher"][p]
    out = {"variance": fit["variance"], "noise": fit["noise"], "same_hand": float(fit["fixed"]["same_hand"][0]),
           "park": pd.Series(fit["fixed"]["park"] - fit["fixed"]["park"].mean(), index=codes(pa["home_team"].to_numpy())[2])}
    for name, c, index, tier_codes, tier_index, levels, total, other, sign in (
            ("batters", b, b_index, bt, bt_index, bat_levels, batter_total, pitcher_total, 1.0),
            ("pitchers", p, p_index, pt, pt_index, pitch_levels, pitcher_total, batter_total, -1.0)):
        n = np.bincount(c, minlength=len(index))
        # A traded player's tier is his busiest stint's.
        tiers = pd.DataFrame({"player": c, "tier": tier_codes}).groupby(["player", "tier"]).size().unstack(fill_value=0).idxmax(axis=1).to_numpy()
        left = y - setting - other
        left = left - left.mean()
        center = float(total.mean())
        out[name] = pd.DataFrame({
            "player_id": pd.Series(index).astype(int), "pa": n, "tier": np.asarray(tier_index)[tiers],
            "runs": sign * np.bincount(c, weights=left, minlength=len(index)),
            "level": sign * (levels[tiers] - center),
            "talent": sign * (np.bincount(c, weights=total, minlength=len(index)) / n - center),
            "raw": np.bincount(c, weights=pa["value"].to_numpy(), minlength=len(index)) / n,
        })
    return out


# ---- positions, replacement and wins ----

FIELD_POSITIONS = ["C", "1B", "2B", "3B", "SS", "LF", "CF", "RF"]


def games_by_position(pa: pd.DataFrame) -> pd.DataFrame:
    """Games each player appeared in at each fielding position, and as a
    designated hitter (he batted and never took the field): one row per
    player, a column per position."""
    fielded = pd.concat([
        pa[["game_pk", f"fielder_{k}"]].dropna().drop_duplicates().rename(columns={f"fielder_{k}": "player_id"}).assign(position=POSITIONS[k])
        for k in range(2, 10)], ignore_index=True)
    fielded["player_id"] = fielded["player_id"].astype(int)
    in_field = set(map(tuple, fielded[["game_pk", "player_id"]].to_numpy()))
    pitched = set(map(tuple, pa[["game_pk", "pitcher"]].drop_duplicates().to_numpy()))
    batted = pa[["game_pk", "batter"]].drop_duplicates().rename(columns={"batter": "player_id"})
    only_hit = [pair not in in_field and pair not in pitched for pair in map(tuple, batted.to_numpy())]
    every = pd.concat([fielded, batted[only_hit].assign(position="DH")], ignore_index=True)
    return every.groupby(["player_id", "position"])["game_pk"].nunique().unstack(fill_value=0).reindex(columns=FIELD_POSITIONS + ["DH"], fill_value=0)


# Runs a plate appearance that the players past a team's roster (ROSTER)
# were below the average of their kind - hitters below the average hitter,
# starters below the average starter, relievers below the average
# reliever - over 2025 and 2026 (scripts/research_mlb_war.py). Constants,
# as in nfl_player_value: one season's own gap moves a few runs a player.
REPLACEMENT = {"bat": 0.038, "SP": 0.014, "RP": 0.025}
# Runs per win at today's scoring (4.5 runs a team a game), by the usual
# rule: 9 x runs an inning x 1.5 + 3. Our own 60 team-seasons (wins on run
# differential) say 11.3, give or take 0.6 - too few seasons to overrule
# what decades of them settled on.
RUNS_PER_WIN = 9.7


def player_table(pa: pd.DataFrame, fit: dict, credits: pd.DataFrame) -> pd.DataFrame:
    """One row per player: position (where he played most; SP or RP for a
    pitcher), games, pa, batting (runs against the average hitter at his
    position), fielding (runs against the average fielder at each
    position he played), bf, pitching (runs saved against the average
    starter or reliever), raa (their sum), replacement (the runs between
    average and a replacement player over his playing time), rar and war.
    """
    games = games_by_position(pa)
    batters = fit["batters"].set_index("player_id")
    pitchers = fit["pitchers"].set_index("player_id")
    out = pd.DataFrame(index=sorted(set(batters.index) | set(pitchers.index) | set(credits["player_id"])))
    out.index.name = "player_id"
    out["pa"] = batters["pa"].reindex(out.index).fillna(0).astype(int)
    out["bf"] = pitchers["pa"].reindex(out.index).fillna(0).astype(int)
    role = pitchers["tier"].str.split(" ").str[0].reindex(out.index)
    most = games.reindex(out.index).fillna(0)
    field_position = most.idxmax(axis=1).where(most.sum(axis=1) > 0)
    is_pitcher = out["bf"] > out["pa"]
    out["position"] = np.where(is_pitcher, role, field_position)
    out["games"] = pd.concat([pa[["game_pk", "batter"]].rename(columns={"batter": "player_id"}),
                              pa[["game_pk", "pitcher"]].rename(columns={"pitcher": "player_id"}),
                              credits[["game_pk", "player_id"]]]).drop_duplicates().groupby("player_id").size().reindex(out.index).fillna(0).astype(int)
    # Batting: against the average hitter at his position.
    bat = batters["runs"].reindex(out.index).fillna(0.0)
    hitters = ~is_pitcher & (out["pa"] > 0)
    by_position = pd.DataFrame({"runs": bat[hitters], "pa": out.loc[hitters, "pa"], "position": out.loc[hitters, "position"]}).groupby("position")[["runs", "pa"]].sum()
    standard = out["position"].map(by_position["runs"] / by_position["pa"]).fillna(0.0)
    out["batting"] = bat - standard * out["pa"]
    out["batting_league"] = bat
    # Fielding: against the average at each position he played.
    by_field = credits.groupby("position")[["runs", "plays"]].sum()
    mine = credits.groupby(["player_id", "position"])[["runs", "plays", "outs_above"]].sum().reset_index()
    mine["above"] = mine["runs"] - mine["plays"] * mine["position"].map(by_field["runs"] / by_field["plays"])
    out["fielding"] = mine.groupby("player_id")["above"].sum().reindex(out.index).fillna(0.0)
    out["chances"] = mine.groupby("player_id")["plays"].sum().reindex(out.index).fillna(0.0)
    # Pitching: against the average starter or reliever.
    saved = pitchers["runs"].reindex(out.index).fillna(0.0)
    by_role = pd.DataFrame({"runs": pitchers["runs"], "bf": pitchers["pa"], "role": pitchers["tier"].str.split(" ").str[0]}).groupby("role")[["runs", "bf"]].sum()
    out["pitching"] = saved - role.map(by_role["runs"] / by_role["bf"]).fillna(0.0) * out["bf"]
    out["raa"] = out["batting"] + out["fielding"] + out["pitching"]
    out["replacement"] = REPLACEMENT["bat"] * out["pa"] + role.map(REPLACEMENT).fillna(0.0) * out["bf"]
    out["rar"] = out["raa"] + out["replacement"]
    out["war"] = out["rar"] / RUNS_PER_WIN
    out["age"] = pd.concat([pa.groupby("batter")["age_bat"].max(), pa.groupby("pitcher")["age_pit"].max()]).groupby(level=0).max().reindex(out.index)
    team = pd.concat([pa[["batter", "bat_team", "game_date"]].set_axis(["player_id", "team", "date"], axis=1),
                      pa[["pitcher", "field_team", "game_date"]].set_axis(["player_id", "team", "date"], axis=1)])
    out["team"] = team.sort_values("date").groupby("player_id")["team"].last().reindex(out.index)
    return out.reset_index()


def team_records(pitches: pd.DataFrame) -> pd.DataFrame:
    """Wins, losses, runs scored and allowed per team from the regular
    season's final scores."""
    p = pitches[pitches["game_type"] == "R"]
    final = p.groupby("game_pk").agg(home=("home_team", "first"), away=("away_team", "first"), home_runs=("post_home_score", "max"), away_runs=("post_away_score", "max"))
    rows = []
    for side, other in (("home", "away"), ("away", "home")):
        rows.append(pd.DataFrame({"team": final[side], "scored": final[f"{side}_runs"], "allowed": final[f"{other}_runs"]}))
    t = pd.concat(rows)
    t["win"] = (t["scored"] > t["allowed"]).astype(int)
    return t.groupby("team").agg(games=("win", "size"), wins=("win", "sum"), scored=("scored", "sum"), allowed=("allowed", "sum"))


# ---- names, leagues and the table ----

DIVISIONS = {
    "AL East": ["BAL", "BOS", "NYY", "TB", "TOR"], "AL Central": ["CWS", "CLE", "DET", "KC", "MIN"],
    "AL West": ["ATH", "HOU", "LAA", "SEA", "TEX"], "NL East": ["ATL", "MIA", "NYM", "PHI", "WSH"],
    "NL Central": ["CHC", "CIN", "MIL", "PIT", "STL"], "NL West": ["AZ", "COL", "LAD", "SD", "SF"],
}
TEAM_DIVISION = {team: division for division, teams in DIVISIONS.items() for team in teams}
# Too little playing time to be worth a row.
MIN_PLAYING_TIME = 30


# Lowercase words that are part of a name, not the start of what he did.
NAME_PARTICLES = {"de", "la", "los", "las", "del", "van", "der", "den", "von", "da", "di", "le", "st."}


def player_names(pitches: pd.DataFrame) -> pd.Series:
    """A name for every pitcher and batter, from the pitches themselves
    (no lookup to download): a pitcher's is in player_name ("Sale,
    Chris"); a batter's is what the descriptions of his plate appearances
    start with ("Willy Adames singles ...", "Willy Adames strikes out
    ...") - the longest opening most of them share, cut at the first word
    that says what he did."""
    names = {}
    pitchers = pitches.dropna(subset=["player_name"]).groupby("pitcher")["player_name"].first()
    for pitcher, name in pitchers.items():
        last, _, first = str(name).partition(", ")
        names[int(pitcher)] = f"{first} {last}".strip()
    ended = pitches[pitches["events"].notna() & pitches["des"].notna()]
    for batter, descriptions in ended.groupby("batter")["des"]:
        if int(batter) in names:
            continue
        words = [d.split() for d in descriptions.head(300)]
        name = []
        for n in range(1, 7):
            openings = pd.Series([" ".join(w[:n]) for w in words if len(w) >= n])
            if openings.empty:
                break
            top = openings.value_counts()
            if top.iloc[0] < 0.6 * len(words):
                break
            candidate = top.index[0].split()
            if candidate[-1].islower() and candidate[-1] not in NAME_PARTICLES:
                break
            name = candidate
        if name:
            names[int(batter)] = " ".join(name)
    return pd.Series(names)


def build(season: int, raw_dir: str = "data/raw") -> dict:
    """The table's data for one season: {season, through, runs_per_win,
    players: [...]}. Every entry: name, pos, kind ("hitter" or
    "pitcher"), team, div ("AL East"; the league is its first word), age,
    games, pa, bf, war, raa (runs above the average at his position), rar
    (above replacement), and the parts of raa he has: batting, fielding,
    pitching."""
    from mlb_metrics import data

    pitches = data.load_persisted_statcast(raw_dir, season, PITCH_COLUMNS)
    if pitches is None or pitches.empty or not (pitches["game_type"] == "R").any():
        return {"season": int(season), "through": None, "runs_per_win": RUNS_PER_WIN, "players": []}
    pa = plate_appearances(pitches)
    balls = fielding_chances(pa)
    credits = fielding_credits(balls)
    table = player_table(pa, fit_plate_appearances(pa, neutral_value(pa, balls)), credits)
    names = player_names(pitches[pitches["game_type"] == "R"])
    players = []
    for row in table.itertuples():
        if row.pa + row.bf < MIN_PLAYING_TIME or not isinstance(row.position, str):
            continue
        pitcher = row.position in ("SP", "RP")
        division = TEAM_DIVISION.get(row.team, "")
        entry = {"name": names.get(row.player_id, str(row.player_id)), "pos": row.position, "kind": "pitcher" if pitcher else "hitter",
                 "team": row.team, "div": division, "age": None if pd.isna(row.age) else int(row.age), "games": int(row.games),
                 "war": round(row.war, 2), "raa": round(row.raa, 1), "rar": round(row.rar, 1)}
        if row.pa:
            entry.update({"pa": int(row.pa), "batting": round(row.batting, 1)})
        if row.chances >= 1:
            entry["fielding"] = round(row.fielding, 1)
        if row.bf:
            entry.update({"bf": int(row.bf), "pitching": round(row.pitching, 1)})
        players.append(entry)
    players.sort(key=lambda e: -e["war"])
    through = pd.to_datetime(pa["game_date"]).max()
    return {"season": int(season), "through": through.strftime("%Y-%m-%d"), "runs_per_win": RUNS_PER_WIN, "players": players}

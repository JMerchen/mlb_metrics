"""The NCAA tournament bracket and its odds.

The field is the real one once the schedule carries it (official_field:
the round-of-64 and First Four games, with their regions and seeds), and
until then a projection from the ratings (projected_field: "if the season
ended today" - each conference's best-rated team as its automatic bid,
the best-rated of the rest at large, seeded straight down the ratings).
The committee picks and seeds on results, not on a rating, so the
projection is a guide to who is good enough, not a forecast of the real
bracket.

simulate() plays the bracket many times from cbb_ratings' game odds
(every game on a neutral floor) and counts how often each team reaches
each round. Games already played are taken as they finished.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from mlb_metrics import cbb_ratings, config

# Top to bottom of a region's bracket: 1 plays 16, the winner plays 8/9...
SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
REGION_NAMES = ["East", "West", "South", "Midwest"]
FIELD_COLUMNS = ["team_id", "seed", "region", "play_in", "bid"]
# Chance of still being alive at the start of each of these.
ROUND_COLUMNS = ["round_of_64", "round_of_32", "sweet_16", "elite_8", "final_four", "final", "champion"]
INDEPENDENTS_CONFERENCE_ID = 43


def projected_field(ratings: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    """A 68-team field from the ratings (FIELD_COLUMNS, best team first):
    the best-rated team of each conference (bid "auto"), then the
    best-rated of everyone else ("at-large"). The four lowest at-large
    teams and the four lowest automatic bids meet in the First Four
    (play_in) for two seeds each. Seeds go straight down the ratings, four
    to a line, and regions snake (the top 1 seed's region gets the worst 2
    seed), so region 1 holds the best team; the Final Four pairs region 1
    with region 4 and region 2 with region 3."""
    rated = ratings.merge(teams[["team_id", "conference_id"]], on="team_id", how="inner")
    rated = rated[rated["games"] > 0] if (rated["games"] > 0).sum() >= 68 else rated
    rated = rated.sort_values("net", ascending=False).reset_index(drop=True)
    in_conference = rated[rated["conference_id"] != INDEPENDENTS_CONFERENCE_ID]
    auto = in_conference.drop_duplicates("conference_id")["team_id"].tolist()
    at_large = [t for t in rated["team_id"] if t not in set(auto)][:68 - len(auto)]
    field = rated[rated["team_id"].isin(auto + at_large)].copy()
    field["bid"] = np.where(field["team_id"].isin(auto), "auto", "at-large")
    play_in = set(field[field["bid"] == "at-large"].tail(4)["team_id"]) | set(field[field["bid"] == "auto"].tail(4)["team_id"])
    field["play_in"] = field["team_id"].isin(play_in)

    # 64 bracket entries: 60 teams and four First Four pairs (two of
    # at-large teams, two of automatic bids, neighbors in the ratings),
    # each pair placed where its average rating puts it.
    entries = [[t] for t in field.loc[~field["play_in"], "team_id"]]
    for bid in ("at-large", "auto"):
        four = field[field["play_in"] & (field["bid"] == bid)]["team_id"].tolist()
        entries += [[four[0], four[1]], [four[2], four[3]]]
    net = field.set_index("team_id")["net"]
    entries.sort(key=lambda entry: -sum(net[t] for t in entry) / len(entry))
    rows = []
    for position, entry in enumerate(entries):
        line, spot = divmod(position, 4)
        region = REGION_NAMES[spot if line % 2 == 0 else 3 - spot]
        rows += [{"team_id": t, "seed": line + 1, "region": region} for t in entry]
    placed = pd.DataFrame(rows).merge(field[["team_id", "play_in", "bid", "net"]], on="team_id")
    return placed.sort_values("net", ascending=False)[FIELD_COLUMNS].reset_index(drop=True)


def official_field(schedules: pd.DataFrame) -> pd.DataFrame | None:
    """The real field from a season's schedule (FIELD_COLUMNS; bid is
    blank), or None until the round of 64 is posted with seeds for all
    four regions. A team in a First Four game is a play_in for that game's
    seed and region."""
    if "notes_headline" not in schedules.columns or "home_current_rank" not in schedules.columns:
        return None
    rows = []
    for _, game in schedules.iterrows():
        round_name, region = cbb_ratings.tournament_round(game["notes_headline"])
        if round_name not in ("First Four", "Round of 64") or region is None:
            continue
        if "season_type" in schedules.columns and game["season_type"] != 3:
            continue
        seeds = {side: pd.to_numeric(game[f"{side}_current_rank"], errors="coerce") for side in ("home", "away")}
        for side, other in (("home", "away"), ("away", "home")):
            team = int(game[f"{side}_id"])
            if team < 0:
                continue   # a First Four winner still to come ("TBD")
            seed = seeds[side]
            if not 1 <= (seed if pd.notna(seed) else 0) <= 16:
                seed = 17 - seeds[other] if round_name == "Round of 64" and pd.notna(seeds[other]) else float("nan")
            rows.append({"team_id": team, "seed": seed, "region": region, "play_in": round_name == "First Four"})
    if not rows:
        return None
    field = pd.DataFrame(rows).dropna(subset=["seed"])
    # A First Four team that won shows up again in the round of 64.
    field = field.sort_values("play_in", ascending=False).drop_duplicates("team_id")
    field["seed"] = field["seed"].astype(int)
    field["bid"] = ""
    slots = field.groupby(["region", "seed"]).size()
    complete = len(slots) == 64 and set(field["region"]) == set(REGION_NAMES) \
        and (slots[field.set_index(["region", "seed"]).index[field["play_in"].to_numpy()]] == 2).all() \
        and (slots <= 2).all() and len(field) == 64 + int(field["play_in"].sum()) // 2
    return field[FIELD_COLUMNS].reset_index(drop=True) if complete else None


def final_four_pairs(season: int, field_kind: str, schedules: pd.DataFrame | None = None,
                     field: pd.DataFrame | None = None) -> tuple[list, bool]:
    """Which regions meet in the national semifinals, as ([[a, b], [c, d]],
    known). A projected bracket pairs its first region with its fourth.
    A real one is read from config.CBB_FINAL_FOUR_PAIRS, or from the Final
    Four games once they're on the schedule; before that it is assumed
    (known False) to be East-West and South-Midwest."""
    if field_kind == "projected":
        return [[REGION_NAMES[0], REGION_NAMES[3]], [REGION_NAMES[1], REGION_NAMES[2]]], True
    if season in config.CBB_FINAL_FOUR_PAIRS:
        return [list(pair) for pair in config.CBB_FINAL_FOUR_PAIRS[season]], True
    if schedules is not None and field is not None and "notes_headline" in schedules.columns:
        region_of = field.set_index("team_id")["region"]
        pairs = []
        for _, game in schedules.iterrows():
            if cbb_ratings.tournament_round(game["notes_headline"])[0] != "Final Four":
                continue
            regions = [region_of.get(int(game["home_id"])), region_of.get(int(game["away_id"]))]
            if all(regions) and regions[0] != regions[1]:
                pairs.append(regions)
        if len(pairs) == 2 and len({r for pair in pairs for r in pair}) == 4:
            return pairs, True
        if len(pairs) == 1:
            rest = [r for r in REGION_NAMES if r not in pairs[0]]
            return [pairs[0], rest], True
    return [["East", "West"], ["South", "Midwest"]], False


def played_results(games: pd.DataFrame) -> dict:
    """{(team, team): winner} for the tournament games already played
    (prepare_games rows), keyed both ways round."""
    out = {}
    for _, g in games[games["tournament"]].iterrows():
        winner = int(g["home_id"]) if g["home_score"] > g["away_score"] else int(g["away_id"])
        out[(int(g["home_id"]), int(g["away_id"]))] = winner
        out[(int(g["away_id"]), int(g["home_id"]))] = winner
    return out


def win_matrix(team_ids: list, ratings: pd.DataFrame, league: dict, results: dict | None = None,
               sigma: float | None = None) -> np.ndarray:
    """P[i, j] = chance team i beats team j on a neutral floor; a pair that
    has already met in this tournament (`results`) is 1 or 0."""
    sigma = config.CBB_MARGIN_SIGMA if sigma is None else sigma
    r = ratings.set_index("team_id").reindex(team_ids)
    net = r["net"].fillna(r["net"].min() if r["net"].notna().any() else 0.0).to_numpy(dtype=float)
    tempo = r["tempo"].fillna(league["tempo"]).to_numpy(dtype=float)
    possessions = tempo[:, None] + tempo[None, :] - league["tempo"]
    P = norm.cdf((net[:, None] - net[None, :]) / 100 * possessions / sigma)
    for (a, b), winner in (results or {}).items():
        if a in team_ids and b in team_ids:
            P[team_ids.index(a), team_ids.index(b)] = 1.0 if winner == a else 0.0
    return P


def simulate(field: pd.DataFrame, ratings: pd.DataFrame, league: dict, pairs: list,
             results: dict | None = None, sims: int | None = None, seed: int = 0) -> pd.DataFrame:
    """Plays the bracket `sims` times. Returns one row per team: team_id
    and, for each of ROUND_COLUMNS, the share of runs in which the team was
    still in at that point (round_of_64 is 1 for everyone but the First
    Four teams; champion is winning it all)."""
    sims = config.CBB_BRACKET_SIMS if sims is None else sims
    rng = np.random.default_rng(seed)
    team_ids = [int(t) for t in field["team_id"]]
    position = {t: i for i, t in enumerate(team_ids)}
    P = win_matrix(team_ids, ratings, league, results)

    def play(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.where(rng.random(a.shape) < P[a, b], a, b)

    # The 64 starting spots, region by region in final-four order, each
    # region top to bottom; a First Four pair plays for its spot.
    region_order = [pairs[0][0], pairs[0][1], pairs[1][0], pairs[1][1]]
    slots = np.zeros((sims, 64), dtype=int)
    by_slot = field.groupby(["region", "seed"])["team_id"].apply(list)
    for r, region in enumerate(region_order):
        for s, seed_number in enumerate(SEED_ORDER):
            entry = [position[int(t)] for t in by_slot[(region, seed_number)]]
            column = np.full(sims, entry[0])
            slots[:, 16 * r + s] = column if len(entry) == 1 else play(column, np.full(sims, entry[1]))
    reached = np.zeros((len(team_ids), len(ROUND_COLUMNS)))
    alive = slots
    for round_index in range(len(ROUND_COLUMNS)):
        reached[:, round_index] = np.bincount(alive.ravel(), minlength=len(team_ids)) / sims
        if alive.shape[1] > 1:
            alive = play(alive[:, 0::2], alive[:, 1::2])
    out = pd.DataFrame(reached, columns=ROUND_COLUMNS)
    out.insert(0, "team_id", team_ids)
    return out

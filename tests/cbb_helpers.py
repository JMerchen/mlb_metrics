"""Small made-up college basketball seasons for the cbb tests."""

import itertools

import pandas as pd

SEED_ORDER = [1, 16, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
REGIONS = ["East", "West", "South", "Midwest"]


def game(game_id, date, home, away, home_score, away_score, neutral=False, done=True, note=None, season_type=2,
         home_seed=99, away_seed=99, season=2027, home_conf=1, away_conf=1):
    """One schedule row in hoopR's columns. A conference of None makes that
    side a non-Division I team."""
    return {
        "game_id": game_id, "season": season, "season_type": season_type, "game_date": date, "neutral_site": neutral,
        "status_type_completed": done, "notes_headline": note,
        "home_id": home, "home_location": f"Team {home}", "home_short_display_name": f"T{home}",
        "home_abbreviation": f"T{home}", "home_conference_id": home_conf, "home_color": "112233", "home_logo": "",
        "home_score": home_score, "home_current_rank": home_seed,
        "away_id": away, "away_location": f"Team {away}", "away_short_display_name": f"T{away}",
        "away_abbreviation": f"T{away}", "away_conference_id": away_conf, "away_color": "445566", "away_logo": "",
        "away_score": away_score, "away_current_rank": away_seed,
    }


def round_robin(strength: dict, home_edge: float = 2.0, possessions: int = 70, conference=None, season: int = 2027) -> pd.DataFrame:
    """Every team hosts every other once. `strength` is {team: (offense,
    defense)} in points per 100 possessions around 100; the scores follow
    the ratings exactly (no luck), on `possessions` possessions."""
    rows = []
    for n, (home, away) in enumerate(itertools.permutations(strength, 2)):
        home_eff = 100 + strength[home][0] - strength[away][1] + home_edge
        away_eff = 100 + strength[away][0] - strength[home][1] - home_edge
        conf = conference or {}
        rows.append(game(1000 + n, f"{season - 1}-12-{1 + n % 28:02d}", home, away, home_eff * possessions / 100,
                         away_eff * possessions / 100, season=season,
                         home_conf=conf.get(home, 1), away_conf=conf.get(away, 1)))
    return pd.DataFrame(rows)


def box_for(schedules: pd.DataFrame, possessions: int = 70) -> pd.DataFrame:
    """Team box scores that add up to `possessions` a side for every game."""
    rows = []
    for _, g in schedules.iterrows():
        for side in ("home", "away"):
            rows.append({"game_id": g["game_id"], "team_id": g[f"{side}_id"], "field_goals_attempted": possessions - 20,
                         "offensive_rebounds": 9, "total_turnovers": 10, "free_throws_attempted": 40, "turnovers": 10})
    return pd.DataFrame(rows)


def bracket_schedule(first_four_done: bool = False, season: int = 2027) -> tuple[pd.DataFrame, dict]:
    """A posted bracket: four regions of 16 seeds, team ids 100*region +
    seed (region 1-4), with a First Four game for each region's 16 seed
    (the second team is 100*region + 17). Before the First Four is played
    the round-of-64 game lists "TBD" (id -1) for that spot. Returns
    (schedule, {team: (region, seed)})."""
    rows, where, n = [], {}, 0
    for r, region in enumerate(REGIONS, start=1):
        for seed in range(1, 17):
            where[100 * r + seed] = (region, seed)
        where[100 * r + 17] = (region, 16)
        n += 1
        rows.append(game(n, "2027-03-16", 100 * r + 16, 100 * r + 17, 70 if first_four_done else 0, 60 if first_four_done else 0,
                         neutral=True, done=first_four_done, season_type=3, home_seed=16, away_seed=16, season=season,
                         note=f"NCAA Men's Basketball Championship - {region} Region - First Four"))
        for top, bottom in zip(SEED_ORDER[0::2], SEED_ORDER[1::2]):
            n += 1
            low = 100 * r + bottom
            if bottom == 16 and not first_four_done:
                low = -1
            rows.append(game(n, "2027-03-18", 100 * r + top, low, 0, 0, neutral=True, done=False, season_type=3,
                             home_seed=top, away_seed=bottom if low > 0 else 99, season=season,
                             away_conf=1 if low > 0 else None,
                             note=f"NCAA Men's Basketball Championship - {region} Region - 1st Round"))
    return pd.DataFrame(rows), where

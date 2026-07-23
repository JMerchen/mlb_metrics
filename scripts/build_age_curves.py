"""Build the Age Curves page's data (docs/data/age_curve_projections.csv,
docs/data/age_curve_comparables.csv, docs/data/age_curve_league.csv) from
this project's own persisted Statcast (current-season traditional stats)
and persisted Lahman data (historical comparables) - see age_curve.py's
module docstring for the method.

Writes one row per (player, metric) - a separate curve/projection for each
of config.AGE_CURVE_HITTER_METRICS (AVG/OBP/SLG/OPS) and
config.AGE_CURVE_PITCHER_METRICS (K9/BB9/HR9/FIP), not one blended number -
see age_curve.py's module docstring for why.

Run this OCCASIONALLY (e.g. weekly), not as part of daily_update.yml -
historical comparables don't move day to day, and current-season stats
only need to move enough to matter every so often, not every single day.
Needs data/raw/lahman/{people,batting,pitching}.parquet - run
scripts/fetch_lahman.py first if that's missing.

Usage:
    python scripts/build_age_curves.py
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import age_curve, config, data, lahman_data, pipeline, traditional_stats

CURRENT_HITTER_COLUMNS = ["key_mlbam", "name_first", "name_last", "age", "AB", "AVG", "OBP", "SLG", "OPS"]
CURRENT_PITCHER_COLUMNS = ["key_mlbam", "name_first", "name_last", "age", "IP", "K9", "BB9", "HR9", "FIP"]


def build_current_player_pool(raw_dir: str, season: int, min_at_bats: int = config.AGE_CURVE_MIN_AB) -> pd.DataFrame:
    """[key_mlbam, name_first, name_last, age, AB, AVG, OBP, SLG, OPS] for
    every qualified current-season hitter - joins this project's own
    persisted Statcast (traditional_stats) to Lahman's People table via
    the chadwick_register crosswalk (lahman_data.build_crosswalk), so
    their age can be computed the same way as any historical Lahman
    season. A current player with no resolvable Lahman/bbref match (a
    brand-new call-up, an international signee not yet in Lahman) is
    excluded, not defaulted - see build_crosswalk's docstring."""
    persisted = data.load_persisted_statcast(raw_dir, season)
    if persisted is None:
        return pd.DataFrame(columns=CURRENT_HITTER_COLUMNS)

    dt = data.completed_events(persisted, ["game_date", "batter", "events"])
    current_stats = traditional_stats.compute_traditional_batting_stats(dt, min_at_bats=min_at_bats)

    names = data.get_name_register()
    chadwick_register = names[["key_mlbam", "key_bbref", "name_first", "name_last"]]
    people = lahman_data.load_persisted_lahman_table(raw_dir, "people")
    if people is None:
        return pd.DataFrame(columns=CURRENT_HITTER_COLUMNS)

    crosswalk = lahman_data.build_crosswalk(chadwick_register, people)
    current_stats = current_stats.merge(crosswalk, on="key_mlbam", how="inner")
    current_stats["yearID"] = season

    aged = lahman_data.attach_age(current_stats, people)
    aged = aged.merge(chadwick_register[["key_mlbam", "name_first", "name_last"]], on="key_mlbam", how="left")

    return aged[CURRENT_HITTER_COLUMNS]


def build_current_pitcher_pool(raw_dir: str, season: int, min_ip: float = config.AGE_CURVE_MIN_IP) -> pd.DataFrame:
    """[key_mlbam, name_first, name_last, age, IP, K9, BB9, HR9, FIP] for
    every qualified current-season pitcher - the pitcher-side counterpart
    to build_current_player_pool, same crosswalk/age-attachment approach."""
    persisted = data.load_persisted_statcast(raw_dir, season)
    if persisted is None:
        return pd.DataFrame(columns=CURRENT_PITCHER_COLUMNS)

    pdf = pipeline.build_pitcher_events(persisted)
    current_stats = traditional_stats.compute_traditional_pitching_stats(pdf, min_ip=min_ip)

    names = data.get_name_register()
    chadwick_register = names[["key_mlbam", "key_bbref", "name_first", "name_last"]]
    people = lahman_data.load_persisted_lahman_table(raw_dir, "people")
    if people is None:
        return pd.DataFrame(columns=CURRENT_PITCHER_COLUMNS)

    crosswalk = lahman_data.build_crosswalk(chadwick_register, people)
    current_stats = current_stats.merge(crosswalk, on="key_mlbam", how="inner")
    current_stats["yearID"] = season

    aged = lahman_data.attach_age(current_stats, people)
    aged = aged.merge(chadwick_register[["key_mlbam", "name_first", "name_last"]], on="key_mlbam", how="left")

    return aged[CURRENT_PITCHER_COLUMNS]


def describe_comparables(
    key_mlbam: int, metric: str, comparables: pd.DataFrame, historical_seasons: pd.DataFrame, people: pd.DataFrame
) -> pd.DataFrame:
    """[key_mlbam, metric, name, yearID, age, value, next_value] for one
    current player's K comparable historical seasons on `metric` - joins
    in Lahman People's name and each comparable's own actual next-season
    value on that same metric (null if they had none - see
    age_curve.project_next_season), for the Age Curves page's "who are
    the comparables" list."""
    names = people[["playerID", "nameFirst", "nameLast"]].drop_duplicates(subset="playerID")
    described = comparables.merge(names, on="playerID", how="left")
    described["name"] = (described["nameFirst"].fillna("") + " " + described["nameLast"].fillna("")).str.strip()

    next_lookup = historical_seasons.rename(columns={"yearID": "next_year", metric: "next_value"})[
        ["playerID", "next_year", "next_value"]
    ]
    described = described.assign(next_year=described["yearID"] + 1).merge(
        next_lookup, on=["playerID", "next_year"], how="left"
    )

    described["key_mlbam"] = key_mlbam
    described["metric"] = metric
    described["value"] = described[metric]
    return described[["key_mlbam", "metric", "name", "yearID", "age", "value", "next_value"]]


def build_projections_for_group(
    current_players: pd.DataFrame, historical_seasons: pd.DataFrame, metrics: list[str], people: pd.DataFrame
) -> tuple[list[dict], list[pd.DataFrame], list[pd.DataFrame]]:
    """Runs find_comparables/project_next_season/league_age_curve for every
    (player, metric) pair in one current-player pool (hitters or pitchers)
    against its own matching historical-seasons table - shared by both
    groups in main() rather than duplicating the loop per position type."""
    rows = []
    comparable_frames = []
    for _, player in current_players.iterrows():
        for metric in metrics:
            comparables = age_curve.find_comparables(player["age"], player[metric], historical_seasons, metric=metric)
            projection = age_curve.project_next_season(comparables, historical_seasons, metric=metric)
            rows.append(
                {
                    "key_mlbam": player["key_mlbam"],
                    "name": f"{player['name_first']} {player['name_last']}".strip(),
                    "metric": metric,
                    "age": player["age"],
                    "value": player[metric],
                    **projection,
                }
            )
            comparable_frames.append(describe_comparables(player["key_mlbam"], metric, comparables, historical_seasons, people))

    league_frames = [
        age_curve.league_age_curve(historical_seasons, metric=metric).assign(metric=metric) for metric in metrics
    ]
    return rows, comparable_frames, league_frames


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--output-dir", default="docs/data")
    parser.add_argument("--season", type=int, default=config.SEASON_START.year)
    args = parser.parse_args()

    people = lahman_data.load_persisted_lahman_table(args.raw_dir, "people")
    batting = lahman_data.load_persisted_lahman_table(args.raw_dir, "batting")
    pitching = lahman_data.load_persisted_lahman_table(args.raw_dir, "pitching")
    if people is None or batting is None or pitching is None:
        print(f"No persisted Lahman data in {args.raw_dir}/lahman/ - run scripts/fetch_lahman.py first.")
        return

    historical_hitter_seasons = age_curve.build_historical_hitter_seasons(batting, people)
    historical_pitcher_seasons = age_curve.build_historical_pitcher_seasons(pitching, people)
    current_hitters = build_current_player_pool(args.raw_dir, args.season)
    current_pitchers = build_current_pitcher_pool(args.raw_dir, args.season)
    print(
        f"{len(historical_hitter_seasons)} qualified historical hitter-seasons, "
        f"{len(historical_pitcher_seasons)} qualified historical pitcher-seasons, "
        f"{len(current_hitters)} qualified current hitters, {len(current_pitchers)} qualified current pitchers."
    )

    hitter_rows, hitter_comparables, hitter_league = build_projections_for_group(
        current_hitters, historical_hitter_seasons, config.AGE_CURVE_HITTER_METRICS, people
    )
    pitcher_rows, pitcher_comparables, pitcher_league = build_projections_for_group(
        current_pitchers, historical_pitcher_seasons, config.AGE_CURVE_PITCHER_METRICS, people
    )
    rows = hitter_rows + pitcher_rows
    comparable_frames = hitter_comparables + pitcher_comparables
    league_frames = hitter_league + pitcher_league

    projections = pd.DataFrame(
        rows,
        columns=[
            "key_mlbam", "name", "metric", "age", "value", "n_comparables", "n_with_next_season",
            "projected_value_mean", "projected_value_p25", "projected_value_p75",
        ],
    )
    comparables_export = (
        pd.concat(comparable_frames, ignore_index=True)
        if comparable_frames
        else pd.DataFrame(columns=["key_mlbam", "metric", "name", "yearID", "age", "value", "next_value"])
    )
    league_curve = pd.concat(league_frames, ignore_index=True)[["metric", "age", "value", "n_seasons"]]

    os.makedirs(args.output_dir, exist_ok=True)
    projections.to_csv(os.path.join(args.output_dir, "age_curve_projections.csv"), index=False)
    comparables_export.to_csv(os.path.join(args.output_dir, "age_curve_comparables.csv"), index=False)
    league_curve.to_csv(os.path.join(args.output_dir, "age_curve_league.csv"), index=False)
    n_metrics = len(config.AGE_CURVE_HITTER_METRICS) + len(config.AGE_CURVE_PITCHER_METRICS)
    print(
        f"Wrote age_curve_projections.csv ({len(projections)} rows, {len(current_hitters)} hitters x "
        f"{len(config.AGE_CURVE_HITTER_METRICS)} metrics + {len(current_pitchers)} pitchers x "
        f"{len(config.AGE_CURVE_PITCHER_METRICS)} metrics = {n_metrics} metric-groups), "
        f"age_curve_comparables.csv ({len(comparables_export)} rows), and age_curve_league.csv ({len(league_curve)} rows)."
    )


if __name__ == "__main__":
    main()

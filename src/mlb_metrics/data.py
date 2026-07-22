"""Data acquisition and persistence.

Historically the pipeline pulled the full season from Statcast fresh on every
run and discarded it after computing that day's metrics - nothing was ever
saved, so there was no way to re-run the pipeline against a past date or
build a backtest. `persist_raw_statcast`/`load_persisted_statcast` fix that by
appending each day's pull to a per-season parquet file that gets committed
alongside the output CSVs.
"""

import os

import pandas as pd

from mlb_metrics import config

# Columns that uniquely identify a single pitch, used to dedupe when merging
# a fresh Statcast pull into the persisted raw dataset.
PITCH_KEY_COLUMNS = ["game_pk", "at_bat_number", "pitch_number"]


def fetch_statcast_range(start_dt, end_dt) -> pd.DataFrame:
    """Pull pitch-by-pitch Statcast data for [start_dt, end_dt] and normalize game_date to datetime."""
    from pybaseball import statcast

    df = statcast(start_dt=str(start_dt), end_dt=str(end_dt))
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def raw_statcast_path(raw_dir: str, season: int) -> str:
    return os.path.join(raw_dir, f"statcast_{season}.parquet")


def load_persisted_statcast(raw_dir: str, season: int) -> pd.DataFrame | None:
    path = raw_statcast_path(raw_dir, season)
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def persist_raw_statcast(df: pd.DataFrame, raw_dir: str, season: int) -> pd.DataFrame:
    """Merge `df` into the persisted raw dataset for `season`, dedupe by pitch, and save.

    Returns the full merged dataset (existing history + new rows).
    """
    os.makedirs(raw_dir, exist_ok=True)
    path = raw_statcast_path(raw_dir, season)

    existing = load_persisted_statcast(raw_dir, season)
    combined = pd.concat([existing, df], ignore_index=True) if existing is not None else df.copy()

    key_columns = [c for c in PITCH_KEY_COLUMNS if c in combined.columns]
    if key_columns:
        combined = combined.drop_duplicates(subset=key_columns, keep="last")
    combined = combined.sort_values("game_date").reset_index(drop=True)

    combined.to_parquet(path, index=False)
    return combined


_name_register_cache: pd.DataFrame | None = None


def get_name_register() -> pd.DataFrame:
    """Cached wrapper around chadwick_register(); the original script called this
    four separate times per run for the same static player-ID lookup table."""
    global _name_register_cache
    if _name_register_cache is None:
        from pybaseball import chadwick_register

        _name_register_cache = chadwick_register()
    return _name_register_cache


def assign_game_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign a per-row `game_id`: a dense integer, ordered by
    (game_date, game_pk), derived directly from Statcast's own `game_pk`
    (MLB's real, already-unique game identifier - the same one
    game_picks_backtest.py uses directly). Sequential integers rather than
    raw game_pk values are kept because game_pk isn't itself
    chronologically monotonic (confirmed empirically - it's assigned by
    MLB's scheduling system, not in play order), and downstream code
    (teams.py, lineup.py) relies on a higher game_id meaning a more recent
    game (e.g. groupby(...)['game_id'].max() for "latest game",
    sort_values(['team', 'game_id']) for chronological order).

    Previously this reconstructed game boundaries from scratch by detecting
    an at_bat_number reset to 1 while walking a single global counter over
    a (game_date, home_team, away_team, pitcher, at_bat_number)-deduped
    table. That only produces correct results if every row belonging to one
    real game is contiguous in the table first - nothing guarantees that:
    persist_raw_statcast only sorts by game_date (a coarse key), and MLB
    plays ~15 games in parallel most days, so their at-bat rows interleave
    in whatever order Statcast/pybaseball originally delivered them.
    Confirmed empirically to badly fragment (and occasionally conflate)
    real games once replayed against a large multi-day dataset - e.g. one
    real batter's games were split into 2-3 different `game_id` values on
    93 of 96 real game dates in one sample. game_pk sidesteps this
    entirely, since it doesn't need to be reconstructed at all.
    """
    game_order = (
        df[["game_date", "game_pk"]]
        .drop_duplicates()
        .sort_values(["game_date", "game_pk"])
        .reset_index(drop=True)
    )
    game_order["game_id"] = game_order.index + 1

    data = df.merge(game_order, on=["game_date", "game_pk"])
    data["ind"] = (
        data["game_id"].astype("str")
        + data["at_bat_number"].astype("str")
        + data["pitch_number"].astype("str")
    ).astype("int")
    return data.set_index("ind").sort_index()


def latest_team_for_batters(df: pd.DataFrame) -> pd.DataFrame:
    """Most recent team each batter appeared for, keyed by key_mlbam."""
    home = (
        df[df["inning_topbot"] == "Bot"][["batter", "home_team", "game_date"]]
        .drop_duplicates()
        .rename(columns={"home_team": "team"})
    )
    away = (
        df[df["inning_topbot"] == "Top"][["batter", "away_team", "game_date"]]
        .drop_duplicates()
        .rename(columns={"away_team": "team"})
    )
    team = pd.concat([home, away])

    latest_player = df[["game_date", "batter"]].groupby("batter", as_index=False).max()
    return (
        latest_player.merge(team, on=["batter", "game_date"], how="left")[["batter", "team"]]
        .rename(columns={"batter": "key_mlbam"})
    )


def latest_team_for_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    """Most recent team each pitcher appeared for, keyed by key_mlbam."""
    homep = (
        df[df["inning_topbot"] == "Bot"][["pitcher", "away_team", "game_date"]]
        .drop_duplicates()
        .rename(columns={"away_team": "team"})
    )
    awayp = (
        df[df["inning_topbot"] == "Top"][["pitcher", "home_team", "game_date"]]
        .drop_duplicates()
        .rename(columns={"home_team": "team"})
    )
    teamp = pd.concat([homep, awayp])

    latest_pitcher = df[["game_date", "pitcher"]].groupby("pitcher", as_index=False).max()
    return (
        latest_pitcher.merge(teamp, on=["pitcher", "game_date"], how="left")[["pitcher", "team"]]
        .rename(columns={"pitcher": "key_mlbam"})
    )


def completed_events(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Filter pitch-by-pitch rows down to one row per completed plate-appearance event."""
    return df[df["events"].isin(config.COUNTED_EVENTS)][columns]


def label_pitcher_roles(data_with_game_id: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, pitching team, pitcher): `is_starter` is True
    for whichever pitcher faced that team's opponent first (lowest
    at_bat_number) in that game, False for every other pitcher who appeared
    for that team in that game.

    Role is determined per game-appearance rather than as a season-long
    per-pitcher label, so a pitcher who both starts and relieves across the
    season (a swingman, an opener's follower, etc.) is classified correctly
    in each individual game rather than bucketed into one role for all of it.
    """
    top = data_with_game_id[data_with_game_id["inning_topbot"] == "Top"][
        ["game_id", "home_team", "pitcher", "at_bat_number"]
    ].rename(columns={"home_team": "team"})
    bot = data_with_game_id[data_with_game_id["inning_topbot"] == "Bot"][
        ["game_id", "away_team", "pitcher", "at_bat_number"]
    ].rename(columns={"away_team": "team"})
    appearances = pd.concat([top, bot])

    first_ab = (
        appearances.groupby(["game_id", "team"], as_index=False)["at_bat_number"]
        .min()
        .rename(columns={"at_bat_number": "first_ab"})
    )
    tagged = appearances.merge(first_ab, on=["game_id", "team"])
    starters = tagged[tagged["at_bat_number"] == tagged["first_ab"]][["game_id", "team", "pitcher"]]
    starters = starters.drop_duplicates().rename(columns={"pitcher": "starting_pitcher"})

    roles = appearances[["game_id", "team", "pitcher"]].drop_duplicates().merge(
        starters, on=["game_id", "team"], how="left"
    )
    roles["is_starter"] = roles["pitcher"] == roles["starting_pitcher"]
    return roles[["game_id", "team", "pitcher", "is_starter"]]


def extract_game_results(data_with_game_id: pd.DataFrame) -> pd.DataFrame:
    """One row per game_id with its final state: [game_id, game_date,
    home_team, away_team, home_score, away_score] - the actual final score,
    found the same way teams.build_team_record finds each game's final
    scoring state (the row with the maximum combined score), but kept in
    home/away form rather than pivoted to a team/opponent perspective.
    Used to reconstruct real historical games (actual outcomes, not a
    schedule-API fetch) for backtesting - see game_picks_backtest.py."""
    totals = data_with_game_id[
        ["game_id", "game_date", "home_team", "away_team", "post_home_score", "post_away_score"]
    ].copy()
    totals["fs"] = totals["post_home_score"] + totals["post_away_score"]
    final_state = totals.groupby("game_id", as_index=False)["fs"].max()
    final_state = final_state.merge(totals, on=["game_id", "fs"]).drop_duplicates(subset="game_id")
    return final_state.rename(columns={"post_home_score": "home_score", "post_away_score": "away_score"})[
        ["game_id", "game_date", "home_team", "away_team", "home_score", "away_score"]
    ]


def assign_batting_order(data_with_game_id: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, team, batter) who batted in that game:
    `batting_order` is their rank (1, 2, 3, ...) by first at_bat_number
    within their team's batting half-innings for that game. Ranks 1-9 are
    the starting lineup in order; rank >9 is a substitute (pinch hitter/
    runner) who entered mid-game - callers that only want the starting
    lineup should filter to batting_order <= 9. A substitute's first at-bat
    necessarily comes after all 9 starters' first at-bats, so this needs no
    special-casing for extra-inning bench-emptying.

    Batting-team convention is the OPPOSITE of label_pitcher_roles's
    pitching-team convention above: Top half -> away_team bats, Bot half ->
    home_team bats (matches latest_team_for_batters, not label_pitcher_roles).
    """
    top = data_with_game_id[data_with_game_id["inning_topbot"] == "Top"][
        ["game_id", "away_team", "batter", "at_bat_number"]
    ].rename(columns={"away_team": "team"})
    bot = data_with_game_id[data_with_game_id["inning_topbot"] == "Bot"][
        ["game_id", "home_team", "batter", "at_bat_number"]
    ].rename(columns={"home_team": "team"})
    appearances = pd.concat([top, bot])

    first_ab = appearances.groupby(["game_id", "team", "batter"], as_index=False)["at_bat_number"].min()
    first_ab["batting_order"] = (
        first_ab.groupby(["game_id", "team"])["at_bat_number"].rank(method="first").astype(int)
    )
    return first_ab[["game_id", "team", "batter", "batting_order"]]

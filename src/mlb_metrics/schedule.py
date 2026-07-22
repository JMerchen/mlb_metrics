"""Today's probable starting pitchers, and (for Automated Game Picks) final
game results, via the MLB Stats API.

`pybaseball` has no schedule/lineup/probable-pitcher support at all - this
is a genuinely new data source (statsapi.mlb.com, free/unauthenticated, via
the `MLB-StatsAPI` package). Confirmed starting lineups aren't available
here - they post only ~2-4 hours before first pitch, structurally too late
for this pipeline's 8am ET run regardless of data source - only probable
pitchers, which are typically settled days in advance.

This queries the lower-level `statsapi.get("schedule", ...)` call (the raw
MLB Stats API JSON) rather than the `statsapi.schedule()` convenience
wrapper. Confirmed via a live run of `scripts/debug_statsapi.py` (the
`Debug statsapi` GitHub Actions workflow, run 29878447354): the convenience
wrapper's `home_probable_pitcher`/`away_probable_pitcher` fields are name
strings only, with no numeric ID anywhere in that response - the raw
endpoint's `teams.{home,away}.probablePitcher.id` is genuinely required to
get the same ID space as Statcast's batter/pitcher columns. The field paths
parsed below (`teams.{home,away}.team.id`, `.probablePitcher.id`) match the
live response exactly. That same live response also confirmed
`status.detailedState` and `teams.{home,away}.score` as top-level fields on
every game (no extra hydrate needed) - but only for an in-progress game, not
a completed one, so `normalize_schedule_games`'s "Final" status handling is
a reasonable extrapolation, not itself live-confirmed. Verify it the same
way (a `Debug statsapi` run against a known past date) before trusting
Automated Game Picks' resolution in production.
"""

import pandas as pd

# Ported from docs/app.js's teamLogo() table (already validated against
# Statcast's own home_team/away_team abbreviations) and inverted. Joining on
# MLB's numeric team ID rather than trying to string-match abbreviations is
# deliberate - MLB Stats API and Baseball Savant are known to disagree on
# some codes (AZ/ARI, CWS/CHW, SD/SDP, SF/SFG, TB/TBR, KC/KCR, WSH/WSN).
TEAM_ID_TO_ABBREV = {
    109: "AZ", 144: "ATL", 110: "BAL", 111: "BOS", 112: "CHC",
    145: "CWS", 113: "CIN", 114: "CLE", 115: "COL", 116: "DET",
    117: "HOU", 118: "KC", 108: "LAA", 119: "LAD", 146: "MIA",
    158: "MIL", 142: "MIN", 121: "NYM", 147: "NYY", 133: "ATH",
    143: "PHI", 134: "PIT", 135: "SD", 137: "SF", 136: "SEA",
    138: "STL", 139: "TB", 140: "TEX", 141: "TOR", 120: "WSH",
}


def normalize_schedule(raw: dict, fallback_date=None) -> pd.DataFrame:
    """Parse a raw `statsapi.get("schedule", ...)` response into one row per
    team per game: [date, team, opponent, probable_pitcher_key_mlbam,
    game_pk, is_home]. `probable_pitcher_key_mlbam` is null if not yet
    announced. Games for a team whose numeric ID isn't in TEAM_ID_TO_ABBREV
    are skipped rather than guessed at. V1 simplification: a doubleheader
    only keeps a team's first game of the day, not both - for a game-level
    view that keeps every game (needed to resolve Automated Game Picks),
    see normalize_schedule_games instead.
    """
    rows = []
    seen_teams = set()

    for date_entry in raw.get("dates", []):
        date = date_entry.get("date", fallback_date)
        for game in date_entry.get("games", []):
            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            home_abbrev = TEAM_ID_TO_ABBREV.get((home.get("team") or {}).get("id"))
            away_abbrev = TEAM_ID_TO_ABBREV.get((away.get("team") or {}).get("id"))
            if home_abbrev is None or away_abbrev is None:
                continue

            for team_abbrev, opponent_abbrev, side, is_home in (
                (home_abbrev, away_abbrev, home, True),
                (away_abbrev, home_abbrev, away, False),
            ):
                if team_abbrev in seen_teams:
                    continue
                seen_teams.add(team_abbrev)
                probable = side.get("probablePitcher") or {}
                rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "team": team_abbrev,
                        "opponent": opponent_abbrev,
                        "probable_pitcher_key_mlbam": probable.get("id"),
                        "game_pk": game.get("gamePk"),
                        "is_home": is_home,
                    }
                )

    return pd.DataFrame(
        rows, columns=["date", "team", "opponent", "probable_pitcher_key_mlbam", "game_pk", "is_home"]
    )


def normalize_schedule_games(raw: dict, fallback_date=None) -> pd.DataFrame:
    """Parse a raw `statsapi.get("schedule", ...)` response into one row per
    GAME (not per team, unlike normalize_schedule): [game_pk, date,
    home_team, away_team, home_probable_pitcher_key_mlbam,
    away_probable_pitcher_key_mlbam, status, home_score, away_score].
    `status` is the raw `detailedState` string (e.g. "Scheduled",
    "In Progress", "Final") - callers should treat anything other than
    "Final" as not-yet-resolvable rather than special-casing every possible
    in-between state. `home_score`/`away_score` are null until the game
    starts posting a score.

    Unlike normalize_schedule, doubleheaders are NOT deduped - dropping game
    2 here would silently lose an entire game's pick/resolution, not just
    reuse a slightly-stale metric like the team-quality join can tolerate.
    Both games of a doubleheader will therefore get an identical predicted
    win probability (the team composite doesn't change between games) -
    a known first-pass simplification, not fixed here.
    """
    rows = []
    for date_entry in raw.get("dates", []):
        date = date_entry.get("date", fallback_date)
        for game in date_entry.get("games", []):
            teams = game.get("teams", {})
            home = teams.get("home", {})
            away = teams.get("away", {})
            home_abbrev = TEAM_ID_TO_ABBREV.get((home.get("team") or {}).get("id"))
            away_abbrev = TEAM_ID_TO_ABBREV.get((away.get("team") or {}).get("id"))
            if home_abbrev is None or away_abbrev is None:
                continue

            home_probable = home.get("probablePitcher") or {}
            away_probable = away.get("probablePitcher") or {}
            rows.append(
                {
                    "game_pk": game.get("gamePk"),
                    "date": pd.Timestamp(date),
                    "home_team": home_abbrev,
                    "away_team": away_abbrev,
                    "home_probable_pitcher_key_mlbam": home_probable.get("id"),
                    "away_probable_pitcher_key_mlbam": away_probable.get("id"),
                    "status": (game.get("status") or {}).get("detailedState"),
                    "home_score": home.get("score"),
                    "away_score": away.get("score"),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "game_pk", "date", "home_team", "away_team",
            "home_probable_pitcher_key_mlbam", "away_probable_pitcher_key_mlbam",
            "status", "home_score", "away_score",
        ],
    )


def _fetch_raw_schedule(date, hydrate: str = "probablePitcher,team") -> dict:
    """Thin wrapper around the raw `statsapi.get("schedule", ...)` call,
    shared by every fetch function below so there's one place to monkeypatch
    in tests and one place documenting the data source."""
    import statsapi

    return statsapi.get("schedule", {"sportId": 1, "date": str(date), "hydrate": hydrate})


def fetch_probable_pitchers(date) -> pd.DataFrame:
    """Today's games and probable starting pitchers, one row per team. See
    module docstring for the data source and its limitations."""
    return normalize_schedule(_fetch_raw_schedule(date), fallback_date=date)


def fetch_todays_games(date) -> pd.DataFrame:
    """Today's games, one row per game - see normalize_schedule_games."""
    return normalize_schedule_games(_fetch_raw_schedule(date), fallback_date=date)


def fetch_game_results(date) -> pd.DataFrame:
    """Games and their final scores for a PAST `date` - same endpoint and
    shape as fetch_todays_games, called backward instead of forward.
    `status`/`home_score`/`away_score` should be populated once each game
    is Final; anything still "Scheduled"/"In Progress"/postponed is not yet
    resolvable and callers should leave it pending rather than guess."""
    return normalize_schedule_games(_fetch_raw_schedule(date), fallback_date=date)

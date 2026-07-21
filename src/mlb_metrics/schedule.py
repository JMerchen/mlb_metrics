"""Today's probable starting pitchers, via the MLB Stats API.

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
live response exactly.
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
    team per game: [date, team, opponent, probable_pitcher_key_mlbam].
    `probable_pitcher_key_mlbam` is null if not yet announced. Games for a
    team whose numeric ID isn't in TEAM_ID_TO_ABBREV are skipped rather than
    guessed at. V1 simplification: a doubleheader only keeps a team's first
    game of the day, not both.
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

            for team_abbrev, opponent_abbrev, side in (
                (home_abbrev, away_abbrev, home),
                (away_abbrev, home_abbrev, away),
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
                    }
                )

    return pd.DataFrame(rows, columns=["date", "team", "opponent", "probable_pitcher_key_mlbam"])


def fetch_probable_pitchers(date) -> pd.DataFrame:
    """Today's games and probable starting pitchers. See module docstring
    for the data source and its limitations."""
    import statsapi

    raw = statsapi.get("schedule", {"sportId": 1, "date": str(date), "hydrate": "probablePitcher,team"})
    return normalize_schedule(raw, fallback_date=date)

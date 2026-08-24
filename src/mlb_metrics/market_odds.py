"""Real ESPN moneyline odds ingestion - quant-analytics item #6 ("no
market benchmark"), slice 2. Promotes slice 1's confirmation-only
`scripts/debug_espn_odds.py` math/fetch logic into a real module once the
real response shape, team-abbreviation crosswalk, and historical depth
(5 real days back, confirmed 2026-08-21 - see README's "Market benchmark"
section) were all confirmed via a real dispatched run.

ESPN's `site.api.espn.com` scoreboard/summary endpoints are public and
keyless (no auth needed), but unofficial - this project's own dev sandbox
blocks the domain directly, same as every other odds-site domain tried;
`fetch_scoreboard`/`fetch_summary` are only exercised for real from a
GitHub Actions runner or the live pipeline, never unit tested directly
(same split `schedule.py` uses: `_fetch_raw_schedule` isn't tested,
`normalize_schedule_games` is). Everything else in this module is pure
parsing/math and IS unit tested.

`fetch_market_home_win_probabilities` matches games by (home_team,
away_team) only, not date+game_pk - ESPN's event IDs are unrelated to
MLB Stats API's game_pk, and the caller already scopes each call to a
single date. This is the same doubleheader simplification
`schedule.normalize_schedule_games` already documents for itself: a real,
disclosed limitation (two same-day games between the same two teams would
collide), not a silent one.
"""

import pandas as pd
import requests

from mlb_metrics import config

SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard"
SUMMARY_URL = "https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/summary"

# The real crosswalk slice 1's dispatch found (GitHub Actions run
# 32516808493, 2026-08-21): ESPN's "ARI"/"CHW" vs. this project's own
# schedule.TEAM_ID_TO_ABBREV "AZ"/"CWS". All other 28 real abbreviations
# ESPN returned matched schedule.TEAM_ID_TO_ABBREV's values exactly.
ESPN_TEAM_ABBREV_FIXUPS = {"ARI": "AZ", "CHW": "CWS"}


def moneyline_to_implied_probability(moneyline: float) -> float:
    """Standard American-odds-to-implied-probability conversion."""
    if moneyline < 0:
        return -moneyline / (-moneyline + 100)
    return 100 / (moneyline + 100)


def devig(home_implied: float, away_implied: float) -> float:
    """Standard proportional de-vig - normalizes both sides' implied
    probabilities to sum to 1.0, removing the bookmaker's overround. The
    simplest, most defensible de-vig method, not the only one that
    exists."""
    return home_implied / (home_implied + away_implied)


def _apply_abbrev_fixup(abbrev):
    return ESPN_TEAM_ABBREV_FIXUPS.get(abbrev, abbrev)


def fetch_scoreboard(date: str) -> dict:
    resp = requests.get(SCOREBOARD_URL, params={"dates": date}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def fetch_summary(event_id: str) -> dict:
    resp = requests.get(SUMMARY_URL, params={"event": event_id}, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _parse_pickcenter_row(summary_json: dict):
    """Pulls the real odds row for config.MARKET_ODDS_PREFERRED_PROVIDER
    ("DraftKings" - the only provider slice 1's dispatch ever saw) out of
    a real summary response's `pickcenter` array, falling back to the
    first available row if that provider is ever absent for a game
    (defensive, not yet observed for real). Returns
    (provider_name, home_moneyline, away_moneyline), or None if
    `pickcenter` is missing/empty."""
    pickcenter = summary_json.get("pickcenter") or []
    if not pickcenter:
        return None
    row = next(
        (r for r in pickcenter if (r.get("provider") or {}).get("name") == config.MARKET_ODDS_PREFERRED_PROVIDER),
        pickcenter[0],
    )
    away = row.get("awayTeamOdds") or {}
    home = row.get("homeTeamOdds") or {}
    provider_name = (row.get("provider") or {}).get("name")
    return provider_name, home.get("moneyLine"), away.get("moneyLine")


def _extract_market_row(event: dict, summary_json: dict):
    """Combines one real ESPN scoreboard event (for real home/away team
    abbreviations, fixed up via ESPN_TEAM_ABBREV_FIXUPS) with that same
    event's real summary response (for pickcenter moneylines) into
    {home_team, away_team, market_home_win_probability, market_provider,
    home_moneyline, away_moneyline}. The raw moneylines are carried
    alongside the de-vigged probability, not just used to compute it and
    discarded - the de-vigged probability answers "who's the better
    forecaster" (kelly.py/scripts/recommend_bets.py's own comments explain
    why real bet sizing needs the RAW, vigged price instead - you're paid
    off at the real price, not a de-vigged one). Returns None if either
    the home/away split or the moneylines are missing - a real, honest
    "no data for this game" case, not a crash."""
    home_abbrev = None
    away_abbrev = None
    for competition in event.get("competitions", []):
        for competitor in competition.get("competitors", []):
            team = competitor.get("team") or {}
            abbrev = _apply_abbrev_fixup(team.get("abbreviation"))
            if competitor.get("homeAway") == "home":
                home_abbrev = abbrev
            elif competitor.get("homeAway") == "away":
                away_abbrev = abbrev

    if home_abbrev is None or away_abbrev is None:
        return None

    parsed = _parse_pickcenter_row(summary_json)
    if parsed is None:
        return None
    provider_name, home_ml, away_ml = parsed
    if home_ml is None or away_ml is None:
        return None

    home_implied = moneyline_to_implied_probability(home_ml)
    away_implied = moneyline_to_implied_probability(away_ml)

    return {
        "home_team": home_abbrev,
        "away_team": away_abbrev,
        "market_home_win_probability": devig(home_implied, away_implied),
        "market_provider": provider_name,
        "home_moneyline": home_ml,
        "away_moneyline": away_ml,
    }


def fetch_market_home_win_probabilities(date) -> pd.DataFrame:
    """Real market home-win probabilities for every game on `date`
    (any pandas-parseable date). One real scoreboard fetch, then one real
    summary fetch per event - each event wrapped in its own try/except so
    one bad/unreachable/oddly-shaped game can't drop the rest (same
    per-item isolation game_predictions.resolve_game_predictions already
    uses per pending date). Returns a DataFrame with columns
    [home_team, away_team, market_home_win_probability, market_provider,
    home_moneyline, away_moneyline], one row per game that had usable
    odds - an empty DataFrame (same columns) if none did. Matching onto
    this project's own picks is by (home_team, away_team) only - see the
    module docstring's doubleheader caveat."""
    date_str = pd.Timestamp(date).strftime("%Y%m%d")
    scoreboard = fetch_scoreboard(date_str)

    rows = []
    for event in scoreboard.get("events", []):
        event_id = event.get("id")
        if event_id is None:
            continue
        try:
            summary = fetch_summary(event_id)
            row = _extract_market_row(event, summary)
        except Exception as exc:
            print(
                f"WARNING: failed to fetch/parse real ESPN odds for event {event_id} on {date_str} "
                f"({exc}); skipping that game."
            )
            continue
        if row is not None:
            rows.append(row)

    columns = [
        "home_team", "away_team", "market_home_win_probability", "market_provider",
        "home_moneyline", "away_moneyline",
    ]
    return pd.DataFrame(rows, columns=columns)

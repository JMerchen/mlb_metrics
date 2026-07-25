"""Fielding-position eligibility for today's DFS hitter pool, via the MLB
Stats API's `people` endpoint (statsapi.get("people", {"personIds": ...}))
- a genuinely new data source for this project (no fielding-position data
exists anywhere else in this codebase). Queried directly by `key_mlbam`
(this project's own player id, the same MLBAM numeric id space
statsapi.mlb.com uses), so no name-matching is needed, unlike a team-roster
lookup - the same numeric-ID-join discipline schedule.py already uses for
team ids.

The `people` endpoint's `primaryPosition.abbreviation` field is confirmed
via the installed MLB-StatsAPI package's own `roster()` implementation
(which parses the structurally identical `position.abbreviation` shape off
the team-roster endpoint) and public MLB Stats API documentation - but NOT
live-confirmed against a real response, since this sandbox cannot reach
statsapi.mlb.com at all (the same network restriction schedule.py's module
docstring documents). See scripts/debug_statsapi_positions.py - run that
through the `Debug statsapi` GitHub Actions workflow (real network access)
before fully trusting the field paths here, the same bootstrapping step
schedule.py's own probable-pitcher field paths went through
(scripts/debug_statsapi.py).

Restricted to a single PRIMARY position per player - NOT DraftKings' real
multi-position eligibility (which DK derives from a player's recent
multi-position starts). A player whose primaryPosition has no DK Classic
MLB fielding slot at all (most commonly "DH", a common primary role for
several everyday hitters today) is excluded outright, not defaulted to a
guessed slot - a real v1 gap, not hidden (see this module's tests and
README's Optimal Lineup section).
"""

import pandas as pd

# MLB Stats API primaryPosition.abbreviation -> DraftKings Classic MLB slot.
# Corner/center outfield positions all collapse to DK's single "OF" slot
# (DK Classic MLB has no separate LF/CF/RF slots). Pitchers (starters or
# relievers) both map to "P" - dfs_optimizer.py's pool only ever includes
# today's probable STARTERS pitcher-side (inherited from dfs.py's own
# starter-only restriction), so "RP" appearing here is for completeness,
# not because a reliever could actually be selected.
POSITION_TO_DK_SLOT = {
    "C": "C",
    "1B": "1B",
    "2B": "2B",
    "3B": "3B",
    "SS": "SS",
    "LF": "OF",
    "CF": "OF",
    "RF": "OF",
    "OF": "OF",
    "P": "P",
    "SP": "P",
    "RP": "P",
}

CHUNK_SIZE = 100  # unconfirmed - see module docstring's live-verification requirement


def normalize_position_eligibility(people: list) -> pd.DataFrame:
    """Parses a list of statsapi `people` response dicts into
    [key_mlbam, primary_position, dk_slot]. A person with no
    primaryPosition, or one with no DK Classic MLB slot (most commonly
    "DH"), is excluded from the result entirely - not defaulted."""
    rows = []
    for person in people:
        key_mlbam = person.get("id")
        primary_position = (person.get("primaryPosition") or {}).get("abbreviation")
        if key_mlbam is None or primary_position is None:
            continue
        dk_slot = POSITION_TO_DK_SLOT.get(primary_position)
        if dk_slot is None:
            continue
        rows.append({"key_mlbam": key_mlbam, "primary_position": primary_position, "dk_slot": dk_slot})
    return pd.DataFrame(rows, columns=["key_mlbam", "primary_position", "dk_slot"])


def fetch_position_eligibility(key_mlbams: list) -> pd.DataFrame:
    """Fetches primaryPosition for every id in `key_mlbams` from the MLB
    Stats API's `people` endpoint, chunked (CHUNK_SIZE ids/call - an
    unconfirmed guess at a safe URL length, see module docstring) since a
    full slate's hitter pool can be several hundred players. Raises on any
    fetch failure (network down, malformed response) rather than returning
    a partial/empty result silently - callers (scripts/build_optimal_lineup.py)
    are expected to catch this and fall back to leaving a prior day's
    output in place, the same resilience pattern
    schedule.fetch_probable_pitchers' callers already use."""
    import statsapi

    key_mlbams = list(dict.fromkeys(key_mlbams))  # de-dupe, preserve order
    all_people = []
    for i in range(0, len(key_mlbams), CHUNK_SIZE):
        chunk = key_mlbams[i:i + CHUNK_SIZE]
        response = statsapi.get("people", {"personIds": ",".join(str(k) for k in chunk)})
        all_people.extend(response.get("people", []))

    return normalize_position_eligibility(all_people)

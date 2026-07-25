"""One-off diagnostic: print the real shape of statsapi's `people` endpoint
response for a handful of real MLBAM player ids, to confirm
roster_positions.py's field-path guesses (`people[i]["id"]`,
`people[i]["primaryPosition"]["abbreviation"]`) before trusting them in
production. This sandbox that built roster_positions.py can't reach
statsapi.mlb.com at all, so this has to run somewhere with real network
access (see .github/workflows/debug_statsapi_positions.yml) - the same
bootstrapping step scripts/debug_statsapi.py already went through for
schedule.py's probable-pitcher field paths.

Usage: python scripts/debug_statsapi_positions.py [id1,id2,...]
  (defaults to a few well-known real MLBAM ids spanning positions -
  Aaron Judge (OF), Mookie Betts (2B/SS/OF), Yadier Molina (C, retired but
  a stable historical id) - if a default id has since become invalid,
  pass real current ids explicitly).
Delete this script (and its workflow) once roster_positions.py's field
paths are confirmed live and battle-tested - it's a bootstrapping tool,
not part of the pipeline.
"""

import json
import sys

DEFAULT_IDS = [592450, 605141, 425877]  # Judge, Betts, Molina


def main():
    ids = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 and sys.argv[1].strip() else DEFAULT_IDS

    import statsapi

    print(f"statsapi version: {getattr(statsapi, '__version__', 'unknown')}")
    print(f"Querying personIds={ids}")

    raw = statsapi.get("people", {"personIds": ",".join(str(i) for i in ids)})
    print("\nTop-level keys:", sorted(raw.keys()))
    people = raw.get("people", [])
    print(f"{len(people)} people returned")
    for person in people:
        print(f"\n--- id={person.get('id')} ---")
        print(json.dumps(person, indent=2, default=str))


if __name__ == "__main__":
    main()

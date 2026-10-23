"""One-off diagnostic: what MLB's Stats API draft prospects feed returns
(/api/v1/draft/prospects/{year}) for each year given, and what
mlb_draft_sources.fetch_mlb_pipeline_draft_prospects makes of it - to
confirm the request and field names against an already-published class
(2026) while the 2027 list is still empty. The development sandbox can't
reach statsapi.mlb.com, so this runs in GitHub Actions
(.github/workflows/debug_mlb_draft_prospects.yml).

Usage: python scripts/debug_mlb_draft_prospects.py [year ...]  (default 2026 2027)
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import requests  # noqa: E402

from mlb_metrics import mlb_draft_sources  # noqa: E402


def main():
    years = [int(y) for y in sys.argv[1:] if y.strip()] or [2026, 2027]
    for year in years:
        url = f"https://statsapi.mlb.com/api/v1/draft/prospects/{year}"
        print(f"\n=== {year}: {url}")
        response = requests.get(url, timeout=30)
        print("status", response.status_code)
        body = response.json()
        print("top-level keys:", sorted(body.keys()))
        prospects = body.get("prospects", [])
        ranked = [p for p in prospects if p.get("rank")]
        print(f"{len(prospects)} prospects, {len(ranked)} with a rank")
        print("school classes:", sorted({str((p.get('school') or {}).get('schoolClass')) for p in prospects})[:20])
        for prospect in (ranked or prospects)[:2]:
            print(json.dumps(prospect, indent=2, default=str)[:1500])
        board = mlb_draft_sources.fetch_mlb_pipeline_draft_prospects(season=year)
        print(f"fetcher result: {len(board)} rows")
        if not board.empty:
            print(board.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

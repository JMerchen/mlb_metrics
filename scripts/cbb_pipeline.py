"""Daily college basketball run. Logic lives in
src/mlb_metrics/cbb_pipeline.py (ratings, game picks log, bracket odds ->
docs/data/cbb.json).

Usage:
    python scripts/cbb_pipeline.py [--season 2027]
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import cbb_pipeline  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    args = parser.parse_args()
    cbb_pipeline.run(season=args.season)

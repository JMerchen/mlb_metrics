"""Weekly NFL game-picks pipeline entrypoint. Logic lives in
src/mlb_metrics/nfl_pipeline.py; see that module's own docstring for the
full orchestration (fetch, resolve pending picks, predict the next
predictable week, log, export).

Usage:
    python scripts/nfl_pipeline.py
    python scripts/nfl_pipeline.py --season 2026
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics.nfl_pipeline import main

if __name__ == "__main__":
    main()

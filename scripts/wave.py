"""Daily pipeline entrypoint. Logic lives in src/mlb_metrics/; see
src/mlb_metrics/pipeline.py for the orchestration and README.md for how to
run this with a specific --as-of-date (e.g. for backtesting)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics.pipeline import main

if __name__ == "__main__":
    main()

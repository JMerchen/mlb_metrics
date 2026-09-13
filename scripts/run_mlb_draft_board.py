"""Runs the real MLB draft-eligible college board - fetches every real
source in mlb_draft_sources.SOURCE_FETCHERS, builds one consensus ranking
(consensus_rankings.build_consensus_ranking via board_runner.run_board),
and writes the full real ranking to data/mlb_draft_board.csv.

Real, honest scope: current college juniors/seniors/eligible
underclassmen not yet drafted - this is NOT the organizational prospect
board (see run_prospect_rankings.py for that).

Usage:
    python scripts/run_mlb_draft_board.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import board_runner, mlb_draft_sources


def main():
    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "mlb_draft_board.csv")
    consensus = board_runner.run_board(
        mlb_draft_sources.SOURCE_FETCHERS, output_path, "MLB Draft-Eligible College Board", top_n=100
    )
    if not consensus.empty:
        pd.set_option("display.width", 200)
        print()
        print(
            consensus[["consensus_rank", "player_name", "consensus_score", "sources_ranked_by"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()

"""Runs the real MLB organizational prospect consensus board - fetches
every real source in prospect_sources.SOURCE_FETCHERS, builds one
consensus ranking (consensus_rankings.build_consensus_ranking via
board_runner.run_board), and writes the full real ranking to
docs/data/prospect_rankings.csv.

Real, honest scope (see prospect_sources.py's own module docstring): a
"prospect" here means any real minor leaguer already in an MLB
organization who has not yet debuted in the majors - this is NOT the
upcoming draft class (see run_mlb_draft_board.py for that).

Usage:
    python scripts/run_prospect_rankings.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import board_runner, prospect_sources


def main():
    output_path = os.path.join(os.path.dirname(__file__), "..", "docs", "data", "prospect_rankings.csv")
    consensus = board_runner.run_board(
        prospect_sources.SOURCE_FETCHERS, output_path, "MLB Prospect Rankings", top_n=100
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

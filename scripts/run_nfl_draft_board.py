"""Runs the real NFL draft-eligible college board - fetches every real
source in nfl_draft_sources.SOURCE_FETCHERS, builds one consensus ranking
(consensus_rankings.build_consensus_ranking via board_runner.run_board),
and writes the full real ranking to docs/data/nfl_draft_board.csv.

Real, honest scope: current college players eligible for the next NFL
draft.

Usage:
    python scripts/run_nfl_draft_board.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from mlb_metrics import board_runner, nfl_draft_sources


def clear_stale_board(output_path: str) -> None:
    """When no source has the next class yet, run_board leaves the old
    file in place - fine after a bad fetch, wrong when that file is last
    year's class, already drafted. Replace a stale board with an empty
    one (header only) so the page says the board isn't out yet instead
    of listing drafted players."""
    if not os.path.exists(output_path):
        return
    existing = pd.read_csv(output_path)
    if existing.empty or "player_name" not in existing.columns:
        return
    drafted = nfl_draft_sources.already_drafted()
    if nfl_draft_sources.current_class_only(existing[["player_name"]], drafted).empty:
        existing.iloc[0:0].to_csv(output_path, index=False)
        print(f"{output_path} was last year's (already drafted) class - cleared it.")


def main():
    output_path = os.path.join(os.path.dirname(__file__), "..", "docs", "data", "nfl_draft_board.csv")
    consensus = board_runner.run_board(
        nfl_draft_sources.SOURCE_FETCHERS, output_path, "NFL Draft-Eligible College Board", top_n=100
    )
    if consensus.empty:
        clear_stale_board(output_path)
    if not consensus.empty:
        pd.set_option("display.width", 200)
        print()
        print(
            consensus[["consensus_rank", "player_name", "consensus_score", "sources_ranked_by"]]
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()

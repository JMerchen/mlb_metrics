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


def _undebuted_only(fetcher):
    """Wraps a real source fetcher so its own output is filtered to this
    board's own real, stated scope (not-yet-debuted minor leaguers) -
    see prospect_sources.filter_undebuted's own docstring for why this
    lives here rather than inside each fetcher: it's this SCRIPT's own
    real business rule (the MLB draft/NFL draft boards, which reuse the
    exact same fetcher/board_runner shape, have no such rule), not a
    property of how any one source's own page is scraped/normalized."""
    def wrapped():
        return prospect_sources.filter_undebuted(fetcher())
    return wrapped


def main():
    output_path = os.path.join(os.path.dirname(__file__), "..", "docs", "data", "prospect_rankings.csv")
    fetchers = {name: _undebuted_only(fetcher) for name, fetcher in prospect_sources.SOURCE_FETCHERS.items()}
    consensus = board_runner.run_board(
        fetchers, output_path, "MLB Prospect Rankings", top_n=100
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

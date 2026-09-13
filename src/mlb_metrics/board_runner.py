"""Shared "fetch every real source -> build one consensus ranking ->
report honestly -> write CSV" runner logic for all 3 consensus boards
this project builds (MLB organizational prospects, MLB draft-eligible
college players, NFL draft-eligible college players) - one real function,
reused by each board's own thin scripts/run_*.py entrypoint, the same
"shared runner, per-board thin script" split
scripts/run_nfl_game_picks_backtest.py already establishes for its own
domain."""

import os

import pandas as pd

from mlb_metrics import consensus_rankings


def run_board(source_fetchers: dict, output_path: str, board_name: str, top_n: int = None) -> pd.DataFrame:
    """Calls every real fetcher in `source_fetchers` ({name: no-arg
    callable} returning a real DataFrame in
    consensus_rankings.build_consensus_ranking's own input contract),
    prints an honest per-source success/failure line (a real fetch
    failure is never hidden), builds one consensus ranking across
    whichever sources actually returned data, writes the FULL real
    ranking to `output_path` as CSV, and returns it (truncated to
    `top_n` rows if given - the written CSV always carries the full real
    ranking regardless).

    If EVERY real source fails on a given run (a real bad day for the
    underlying sites, not a hypothetical), nothing is written and the
    prior day's CSV is left in place - a real, honest degrade (keep
    yesterday's real board rather than overwrite it with an empty file)
    rather than destroying good historical data for one bad run, same
    "don't let one bad day erase real history" posture this project
    already applies to persisted raw data elsewhere."""
    print(f"Fetching real sources for {board_name}...")
    source_rankings = {}
    for name, fetcher in source_fetchers.items():
        df = fetcher()
        source_rankings[name] = df
        status = f"{len(df)} real players" if len(df) else "FAILED (see fetch log above)"
        print(f"  {name}: {status}")

    succeeded = sum(1 for df in source_rankings.values() if len(df) > 0)
    if succeeded == 0:
        print(f"No real sources returned data for {board_name} - writing nothing, leaving the prior CSV in place.")
        return pd.DataFrame()

    consensus = consensus_rankings.build_consensus_ranking(source_rankings)
    print(f"\n{board_name}: {len(consensus)} real players ranked by {succeeded} of {len(source_fetchers)} real sources.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    _write_csv(consensus, output_path)
    print(f"Wrote {output_path}")

    return consensus.head(top_n) if top_n else consensus


def _write_csv(df: pd.DataFrame, path: str) -> None:
    """CSV has no real concept of a nested dict - `source_ranks` (see
    consensus_rankings.build_consensus_ranking's own docstring) is
    flattened to its real string repr so the underlying per-source
    transparency survives in the committed file rather than being
    silently dropped."""
    out = df.copy()
    if "source_ranks" in out.columns:
        out["source_ranks"] = out["source_ranks"].apply(str)
    out.to_csv(path, index=False)

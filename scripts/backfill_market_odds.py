"""One-time seed for market_home_win_probability on the real game-picks
log (data/predictions/game_predictions.csv) - quant-analytics item #6
("no market benchmark"), slice 2. Going forward, pipeline.run() logs real
ESPN market odds alongside every new day's picks automatically
(market_odds.fetch_market_home_win_probabilities); this script exists to
seed the confirmed-reachable historical window that already accumulated
before slice 2 shipped, so the real "beat the closing line" comparison
has more than a few days of real data the moment it lands.

Defaults to the real depth quant-analytics item #6 slice 1's confirmation
dispatch actually confirmed ESPN still serves odds for (5 days back,
checked 2026-08-16 - see README's "Market benchmark" section and
config.MARKET_ODDS_BACKFILL_DAYS_BACK). A larger --days-back can be
passed, but real depth beyond 5 days is unconfirmed - dates ESPN no
longer serves odds for simply come back empty, not an error.

Usage:
    python scripts/backfill_market_odds.py
    python scripts/backfill_market_odds.py --days-back 10
"""

import argparse
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, game_evaluation, market_odds


def backfill_market_probabilities(log_path: str, fetch_fn, dates) -> pd.DataFrame:
    """Fills in market_home_win_probability on the game-predictions log at
    `log_path`, matching real rows by (date, home_team, away_team) for
    each date in `dates` - ONLY where that column is currently null, never
    overwriting a value already there (including one an earlier backfill
    run already found). `fetch_fn(date) -> pd.DataFrame` is injected - same
    dependency-injection pattern game_predictions.resolve_game_predictions
    already uses for fetch_results_fn - so this is testable without real
    network. A date whose fetch fails is skipped (warned, not fatal) so
    one bad date can't block the rest. Writes the updated log back to
    `log_path` and returns it.

    Defended against a real doubleheader collision: market_odds.py matches
    by (home_team, away_team) only (no date/game key - a disclosed,
    pre-existing limitation, same one game_predictions.advise_bets already
    guards against on the live bet-advice side). If `market` has more than
    one row for the same team pair on a date, `.set_index(...).loc[key]`
    stops returning a scalar for EVERY key that day (not just the
    duplicated one) once the index itself is non-unique, and a naive
    `.at[idx, col] = ...` assignment silently stores that non-scalar
    object into the cell - it round-trips through `to_csv` as a garbled
    string repr instead of a real float. Any duplicated team pair is
    dropped from `market` before the lookup, with a warning, rather than
    guessing which of the real games a plain team-pair match can't tell
    apart - the same "drop both, don't guess" choice
    game_predictions.select_game_picks already makes for advised bets."""
    log = pd.read_csv(log_path, parse_dates=["date"])
    if "market_home_win_probability" not in log.columns:
        log["market_home_win_probability"] = pd.NA

    for date in dates:
        date = pd.Timestamp(date)
        try:
            market = fetch_fn(date)
        except Exception as exc:
            print(f"WARNING: failed to fetch real ESPN market odds for {date.date()} ({exc}); skipping that date.")
            continue
        if market is None or market.empty:
            continue

        dupe_mask = market.duplicated(subset=["home_team", "away_team"], keep=False)
        if dupe_mask.any():
            dupe_matchups = sorted(set(zip(market.loc[dupe_mask, "home_team"], market.loc[dupe_mask, "away_team"])))
            print(
                f"WARNING: real ESPN market fetch for {date.date()} has more than one row for "
                f"matchup(s) {dupe_matchups} (a real doubleheader collision - market_odds.py "
                f"matches teams only, no date/game key); dropping those matchups for this "
                f"backfill rather than guessing which value belongs to which real game."
            )
            market = market[~dupe_mask]

        market_by_matchup = market.set_index(["home_team", "away_team"])["market_home_win_probability"]
        day_mask = (log["date"] == date) & log["market_home_win_probability"].isna()
        for idx in log.index[day_mask]:
            key = (log.at[idx, "home_team"], log.at[idx, "away_team"])
            if key in market_by_matchup.index:
                log.at[idx, "market_home_win_probability"] = float(market_by_matchup.loc[key])

    log.to_csv(log_path, index=False)
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-path", default="data/predictions/game_predictions.csv")
    parser.add_argument("--days-back", type=int, default=config.MARKET_ODDS_BACKFILL_DAYS_BACK)
    args = parser.parse_args()

    if not os.path.exists(args.log_path):
        print(f"{args.log_path} does not exist - nothing to backfill.")
        return

    log = pd.read_csv(args.log_path, parse_dates=["date"])
    resolved_dates = sorted(log.loc[log["game_played"] == 1, "date"].unique(), reverse=True)
    dates = resolved_dates[: args.days_back]

    if not dates:
        print("No real resolved dates found in the log - nothing to backfill.")
        return

    print(
        f"Backfilling real ESPN market odds for {len(dates)} real resolved date(s): "
        f"{[pd.Timestamp(d).date() for d in dates]}"
    )
    backfill_market_probabilities(args.log_path, market_odds.fetch_market_home_win_probabilities, dates)

    updated = pd.read_csv(args.log_path, parse_dates=["date"])
    _, summary = game_evaluation.build_game_picks_export(updated)
    print("\nReal market comparison summary after backfill:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

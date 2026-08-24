"""Daily bet-sizing report - a follow-up to quant-analytics item #6 (the
market benchmark). Turns "the model's probability disagrees with the
market's real price" into an actual recommended stake for that day's
still-unresolved logged game picks.

This is a REPORT a human reads and acts on manually, nothing more:
- Single straight bets only, never parlays. Parlays compound the book's
  vig across every leg (worse EV by construction) and would need real
  joint-probability modeling this project doesn't have
  (predictions._diversify_second_pick is an explicit sign-only proxy,
  not a real correlation estimate) to ever be justified.
- No execution/order-placement anywhere. Retail sportsbooks don't offer
  public betting APIs to individuals, and actively limit/ban bettors who
  show a persistent edge - "physically placing the bet" means a human
  reads this report's table and does it themselves.
- "Favorite" is irrelevant here - a bet is recommended on whichever SIDE
  (home or away) the model's own probability diverges enough from that
  side's real market price, in either direction.

**Real, current validation status**: game_evaluation.py's own
beat_closing_line_rate is NOT yet backed by a real statistically
meaningful sample (see the confidence banner this script always prints).
This script shows real, honestly-computed numbers - it is not a proven
betting strategy, and never claims to be.

Usage:
    python scripts/recommend_bets.py
    python scripts/recommend_bets.py --date 2026-08-24 --bankroll 1000
    python scripts/recommend_bets.py --kelly-fraction 0.25 --min-edge 0.03 --out bets.csv
"""

import argparse
import datetime
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import config, game_evaluation, kelly, market_odds, schedule

RECOMMENDATION_COLUMNS = [
    "date", "game_pk", "side", "team", "opponent", "moneyline",
    "model_probability", "market_implied_probability", "edge", "kelly_stake_fraction",
]


def _print_confidence_banner(log_path: str) -> None:
    """Always prints the real beat_closing_line_rate/n so far - real
    numbers, honestly labeled, not hidden and not oversold (same
    convention decision_theory.py's own streak=0 caveat already
    establishes for this project)."""
    if not os.path.exists(log_path):
        print("No game-predictions log found yet - nothing to base a confidence check on.")
        return
    log = pd.read_csv(log_path, parse_dates=["date"])
    _, summary = game_evaluation.build_game_picks_export(log)
    n = int(summary.loc[0, "n_beat_closing_line_compared"]) if not summary.empty else 0
    rate = summary.loc[0, "beat_closing_line_rate"] if not summary.empty else float("nan")
    rate_str = f"{rate:.1%}" if pd.notna(rate) else "n/a"

    # flush=True on every line here: this banner MUST appear before any
    # later SystemExit message (which Python writes straight to stderr,
    # unbuffered) in a real CI log - stdout is block-buffered when not
    # attached to a terminal, so without an explicit flush this banner
    # can print AFTER a later refusal message despite running first,
    # confirmed for real via a GitHub Actions dispatch (run 32679390305).
    print("=" * 72, flush=True)
    print(f"Real beat_closing_line_rate so far: {rate_str} (n={n} real market-compared games)", flush=True)
    if n < config.KELLY_MIN_GAMES_FOR_CONFIDENCE:
        print(
            f"WARNING: n={n} is well below a real statistically meaningful sample "
            f"(config.KELLY_MIN_GAMES_FOR_CONFIDENCE={config.KELLY_MIN_GAMES_FOR_CONFIDENCE}). "
            "The numbers below are real, honestly computed edges - NOT a validated betting "
            "strategy. Do not size real money off this until beat_closing_line_rate has real "
            "statistical power behind it.",
            flush=True,
        )
    print("=" * 72, flush=True)


def _load_target_date_picks(log_path: str, target_date: pd.Timestamp) -> pd.DataFrame:
    """Real, unresolved logged game picks for `target_date` - HARD refuses
    (raises SystemExit) rather than silently falling back to a different
    date or a stale one, since this script's output is literally an
    instruction to place real money. Same "safe failure mode, not a
    wrong-data one" philosophy game_predictions.resolve_game_predictions
    already applies to its own ambiguous-status games."""
    if not os.path.exists(log_path):
        raise SystemExit(f"{log_path} does not exist - run the daily pipeline first.")

    log = pd.read_csv(log_path, parse_dates=["date"])
    day = log[(log["date"] == target_date) & (log["metric"] == "GamePick_Win_Probability")]
    if day.empty:
        raise SystemExit(
            f"No game picks logged for {target_date.date()} - run the daily pipeline for "
            f"this date first rather than betting on a stale or guessed date."
        )

    pending = day[day["game_played"].isna()]
    if pending.empty:
        raise SystemExit(f"Every logged game pick for {target_date.date()} is already resolved - nothing to bet on.")
    return pending


def _real_game_statuses(target_date) -> dict:
    """game_pk -> real MLB Stats API status for `target_date`, via
    schedule.fetch_todays_games - a second, independent, cheap guard
    against ever recommending a stake on a game that's already started or
    finished. Both game_predictions.csv's game_pk and this function's
    game_pk come from the same MLB Stats API source, so matching on it
    here is exact (unlike matching against ESPN's market data, which
    needs the team-abbreviation crosswalk in market_odds.py)."""
    games = schedule.fetch_todays_games(target_date)
    return dict(zip(games["game_pk"], games["status"]))


def build_bet_recommendations(
    todays_picks: pd.DataFrame,
    market: pd.DataFrame,
    kelly_fraction_multiplier: float = config.KELLY_FRACTION_MULTIPLIER,
    min_edge: float = config.KELLY_MIN_EDGE,
) -> pd.DataFrame:
    """Pure, testable core - no network. For each logged game, derives the
    model's real probability for BOTH sides from its own logged
    predicted_winner/predicted_probability (predicted_probability is
    always the model's OWN favored side's probability, >= 0.5 by
    construction - the other side's model probability is the complement).
    Compares each side against the REAL vigged market price
    (market_odds.moneyline_to_implied_probability on the real raw
    moneyline) - deliberately NOT the de-vigged market_home_win_probability
    column, which exists for a different purpose (fairly scoring forecast
    skill with the book's own edge removed). A real bet is paid off at
    the real, vigged price, so Kelly's net-odds term has to come from
    that same real price too. Where edge >= min_edge, sizes a stake via
    kelly.kelly_fraction.

    Provably never recommends both sides of one game when min_edge > 0:
    edge_home + edge_away = 1 - (home_implied + away_implied) = -vig, and
    a real book's implied probabilities always sum to > 1 (see
    market_odds.devig's own test), so the two edges can never both be
    positive at once. Kept as a defensive check anyway, not silently
    trusted - if a row's real data is malformed enough that both sides
    somehow clear min_edge together, that's a real data-quality anomaly
    (e.g. a stale matching collision), not a real arbitrage in this
    pipeline's actual data - both sides are dropped for that game rather
    than either one being guessed at."""
    merged = todays_picks.merge(market, on=["home_team", "away_team"], how="left")

    rows = []
    for _, pick in merged.iterrows():
        if pd.isna(pick.get("home_moneyline")) or pd.isna(pick.get("away_moneyline")):
            continue  # no real market data for this game - skip, not fatal

        home_favored = pick["predicted_winner"] == pick["home_team"]
        home_model_probability = pick["predicted_probability"] if home_favored else 1 - pick["predicted_probability"]
        away_model_probability = 1 - home_model_probability

        sides = [
            ("home", pick["home_team"], pick["away_team"], home_model_probability, pick["home_moneyline"]),
            ("away", pick["away_team"], pick["home_team"], away_model_probability, pick["away_moneyline"]),
        ]

        candidates = []
        for side, team, opponent, model_probability, moneyline in sides:
            market_implied = market_odds.moneyline_to_implied_probability(moneyline)
            edge = model_probability - market_implied
            if edge >= min_edge:
                stake_fraction = kelly.kelly_fraction(model_probability, moneyline, kelly_fraction_multiplier)
                candidates.append({
                    "date": pick["date"], "game_pk": pick["game_pk"], "side": side,
                    "team": team, "opponent": opponent, "moneyline": moneyline,
                    "model_probability": model_probability, "market_implied_probability": market_implied,
                    "edge": edge, "kelly_stake_fraction": stake_fraction,
                })

        if len(candidates) > 1:
            print(
                f"WARNING: both sides of game_pk={pick['game_pk']} cleared min_edge "
                f"simultaneously - a real data-quality anomaly (see build_bet_recommendations' "
                f"own docstring), not a real arbitrage. Dropping both sides for this game."
            )
            continue
        rows.extend(candidates)

    return pd.DataFrame(rows, columns=RECOMMENDATION_COLUMNS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--log-path", default="data/predictions/game_predictions.csv")
    parser.add_argument("--date", type=datetime.date.fromisoformat, default=schedule.today_local())
    parser.add_argument("--bankroll", type=float, default=None)
    parser.add_argument("--kelly-fraction", type=float, default=config.KELLY_FRACTION_MULTIPLIER)
    parser.add_argument("--min-edge", type=float, default=config.KELLY_MIN_EDGE)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    _print_confidence_banner(args.log_path)

    target_date = pd.Timestamp(args.date)
    todays_picks = _load_target_date_picks(args.log_path, target_date)

    real_statuses = _real_game_statuses(args.date)
    scheduled_mask = todays_picks["game_pk"].map(real_statuses) == "Scheduled"
    not_scheduled = todays_picks[~scheduled_mask]
    if not not_scheduled.empty:
        print(
            f"Skipping {len(not_scheduled)} game(s) whose real MLB Stats API status isn't "
            f"'Scheduled' (already started/finished, or not found for {target_date.date()}) - "
            f"never bet on those."
        )
    todays_picks = todays_picks[scheduled_mask]

    if todays_picks.empty:
        print(f"No real still-scheduled games left to evaluate for {target_date.date()}.")
        return

    market = market_odds.fetch_market_home_win_probabilities(target_date)
    recommendations = build_bet_recommendations(todays_picks, market, args.kelly_fraction, args.min_edge)

    if recommendations.empty:
        print(
            f"No qualifying edge found for {target_date.date()} - no bets recommended. "
            f"This is a real, expected outcome (most days should have none), not a bug."
        )
        return

    if args.bankroll is not None:
        recommendations["recommended_stake_dollars"] = recommendations["kelly_stake_fraction"] * args.bankroll

    print(f"\nReal bet recommendations for {target_date.date()}:")
    print(recommendations.to_string(index=False))

    if args.out:
        recommendations.to_csv(args.out, index=False)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()

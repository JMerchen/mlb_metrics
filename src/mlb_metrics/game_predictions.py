"""Daily game-pick logging and outcome resolution - the game-level analog
of predictions.py. Kept as a separate module rather than extending
predictions.py: the entity (a game, keyed on game_pk) and its resolution
data source (final scores via schedule.fetch_game_results, not Statcast)
both differ enough from the batter-pick case that threading two entity
types through one module would hurt readability more than sharing code
would help.

Also owns `advise_bets` (real Kelly-edge bet-advice logic, formerly
scripts/recommend_bets.py's own private copy) - deliberately placed here
rather than in kelly.py (which stays pure scalar math, no pandas/config)
or a new module, since this IS the module that decides what goes into
each day's logged game-pick row, and bet advice is now one more real,
resolvable field on that same row (bet_advised/bet_side/bet_moneyline/
bet_stake_dollars/bet_profit_dollars) - not a separate concern with its
own lifecycle. scripts/recommend_bets.py calls this same function rather
than keeping its own copy, so there's exactly one real implementation of
"should a bet be advised on this game."
"""

import os

import pandas as pd

from mlb_metrics import config, kelly, market_odds

GAME_PREDICTION_COLUMNS = [
    "date", "game_pk", "home_team", "away_team", "predicted_winner",
    "predicted_probability", "above_threshold", "metric", "actual_winner", "game_played", "model_version",
    "market_home_win_probability",
    "bet_advised", "bet_side", "bet_team", "bet_moneyline", "bet_stake_fraction", "bet_stake_dollars",
    "bet_profit_dollars",
]

BET_ADVICE_COLUMNS = [
    "date", "game_pk", "side", "team", "opponent", "moneyline",
    "model_probability", "market_implied_probability", "edge", "kelly_stake_fraction",
]

# Same purpose as predictions.LEGACY_MODEL_VERSION - tagged onto any row
# logged before model_version existed, or reconstructed by a historical
# replay (see game_picks_backtest.py).
LEGACY_MODEL_VERSION = "legacy"


def advise_bets(
    todays_picks: pd.DataFrame,
    market: pd.DataFrame,
    kelly_fraction_multiplier: float,
    min_edge: float,
) -> pd.DataFrame:
    """Pure, testable core shared by both the live daily pipeline
    (select_game_picks below) and the standalone scripts/recommend_bets.py
    report - single source of truth for "should a real bet be advised on
    this game." No network. For each logged game, derives the model's real
    probability for BOTH sides from its own logged predicted_winner/
    predicted_probability (predicted_probability is always the model's OWN
    favored side's probability, >= 0.5 by construction - the other side's
    model probability is the complement). Compares each side against the
    REAL vigged market price (market_odds.moneyline_to_implied_probability
    on the real raw moneyline) - deliberately NOT the de-vigged
    market_home_win_probability column, which exists for a different
    purpose (fairly scoring forecast skill with the book's own edge
    removed). A real bet is paid off at the real, vigged price, so Kelly's
    net-odds term has to come from that same real price too. Where
    edge >= min_edge, sizes a stake via kelly.kelly_fraction.

    Provably never recommends both sides of one game when min_edge > 0:
    edge_home + edge_away = 1 - (home_implied + away_implied) = -vig, and a
    real book's implied probabilities always sum to > 1 (see
    market_odds.devig's own test), so the two edges can never both be
    positive at once. Kept as a defensive check anyway, not silently
    trusted - if a row's real data is malformed enough that both sides
    somehow clear min_edge together, that's a real data-quality anomaly
    (e.g. a stale matching collision), not a real arbitrage in this
    pipeline's actual data - both sides are dropped for that game rather
    than either one being guessed at. Returns at most one row per
    game_pk."""
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
                f"simultaneously - a real data-quality anomaly (see advise_bets' own "
                f"docstring), not a real arbitrage. Dropping both sides for this game."
            )
            continue
        rows.extend(candidates)

    return pd.DataFrame(rows, columns=BET_ADVICE_COLUMNS)


def select_game_picks(
    win_probabilities: pd.DataFrame,
    date,
    min_probability: float = config.GAME_PICK_MIN_PROBABILITY,
    metric: str = "GamePick_Win_Probability",
    model_version: str = config.GAME_PICK_MODEL_VERSION,
    market_probabilities: pd.DataFrame | None = None,
    kelly_fraction_multiplier: float = config.KELLY_FRACTION_MULTIPLIER,
    min_edge: float = config.KELLY_MIN_EDGE,
) -> pd.DataFrame:
    """Turn game_picks.compute_game_win_probabilities' output into the
    day's logged games - EVERY scheduled game, not just the ones that clear
    `min_probability`. `above_threshold` flags whether the favored side's
    win probability clears `min_probability` ("real separation between the
    two teams") - the dashboard still publishes the complete slate and
    highlights the flagged games, but game_evaluation.build_game_picks_export's
    real tracking is scoped to `bet_advised` (below), not `above_threshold` -
    a model being confident in its own favorite is a different question
    from the market actually disagreeing with it. `predicted_winner` is
    whichever side (home_team/away_team) has the higher probability;
    `predicted_probability` is that side's probability (always >= 0.5).
    `model_version` (see config.GAME_PICK_MODEL_VERSION) is stamped onto
    every row so game_evaluation.py/the dashboard can segment by which
    win-probability logic actually produced a pick - same reasoning as
    predictions.select_picks' own model_version.

    `market_probabilities` (default None - unchanged behavior for every
    existing caller/test, including game_picks_backtest.py's real calls
    that never pass one) is market_odds.fetch_market_home_win_probabilities'
    output, a real DataFrame keyed by (home_team, away_team) with a
    market_home_win_probability column - left-merged in when given, so a
    day with no market data (fetch failed, or the caller doesn't pass one)
    still logs picks normally with that column all-NaN.

    When `market_probabilities` also carries real `home_moneyline`/
    `away_moneyline` (the raw, vigged price - a defensive column check,
    not just an empty check, so any older caller/test still passing the
    original 3-column de-vigged-only frame keeps working with
    `bet_advised` simply staying False), calls `advise_bets` to decide
    whether a real bet is advised on each game and logs the result
    (`bet_advised`/`bet_side`/`bet_team`/`bet_moneyline`/`bet_stake_fraction`/
    `bet_stake_dollars` - `bet_profit_dollars` stays null here, filled in
    only once resolve_game_predictions later knows the real outcome).
    `bet_stake_dollars = bet_stake_fraction * config.KELLY_REFERENCE_BANKROLL`,
    a fixed NOTIONAL bankroll purely so logged dollar amounts are
    comparable day to day, not a live/compounding one (see
    config.KELLY_REFERENCE_BANKROLL's own docstring).

    advise_bets returns at most one row per game_pk in the normal case,
    but `market_probabilities` matches by (home_team, away_team) only (see
    market_odds.py's own doubleheader caveat) - a real doubleheader with
    two ESPN-reported events for the same two teams could in principle
    produce two advice rows for the same game_pk. Defended against
    explicitly here (not silently trusted): any game_pk appearing more
    than once in the advice is dropped from consideration entirely (with
    a warning) rather than picking one arbitrarily, so the final merge
    below is always safely one-to-one."""
    df = win_probabilities.copy()
    favors_home = df["home_win_probability"] >= 0.5
    df["predicted_winner"] = df["home_team"].where(favors_home, df["away_team"])
    df["predicted_probability"] = df["home_win_probability"].where(favors_home, 1 - df["home_win_probability"])
    df["above_threshold"] = df["predicted_probability"] >= min_probability

    picks = df.copy()
    picks["date"] = pd.Timestamp(date)
    picks["metric"] = metric
    picks["actual_winner"] = pd.NA
    picks["game_played"] = pd.NA
    picks["model_version"] = model_version

    has_moneylines = (
        market_probabilities is not None
        and not market_probabilities.empty
        and {"home_moneyline", "away_moneyline"}.issubset(market_probabilities.columns)
    )

    if market_probabilities is not None and not market_probabilities.empty:
        picks = picks.merge(
            market_probabilities[["home_team", "away_team", "market_home_win_probability"]],
            on=["home_team", "away_team"],
            how="left",
        )
    else:
        picks["market_home_win_probability"] = pd.NA

    if has_moneylines:
        advice = advise_bets(picks, market_probabilities, kelly_fraction_multiplier, min_edge)
        dupe_game_pks = advice.loc[advice["game_pk"].duplicated(keep=False), "game_pk"].unique()
        if len(dupe_game_pks) > 0:
            print(
                f"WARNING: advise_bets returned more than one row for game_pk(s) "
                f"{list(dupe_game_pks)} - a real doubleheader/matching collision (see "
                f"market_odds.py's own doubleheader caveat), not a data error to guess "
                f"through. Dropping those games from bet advice entirely."
            )
            advice = advice[~advice["game_pk"].isin(dupe_game_pks)]

        picks = picks.merge(
            advice[["game_pk", "side", "team", "moneyline", "kelly_stake_fraction"]].rename(
                columns={
                    "side": "bet_side", "team": "bet_team",
                    "moneyline": "bet_moneyline", "kelly_stake_fraction": "bet_stake_fraction",
                }
            ),
            on="game_pk",
            how="left",
        )
        picks["bet_advised"] = picks["game_pk"].isin(advice["game_pk"])
        picks["bet_stake_dollars"] = picks["bet_stake_fraction"] * config.KELLY_REFERENCE_BANKROLL
    else:
        picks["bet_advised"] = False
        picks["bet_side"] = pd.NA
        picks["bet_team"] = pd.NA
        picks["bet_moneyline"] = pd.NA
        picks["bet_stake_fraction"] = pd.NA
        picks["bet_stake_dollars"] = pd.NA

    picks["bet_profit_dollars"] = pd.NA

    return picks[GAME_PREDICTION_COLUMNS]


def append_game_predictions(picks: pd.DataFrame, log_path: str) -> pd.DataFrame:
    """Append `picks` to the game-predictions log at `log_path`, deduping
    on (date, game_pk, metric) so a re-run doesn't create duplicate log
    entries. Existing rows (including already-resolved outcomes) always win
    over a re-logged pick for the same key."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path, parse_dates=["date"])
        if "model_version" not in existing.columns:
            existing["model_version"] = LEGACY_MODEL_VERSION  # migrate a log written before model_version existed
        if "above_threshold" not in existing.columns:
            # Every row logged before this column existed already cleared
            # the old hard filter select_game_picks used to apply - True is
            # the factually correct backfill, not an arbitrary default.
            existing["above_threshold"] = True
        if "market_home_win_probability" not in existing.columns:
            # A log written before slice 2 genuinely has no real market
            # data for those rows - NaN, not a guess.
            existing["market_home_win_probability"] = pd.NA
        if "bet_advised" not in existing.columns:
            # A row logged before bet advice existed genuinely never had a
            # bet advised - False is the factually correct backfill, not
            # an arbitrary default (same reasoning as above_threshold's
            # own backfill above).
            existing["bet_advised"] = False
            for col in ("bet_side", "bet_team", "bet_moneyline", "bet_stake_fraction",
                        "bet_stake_dollars", "bet_profit_dollars"):
                existing[col] = pd.NA
        combined = pd.concat([picks, existing], ignore_index=True)
    else:
        combined = picks

    combined = combined.drop_duplicates(subset=["date", "game_pk", "metric"], keep="last")
    combined = combined.sort_values(["date", "game_pk"]).reset_index(drop=True)
    combined.to_csv(log_path, index=False)
    return combined


def resolve_game_predictions(log_path: str, fetch_results_fn, as_of_date) -> pd.DataFrame:
    """Fill in `actual_winner`/`game_played` for any still-pending rows
    (`game_played` is null) with a date strictly before `as_of_date`, by
    calling `fetch_results_fn(date)` (i.e. schedule.fetch_game_results) once
    per distinct pending date. `as_of_date` is explicit rather than reading
    the wall clock - same no-implicit-"now" philosophy as pipeline.run()
    itself (see its module docstring), and it keeps this function's
    behavior fully determined by its inputs, not by when it happens to run.
    Unlike Statcast (one bulk range fetch), statsapi has no bulk date-range
    endpoint, so this is necessarily N fetches for N distinct pending dates
    - each one is try/except-guarded so one bad/unreachable date can't
    block resolving the others or raise out of this function.

    Only a `status == "Final"` game is resolved (winner = whichever side
    scored more). Any other status (Scheduled, In Progress, postponed,
    etc.) is left pending - the exact status strings MLB Stats API uses for
    a postponed/cancelled game aren't confirmed in this codebase, so rather
    than guess at them and risk mis-resolving a game, those picks simply
    stay pending indefinitely in this first pass (a safe failure mode, not
    a wrong-data one)."""
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=GAME_PREDICTION_COLUMNS)

    log = pd.read_csv(log_path, parse_dates=["date"])
    if log.empty:
        return log
    if "game_played" not in log.columns:
        log["game_played"] = pd.NA  # migrate a log written before game_played existed
    if "model_version" not in log.columns:
        log["model_version"] = LEGACY_MODEL_VERSION  # migrate a log written before model_version existed
    if "above_threshold" not in log.columns:
        log["above_threshold"] = True  # migrate a log written before above_threshold existed (see append_game_predictions)
    if "market_home_win_probability" not in log.columns:
        log["market_home_win_probability"] = pd.NA  # migrate a log written before slice 2's market column existed
    if "bet_advised" not in log.columns:
        # A row logged before bet advice existed genuinely never had a bet
        # advised - False is the factually correct backfill, not an
        # arbitrary default (see append_game_predictions's identical guard).
        log["bet_advised"] = False
        for col in ("bet_side", "bet_team", "bet_moneyline", "bet_stake_fraction",
                    "bet_stake_dollars", "bet_profit_dollars"):
            log[col] = pd.NA
    # A log with no resolved games yet round-trips actual_winner as an
    # all-null float64 column (empty strings -> NaN on read) - cast back to
    # object so assigning a team abbreviation string into it doesn't raise.
    log["actual_winner"] = log["actual_winner"].astype("object")

    pending_dates = log.loc[
        log["game_played"].isna() & (log["date"] < pd.Timestamp(as_of_date)), "date"
    ].unique()

    for date in pending_dates:
        date = pd.Timestamp(date)
        try:
            results = fetch_results_fn(date.date())
        except Exception as exc:
            print(
                f"WARNING: failed to fetch game results for {date.date()} ({exc}); "
                f"leaving that date's game picks pending."
            )
            continue
        if results.empty:
            continue

        results = results[["game_pk", "status", "home_score", "away_score"]].rename(
            columns={"status": "_status", "home_score": "_home_score", "away_score": "_away_score"}
        )
        merged = log.merge(results, on="game_pk", how="left")

        day_mask = (log["date"] == date) & log["game_played"].isna()
        final = day_mask & (merged["_status"] == "Final")
        home_won = final & (merged["_home_score"] > merged["_away_score"])
        away_won = final & (merged["_away_score"] > merged["_home_score"])

        log.loc[final, "game_played"] = 1
        log.loc[home_won, "actual_winner"] = log.loc[home_won, "home_team"]
        log.loc[away_won, "actual_winner"] = log.loc[away_won, "away_team"]

        # Real money won/lost, only for newly-finalized rows where a bet
        # was actually advised (bet_advised is never NaN - always True or
        # False - but == True side-steps any object-dtype ambiguity after
        # a real CSV round-trip, same convention this project's own tests
        # already use for boolean columns read back from CSV).
        advised_final = final & (log["bet_advised"] == True)  # noqa: E712
        team_won = (home_won & (log["bet_team"] == log["home_team"])) | (
            away_won & (log["bet_team"] == log["away_team"])
        )
        bet_won = advised_final & team_won
        bet_lost = advised_final & ~team_won
        log.loc[bet_won, "bet_profit_dollars"] = log.loc[bet_won, "bet_stake_dollars"] * log.loc[
            bet_won, "bet_moneyline"
        ].apply(kelly.moneyline_to_net_odds)
        log.loc[bet_lost, "bet_profit_dollars"] = -log.loc[bet_lost, "bet_stake_dollars"]

    log.to_csv(log_path, index=False)
    return log

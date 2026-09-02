"""Weekly NFL game-pick logging and outcome resolution - the NFL analog of
game_predictions.py (see that module's own docstring for the full
reasoning behind logging in UNITS, bet_units as the single "was a bet
advised" signal, etc. - not repeated here). Real NFL games happen roughly
weekly rather than daily, so this logs by REAL calendar `date`
(`schedules_*.parquet`'s own `gameday`) plus explicit `season`/`week`
columns, keyed on `game_id` (nflreadpy's own real game id, e.g.
"2025_01_DAL_PHI") rather than MLB's `game_pk`.

Uses its OWN separate log (data/predictions/nfl_game_predictions.csv - a
literal path default, same convention scripts/recommend_bets.py/
run_game_picks_backtest.py already use for MLB's own log rather than a
config constant) - never mixed with MLB's - and its
own calibration/model-version constants (nfl_game_picks.apply_calibration/
config.NFL_GAME_PICK_MODEL_VERSION). Reuses kelly.py/config.KELLY_*
completely UNCHANGED (kelly.py has zero MLB-specific logic - pure scalar
math), including config.KELLY_DAILY_UNIT_CAP/KELLY_MAX_SINGLE_BET_UNIT_CAP -
a deliberate, non-hidden design choice: these caps apply PER SPORT (MLB's
own daily log and this one are sized independently, never pooled), same
as the approved plan states explicitly.

Unlike MLB's resolve_game_predictions (which needs a per-date statsapi
fetch loop - MLB Stats API has no bulk date-range results endpoint), a
single already-fetched `schedules_df` slice covers an ENTIRE season's real
final scores at once (nfl_data.fetch_schedules), so resolving pending NFL
picks here is a single real DataFrame match on `game_id`, no fetch
function/as_of_date indirection needed.
"""

import os

import pandas as pd

from mlb_metrics import config, game_picks, kelly, market_odds

NFL_GAME_PREDICTION_COLUMNS = [
    "date", "season", "week", "game_id", "home_team", "away_team", "predicted_winner",
    "predicted_probability", "above_threshold", "metric", "actual_winner", "game_played", "model_version",
    "market_home_win_probability",
    "bet_units", "bet_side", "bet_team", "bet_moneyline", "bet_stake_fraction", "bet_profit_units",
    "home_win_probability_pessimistic", "away_win_probability_pessimistic",
]

NFL_BET_ADVICE_COLUMNS = [
    "date", "game_id", "side", "team", "opponent", "moneyline",
    "model_probability", "market_implied_probability", "edge", "kelly_stake_fraction",
]


def advise_bets(
    todays_picks: pd.DataFrame,
    market: pd.DataFrame,
    kelly_fraction_multiplier: float,
    min_edge: float,
) -> pd.DataFrame:
    """Direct structural mirror of game_predictions.advise_bets (see that
    function's own docstring for the full reasoning - real vigged-price
    edge detection, uncertainty-scaled sizing when a pessimistic
    probability is present, the per-bet/daily unit caps, the "both sides
    cleared min_edge" defensive drop), with `game_id` in place of
    `game_pk` throughout. Returns at most one row per game_id."""
    merged = todays_picks.merge(market, on=["home_team", "away_team"], how="left")

    rows = []
    for _, pick in merged.iterrows():
        if pd.isna(pick.get("home_moneyline")) or pd.isna(pick.get("away_moneyline")):
            continue  # no real market data for this game - skip, not fatal

        home_favored = pick["predicted_winner"] == pick["home_team"]
        home_model_probability = pick["predicted_probability"] if home_favored else 1 - pick["predicted_probability"]
        away_model_probability = 1 - home_model_probability

        home_pessimistic = pick.get("home_win_probability_pessimistic")
        away_pessimistic = pick.get("away_win_probability_pessimistic")
        if pd.notna(home_pessimistic) and pd.notna(away_pessimistic):
            home_sizing_probability = home_pessimistic
            away_sizing_probability = away_pessimistic
        else:
            home_sizing_probability = home_model_probability
            away_sizing_probability = away_model_probability

        sides = [
            ("home", pick["home_team"], pick["away_team"], home_model_probability, home_sizing_probability, pick["home_moneyline"]),
            ("away", pick["away_team"], pick["home_team"], away_model_probability, away_sizing_probability, pick["away_moneyline"]),
        ]

        candidates = []
        for side, team, opponent, model_probability, sizing_probability, moneyline in sides:
            market_implied = market_odds.moneyline_to_implied_probability(moneyline)
            edge = model_probability - market_implied
            if edge >= min_edge:
                stake_fraction = kelly.kelly_fraction(sizing_probability, moneyline, kelly_fraction_multiplier)
                candidates.append({
                    "date": pick["date"], "game_id": pick["game_id"], "side": side,
                    "team": team, "opponent": opponent, "moneyline": moneyline,
                    "model_probability": model_probability, "market_implied_probability": market_implied,
                    "edge": edge, "kelly_stake_fraction": stake_fraction,
                })

        if len(candidates) > 1:
            print(
                f"WARNING: both sides of game_id={pick['game_id']} cleared min_edge "
                f"simultaneously - a real data-quality anomaly (see advise_bets' own "
                f"docstring), not a real arbitrage. Dropping both sides for this game."
            )
            continue
        rows.extend(candidates)

    result = pd.DataFrame(rows, columns=NFL_BET_ADVICE_COLUMNS)
    if result.empty:
        return result

    single_bet_cap_fraction = config.KELLY_MAX_SINGLE_BET_UNIT_CAP * config.UNIT_SIZE_FRACTION
    result["kelly_stake_fraction"] = result["kelly_stake_fraction"].clip(upper=single_bet_cap_fraction)

    daily_cap_fraction = config.KELLY_DAILY_UNIT_CAP * config.UNIT_SIZE_FRACTION
    daily_total = result.groupby("date")["kelly_stake_fraction"].transform("sum")
    over_cap = daily_total > daily_cap_fraction
    result.loc[over_cap, "kelly_stake_fraction"] = (
        result.loc[over_cap, "kelly_stake_fraction"] * daily_cap_fraction / daily_total[over_cap]
    )
    return result


def select_game_picks(
    win_probabilities: pd.DataFrame,
    schedule_games_df: pd.DataFrame,
    min_probability: float = config.NFL_GAME_PICK_MIN_PROBABILITY,
    metric: str = "NFL_GamePick_Win_Probability",
    model_version: str = config.NFL_GAME_PICK_MODEL_VERSION,
    market_probabilities: pd.DataFrame | None = None,
    kelly_fraction_multiplier: float = config.KELLY_FRACTION_MULTIPLIER,
    min_edge: float = config.KELLY_MIN_EDGE,
    confidence: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Turn nfl_game_picks.compute_game_win_probabilities' output into a
    week's logged games - direct structural mirror of
    game_predictions.select_game_picks (see that function's own docstring
    for the full reasoning behind every column/branch below). `schedule_games_df`
    supplies the real `date`(`gameday`)/`season`/`week` columns this NFL
    log carries that MLB's own doesn't need (`win_probabilities` itself
    only carries [game_id, season, week, home_team, away_team,
    home_win_probability] - see nfl_game_picks.compute_game_win_probabilities).

    `market_probabilities` (default None) is expected to carry real
    `home_team`/`away_team`/`home_moneyline`/`away_moneyline` (and,
    optionally, a de-vigged `market_home_win_probability` - the same
    real schedules_*.parquet slice covers both, unlike MLB which needed a
    separate market_odds.py scrape). `confidence` (default None) is
    nfl_team_strength.assemble_team_metrics' output, carrying real
    win_rate_CI_Low/CI_High for game_picks.apply_kelly_uncertainty's
    uncertainty-scaled sizing (reused directly - see that function's own
    docstring, it's already sport-agnostic)."""
    df = win_probabilities.copy()
    favors_home = df["home_win_probability"] >= 0.5
    df["predicted_winner"] = df["home_team"].where(favors_home, df["away_team"])
    df["predicted_probability"] = df["home_win_probability"].where(favors_home, 1 - df["home_win_probability"])
    df["above_threshold"] = df["predicted_probability"] >= min_probability

    picks = df.merge(
        schedule_games_df[["game_id", "gameday"]].rename(columns={"gameday": "date"}), on="game_id", how="left"
    )
    picks["date"] = pd.to_datetime(picks["date"])
    picks["metric"] = metric
    picks["actual_winner"] = pd.NA
    picks["game_played"] = pd.NA
    picks["model_version"] = model_version

    has_moneylines = (
        market_probabilities is not None
        and not market_probabilities.empty
        and {"home_moneyline", "away_moneyline"}.issubset(market_probabilities.columns)
    )

    if market_probabilities is not None and not market_probabilities.empty and "market_home_win_probability" in market_probabilities.columns:
        picks = picks.merge(
            market_probabilities[["home_team", "away_team", "market_home_win_probability"]],
            on=["home_team", "away_team"],
            how="left",
        )
    else:
        picks["market_home_win_probability"] = pd.NA

    if confidence is not None and not confidence.empty:
        # game_picks.apply_kelly_uncertainty is fully generic (only
        # references home_team/away_team/win_rate_CI_Low/CI_High/
        # home_win_probability, all present here too) - reused unchanged
        # rather than duplicated, matching this project's own "reuse the
        # genuinely sport-agnostic math directly" precedent (kelly.py,
        # market_odds.py, teams.compute_team_win_rate_ci).
        pessimistic = game_picks.apply_kelly_uncertainty(win_probabilities, confidence)
        picks = picks.merge(
            pessimistic[["game_id", "home_win_probability_pessimistic", "away_win_probability_pessimistic"]],
            on="game_id",
            how="left",
        )
    else:
        picks["home_win_probability_pessimistic"] = pd.NA
        picks["away_win_probability_pessimistic"] = pd.NA

    if has_moneylines:
        advice = advise_bets(picks, market_probabilities, kelly_fraction_multiplier, min_edge)
        dupe_game_ids = advice.loc[advice["game_id"].duplicated(keep=False), "game_id"].unique()
        if len(dupe_game_ids) > 0:
            print(
                f"WARNING: advise_bets returned more than one row for game_id(s) "
                f"{list(dupe_game_ids)} - a real data-quality anomaly, not a data error to "
                f"guess through. Dropping those games from bet advice entirely."
            )
            advice = advice[~advice["game_id"].isin(dupe_game_ids)]

        picks = picks.merge(
            advice[["game_id", "side", "team", "moneyline", "kelly_stake_fraction"]].rename(
                columns={
                    "side": "bet_side", "team": "bet_team",
                    "moneyline": "bet_moneyline", "kelly_stake_fraction": "bet_stake_fraction",
                }
            ),
            on="game_id",
            how="left",
        )
        picks["bet_units"] = (picks["bet_stake_fraction"] / config.UNIT_SIZE_FRACTION).fillna(0.0)
    else:
        picks["bet_units"] = 0.0
        picks["bet_side"] = pd.NA
        picks["bet_team"] = pd.NA
        picks["bet_moneyline"] = pd.NA
        picks["bet_stake_fraction"] = pd.NA

    picks["bet_profit_units"] = pd.NA

    return picks[NFL_GAME_PREDICTION_COLUMNS]


def append_game_predictions(picks: pd.DataFrame, log_path: str) -> pd.DataFrame:
    """Append `picks` to the NFL game-predictions log at `log_path`,
    deduping on (game_id, metric) - direct mirror of
    game_predictions.append_game_predictions, with `game_id` in place of
    (date, game_pk) as the real dedup key (a game_id is already globally
    unique across a season - no separate date component needed the way
    MLB's game_pk, which can repeat across seasons, requires)."""
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)

    if os.path.exists(log_path):
        existing = pd.read_csv(log_path, parse_dates=["date"])
        combined = pd.concat([picks, existing], ignore_index=True)
    else:
        combined = picks

    combined = combined.drop_duplicates(subset=["game_id", "metric"], keep="last")
    combined = combined.sort_values(["date", "game_id"]).reset_index(drop=True)
    combined.to_csv(log_path, index=False)
    return combined


def resolve_game_predictions(log_path: str, schedules_df: pd.DataFrame) -> pd.DataFrame:
    """Fills in `actual_winner`/`game_played` for any still-pending rows
    (`game_played` null) by matching against `schedules_df`'s OWN real
    `home_score`/`away_score` for that `game_id`. Unlike MLB's
    resolve_game_predictions (a per-pending-date statsapi fetch loop -
    MLB Stats API has no bulk date-range results endpoint), a single
    already-fetched `schedules_df` (nfl_data.fetch_schedules) covers
    however many games are pending at once - no fetch function/as_of_date
    indirection needed here."""
    if not os.path.exists(log_path):
        return pd.DataFrame(columns=NFL_GAME_PREDICTION_COLUMNS)

    log = pd.read_csv(log_path, parse_dates=["date"])
    if log.empty:
        return log
    # A log with no resolved games yet round-trips actual_winner as an
    # all-null float64 column (empty strings -> NaN on read) - cast back
    # to object so assigning a team abbreviation string into it doesn't raise.
    log["actual_winner"] = log["actual_winner"].astype("object")

    results = schedules_df[["game_id", "home_score", "away_score"]].rename(
        columns={"home_score": "_home_score", "away_score": "_away_score"}
    )
    merged = log.merge(results, on="game_id", how="left")

    pending = log["game_played"].isna()
    final = pending & merged["_home_score"].notna() & merged["_away_score"].notna()
    home_won = final & (merged["_home_score"] > merged["_away_score"])
    away_won = final & (merged["_away_score"] > merged["_home_score"])

    log.loc[final, "game_played"] = 1
    log.loc[home_won, "actual_winner"] = log.loc[home_won, "home_team"]
    log.loc[away_won, "actual_winner"] = log.loc[away_won, "away_team"]

    # Real units won/lost, only for newly-finalized rows where a real bet
    # was advised (bet_units > 0 - the single "was a bet advised" signal).
    advised_final = final & (pd.to_numeric(log["bet_units"], errors="coerce") > 0)
    team_won = (home_won & (log["bet_team"] == log["home_team"])) | (
        away_won & (log["bet_team"] == log["away_team"])
    )
    bet_won = advised_final & team_won
    bet_lost = advised_final & ~team_won
    log.loc[bet_won, "bet_profit_units"] = log.loc[bet_won, "bet_units"] * log.loc[
        bet_won, "bet_moneyline"
    ].apply(kelly.moneyline_to_net_odds)
    log.loc[bet_lost, "bet_profit_units"] = -log.loc[bet_lost, "bet_units"]

    log.to_csv(log_path, index=False)
    return log

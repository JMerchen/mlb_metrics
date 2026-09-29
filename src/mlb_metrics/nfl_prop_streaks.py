"""Players who have cleared the same prop line in EVERY game of the
season so far, ranked by how likely they are to do it again.

Direct user request (2026-09-29): "a section of bets that have hit every
week of that season to that point... if a tight end has had 3+ receptions
every week, that might be listed... just listing the ten that have the
best odds of happening... something with odds of -1100 probably shouldn't
show, because winning means very little and losing means losing money."

THE STREAK IS THE SCREEN, NOT THE SIGNAL, and that distinction is the
whole design. A perfect season-to-date streak sounds like strong
evidence and mostly is not, because the number of games is small and the
number of players is large. Measured on the live 2026 board with two
weeks played: 269 skill players have a full game log, and 59 of them
have cleared 3+ receptions every single week. But a player whose true
weekly rate is 70% runs a perfect two-game streak 49% of the time, and
one at 60% does it 36% of the time. Most of that list is the binomial
being itself.

So the streak decides who is ELIGIBLE - it guarantees a player has
actually been doing the thing, with no bad weeks hiding inside an
average - and `hit_probability` decides the ORDER. That probability
comes from the same projection the main board uses (usage share x team
volume x shrunk per-play efficiency x opponent adjustment) pushed
through `prop_market.probability_over`, so it reflects the player's
whole history and this week's matchup rather than two lucky games. A
ranking built on streak length instead would be a ranking of luck.

THE LINE each player is measured at is the highest standard prop number
they have cleared in every game - their season floor, rounded down to a
real line. A receiver whose worst week was 12 yards carries a 9.5 line,
not his average. That choice is what makes the list mean "this has
literally never failed", and it also self-regulates the odds: a line
sitting at a player's floor lands in a bettable probability range rather
than the 99% a line at his average would imply.

THE JUICE CAP is the user's own stated constraint and is a judgment
call, not a measured optimum. `config.NFL_STREAK_PROBABILITY_BAND` keeps
the board between roughly -122 and -300 in fair-odds terms. Without it
the list fills with exactly what the request asked to exclude: capping
at 85% instead of 75% still produced a top ten priced around -550, where
five wins are needed to cover one loss.

WHAT THIS IS NOT. There are still no sportsbook prop lines anywhere in
this project (see `prop_market`'s own docstring, and the 2026-09-17
confirmation that every odds domain is unreachable from this sandbox).
`fair_odds` is THIS MODEL's implied price, not a price anyone is
offering, and nothing here can tell you whether a book disagrees. The
request asked for "the ones where vegas might not agree"; that half is
not deliverable until a feed exists, and pretending otherwise by
comparing our number to a number we invented would be worse than saying
so. What the board does deliver is the other half: the repeat
performers, honestly priced by our own model, with the unpayable ones
filtered out.

SACKS ARE EXCLUDED. Turning a line into a probability needs the outcome
spread from `config.PROP_OUTCOME_SIGMA_K`, and Sacks has no fitted value
there because it has no projection to take residuals against (see
`nfl_prop_projections.compute_sacks_allowed_per_game`). A guessed spread
would produce confident-looking probabilities backed by nothing.
"""

import math
import os

import numpy as np
import pandas as pd

from mlb_metrics import config, prop_market


# (weekly stat column, the increment real books post lines at) per
# category. Receptions move in whole numbers so the lines sit at k-0.5;
# yardage props are posted in 5-yard steps, so 4.5 / 9.5 / 14.5 and so
# on. Sacks is deliberately absent - see the module docstring.
STREAK_CATEGORY_LINES = {
    "Receptions": ("receptions", 1.0),
    "Receiving Yards": ("receiving_yards", 5.0),
    "Rushing Yards": ("rushing_yards", 5.0),
    "Passing Yards": ("passing_yards", 5.0),
}

STREAK_COLUMNS = [
    "player_id", "player_name", "team", "opponent", "game", "category",
    "line", "games_played", "season_low", "projection",
    "hit_probability", "fair_odds",
]


def highest_line_always_cleared(season_low: float, step: float) -> float:
    """The largest standard prop line strictly below `season_low`.

    Lines sit at `k * step - 0.5`, so a receiver whose worst game was 12
    yards carries 9.5 (he cleared it) rather than 14.5 (he did not), and
    a back whose worst game was exactly 4 receptions carries 3.5. Using
    the season LOW rather than the average is what lets the board claim
    the line has never failed.

    Returns 0.0 when no positive line qualifies - a player who was held
    scoreless in any game has no streak at any line.
    """
    if not np.isfinite(season_low) or season_low <= 0 or step <= 0:
        return 0.0
    steps = math.ceil((season_low + 0.5) / step) - 1
    line = steps * step - 0.5
    return float(line) if line > 0 else 0.0


def probability_to_american_odds(probability: float) -> float:
    """Fair American odds for a probability, with no vig applied.

    Deliberately NOT called a price: this is what the model thinks the
    bet is worth, not what any book is offering. A -300 here means the
    model gives it a 75% chance, so three wins are needed to cover one
    loss - which is the arithmetic the juice cap exists to keep in view.
    """
    probability = min(max(float(probability), 1e-6), 1 - 1e-6)
    if probability >= 0.5:
        return -round(100.0 * probability / (1.0 - probability))
    return round(100.0 * (1.0 - probability) / probability)


def compute_season_streaks(weekly_df: pd.DataFrame, season: int) -> pd.DataFrame:
    """One row per (player, category) for players who played EVERY week of
    `season` so far, carrying the highest line they cleared in all of them.

    Requiring a full game log is literal to the request ("hit every week
    of that season to that point") and is stricter than it may sound: a
    player who missed week 1 is excluded entirely, however well he has
    played since. That is the honest reading - a streak with a gap in it
    is not a streak - but it does mean the board thins out after an
    injury week.
    """
    if weekly_df.empty:
        return pd.DataFrame(columns=["player_id", "category", "line", "games_played", "season_low"])

    current = weekly_df[weekly_df["season"] == season]
    if "season_type" in current.columns:
        current = current[current["season_type"] == "REG"]
    if current.empty:
        return pd.DataFrame(columns=["player_id", "category", "line", "games_played", "season_low"])

    weeks_played = current["week"].nunique()
    rows = []
    for category, (column, step) in STREAK_CATEGORY_LINES.items():
        if column not in current.columns:
            continue
        grouped = current.groupby("player_id").agg(
            games_played=("week", "nunique"), season_low=(column, "min")
        )
        # Only players present for every week so far - a missed week
        # breaks the streak rather than being skipped over.
        grouped = grouped[grouped["games_played"] == weeks_played]
        for player_id, row in grouped.iterrows():
            line = highest_line_always_cleared(row["season_low"], step)
            if line <= 0:
                continue
            rows.append({
                "player_id": player_id,
                "category": category,
                "line": line,
                "games_played": int(row["games_played"]),
                "season_low": float(row["season_low"]),
            })

    return pd.DataFrame(rows, columns=["player_id", "category", "line", "games_played", "season_low"])


def build_streak_board(
    edges_df: pd.DataFrame,
    weekly_df: pd.DataFrame,
    season: int,
    top_n: int = None,
    probability_band: tuple = None,
) -> pd.DataFrame:
    """The board: streak-eligible players ranked by how likely the model
    thinks they are to clear their line again, filtered to a payout band.

    Ranking on `hit_probability` rather than on streak length is the
    point - see the module docstring for why a two-game streak is mostly
    noise. The band drops both ends: prices so short that a single loss
    swallows many wins, and probabilities too low to call a repeat
    performance likely at all.
    """
    top_n = config.NFL_STREAK_TOP_N if top_n is None else top_n
    low, high = config.NFL_STREAK_PROBABILITY_BAND if probability_band is None else probability_band

    streaks = compute_season_streaks(weekly_df, season)
    if streaks.empty or edges_df.empty:
        return pd.DataFrame(columns=STREAK_COLUMNS)

    available = [c for c in ["player_id", "player_name", "team", "opponent", "game", "category", "projection"]
                 if c in edges_df.columns]
    board = streaks.merge(edges_df[available], on=["player_id", "category"], how="inner")
    board = board[board["projection"].notna() & (board["projection"] > 0)]
    if board.empty:
        return pd.DataFrame(columns=STREAK_COLUMNS)

    board["hit_probability"] = [
        float(prop_market.probability_over(projection, line, category))
        for projection, line, category in zip(board["projection"], board["line"], board["category"])
    ]
    # A category with no fitted outcome spread yields NaN rather than a
    # made-up probability; drop those instead of ranking on nothing.
    board = board[board["hit_probability"].notna()]
    board["fair_odds"] = [probability_to_american_odds(p) for p in board["hit_probability"]]

    board = board[(board["hit_probability"] >= low) & (board["hit_probability"] <= high)]
    if board.empty:
        return pd.DataFrame(columns=STREAK_COLUMNS)

    # Probability first; a longer streak breaks ties, because at equal
    # model confidence the player with more games behind him has the
    # better-evidenced floor.
    board = board.sort_values(
        ["hit_probability", "games_played"], ascending=[False, False]
    ).head(top_n)

    for column in STREAK_COLUMNS:
        if column not in board.columns:
            board[column] = pd.NA
    return board[STREAK_COLUMNS].reset_index(drop=True)


def write_streak_board_csv(
    edges_df: pd.DataFrame, weekly_df: pd.DataFrame, season: int, output_path: str
) -> pd.DataFrame:
    """Writes the streak board to `output_path`, leaving any prior file in
    place when nothing qualifies - the same "don't let one quiet week
    erase real history" posture the props board itself already takes.
    """
    board = build_streak_board(edges_df, weekly_df, season)
    if board.empty:
        print("No prop streaks qualified this week - writing nothing, leaving the prior CSV in place.")
        return board

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    board.to_csv(output_path, index=False)
    print(f"Wrote {output_path} ({len(board)} streak rows)")
    return board

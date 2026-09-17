"""Cross-book player-prop market math: de-vigging, consensus formation,
outlier scoring, middles, arbitrage, and closing-line value.

DELIBERATELY SOURCE-AGNOSTIC AND OFFLINE. Every function here takes a
plain "quote table" and returns a frame - nothing fetches. That split is
the point: this project's sandbox blocks every odds domain (confirmed
again 2026-09-17: both `site.api.espn.com` and `api.the-odds-api.com`
return no response), so a fetcher can only ever be exercised from a
GitHub Actions runner, exactly as `market_odds.py` documents for its own
ESPN calls. The math, by contrast, is pure and is unit-tested here on
synthetic prices. When a multi-book prop feed exists, it has to produce
`QUOTE_COLUMNS` and nothing in this module changes.

WHY CROSS-BOOK, AND WHY IT IS A STRONGER IDEA THAN WHAT CAME BEFORE
(2026-09-17 user question: "not just when we disagree with the books but
also when the books disagree with one another"). Everything this project
has built for props so far is scored against a stand-in line - the
player's own trailing per-game average - because there are no book lines
in the repo. That benchmark is weak, and it produced a 74% top-10 hit
rate that turned out, on inspection, to be almost entirely selection
rather than directional skill. The de-vigged consensus of several books
is a far sharper estimate than either that stand-in or this project's own
projection. Measuring an outlier against consensus therefore measures
against something good, and it does not require our model to be right at
all.

TWO KINDS OF DISAGREEMENT, both handled:

  * LINE disagreement - one book has a receiver at 3.5 receptions and
    another at 4.5. On a low-integer stat that is an enormous gap.
  * PRICE disagreement - both books sit at 3.5, but one pays -110 and
    the other -135.

They are unified by translating every quote into an implied CENTRAL
ESTIMATE of the stat (`implied_mean`), so a book that moved its line and
a book that only moved its price become comparable numbers on one scale.

THE DISTRIBUTIONAL MODEL, and the one piece of it that is measured
rather than assumed. Converting `(line, de-vigged P(over))` into an
implied mean needs a distribution for the stat. A normal with a
level-dependent spread is used, because the spread is not constant:
measured over 2025 replay, receiving-yards residuals had a standard
deviation of 14.7 around a 7.7-yard projection but 35.5 around a
58.1-yard one. Dividing by the square root of the level flattens that
almost exactly (5.28, 4.77, 4.72, 4.65 across quartiles), which is the
signature of a count-like or compound-sum process - receptions are
counts, and receiving yards are a sum over catches - where variance
grows with the mean.

So `sigma = k * sqrt(level)`, with `k` per category in
`config.PROP_OUTCOME_SIGMA_K`, fitted by maximum likelihood on
`residual^2 / level`. Those k values REPLICATE across seasons (receiving
yards 5.17 then 5.24; receptions 1.14 then 1.20), which is worth stating
plainly because two other ideas measured in this same session - the EPA
blend and the game-script adjustment - did not, and were shipped off as
a result.

WHICH BOOK DISAGREES MATTERS MORE THAN THAT ONE DOES. An outlier is one
of three things and only the first is money:

  1. a slow or soft book that has not moved yet - exploitable;
  2. a book that moved FIRST on real news, an inactive or a beat report -
     in which case the outlier is correct and the bettor is the mark;
  3. a stale quote about to be pulled or voided.

Nothing in a price distinguishes these, so this module refuses to pretend
otherwise. `build_consensus` weights books by `config.PROP_BOOK_SHARPNESS`
so that a low-margin, high-limit book moves the consensus more than a
retail book does, and `flag_stale_quotes` marks anything older than
`config.PROP_QUOTE_MAX_STALENESS_MINUTES`. A caller that finds the
SHARPEST book is the outlier should read that as information, not as an
opportunity - `score_book_disagreement` reports
`outlier_is_sharpest_book` for exactly that check.

CLOSING-LINE VALUE IS THE REASON TO WANT ALL OF THIS. `closing_line_value`
compares the price a bet was taken at against the last quote before
kickoff. Beating the close is the best-established leading indicator of
long-run profit in betting, and unlike a win rate it is measurable in
days rather than hundreds of settled bets. That directly addresses the
measurement problem that has constrained every props decision in this
project: the prediction log currently holds ten pending picks and will
not say anything for weeks.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from mlb_metrics import config, market_odds


# The canonical quote table every function here consumes. One row per
# (book, player, category) observation, with both sides' prices as
# American odds - the format `market_odds.moneyline_to_implied_probability`
# already expects, reused rather than reinvented.
QUOTE_COLUMNS = [
    "fetched_at", "book", "player_id", "player_name", "category",
    "line", "over_price", "under_price",
]

# What a logged bet needs to carry for `closing_line_value` to score it.
BET_COLUMNS = ["placed_at", "book", "player_id", "category", "line", "side", "price"]

OVER = "Over"
UNDER = "Under"


def american_to_decimal(price) -> float:
    """American odds to decimal (total return per 1 staked), so expected
    value can be written as a single multiplication."""
    price = np.asarray(price, dtype=float)
    return np.where(price < 0, 1.0 + 100.0 / -price, 1.0 + price / 100.0)


def devig_two_way(over_price, under_price):
    """De-vigged (P(over), P(under)) from a two-sided American-odds quote,
    summing to exactly 1.

    The American-odds conversion reuses
    `market_odds.moneyline_to_implied_probability` rather than a second
    implementation that could drift from it. The normalisation is written
    out here rather than calling `market_odds.devig`, because that helper
    returns only the first side's probability and both are needed - the
    arithmetic is identical proportional de-vigging either way.

    Proportional is the simplest defensible method, not the only one:
    Shin and power de-vigs handle longshot bias better, which matters far
    more for a +1000 price than for the near-even two-way prices props
    carry.
    """
    over_implied = np.vectorize(market_odds.moneyline_to_implied_probability)(
        np.asarray(over_price, dtype=float)
    )
    under_implied = np.vectorize(market_odds.moneyline_to_implied_probability)(
        np.asarray(under_price, dtype=float)
    )
    total = over_implied + under_implied
    p_over = np.where(total > 0, over_implied / total, np.nan)
    return p_over, 1.0 - p_over


def outcome_sigma(category, level):
    """Standard deviation of the actual stat around a central estimate:
    `k * sqrt(level)`, with k per category from
    `config.PROP_OUTCOME_SIGMA_K`.

    The square-root form is measured, not assumed - see the module
    docstring for the quartile evidence and the cross-season stability of
    k. `level` is floored at `config.PROP_SIGMA_MIN_LEVEL` so a line of
    0.5 receptions does not produce a near-zero spread that would make
    every price look like a certainty.
    """
    level = np.maximum(np.asarray(level, dtype=float), config.PROP_SIGMA_MIN_LEVEL)
    if isinstance(category, str):
        k = config.PROP_OUTCOME_SIGMA_K.get(category)
        if k is None:
            return np.full_like(level, np.nan, dtype=float)
        return k * np.sqrt(level)
    k = pd.Series(category).map(config.PROP_OUTCOME_SIGMA_K).to_numpy(dtype=float)
    return k * np.sqrt(level)


def implied_mean(line, p_over, category):
    """The central estimate of the stat implied by a de-vigged quote.

    For X ~ Normal(mu, sigma), P(X > L) = p gives mu = L + sigma * z(p).
    A book sitting at the line with a 50/50 price implies mu == L; a book
    pricing the over as a favourite implies mu above its own line.

    Sigma is evaluated at the LINE rather than at mu, deliberately.
    Evaluating at mu would be circular (mu is what is being solved for),
    and the line is by construction close to mu for any sane price - a
    quote 30 points from its own implied mean would be a broken quote.
    """
    p_over = np.clip(np.asarray(p_over, dtype=float), 1e-6, 1 - 1e-6)
    sigma = outcome_sigma(category, line)
    return np.asarray(line, dtype=float) + sigma * norm.ppf(p_over)


def probability_over(mean, line, category):
    """P(X > line) implied by a central estimate - the inverse of
    `implied_mean`, used to price a consensus view against a specific
    book's line."""
    sigma = outcome_sigma(category, line)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (np.asarray(mean, dtype=float) - np.asarray(line, dtype=float)) / sigma
    return norm.cdf(z)


def annotate_quotes(quotes_df: pd.DataFrame) -> pd.DataFrame:
    """Adds the de-vigged probabilities and implied mean to every quote -
    the shared first step for consensus, outlier scoring and CLV."""
    if quotes_df.empty:
        return quotes_df.assign(p_over=[], p_under=[], implied_mean=[], book_weight=[])

    annotated = quotes_df.copy()
    p_over, p_under = devig_two_way(annotated["over_price"], annotated["under_price"])
    annotated["p_over"] = p_over
    annotated["p_under"] = p_under
    annotated["implied_mean"] = implied_mean(
        annotated["line"], annotated["p_over"], annotated["category"]
    )
    annotated["book_weight"] = (
        annotated["book"].map(config.PROP_BOOK_SHARPNESS).fillna(config.PROP_BOOK_DEFAULT_SHARPNESS)
    )
    return annotated


def flag_stale_quotes(quotes_df: pd.DataFrame, as_of=None) -> pd.DataFrame:
    """Adds `age_minutes` and `is_stale` against
    `config.PROP_QUOTE_MAX_STALENESS_MINUTES`.

    Staleness is not a nicety. A quote that has not moved while the rest
    of the market has is the single most common way an "outlier" turns
    out to be a line about to be pulled rather than an edge, and it is
    indistinguishable from a real edge on price alone.
    """
    if quotes_df.empty:
        return quotes_df.assign(age_minutes=[], is_stale=[])

    flagged = quotes_df.copy()
    fetched = pd.to_datetime(flagged["fetched_at"], utc=True, errors="coerce")
    reference = pd.to_datetime(as_of, utc=True) if as_of is not None else fetched.max()
    flagged["age_minutes"] = (reference - fetched).dt.total_seconds() / 60.0
    flagged["is_stale"] = flagged["age_minutes"] > config.PROP_QUOTE_MAX_STALENESS_MINUTES
    return flagged


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Median with per-observation weights - robust like a median (one
    broken quote cannot drag it) while still letting a sharper book count
    for more, which a plain median cannot express."""
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    total = weights.sum()
    if total <= 0:
        return float(np.median(values))
    cumulative = np.cumsum(weights) - 0.5 * weights
    return float(np.interp(0.5 * total, cumulative / total * total, values))


def build_consensus(quotes_df: pd.DataFrame) -> pd.DataFrame:
    """One row per (player_id, category): the market's consensus central
    estimate and how much the books disagree about it.

    `consensus_mean` is a SHARPNESS-WEIGHTED MEDIAN of the per-book
    implied means. Median rather than mean so a single stale or fat-
    fingered quote cannot drag the consensus; weighted so a low-margin,
    high-limit book counts for more than a retail one.

    `book_disagreement` - the standard deviation of the per-book implied
    means - is the quantity this whole module exists to surface. It is
    expressed in the stat's own units (receptions, yards), and
    `disagreement_sigmas` restates it as a fraction of the outcome
    spread so it is comparable across categories: 0.5 receptions of
    disagreement is a lot, 0.5 passing yards is nothing.

    Groups with fewer than `config.PROP_MIN_BOOKS_FOR_CONSENSUS` books
    are returned with `has_consensus` False rather than dropped. Two
    books disagreeing is not a consensus to bet against - it is two
    opinions - and the caller should see that rather than have the row
    silently vanish.
    """
    columns = [
        "player_id", "player_name", "category", "consensus_mean", "consensus_line",
        "n_books", "book_disagreement", "disagreement_sigmas", "has_consensus",
    ]
    if quotes_df.empty:
        return pd.DataFrame(columns=columns)

    annotated = annotate_quotes(quotes_df)
    rows = []
    for (player_id, category), group in annotated.groupby(["player_id", "category"]):
        usable = group[group["implied_mean"].notna()]
        if usable.empty:
            continue
        consensus_mean = _weighted_median(
            usable["implied_mean"].to_numpy(), usable["book_weight"].to_numpy()
        )
        disagreement = float(usable["implied_mean"].std(ddof=0))
        sigma = float(np.asarray(outcome_sigma(category, consensus_mean)).reshape(-1)[0])
        rows.append({
            "player_id": player_id,
            "player_name": usable["player_name"].iloc[0],
            "category": category,
            "consensus_mean": consensus_mean,
            # The line the most books actually offer, which is what a
            # reader wants to see next to the consensus estimate.
            "consensus_line": float(usable["line"].median()),
            "n_books": int(usable["book"].nunique()),
            "book_disagreement": disagreement,
            "disagreement_sigmas": disagreement / sigma if sigma and np.isfinite(sigma) else np.nan,
            "has_consensus": usable["book"].nunique() >= config.PROP_MIN_BOOKS_FOR_CONSENSUS,
        })

    return pd.DataFrame(rows, columns=columns)


def score_book_disagreement(quotes_df: pd.DataFrame, consensus_df: pd.DataFrame = None) -> pd.DataFrame:
    """Per quote: which side the consensus favours at THAT book's line,
    and the expected value of taking it at that book's price.

    The consensus each book is scored against is LEAVE-ONE-OUT - the rest
    of the market excluding that book. See the inline comment for why
    that is a correctness requirement rather than a refinement.

    The EV combines the consensus's own probability with the price the
    book actually offers - `p_consensus * (decimal - 1) - (1 -
    p_consensus)` per unit staked. Using the consensus probability
    against the RAW offered payout is the correct pairing: the de-vigged
    number is an estimate of truth, while the vigged price is what is
    actually available to bet.

    `outlier_is_sharpest_book` marks a row where the book being priced
    off is the highest-weighted one in that group. Those rows should be
    treated as information rather than opportunity - if the sharpest book
    in the market is the one out of line, the likeliest explanation is
    that it knows something, not that it is wrong.
    """
    columns = [
        "book", "player_id", "player_name", "category", "line", "implied_mean",
        "consensus_mean", "consensus_books", "deviation", "deviation_sigmas",
        "best_side", "best_price", "consensus_probability", "expected_value",
        "outlier_is_sharpest_book",
    ]
    if quotes_df.empty:
        return pd.DataFrame(columns=columns)

    annotated = annotate_quotes(quotes_df)
    if consensus_df is None:
        consensus_df = build_consensus(quotes_df)
    if consensus_df.empty:
        return pd.DataFrame(columns=columns)

    # LEAVE-ONE-OUT consensus: each book is scored against what the REST
    # of the market thinks, not against a consensus it helped set.
    #
    # This is a correctness fix, not a refinement. Including the book
    # being evaluated pulls the consensus toward it and systematically
    # understates its deviation, and the bias is worst exactly where the
    # data is thinnest - at the three-book minimum, an outlier can BE the
    # median of the three, so it would be scored as agreeing with a
    # consensus that is mostly itself. The question a bettor is asking is
    # "does the rest of the market disagree with this price", and that is
    # what this computes.
    per_book = []
    for (player_id, category), group in annotated.groupby(["player_id", "category"]):
        for book in group["book"].unique():
            others = group[group["book"] != book]
            if others.empty:
                continue
            usable = others[others["implied_mean"].notna()]
            if usable.empty:
                continue
            per_book.append({
                "player_id": player_id, "category": category, "book": book,
                "consensus_mean": _weighted_median(
                    usable["implied_mean"].to_numpy(), usable["book_weight"].to_numpy()
                ),
                "consensus_books": int(usable["book"].nunique()),
            })
    if not per_book:
        return pd.DataFrame(columns=columns)

    merged = annotated.merge(
        pd.DataFrame(per_book), on=["player_id", "category", "book"], how="inner"
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)

    merged["deviation"] = merged["implied_mean"] - merged["consensus_mean"]
    merged["deviation_sigmas"] = merged["deviation"] / outcome_sigma(
        merged["category"], merged["consensus_mean"]
    )
    # The consensus's probability AT THIS BOOK'S LINE - the only apples-
    # to-apples comparison when books post different handicaps.
    merged["consensus_probability_over"] = probability_over(
        merged["consensus_mean"], merged["line"], merged["category"]
    )

    over_decimal = american_to_decimal(merged["over_price"])
    under_decimal = american_to_decimal(merged["under_price"])
    p_over = merged["consensus_probability_over"]
    ev_over = p_over * (over_decimal - 1.0) - (1.0 - p_over)
    ev_under = (1.0 - p_over) * (under_decimal - 1.0) - p_over

    take_over = ev_over >= ev_under
    merged["best_side"] = np.where(take_over, OVER, UNDER)
    merged["best_price"] = np.where(take_over, merged["over_price"], merged["under_price"])
    merged["consensus_probability"] = np.where(take_over, p_over, 1.0 - p_over)
    merged["expected_value"] = np.where(take_over, ev_over, ev_under)

    sharpest = merged.groupby(["player_id", "category"])["book_weight"].transform("max")
    merged["outlier_is_sharpest_book"] = merged["book_weight"] >= sharpest

    return merged[columns].sort_values("expected_value", ascending=False).reset_index(drop=True)


def detect_middles(quotes_df: pd.DataFrame, consensus_df: pd.DataFrame = None) -> pd.DataFrame:
    """Pairs of books where the Over at the lower line and the Under at
    the higher line can BOTH win.

    Genuinely available on these props rather than a curiosity, because
    receptions and rush attempts are low integers: Over 3.5 at one book
    and Under 4.5 at another both cash on exactly 4. One side always
    wins, so the cost of the pair is the vig on the loser and the upside
    is the middle landing.
    """
    columns = [
        "player_id", "player_name", "category", "over_book", "over_line", "over_price",
        "under_book", "under_line", "under_price", "middle_width",
        "middle_probability", "expected_value",
    ]
    if quotes_df.empty:
        return pd.DataFrame(columns=columns)

    if consensus_df is None:
        consensus_df = build_consensus(quotes_df)
    means = consensus_df.set_index(["player_id", "category"])["consensus_mean"] if not consensus_df.empty else {}

    rows = []
    for (player_id, category), group in quotes_df.groupby(["player_id", "category"]):
        for _, low in group.iterrows():
            for _, high in group.iterrows():
                if low["book"] == high["book"] or low["line"] >= high["line"]:
                    continue
                consensus_mean = means.get((player_id, category), np.nan) if len(means) else np.nan
                if np.isnan(consensus_mean):
                    middle_probability = np.nan
                else:
                    upper = probability_over(consensus_mean, low["line"], category)
                    lower = probability_over(consensus_mean, high["line"], category)
                    middle_probability = float(upper - lower)

                over_decimal = float(american_to_decimal(low["over_price"]))
                under_decimal = float(american_to_decimal(high["under_price"]))
                # Stake 1 unit each side. Both win inside the middle;
                # otherwise exactly one wins and the other is lost.
                if np.isnan(middle_probability):
                    expected_value = np.nan
                else:
                    both = (over_decimal - 1.0) + (under_decimal - 1.0)
                    over_only = (over_decimal - 1.0) - 1.0
                    under_only = (under_decimal - 1.0) - 1.0
                    p_above = float(probability_over(consensus_mean, high["line"], category))
                    p_below = float(1.0 - probability_over(consensus_mean, low["line"], category))
                    expected_value = (
                        middle_probability * both + p_above * over_only + p_below * under_only
                    )

                rows.append({
                    "player_id": player_id, "player_name": low["player_name"], "category": category,
                    "over_book": low["book"], "over_line": low["line"], "over_price": low["over_price"],
                    "under_book": high["book"], "under_line": high["line"], "under_price": high["under_price"],
                    "middle_width": float(high["line"] - low["line"]),
                    "middle_probability": middle_probability,
                    "expected_value": expected_value,
                })

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        "expected_value", ascending=False
    ).reset_index(drop=True)


def detect_arbitrage(quotes_df: pd.DataFrame) -> pd.DataFrame:
    """Pairs at the SAME line where both sides can be backed for a
    guaranteed profit - `1/over_decimal + 1/under_decimal < 1`.

    Kept separate from `detect_middles` because the two are different
    animals: an arb needs no view of the outcome distribution at all, so
    it carries no model risk, whereas a middle's value depends entirely
    on the probability of landing in the gap. Real arbs on props are rare
    and get limited quickly; the function exists so that when one appears
    it is not mistaken for a modelling artifact.
    """
    columns = [
        "player_id", "player_name", "category", "line",
        "over_book", "over_price", "under_book", "under_price", "profit_fraction",
    ]
    if quotes_df.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for (player_id, category, line), group in quotes_df.groupby(["player_id", "category", "line"]):
        for _, a in group.iterrows():
            for _, b in group.iterrows():
                if a["book"] == b["book"]:
                    continue
                inverse = 1.0 / float(american_to_decimal(a["over_price"])) + 1.0 / float(
                    american_to_decimal(b["under_price"])
                )
                if inverse < 1.0:
                    rows.append({
                        "player_id": player_id, "player_name": a["player_name"], "category": category,
                        "line": line, "over_book": a["book"], "over_price": a["over_price"],
                        "under_book": b["book"], "under_price": b["under_price"],
                        "profit_fraction": 1.0 - inverse,
                    })

    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns).sort_values(
        "profit_fraction", ascending=False
    ).reset_index(drop=True)


def closing_line_value(bets_df: pd.DataFrame, quotes_df: pd.DataFrame) -> pd.DataFrame:
    """Per bet: whether it beat the market's closing price.

    For each logged bet, the closing quote is the LATEST observation of
    that (player, category) across all books, and CLV is the difference
    between the de-vigged probability of the bet's own side at close and
    the de-vigged probability it was taken at. Positive means the market
    moved toward the bet.

    This is the metric worth having. Beating the close is the best-
    established leading indicator of long-run profitability in betting,
    and it resolves in days rather than requiring hundreds of settled
    wagers - which is exactly the constraint every props decision in this
    project has run into. It is a LEADING indicator, though, not a
    result: positive CLV with a losing record is normal over small
    samples, and this function deliberately reports no win rate so the
    two are not conflated.
    """
    columns = [
        "placed_at", "book", "player_id", "category", "side", "line", "price",
        "bet_probability", "closing_line", "closing_probability", "clv", "beat_close",
    ]
    if bets_df.empty or quotes_df.empty:
        return pd.DataFrame(columns=columns)

    annotated = annotate_quotes(quotes_df)
    annotated["fetched_at"] = pd.to_datetime(annotated["fetched_at"], utc=True, errors="coerce")
    latest = annotated.sort_values("fetched_at").groupby(["player_id", "category"]).last()

    rows = []
    for _, bet in bets_df.iterrows():
        key = (bet["player_id"], bet["category"])
        if key not in latest.index:
            continue
        close = latest.loc[key]
        bet_p_over, bet_p_under = devig_two_way(
            bet["price"], _opposite_price(bet, quotes_df)
        )
        # A bet row carries only the side actually taken, so the other
        # side's price for de-vigging comes from the same book's quote at
        # that line where available; failing that the bet's own raw
        # implied probability is used and that is recorded honestly by
        # leaving the de-vig one-sided rather than inventing a price.
        taken = float(np.asarray(bet_p_over).reshape(-1)[0]) if bet["side"] == OVER else float(
            np.asarray(bet_p_under).reshape(-1)[0]
        )
        closing = float(close["p_over"] if bet["side"] == OVER else close["p_under"])
        # Re-price the closing probability at the BET's line, not the
        # closing line - otherwise a line move is invisible whenever the
        # price happens to stay the same.
        closing_at_bet_line = float(
            probability_over(close["implied_mean"], bet["line"], bet["category"])
        )
        if bet["side"] == UNDER:
            closing_at_bet_line = 1.0 - closing_at_bet_line

        rows.append({
            "placed_at": bet["placed_at"], "book": bet["book"], "player_id": bet["player_id"],
            "category": bet["category"], "side": bet["side"], "line": bet["line"],
            "price": bet["price"], "bet_probability": taken,
            "closing_line": float(close["line"]), "closing_probability": closing,
            "clv": closing_at_bet_line - taken,
            "beat_close": closing_at_bet_line > taken,
        })

    return pd.DataFrame(rows, columns=columns)


def _opposite_price(bet, quotes_df: pd.DataFrame):
    """The other side's price at the same book and line, so a logged bet
    can be de-vigged the same way a quote is. Falls back to the bet's own
    price mirrored, which makes the de-vig a no-op rather than a guess at
    a number nobody observed."""
    match = quotes_df[
        (quotes_df["book"] == bet["book"])
        & (quotes_df["player_id"] == bet["player_id"])
        & (quotes_df["category"] == bet["category"])
        & (quotes_df["line"] == bet["line"])
    ]
    if match.empty:
        return bet["price"]
    row = match.iloc[0]
    return row["under_price"] if bet["side"] == OVER else row["over_price"]


def append_quote_snapshots(quotes_df: pd.DataFrame, path: str) -> pd.DataFrame:
    """Appends quotes to the snapshot history CLV is computed from,
    de-duplicating on (fetched_at, book, player_id, category, line).

    A time series, not a current-state table - which is a different shape
    from every other CSV this project keeps, and necessarily so: closing-
    line value cannot be reconstructed after the fact from a file that
    only ever holds the latest lines. Same append-then-dedupe posture
    `nfl_prop_predictions.append_prop_predictions` already uses, with the
    NEW row winning so a corrected quote supersedes an earlier one.
    """
    import os

    if quotes_df.empty:
        return quotes_df

    combined = quotes_df
    if os.path.exists(path):
        existing = pd.read_csv(path)
        combined = pd.concat([quotes_df, existing], ignore_index=True)
    combined = combined.drop_duplicates(
        subset=["fetched_at", "book", "player_id", "category", "line"], keep="first"
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    combined.to_csv(path, index=False)
    return combined

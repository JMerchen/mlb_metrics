"""Consensus ranking engine: turns several independently-published player
rankings for the SAME player pool into one aggregated ranking - the shared
"make multiple models talk to one another" mechanism behind all three
boards this project builds this way (MLB organizational prospects, MLB
draft-eligible college players, NFL draft-eligible college players).

Real, deliberate scope: this module owns ONLY the aggregation arithmetic
(name matching + rank blending), never the fetching or publishing of any
individual source's own ranking - each board's own `*_sources.py` module
(prospect_sources.py, mlb_draft_sources.py, nfl_draft_sources.py) is
responsible for producing a real DataFrame per source in this module's own
input contract, and each board's own runner script calls
`build_consensus_ranking` once across whichever sources it could actually
fetch that day. This mirrors game_picks.py/nfl_game_picks.py's own
"assemble raw signals elsewhere, blend them here" split.

Why average-of-imputed-rank, not average-of-only-the-sources-that-list-a-
player: a player who appears on only 1 of 8 real published boards (at, say,
#40) is a fundamentally less-agreed-upon evaluation than a player every
board places at #40 - a naive "average rank among sources that rank them"
would treat both identically, hiding real disagreement. Instead, a player
missing from a given source's list is imputed as ranked ONE SPOT past that
source's own real length (`len(list) + unranked_penalty_rows`) for that
source's contribution to the average - a real, soft, source-length-aware
penalty (missing from a 300-player board is a much weaker signal than
missing from a 100-player board, and this imputation scales with each
source's own real length rather than one arbitrary global constant), not
an outright exclusion. This is the same "impute the missing case as a real,
bounded number rather than dropping it" reasoning `helpers.shrink_rate`
already uses elsewhere in this project, adapted to ranks instead of rates.
"""

import re
import unicodedata

import pandas as pd

from mlb_metrics import config

# Real, common name suffixes that vary across publishers for the exact same
# player ("Jr." on one board, dropped entirely on another) - stripped
# during normalization so they never cause a real, avoidable mismatch.
_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_player_name(name: str) -> str:
    """A real, deterministic join key for matching the SAME player across
    independently-formatted published rankings - lowercased, accents
    stripped (Unicode NFKD decomposition, e.g. "Acuña" -> "Acuna" - a real,
    common divergence between publishers, not a hypothetical one),
    periods/apostrophes/hyphens removed, common generational suffixes
    dropped as separate tokens, whitespace collapsed. Not a universal
    solution to entity resolution (a genuine nickname vs. legal-name split,
    e.g. "Nick" vs. "Nicholas", still won't match) - `config.PLAYER_NAME_ALIASES`
    exists for real, discovered mismatches to be fixed by hand as they
    surface in production, the same "note discoveries, extend later"
    posture this project already uses elsewhere (e.g. QB-continuity roster
    name matching)."""
    if not isinstance(name, str) or not name.strip():
        return ""
    canonical = config.PLAYER_NAME_ALIASES.get(name.strip(), name)
    decomposed = unicodedata.normalize("NFKD", canonical)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    stripped = re.sub(r"[.''\-]", "", stripped.lower())
    tokens = [t for t in stripped.split() if t not in _SUFFIXES]
    return " ".join(tokens).strip()


def build_consensus_ranking(
    source_rankings: dict[str, pd.DataFrame],
    name_col: str = "player_name",
    rank_col: str = "rank",
    unranked_penalty_rows: int = None,
) -> pd.DataFrame:
    """Real rank aggregation across `source_rankings` - {source_name:
    DataFrame} where each DataFrame is one real, independently-published
    ranking (must carry at least `name_col`/`rank_col`; any other real
    columns, e.g. `position`/`team`/`url`, are carried through from
    whichever source's row happens to be picked as that player's canonical
    display row - see below). A source with zero real rows contributes
    nothing (real fetch failure, not a fabricated participant) - callers
    should log/report which sources actually returned data, this function
    only aggregates what it's given.

    `unranked_penalty_rows` defaults to `config.CONSENSUS_UNRANKED_PENALTY_ROWS`
    (see this module's own docstring for the full reasoning on why a
    missing-from-a-source player is imputed rather than excluded).

    Returns one real row per player appearing on AT LEAST one source,
    sorted by `consensus_rank` (1 = most consensus-favored), with:
    - `consensus_score`: the real mean imputed rank across ALL sources
      (lower is better) - the actual sort key.
    - `sources_ranked_by`: how many of the real sources actually listed
      this player (not imputed) - real, honest agreement-count, so a
      caller/UI can show "ranked by 6 of 8 sources" rather than hiding it.
    - `source_ranks`: a real {source_name: rank_or_None} dict per player -
      None means that source didn't list them (imputed, not a fabricated
      rank) - the literal "show the underlying models" transparency this
      board is meant to provide, not just a black-box output number.
    - every other real column, MERGED across every real source that
      ranked this player: for each such column, the value comes from the
      BEST-ranked source that has a real (non-null) value for it, falling
      back to the next-best-ranked source that does, and so on - a real,
      deterministic tie-break (not arbitrary), but no longer an
      all-or-nothing "one canonical row" - a player whose best source
      doesn't carry a given field (e.g. a stats-only source with no
      `team`/`ETA` columns at all) still gets that field filled in from
      whichever other real source that ranked them actually has it,
      rather than left blank just because the best source didn't happen
      to publish it. Real, honest limit: a field stays blank only when
      NO real source that ranked this player had a value for it.
    A player appearing in zero real sources never appears in the output -
    there is no real ranking to aggregate for them."""
    penalty = config.CONSENSUS_UNRANKED_PENALTY_ROWS if unranked_penalty_rows is None else unranked_penalty_rows

    real_sources = {name: df for name, df in source_rankings.items() if df is not None and len(df) > 0}
    if not real_sources:
        return pd.DataFrame(columns=[
            "consensus_rank", "consensus_score", "sources_ranked_by", name_col, "source_ranks",
        ])

    prepared = {}
    for source_name, df in real_sources.items():
        working = df.copy()
        working["_norm_name"] = working[name_col].apply(normalize_player_name)
        working = working[working["_norm_name"] != ""]
        prepared[source_name] = working

    all_norm_names = sorted(set().union(*(set(df["_norm_name"]) for df in prepared.values())))

    rows = []
    for norm_name in all_norm_names:
        source_ranks = {}
        effective_ranks = []
        matches_by_rank = []
        for source_name, df in prepared.items():
            source_len = len(df)
            match = df[df["_norm_name"] == norm_name]
            if match.empty:
                source_ranks[source_name] = None
                effective_ranks.append(source_len + penalty)
                continue
            real_rank = float(match.iloc[0][rank_col])
            source_ranks[source_name] = real_rank
            effective_ranks.append(real_rank)
            matches_by_rank.append((real_rank, match.iloc[0]))

        matches_by_rank.sort(key=lambda pair: pair[0])
        sources_ranked_by = sum(1 for v in source_ranks.values() if v is not None)

        # Merge every real column across ALL sources that ranked this
        # player (best-ranked source's value wins per column, falling
        # back to the next-best real source that has one) rather than
        # taking one canonical row wholesale - see this function's own
        # docstring for why.
        row = {}
        for _, candidate_row in matches_by_rank:
            for col, value in candidate_row.items():
                if col == "_norm_name":
                    continue
                if col not in row or pd.isna(row[col]):
                    row[col] = value
        row[name_col] = matches_by_rank[0][1][name_col]
        row["consensus_score"] = sum(effective_ranks) / len(effective_ranks)
        row["sources_ranked_by"] = sources_ranked_by
        row["source_ranks"] = source_ranks
        rows.append(row)

    result = pd.DataFrame(rows).sort_values("consensus_score", kind="stable").reset_index(drop=True)
    result.insert(0, "consensus_rank", range(1, len(result) + 1))
    return result

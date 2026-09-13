import pandas as pd

from mlb_metrics import config, consensus_rankings as cr


def test_normalize_player_name_strips_accents_punctuation_and_suffixes():
    assert cr.normalize_player_name("Ronald Acuña Jr.") == "ronald acuna"
    assert cr.normalize_player_name("C.J. Stroud III") == "cj stroud"
    assert cr.normalize_player_name("  Bo   Bichette  ") == "bo bichette"


def test_normalize_player_name_handles_empty_and_non_string():
    assert cr.normalize_player_name("") == ""
    assert cr.normalize_player_name(None) == ""
    assert cr.normalize_player_name(float("nan")) == ""


def test_normalize_player_name_applies_configured_alias(monkeypatch):
    monkeypatch.setitem(config.PLAYER_NAME_ALIASES, "Nick Kurtz Jr.", "Nicholas Kurtz")

    assert cr.normalize_player_name("Nick Kurtz Jr.") == cr.normalize_player_name("Nicholas Kurtz")


def _source(rows):
    """rows: list of (player_name, rank) tuples - one real published
    ranking's own shape."""
    return pd.DataFrame([{"player_name": name, "rank": rank} for name, rank in rows])


def test_build_consensus_ranking_hand_computed_two_sources():
    # Source A (3 real ranked players): Alice=1, Bob=2, Cara=3.
    # Source B (3 real ranked players): Bob=1, Cara=2, Dave=3.
    # Alice is missing from B -> imputed at len(B)+1 = 4 for B's contribution.
    # Dave is missing from A -> imputed at len(A)+1 = 4 for A's contribution.
    source_a = _source([("Alice", 1), ("Bob", 2), ("Cara", 3)])
    source_b = _source([("Bob", 1), ("Cara", 2), ("Dave", 3)])

    result = cr.build_consensus_ranking({"A": source_a, "B": source_b})

    # Alice: (1 + 4) / 2 = 2.5 | Bob: (2 + 1) / 2 = 1.5 | Cara: (3 + 2) / 2 = 2.5 | Dave: (4 + 3) / 2 = 3.5
    scores = result.set_index("player_name")["consensus_score"]
    assert scores["Bob"] == 1.5
    assert scores["Alice"] == 2.5
    assert scores["Cara"] == 2.5
    assert scores["Dave"] == 3.5

    # Real, honest agreement counts - Alice/Dave real-ranked by only 1 of 2
    # sources, Bob/Cara by both.
    ranked_by = result.set_index("player_name")["sources_ranked_by"]
    assert ranked_by["Alice"] == 1
    assert ranked_by["Bob"] == 2
    assert ranked_by["Cara"] == 2
    assert ranked_by["Dave"] == 1

    # Consensus order: Bob (best), then Alice/Cara tied (stable sort keeps
    # their real alphabetical-by-normalized-name insertion order), then Dave.
    assert list(result["player_name"]) == ["Bob", "Alice", "Cara", "Dave"]
    assert list(result["consensus_rank"]) == [1, 2, 3, 4]

    # Real per-source transparency - Alice's real rank in A, imputed (None)
    # in B, not a fabricated number.
    alice_sources = result.set_index("player_name").loc["Alice", "source_ranks"]
    assert alice_sources == {"A": 1.0, "B": None}


def test_build_consensus_ranking_picks_canonical_row_from_players_best_source():
    # Bob's extra columns (e.g. team) should come from whichever source
    # ranked him BEST (source B, rank 1) - not source A (rank 2) - a real,
    # deterministic tie-break, not arbitrary dict-iteration order.
    source_a = pd.DataFrame([{"player_name": "Bob", "rank": 2, "team": "STALE"}])
    source_b = pd.DataFrame([{"player_name": "Bob", "rank": 1, "team": "CURRENT"}])

    result = cr.build_consensus_ranking({"A": source_a, "B": source_b})

    assert result.loc[0, "team"] == "CURRENT"


def test_build_consensus_ranking_ignores_empty_source():
    # A source that returned zero real rows (a real fetch failure) must
    # contribute nothing - neither ranking a player nor imputing a penalty
    # for players missing from it.
    source_a = _source([("Alice", 1), ("Bob", 2)])
    empty_source = pd.DataFrame(columns=["player_name", "rank"])

    with_empty = cr.build_consensus_ranking({"A": source_a, "Empty": empty_source})
    without_empty = cr.build_consensus_ranking({"A": source_a})

    pd.testing.assert_frame_equal(with_empty, without_empty)


def test_build_consensus_ranking_respects_unranked_penalty_override():
    source_a = _source([("Alice", 1)])
    source_b = _source([("Bob", 1)])

    result = cr.build_consensus_ranking({"A": source_a, "B": source_b}, unranked_penalty_rows=5)

    # Alice missing from B (len 1) -> imputed at 1 + 5 = 6 -> mean (1+6)/2 = 3.5.
    assert result.set_index("player_name").loc["Alice", "consensus_score"] == 3.5


def test_build_consensus_ranking_empty_input_returns_empty_frame():
    result = cr.build_consensus_ranking({})

    assert result.empty
    assert list(result.columns) == [
        "consensus_rank", "consensus_score", "sources_ranked_by", "player_name", "source_ranks",
    ]

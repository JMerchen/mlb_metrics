import pandas as pd

from mlb_metrics import board_runner


def test_run_board_writes_csv_and_returns_consensus(tmp_path):
    fetchers = {
        "A": lambda: pd.DataFrame([{"rank": 1, "player_name": "Alice"}, {"rank": 2, "player_name": "Bob"}]),
        "B": lambda: pd.DataFrame([{"rank": 1, "player_name": "Bob"}]),
    }
    output_path = str(tmp_path / "board.csv")

    result = board_runner.run_board(fetchers, output_path, "Test Board")

    # Alice: A=1, imputed in B (len 1) at 1+1=2 -> mean 1.5.
    # Bob: A=2, B=1 -> mean 1.5. A real tie - stable sort keeps the
    # alphabetical-by-normalized-name insertion order (alice, bob).
    assert list(result["player_name"]) == ["Alice", "Bob"]
    written = pd.read_csv(output_path)
    assert list(written["player_name"]) == ["Alice", "Bob"]
    # source_ranks is a real per-player dict - flattened to a string for CSV.
    assert isinstance(written.loc[0, "source_ranks"], str)


def test_run_board_respects_top_n_for_the_returned_frame_but_not_the_csv(tmp_path):
    fetchers = {
        "A": lambda: pd.DataFrame([
            {"rank": 1, "player_name": "Alice"}, {"rank": 2, "player_name": "Bob"}, {"rank": 3, "player_name": "Cara"},
        ]),
    }
    output_path = str(tmp_path / "board.csv")

    result = board_runner.run_board(fetchers, output_path, "Test Board", top_n=1)

    assert len(result) == 1
    written = pd.read_csv(output_path)
    assert len(written) == 3


def test_run_board_writes_nothing_when_every_source_fails(tmp_path):
    output_path = tmp_path / "board.csv"
    output_path.write_text("stale,previous,data\n1,2,3\n")
    fetchers = {"A": lambda: pd.DataFrame(columns=["rank", "player_name"])}

    result = board_runner.run_board(fetchers, str(output_path), "Test Board")

    assert result.empty
    # The prior real file is left untouched - a real, honest degrade, not
    # an empty file overwriting good history.
    assert output_path.read_text() == "stale,previous,data\n1,2,3\n"


def test_run_board_isolates_one_fetchers_unexpected_exception_from_the_others(tmp_path):
    # Real, confirmed necessary fix (2026-09-15 live CI run): a bug
    # INSIDE one fetcher's own post-parsing code (not its own
    # network/parsing try/except) once crashed this entire function,
    # taking every other real source down with it for that run.
    def _broken():
        raise TypeError("arg must be a list, tuple, 1-d array, or Series")

    fetchers = {
        "Working": lambda: pd.DataFrame([{"rank": 1, "player_name": "Alice"}]),
        "Broken": _broken,
    }
    output_path = str(tmp_path / "board.csv")

    result = board_runner.run_board(fetchers, output_path, "Test Board")

    assert list(result["player_name"]) == ["Alice"]

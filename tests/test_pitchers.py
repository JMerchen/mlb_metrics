import pandas as pd
import pytest

from mlb_metrics import pitchers


def _events(pitcher, rows):
    return [
        {"pitcher": pitcher, "game_date": pd.Timestamp(date), "events": events} for date, events in rows
    ]


def test_compute_pave_blends_windows_with_strikeout_adjustment():
    # Windows relative to latest=2026-06-20: full/30d/81d/15d cutoffs are
    # 2026-03-25(season start-ish)/05-21/03-31/06-05.
    rows = _events(1, [
        ("2026-03-01", "field_out"),                # only in full
        ("2026-04-15", "single"), ("2026-04-15", "field_out"),   # in 81d, not 30d/15d
        ("2026-06-01", "single"), ("2026-06-01", "strikeout"),   # in 30d, not 15d
        ("2026-06-18", "single"),                    # in 15d
    ])
    pdf = pd.DataFrame(rows)

    pave = pitchers.compute_pave(pdf).set_index("key_mlbam")

    assert pave.loc[1, "at_bats"] == 6
    assert pave.loc[1, "hits"] == 3
    assert pave.loc[1, "TBA"] == 3
    # full: baa=3/6=.5, stk_rate=1/6 -> pave_full=.5/(5/6)=.6
    # 30d: baa=2/3, stk_rate=1/3 -> pave_30 = (2/3)/(2/3) = 1.0
    # 81d: baa=3/5, stk_rate=1/5 -> pave_81 = .6/.8=.75
    # 15d: baa=1/1, stk_rate=0 -> pave_15=1.0
    expected = 0.6 * 0.3 + 1.0 * 0.265 + 0.75 * 0.23 + 1.0 * 0.205
    assert pave.loc[1, "PAVE"] == pytest.approx(expected)


def test_assemble_pitchers_normalizes_pave_plus_to_qualified_mean():
    # Pitcher 1: same data as above (PAVE ~= 0.8225, at_bats=6, qualifies).
    rows = _events(1, [
        ("2026-03-01", "field_out"),
        ("2026-04-15", "single"), ("2026-04-15", "field_out"),
        ("2026-06-01", "single"), ("2026-06-01", "strikeout"),
        ("2026-06-18", "single"),
    ])
    # Pitcher 2: a single at-bat (well under the 75%-of-max qualified
    # threshold), so it should NOT count toward the qualified-average
    # baseline, but should still get a PAVE_PLUS relative to it.
    rows += _events(2, [("2026-06-18", "single")])

    pdf = pd.DataFrame(rows)
    names = pd.DataFrame([
        {"key_mlbam": 1, "name_first": "Ace", "name_last": "One"},
        {"key_mlbam": 2, "name_first": "Rook", "name_last": "Two"},
    ])
    latest_pitcher_team = pd.DataFrame([
        {"key_mlbam": 1, "team": "NYY"},
        {"key_mlbam": 2, "team": "BOS"},
    ])

    result = pitchers.assemble_pitchers(pdf, names, latest_pitcher_team).set_index("key_mlbam")

    assert result.loc[1, "PAVE_PLUS"] == pytest.approx(1.0)
    assert result.loc[2, "PAVE_PLUS"] == pytest.approx(1.0 / 0.8225, rel=1e-3)

    # baa blended = .5*.3 + (2/3)*.265 + .6*.23 + 1*.205 = .669675 -> Expected_Hits = baa * 22
    assert result.loc[1, "Expected_Hits"] == pytest.approx(0.669675 * 22, rel=1e-3)

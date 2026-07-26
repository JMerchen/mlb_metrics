import pandas as pd
import pytest

from mlb_metrics import pitchers


def _events(pitcher, rows):
    return [
        {"pitcher": pitcher, "game_date": pd.Timestamp(date), "events": events} for date, events in rows
    ]


def test_compute_pave_blends_windows_excluding_only_walks_and_hbp_from_ab():
    # Windows relative to latest=2026-06-20: full/30d/81d/15d cutoffs are
    # 2026-03-25(season start-ish)/05-21/03-31/06-05.
    #
    # A "walk" (not a strikeout) is the only non-AB event here, on purpose:
    # PAVE used to exclude strikeouts from the AB denominator too (a real
    # bug - see pitchers.py's module docstring), which this fixture would
    # NOT have caught, since it has no walk/HBP-vs-strikeout distinction to
    # exercise unless a real non-AB event (walk/HBP) is present. The
    # strikeout below must stay IN every window's AB count.
    rows = _events(1, [
        ("2026-03-01", "walk"),                      # only in full
        ("2026-04-15", "single"), ("2026-04-15", "field_out"),   # in 81d, not 30d/15d
        ("2026-06-01", "single"), ("2026-06-01", "strikeout"),   # in 30d, not 15d
        ("2026-06-18", "single"),                    # in 15d
    ])
    pdf = pd.DataFrame(rows)

    pave = pitchers.compute_pave(pdf).set_index("key_mlbam")

    assert pave.loc[1, "at_bats"] == 6
    assert pave.loc[1, "hits"] == 3
    assert pave.loc[1, "TBA"] == 3
    # full: baa=3/6=.5, non_ab_rate=1/6 (the one walk) -> pave_full=.5/(5/6)=.6
    # 30d: baa=2/3, non_ab_rate=0 (the strikeout stays IN the AB count) -> pave_30=2/3
    # 81d: baa=3/5, non_ab_rate=0 -> pave_81=.6
    # 15d: baa=1/1, non_ab_rate=0 -> pave_15=1.0
    expected = 0.6 * 0.3 + (2 / 3) * 0.265 + 0.6 * 0.23 + 1.0 * 0.205
    assert pave.loc[1, "PAVE"] == pytest.approx(expected)

    # power_a (total bases allowed per PA): full=3/6=.5, 30d=2/3, 81d=3/5=.6,
    # 15d=1/1=1.0 - happens to equal "baa" here since every hit in this
    # fixture is a single (tba == hit for a single), but power_a and baa are
    # independently computed and diverge whenever extra-base hits appear.
    expected_power_a = 0.5 * 0.3 + (2 / 3) * 0.265 + 0.6 * 0.23 + 1.0 * 0.205
    assert pave.loc[1, "power_a"] == pytest.approx(expected_power_a)


def test_assemble_pitchers_normalizes_pave_plus_to_qualified_mean():
    # Pitcher 1: same data as the compute_pave test above (blended
    # PAVE ~= 0.699667, at_bats=6, qualifies).
    rows = _events(1, [
        ("2026-03-01", "walk"),
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

    pitcher_1_pave = 0.6 * 0.3 + (2 / 3) * 0.265 + 0.6 * 0.23 + 1.0 * 0.205
    assert result.loc[1, "PAVE_PLUS"] == pytest.approx(1.0)
    assert result.loc[2, "PAVE_PLUS"] == pytest.approx(1.0 / pitcher_1_pave, rel=1e-3)

    # baa blended = .5*.3 + (2/3)*.265 + .6*.23 + 1*.205 = .669675 -> Expected_Hits = baa * 22
    assert result.loc[1, "Expected_Hits"] == pytest.approx(0.669675 * 22, rel=1e-3)

    # power_a blended is the same .669675 here (every hit in this fixture is
    # a single) - pitcher 1 (the only qualified pitcher) is its own
    # baseline, so Power_A_PLUS(1) == 1.0; pitcher 2 (a single-AB sample,
    # power_a=1.0 since its lone at-bat was a single) is above that baseline.
    assert result.loc[1, "Power_A_PLUS"] == pytest.approx(1.0)
    assert result.loc[2, "Power_A_PLUS"] == pytest.approx(1.0 / 0.669675, rel=1e-3)


def test_compute_pitcher_throws_one_row_per_pitcher():
    pdf = pd.DataFrame([
        {"pitcher": 1, "game_date": pd.Timestamp("2026-06-01"), "events": "single", "p_throws": "L"},
        {"pitcher": 1, "game_date": pd.Timestamp("2026-06-02"), "events": "field_out", "p_throws": "L"},
        {"pitcher": 2, "game_date": pd.Timestamp("2026-06-01"), "events": "strikeout", "p_throws": "R"},
    ])

    throws = pitchers.compute_pitcher_throws(pdf).set_index("key_mlbam")

    assert throws.loc[1, "Throws"] == "L"
    assert throws.loc[2, "Throws"] == "R"


def test_compute_pitcher_throws_missing_column_returns_empty_not_a_crash():
    pdf = pd.DataFrame([{"pitcher": 1, "game_date": pd.Timestamp("2026-06-01"), "events": "single"}])

    throws = pitchers.compute_pitcher_throws(pdf)

    assert throws.empty
    assert list(throws.columns) == ["key_mlbam", "Throws"]


def test_assemble_pitchers_carries_throws_and_degrades_gracefully_without_it():
    rows = _events(1, [("2026-06-18", "single")])
    pdf = pd.DataFrame(rows)
    pdf["p_throws"] = "L"
    names = pd.DataFrame([{"key_mlbam": 1, "name_first": "Ace", "name_last": "One"}])
    latest_pitcher_team = pd.DataFrame([{"key_mlbam": 1, "team": "NYY"}])

    with_throws = pitchers.assemble_pitchers(pdf, names, latest_pitcher_team).set_index("key_mlbam")
    assert with_throws.loc[1, "Throws"] == "L"

    without_throws = pitchers.assemble_pitchers(
        pdf.drop(columns=["p_throws"]), names, latest_pitcher_team
    ).set_index("key_mlbam")
    assert pd.isna(without_throws.loc[1, "Throws"])


def _bullpen_row(team, pitcher, date, events, is_starter):
    return {
        "team": team,
        "pitcher": pitcher,
        "game_date": pd.Timestamp(date),
        "events": events,
        "is_starter": is_starter,
    }


def test_compute_bullpen_pave_excludes_starters_and_pools_by_team():
    pitcher_1_pave = 0.6 * 0.3 + (2 / 3) * 0.265 + 0.6 * 0.23 + 1.0 * 0.205
    rows = [
        # Team X's bullpen (two different relievers pooled together): the
        # exact same events as the compute_pave test above, split across two
        # pitchers, so Bullpen_PAVE(X) should equal that test's blended PAVE.
        _bullpen_row("X", 11, "2026-03-01", "walk", False),
        _bullpen_row("X", 11, "2026-04-15", "single", False),
        _bullpen_row("X", 12, "2026-04-15", "field_out", False),
        _bullpen_row("X", 12, "2026-06-01", "single", False),
        _bullpen_row("X", 11, "2026-06-01", "strikeout", False),
        _bullpen_row("X", 12, "2026-06-18", "single", False),
        # Team X's starter on the same date - must NOT affect Bullpen_PAVE(X)
        # despite a wildly different stat line (a lone home run allowed).
        _bullpen_row("X", 99, "2026-06-18", "home_run", True),
        # Team Y's bullpen: a single at-bat, mirroring pitcher 2 above (PAVE=1.0).
        _bullpen_row("Y", 21, "2026-06-18", "single", False),
    ]
    pdf_with_role = pd.DataFrame(rows)

    result = pitchers.compute_bullpen_pave(pdf_with_role).set_index("team")

    assert result.loc["X", "Bullpen_AtBats"] == 6
    mean_pave = (pitcher_1_pave + 1.0) / 2
    assert result.loc["X", "Bullpen_PAVE_PLUS"] == pytest.approx(pitcher_1_pave / mean_pave, rel=1e-3)
    assert result.loc["Y", "Bullpen_PAVE_PLUS"] == pytest.approx(1.0 / mean_pave, rel=1e-3)

    # Same underlying at-bats as the power_a test above: team X's power_a
    # blended = .669675 (all singles, so it matches PAVE here); team Y's
    # lone at-bat is also a single, so power_a=1.0 too.
    mean_power_a = (0.669675 + 1.0) / 2
    assert result.loc["X", "Bullpen_Power_A_PLUS"] == pytest.approx(0.669675 / mean_power_a, rel=1e-3)
    assert result.loc["Y", "Bullpen_Power_A_PLUS"] == pytest.approx(1.0 / mean_power_a, rel=1e-3)

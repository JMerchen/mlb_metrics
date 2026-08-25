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


def test_compute_bullpen_recent_workload_sums_relief_outs_within_window():
    # Latest date = 2026-06-18, recent_days=2 -> cutoff 2026-06-16.
    rows = [
        _bullpen_row("X", 11, "2026-06-10", "field_out", False),   # outside window
        _bullpen_row("X", 11, "2026-06-17", "field_out", False),   # 1 out, in window
        _bullpen_row("X", 12, "2026-06-18", "strikeout", False),   # 1 out, in window
        _bullpen_row("X", 12, "2026-06-18", "single", False),      # 0 outs, in window
        # Team X's starter on the same dates - must NOT count.
        _bullpen_row("X", 99, "2026-06-18", "strikeout", True),
        # Team Y: only outside the window.
        _bullpen_row("Y", 21, "2026-06-10", "strikeout", False),
    ]
    pdf_with_role = pd.DataFrame(rows)

    result = pitchers.compute_bullpen_recent_workload(pdf_with_role, recent_days=2).set_index("team")

    assert result.loc["X", "Bullpen_Recent_Outs"] == 2
    assert "Y" not in result.index


def test_compute_bullpen_recent_workload_empty_input_returns_empty_not_crash():
    result = pitchers.compute_bullpen_recent_workload(pd.DataFrame(columns=["game_date", "team", "is_starter", "events"]))

    assert result.empty
    assert list(result.columns) == ["team", "Bullpen_Recent_Outs"]


def test_compute_bullpen_recent_workload_no_relief_appearances_in_window_returns_empty():
    # Every appearance in the window is a starter's - real 0 relief workload,
    # not a fabricated row.
    rows = [_bullpen_row("X", 99, "2026-06-18", "strikeout", True)]
    pdf_with_role = pd.DataFrame(rows)

    result = pitchers.compute_bullpen_recent_workload(pdf_with_role, recent_days=2)

    assert result.empty


def test_compute_bullpen_distinct_relievers_counts_different_pitchers_not_outs():
    # Latest date = 2026-06-18, recent_days=2 -> cutoff 2026-06-16. Team X
    # uses 2 different relievers in the window (real breadth); Team Y uses
    # the SAME reliever twice on different days (1 distinct, despite 2
    # appearances) - the count must reflect distinct arms, not raw outings.
    rows = [
        _bullpen_row("X", 11, "2026-06-17", "field_out", False),
        _bullpen_row("X", 12, "2026-06-18", "strikeout", False),
        _bullpen_row("Y", 21, "2026-06-17", "field_out", False),
        _bullpen_row("Y", 21, "2026-06-18", "strikeout", False),
        # Team X's starter must not count.
        _bullpen_row("X", 99, "2026-06-18", "strikeout", True),
    ]
    pdf_with_role = pd.DataFrame(rows)

    result = pitchers.compute_bullpen_distinct_relievers(pdf_with_role, recent_days=2).set_index("team")

    assert result.loc["X", "Bullpen_Distinct_Relievers"] == 2
    assert result.loc["Y", "Bullpen_Distinct_Relievers"] == 1


def test_compute_bullpen_distinct_relievers_empty_input_returns_empty_not_crash():
    result = pitchers.compute_bullpen_distinct_relievers(
        pd.DataFrame(columns=["game_date", "team", "is_starter", "pitcher"])
    )

    assert result.empty
    assert list(result.columns) == ["team", "Bullpen_Distinct_Relievers"]


def test_compute_bullpen_back_to_back_relievers_requires_appearance_on_both_latest_days():
    # Latest date = 2026-06-18, so "back-to-back" here means 2026-06-17 AND
    # 2026-06-18. Pitcher 11 appears on both (real back-to-back). Pitcher
    # 12 appears only on 06-18 (not back-to-back - fresh yesterday).
    # Pitcher 13 appears only on 06-16 (two days before latest - a real
    # off-day gap, not back-to-back either).
    rows = [
        _bullpen_row("X", 11, "2026-06-17", "field_out", False),
        _bullpen_row("X", 11, "2026-06-18", "strikeout", False),
        _bullpen_row("X", 12, "2026-06-18", "single", False),
        _bullpen_row("X", 13, "2026-06-16", "field_out", False),
    ]
    pdf_with_role = pd.DataFrame(rows)

    result = pitchers.compute_bullpen_back_to_back_relievers(pdf_with_role).set_index("team")

    assert result.loc["X", "Bullpen_Back_To_Back_Relievers"] == 1


def test_compute_bullpen_back_to_back_relievers_off_day_gap_reads_zero():
    # Team's two most recent appearances are 3 days apart (a real
    # scheduled off day in between) - nobody can be "back-to-back" across
    # that gap, so this must read a real 0, not compare across it.
    rows = [
        _bullpen_row("X", 11, "2026-06-15", "field_out", False),
        _bullpen_row("X", 11, "2026-06-18", "strikeout", False),
    ]
    pdf_with_role = pd.DataFrame(rows)

    result = pitchers.compute_bullpen_back_to_back_relievers(pdf_with_role)

    assert result.empty


def test_compute_bullpen_back_to_back_relievers_empty_input_returns_empty_not_crash():
    result = pitchers.compute_bullpen_back_to_back_relievers(
        pd.DataFrame(columns=["game_date", "team", "is_starter", "pitcher"])
    )

    assert result.empty
    assert list(result.columns) == ["team", "Bullpen_Back_To_Back_Relievers"]


def _pitches(pitcher, rows):
    return [{"pitcher": pitcher, "game_date": pd.Timestamp(date), "pitch_type": pitch_type} for date, pitch_type in rows]


def test_compute_pitch_arsenal_exact_arithmetic_all_windows():
    # Windows relative to latest=2026-06-20 (config.PAVE_WINDOWS: full/30d/
    # 81d/15d, cutoffs 2026-05-21/03-31/06-05).
    rows = _pitches(1, [
        ("2026-03-01", "FF"),                    # only in full
        ("2026-04-15", "SL"), ("2026-04-15", "CH"),   # in 81d, not 30d/15d
        ("2026-06-01", "FF"), ("2026-06-01", "FF"),   # in 30d, not 15d
        ("2026-06-18", "SL"),                    # in 15d
    ])
    all_pitches = pd.DataFrame(rows)

    result = pitchers.compute_pitch_arsenal(all_pitches).set_index("key_mlbam")

    assert result.loc[1, "pitches_thrown"] == 6
    # full (6 pitches): FF=3, SL=2, CH=1 -> 0.5/0.3333/0.1667
    # 30d (3 pitches): FF=2, SL=1, CH=0 -> 0.6667/0.3333/0
    # 81d (5 pitches): FF=2, SL=2, CH=1 -> 0.4/0.4/0.2
    # 15d (1 pitch): FF=0, SL=1, CH=0 -> 0/1.0/0
    expected_fastball = 0.5 * 0.300 + (2 / 3) * 0.265 + 0.4 * 0.230 + 0 * 0.205
    expected_breaking = (1 / 3) * 0.300 + (1 / 3) * 0.265 + 0.4 * 0.230 + 1.0 * 0.205
    expected_offspeed = (1 / 6) * 0.300 + 0 * 0.265 + 0.2 * 0.230 + 0 * 0.205
    assert result.loc[1, "Fastball_Rate"] == pytest.approx(expected_fastball)
    assert result.loc[1, "Breaking_Rate"] == pytest.approx(expected_breaking)
    assert result.loc[1, "Offspeed_Rate"] == pytest.approx(expected_offspeed)
    total = result.loc[1, "Fastball_Rate"] + result.loc[1, "Breaking_Rate"] + result.loc[1, "Offspeed_Rate"]
    assert total == pytest.approx(1.0)


def test_compute_pitch_arsenal_excludes_unclassifiable_pitch_types():
    # A pitchout ("PO") and a real null both have no real family - dropped
    # from the denominator entirely, not counted as a fourth bucket.
    rows = _pitches(1, [("2026-06-18", "FF"), ("2026-06-18", "PO"), ("2026-06-18", None)])
    all_pitches = pd.DataFrame(rows)

    result = pitchers.compute_pitch_arsenal(all_pitches).set_index("key_mlbam")

    assert result.loc[1, "pitches_thrown"] == 1
    assert result.loc[1, "Fastball_Rate"] == pytest.approx(1.0)
    assert result.loc[1, "Breaking_Rate"] == 0
    assert result.loc[1, "Offspeed_Rate"] == 0


def test_assemble_pitchers_merges_pitch_arsenal_when_provided():
    rows = _events(1, [("2026-06-18", "single")])
    pdf = pd.DataFrame(rows)
    names = pd.DataFrame([{"key_mlbam": 1, "name_first": "Test", "name_last": "Pitcher"}])
    latest_team = pd.DataFrame([{"key_mlbam": 1, "team": "NYY"}])
    pitch_arsenal = pd.DataFrame([
        {"key_mlbam": 1, "Fastball_Rate": 0.6, "Breaking_Rate": 0.3, "Offspeed_Rate": 0.1, "pitches_thrown": 50}
    ])

    result = pitchers.assemble_pitchers(pdf, names, latest_team, pitch_arsenal).set_index("key_mlbam")

    assert result.loc[1, "Fastball_Rate"] == pytest.approx(0.6)
    assert result.loc[1, "Breaking_Rate"] == pytest.approx(0.3)
    assert result.loc[1, "Offspeed_Rate"] == pytest.approx(0.1)


def test_assemble_pitchers_omits_arsenal_columns_when_not_provided():
    rows = _events(1, [("2026-06-18", "single")])
    pdf = pd.DataFrame(rows)
    names = pd.DataFrame([{"key_mlbam": 1, "name_first": "Test", "name_last": "Pitcher"}])
    latest_team = pd.DataFrame([{"key_mlbam": 1, "team": "NYY"}])

    result = pitchers.assemble_pitchers(pdf, names, latest_team)

    assert "Fastball_Rate" not in result.columns


def test_assemble_pitchers_pitch_arsenal_missing_pitcher_is_null_not_zero():
    # A pitcher present in `pdf` but absent from pitch_arsenal (no real
    # pitches tracked in that window) must get null rates, not a fabricated
    # 0/0/0 that would misrepresent "unknown mix" as "throws nothing."
    rows = _events(1, [("2026-06-18", "single")])
    pdf = pd.DataFrame(rows)
    names = pd.DataFrame([{"key_mlbam": 1, "name_first": "Test", "name_last": "Pitcher"}])
    latest_team = pd.DataFrame([{"key_mlbam": 1, "team": "NYY"}])
    pitch_arsenal = pd.DataFrame(columns=["key_mlbam", "Fastball_Rate", "Breaking_Rate", "Offspeed_Rate"])

    result = pitchers.assemble_pitchers(pdf, names, latest_team, pitch_arsenal).set_index("key_mlbam")

    assert pd.isna(result.loc[1, "Fastball_Rate"])

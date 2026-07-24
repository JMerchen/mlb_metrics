import pandas as pd
import pytest

from mlb_metrics import pitcher_form


def _start(pitcher, game_id, date, events, is_starter=True, team="BOS"):
    return [
        {"pitcher": pitcher, "game_id": game_id, "game_date": pd.Timestamp(date), "events": e, "is_starter": is_starter, "team": team}
        for e in events
    ]


def _events_for(outs_events_count, so, bb, hr, extra_outs_events="field_out"):
    """Builds an events list with the given strikeout/walk/HR counts plus
    enough extra_outs_events (an out-recording event) to reach the target
    out count exactly (so/hr each count as their own out too, matching
    helpers.OUTS_BY_EVENT: strikeout=1 out, home_run=0 outs, walk=0 outs)."""
    events = ["strikeout"] * so + ["walk"] * bb + ["home_run"] * hr
    remaining_outs = outs_events_count - so
    events += [extra_outs_events] * remaining_outs
    return events


def test_compute_pitcher_dfs_form_blends_windows_and_excludes_relief():
    # Windows relative to latest=2026-06-20 (start 4): 60d/30d/15d cutoffs
    # are 2026-04-21 / 2026-05-21 / 2026-06-05.
    rows = []
    # Start 1 (2026-03-01): only in "full" window. 15 outs (5 IP), 3 K, 1 BB, 1 HR.
    rows += _start(1, "g1", "2026-03-01", _events_for(15, so=3, bb=1, hr=1))
    # Start 2 (2026-05-10): full + 60d. 18 outs (6 IP), 6 K, 2 BB, 0 HR.
    rows += _start(1, "g2", "2026-05-10", _events_for(18, so=6, bb=2, hr=0))
    # Start 3 (2026-05-25): full + 60d + 30d. 21 outs (7 IP), 8 K, 1 BB, 1 HR.
    rows += _start(1, "g3", "2026-05-25", _events_for(21, so=8, bb=1, hr=1))
    # Start 4 (2026-06-20, latest): all four windows. 18 outs (6 IP), 7 K, 0 BB, 0 HR.
    rows += _start(1, "g4", "2026-06-20", _events_for(18, so=7, bb=0, hr=0))
    # A relief appearance for the same pitcher - must be excluded entirely.
    rows += _start(1, "g5", "2026-06-19", ["strikeout"], is_starter=False)

    pdf_with_role = pd.DataFrame(rows)

    result = pitcher_form.compute_pitcher_dfs_form(pdf_with_role).set_index("key_mlbam")

    assert result.loc[1, "starts"] == 4
    assert result.loc[1, "IP"] == pytest.approx(24.0)

    # Per-window rates (IP = outs/3, X9 = X*9/IP):
    k9_full, bb9_full, hr9_full, ips_full = 24 * 9 / 24, 4 * 9 / 24, 2 * 9 / 24, 24 / 4
    k9_60, bb9_60, hr9_60, ips_60 = 21 * 9 / 19, 3 * 9 / 19, 1 * 9 / 19, 19 / 3
    k9_30, bb9_30, hr9_30, ips_30 = 15 * 9 / 13, 1 * 9 / 13, 1 * 9 / 13, 13 / 2
    k9_15, bb9_15, hr9_15, ips_15 = 7 * 9 / 6, 0.0, 0.0, 6 / 1

    weights = {"full": 0.30, "60": 0.25, "30": 0.25, "15": 0.20}
    expected_k9 = k9_full * weights["full"] + k9_60 * weights["60"] + k9_30 * weights["30"] + k9_15 * weights["15"]
    expected_bb9 = bb9_full * weights["full"] + bb9_60 * weights["60"] + bb9_30 * weights["30"] + bb9_15 * weights["15"]
    expected_hr9 = hr9_full * weights["full"] + hr9_60 * weights["60"] + hr9_30 * weights["30"] + hr9_15 * weights["15"]
    expected_ips = ips_full * weights["full"] + ips_60 * weights["60"] + ips_30 * weights["30"] + ips_15 * weights["15"]

    assert result.loc[1, "K9"] == pytest.approx(expected_k9)
    assert result.loc[1, "BB9"] == pytest.approx(expected_bb9)
    assert result.loc[1, "HR9"] == pytest.approx(expected_hr9)
    assert result.loc[1, "IP_per_start"] == pytest.approx(expected_ips)


def test_compute_pitcher_dfs_form_zero_starts_does_not_divide_by_zero():
    # Only a relief appearance - no starts at all for this pitcher.
    rows = _start(1, "g1", "2026-06-20", ["strikeout"], is_starter=False)
    pdf_with_role = pd.DataFrame(rows)

    result = pitcher_form.compute_pitcher_dfs_form(pdf_with_role)

    assert result.empty

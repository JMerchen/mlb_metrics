"""Tests for scripts/build_bts_assistant_page.py: the curated data
payload (NaN handling in particular - the page embeds this as inline JS,
so a stray NaN must become a real null, not the bare `NaN` token pandas/
json would otherwise emit) and the template marker substitution."""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_bts_assistant_page.py"


def _load_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("build_bts_assistant_page", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_docs_data(tmp_path):
    docs_data = tmp_path / "docs_data"
    docs_data.mkdir()

    pd.DataFrame([
        {"date": "2026-06-20", "rank": 1, "name": "Test Player", "predicted_probability": 0.9,
         "combined_probability": 0.85, "grade": "recommended", "status": "pending"},
    ]).to_csv(docs_data / "beat_the_streak_picks.csv", index=False)

    pd.DataFrame([
        {"model_version": "all_time", "metric": "Game_Hit_Probability", "max_picks": 2, "min_probability": 0.77,
         "n_days_resolved": 10, "day_survival_rate": 0.5, "day_survival_rate_ci_low": 0.3,
         "day_survival_rate_ci_high": 0.7, "longest_streak": 5, "current_streak": 2},
    ]).to_csv(docs_data / "beat_the_streak_summary.csv", index=False)

    pd.DataFrame([
        {"model_version": "all_time", "metric": "Game_Hit_Probability", "max_picks": 2, "min_probability": 0.77,
         "n_days_resolved": 10, "day_survival_rate": 0.5, "day_survival_rate_ci_low": 0.3,
         "day_survival_rate_ci_high": 0.7, "longest_streak": 5, "current_streak": 2},
    ]).to_csv(docs_data / "beat_the_streak_summary_by_version.csv", index=False)

    # Batter 1 qualifies (30+ PA) and has a real avg_batting_order; batter
    # 2 qualifies but has never started for their current team (a real
    # NaN avg_batting_order, not a fake 0 - see lineup.py). Batter 3 has
    # too few PA and must be excluded entirely.
    pd.DataFrame([
        {"key_mlbam": 1, "name_first": "Full", "name_last": "Timer", "team": "NYY", "PA_L": 20, "PA_R": 20,
         "WAVE": 0.28, "Game_Hit_Probability": 0.65, "Approach": 0.4, "avg_batting_order": 2.0, "start_rate": 1.0},
        {"key_mlbam": 2, "name_first": "Never", "name_last": "Started", "team": "BOS", "PA_L": 20, "PA_R": 20,
         "WAVE": 0.24, "Game_Hit_Probability": 0.55, "Approach": 0.3, "avg_batting_order": float("nan"), "start_rate": 0.0},
        {"key_mlbam": 3, "name_first": "Too", "name_last": "Few", "team": "SEA", "PA_L": 5, "PA_R": 5,
         "WAVE": 0.30, "Game_Hit_Probability": 0.70, "Approach": 0.5, "avg_batting_order": 1.0, "start_rate": 1.0},
    ]).to_csv(docs_data / "wave.csv", index=False)

    return docs_data


def test_build_payload_replaces_nan_with_none(tmp_path):
    module = _load_module()
    docs_data = _write_docs_data(tmp_path)

    payload = module.build_payload(str(docs_data))

    by_key = {row["name"]: row for row in payload["allQualified"]}
    assert by_key["Full Timer"]["avg_batting_order"] == 2.0
    assert by_key["Never Started"]["avg_batting_order"] is None
    assert "Too Few" not in by_key  # excluded by the PA qualifier


def test_build_payload_picks_the_latest_date_only(tmp_path):
    module = _load_module()
    docs_data = _write_docs_data(tmp_path)
    # Add an older date to the picks log - only the latest date's picks
    # should surface as "today's picks".
    picks = pd.read_csv(docs_data / "beat_the_streak_picks.csv")
    older = picks.copy()
    older["date"] = "2026-06-01"
    older["name"] = "Old Pick"
    pd.concat([older, picks]).to_csv(docs_data / "beat_the_streak_picks.csv", index=False)

    payload = module.build_payload(str(docs_data))

    assert payload["asOf"] == "2026-06-20"
    assert [p["name"] for p in payload["todaysPicks"]] == ["Test Player"]


def test_build_html_embeds_valid_json_between_markers(tmp_path):
    module = _load_module()
    docs_data = _write_docs_data(tmp_path)

    html = module.build_html(str(docs_data))

    start = html.index(module.DATA_START) + len(module.DATA_START)
    end = html.index(module.DATA_END)
    embedded = json.loads(html[start:end])  # must be valid JSON, not a bare `NaN` token
    assert embedded["asOf"] == "2026-06-20"
    assert html.count("<script") == html.count("</script>")

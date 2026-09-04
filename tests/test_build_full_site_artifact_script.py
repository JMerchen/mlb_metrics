"""Tests for scripts/build_full_site_artifact.py's data/text-transform
helpers - the DOM/JS behavior itself (loadAll wiring, the real DFS
optimizer, tab switching, id-collision fixes) is validated separately
against the actual generated page in a real browser (Playwright), not
practical to re-derive as pytest units; these cover the payload curation
and the brace-balanced function extraction/patching that a naive regex
got wrong once already (see the module's own docstring).

Also includes one real end-to-end run of build_html() against this
repo's actual docs/ (the same thing the daily resync job does, ~1s in
practice) - confirmed correct beyond just this structural check via a
manual CLI run plus a real-browser Playwright pass across all 6 tabs,
the player search, and the real DFS optimizer."""

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "build_full_site_artifact.py"


def _load_module():
    sys.path.insert(0, str(REPO_ROOT / "src"))
    spec = importlib.util.spec_from_file_location("build_full_site_artifact", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_records_replaces_nan_and_stringifies_dates():
    module = _load_module()
    df = pd.DataFrame([
        {"a": 1.0, "b": float("nan"), "d": pd.Timestamp("2026-06-20")},
        {"a": 2.0, "b": 3.0, "d": pd.Timestamp("2026-06-21")},
    ])
    records = module._records(df)
    assert records[0]["b"] is None
    assert records[0]["d"] == "2026-06-20"
    assert records[1]["b"] == 3.0


def test_extract_balanced_function_skips_nested_braces():
    module = _load_module()
    js = (
        "function outer(x){\n"
        "if(x){\n"
        "return 1\n"
        "}\n"
        "const obj = {a: 1, b: 2}\n"
        "return obj\n"
        "}\n"
        "function afterward(){ return 2 }\n"
    )
    start, end = module._extract_balanced_function(js, "function outer(x)", "test.js")
    extracted = js[start:end]
    assert extracted.startswith("function outer(x){")
    assert extracted.endswith("}")
    assert "return obj" in extracted
    assert "afterward" not in extracted  # didn't overshoot into the next function


def test_patch_loadcsv_replaces_fetch_based_body():
    module = _load_module()
    js = (
        'let x = 1\n'
        'async function loadCSV(path){\n'
        'const response = await fetch(`${path}?t=${Date.now()}`, {cache:"no-store"})\n'
        'if(!response.ok){\n'
        'throw new Error(`Failed to load ${path}: ${response.status}`)\n'
        '}\n'
        'const text = await response.text()\n'
        'return Papa.parse(text, {header:true, skipEmptyLines:true}).data\n'
        '}\n'
        'async function loadAll(){ wave = await loadCSV("./data/wave.csv") }\n'
    )
    patched = module._patch_loadcsv(js, "test.js")
    assert "fetch(" not in patched
    assert "Papa.parse" not in patched
    assert "window.SITE_DATA" in patched
    assert 'loadCSV("./data/wave.csv")' in patched  # call sites untouched


def test_build_data_payload_keys_by_csv_filename_stem(tmp_path):
    module = _load_module()
    docs_data = tmp_path / "data"
    docs_data.mkdir()
    pd.DataFrame([{"key_mlbam": 1, "WAVE": 0.25}]).to_csv(docs_data / "wave.csv", index=False)
    pd.DataFrame([{"team": "NYY", "Confidence": 1.1}]).to_csv(docs_data / "confidence.csv", index=False)

    payload = module.build_data_payload(str(docs_data))

    assert set(payload.keys()) == {"wave", "confidence"}
    assert payload["wave"][0]["key_mlbam"] == 1
    json.dumps(payload, allow_nan=False)  # must be real JSON, no bare NaN


def test_build_html_end_to_end_against_real_docs(tmp_path):
    """Runs the real build against this repo's actual docs/ - the same
    thing the daily resync job does - and checks the output is
    structurally sound (every template marker resolved, both embedded
    JSON payloads parse, every <script> is balanced)."""
    module = _load_module()
    html = module.build_html(str(REPO_ROOT / "docs"))

    # __SITE_DATA_JSON__/__END_SITE_DATA_JSON__ (like the assistant
    # fragment's own __DATA_JSON__/__END_DATA_JSON__) are deliberately
    # KEPT in the output as comment delimiters around the embedded JSON,
    # not "unresolved" - excluded here, same as the assistant markers.
    stripped = html
    for marker in ["/*__SITE_DATA_JSON__*/", "/*__END_SITE_DATA_JSON__*/", "/*__DATA_JSON__*/", "/*__END_DATA_JSON__*/"]:
        stripped = stripped.replace(marker, "")
    unresolved = [
        marker for marker in [
            "__STYLE_CSS__", "__METHODOLOGY_STYLE__", "__INDEX_CONTENT__", "__DFS_CONTENT__",
            "__NFL_CONTENT__", "__AGE_CURVES_CONTENT__", "__METHODOLOGY_CONTENT__",
            "__ASSISTANT_CONTENT__", "__PAGE_SCRIPTS__",
        ]
        if marker in stripped
    ]
    # Comparing the boolean/list, not `marker not in stripped` directly -
    # a plain `assert x not in huge_string` makes pytest's assertion
    # rewriter build a difflib comparison over the whole (multi-MB)
    # string on failure, which is pathologically slow.
    assert unresolved == []

    i = html.index("/*__SITE_DATA_JSON__*/") + len("/*__SITE_DATA_JSON__*/")
    j = html.index("/*__END_SITE_DATA_JSON__*/")
    site_data = json.loads(html[i:j])
    assert "wave" in site_data and "nfl_bestball_rankings" in site_data

    assert html.count("<script") == html.count("</script>")
    # Top-level pages: Assistant, MLB (an umbrella - see below), NFL,
    # Methodology.
    missing_sections = [
        pid for pid in ["page-assistant", "page-mlb", "page-nfl", "page-methodology"]
        if f'<section id="{pid}"' not in html
    ]
    assert missing_sections == []  # see the note above `in`/`not in` on the full `html` string in assert

    # Predictive Metrics/DFS Rankings/Age Curves are nested subpages under
    # the MLB umbrella tab, not their own top-level sections.
    missing_mlb_subpages = [
        pid for pid in ["mlb-predictive", "mlb-dfs", "mlb-age-curves"]
        if f'<div id="{pid}"' not in html
    ]
    assert missing_mlb_subpages == []

    # The real collision this module's docstring documents fixing.
    for dup_id in ["playerSearch", "playerChoices", "playerResult"]:
        assert html.count(f'id="{dup_id}"') == 1

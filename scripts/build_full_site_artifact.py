"""Builds the standalone HTML for the full-site Claude Artifact - the
whole real docs/ dashboard (MLB Picks and Stats, Age Curves, Payroll,
NFL, Methodology) plus the Beat the Streak Assistant chat, all under one
claude.ai URL. Ports the real docs/*.js/*.html/*.css almost verbatim
(same code) rather than reimplementing:

1. Every docs/data/*.csv is pre-parsed to JSON and embedded (an artifact
   page's own network is blocked, so the real `fetch("./data/X.csv")`
   calls the site normally makes cannot work here - see loadCSV below).
2. Each page script's `loadCSV(path)` is redefined to look up the
   embedded JSON by filename instead of fetching - a single, mechanical,
   well-contained swap; everything downstream (rendering, the NFL Draft
   Assistant) is untouched, so it behaves
   exactly like the live site given the same data.
3. No DOM ids or global function names collide across the pages, which
   all share one DOM and global scope here (the Age Curves page uses its
   own age-prefixed ids and functions for that reason).
4. mlb_render.js's headshot/cap URL helpers (real MLB CDN URLs) return
   "" - an artifact page can't load images from external hosts, so the
   pictures fall back to initials and team codes instead of 404ing.
Run manually, or by the daily resync job (same idea as
build_bts_assistant_page.py, which this supersedes) after the daily
pipeline refreshes docs/data/*.csv.

Usage:
    python scripts/build_full_site_artifact.py [--output PATH]
"""

import argparse
import glob
import importlib.util
import json
import os
import re

import pandas as pd

DOCS_DIR = os.path.join(os.path.dirname(__file__), "..", "docs")
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "full_site_template.html")
DATA_START = "/*__SITE_DATA_JSON__*/"
DATA_END = "/*__END_SITE_DATA_JSON__*/"


def _load_bts_assistant_module():
    """scripts/ isn't a package - load build_bts_assistant_page.py by
    path (same pattern the project's own script tests use) so its
    already-tested build_html() can be reused verbatim for the Assistant
    tab's curated data + markup, instead of re-deriving it here."""
    path = os.path.join(os.path.dirname(__file__), "build_bts_assistant_page.py")
    spec = importlib.util.spec_from_file_location("build_bts_assistant_page", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records(df: pd.DataFrame) -> list:
    """See build_bts_assistant_page.py's identical helper - real NaN
    cells become real null, not a bare `NaN` token, and Timestamp/date
    columns are stringified first (also not JSON-serializable)."""
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].dt.strftime("%Y-%m-%d")
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols):
        df[numeric_cols] = df[numeric_cols].round(4)
    return df.astype(object).where(df.notna(), None).to_dict("records")


def build_data_payload(docs_data_dir: str) -> dict:
    """One key per docs/data/*.csv (filename stem -> parsed rows) - every
    CSV any ported page script reads via loadCSV(), keyed exactly how
    the loadCSV patch below looks them up."""
    payload = {}
    for path in sorted(glob.glob(os.path.join(docs_data_dir, "*.csv"))):
        stem = os.path.splitext(os.path.basename(path))[0]
        df = pd.read_csv(path)
        payload[stem] = _records(df)
    # JSON data files (the NFL what-if panel's) ride along as-is, looked up
    # by the page script before it tries the network.
    for path in sorted(glob.glob(os.path.join(docs_data_dir, "*.json"))):
        with open(path) as fh:
            payload[os.path.splitext(os.path.basename(path))[0]] = json.load(fh)
    return payload


LOADCSV_REPLACEMENT = (
    'async function loadCSV(path){\n'
    'const key = path.replace(/^\\.\\/data\\//, "").replace(/\\.csv.*$/, "");\n'
    'return (window.SITE_DATA && window.SITE_DATA[key]) || [];\n'
    '}'
)


def _extract_balanced_function(js: str, signature: str, filename: str) -> tuple[int, int]:
    """(start, end) character offsets of the function whose header is
    `signature` (e.g. "async function loadCSV(path)"), found by counting
    braces from its opening `{` rather than a regex - these files nest
    their own `{...}` blocks (an `if`, an object literal), so a naive
    "up to the next closing brace" match stops at the FIRST nested one,
    not the function's real end (a real bug this caught: the original
    regex-based loadCSV patch left the rest of the old function body
    dangling after the replacement, a JS syntax error)."""
    start = js.find(signature)
    if start == -1:
        raise ValueError(f"{filename}: expected to find `{signature}` but didn't")
    body_start = js.index("{", start)
    depth = 0
    for i in range(body_start, len(js)):
        if js[i] == "{":
            depth += 1
        elif js[i] == "}":
            depth -= 1
            if depth == 0:
                return start, i + 1
    raise ValueError(f"{filename}: `{signature}` has no balanced closing brace")


def _patch_loadcsv(js: str, filename: str) -> str:
    start, end = _extract_balanced_function(js, "async function loadCSV(path)", filename)
    return js[:start] + LOADCSV_REPLACEMENT + js[end:]


def _patch_app_js(js: str) -> str:
    return _patch_loadcsv(js, "app.js")


def _patch_mlb_render_js(js: str) -> str:
    """mlb_render.js's headshot/cap URLs point at MLB's CDNs
    (img.mlbstatic.com, mlbstatic.com), which an artifact page can't load.
    Both return "" here instead, so every picture falls back to its text
    (initials, team code) rather than a broken image. Matches each
    function by its exact signature and raises if either has changed
    shape."""
    for signature in ("function mlbHeadshotUrl(id)", "function mlbLogoUrl(team, onLight = false)"):
        start, end = _extract_balanced_function(js, signature, "mlb_render.js")
        js = js[:start] + f'{signature}{{\nreturn ""\n}}' + js[end:]
    return js


def _patch_nfl_stats_js(js: str) -> str:
    """Same as _patch_mlb_render_js for the NFL Stats tab's logo (ESPN) and
    headshot (nfl.com) URLs."""
    for signature in ("function nflLogoUrl(team)", "function nflHeadshotUrl(url)"):
        start, end = _extract_balanced_function(js, signature, "nfl_stats.js")
        js = js[:start] + f'{signature}{{\nreturn ""\n}}' + js[end:]
    return js


PAGE_JS_PATCHERS = {
    "mlb_render.js": _patch_mlb_render_js,
    "nfl_stats.js": _patch_nfl_stats_js,
    "app.js": _patch_app_js,
    # payroll.js before nfl.js: the NFL page's Payroll tab starts its view.
    "payroll.js": lambda js: _patch_loadcsv(js, "payroll.js"),
    "nfl.js": lambda js: _patch_loadcsv(js, "nfl.js"),
    # mlb_sim_view.js after app.js: it reads probable starters with app.js's loadCSV.
    "mlb_sim_view.js": lambda js: js,
    "age-curves.js": lambda js: _patch_loadcsv(js, "age-curves.js"),
}

# Verbatim - no loadCSV/network/image dependency to patch. Each file gets
# its own <script> tag (see page_scripts below), in this order, before the
# patched page scripts that call into them: teams.js/summary_render.js
# first (team nicknames and the plain-language picks rendering, used by the
# Assistant tab and the MLB/NFL tabs), then the shared rendering for the
# Age Curves, Prospects (consensus_board.js), Player Props, Stats units and
# What-if panels.
VERBATIM_JS = [
    "teams.js", "summary_render.js",
    "age_render.js", "consensus_board.js", "player_props.js", "nfl_units.js", "prop_cards.js",
    "nfl_whatif.js", "nfl_sim.js", "payroll_render.js", "mlb_sim.js",
]


REAL_SITE_ONLY_NAV_MARKERS = (
    # The MLB umbrella's own sub-header + Picks/Stats/DFS/Age
    # Curves subtabs, and NFL's own sub-header - added to the real
    # docs/*.html pages so their own nav mirrors the artifact's MLB/NFL
    # umbrella tabs when browsing the real site directly. The merged
    # artifact already builds its own copy of both higher up the DOM (see
    # full_site_template.html), so these must NOT also flow into the
    # page's own extracted content - besides being a visual duplicate,
    # <a href="./stats.html">-style links wouldn't work inside the
    # single-page artifact anyway.
    '<div class="sportHeader">',
    '<div class="tabGroup" id="mlbSubTabs">',
)


def _skip_real_site_only_nav(html: str, start: int) -> int:
    pos = start
    for marker in REAL_SITE_ONLY_NAV_MARKERS:
        candidate = html[pos:].lstrip()
        if candidate.startswith(marker):
            marker_pos = pos + (len(html[pos:]) - len(candidate))
            pos = html.index("</div>", marker_pos) + len("</div>")
    return pos


def _extract_page_content(html: str, filename: str) -> str:
    """Everything between the shared header/nav (dropped - the merged
    page builds ONE top-level nav instead) and the trailing <script>
    tags (dropped - scripts are assembled separately, once, in a fixed
    dependency order)."""
    # The nav's last link (the "How it works" tab, formerly labelled
    # "Methodology"). Matched with its href so a "how it works" link in a
    # page's own text can't be mistaken for it.
    nav_marker = 'href="./methodology.html">How it works</a>'
    nav_end = html.find(nav_marker)
    if nav_end == -1:
        raise ValueError(f"{filename}: expected nav marker {nav_marker!r} not found")
    # Two </div> closes follow the nav in every page: the pageTabs
    # wrapper, then siteHeader itself.
    closes = 0
    pos = nav_end
    while closes < 2:
        pos = html.index("</div>", pos) + len("</div>")
        closes += 1
    content_start = _skip_real_site_only_nav(html, pos)
    # methodology.html has no <script> tag at all (purely static) -
    # fall back to </body> for that one page.
    script_pos = html.find("<script", content_start)
    content_end = script_pos if script_pos != -1 else html.index("</body>", content_start)
    content = html[content_start:content_end]
    # The outer .container's own closing </div> (paired with the opening
    # tag stripped along with siteHeader) is the LAST </div> before the
    # scripts - drop it too so this content sits cleanly inside this
    # page's own wrapper <section> instead of an orphaned close tag.
    last_div = content.rstrip().rfind("</div>")
    if last_div == -1 or not content[last_div:].strip() == "</div>":
        raise ValueError(f"{filename}: expected a trailing </div> (the .container close) at the end of the page content")
    return content[:last_div]


def _extract_methodology_style(html: str) -> str:
    match = re.search(r"<style>(.*?)</style>", html, re.S)
    if not match:
        raise ValueError("methodology.html: expected a page-scoped <style> block")
    return match.group(1)


def build_html(docs_dir: str) -> str:
    data_payload = build_data_payload(os.path.join(docs_dir, "data"))

    def read(name):
        with open(os.path.join(docs_dir, name)) as f:
            return f.read()

    # index.html is now the real site's own "Today's Summary" landing page
    # (see docs/today.js) - not part of this build at all, since the
    # artifact already has its own equivalent (the Assistant tab, with a
    # live chat on top). The MLB Picks content this artifact
    # wants lives at predictive.html (and its Stats at stats.html).
    index_content = _extract_page_content(read("predictive.html"), "predictive.html")
    stats_content = _extract_page_content(read("stats.html"), "stats.html")
    nfl_content = _extract_page_content(read("nfl.html"), "nfl.html")
    age_curves_content = _extract_page_content(read("age-curves.html"), "age-curves.html")
    payroll_content = _extract_page_content(read("payroll.html"), "payroll.html")
    methodology_html = read("methodology.html")
    methodology_content = _extract_page_content(methodology_html, "methodology.html")
    methodology_style = _extract_methodology_style(methodology_html)

    js_blocks = []
    for filename in VERBATIM_JS:
        js_blocks.append(read(filename))
    for filename, patcher in PAGE_JS_PATCHERS.items():
        js_blocks.append(patcher(read(filename)))

    style_css = read("style.css")

    # The Assistant tab reuses build_bts_assistant_page.py's own
    # build_html() wholesale (its already-tested curated payload +
    # markup + chat script, complete and self-contained) rather than
    # re-deriving that logic here - it becomes one section among the
    # real site's other tabs instead of standing alone.
    assistant_module = _load_bts_assistant_module()
    # include_style=False: this outer template already embeds style.css
    # once, globally (see below) - the Assistant fragment's own .wrap/
    # .stat-row/.panel/... CSS now lives there too, so it doesn't need
    # its own copy embedded a second time.
    assistant_content = assistant_module.build_html(
        os.path.join(docs_dir, "data"), include_style=False, include_shared_js=False
    )

    with open(TEMPLATE_PATH) as f:
        template = f.read()

    template = template.replace("/*__STYLE_CSS__*/", style_css)
    template = template.replace("/*__METHODOLOGY_STYLE__*/", methodology_style)
    template = template.replace("<!--__INDEX_CONTENT__-->", index_content)
    template = template.replace("<!--__STATS_CONTENT__-->", stats_content)
    template = template.replace("<!--__NFL_CONTENT__-->", nfl_content)
    template = template.replace("<!--__AGE_CURVES_CONTENT__-->", age_curves_content)
    template = template.replace("<!--__PAYROLL_CONTENT__-->", payroll_content)
    template = template.replace("<!--__METHODOLOGY_CONTENT__-->", methodology_content)
    template = template.replace("<!--__ASSISTANT_CONTENT__-->", assistant_content)
    # Each file gets its OWN <script> tag, matching the real site's
    # separate <script src="..."> tags - NOT one shared tag. A real bug
    # caught in testing: concatenating them into a single <script> hoists
    # every top-level `function` declaration (loadAll, buildTable, ...)
    # to the top of that ONE shared scope before any code runs, so the
    # LAST file's same-named function silently wins even for an EARLIER
    # file's own bottom-of-file call to itself - app.js's `loadAll()`
    # call was actually invoking age-curves.js's loadAll, since both are
    # named `loadAll` and age-curves.js's declaration (textually later)
    # hoisted over app.js's. Separate <script> tags keep each file's
    # declarations properly scoped to their own execution unit, exactly
    # like the real multi-file site already relies on.
    page_scripts = "\n".join(f"<script>\n{block}\n</script>" for block in js_blocks)
    template = template.replace("<!--__PAGE_SCRIPTS__-->", page_scripts)

    start = template.index(DATA_START) + len(DATA_START)
    end = template.index(DATA_END)
    data_json = json.dumps(data_payload, separators=(",", ":"), allow_nan=False)
    return template[:start] + data_json + template[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs-dir", default=DOCS_DIR)
    parser.add_argument("--output", default="/tmp/full_site_assistant.html")
    args = parser.parse_args()

    html = build_html(args.docs_dir)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"Wrote {args.output} ({len(html)} bytes).")


if __name__ == "__main__":
    main()

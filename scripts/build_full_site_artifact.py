"""Builds the standalone HTML for the full-site Claude Artifact - the
whole real docs/ dashboard (Predictive Metrics, Age Curves, DFS Rankings,
NFL, Methodology) plus the Beat the Streak Assistant chat, all under one
claude.ai URL. Ports the real docs/*.js/*.html/*.css almost verbatim
(same code, same tested DFS optimizer) rather than reimplementing:

1. Every docs/data/*.csv is pre-parsed to JSON and embedded (an artifact
   page's own network is blocked, so the real `fetch("./data/X.csv")`
   calls the site normally makes cannot work here - see loadCSV below).
2. Each page script's `loadCSV(path)` is redefined to look up the
   embedded JSON by filename instead of fetching - a single, mechanical,
   well-contained swap; everything downstream (rendering, the real DFS
   knapsack solver, the NFL Draft Assistant) is untouched, so it behaves
   exactly like the live site given the same data.
3. Two real DOM-id/function-name collisions get fixed (age-curves.js's
   own `searchPlayer`/`showPlayer`/`#playerSearch`/`#playerChoices`/
   `#playerResult` clash with app.js's identically-named ones - both
   pages now coexist in one DOM/global scope, which they never did
   live) - renamed to an `age`-prefixed set, verified to be the ONLY
   collision across the whole site (checked every id="" and every
   function used from an onclick/oninput attribute for duplicates).
4. `playerImage`/`teamLogo` (real MLB headshot/logo CDN URLs) are
   replaced with a local inline-SVG placeholder - an artifact page can't
   load images from arbitrary external hosts either, so those would
   otherwise silently 404.
5. `nfl_dfs_solver.js` is NOT ported: grepping the whole docs/ tree finds
   no page that actually loads it (no NFL DFS tab exists in nfl.html
   today - only its own test file references it) - it isn't part of the
   live site to port, and skipping it also removes what would otherwise
   be a second solver-globals collision with dfs_solver.js.

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

# A generic, local, no-network placeholder - real player photos/team
# logos come from external CDNs (img.mlbstatic.com, mlbstatic.com) an
# artifact page can't load. One neutral silhouette works for both use
# sites (playerImage/teamLogo both just interpolate this into an <img
# src="...">, see _patch_app_js below).
PLACEHOLDER_IMAGE = (
    "data:image/svg+xml;base64,"
    "PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCA0"
    "OCA0OCI+PGNpcmNsZSBjeD0iMjQiIGN5PSIyNCIgcj0iMjQiIGZpbGw9IiMyNjJiMzQiLz48"
    "Y2lyY2xlIGN4PSIyNCIgY3k9IjE4IiByPSI4IiBmaWxsPSIjNmI2NDU5Ii8+PHBhdGggZD0i"
    "TTggNDRjMC05IDcuMi0xNiAxNi0xNnMxNiA3IDE2IDE2IiBmaWxsPSIjNmI2NDU5Ii8+PC9z"
    "dmc+"
)


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
    js = _patch_loadcsv(js, "app.js")
    # playerImage/teamLogo point at real external CDNs (img.mlbstatic.com,
    # mlbstatic.com) an artifact page can't load - swap both for the
    # local placeholder. Matches each function's exact real body (the
    # `return \`https://...\`` line's real interpolation differs between
    # the two, so patched separately) rather than a loose regex, so this
    # raises loudly if either function's source ever changes shape.
    player_image_old = (
        'function playerImage(id){\n\n'
        'if(\n!id\n){\n\n'
        'return ""\n\n'
        '}\n\n'
        'return `https://img.mlbstatic.com/mlb-photos/image/upload/w_240,q_auto:best/v1/people/${String(id).trim()}/headshot/67/current`\n\n'
        '}'
    )
    if player_image_old not in js:
        raise ValueError("app.js: playerImage() body not found where expected - check the real source hasn't changed shape")
    js = js.replace(player_image_old, f'function playerImage(id){{\nreturn id ? "{PLACEHOLDER_IMAGE}" : "";\n}}')

    start, end = _extract_balanced_function(js, "function teamLogo(team)", "app.js")
    original = js[start:end]
    ids_match = re.search(r"const ids = \{.*?\n\}", original, re.S)
    if not ids_match:
        raise ValueError("app.js: teamLogo()'s `ids` team-code table not found where expected")
    new_team_logo = (
        f"function teamLogo(team){{\n{ids_match.group(0)}\n"
        f'return ids[String(team).trim().toUpperCase()] ? "{PLACEHOLDER_IMAGE}" : "";\n}}'
    )
    js = js[:start] + new_team_logo + js[end:]
    return js


def _patch_age_curves_js(js: str) -> str:
    js = _patch_loadcsv(js, "age-curves.js")
    # Real collision with app.js: both files define searchPlayer/
    # showPlayer with DIFFERENT behavior (hitter/pitcher search vs. age-
    # curve search) - both are onclick-reachable once merged into one
    # page/global scope, so whichever loaded last would silently win for
    # BOTH tabs. Rename age-curves.js's pair (word-boundary so
    # e.g. `searchPlayerX` is untouched, though none exists).
    js = re.sub(r"\bsearchPlayer\b", "ageSearchPlayer", js)
    js = re.sub(r"\bshowPlayer\b", "ageShowPlayer", js)
    # Same story for the 3 DOM ids app.js's Player Search section already
    # owns (#playerSearch/#playerChoices/#playerResult) - verified via a
    # full id="" audit across every docs/*.html to be the only collision.
    for old, new in [("playerSearch", "ageplayerSearch"), ("playerChoices", "ageplayerChoices"), ("playerResult", "ageplayerResult")]:
        js = re.sub(rf'"{old}"', f'"{new}"', js)
    return js


PAGE_JS_PATCHERS = {
    "app.js": _patch_app_js,
    "dfs.js": lambda js: _patch_loadcsv(js, "dfs.js"),
    "nfl.js": lambda js: _patch_loadcsv(js, "nfl.js"),
    "age-curves.js": _patch_age_curves_js,
}

# Verbatim - no loadCSV/network/image dependency to patch (see module
# docstring point 5 for why nfl_dfs_solver.js is deliberately excluded).
VERBATIM_JS = ["dfs_solver.js", "nfl_draft_assistant.js"]


def _extract_page_content(html: str, filename: str) -> str:
    """Everything between the shared header/nav (dropped - the merged
    page builds ONE top-level nav instead) and the trailing <script>
    tags (dropped - scripts are assembled separately, once, in a fixed
    dependency order)."""
    nav_end = html.find(">Methodology</a>")
    if nav_end == -1:
        raise ValueError(f"{filename}: expected nav marker '>Methodology</a>' not found")
    # Two </div> closes follow the nav in every page: the pageTabs
    # wrapper, then siteHeader itself.
    closes = 0
    pos = nav_end
    while closes < 2:
        pos = html.index("</div>", pos) + len("</div>")
        closes += 1
    content_start = pos
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

    index_content = _extract_page_content(read("index.html"), "index.html")
    dfs_content = _extract_page_content(read("dfs.html"), "dfs.html")
    nfl_content = _extract_page_content(read("nfl.html"), "nfl.html")
    age_curves_content = _extract_page_content(read("age-curves.html"), "age-curves.html")
    methodology_html = read("methodology.html")
    methodology_content = _extract_page_content(methodology_html, "methodology.html")
    methodology_style = _extract_methodology_style(methodology_html)

    # Rewrite the DOM ids/onclick text this SPECIFIC page's static markup
    # references, matching age-curves.js's own renames above.
    for old, new in [
        ('id="playerSearch"', 'id="ageplayerSearch"'),
        ('id="playerChoices"', 'id="ageplayerChoices"'),
        ('id="playerResult"', 'id="ageplayerResult"'),
        ("searchPlayer()", "ageSearchPlayer()"),
    ]:
        age_curves_content = age_curves_content.replace(old, new)

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
    assistant_content = assistant_module.build_html(os.path.join(docs_dir, "data"))

    with open(TEMPLATE_PATH) as f:
        template = f.read()

    template = template.replace("/*__STYLE_CSS__*/", style_css)
    template = template.replace("/*__METHODOLOGY_STYLE__*/", methodology_style)
    template = template.replace("<!--__INDEX_CONTENT__-->", index_content)
    template = template.replace("<!--__DFS_CONTENT__-->", dfs_content)
    template = template.replace("<!--__NFL_CONTENT__-->", nfl_content)
    template = template.replace("<!--__AGE_CURVES_CONTENT__-->", age_curves_content)
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

"""The scheduled jobs save what they build.

Four times a file the NFL page reads was rebuilt by the weekly job and
then thrown away, because the job's commit step named its files one by one
and the new one was not on the list. The step now takes them by pattern;
this checks that every data file the page loads is matched by it."""

import fnmatch
import os
import re

ROOT = os.path.join(os.path.dirname(__file__), "..")


def _read(*parts):
    with open(os.path.join(ROOT, *parts)) as f:
        return f.read()


def _commit_patterns(workflow_text):
    """The paths (some with *) the job's commit step adds."""
    step = workflow_text[workflow_text.index("git add"):]
    return set(re.findall(r"(?:docs/data|data)/[\w./*-]+", step))


def test_every_data_file_the_nfl_page_loads_is_saved_by_the_weekly_job():
    page = _read("docs", "nfl.js")
    loaded = set(re.findall(r'\./data/(nfl_[\w]+\.csv)', page))
    loaded |= {f"{name}.json" for name in re.findall(r'loadNflWhatIfFile\("([\w]+)"\)', page)}
    loaded |= {f"{name}.json" for name in re.findall(r'file: "([\w]+)"', page)}
    assert "nfl_matchup_history.csv" in loaded and "nfl_play_calling.json" in loaded          # the two that were lost
    patterns = _commit_patterns(_read(".github", "workflows", "nfl_weekly_update.yml"))
    # The draft board is another job's.
    missing = sorted(name for name in loaded - {"nfl_draft_board.csv"}
                     if not any(fnmatch.fnmatch(f"docs/data/{name}", pattern) for pattern in patterns))
    assert not missing, f"the weekly NFL job builds but never commits: {missing}"


def test_the_pattern_reader_understands_the_commit_step():
    patterns = _commit_patterns("run: |\n  git add docs/data/a.csv data/raw/nfl/\n  for f in docs/data/nfl_* docs/data/bet_evidence.json; do\n")
    assert patterns == {"docs/data/a.csv", "data/raw/nfl/", "docs/data/nfl_*", "docs/data/bet_evidence.json"}
    assert fnmatch.fnmatch("docs/data/nfl_matchup_history.csv", "docs/data/nfl_*")

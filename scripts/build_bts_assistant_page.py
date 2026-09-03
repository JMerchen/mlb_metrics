"""Builds the standalone HTML for the "Beat the Streak Assistant" Claude
Artifact - a read-only chat page (uses the Artifact `sample` capability,
see https://code.claude.com/docs) that answers visitor questions about
the site's real, currently-live Beat the Streak, MLB Automated Game
Picks, and NFL Automated Game Picks data. It cannot write to this repo
or modify any pick/model - it only ever sees a curated JSON snapshot of
already-public docs/data/*.csv files, embedded into the page at build
time.

Run manually, or by a scheduled job that re-runs this after the daily
pipeline (.github/workflows/daily_update.yml, 10:23 UTC) has refreshed
docs/data/*.csv, then republishes the artifact from the output file so
the assistant's data never drifts far from what's live on the dashboard.

Usage:
    python scripts/build_bts_assistant_page.py [--output PATH]
"""

import argparse
import json
import os

import pandas as pd

TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "bts_assistant_template.html")
DATA_START = "/*__DATA_JSON__*/"
DATA_END = "/*__END_DATA_JSON__*/"


def _records(df: pd.DataFrame) -> list:
    """df.round(4).to_dict('records'), with real NaN cells replaced by
    None/null instead of a bare `NaN` token - this payload is inlined as
    JS source (not JSON.parse'd), where `NaN` is a valid identifier, so
    it would still run, but it isn't valid JSON and isn't what
    JSON.stringify(NaN) (called on this data elsewhere in the page, e.g.
    tool results) round-trips to anyway (that already silently becomes
    null) - null everywhere is simpler and more portable than relying on
    that implicit coercion."""
    rounded = df.round(4)
    return rounded.astype(object).where(rounded.notna(), None).to_dict("records")


GAME_PICK_COLUMNS = [
    "date", "home_team", "away_team", "predicted_winner", "predicted_probability",
    "above_threshold", "status", "bet_side", "bet_team", "bet_units", "bet_profit_units",
    "market_predicted_winner_probability",
]


def _game_picks_section(docs_data_dir: str, prefix: str, extra_columns: list | None = None) -> dict:
    """Curates one sport's Automated Game Picks CSVs (`{prefix}_picks.csv`/
    `{prefix}_summary.csv`/`{prefix}_summary_by_version.csv`) into
    {asOf, upcomingPicks, recentHistory, summary, summaryByVersion,
    fullHistory} - the same "small slice in the prompt, full table behind
    a tool" split beat_the_streak/wave use. `extra_columns` (e.g. NFL's
    ["season", "week"]) are appended to GAME_PICK_COLUMNS for that sport
    only, since MLB has no such concept."""
    columns = GAME_PICK_COLUMNS + (extra_columns or [])
    picks = pd.read_csv(os.path.join(docs_data_dir, f"{prefix}_picks.csv"), parse_dates=["date"])
    summary = pd.read_csv(os.path.join(docs_data_dir, f"{prefix}_summary.csv"))
    by_version = pd.read_csv(os.path.join(docs_data_dir, f"{prefix}_summary_by_version.csv"))

    if picks.empty:
        return {
            "asOf": None, "upcomingPicks": [], "recentHistory": [],
            "summary": _records(summary), "summaryByVersion": _records(by_version), "fullHistory": [],
        }

    latest_date = picks["date"].max()
    # A real Timestamp isn't JSON-serializable (and _records's .round(4)
    # is meaningless on a datetime column, just a spurious warning) -
    # stringify before selecting columns, same "date" shape
    # beat_the_streak_picks.csv's rows already use in this payload.
    picks = picks.assign(date=picks["date"].dt.strftime("%Y-%m-%d"))
    latest_date_str = latest_date.strftime("%Y-%m-%d")
    upcoming = picks[picks["date"] == latest_date_str].sort_values(["home_team"])[columns]
    history = picks[picks["date"] < latest_date_str].sort_values("date", ascending=False)[columns]

    return {
        "asOf": str(latest_date.date()),
        "upcomingPicks": _records(upcoming),
        "recentHistory": _records(history.head(8)),
        "summary": _records(summary),
        "summaryByVersion": _records(by_version),
        "fullHistory": _records(history),
    }


def build_payload(docs_data_dir: str) -> dict:
    picks = pd.read_csv(os.path.join(docs_data_dir, "beat_the_streak_picks.csv"), parse_dates=["date"])
    summary = pd.read_csv(os.path.join(docs_data_dir, "beat_the_streak_summary.csv"))
    by_version = pd.read_csv(os.path.join(docs_data_dir, "beat_the_streak_summary_by_version.csv"))
    wave = pd.read_csv(os.path.join(docs_data_dir, "wave.csv"))

    if picks.empty:
        todays_picks: list = []
        as_of = pd.Timestamp.today().date().isoformat()
    else:
        today = picks["date"].max()
        todays_picks = _records(
            picks[picks["date"] == today]
            .sort_values("rank")[["rank", "name", "predicted_probability", "combined_probability", "grade", "status"]]
        )
        as_of = str(today.date())

    qualified = wave[(wave["PA_L"] + wave["PA_R"]) >= 30].copy()
    qualified["PA"] = qualified["PA_L"] + qualified["PA_R"]
    qualified["name"] = qualified["name_first"] + " " + qualified["name_last"]
    all_q = qualified[
        ["name", "team", "WAVE", "Game_Hit_Probability", "Approach", "PA", "avg_batting_order", "start_rate"]
    ].sort_values("Approach", ascending=False)

    return {
        "asOf": as_of,
        "todaysPicks": todays_picks,
        "streakSummary": _records(summary),
        "streakByVersion": _records(by_version),
        "topLeaders": _records(all_q.head(20)),
        "allQualified": _records(all_q),
        "mlbGamePicks": _game_picks_section(docs_data_dir, "game_picks"),
        "nflGamePicks": _game_picks_section(docs_data_dir, "nfl_game_picks", extra_columns=["season", "week"]),
    }


def build_html(docs_data_dir: str) -> str:
    payload = build_payload(docs_data_dir)
    with open(TEMPLATE_PATH) as f:
        template = f.read()

    start = template.index(DATA_START) + len(DATA_START)
    end = template.index(DATA_END)
    # allow_nan=False: fail the build loudly if any NaN slipped past
    # _records above, rather than silently emitting an invalid `NaN`
    # token into the published page.
    return template[:start] + json.dumps(payload, separators=(",", ":"), allow_nan=False) + template[end:]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs-data-dir", default="docs/data")
    parser.add_argument("--output", default="/tmp/bts_assistant.html")
    args = parser.parse_args()

    html = build_html(args.docs_data_dir)
    with open(args.output, "w") as f:
        f.write(html)
    print(f"Wrote {args.output} ({len(html)} bytes).")


if __name__ == "__main__":
    main()

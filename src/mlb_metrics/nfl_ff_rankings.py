"""Real, current-moment FantasyPros Expert Consensus Rank (ECR) data,
crosswalked to this project's own `player_id` (GSIS id) space - built for
the NFL Bestball Draft Assistant's real "is this pick a reach relative to
consensus" signal (`docs/nfl_draft_assistant.js`).

Not literally raw historical Average Draft Position (real ADP from
completed drafts) - every raw ADP-provider domain (Fantasy Football
Calculator, FantasyPros itself, MyFantasyLeague) is blocked by this
sandbox's network egress policy, and this project has no HTTP client
dependency to reach one directly even outside this sandbox (see
`requirements.txt`). ECR is a real, honestly-labeled substitute: the same
"where does the market expect this player to go" question raw ADP
answers, sourced via `nflreadpy` (already an installed dependency, so no
new package), confirmed live via `scripts/debug_nfl_data.py`
(`.github/workflows/debug_nfl_data.yml`, run 31513644704, 2026-08-11).

Both real source tables (`nfl_data.fetch_ff_rankings`/`fetch_ff_playerids`)
are CURRENT-moment snapshots, not season-keyed historical data - refetched
live at build time (see `scripts/build_nfl_bestball_rankings.py`), never
persisted to `data/raw/nfl/` the way real season stats are."""

import pandas as pd

POSITIONS = ("QB", "RB", "WR", "TE")

# Real confirmed FantasyPros ecr_type value for "best-ball overall" - the
# one real slice (of `['bo', 'bp', 'do', 'dp', 'drk', 'dsf', 'ro', 'rp',
# 'rsf']`, confirmed live via the same debug run cited above) that matches
# this project's bestball focus directly: a real, cross-position-
# comparable consensus rank, already shaped like a real draft pick number
# (QB1 might sit at real ecr 15, the top RB at real ecr 1, etc.) rather
# than a per-position-only rank.
BEST_BALL_OVERALL_ECR_TYPE = "bo"


def compute_ff_rankings_export(ff_rankings_df: pd.DataFrame, ff_playerids_df: pd.DataFrame) -> pd.DataFrame:
    """[player_id, ecr, ecr_sd, ecr_best, ecr_worst] - real FantasyPros
    Best Ball Overall ECR, crosswalked to this project's own `player_id`
    (GSIS id) space.

    Filters `ff_rankings_df` to `ecr_type == "bo"` (see
    `BEST_BALL_OVERALL_ECR_TYPE`'s own comment) and to real QB/RB/WR/TE
    (`POSITIONS` - DST is excluded, matching `nfl_bestball.POSITIONS`'s
    own real scope; this whole bestball feature doesn't rank DST). If more
    than one real `scrape_date` is present (e.g. this export function is
    ever called against accumulated/cached data spanning multiple real
    fetches), only the most recent real snapshot is kept - a real "use the
    freshest consensus" rule, not an average across stale snapshots.

    Crosswalked via `ff_playerids_df`'s real `fantasypros_id` column
    (float-typed in the source - cast to a comparable int before joining
    against `ff_rankings_df`'s own real `id` column) to real `gsis_id`,
    renamed to this project's own `player_id`. Inner merge - a real player
    missing from the real crosswalk (confirmed live: real ~70.5% coverage,
    meaningfully lower than the 99.7% `pfr_id` crosswalk `compute_player_
    snap_share` uses) is simply absent from the output, never a fabricated
    row. `sd`/`best`/`worst` are renamed to `ecr_sd`/`ecr_best`/`ecr_worst`
    to avoid colliding with this project's own generic-sounding column
    names elsewhere."""
    rankings = ff_rankings_df[
        (ff_rankings_df["ecr_type"] == BEST_BALL_OVERALL_ECR_TYPE) & (ff_rankings_df["pos"].isin(POSITIONS))
    ].copy()

    if "scrape_date" in rankings.columns and rankings["scrape_date"].notna().any():
        latest = rankings["scrape_date"].max()
        rankings = rankings[rankings["scrape_date"] == latest]

    crosswalk = ff_playerids_df.dropna(subset=["fantasypros_id", "gsis_id"]).copy()
    crosswalk["fantasypros_id"] = crosswalk["fantasypros_id"].astype(int)
    crosswalk = crosswalk.drop_duplicates("fantasypros_id")[["fantasypros_id", "gsis_id"]]

    merged = rankings.merge(crosswalk, left_on="id", right_on="fantasypros_id", how="inner")
    merged = merged.rename(columns={"gsis_id": "player_id", "sd": "ecr_sd", "best": "ecr_best", "worst": "ecr_worst"})
    return merged[["player_id", "ecr", "ecr_sd", "ecr_best", "ecr_worst"]].drop_duplicates("player_id")

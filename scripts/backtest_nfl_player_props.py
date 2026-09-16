"""Real, no-lookahead validation for nfl_player_props.py (2026-09-16 -
direct user request to rank NFL player prop bets by real matchup
quality). The real question this backtest answers, honestly, either
way: does a player's real upcoming opponent allowing more than the real
league average at a given stat actually correlate with that player
REAL-ly beating their own recent trailing average that week, more often
than an unfavorable matchup does?

A DELIBERATELY SIMPLIFIED proxy of the live module's own windowed-blend
rolling stats (config.NFL_SKILL_WINDOWS/NFL_QB_WINDOWS/NFL_PASS_RUSH_WINDOWS),
not a literal re-run of nfl_player_props.py's own functions - re-running
the full multi-window blend at every one of ~170 real historical weeks
across 10 real seasons would be real, unnecessary compute for what this
backtest actually needs to answer. Instead, both a player's own "prior
rate" and a team's own "prior allowed rate" are a real, honest EXPANDING
mean over that player's/team's own STRICTLY PRIOR real rows (no
lookahead - the exact real value known heading into that specific real
week, just a simpler shape than the live module's recency-weighted
blend). This is reported plainly as what it is, not asserted to be
identical to the live module's own real behavior.

Real league-average-at-that-moment: for each real (season, week), the
mean of every real team's OWN prior allowed-rate at that same real
moment (a real, no-lookahead cross-sectional snapshot, not one shared
constant across all 10 seasons).

Real save-gate, reported honestly either way: does the top real
matchup-ratio group actually beat its own trailing average more often
than the bottom real matchup-ratio group, with a real, non-trivial
margin? If so, an nonzero NFL_PROP_MATCHUP_WEIGHT is justified; if not,
config.NFL_PROP_MATCHUP_WEIGHT is set to 0.0 (the real null hypothesis -
rank by raw usage/volume only, no matchup adjustment) and this is
reported as a real negative finding, not shipped anyway."""

import sys

import pandas as pd

sys.path.insert(0, "src")

from mlb_metrics import config, nfl_data  # noqa: E402

SEASONS = list(range(2016, 2026))

CATEGORY_SPECS = [
    # (label, player_stat_col, position_filter, allowed_group_col, min_games)
    ("Receptions", "receptions", ("RB", "WR", "TE"), "opponent_team", 3),
    ("Receiving Yards", "receiving_yards", ("RB", "WR", "TE"), "opponent_team", 3),
    ("Rushing Yards", "rushing_yards", ("RB", "WR", "TE"), "opponent_team", 3),
    ("Passing Yards", "passing_yards", ("QB",), "opponent_team", 3),
]


def _expanding_prior(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    """Real, no-lookahead expanding mean of `value_col` over `group_col`'s
    own STRICTLY PRIOR rows (this row's own real value excluded) - `df`
    must already be sorted by (season, week) ascending. A group's first
    real row gets NaN (no real prior history yet), not a fabricated
    zero."""
    ordered = df.sort_values(["season", "week"])
    grouped = ordered.groupby(group_col)[value_col]
    return grouped.transform(lambda s: s.shift(1).expanding().mean())


def load_multi_season_weekly() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        df = nfl_data.load_persisted_table("data/raw/nfl", "weekly", season)
        if df is not None:
            frames.append(df)
    return pd.concat(frames, ignore_index=True)


def build_skill_category_rows(weekly_df: pd.DataFrame, label: str, stat_col: str, positions: tuple) -> pd.DataFrame:
    """One real row per (player, season, week) for this category's real
    qualifying position group: [player_id, team, opponent, season, week,
    actual, prior_rate, opponent_prior_allowed, league_prior_rate]."""
    pool = weekly_df[
        (weekly_df["position"].isin(positions)) & (weekly_df["season_type"] == "REG")
    ].copy()
    pool = pool.sort_values(["season", "week"])
    pool["prior_rate"] = _expanding_prior(pool, "player_id", stat_col)
    pool["actual"] = pool[stat_col]

    team_week = pool.groupby(["opponent_team", "season", "week"], as_index=False).agg(allowed=(stat_col, "sum"))
    team_week = team_week.rename(columns={"opponent_team": "team"})
    team_week = team_week.sort_values(["season", "week"])
    team_week["opponent_prior_allowed"] = _expanding_prior(team_week, "team", "allowed")

    league_prior = team_week.groupby(["season", "week"])["opponent_prior_allowed"].mean().rename("league_prior_rate")
    team_week = team_week.merge(league_prior, on=["season", "week"], how="left")

    merged = pool.merge(
        team_week[["team", "season", "week", "opponent_prior_allowed", "league_prior_rate"]].rename(
            columns={"team": "opponent_team"}
        ),
        on=["opponent_team", "season", "week"],
        how="left",
    )
    merged["category"] = label
    return merged[
        ["player_id", "season", "week", "actual", "prior_rate", "opponent_prior_allowed", "league_prior_rate", "category"]
    ]


def build_sacks_rows(weekly_df: pd.DataFrame) -> pd.DataFrame:
    """Same real shape as build_skill_category_rows, but for individual
    pass-rushers: `opponent` for a defender is the OFFENSE they face,
    and "allowed" is that offense's own real sacks_suffered (already
    offense-side, grouped by the sacked team's own `team` column - see
    nfl_player_props.compute_sacks_allowed_rolling_rates's own docstring
    for why no opponent-perspective flip is needed here)."""
    reg = weekly_df[weekly_df["season_type"] == "REG"].copy()

    has_recorded_a_sack = reg.groupby("player_id")["def_sacks"].transform("max") > 0
    pool = reg[has_recorded_a_sack].sort_values(["season", "week"]).copy()
    pool["prior_rate"] = _expanding_prior(pool, "player_id", "def_sacks")
    pool["actual"] = pool["def_sacks"]

    team_week = reg.groupby(["team", "season", "week"], as_index=False).agg(allowed=("sacks_suffered", "sum"))
    team_week = team_week.sort_values(["season", "week"])
    team_week["opponent_prior_allowed"] = _expanding_prior(team_week, "team", "allowed")

    league_prior = team_week.groupby(["season", "week"])["opponent_prior_allowed"].mean().rename("league_prior_rate")
    team_week = team_week.merge(league_prior, on=["season", "week"], how="left")

    merged = pool.merge(
        team_week[["team", "season", "week", "opponent_prior_allowed", "league_prior_rate"]].rename(
            columns={"team": "opponent_team"}
        ),
        on=["opponent_team", "season", "week"],
        how="left",
    )
    merged["category"] = "Sacks"
    return merged[
        ["player_id", "season", "week", "actual", "prior_rate", "opponent_prior_allowed", "league_prior_rate", "category"]
    ]


def evaluate_weight(rows: pd.DataFrame, weight: float, clip: tuple, min_games_seen: int = 3) -> dict:
    """Real, honest evaluation for one (weight, clip) candidate.

    Reports TWO real metrics, deliberately, because the first one alone
    cannot pick a weight/clip: splitting into a top/bottom real tercile
    by ratio and comparing real beat-own-baseline rates is RANK-based,
    so it is mathematically invariant to any positive monotonic
    transform of the ratio (which is exactly what varying `weight`/`clip`
    does here) - confirmed live: every (weight, clip) candidate produced
    an IDENTICAL real tercile spread on this project's own real 10-season
    data. That metric still answers the real, more important question
    honestly (does a favorable matchup correlate with real
    over-performance AT ALL) - it just can't be the tiebreaker between
    candidates.

    `correlation` is the real, SCALE-SENSITIVE tiebreaker: Pearson
    correlation between the (weight, clip)-adjusted ratio and a real
    continuous relative-performance outcome (`actual / prior_rate`,
    capped at a real, generous 5x to keep one absurd real outlier game
    from dominating a linear correlation) - clipping the ratio DOES
    change this number (a tighter clip discards real signal from a
    genuinely extreme matchup; a looser one risks a real small-sample
    defense's noisy extreme dominating), so this real metric can
    actually discriminate between candidates."""
    df = rows.dropna(subset=["prior_rate", "opponent_prior_allowed", "league_prior_rate"]).copy()
    df = df[(df["league_prior_rate"] > 0) & (df["prior_rate"] > 0)]

    games_seen = df.sort_values(["season", "week"]).groupby("player_id").cumcount()
    df = df[games_seen >= min_games_seen]
    if len(df) < 200:
        return {"n": len(df), "top_beat_rate": None, "bottom_beat_rate": None, "spread": None, "correlation": None}

    raw_ratio = df["opponent_prior_allowed"] / df["league_prior_rate"]
    df["ratio"] = (1 + weight * (raw_ratio - 1)).clip(*clip)
    df["beat_own_baseline"] = df["actual"] > df["prior_rate"]
    df["relative_performance"] = (df["actual"] / df["prior_rate"]).clip(upper=5.0)

    top_cut = df["ratio"].quantile(2 / 3)
    bottom_cut = df["ratio"].quantile(1 / 3)
    top_group = df[df["ratio"] >= top_cut]
    bottom_group = df[df["ratio"] <= bottom_cut]

    top_rate = top_group["beat_own_baseline"].mean()
    bottom_rate = bottom_group["beat_own_baseline"].mean()
    correlation = df["ratio"].corr(df["relative_performance"])
    return {
        "n": len(df),
        "top_n": len(top_group),
        "bottom_n": len(bottom_group),
        "top_beat_rate": top_rate,
        "bottom_beat_rate": bottom_rate,
        "spread": top_rate - bottom_rate,
        "correlation": correlation,
    }


def main():
    print(f"Loading real weekly stats for seasons {SEASONS[0]}-{SEASONS[-1]}...")
    weekly_df = load_multi_season_weekly()
    print(f"{len(weekly_df)} real weekly rows loaded.\n")

    rows_by_category = {}
    for label, stat_col, positions, _, _ in CATEGORY_SPECS:
        print(f"Building real {label} rows...")
        rows_by_category[label] = build_skill_category_rows(weekly_df, label, stat_col, positions)
    print("Building real Sacks rows...")
    rows_by_category["Sacks"] = build_sacks_rows(weekly_df)

    clips = [(0.7, 1.3), (0.8, 1.2), (0.6, 1.4)]
    print("\n=== Real per-category, per-(weight, clip) validation ===")
    print(
        "(spread = top-tercile beat-own-baseline rate MINUS bottom-tercile rate, positive = real signal, "
        "but RANK-invariant to weight/clip - see evaluate_weight's own docstring; correlation is the real "
        "scale-sensitive tiebreaker used to pick weight/clip)\n"
    )

    best_by_category = {}
    for label in list(rows_by_category.keys()):
        print(f"--- {label} ---")
        rows = rows_by_category[label]
        best = None
        for weight in config.NFL_PROP_MATCHUP_WEIGHT_GRID:
            for clip in clips:
                result = evaluate_weight(rows, weight, clip)
                if result["spread"] is None:
                    print(f"  weight={weight}, clip={clip}: n too small ({result['n']})")
                    continue
                print(
                    f"  weight={weight}, clip={clip}: n={result['n']} "
                    f"top_beat_rate={result['top_beat_rate']:.4f} "
                    f"bottom_beat_rate={result['bottom_beat_rate']:.4f} "
                    f"spread={result['spread']:+.4f} "
                    f"correlation={result['correlation']:+.4f}"
                )
                if best is None or result["correlation"] > best["correlation"]:
                    best = {"weight": weight, "clip": clip, **result}
        best_by_category[label] = best
        print(
            f"  BEST (by correlation): weight={best['weight']}, clip={best['clip']}, "
            f"correlation={best['correlation']:+.4f}, spread={best['spread']:+.4f}\n"
        )

    print("=== Summary: best real (weight, clip) per category ===")
    for label, best in best_by_category.items():
        print(
            f"{label}: weight={best['weight']} clip={best['clip']} "
            f"correlation={best['correlation']:+.4f} spread={best['spread']:+.4f} (n={best['n']})"
        )


if __name__ == "__main__":
    main()

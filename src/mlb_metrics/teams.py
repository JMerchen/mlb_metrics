"""Team-level metrics: Strength, current, pyth_Strength, SOS, pyth_SOS,
Confidence, pyth_Confidence, offensive_edge, suppression_resistance, and
home-run dependency stats. Unlike the hitter metrics, the original script's
rolling windows here already used `.shift(1)` to avoid same-game leakage -
that pattern is preserved as-is.

Park_Factor (compute_park_factors) is a venue-level signal, not a team-
strength one - it's included here because a team's home venue and the team
itself are a 1:1 mapping in this project's data (see that function's
docstring), so the same per-team row shape fits it directly.
"""

import numpy as np
import pandas as pd

from mlb_metrics import config, helpers

# Team "bases" valuation for offensive_edge counts a walk as a base; this is
# deliberately different from helpers.total_bases (used for hitter WTB/PAVE),
# which does not count walks. Both are faithful to the original script.
_OFFENSIVE_EDGE_BASES_BY_EVENT = {"single": 1, "double": 2, "triple": 3, "home_run": 4, "walk": 1}


def _homer_flag(events: pd.Series) -> pd.Series:
    return np.where(events == "home_run", "homer", "non_homer")


def compute_home_run_stats(data: pd.DataFrame):
    """Returns (for_merge, sus): home-run reliance/dependency stats per team."""
    hhr = data[data["inning_topbot"] == "Bot"][
        ["home_team", "events", "home_score", "post_home_score", "game_id"]
    ].rename(columns={"home_team": "team", "home_score": "pre_score", "post_home_score": "post_score"})
    ahr = data[data["inning_topbot"] == "Top"][
        ["away_team", "events", "away_score", "post_away_score", "game_id"]
    ].rename(columns={"away_team": "team", "away_score": "pre_score", "post_away_score": "post_score"})
    hr = pd.concat([hhr, ahr])
    hr["hr"] = _homer_flag(hr["events"])

    gp = hr[["team", "game_id"]].drop_duplicates().groupby("team", as_index=False).count()
    gp = gp.rename(columns={"game_id": "game_count"})
    total_homers = hr[hr["hr"] == "homer"][["team", "hr"]].groupby("team", as_index=False).count()
    hpg = total_homers.merge(gp, on="team")
    hpg["homer_per_game"] = hpg["hr"] / hpg["game_count"]
    hpg = hpg[["team", "homer_per_game"]]

    gwh = hr[hr["hr"] == "homer"][["team", "game_id", "hr"]].groupby(["team", "game_id"], as_index=False).count()
    game = hr[["team", "game_id"]].drop_duplicates()
    gwh = game.merge(gwh, on=["team", "game_id"], how="left").fillna(0)
    ghi = gwh[gwh["hr"] > 0].groupby("team", as_index=False).count().rename(columns={"hr": "games_homered_in"})
    gnhi = gwh[gwh["hr"] == 0].groupby("team", as_index=False).count().rename(columns={"hr": "games_not_homered_in"})
    ghi = ghi.merge(gnhi, on="team", how="left")
    ghi["game_homer_rate"] = ghi["games_homered_in"] / (ghi["games_homered_in"] + ghi["games_not_homered_in"])
    ghi = ghi[["team", "game_homer_rate"]]

    hr = hr.fillna(0)
    hr = hr[hr["pre_score"] != hr["post_score"]][["team", "hr", "pre_score", "post_score"]]
    hr = hr.groupby(["team", "hr"], as_index=False).sum()
    hr["runs_scored"] = hr["post_score"] - hr["pre_score"]
    hr = hr.pivot(index="team", columns="hr", values="runs_scored").fillna(0)
    # Early in a season (or any short as-of-date history slice - see
    # dfs_backtest.assemble_ml_training_rows, which replays from day one),
    # no home run may have been scored yet at all, so the pivot can be
    # missing the "homer" column entirely (or, in principle, "non_homer") -
    # add it back as all-zero rather than crashing on a real but temporary
    # data state.
    for column in ("homer", "non_homer"):
        if column not in hr.columns:
            hr[column] = 0.0
    hr = hr.reset_index()
    hr["runs_scored"] = hr["homer"] + hr["non_homer"]
    hr["home_run_reliance"] = hr["homer"] / hr["runs_scored"]
    hr = hr.merge(hpg, on="team").merge(ghi, on="team")
    for_merge = hr[["team", "home_run_reliance", "homer_per_game", "game_homer_rate"]]

    sus = data[["home_team", "events"]].rename(columns={"home_team": "team"})
    sus["hr"] = _homer_flag(sus["events"])
    sus = sus[sus["hr"] == "homer"][["team", "hr"]].groupby("team").count().reset_index()
    games = (
        data[["home_team", "game_id"]]
        .drop_duplicates()
        .groupby("home_team")
        .count()
        .reset_index()
        .rename(columns={"home_team": "team", "game_id": "game_count"})
    )
    asus = data[data["inning_topbot"] == "Top"][["home_team", "events"]].rename(columns={"home_team": "team"})
    asus["hr"] = _homer_flag(asus["events"])
    asus = asus[asus["hr"] == "homer"].rename(columns={"hr": "away_hr"})[["team", "away_hr"]]
    asus = asus.groupby("team").count().reset_index()

    sus = sus.merge(games, on="team")
    sus["hr_rate"] = sus["hr"] / sus["game_count"]
    sus = sus.merge(asus, on="team")
    sus["away_hr_rate"] = sus["away_hr"] / sus["game_count"]
    sus["team_home_run_rate"] = sus["hr_rate"] - sus["away_hr_rate"]
    sus = sus[["team", "team_home_run_rate", "away_hr_rate"]]

    return for_merge, sus


def compute_offensive_edge(data: pd.DataFrame) -> pd.DataFrame:
    """Returns [team, offensive_edge, team_bases_pg]. offensive_edge is
    rolling bases-scored-per-game minus the upcoming opponent's rolling
    bases-allowed-per-game, blended across OFFENSIVE_EDGE_WINDOWS.
    team_bases_pg (the pure offense half, before that opponent subtraction)
    is exposed separately for pitcher_matchup.py - offensive_edge itself
    nets out whichever opponent this team's OWN most recent game happened
    to be against, not today's actual matchup, so it's contaminated for
    that use; team_bases_pg has no opponent term at all and is the correct
    building block for "how good is the offense a given pitcher faces
    today," looked up by that pitcher's own opponent."""

    def bases_value(events: pd.Series) -> pd.Series:
        return events.map(_OFFENSIVE_EDGE_BASES_BY_EVENT).fillna(0).astype(int)

    top = data[data["inning_topbot"] == "Top"][["game_id", "away_team", "events", "home_team"]]
    top = top.rename(columns={"away_team": "team", "home_team": "opp"})
    bot = data[data["inning_topbot"] == "Bot"][["game_id", "home_team", "events", "away_team"]]
    bot = bot.rename(columns={"home_team": "team", "away_team": "opp"})
    bases = pd.concat([top, bot])
    bases["bases"] = bases_value(bases["events"])
    bases = bases.groupby(["team", "opp", "game_id"], as_index=False)["bases"].sum()

    allowed = bases.rename(columns={"opp": "team", "team": "opp", "bases": "bases_allowed"})
    full_bases = bases.merge(allowed, on=["team", "opp", "game_id"]).sort_values(["team", "game_id"])

    full_bases["bases_shift"] = full_bases.groupby("team")["bases"].shift(1)
    full_bases["bases_allowed_shift"] = full_bases.groupby("team")["bases_allowed"].shift(1)

    def col(prefix, days_back):
        return f"{prefix}_full" if days_back is None else f"{prefix}_{days_back}"

    for days_back, _weight in config.OFFENSIVE_EDGE_WINDOWS:
        if days_back is None:
            full_bases[col("bases_pg", None)] = (
                full_bases.groupby("team")["bases_shift"].expanding().mean().reset_index(level=0, drop=True)
            )
            full_bases[col("bases_allowed_pg", None)] = (
                full_bases.groupby("team")["bases_allowed_shift"].expanding().mean().reset_index(level=0, drop=True)
            )
        else:
            full_bases[col("bases_pg", days_back)] = (
                full_bases.groupby("team")["bases_shift"].rolling(days_back, min_periods=1).mean().reset_index(level=0, drop=True)
            )
            full_bases[col("bases_allowed_pg", days_back)] = (
                full_bases.groupby("team")["bases_allowed_shift"].rolling(days_back, min_periods=1).mean().reset_index(level=0, drop=True)
            )

    full_bases["team_bases_pg"] = sum(
        full_bases[col("bases_pg", d)] * w for d, w in config.OFFENSIVE_EDGE_WINDOWS
    )
    full_bases["team_bases_allowed_pg"] = sum(
        full_bases[col("bases_allowed_pg", d)] * w for d, w in config.OFFENSIVE_EDGE_WINDOWS
    )

    opp_allowed = full_bases[["team", "game_id", "team_bases_allowed_pg"]].rename(
        columns={"team": "opp", "team_bases_allowed_pg": "opp_bases_allowed_pg"}
    )
    full_bases = full_bases.merge(opp_allowed, on=["opp", "game_id"], how="left").fillna(0)
    full_bases["offensive_edge"] = full_bases["team_bases_pg"] - full_bases["opp_bases_allowed_pg"]

    latest_game = full_bases.groupby("team", as_index=False)["game_id"].max()
    return latest_game.merge(full_bases, on=["team", "game_id"])[["team", "offensive_edge", "team_bases_pg"]]


def compute_park_factors(data: pd.DataFrame) -> pd.DataFrame:
    """One row per team's home venue: Park_Factor - that venue's average
    combined (both teams') final runs per game, normalized to the
    across-all-venues average (mean 1.0, same ratio convention as
    PAVE_PLUS/Bullpen_PAVE_PLUS). A hitter-friendly park (Coors-like) reads
    above 1.0; a pitcher's park reads below 1.0.

    Uses `home_team` as the venue identifier - Statcast's pitch-level data
    doesn't carry a separate ballpark id, and a team's home games are all
    at the same park for the entire season (interleague neutral-site
    games are rare enough not to special-case here) - so this is
    equivalent to a real per-park factor without needing a park lookup
    table."""
    totals = data[["game_id", "home_team", "post_home_score", "post_away_score"]].copy()
    totals["combined_score"] = totals["post_home_score"] + totals["post_away_score"]
    per_game = totals.groupby(["game_id", "home_team"], as_index=False)["combined_score"].max()

    venue = per_game.groupby("home_team", as_index=False)["combined_score"].mean()
    venue = venue.rename(columns={"home_team": "team", "combined_score": "runs_per_game_at_home"})

    league_avg = venue["runs_per_game_at_home"].mean()
    venue["Park_Factor"] = venue["runs_per_game_at_home"] / league_avg
    return venue[["team", "Park_Factor"]]


def compute_team_win_rate_ci(record: pd.DataFrame) -> pd.DataFrame:
    """One row per team: games_played, win_rate (real season-to-date
    wins / games), and a real Wilson score confidence interval
    (helpers.wilson_ci) on that binomial proportion. Real follow-up
    (2026-08-25 - "we need the units risked to not be arbitrary"): feeds
    game_picks.apply_kelly_uncertainty, which supplements
    config.KELLY_FRACTION_MULTIPLIER's flat shrinkage with a real
    per-game conservative estimate.

    Uses config.KELLY_UNCERTAINTY_CI_ALPHA (a ~68%/1-standard-error CI),
    NOT helpers.wilson_ci's own 95% default - real follow-up (2026-08-26,
    "maybe we scaled it down too much... this is now taking out games
    where the line is genuinely appealing"): a real 95% CI's half-width
    sits at roughly 8 points EVEN AT A NEAR-FULL SEASON'S ~130 games (the
    real sqrt(n) precision floor of a binomial proportion at that sample
    size, not a fluke), which combined across two teams was quietly
    erasing genuine, honest edges rather than just implausible ones - see
    config.KELLY_UNCERTAINTY_CI_ALPHA's own docstring for the worked
    example. This narrower interval still genuinely widens early in the
    season (few real games played) and narrows as more real games
    accumulate, same as before - just calibrated to a less extreme real
    confidence level, with no hand-picked shrinkage constant.

    `record` must be build_team_record's own shape (one row per team per
    game, real win/loss)."""
    games = record.groupby("team", as_index=False).agg(games_played=("win", "size"), wins=("win", "sum"))
    games["win_rate"] = games["wins"] / games["games_played"]
    ci_low, ci_high = helpers.wilson_ci(games["wins"], games["games_played"], alpha=config.KELLY_UNCERTAINTY_CI_ALPHA)
    games["win_rate_CI_Low"] = ci_low
    games["win_rate_CI_High"] = ci_high
    return games[["team", "games_played", "win_rate", "win_rate_CI_Low", "win_rate_CI_High"]]


def compute_umpire_factor(data: pd.DataFrame) -> pd.DataFrame:
    """One row per real home-plate umpire (Statcast's own `umpire` id -
    real, not derived): Umpire_Factor - that umpire's own real hit rate on
    completed at-bats (config.COUNTED_EVENTS), normalized to the
    across-all-umpires average (mean 1.0, same ratio convention as
    Park_Factor/PAVE_PLUS). Exploratory candidate feature (2026-08-24
    feature-search follow-up - "test feature significance before
    committing to the model"): NOT yet wired into any live signal - see
    scripts/train_hitter_hit_model.py's significance report for whether a
    real bar was cleared before this goes anywhere near
    dfs_ml.HITTER_FEATURE_COLUMNS.

    A row with a missing/null umpire (an older/narrower synthetic fixture)
    is excluded from the average entirely, not counted as a real 0."""
    completed = data[data["events"].isin(config.COUNTED_EVENTS)][["umpire", "events"]].copy()
    completed = completed[completed["umpire"].notna()]
    if completed.empty:
        return pd.DataFrame(columns=["umpire", "Umpire_Factor"])

    completed["hit"] = helpers.is_hit(completed["events"])
    league_avg = completed["hit"].mean()

    per_umpire = completed.groupby("umpire", as_index=False)["hit"].mean()
    per_umpire["Umpire_Factor"] = per_umpire["hit"] / league_avg
    return per_umpire[["umpire", "Umpire_Factor"]]


def build_team_record(data: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: opponent, win/loss, runs scored/allowed."""
    game_totals = data[
        ["home_team", "away_team", "game_date", "game_id", "post_home_score", "post_away_score"]
    ].copy()
    game_totals["fs"] = game_totals["post_home_score"] + game_totals["post_away_score"]
    final_state = game_totals.groupby("game_id", as_index=False)["fs"].max()
    final_state = final_state.merge(game_totals, on=["game_id", "fs"]).drop_duplicates()[
        ["home_team", "away_team", "game_date", "game_id", "post_home_score", "post_away_score"]
    ]

    away = final_state.rename(
        columns={"away_team": "team", "home_team": "opp", "post_home_score": "ra", "post_away_score": "rs"}
    )
    home = final_state.rename(
        columns={"home_team": "team", "away_team": "opp", "post_home_score": "rs", "post_away_score": "ra"}
    )
    record = pd.concat([away, home]).sort_values(["team", "game_date"])
    record["win"] = (record["rs"] > record["ra"]).astype(int)
    record["loss"] = (record["rs"] < record["ra"]).astype(int)
    return record[["opp", "team", "game_date", "game_id", "win", "loss", "rs", "ra"]]


def compute_suppression_resistance(record: pd.DataFrame) -> pd.DataFrame:
    """1 - (blended pct of games scored under 3 runs), z-normalized."""
    under3 = record.sort_values(["team", "game_id"]).copy()
    under3["under3"] = (under3["rs"] < 3).astype(int)

    for days_back, _weight in config.SUPPRESSION_WINDOWS:
        under3[f"under3_pct_{days_back}"] = (
            under3.groupby("team")["under3"].rolling(days_back, min_periods=1).mean().reset_index(level=0, drop=True)
        )

    latest_game = under3.groupby("team", as_index=False)["game_id"].max()
    max3 = latest_game.merge(under3, on=["team", "game_id"])

    max3["suppression_resistance"] = 1 - sum(
        max3[f"under3_pct_{d}"] * w for d, w in config.SUPPRESSION_WINDOWS
    )
    max3 = max3[["team", "suppression_resistance"]]

    mean_sup = max3["suppression_resistance"].mean()
    std_sup = max3["suppression_resistance"].std()
    max3["suppression_resistance"] = 1 + (
        (max3["suppression_resistance"] - mean_sup) / std_sup * config.NORMALIZATION_Z_SCALE
    )
    return max3


def _compute_exclusion_windows(group: pd.DataFrame, windows) -> pd.DataFrame:
    """Per-game win rate over the last N games, excluding games against that
    game's specific upcoming opponent (so a strength score isn't inflated by
    a long series against one weak team)."""
    # pandas' groupby(...).apply() excludes the grouping column from `group`
    # itself, so it has to be restored explicitly for the two chained
    # groupby("team") calls in compute_strength_metrics to keep working.
    team = group.name
    group = group.copy()
    group["team"] = team
    results = {f"rolling_{w}": [] for w in windows}
    full_history = []

    for i in range(len(group)):
        opp = group.iloc[i]["opp"]
        history_excl = group.iloc[:i]
        history_excl = history_excl[history_excl["opp"] != opp]

        full_history.append(history_excl["win"].mean() if len(history_excl) else np.nan)
        for w in windows:
            results[f"rolling_{w}"].append(
                history_excl["win"].tail(w).mean() if len(history_excl) else np.nan
            )

    for key, values in results.items():
        group[key] = values
    group["full"] = full_history
    return group


def _compute_current_windows(group: pd.DataFrame, windows) -> pd.DataFrame:
    """Same as `_compute_exclusion_windows` but without excluding the upcoming opponent."""
    # See _compute_exclusion_windows - group.name has to be captured before
    # any copy/reassignment, since pandas excludes "team" from `group` itself.
    team = group.name
    group = group.copy()
    group["team"] = team
    results = {f"roll_{w}_cur": [] for w in windows}
    full_history = []

    for i in range(len(group)):
        history = group.iloc[:i]
        full_history.append(history["win"].mean() if len(history) else np.nan)
        for w in windows:
            results[f"roll_{w}_cur"].append(history["win"].tail(w).mean() if len(history) else np.nan)

    for key, values in results.items():
        group[key] = values
    group["full_cur"] = full_history
    return group


def compute_strength_metrics(record: pd.DataFrame):
    """Returns (current_strength, sos): rolling win%, Pythagorean win%, and
    strength-of-schedule per team, as of each team's most recent game."""
    windows = [w for w, _ in config.TEAM_STRENGTH_WINDOWS if w is not None]

    nf = record.sort_values(["team", "game_id"]).reset_index(drop=True)
    nf = nf.groupby("team", group_keys=False).apply(lambda g: _compute_exclusion_windows(g, windows))
    nf = nf.fillna(0)

    nf = nf.groupby("team", group_keys=False).apply(lambda g: _compute_current_windows(g, windows))
    nf = nf.fillna(0)

    nf["rs_shift"] = nf.groupby("team")["rs"].shift(1)
    nf["ra_shift"] = nf.groupby("team")["ra"].shift(1)

    for w in windows:
        nf[f"rs_{w}"] = nf.groupby("team")["rs_shift"].rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)
        nf[f"ra_{w}"] = nf.groupby("team")["ra_shift"].rolling(w, min_periods=1).sum().reset_index(level=0, drop=True)

    nf["rs_full"] = nf.groupby("team")["rs_shift"].cumsum()
    nf["ra_full"] = nf.groupby("team")["ra_shift"].cumsum()

    exp = config.PYTHAGOREAN_EXPONENT
    for w in windows:
        nf[f"pyth_{w}"] = (nf[f"rs_{w}"] ** exp) / ((nf[f"rs_{w}"] ** exp) + (nf[f"ra_{w}"] ** exp))
    nf["pyth_full"] = (nf["rs_full"] ** exp) / ((nf["rs_full"] ** exp) + (nf["ra_full"] ** exp))
    nf = nf.fillna(0)

    def blend(col_for_window):
        total = None
        for days_back, weight in config.TEAM_STRENGTH_WINDOWS:
            term = nf[col_for_window(days_back)] * weight
            total = term if total is None else total + term
        return total

    nf["strength"] = blend(lambda d: "full" if d is None else f"rolling_{d}")
    nf["current_strength"] = blend(lambda d: "full_cur" if d is None else f"roll_{d}_cur")
    nf["pyth_strength"] = blend(lambda d: "pyth_full" if d is None else f"pyth_{d}")

    sos = nf.groupby("opp", as_index=False)[["strength", "pyth_strength"]].mean()
    sos = sos.rename(columns={"opp": "team", "strength": "SOS", "pyth_strength": "pyth_SOS"})

    latest_game = nf.groupby("team", as_index=False)["game_id"].max()
    current_strength = latest_game.merge(nf, on=["team", "game_id"])[
        ["team", "strength", "current_strength", "pyth_strength"]
    ]
    return current_strength, sos


def assemble_team_metrics(data: pd.DataFrame, bullpen_pave: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the final team output table (equivalent to the original script's `master` dataframe).

    `bullpen_pave` is the output of pitchers.compute_bullpen_pave() - team
    Bullpen_PAVE_PLUS/Bullpen_BAA/etc, describing how tough *this* team's
    bullpen is to hit against. It's optional so teams.py stays independently
    testable/usable without requiring pitcher-role data.

    Also carries games_played/win_rate/win_rate_CI_Low/win_rate_CI_High
    (compute_team_win_rate_ci) - real season-to-date record with a Wilson
    confidence interval, feeding game_picks.apply_kelly_uncertainty's
    real bet-sizing (see that function's docstring).
    """
    for_merge, sus = compute_home_run_stats(data)
    latest_off = compute_offensive_edge(data)
    record = build_team_record(data)
    suppression = compute_suppression_resistance(record)
    current_strength, sos = compute_strength_metrics(record)
    park_factors = compute_park_factors(data)

    master = current_strength.merge(sos, on="team")

    def normalize(col: str) -> pd.Series:
        mean = master[col].mean()
        std = master[col].std()
        return 1 + ((master[col] - mean) / std * config.NORMALIZATION_Z_SCALE)

    master = master[["team"]].assign(
        Strength=normalize("strength"),
        SOS=normalize("SOS"),
        current=normalize("current_strength"),
        pyth_Strength=normalize("pyth_strength"),
        pyth_SOS=normalize("pyth_SOS"),
    ).drop_duplicates().reset_index(drop=True)

    master["Confidence"] = master["Strength"] + master["SOS"] * config.CONFIDENCE_SOS_WEIGHT
    master["pyth_Confidence"] = master["pyth_Strength"] + master["pyth_SOS"] * config.CONFIDENCE_SOS_WEIGHT
    for col in ("Confidence", "pyth_Confidence"):
        mean = master[col].mean()
        std = master[col].std()
        master[col] = 1 + ((master[col] - mean) / std * config.NORMALIZATION_Z_SCALE)

    master["Confidence_Delta"] = master["Confidence"] - master["pyth_Confidence"]
    master = master.merge(latest_off, on="team").merge(for_merge, on="team").merge(sus, on="team").merge(
        suppression, on="team"
    )

    mean_edge = master["offensive_edge"].mean()
    std_edge = master["offensive_edge"].std()
    master["offensive_edge"] = 1 + ((master["offensive_edge"] - mean_edge) / std_edge * config.NORMALIZATION_Z_SCALE)

    master = master.drop_duplicates()
    master["true_power"] = (master["offensive_edge"] + master["suppression_resistance"]) / 2
    master = master.merge(park_factors, on="team", how="left")

    win_rate_ci = compute_team_win_rate_ci(record)
    master = master.merge(win_rate_ci, on="team", how="left")

    output_columns = [
        "team", "current", "Strength", "pyth_Strength", "SOS", "pyth_SOS",
        "Confidence", "pyth_Confidence", "Confidence_Delta", "true_power",
        "offensive_edge", "team_bases_pg", "suppression_resistance", "home_run_reliance",
        "homer_per_game", "game_homer_rate", "team_home_run_rate", "away_hr_rate",
        "Park_Factor", "games_played", "win_rate", "win_rate_CI_Low", "win_rate_CI_High",
    ]
    if bullpen_pave is not None:
        master = master.merge(bullpen_pave, on="team", how="left")
        output_columns += [
            "Bullpen_PAVE", "Bullpen_PAVE_PLUS", "Bullpen_Power_A_PLUS",
            "Bullpen_BAA", "Bullpen_Power_A", "Bullpen_HR_Per", "Bullpen_AtBats",
        ]

    master = master[output_columns]
    return master.sort_values("team").reset_index(drop=True)

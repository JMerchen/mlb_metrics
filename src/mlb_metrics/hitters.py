"""Hitter metrics: WAVE (at-bat based hit probability), Game_Hit_Probability
(game-level hit rate), WTB (weighted total bases), and WHOPS/RC (weighted
OPS / runs-created proxy - computed but, as in the original script, not
currently written to any output; kept here as a reusable building block for
future matchup work rather than wired into the CSV pipeline).

All three of WAVE/WHOPS/WTB share the same shape: split at-bats by the
opposing pitcher's throwing hand, compute a rate per recency window, and
blend the windows with fixed weights (see config.py). `_blend_windows`
implements that shared shape once instead of the ~6x duplicated inline
blocks in the original script.

Note on population: each blended side-metric is anchored on batters who
have at least one full-season at-bat *against that side* (matching the
original's `rfull.merge(...)` anchor) - a batter who has only ever faced
left-handed pitching, for instance, is dropped from the R-side blend and
picks up a WAVE_R of 0 rather than being excluded outright, exactly as in
the original.
"""

import pandas as pd

from mlb_metrics import config, helpers


def _window_key(days_back):
    return "full" if days_back is None else f"d{days_back}"


def _slice_by_days(frame: pd.DataFrame, latest, days_back):
    if days_back is None:
        return frame
    cutoff = latest - pd.Timedelta(days=days_back)
    return frame[frame["game_date"] >= cutoff]


def _side_window_agg(dt: pd.DataFrame, latest, days_back, throws: str, stat_fns: dict) -> pd.DataFrame:
    """One row per batter: `n` (at-bat count) plus the sum of each stat in
    `stat_fns`, restricted to the window and the given opposing-pitcher hand.

    Each `stat_fns` value is `fn(side_df) -> Series`, given the WHOLE
    windowed/sided frame (not just its `events` column) - needed by stats
    like `helpers.estimate_rbi` that read more than one column
    (`bat_score`/`post_bat_score`). Existing `events`-only classifiers
    (`helpers.is_hit`, etc.) are passed as `lambda df: helpers.is_hit(df["events"])`."""
    window_df = _slice_by_days(dt, latest, days_back)
    side_df = window_df[window_df["p_throws"] == throws].copy()
    side_df["n"] = 1
    for name, fn in stat_fns.items():
        side_df[name] = fn(side_df)
    cols = ["batter", "n"] + list(stat_fns.keys())
    return side_df[cols].groupby("batter", as_index=False).sum()


def _blend_windows(dt: pd.DataFrame, windows, throws: str, stat_fns: dict, rate_fns: dict):
    """Blend one or more rate functions across `windows` for one throws-side.

    rate_fns: dict of name -> function(window_agg_df) -> per-batter rate Series.
    Returns (blended_df, full_season_counts) where blended_df has columns
    [batter, *rate_fns.keys()] and full_season_counts has [batter, n].
    """
    latest = dt["game_date"].max()
    blended = {name: None for name in rate_fns}
    full_counts = None

    for days_back, weight in windows:
        agg = _side_window_agg(dt, latest, days_back, throws, stat_fns)
        base = agg[["batter"]]
        for name, rate_fn in rate_fns.items():
            rate = rate_fn(agg).fillna(0)
            contribution = base.assign(rate=rate.values).set_index("batter")["rate"] * weight
            blended[name] = (
                contribution if blended[name] is None else blended[name].add(contribution, fill_value=0)
            )
        if days_back is None:
            full_counts = agg[["batter", "n"]]

    result = pd.DataFrame(blended)
    result.index.name = "batter"
    result = result.reset_index()
    return result, full_counts


def compute_wave(dt: pd.DataFrame) -> pd.DataFrame:
    """Recency-weighted at-bat hit rate vs L/R pitching, converted to a
    per-game hit probability. Returns key_mlbam, WAVE, WAVE_L, WAVE_R,
    l_at_bat, r_at_bat, probability, probability_L, probability_R."""
    stat_fns = {"hit": lambda df: helpers.is_hit(df["events"])}
    rate_fns = {"rate": lambda agg: agg["hit"] / agg["n"]}

    r_blended, r_full = _blend_windows(dt, config.WAVE_WINDOWS, "R", stat_fns, rate_fns)
    l_blended, l_full = _blend_windows(dt, config.WAVE_WINDOWS, "L", stat_fns, rate_fns)

    wave = r_full.rename(columns={"n": "r_at_bat"})
    wave = wave.merge(r_blended.rename(columns={"rate": "WAVE_R"}), on="batter", how="left")
    wave = wave.merge(l_full.rename(columns={"n": "l_at_bat"}), on="batter", how="left")
    wave = wave.merge(l_blended.rename(columns={"rate": "WAVE_L"}), on="batter", how="left")
    wave = wave.fillna(0)

    wave["abt"] = wave["l_at_bat"] + wave["r_at_bat"]
    wave["abtl"] = wave["l_at_bat"] / wave["abt"]
    wave["abtr"] = wave["r_at_bat"] / wave["abt"]
    wave["WAVE"] = wave["WAVE_R"] * wave["abtr"] + wave["WAVE_L"] * wave["abtl"]
    wave = wave.rename(columns={"batter": "key_mlbam"})

    wave["probability"] = 1 - (1 - wave["WAVE"]) ** config.WAVE_TRIALS_PER_GAME
    wave["probability_L"] = 1 - (1 - wave["WAVE_L"]) ** config.WAVE_TRIALS_PER_GAME
    wave["probability_R"] = 1 - (1 - wave["WAVE_R"]) ** config.WAVE_TRIALS_PER_GAME

    return wave[
        [
            "key_mlbam", "WAVE", "WAVE_L", "WAVE_R", "l_at_bat", "r_at_bat",
            "probability", "probability_L", "probability_R",
        ]
    ]


def compute_whops(dt: pd.DataFrame) -> pd.DataFrame:
    """Weighted OPS and a runs-created proxy, split by throws and blended
    across windows. Computed for completeness but - as in the original
    script - not currently merged into any output table."""
    stat_fns = {
        "ob": lambda df: helpers.is_on_base(df["events"]),
        "sv": lambda df: helpers.total_bases(df["events"]),
        "ab": lambda df: helpers.is_official_at_bat(df["events"]),
    }

    def ops_rate(agg):
        return agg["ob"] / agg["n"] + agg["sv"] / agg["n"]

    def rc_rate(agg):
        obp = agg["ob"] / agg["n"]
        slg = agg["sv"] / agg["n"]
        return slg * obp * (agg["ab"] / agg["n"])

    rate_fns = {"ops": ops_rate, "rc": rc_rate}
    windows = config.WHOPS_WTB_WINDOWS

    r_blended, r_full = _blend_windows(dt, windows, "R", stat_fns, rate_fns)
    l_blended, l_full = _blend_windows(dt, windows, "L", stat_fns, rate_fns)

    whops = r_full.rename(columns={"n": "pa_rfull"})
    whops = whops.merge(r_blended.rename(columns={"ops": "whops_R", "rc": "rcw_R"}), on="batter", how="left")
    whops = whops.merge(l_full.rename(columns={"n": "pa_lfull"}), on="batter", how="left")
    whops = whops.merge(l_blended.rename(columns={"ops": "whops_L", "rc": "rcw_L"}), on="batter", how="left")
    whops = whops.fillna(0)

    whops["abt"] = whops["pa_lfull"] + whops["pa_rfull"]
    whops["abtl"] = whops["pa_lfull"] / whops["abt"]
    whops["abtr"] = whops["pa_rfull"] / whops["abt"]
    whops["whops"] = whops["whops_R"] * whops["abtr"] + whops["whops_L"] * whops["abtl"]
    whops["rcw"] = whops["rcw_R"] * whops["abtr"] + whops["rcw_L"] * whops["abtl"]
    whops["rope"] = whops["rcw"] + whops["whops"]
    whops = whops.rename(columns={"batter": "key_mlbam"})

    return whops[
        ["key_mlbam", "whops", "whops_R", "whops_L", "pa_lfull", "pa_rfull", "rcw_R", "rcw_L", "rcw", "rope"]
    ]


def compute_wtb(dt: pd.DataFrame) -> pd.DataFrame:
    """Recency-weighted slugging value, blended by throws, plus Expected_Bases."""
    stat_fns = {"sv": lambda df: helpers.total_bases(df["events"])}
    rate_fns = {"slg": lambda agg: agg["sv"] / agg["n"]}
    windows = config.WHOPS_WTB_WINDOWS

    r_blended, r_full = _blend_windows(dt, windows, "R", stat_fns, rate_fns)
    l_blended, l_full = _blend_windows(dt, windows, "L", stat_fns, rate_fns)

    wtb = r_full.rename(columns={"n": "pa_rfull"})
    wtb = wtb.merge(r_blended.rename(columns={"slg": "WTB_r"}), on="batter", how="left")
    wtb = wtb.merge(l_full.rename(columns={"n": "pa_lfull"}), on="batter", how="left")
    wtb = wtb.merge(l_blended.rename(columns={"slg": "WTB_l"}), on="batter", how="left")
    wtb = wtb.fillna(0)

    wtb["ab"] = wtb["pa_lfull"] + wtb["pa_rfull"]
    wtb["abr"] = wtb["pa_rfull"] / wtb["ab"]
    wtb["abl"] = wtb["pa_lfull"] / wtb["ab"]
    wtb["WTB"] = wtb["WTB_r"] * wtb["abr"] + wtb["WTB_l"] * wtb["abl"]
    wtb["Expected_Bases"] = wtb["WTB"] * config.WTB_TRIALS_PER_GAME
    wtb = wtb.rename(columns={"batter": "key_mlbam"})

    return wtb[["key_mlbam", "pa_lfull", "pa_rfull", "WTB_l", "WTB_r", "WTB", "Expected_Bases"]]


def compute_extended_dk_rates(dt: pd.DataFrame) -> pd.DataFrame:
    """Recency-windowed BB/HBP/RBI rates for DK Classic MLB scoring
    (dfs.py's compute_hitter_dk_points), reusing the exact same windowed
    blend machinery WAVE/WHOPS/WTB already use. Deliberately reuses
    config.WHOPS_WTB_WINDOWS (not a new dedicated window constant) - these
    draw from the identical per-PA event stream WTB already blends, at the
    same row granularity, so a separate window schedule would be an
    unjustified extra knob, not a real architectural need (contrast with
    pitcher_form.py's DFS_PITCHER_WINDOWS, which is deliberately separate
    because it blends START-level, not AT-BAT-level, data).

    `dt` must carry `bat_score`/`post_bat_score` (see
    pipeline.build_pitch_events) for helpers.estimate_rbi.

    Returns [key_mlbam, Expected_BB, Expected_HBP, Expected_RBI] - a
    per-game expected count for each, converted from the blended per-PA
    rate via config.WTB_TRIALS_PER_GAME (the same at-bat-per-game
    assumption Expected_Bases already uses)."""
    stat_fns = {
        "bb": lambda df: helpers.is_walk_for_dk_scoring(df["events"]),
        "hbp": lambda df: helpers.is_hit_by_pitch(df["events"]),
        "rbi": helpers.estimate_rbi,
    }
    rate_fns = {
        "bb_rate": lambda agg: agg["bb"] / agg["n"],
        "hbp_rate": lambda agg: agg["hbp"] / agg["n"],
        "rbi_rate": lambda agg: agg["rbi"] / agg["n"],
    }
    windows = config.WHOPS_WTB_WINDOWS

    r_blended, r_full = _blend_windows(dt, windows, "R", stat_fns, rate_fns)
    l_blended, l_full = _blend_windows(dt, windows, "L", stat_fns, rate_fns)

    r_blended = r_blended.rename(columns={"bb_rate": "BB_rate_R", "hbp_rate": "HBP_rate_R", "rbi_rate": "RBI_rate_R"})
    l_blended = l_blended.rename(columns={"bb_rate": "BB_rate_L", "hbp_rate": "HBP_rate_L", "rbi_rate": "RBI_rate_L"})

    result = r_full.rename(columns={"n": "pa_rfull"})
    result = result.merge(r_blended, on="batter", how="left")
    result = result.merge(l_full.rename(columns={"n": "pa_lfull"}), on="batter", how="left")
    result = result.merge(l_blended, on="batter", how="left")
    result = result.fillna(0)

    result["abt"] = result["pa_lfull"] + result["pa_rfull"]
    result["abtl"] = result["pa_lfull"] / result["abt"]
    result["abtr"] = result["pa_rfull"] / result["abt"]

    for stat in ("BB", "HBP", "RBI"):
        blended_rate = result[f"{stat}_rate_R"] * result["abtr"] + result[f"{stat}_rate_L"] * result["abtl"]
        result[f"Expected_{stat}"] = (blended_rate * config.WTB_TRIALS_PER_GAME).fillna(0)

    result = result.rename(columns={"batter": "key_mlbam"})
    return result[["key_mlbam", "Expected_BB", "Expected_HBP", "Expected_RBI"]]


def compute_game_hit_probability(data_with_game_id: pd.DataFrame) -> pd.DataFrame:
    """Rate of *games* (not at-bats) with >=1 hit, blended across windows.
    This is the closest existing analog to a Beat-the-Streak pick probability."""
    completed = data_with_game_id[data_with_game_id["events"].isin(config.COUNTED_EVENTS)][
        ["game_date", "batter", "events", "game_id"]
    ].copy()
    completed["had_hit"] = helpers.is_hit(completed["events"])

    game_hits = completed.groupby(["batter", "game_id", "game_date"], as_index=False)["had_hit"].max()
    game_hits["game"] = 1
    latest = game_hits["game_date"].max()

    blended = None
    for days_back, weight in config.GAME_HIT_PROB_WINDOWS:
        window_games = _slice_by_days(game_hits, latest, days_back)
        agg = window_games.groupby("batter", as_index=False)[["had_hit", "game"]].sum()
        rate = (agg["had_hit"] / agg["game"]).fillna(0)
        contribution = agg[["batter"]].assign(rate=rate.values).set_index("batter")["rate"] * weight
        blended = contribution if blended is None else blended.add(contribution, fill_value=0)

    result = blended.reset_index()
    result.columns = ["batter", "Game_Hit_Probability"]
    return result.rename(columns={"batter": "key_mlbam"})


def compute_last_game_dates(data_with_game_id: pd.DataFrame) -> pd.DataFrame:
    """[key_mlbam, Last_Game_Date] - the most recent date each batter
    recorded a completed plate appearance. Unlike compute_current_hit_streaks's
    walk, finding the single most recent date doesn't need the game_id-safe
    chronological ordering (a doubleheader's two games share one game_date
    either way) - a plain per-batter max is correct.

    Feeds assemble_hitters's Last_Game_Date column, which
    predictions.select_picks gates official picks on (a hitter who hasn't
    played recently shouldn't be pick-eligible just because their
    season-long rates are still high)."""
    completed = data_with_game_id[data_with_game_id["events"].isin(config.COUNTED_EVENTS)][
        ["game_date", "batter"]
    ]
    result = completed.groupby("batter", as_index=False)["game_date"].max()
    return result.rename(columns={"batter": "key_mlbam", "game_date": "Last_Game_Date"})


def compute_current_hit_streaks(
    data_with_game_id: pd.DataFrame, recent_days: int = config.HIT_STREAK_RECENT_DAYS
) -> pd.DataFrame:
    """[key_mlbam, Current_Hit_Streak, last_game_date] - each recently-active
    batter's real current consecutive-games-with-a-hit streak, counted
    backward from their own most recent game. Same per-game hit indicator
    compute_game_hit_probability builds - grouped by (batter, game_id,
    game_date), NOT game_date alone (see data.assign_game_ids's docstring:
    date-only grouping fragments a doubleheader's two games and has already
    been a real bug here) - just walked in game_id order per batter instead
    of aggregated into a rate.

    A batter whose most recent game is more than `recent_days` before the
    latest game_date in the data is excluded entirely - a streak frozen by
    weeks of inactivity isn't a real "current" one. Batters within that
    window are still included even at Current_Hit_Streak=0 (their last game
    was a miss); it's the caller's job to decide what counts as "on a
    streak" for display purposes."""
    completed = data_with_game_id[data_with_game_id["events"].isin(config.COUNTED_EVENTS)][
        ["game_date", "batter", "events", "game_id"]
    ].copy()
    completed["had_hit"] = helpers.is_hit(completed["events"])

    game_hits = completed.groupby(["batter", "game_id", "game_date"], as_index=False)["had_hit"].max()
    if game_hits.empty:
        return pd.DataFrame(columns=["key_mlbam", "Current_Hit_Streak", "last_game_date"])

    latest = game_hits["game_date"].max()
    game_hits = game_hits.sort_values("game_id")

    rows = []
    for batter, group in game_hits.groupby("batter", sort=False):
        last_game_date = group["game_date"].iloc[-1]
        if (latest - last_game_date).days > recent_days:
            continue
        streak = 0
        for had_hit in group["had_hit"].iloc[::-1]:
            if not had_hit:
                break
            streak += 1
        rows.append({"key_mlbam": batter, "Current_Hit_Streak": streak, "last_game_date": last_game_date})

    return pd.DataFrame(rows, columns=["key_mlbam", "Current_Hit_Streak", "last_game_date"])


def assemble_hitters(
    dt: pd.DataFrame,
    data_with_game_id: pd.DataFrame,
    names: pd.DataFrame,
    latest_team: pd.DataFrame,
    lineup_consistency: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build the final hitter output table (equivalent to the original script's `WAVE` dataframe).

    `lineup_consistency` (see lineup.compute_lineup_consistency) is optional
    so existing callers/tests are unaffected; when given, it adds
    avg_batting_order/start_rate columns that predictions.select_picks's
    lineup qualifiers key off of. avg_batting_order is deliberately left
    null (not filled to 0) for a batter with no recorded starts - a null
    correctly fails the "top half of the order" qualifier downstream,
    whereas a 0 would look like the best possible batting slot.
    """
    wave = compute_wave(dt)
    wtb = compute_wtb(dt)
    extended_dk_rates = compute_extended_dk_rates(dt)
    game_hit_prob = compute_game_hit_probability(data_with_game_id)
    last_game = compute_last_game_dates(data_with_game_id)

    hitters = wave.merge(
        wtb[["key_mlbam", "pa_lfull", "pa_rfull", "Expected_Bases"]], on="key_mlbam", how="left"
    )
    hitters = hitters.merge(extended_dk_rates, on="key_mlbam", how="left")
    hitters = hitters.merge(game_hit_prob, on="key_mlbam", how="left")
    hitters = hitters.merge(names, on="key_mlbam", how="left")
    hitters = hitters.merge(latest_team, on="key_mlbam", how="left")

    hitters["Consistency"] = hitters["Game_Hit_Probability"] - hitters["probability"]
    hitters["Approach"] = hitters["Game_Hit_Probability"] * hitters["probability"]

    hitters = hitters[
        [
            "key_mlbam", "name_first", "name_last", "team",
            "pa_lfull", "pa_rfull",
            "WAVE", "WAVE_L", "WAVE_R", "probability_L", "probability_R", "probability",
            "Game_Hit_Probability", "Consistency", "Approach", "Expected_Bases",
            "Expected_BB", "Expected_HBP", "Expected_RBI",
        ]
    ].fillna(0)

    # Merged AFTER the fillna(0) block above (like lineup_consistency below)
    # rather than before - a missing/NaT Last_Game_Date must stay NaT, not
    # get coerced to 0 by that blanket fillna, since predictions.select_picks's
    # recency gate relies on NaT correctly failing the comparison (same
    # "null loses, isn't filled to a value that would wrongly pass"
    # precedent avg_batting_order already established).
    hitters = hitters.merge(last_game, on="key_mlbam", how="left")

    if lineup_consistency is not None:
        hitters = hitters.merge(lineup_consistency, on="key_mlbam", how="left")
        hitters["start_rate"] = hitters["start_rate"].fillna(0)

    hitters = hitters.sort_values("Game_Hit_Probability", ascending=False)
    return hitters.rename(columns={"pa_lfull": "PA_L", "pa_rfull": "PA_R"})

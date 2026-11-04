"""Measures how the weather changes plays, for the NFL game simulation's
SIM_WEATHER table (docs/nfl_sim.js).

Every regular-season snap in an outdoor (or open-roof) game, 2016-2025,
with that game's weather: wind over 5 mph and degrees below 40°F from
nflverse's schedule readings, and rain or snow from the play-by-play's
weather text (nfl_weather.precip_from_text). For each kind of play the
odds (or yards) are regressed on all four at once, so a windy, rainy game
isn't counted twice:
    pass_rate   pass called on 1st/2nd down, quarters 1-3, score within 8
    comp        pass completed (sacks out)
    int         pass intercepted
    sack        dropback sacked
    fumble      run or catch fumbled and lost
    fg          field goal good, given its distance
are log-odds per unit, and
    pass_yds    yards per completion
    rush_yds    yards per carry
fractions of the average per unit. The simulation applies them to each
team's own rates, so a soaked game throws less, completes less and gains
less through the air; how many points get scored still comes from the
points model's total (nfl_totals.py), which the simulation is calibrated to.

RESULT (2026-11-04, 222,061 outdoor snaps, 1,707 games), log-odds or
yards fraction per unit (p):
    pass_rate  wind -0.0054 (0.000)  cold -0.0049 (0.003)  rain -0.136 (0.000)  snow -0.193 (0.003)
    comp       wind -0.0063 (0.000)  cold -0.0042 (0.002)  rain -0.122 (0.000)  snow -0.192 (0.000)
    sack       snow +0.261 (0.010); wind, cold, rain nothing
    pass_yds   wind -0.0028 (0.000); cold, rain, snow nothing
    int, fumble, rush_yds: nothing (all p > 0.05)
    fg         rain -0.253 (0.053), snow -0.632 (0.068), wind -0.012 (0.146) -
               short of p < 0.05 (coaches skip long kicks in wind), left out
docs/nfl_sim.js's SIM_WEATHER is the p < 0.05 part of this.

Usage:
    python scripts/measure_nfl_weather_effects.py
"""

import os
import sys

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mlb_metrics import nfl_weather  # noqa: E402

RAW_DIR = "data/raw/nfl"
SEASONS = range(2016, 2026)
TERMS = ["wind", "cold", "rain", "snow"]
COLUMNS = ["game_id", "season_type", "roof", "weather", "down", "qtr", "score_differential", "play_type",
           "pass", "rush", "qb_kneel", "qb_spike", "sack", "complete_pass", "interception", "fumble_lost",
           "yards_gained", "field_goal_attempt", "field_goal_result", "kick_distance"]


def load() -> pd.DataFrame:
    frames = []
    for season in SEASONS:
        pbp = pd.read_parquet(os.path.join(RAW_DIR, f"pbp_{season}.parquet"), columns=COLUMNS)
        schedule = pd.read_parquet(os.path.join(RAW_DIR, f"schedules_{season}.parquet"))
        pbp = pbp[(pbp["season_type"] == "REG") & pbp["roof"].isin(["outdoors", "open"])]
        game = schedule.set_index("game_id")
        pbp["wind"] = (pbp["game_id"].map(game["wind"]) - nfl_weather.WIND_FROM_MPH).clip(lower=0)
        pbp["cold"] = (nfl_weather.COLD_BELOW_F - pbp["game_id"].map(game["temp"])).clip(lower=0)
        kind = pbp["weather"].map(nfl_weather.precip_from_text)
        pbp["rain"] = (kind == "rain").astype(float)
        pbp["snow"] = (kind == "snow").astype(float)
        frames.append(pbp.dropna(subset=["wind", "cold"]))
    return pd.concat(frames, ignore_index=True)


def logit(rows: pd.DataFrame, outcome: str, extra=()) -> dict:
    rows = rows.dropna(subset=[outcome, *extra])
    model = sm.Logit(rows[outcome].astype(float), sm.add_constant(rows[TERMS + list(extra)])).fit(disp=0)
    return {t: (round(float(model.params[t]), 4), round(float(model.pvalues[t]), 4)) for t in TERMS}


def yards(rows: pd.DataFrame) -> dict:
    rows = rows.dropna(subset=["yards_gained"])
    model = sm.OLS(rows["yards_gained"], sm.add_constant(rows[TERMS])).fit()
    base = float(model.params["const"])
    return {t: (round(float(model.params[t]) / base, 4), round(float(model.pvalues[t]), 4)) for t in TERMS}


def main():
    plays = load()
    scrimmage = plays[((plays["pass"] == 1) | (plays["rush"] == 1)) & (plays["qb_kneel"] != 1) & (plays["qb_spike"] != 1)]
    passes = scrimmage[scrimmage["pass"] == 1]
    attempts = passes[passes["sack"] != 1]
    runs = scrimmage[scrimmage["pass"] != 1]
    neutral = scrimmage[scrimmage["down"].isin([1, 2]) & (scrimmage["qtr"] <= 3)
                        & (scrimmage["score_differential"].abs() <= 8)]
    kicks = plays[plays["field_goal_attempt"] == 1].assign(
        good=lambda d: (d["field_goal_result"] == "made").astype(float))
    print(f"{len(scrimmage):,} outdoor snaps; {plays.game_id.nunique()} games")
    results = {
        "pass_rate": logit(neutral.assign(is_pass=neutral["pass"]), "is_pass"),
        "comp": logit(attempts, "complete_pass"),
        "int": logit(attempts, "interception"),
        "sack": logit(passes, "sack"),
        "fumble": logit(pd.concat([runs, attempts[attempts["complete_pass"] == 1]]), "fumble_lost"),
        "fg": logit(kicks, "good", extra=["kick_distance"]),
        "pass_yds": yards(attempts[attempts["complete_pass"] == 1]),
        "rush_yds": yards(runs),
    }
    for name, terms in results.items():
        print(f"{name:10s}", "  ".join(f"{t} {v:+.4f} (p {p:.3f})" for t, (v, p) in terms.items()))
    print("\nSIM_WEATHER (p < 0.05 kept, others 0):")
    print({name: {t: (v if p < 0.05 else 0) for t, (v, p) in terms.items()} for name, terms in results.items()})


if __name__ == "__main__":
    main()

"""Backtest: do experience, guard shooting and free-throw shooting say
something the college basketball ratings don't - especially about
tournament upsets?

User question (2026-11-10): "What about rebounding, tempo, free throw
shooting, and looking for teams with good perimeter shooting guards from
lower ranked teams (especially upper class who may not be as nervous)?"
(Rebounding and tempo are in backtest_cbb_style_elo.py.)

From hoopR's player box scores (every Division I game since 2008), each
team's make-up going INTO each game, from its earlier games that season:
    experience         minutes-weighted college seasons already played
                       (a season counts when the player logged 100+
                       minutes in it; so 0 = first year, 3 = fourth)
    guard_experience   the same for its guards
    guard_threes       threes made per game by its guards
    guard_three_pct    its guards' three-point percentage (100 attempts
                       of the league's mixed in)
    upper_guard_threes threes made per game by guards in at least their
                       third college season - the "upper-class shooting
                       guards" of the question
    ft_pct             team free-throw percentage (100 attempts mixed in)

The base is the week-by-week replay of backtest_cbb_ratings.py. Scored on
2013 on (experience needs earlier seasons to count).

1. The margin: is the miss of our margin explained by the two teams'
   differences in these? Walk-forward, like every add-on test here.
2. March: does any of it matter MORE in the NCAA tournament than in the
   regular season? (the "not as nervous" idea)
3. Upsets: with our chance held fixed, do the underdog's traits move the
   upset odds - in the NCAA tournament, on neutral floors, and overall?
4. The question as asked: tournament underdogs split by how many threes
   their upper-class guards make.

Usage:
    python3 -I scripts/backtest_cbb_players.py --raw DIR --players DIR
(--raw as backtest_cbb_style_elo.py; --players holds hoopR's
player_box_<season>.parquet files, downloaded there when missing.)

RESULT (2026-11-10; 68,665 games 2012-2026 with both teams at least five
games in, scored from 2013: 64,181 games, 865 in the NCAA tournament;
base log loss 0.5422, margin off by 8.765):
  1. The margin, walk-forward. Experience: -0.0007 (95% CI -0.0010 to
     -0.0005), 14 of 14 seasons - a more experienced team beats its rating
     by about half a point per standard deviation (t 6.1) - but +0.0007 on
     tournament games. Guard shooting: -0.0006 (11 of 14 seasons), and the
     sign is the opposite of the idea: a team whose guards shoot a high
     percentage does WORSE than its rating (-0.57 points per standard
     deviation, t -11), since some of a high percentage is luck that the
     rating has already banked. Free-throw percentage: -0.0001, nothing.
     All six: -0.0009 overall, +0.0019 (worse) on tournament games.
  2. March (932 tournament games): nothing matters more there. The one
     term that differs from the regular season is experience, and it
     points the other way: -1.2 points per standard deviation in the
     tournament (t -1.9; differs from the regular season p 0.01).
  3. Upsets, our chance held fixed. NCAA tournament (865 games, 253
     upsets; we said 28.0%, it was 29.2%): no trait is reliable - upper-
     class guard threes +20% upset odds per standard deviation (p 0.25),
     experience -18% (p 0.19), guard three-point percentage -5% (p 0.57),
     free-throw percentage +6% (p 0.48). Neutral floors (4,106) and every
     game (39,873): effects of 3-8%, with favorites' experience lowering
     upset odds a little (-7%, p 0.00-0.06).
  4. NCAA underdogs we gave 15-45% (560 games; they won 33.9%, we said
     31.7%), by thirds of upper-class guard threes: 34.6%, 33.5%, 33.7%.
     By guard three-point percentage: 37.4%, 31.7%, 32.6%. By experience:
     34.2%, 34.4%, 33.2%. Underdogs with experienced guards who shoot
     well: 34.9% of 152 (we said 32.5%); everyone else 33.6% of 408 (we
     said 31.4%) - the same two points above our number either way.
  So: experienced, good-shooting guards do not mark the underdogs that
  win, and nothing here helps tournament picks. Not added to the model.
"""

import argparse
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import requests
import statsmodels.api as sm
from scipy.stats import norm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import backtest_cbb_ratings as base  # noqa: E402
import backtest_cbb_style_elo as style_elo  # noqa: E402
from mlb_metrics import cbb_data, cbb_ratings  # noqa: E402

SEASONS = base.SEASONS
FIRST_USED = 2012           # experience is counted from 2008's box scores, so it is whole from here
TRAITS = ["experience", "guard_experience", "guard_threes", "guard_three_pct", "upper_guard_threes", "ft_pct"]
MIN_SEASON_MINUTES = 100
UPPER_CLASS = 2             # at least two earlier college seasons
PLAYER_COLUMNS = ["game_id", "season", "game_date", "athlete_id", "team_id", "minutes", "three_point_field_goals_made",
                  "three_point_field_goals_attempted", "free_throws_made", "free_throws_attempted",
                  "athlete_position_abbreviation"]


def load_players(directory: str, season: int) -> pd.DataFrame:
    path = os.path.join(directory, f"player_box_{season}.parquet")
    if not os.path.exists(path):
        os.makedirs(directory, exist_ok=True)
        url = f"{cbb_data.BASE_URL}/player_box/parquet/player_box_{season}.parquet"
        response = requests.get(url, timeout=240)
        response.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(response.content)
    box = pd.read_parquet(path, columns=PLAYER_COLUMNS)
    for column in PLAYER_COLUMNS[5:10]:
        box[column] = pd.to_numeric(box[column], errors="coerce").fillna(0.0)
    box = box[box["athlete_id"].notna() & (box["minutes"] > 0)].copy()
    box["game_id"] = box["game_id"].astype(str)
    box["team_id"] = box["team_id"].astype(int)
    box["guard"] = box["athlete_position_abbreviation"].astype(str).isin(["G", "PG", "SG", "G-F"])
    return box


def add_experience(boxes: dict) -> None:
    """Adds `experience` to each season's player rows: how many earlier
    seasons the player logged MIN_SEASON_MINUTES or more in (0-4)."""
    seasons_played = {}
    for season in sorted(boxes):
        minutes = boxes[season].groupby("athlete_id")["minutes"].sum()
        boxes[season]["experience"] = boxes[season]["athlete_id"].map(
            lambda a: min(len(seasons_played.get(a, ())), 4)).astype(float)
        for athlete in minutes.index[minutes >= MIN_SEASON_MINUTES]:
            seasons_played.setdefault(athlete, set()).add(season)


def team_traits(box: pd.DataFrame) -> pd.DataFrame:
    """TRAITS for each team going into each of its games (season-to-date,
    earlier games only), as [game_id, team_id, *TRAITS]."""
    b = box.copy()
    b["min_exp"] = b["minutes"] * b["experience"]
    b["g_min"] = b["minutes"].where(b["guard"], 0.0)
    b["g_min_exp"] = b["min_exp"].where(b["guard"], 0.0)
    b["g_tpm"] = b["three_point_field_goals_made"].where(b["guard"], 0.0)
    b["g_tpa"] = b["three_point_field_goals_attempted"].where(b["guard"], 0.0)
    b["ug_tpm"] = b["g_tpm"].where(b["experience"] >= UPPER_CLASS, 0.0)
    sums = ["minutes", "min_exp", "g_min", "g_min_exp", "g_tpm", "g_tpa", "ug_tpm", "free_throws_made", "free_throws_attempted"]
    games = b.groupby(["team_id", "game_id", "game_date"], as_index=False)[sums].sum().sort_values(["team_id", "game_date", "game_id"])
    group = games.groupby("team_id")
    before = {c: group[c].cumsum() - games[c] for c in sums}
    played = group.cumcount()
    league_three = games["g_tpm"].sum() / max(games["g_tpa"].sum(), 1)
    league_ft = games["free_throws_made"].sum() / max(games["free_throws_attempted"].sum(), 1)
    out = games[["game_id", "team_id"]].copy()
    out["games_before"] = played.to_numpy()
    out["experience"] = (before["min_exp"] / before["minutes"].where(before["minutes"] > 0)).to_numpy()
    out["guard_experience"] = (before["g_min_exp"] / before["g_min"].where(before["g_min"] > 0)).to_numpy()
    out["guard_threes"] = (before["g_tpm"] / played.where(played > 0)).to_numpy()
    out["upper_guard_threes"] = (before["ug_tpm"] / played.where(played > 0)).to_numpy()
    out["guard_three_pct"] = ((before["g_tpm"] + 100 * league_three) / (before["g_tpa"] + 100)).to_numpy()
    out["ft_pct"] = ((before["free_throws_made"] + 100 * league_ft) / (before["free_throws_attempted"] + 100)).to_numpy()
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--players", required=True)
    args = parser.parse_args()
    pd.set_option("display.width", 250)

    games_by_season, boxes = {}, {}
    for season in SEASONS:
        schedules = style_elo.load_raw(args.raw, "schedules", season)
        games_by_season[season] = cbb_ratings.prepare_games(schedules, style_elo.load_raw(args.raw, "team_box", season))
        base._RANKED[season] = cbb_ratings.preseason_ranked(schedules)
        boxes[season] = load_players(args.players, season)
    add_experience(boxes)
    traits = pd.concat([team_traits(boxes[season]) for season in SEASONS], ignore_index=True)
    all_games = pd.concat(games_by_season.values(), ignore_index=True)

    rows = base.replay(games_by_season).merge(all_games[["game_id", "home_id", "away_id"]], on="game_id")
    for side in ("home", "away"):
        rows = rows.merge(traits.rename(columns={c: f"{side}_{c}" for c in traits.columns if c != "game_id"})
                          .rename(columns={f"{side}_team_id": f"{side}_id"}), on=["game_id", f"{side}_id"], how="inner")
    rows = rows[(rows["season"] >= FIRST_USED) & (rows["home_games_before"] >= 5) & (rows["away_games_before"] >= 5)]
    rows = rows.dropna(subset=[f"{side}_{t}" for side in ("home", "away") for t in TRAITS])
    rows = rows.sort_values(["date", "game_id"]).reset_index(drop=True)
    rows["miss"] = rows["actual"] - rows["margin"]
    style_elo.FIRST_SCORED = FIRST_USED + 1
    sigma = np.full(len(rows), np.nan)
    for season in sorted(rows["season"].unique()):
        if season > FIRST_USED:
            sigma[(rows["season"] == season).to_numpy()] = np.sqrt((rows.loc[rows["season"] < season, "miss"] ** 2).mean())
    rows["sigma"] = sigma
    rows["p"] = norm.cdf(rows["margin"] / rows["sigma"])
    for t in TRAITS:
        rows[f"d_{t}"] = rows[f"home_{t}"] - rows[f"away_{t}"]
    diffs = [f"d_{t}" for t in TRAITS]
    scored = rows["season"] > FIRST_USED
    s = rows[scored].reset_index(drop=True)
    print(f"{len(rows)} games with both teams' make-up (each at least 5 games into its season), {FIRST_USED}-{rows['season'].max()};"
          f" scored from {FIRST_USED + 1}: {len(s)} games, {int(s['tournament'].sum())} in the NCAA tournament")
    team_side = pd.concat([rows[[f"home_{t}" for t in TRAITS]].set_axis(TRAITS, axis=1), rows[[f"away_{t}" for t in TRAITS]].set_axis(TRAITS, axis=1)])
    print("a team's make-up, average and spread:", {t: f"{team_side[t].mean():.2f} +/- {team_side[t].std():.2f}" for t in TRAITS})
    print(f"base log loss {style_elo.log_loss(s['p'], s['actual'] > 0).mean():.4f}, margin off by {s['miss'].abs().mean():.3f}")

    print("\n1. THE MARGIN (walk-forward)")
    for name, features in (("experience", ["d_experience"]), ("guard shooting (3 terms)", ["d_guard_threes", "d_guard_three_pct", "d_upper_guard_threes"]),
                           ("free-throw percentage", ["d_ft_pct"]), ("all six", diffs)):
        rows["adj"] = style_elo.walk_forward(rows, features, "miss")
        s = rows[scored].reset_index(drop=True)
        style_elo.compare(name, s, norm.cdf((s["margin"] + s["adj"]) / s["sigma"]), s["p"], s["margin"] + s["adj"])
    z = (rows[diffs] - rows[diffs].mean()) / rows[diffs].std()
    fit = sm.OLS(rows["miss"], sm.add_constant(z)).fit()
    print("  points of margin per standard deviation of each difference, all games (t): "
          + ", ".join(f"{t} {fit.params[f'd_{t}']:+.2f} ({fit.tvalues[f'd_{t}']:+.1f})" for t in TRAITS))

    print("\n2. MARCH: the same, NCAA tournament games only, and whether it differs from the regular season")
    t_rows = rows[rows["tournament"]]
    zt = (t_rows[diffs] - rows[diffs].mean()) / rows[diffs].std()
    fit_t = sm.OLS(t_rows["miss"], sm.add_constant(zt)).fit()
    both = z.copy()
    for d in diffs:
        both[f"{d}_x_tourney"] = z[d] * rows["tournament"].astype(float)
    fit_x = sm.OLS(rows["miss"], sm.add_constant(both)).fit()
    print(f"  {len(t_rows)} tournament games. Points of margin per standard deviation (t), then the p-value that March differs:")
    print("    " + ", ".join(f"{t} {fit_t.params[f'd_{t}']:+.2f} ({fit_t.tvalues[f'd_{t}']:+.1f}; differs p {fit_x.pvalues[f'd_{t}_x_tourney']:.2f})"
                             for t in TRAITS))

    print("\n3. UPSETS: change in the underdog's odds per standard deviation, our chance held fixed (p)")
    s = rows[scored].copy()
    s["fav_home"] = s["margin"] > 0
    s["p_fav"] = np.where(s["fav_home"], s["p"], 1 - s["p"])
    s["upset"] = (s["actual"] > 0) != s["fav_home"]
    for t in TRAITS:
        s[f"dog_{t}"] = np.where(s["fav_home"], s[f"away_{t}"], s[f"home_{t}"])
        s[f"fav_{t}"] = np.where(s["fav_home"], s[f"home_{t}"], s[f"away_{t}"])
    s["logit"] = np.log((1 - s["p_fav"]) / s["p_fav"])
    columns = [f"dog_{t}" for t in TRAITS] + ["fav_experience", "fav_ft_pct"]
    pools = {"NCAA tournament": s[s["tournament"]],
             "neutral floor, favorite 60-90%": s[s["neutral"] & s["p_fav"].between(0.6, 0.9)],
             "every game, favorite 60-90%": s[s["p_fav"].between(0.6, 0.9)]}
    for name, pool in pools.items():
        zp = (pool[columns] - pool[columns].mean()) / pool[columns].std()
        fit = sm.Logit(pool["upset"].astype(float), sm.add_constant(pd.concat([pool[["logit"]], zp], axis=1))).fit(disp=0)
        print(f"  {name}: {len(pool)} games, {int(pool['upset'].sum())} upsets ({pool['upset'].mean():.3f}; we said {(1 - pool['p_fav']).mean():.3f})")
        print("    " + ", ".join(f"{c} {np.exp(fit.params[c]) - 1:+.1%} ({fit.pvalues[c]:.2f})" for c in columns))

    print("\n4. NCAA underdogs we gave 15-45%, split in thirds (upset rate; what we said)")
    t = s[s["tournament"] & (1 - s["p_fav"]).between(0.15, 0.45)]
    print(f"  {len(t)} games, {int(t['upset'].sum())} upsets")
    for trait in ("dog_upper_guard_threes", "dog_guard_threes", "dog_guard_three_pct", "dog_experience", "dog_guard_experience", "dog_ft_pct"):
        thirds = pd.qcut(t[trait], 3, labels=["low", "middle", "high"])
        g = t.groupby(thirds, observed=True).agg(games=("upset", "size"), won=("upset", "mean"), said=("p_fav", lambda x: (1 - x).mean()))
        print(f"  {trait:24s} " + "; ".join(f"{k}: {v['won']:.3f} of {int(v['games'])} (said {v['said']:.3f})" for k, v in g.iterrows()))
    # The exact combination: experienced guards AND they shoot.
    combo = (t["dog_guard_experience"] > t["dog_guard_experience"].median()) & (t["dog_guard_three_pct"] > t["dog_guard_three_pct"].median())
    print(f"  experienced guards who shoot well (both above the middle): {t.loc[combo, 'upset'].mean():.3f} of {int(combo.sum())}"
          f" (said {(1 - t.loc[combo, 'p_fav']).mean():.3f}); everyone else {t.loc[~combo, 'upset'].mean():.3f} of {int((~combo).sum())}"
          f" (said {(1 - t.loc[~combo, 'p_fav']).mean():.3f})")


if __name__ == "__main__":
    main()

"""Backtest: do Elo or style matchups add to the college basketball
ratings' game predictions?

User request (2026-11-10): "test style matchups and Elo." The live model
(cbb_ratings.py) gives every team an offense and a defense number over
the whole season, so a game is one team's offense against the other's
defense at the two teams' pace - nothing about HOW they play, and no
separate memory of results.

The base is the same week-by-week replay as backtest_cbb_ratings.py
(ratings refit each week from earlier games only), scored on 2011-2026.
Everything added on top is also known before tip-off, and every fit on
top is walk-forward: fit on the seasons before, scored on the season.

1. Elo. A plain Elo over every Division I game since 2008, updated game
   by game (margin-of-victory multiplier, home edge, part of each rating
   kept between seasons), over a small grid of settings. Scored alone, and
   blended with our margin (regression of the real margin on both).

2. Style, the average. Each team's season-to-date style before the game
   (five games of league average mixed in): how often it shoots threes,
   turns it over, gets offensive rebounds and gets to the line, and what
   its defense allows of each. For each of the four, the matchup term is
   (offense's lean) x (defense's lean), home side minus away side - a
   three-happy offense against a defense that gives up threes. Also the
   plain differences (does a style beat its rating?). The miss of our
   margin is regressed on them.

3. Style, the spread. Does the size of the miss depend on the game - the
   pace (fewer possessions, less room for the better team) or how many
   threes get shot (more luck)? log(miss squared) is regressed on those,
   which gives each game its own spread for the win chance instead of one
   number for all.

4. Upsets. NCAA tournament games and every neutral-floor game with a
   60-90% favorite: does the underdog's style say more than our chance
   already does?

Usage:
    PYTHONPATH=src python3 scripts/backtest_cbb_style_elo.py --raw DIR
(DIR holds hoopR's full mbb_schedule_<season>.parquet and
team_box_<season>.parquet files; missing ones are downloaded into it.)

RESULT: not run yet.
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

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import backtest_cbb_ratings as base  # noqa: E402
from mlb_metrics import cbb_data, cbb_ratings, config  # noqa: E402

SEASONS = base.SEASONS
FIRST_SCORED = SEASONS[0] + 3          # walk-forward fits need a few seasons behind them
STYLE = ["three_rate", "turnover_rate", "oreb_rate", "ft_rate"]
SHRINK_GAMES = 5.0
RNG = np.random.default_rng(0)


def load_raw(raw: str, table: str, season: int) -> pd.DataFrame:
    name = f"mbb_schedule_{season}.parquet" if table == "schedules" else f"team_box_{season}.parquet"
    path = os.path.join(raw, name)
    if not os.path.exists(path):
        os.makedirs(raw, exist_ok=True)
        response = requests.get(cbb_data.table_url(table, season), timeout=180)
        response.raise_for_status()
        with open(path, "wb") as fh:
            fh.write(response.content)
    return pd.read_parquet(path)


def team_games(box: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """One row per team per Division I game with its own and its
    opponent's box-score counts."""
    num = lambda c: pd.to_numeric(box[c], errors="coerce")
    turnovers = num("total_turnovers").fillna(num("turnovers")) if "total_turnovers" in box.columns else num("turnovers")
    own = pd.DataFrame({
        "game_id": box["game_id"].astype(str), "team_id": box["team_id"].astype(int),
        "fga": num("field_goals_attempted"), "tpa": num("three_point_field_goals_attempted"),
        "fta": num("free_throws_attempted"), "oreb": num("offensive_rebounds"), "dreb": num("defensive_rebounds"),
        "tov": turnovers,
    }).drop_duplicates(["game_id", "team_id"])
    d1 = games[games["home_d1"] & games["away_d1"]][["game_id", "season", "date", "home_id", "away_id", "possessions"]]
    sides = pd.concat([d1.rename(columns={"home_id": "team_id", "away_id": "opp_id"}),
                       d1.rename(columns={"away_id": "team_id", "home_id": "opp_id"})], ignore_index=True)
    rows = sides.merge(own, on=["game_id", "team_id"], how="inner")
    rows = rows.merge(own.rename(columns={c: f"opp_{c}" for c in own.columns if c not in ("game_id", "team_id")})
                      .rename(columns={"team_id": "opp_id"}), on=["game_id", "opp_id"], how="inner")
    return rows.dropna()


def style_before(rows: pd.DataFrame) -> pd.DataFrame:
    """Each team's style going INTO each game: season-to-date rates from
    its earlier games only, with SHRINK_GAMES games of the league's average
    mixed in. off_* is what the team does, def_* what its opponents do
    against it; `pace` is its possessions a game."""
    rows = rows.sort_values(["season", "team_id", "date", "game_id"]).reset_index(drop=True)
    pairs = {   # rate: (numerator, denominator) from the offense's side
        "three_rate": ("tpa", "fga"), "turnover_rate": ("tov", "possessions"), "ft_rate": ("fta", "fga"),
    }
    rows["oreb_chances"] = rows["oreb"] + rows["opp_dreb"]
    rows["opp_oreb_chances"] = rows["opp_oreb"] + rows["dreb"]
    rows["opp_possessions"] = rows["possessions"]
    pairs["oreb_rate"] = ("oreb", "oreb_chances")
    out = rows[["game_id", "season", "team_id"]].copy()
    group = rows.groupby(["season", "team_id"], sort=False)
    played = group.cumcount()
    out["games_before"] = played
    out["pace"] = ((group["possessions"].cumsum() - rows["possessions"]) + SHRINK_GAMES * rows["possessions"].mean()) \
        / (played + SHRINK_GAMES)
    for name, (numerator, denominator) in pairs.items():
        for side, prefix in (("off", ""), ("def", "opp_")):
            n, d = rows[prefix + numerator], rows[prefix + denominator]
            league = n.sum() / d.sum()
            per_game = d.mean()
            cum_n = n.groupby([rows["season"], rows["team_id"]]).cumsum() - n
            cum_d = d.groupby([rows["season"], rows["team_id"]]).cumsum() - d
            out[f"{side}_{name}"] = (cum_n + SHRINK_GAMES * per_game * league) / (cum_d + SHRINK_GAMES * per_game)
    return out


def elo_margins(games: pd.DataFrame, k: float, keep: float, home: float) -> pd.Series:
    """The home side's Elo edge (in Elo points, home edge included unless
    neutral) before each Division I game, indexed by game_id."""
    g = games[games["home_d1"] & games["away_d1"]].sort_values(["date", "game_id"])
    rating, season_of, edges = {}, None, []
    for season, home_id, away_id, hs, as_, neutral in zip(g["season"], g["home_id"], g["away_id"], g["home_score"],
                                                          g["away_score"], g["neutral"]):
        if season != season_of:
            rating = {t: 1500 + keep * (r - 1500) for t, r in rating.items()}
            season_of = season
        rh, ra = rating.get(home_id, 1500.0), rating.get(away_id, 1500.0)
        edge = rh - ra + (0.0 if neutral else home)
        edges.append(edge)
        expected = 1 / (1 + 10 ** (-edge / 400))
        margin = hs - as_
        winner_edge = edge if margin > 0 else -edge
        multiplier = np.log(abs(margin) + 1) * 2.2 / (winner_edge * 0.001 + 2.2)
        shift = k * multiplier * ((1.0 if margin > 0 else 0.0) - expected)
        rating[home_id], rating[away_id] = rh + shift, ra - shift
    return pd.Series(edges, index=g["game_id"].to_numpy())


def log_loss(p, won) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return -np.where(np.asarray(won, dtype=bool), np.log(p), np.log(1 - p))


def compare(name: str, rows: pd.DataFrame, new_p, base_p, new_margin=None) -> None:
    """Prints the new chance against the base one on the same games: log
    loss, the change with a bootstrap interval, seasons better, and the
    same for the parts of the season and the NCAA tournament."""
    won = rows["actual"].to_numpy() > 0
    diff = log_loss(new_p, won) - log_loss(base_p, won)
    boots = [diff[RNG.integers(0, len(diff), len(diff))].mean() for _ in range(500)]
    by_season = pd.Series(diff).groupby(rows["season"].to_numpy()).mean()
    line = (f"  {name:34s} log loss {log_loss(new_p, won).mean():.4f} ({diff.mean():+.4f}, 95% CI {np.percentile(boots, 2.5):+.4f}"
            f" to {np.percentile(boots, 97.5):+.4f}), better in {(by_season < 0).sum()} of {len(by_season)} seasons")
    if new_margin is not None:
        line += f", margin off by {np.abs(rows['actual'].to_numpy() - np.asarray(new_margin)).mean():.3f}"
    month = rows["date"].dt.month.to_numpy()
    parts = {"Nov-Dec": np.isin(month, [11, 12]), "Jan-Feb": np.isin(month, [1, 2]), "Mar-Apr": np.isin(month, [3, 4]),
             "NCAA": rows["tournament"].to_numpy(dtype=bool)}
    line += " | " + ", ".join(f"{part} {diff[mask].mean():+.4f}" for part, mask in parts.items())
    print(line)


def walk_forward(rows: pd.DataFrame, features: list, target: str) -> np.ndarray:
    """For each season from FIRST_SCORED on, an OLS of `target` on
    `features` fit on the seasons before, predicted for that season.
    NaN before FIRST_SCORED."""
    out = np.full(len(rows), np.nan)
    for season in sorted(rows["season"].unique()):
        if season < FIRST_SCORED:
            continue
        train, test = rows[rows["season"] < season], rows["season"] == season
        fit = sm.OLS(train[target], sm.add_constant(train[features], has_constant="add")).fit()
        out[test.to_numpy()] = fit.predict(sm.add_constant(rows.loc[test, features], has_constant="add"))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    args = parser.parse_args()
    pd.set_option("display.width", 250)

    games_by_season, style_rows = {}, []
    for season in SEASONS:
        schedules, box = load_raw(args.raw, "schedules", season), load_raw(args.raw, "team_box", season)
        games_by_season[season] = cbb_ratings.prepare_games(schedules, box)
        base._RANKED[season] = cbb_ratings.preseason_ranked(schedules)
        style_rows.append(style_before(team_games(box, games_by_season[season])))
    style = pd.concat(style_rows, ignore_index=True)
    all_games = pd.concat(games_by_season.values(), ignore_index=True)

    rows = base.replay(games_by_season)
    rows = rows.merge(all_games[["game_id", "home_id", "away_id"]], on="game_id")
    for side in ("home", "away"):
        rows = rows.merge(style.drop(columns=["season"]).rename(
            columns={c: f"{side}_{c}" for c in style.columns if c not in ("game_id", "season")}
        ).rename(columns={f"{side}_team_id": f"{side}_id"}), on=["game_id", f"{side}_id"], how="inner")
    rows = rows.sort_values(["date", "game_id"]).reset_index(drop=True)
    rows["miss"] = rows["actual"] - rows["margin"]
    print(f"{len(rows)} replayed games with both teams' style, {rows['season'].min()}-{rows['season'].max()};"
          f" scored from {FIRST_SCORED}")
    scored = rows["season"] >= FIRST_SCORED

    # The base chance, with the spread fit walk-forward too (so every comparison is like for like).
    sigma = np.full(len(rows), np.nan)
    for season in sorted(rows["season"].unique()):
        if season >= FIRST_SCORED:
            sigma[(rows["season"] == season).to_numpy()] = np.sqrt((rows.loc[rows["season"] < season, "miss"] ** 2).mean())
    rows["sigma"] = sigma
    rows["p"] = norm.cdf(rows["margin"] / rows["sigma"])
    s = rows[scored].reset_index(drop=True)
    won = s["actual"] > 0
    print(f"base: {len(s)} games, log loss {log_loss(s['p'], won).mean():.4f}, margin off by {s['miss'].abs().mean():.3f},"
          f" favorite won {((s['p'] > 0.5) == won).mean():.3f}; NCAA tournament {int(s['tournament'].sum())} games,"
          f" log loss {log_loss(s.loc[s['tournament'], 'p'], won[s['tournament']]).mean():.4f}")

    # ---- 1. Elo ---------------------------------------------------------------------------------------
    print("\n1. ELO")
    best = None
    for k in (20.0, 30.0, 45.0):
        for keep in (0.6, 0.75, 0.9):
            for home in (70.0, 100.0):
                edge = rows["game_id"].map(elo_margins(all_games, k, keep, home))
                alone = 1 / (1 + 10 ** (-edge[scored] / 400))
                loss = log_loss(alone, rows.loc[scored, "actual"] > 0).mean()
                if best is None or loss < best[0]:
                    best = (loss, k, keep, home, edge)
    loss, k, keep, home, edge = best
    print(f"  best Elo alone (K {k:.0f}, keeps {keep:.0%} between seasons, home edge {home:.0f}): log loss {loss:.4f}"
          f" (the 18 settings tried are picked on these same games, so this flatters Elo slightly)")
    rows["elo_edge"] = edge
    rows["elo_only"] = walk_forward(rows, ["elo_edge"], "actual")
    rows["blend"] = walk_forward(rows, ["margin", "elo_edge"], "actual")
    weights = sm.OLS(rows["actual"], sm.add_constant(rows[["margin", "elo_edge"]])).fit()
    print(f"  real margin on both, all seasons: our margin {weights.params['margin']:.3f} (t {weights.tvalues['margin']:.1f}),"
          f" Elo edge per 100 points {100 * weights.params['elo_edge']:.3f} (t {weights.tvalues['elo_edge']:.1f})")
    s = rows[scored].reset_index(drop=True)
    for name, column in (("Elo alone, as a margin", "elo_only"), ("our margin + Elo", "blend")):
        resid = np.sqrt(((s["actual"] - s[column]) ** 2).groupby(s["season"]).transform("mean"))   # same-season spread: generous to the new one
        compare(name, s, norm.cdf(s[column] / s["sigma"]), s["p"], s[column])

    # ---- 2. Style, the average ---------------------------------------------------------------------------
    print("\n2. STYLE AND THE MARGIN")
    league = {f"{side}_{name}": pd.concat([rows[f"home_{side}_{name}"], rows[f"away_{side}_{name}"]]).mean()
              for side in ("off", "def") for name in STYLE}
    matchup, plain = [], []
    for name in STYLE:
        lean = lambda team, side: rows[f"{team}_{side}_{name}"] - league[f"{side}_{name}"]
        rows[f"x_{name}"] = lean("home", "off") * lean("away", "def") - lean("away", "off") * lean("home", "def")
        rows[f"d_off_{name}"] = rows[f"home_off_{name}"] - rows[f"away_off_{name}"]
        rows[f"d_def_{name}"] = rows[f"home_def_{name}"] - rows[f"away_def_{name}"]
        matchup.append(f"x_{name}")
        plain += [f"d_off_{name}", f"d_def_{name}"]
    for name, features in (("matchup terms (4)", matchup), ("plain style differences (8)", plain), ("both (12)", matchup + plain)):
        rows["adj"] = walk_forward(rows, features, "miss")
        s = rows[scored].reset_index(drop=True)
        compare(name, s, norm.cdf((s["margin"] + s["adj"]) / s["sigma"]), s["p"], s["margin"] + s["adj"])
    fit = sm.OLS(rows["miss"], sm.add_constant((rows[matchup + plain] - rows[matchup + plain].mean()) / rows[matchup + plain].std())).fit()
    print("  points of margin per standard deviation of each term, all seasons (t):")
    print("   ", ", ".join(f"{c} {fit.params[c]:+.2f} ({fit.tvalues[c]:+.1f})" for c in matchup + plain))

    # ---- 3. Style, the spread ----------------------------------------------------------------------------
    print("\n3. STYLE AND THE SPREAD OF THE MISS")
    rows["log_miss2"] = np.log(rows["miss"] ** 2 + 1.0)
    rows["pace_sum"] = rows["home_pace"] + rows["away_pace"]
    rows["threes"] = rows["home_off_three_rate"] + rows["away_off_three_rate"]
    rows["lopsided"] = rows["margin"].abs()
    for name, features in (("pace", ["pace_sum"]), ("threes", ["threes"]), ("how lopsided", ["lopsided"]),
                           ("pace + threes + lopsided", ["pace_sum", "threes", "lopsided"])):
        rows["fit_log"] = walk_forward(rows, features, "log_miss2")
        # A game's spread: the season's base spread scaled by its fitted miss against the average fitted miss.
        rows["sigma_game"] = rows["sigma"] * np.sqrt(np.exp(rows["fit_log"] - rows.groupby("season")["fit_log"].transform("mean")))
        s = rows[scored].reset_index(drop=True)
        compare(name, s, norm.cdf(s["margin"] / s["sigma_game"]), s["p"])
    fit = sm.OLS(rows["miss"].abs(), sm.add_constant(rows[["pace_sum", "threes", "lopsided"]])).fit()
    sd = rows[["pace_sum", "threes", "lopsided"]].std()
    print("  size of the miss, points per standard deviation (t): "
          + ", ".join(f"{c} {fit.params[c] * sd[c]:+.2f} ({fit.tvalues[c]:+.1f})" for c in sd.index))
    s = rows[scored]
    thirds = pd.qcut(s["pace_sum"], 3, labels=["slow", "middle", "fast"])
    print("  sd of the miss by pace:", s.groupby(thirds, observed=True)["miss"].std().round(2).to_dict(),
          "| by threes:", s.groupby(pd.qcut(s["threes"], 3, labels=["few", "middle", "many"]), observed=True)["miss"].std().round(2).to_dict())

    # ---- 4. Upsets ---------------------------------------------------------------------------------------
    print("\n4. UPSETS: does the underdog's style say more than our chance does?")
    s = rows[scored].copy()
    s["fav_home"] = s["margin"] > 0
    s["p_fav"] = np.where(s["fav_home"], s["p"], 1 - s["p"])
    s["upset"] = (s["actual"] > 0) != s["fav_home"]
    for column in ("off_three_rate", "def_turnover_rate", "off_oreb_rate", "off_turnover_rate", "pace"):
        s[f"dog_{column}"] = np.where(s["fav_home"], s[f"away_{column}"], s[f"home_{column}"])
        s[f"fav_{column}"] = np.where(s["fav_home"], s[f"home_{column}"], s[f"away_{column}"])
    s["logit"] = np.log((1 - s["p_fav"]) / s["p_fav"])
    traits = ["dog_off_three_rate", "dog_def_turnover_rate", "dog_off_oreb_rate", "dog_off_turnover_rate", "dog_pace",
              "fav_off_three_rate", "fav_off_turnover_rate"]
    pools = {"NCAA tournament": s[s["tournament"]],
             "neutral floor, favorite 60-90%": s[s["neutral"] & s["p_fav"].between(0.6, 0.9)],
             "every game, favorite 60-90%": s[s["p_fav"].between(0.6, 0.9)]}
    for name, pool in pools.items():
        z = (pool[traits] - pool[traits].mean()) / pool[traits].std()
        fit = sm.Logit(pool["upset"].astype(float), sm.add_constant(pd.concat([pool[["logit"]], z], axis=1))).fit(disp=0)
        print(f"  {name}: {len(pool)} games, {int(pool['upset'].sum())} upsets ({pool['upset'].mean():.3f}; we said"
              f" {(1 - pool['p_fav']).mean():.3f}). Change in upset odds per standard deviation (p):")
        print("    " + ", ".join(f"{t.replace('_rate', '')} {np.exp(fit.params[t]) - 1:+.1%} ({fit.pvalues[t]:.2f})" for t in traits))
    # Does it sort tournament underdogs? Underdogs given 20-45% by us, split by three-point rate.
    t = s[s["tournament"] & (1 - s["p_fav"]).between(0.2, 0.45)]
    high = t["dog_off_three_rate"] > t["dog_off_three_rate"].median()
    print(f"  NCAA underdogs we gave 20-45% ({len(t)}): three-happy half won {t.loc[high, 'upset'].mean():.3f} (said"
          f" {(1 - t.loc[high, 'p_fav']).mean():.3f}), the other half {t.loc[~high, 'upset'].mean():.3f} (said"
          f" {(1 - t.loc[~high, 'p_fav']).mean():.3f})")


if __name__ == "__main__":
    main()

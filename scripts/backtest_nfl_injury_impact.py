"""How much does a team's offense or defense lose when starters at a
position miss a game - and does the game-pick model already know?

Starters come from snap counts: a player's share of his side's snaps over
the team's previous 4 games (0 for a game he missed), and he's a regular
when that's at least half. A regular with no snaps this game is out, so
each position group gets a "missing" share: the snap share of its absent
regulars over all its regulars' (0 = everyone played, 1 = the whole group
out). Groups: RB, WR, TE, OL (offense); DL, LB, CB, S (defense). QBs are
left to QB+, which already rates whoever starts.

1. Unit cost: each team-game's EPA per play (dropbacks and designed runs,
   offense and defense) against the team's and opponent's previous 8
   games (play-weighted), plus each group's missing share - the
   coefficient is EPA per play lost with the whole group out.
2. Win probability: home win ~ the game-pick model's walk-forward
   log-odds (QB+ features, 2018-2026) + home minus away missing share per
   group - log-odds the model misses - and the same against Vegas's
   closing line, to see which absences Vegas prices and we don't.

RESULT (2026-10-26, 2016-2026 snap counts; 2,221 games 2018-2026 for 2):

  EPA per play with the whole group out (one of five linemen = 1/5 of it):
      OL     dropbacks -.116 (p < .001)   runs -.057 (p = .02)
      TE     dropbacks -.038 (p = .05)
      RB     runs -.029 (p = .03)
      CB     dropbacks allowed +.064 (p = .004)
      LB     dropbacks allowed +.056 (p = .03)
      WR, DL, S: no clear unit effect

  Home-win log-odds the model misses, per whole group (home minus away):
      OL -.85 (p < .001)   CB -.45 (p = .003)   DL -.42 (p = .04)
      WR -.39 (p = .02)    LB -.18, S -.08, TE -.05, RB +.04 (not significant)
  Vegas prices every group (-.07 to -.20 per whole group against our line),
  most for WR, DL, OL and CB.

"Out" here is "took no snaps", which is known when inactives are announced
90 minutes before kickoff; the injury report's Out/Doubtful is the
earlier, noisier version. These per-group log-odds drive the injury
toggles on the NFL page's what-if panel (nfl_whatif.INJURY_LOG_ODDS).

Usage:
    python scripts/backtest_nfl_injury_impact.py --rows rows_qb_plus.parquet
(--rows: game-pick feature rows from nfl_game_picks.build_game_features_qb_plus,
as built by assemble_nfl_game_pick_log; rebuilt when missing.)
"""

import argparse
import io
import os
import sys
from contextlib import redirect_stdout

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import statsmodels.api as sm

import grid_search_nfl_game_picks as grid
from mlb_metrics import nfl_data, nfl_game_picks, nfl_game_picks_backtest as backtest, nfl_totals, nfl_units, nfl_whatif

SEASONS = list(range(2016, 2027))
GROUPS = nfl_whatif.GROUPS
LOOKBACK_GAMES = nfl_whatif.LOOKBACK_GAMES
REGULAR_SHARE = nfl_whatif.REGULAR_SHARE
PRIOR_GAMES = 8


def missing_shares(snaps: pd.DataFrame) -> pd.DataFrame:
    """One row per (game, team): each group's missing share."""
    snaps = snaps[snaps["game_type"] == "REG"].copy()
    snaps["team"] = nfl_totals._canonical(snaps["team"])
    team_games = snaps[["season", "week", "game_id", "team"]].drop_duplicates().sort_values(["team", "season", "week"])
    team_games["n"] = team_games.groupby("team").cumcount()
    out = team_games[["game_id", "team"]].copy()
    for group, (side, positions) in GROUPS.items():
        pct = f"{side}_pct"
        rows = snaps[snaps["position"].isin(positions)][["game_id", "team", "pfr_player_id", pct]]
        # Player x team-game grid of snap shares, 0 where he didn't play.
        grid_ = rows.pivot_table(index=["team", "game_id"], columns="pfr_player_id", values=pct, aggfunc="max")
        shares = []
        for team, g in grid_.groupby(level="team"):
            order = team_games[team_games["team"] == team].set_index("game_id")["n"]
            g = g.droplevel("team").reindex(order.sort_values().index)
            played = g.notna() & (g > 0)
            prior = g.fillna(0).shift(1).rolling(LOOKBACK_GAMES, min_periods=2).mean()
            regular = prior >= REGULAR_SHARE
            expected = prior.where(regular, 0).sum(axis=1)
            missing = prior.where(regular & ~played, 0).sum(axis=1)
            shares.append(pd.DataFrame({"team": team, "game_id": g.index,
                                        f"miss_{group}": (missing / expected.replace(0, np.nan)).values}))
        out = out.merge(pd.concat(shares), on=["game_id", "team"], how="left")
    return out


def unit_cost(units: pd.DataFrame, missing: pd.DataFrame) -> None:
    units = units.copy()
    units["team"], units["opp"] = nfl_totals._canonical(units["team"]), nfl_totals._canonical(units["opp"])
    units = units.sort_values(["season", "week"])

    def prior(key, num, den):
        g = units.groupby(key)
        s = g[num].transform(lambda x: x.shift(1).rolling(PRIOR_GAMES, min_periods=3).sum())
        n = g[den].transform(lambda x: x.shift(1).rolling(PRIOR_GAMES, min_periods=3).sum())
        return s / n

    off = missing.add_prefix("off_").rename(columns={"off_game_id": "game_id", "off_team": "team"})
    dfn = missing.add_prefix("def_").rename(columns={"def_game_id": "game_id", "def_team": "opp"})
    data = units.merge(off, on=["game_id", "team"], how="left").merge(dfn, on=["game_id", "opp"], how="left")
    offense = [f"off_miss_{g}" for g, (side, _) in GROUPS.items() if side == "offense"]
    defense = [f"def_miss_{g}" for g, (side, _) in GROUPS.items() if side == "defense"]
    print("EPA per play lost with a whole group out (offense's groups for its own plays, the defense's for plays it faces):\n")
    rows = []
    for label, num, den in (("dropbacks", "dropback_epa", "dropbacks"), ("designed runs", "run_epa", "runs")):
        d = data.assign(actual=data[num] / data[den], team_prior=prior("team", num, den), opp_prior=prior("opp", num, den))
        d = d.dropna(subset=["actual", "team_prior", "opp_prior"] + offense + defense)
        d = d[d[den] >= 5]
        X = sm.add_constant(d[["team_prior", "opp_prior"] + offense + defense].astype(float))
        fit = sm.WLS(d["actual"], X, weights=d[den]).fit()
        for col in offense + defense:
            rows.append({"plays": label, "group": col.replace("_miss_", " "), "epa_per_play": fit.params[col],
                         "se": fit.bse[col], "p": fit.pvalues[col],
                         "games_with_any_missing": int((d[col] > 0).sum())})
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.round(4).to_string(index=False))


def win_effect(rows: pd.DataFrame, missing: pd.DataFrame, schedules: pd.DataFrame) -> None:
    pred = grid.walk_forward(rows, 0.03, list(range(2018, 2027)))
    games = pred.merge(rows[["game_id", "home_team", "away_team"]], on="game_id")
    games["home_team"], games["away_team"] = nfl_totals._canonical(games["home_team"]), nfl_totals._canonical(games["away_team"])
    m = missing.set_index(["game_id", "team"])
    cols = [c for c in missing.columns if c.startswith("miss_")]
    home = m.reindex(pd.MultiIndex.from_frame(games[["game_id", "home_team"]]))[cols].to_numpy()
    away = m.reindex(pd.MultiIndex.from_frame(games[["game_id", "away_team"]]))[cols].to_numpy()
    diff = pd.DataFrame(home - away, columns=[f"{c}_home_minus_away" for c in cols], index=games.index).fillna(0)
    sched = schedules.set_index("game_id")
    imp = lambda ml: np.where(ml < 0, -ml / (-ml + 100), 100 / (ml + 100))  # noqa: E731
    h, a = imp(sched["home_moneyline"]), imp(sched["away_moneyline"])
    vegas = pd.Series(h / (h + a), index=sched.index).reindex(games["game_id"]).to_numpy()

    def logit(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return np.log(p / (1 - p))

    ok = ~np.isnan(vegas)
    ours = pd.Series(logit(games["p"].to_numpy()), index=games.index, name="logit_ours")
    print(f"\nHome win ~ model log-odds + (home minus away) missing share, {ok.sum()} games 2018-2026:\n")
    fit = sm.Logit(games.loc[ok, "y"], sm.add_constant(pd.concat([ours, diff], axis=1)[ok])).fit(disp=0)
    print(fit.summary2().tables[1].round(3).to_string())
    gap = pd.Series(logit(vegas[ok]) - ours[ok].to_numpy(), index=games.index[ok])
    print("\nVegas log-odds minus ours ~ the same (what Vegas prices that we don't):\n")
    print(sm.OLS(gap, sm.add_constant(diff[ok])).fit().summary2().tables[1].round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default=grid.RAW_DIR)
    parser.add_argument("--rows", default=None)
    args = parser.parse_args()

    snaps = pd.concat([nfl_data.load_persisted_table(args.raw_dir, "snap_counts", s) for s in SEASONS], ignore_index=True)
    missing = missing_shares(snaps)
    units = pd.concat([nfl_units.unit_game_epa(nfl_data.load_persisted_table(args.raw_dir, "pbp", s)[nfl_units.UNIT_PBP_COLUMNS])
                       for s in SEASONS], ignore_index=True)
    unit_cost(units, missing)

    grid.SEASONS = SEASONS
    if args.rows and os.path.exists(args.rows):
        rows = pd.read_parquet(args.rows)
    else:
        grid._init()
        with redirect_stdout(io.StringIO()):
            snapshots = backtest.build_multi_season_history(
                grid._DATA["schedules"], grid._DATA["team_stats"], grid._DATA["weekly"], grid._DATA["snap_counts"],
                grid._DATA["rosters_weekly"], grid._DATA["pbp"], SEASONS)
        rows = backtest.assemble_nfl_game_pick_log(snapshots, nfl_game_picks.build_game_features_qb_plus)
        rows = rows[rows["home_won"].notna()]
        if args.rows:
            rows.to_parquet(args.rows)
    grid_cols = grid.nfl_game_picks.DISAGGREGATED_FEATURE_COLUMNS
    grid.nfl_game_picks.DISAGGREGATED_FEATURE_COLUMNS = nfl_game_picks.QB_PLUS_FEATURE_COLUMNS
    try:
        win_effect(rows, missing, grid._load("schedules"))
    finally:
        grid.nfl_game_picks.DISAGGREGATED_FEATURE_COLUMNS = grid_cols


if __name__ == "__main__":
    main()

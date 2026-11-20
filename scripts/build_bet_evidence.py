"""How strong is the evidence behind each bet on the site? Writes
docs/data/bet_evidence.json for the NFL Picks tab's track record.

Each series of bets, in the order made, gets the anytime-valid numbers
from src/mlb_metrics/anytime_valid.py: the evidence bankroll (1 unit bet
against "no edge"; 20 is the usual bar), the range of win rates not ruled
out, and the fade alarm. Read that module's notes first.

Two kinds of series:
  replay   every bet a rule would have made over past seasons
           (data/backtest/nfl_*_replay.csv, rebuilt with --replay). The
           rule was chosen with these results in view, so their evidence
           is an upper bound.
  live     bets logged on the site before the games (docs/data/*_picks.csv).
           The clean test; it grows by a few bets a week.

RESULT (2026-11-20):
  Spread bets, replay 2002-2026: 849 graded, 57.4%, +80.7 units. The
    bankroll is 20.6 - at the bar, not far past it - after peaking at 64.8
    in November 2021, and it sat BELOW 1 until 2014 (2002-13 barely beat
    the vig). Win rates not ruled out: 52.4% to 61.8%; break-even is 52.4%.
    Fade alarm against break-even: 63 of the 1,000 that trips it.
  Over/under leans, replay 2019-2025 at the real prices: 174 graded,
    58.6%, +21.9 units; bankroll 2.75 (peak 3.0), win rates not ruled out
    47.1% to 69.2%. Thinner than "58.6%" sounds: 174 bets cannot carry
    more, and break-even is well inside the range.
  Our winner in every game at the closing moneyline, 2006-2026: 5,445
    bets, -88.7 units, bankroll 0.016. No edge, as the site says: picks,
    not bets.
  Live series are all too short to say anything yet (bankrolls 1.3).
  MLB moneyline bets (stopped): 124 bets, -42.3 units, bankroll 0.09.

Usage:
  python scripts/build_bet_evidence.py                      # the JSON, from what is on disk
  python scripts/build_bet_evidence.py --replay --cache DIR # also rebuild the replay files
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlb_metrics import anytime_valid as av, config  # noqa: E402

BREAK_EVEN = 110.0 / 210.0
REPLAY_DIR = os.path.join("data", "backtest")
REPLAY_FILES = {
    "spread": "nfl_spread_bets_replay.csv", "total": "nfl_total_leans_replay.csv", "winner": "nfl_winners_moneyline_replay.csv",
}


def payout(price) -> np.ndarray:
    """Profit per unit staked on a win at an American price (-110 if missing)."""
    price = pd.to_numeric(pd.Series(price), errors="coerce").fillna(-110.0).to_numpy()
    return np.where(price > 0, price / 100.0, 100.0 / np.abs(price))


def rebuild_replays(cache: str, out_dir: str) -> None:
    """Every bet each rule would have made, one row each, oldest first."""
    import statsmodels.api as sm

    import backtest_nfl_totals as totals_replay
    import research_nfl_spread_edge as spread_replay
    from mlb_metrics import nfl_totals

    os.makedirs(out_dir, exist_ok=True)
    games, weekly = spread_replay.load(cache)
    d = spread_replay.replay(games, weekly).sort_values(["gameday", "game_id"])
    home = d["gap"] > 0
    bets = d[d["gap"].abs() >= config.NFL_SPREAD_BET_MIN_GAP]
    side = home.loc[bets.index]
    pd.DataFrame({
        "season": bets["season"], "week": bets["week"], "date": bets["gameday"], "game_id": bets["game_id"],
        "team": np.where(side, bets["home_team"], bets["away_team"]),
        "line": np.where(side, -bets["spread_line"], bets["spread_line"]), "price": -110,
        "won": np.where(bets["cover"] == 0, np.nan, (side == (bets["cover"] > 0)).astype(float)),
    }).to_csv(os.path.join(out_dir, REPLAY_FILES["spread"]), index=False)

    priced = d[d["home_moneyline"].notna() & d["away_moneyline"].notna() & (d["our_margin"] != 0)]
    pick_home = priced["our_margin"] > 0
    margin = priced["home_score"] - priced["away_score"]
    pd.DataFrame({
        "season": priced["season"], "week": priced["week"], "date": priced["gameday"], "game_id": priced["game_id"],
        "team": np.where(pick_home, priced["home_team"], priced["away_team"]),
        "price": np.where(pick_home, priced["home_moneyline"], priced["away_moneyline"]),
        "won": np.where(margin == 0, np.nan, (pick_home == (margin > 0)).astype(float)),
    }).to_csv(os.path.join(out_dir, REPLAY_FILES["winner"]), index=False)

    team_games, qb_games, schedules = totals_replay.load()
    columns = nfl_totals.TOTALS_FEATURE_COLUMNS
    features = totals_replay.weekly_features(team_games, qb_games, schedules).dropna(subset=columns + ["points"])
    scored = []
    for season in totals_replay.SCORED_SEASONS:
        model = totals_replay.fit(features[features["season"] < season])
        test = features[features["season"] == season].copy()
        # Rain unknown, as when the lean is posted (scripts/research_nfl_bets.py).
        test["ours"] = model.predict(sm.add_constant(test.assign(wx_rain=0.0)[columns], has_constant="add"))
        scored.append(test)
    rows = pd.concat(scored)
    t = rows.groupby("game_id").agg(ours=("ours", "sum"), total=("points", "sum"), sides=("points", "size")).reset_index()
    t = t[t["sides"] == 2].merge(
        schedules[["game_id", "season", "week", "gameday", "total_line", "over_odds", "under_odds"]], on="game_id")
    t = t[t["total_line"].notna() & (t["season"] >= 2019)]
    t = t[(t["ours"] - t["total_line"]).abs() >= config.NFL_TOTAL_LEAN_MIN_GAP].sort_values(["gameday", "game_id"])
    over = t["ours"] > t["total_line"]
    miss = t["total"] - t["total_line"]
    pd.DataFrame({
        "season": t["season"], "week": t["week"], "date": t["gameday"], "game_id": t["game_id"],
        "team": np.where(over, "over", "under"), "line": t["total_line"],
        "price": np.where(over, t["over_odds"], t["under_odds"]),
        "won": np.where(miss == 0, np.nan, (over == (miss > 0)).astype(float)),
    }).to_csv(os.path.join(out_dir, REPLAY_FILES["total"]), index=False)


def series(key, label, kind, note, frame, one_price=True) -> dict | None:
    """One series' numbers. frame: date, won (1/0/NaN) and either `price`
    or `ret` (profit per unit staked), oldest first. one_price: the bets
    all pay about the same (-110), so a win rate means something."""
    if frame is None or frame.empty:
        return None
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    won = frame["won"].to_numpy(dtype=float)
    ret = frame["ret"].to_numpy(dtype=float) if "ret" in frame.columns else np.where(
        np.isnan(won), 0.0, np.where(won > 0, payout(frame["price"]), -1.0))
    path = av.wealth_path(ret)
    numbers = av.evidence(ret)
    graded = won[~np.isnan(won)]
    out = {
        "key": key, "label": label, "kind": kind, "note": note,
        "bets": int(len(frame)), "won": int((graded > 0).sum()), "lost": int((graded == 0).sum()),
        "win_rate": round(float(graded.mean()), 4) if graded.size else None,
        "units": round(float(frame["profit"].sum()) if "profit" in frame.columns else numbers["units"], 2), "wealth": round(numbers["wealth"], 3), "peak": round(numbers["peak"], 3),
        "peak_date": str(frame["date"].iloc[numbers["peak_at"]]) if numbers["peak_at"] is not None else None,
        "first": str(frame["date"].iloc[0]), "last": str(frame["date"].iloc[-1]),
    }
    if one_price and graded.size:
        rates = av.confidence_sequence(won)
        fade = av.fade_detector(won, BREAK_EVEN)
        dated = frame.loc[~np.isnan(won), "date"].reset_index(drop=True)
        out.update({
            "rate_low": round(rates["low"], 3), "rate_high": round(rates["high"], 3),
            "fade": round(fade["value"], 1), "fade_peak": round(fade["peak"], 1),
            "fade_tripped": str(dated.iloc[fade["tripped_at"]]) if fade["tripped_at"] is not None else None,
        })
    # The bankroll at the end of each season (a replay) or after each bet.
    if "season" in frame.columns and frame["season"].nunique() > 2:
        last = frame.groupby("season").tail(1)
        out["path"] = [{"at": str(int(s)), "wealth": round(float(path[i]), 3)} for i, s in zip(last.index, last["season"])]
    else:
        out["path"] = [{"at": str(d), "wealth": round(float(w), 3)} for d, w in zip(frame["date"], path)]
    return out


def live_frames(data_dir: str) -> dict:
    """The bets logged on the site, graded ones only."""
    out = {}
    nfl_path = os.path.join(data_dir, "nfl_game_picks_picks.csv")
    if os.path.exists(nfl_path):
        nfl = pd.read_csv(nfl_path)
        result = {"win": 1.0, "loss": 0.0, "push": np.nan}
        for key, flag, graded in (("spread", "spread_bet", "spread_bet_result"), ("total", "total_lean", "total_lean_result")):
            if flag in nfl.columns and graded in nfl.columns:
                rows = nfl[nfl[flag].notna() & nfl[graded].isin(list(result))]
                out[key] = pd.DataFrame({"date": rows["date"], "won": rows[graded].map(result), "price": -110})
        out["nfl_moneyline"] = _staked(nfl)
    mlb_path = os.path.join(data_dir, "game_picks_picks.csv")
    if os.path.exists(mlb_path):
        out["mlb_moneyline"] = _staked(pd.read_csv(mlb_path))
    return out


def _staked(picks: pd.DataFrame) -> pd.DataFrame:
    """Moneyline bets of varying size: the return per unit staked."""
    if "bet_units" not in picks.columns:
        return pd.DataFrame()
    rows = picks[(pd.to_numeric(picks["bet_units"], errors="coerce") > 0) & picks["bet_profit_units"].notna()
                 & picks["status"].isin(["win", "loss"])]
    ret = (rows["bet_profit_units"] / rows["bet_units"]).clip(lower=-1.0)
    return pd.DataFrame({"date": rows["date"], "won": (ret > 0).astype(float), "ret": ret, "profit": rows["bet_profit_units"]})


def build(data_dir: str, replay_dir: str) -> dict:
    def replay(name):
        path = os.path.join(replay_dir, REPLAY_FILES[name])
        return pd.read_csv(path) if os.path.exists(path) else None

    live = live_frames(data_dir)
    chosen = "The rule was chosen with these results in view, so this is the most the evidence could be."
    clean = "Posted on the site before the games: the clean test. It grows by a few bets a week."
    rows = [
        series("nfl_spread_replay", "Spread bets, every one since 2002", "replay", chosen, replay("spread")),
        series("nfl_spread_live", "Spread bets posted here", "live", clean, live.get("spread")),
        series("nfl_total_replay", "Over/under leans, every one 2019-2025", "replay", chosen, replay("total")),
        series("nfl_total_live", "Over/under leans posted here", "live", clean, live.get("total")),
        series("nfl_winner_replay", "Our winner in every game, at the closing moneyline since 2006", "replay",
               "Why our winners are shown as picks, not bets.", replay("winner"), one_price=False),
        series("nfl_moneyline_live", "Moneyline bets posted here", "live", clean, live.get("nfl_moneyline"), one_price=False),
        series("mlb_moneyline_live", "Baseball moneyline bets posted here (stopped)", "live",
               "Why we stopped suggesting them.", live.get("mlb_moneyline"), one_price=False),
    ]
    return {
        "bar": av.EVIDENCE_BAR, "confidence": av.CONFIDENCE, "alarm": av.ALARM_LEVEL, "break_even": round(BREAK_EVEN, 4),
        "series": [r for r in rows if r],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="docs/data")
    parser.add_argument("--replay-dir", default=REPLAY_DIR)
    parser.add_argument("--replay", action="store_true", help="Rebuild the replay files first (needs --cache).")
    parser.add_argument("--cache", help="Folder for the downloaded games file the spread replay uses.")
    args = parser.parse_args()
    if args.replay:
        if not args.cache:
            parser.error("--replay needs --cache")
        rebuild_replays(args.cache, args.replay_dir)
    out = build(args.data_dir, args.replay_dir)
    with open(os.path.join(args.data_dir, "bet_evidence.json"), "w") as f:
        json.dump(out, f, separators=(",", ":"))
    for s in out["series"]:
        rate = f", {s['win_rate']:.1%}" if s.get("win_rate") is not None else ""
        kept = f", rates not ruled out {s['rate_low']:.1%}-{s['rate_high']:.1%}, fade {s['fade']}" if "rate_low" in s else ""
        print(f"{s['label']}: {s['bets']} bets{rate}, {s['units']:+.1f} units, bankroll {s['wealth']} (peak {s['peak']}){kept}")


if __name__ == "__main__":
    main()

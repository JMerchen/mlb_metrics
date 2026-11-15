"""Do the college basketball ratings (cbb_ratings.py) beat a betting line?
The first test of the college model against sportsbook spreads and totals
rather than results.

Every game is predicted from ratings fit on the games before its date (a
fresh fit each day, last season's final ratings as the prior - what the
daily job does), then set beside the line ESPN recorded for it
(scripts/backfill_cbb_odds.py -> data/raw/market_odds/espn_cbb_odds_*):
the closing spread and total, and the opening ones.

Printed: how far each of us missed, then our side of the spread and of the
total by how far our number sits from the line - against the closing line
(the hard test) and the opening one (what a bettor early in the day gets)
- by part of the season and by how small the schools are, and whether the
line moved toward us after it opened. 52.4% breaks even at -110.

RESULT (2026-11-15; 22,940 games 2023-2026 with a DraftKings closing line,
16,695 of them 2024-2026 with a usable opening one):
  The line is closer than we are: margins off by 8.87 against our 9.08;
  totals 13.04 against 13.43.
  Our side of the CLOSING spread: 49.2% of 8,897 at 2+ points apart, 49.1%
  of 2,500 at 4+, 50.9% of 743 at 6+, 50.6% of 249 at 8+. No season, part
  of the season or size of school is different (small schools 49.5%).
  Our side of the OPENING spread: 50.2% at 2+, 48.5% of 1,467 at 4+, 50.3%
  of 384 at 6+. The line then moved toward our number 41% of the time and
  away 30%, by a fifth of a point on average - not enough to bet.
  Totals: against the close our side LOSES (48.3% of 4,602 at 5+, 47.5% of
  3,082 at 6+; fading it would only break even); against the opener 51.5%
  of 2,982 at 5+, short of 52.4%, with the line moving half a point our
  way.
  So: no college bet. These are picks, and the page says so.
  THE TRAP IN THIS DATA: ESPN records some opening spreads with the wrong
  sign. Before those were dropped (with_lines), our side of the "opening"
  spread at 6+ apart read 55.9% of 508 and at 8+ 61.7% of 193 - every bit
  of it from 124 games whose line "moved" 4+ points (73% winners); the
  other 384 won 50.3%.

Usage:
    PYTHONPATH=src python3 scripts/research_cbb_vs_lines.py --cache DIR [--seasons 2023 2024 2025 2026]
(DIR holds the hoopR season files, downloaded as needed, and the day-by-day
predictions once made, so a second run is quick.)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mlb_metrics import cbb_data, cbb_odds, cbb_ratings, config  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
WIN = 100 / 110
MAX_SPREAD_MOVE, MAX_TOTAL_MOVE = 4.0, 6.0   # an opener further than this from the close is treated as an error


def daily_predictions(cache: str, seasons: list) -> pd.DataFrame:
    """One row per game between two rated Division I teams in `seasons`:
    our margin and total from the ratings as of that morning, the result,
    and each team's rank by net rating that day."""
    path = os.path.join(cache, f"cbb_daily_predictions_{min(seasons)}_{max(seasons)}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    out, prior = [], None
    for season in range(config.CBB_FIRST_SEASON, max(seasons) + 1):
        schedules = cbb_data.load_table("schedules", season, cache)
        games = cbb_ratings.prepare_games(schedules, cbb_data.load_table("team_box", season, cache))
        ranked = cbb_ratings.preseason_ranked(schedules)
        if season in seasons and prior is not None:
            d1 = games[games["home_d1"] & games["away_d1"]]
            for day, today in d1.groupby(d1["date"].dt.normalize()):
                ratings, league = cbb_ratings.fit_ratings(games, prior=prior, as_of=day, ranked=ranked)
                r = ratings.set_index("team_id")
                rank = r["net"].rank(ascending=False)
                known = today[today["home_id"].isin(r.index) & today["away_id"].isin(r.index)]
                if known.empty:
                    continue
                p = cbb_ratings.predict_games(r["net"].reindex(known["home_id"]), r["net"].reindex(known["away_id"]),
                                              r["tempo"].reindex(known["home_id"]), r["tempo"].reindex(known["away_id"]),
                                              known["neutral"], league)
                possessions = p["possessions"].to_numpy()
                # Both teams' points: the home edge cancels in the sum.
                total = possessions / 100 * (2 * league["efficiency"]
                                             + r["offense"].reindex(known["home_id"]).to_numpy() - r["defense"].reindex(known["away_id"]).to_numpy()
                                             + r["offense"].reindex(known["away_id"]).to_numpy() - r["defense"].reindex(known["home_id"]).to_numpy())
                out.append(pd.DataFrame({
                    "game_id": known["game_id"].astype("int64").to_numpy(), "season": season, "date": known["date"].to_numpy(),
                    "neutral": known["neutral"].to_numpy(), "tournament": known["tournament"].to_numpy(),
                    "our_margin": p["margin"].to_numpy(), "our_total": total,
                    "margin": (known["home_score"] - known["away_score"]).to_numpy(),
                    "points": (known["home_score"] + known["away_score"]).to_numpy(),
                    "home_rank": rank.reindex(known["home_id"]).to_numpy(), "away_rank": rank.reindex(known["away_id"]).to_numpy(),
                    "home_games": r["games"].reindex(known["home_id"]).to_numpy(), "away_games": r["games"].reindex(known["away_id"]).to_numpy(),
                }))
            print(f"  {season}: {sum(len(f) for f in out if f['season'].iloc[0] == season)} games predicted", flush=True)
        prior, _ = cbb_ratings.fit_ratings(games, prior=prior, ranked=ranked)
    rows = pd.concat(out, ignore_index=True)
    rows.to_parquet(path, index=False)
    return rows


def with_lines(predictions: pd.DataFrame, seasons: list, raw_dir: str) -> pd.DataFrame:
    """The predictions with each game's consensus lines: market_margin and
    market_total at the close, and open_margin / open_total (home minus
    away; ESPN's spread is the home team's, negative = favored)."""
    frames = []
    for season in seasons:
        path = os.path.join(raw_dir, "market_odds", f"espn_cbb_odds_{season}.parquet")
        if os.path.exists(path):
            frames.append(cbb_odds.consensus_lines(pd.read_parquet(path)))
    if not frames:
        return predictions.iloc[0:0]
    lines = pd.concat(frames, ignore_index=True)
    lines["game_id"] = lines["game_id"].astype("int64")
    d = predictions.merge(lines, on="game_id", how="inner")
    d["market_margin"], d["open_margin"] = -d["spread"], -d["spread_open"]
    d["market_total"], d["open_total"] = d["total"], d["total_open"]
    # ESPN's opening numbers are sometimes wrong - most often the spread
    # with its sign flipped (opened -21.5, closed +21.5). A real line all
    # but never moves this far (99% of spreads move 5 points or less), and
    # a wrong opener looks like a huge edge for anyone near the truth: left
    # in, our side of the "opening" spread at 6+ points apart read 55.9% of
    # 508, all of it from 124 such games. Those openers are dropped.
    d.loc[(d["market_margin"] - d["open_margin"]).abs() >= MAX_SPREAD_MOVE, "open_margin"] = np.nan
    d.loc[(d["market_total"] - d["open_total"]).abs() >= MAX_TOTAL_MOVE, "open_total"] = np.nan
    return d


def record(label, d, ours: str, line: str, actual: str, low: float, width: int = 46) -> None:
    """Prints how our side of `line` did where our number is `low`+ from it."""
    d = d[d[ours].notna() & d[line].notna()]
    gap, result = d[ours] - d[line], d[actual] - d[line]
    use = (gap.abs() >= low) & (result != 0)
    won = ((gap > 0) == (result > 0))[use]
    n = int(use.sum())
    if not n:
        print(f"  {label:{width}s} none")
        return
    units = np.where(won, WIN, -1.0)
    seasons = pd.Series(units, index=won.index).groupby(d.loc[use, "season"]).sum()
    print(f"  {label:{width}s} {n:6d} bets, won {won.mean():.1%}, {units.sum():+8.1f} units, "
          f"winning seasons {int((seasons > 0).sum())}/{len(seasons)}, z {(won.mean() - 0.5) / np.sqrt(0.25 / n):+.2f}")


def beyond_line(d, ours: str, line: str, actual: str) -> str:
    import statsmodels.api as sm
    d = d[d[ours].notna() & d[line].notna()]
    fit = sm.OLS(d[actual], sm.add_constant(pd.DataFrame({"line": d[line], "ours": d[ours]}))).fit(cov_type="HC1")
    return f"line {fit.params['line']:.3f}, ours {fit.params['ours']:+.3f} (t {fit.tvalues['ours']:+.2f})"


def report(d: pd.DataFrame, kind: str) -> None:
    ours, actual = ("our_margin", "margin") if kind == "spread" else ("our_total", "points")
    close, opening = ("market_margin", "open_margin") if kind == "spread" else ("market_total", "open_total")
    have = d[d[close].notna()]
    print(f"\n==== The {kind}: {len(have)} games with a closing line, {int(d[opening].notna().sum())} with an opening one")
    print(f"  average miss: ours {(have[actual] - have[ours]).abs().mean():.2f}, the closing line {(have[actual] - have[close]).abs().mean():.2f}")
    print(f"  the result regressed on the closing line and ours: {beyond_line(d, ours, close, actual)}")
    print(f"  ... on the opening line and ours: {beyond_line(d, ours, opening, actual)}")
    gaps = (2, 3, 4, 5, 6, 8) if kind == "spread" else (3, 4, 5, 6, 8, 10)
    for name, line in (("closing", close), ("opening", opening)):
        print(f" Our side of the {name} line, by how far ours is from it")
        for low in gaps:
            record(f"{low}+ points apart", d, ours, line, actual, low)
    mid = gaps[2]
    month = d["date"].dt.month
    print(f" At {mid}+ points from the closing line, by when and who")
    for label, part in (("November-December", d[month.isin([11, 12])]), ("January-February", d[month.isin([1, 2])]),
                        ("March-April", d[month.isin([3, 4])]),
                        ("both teams with 8+ games played", d[(d["home_games"] >= 8) & (d["away_games"] >= 8)]),
                        ("both teams ranked outside our top 150", d[(d["home_rank"] > 150) & (d["away_rank"] > 150)]),
                        ("a team inside our top 50", d[(d["home_rank"] <= 50) | (d["away_rank"] <= 50)])):
        record(label, part, ours, close, actual, mid)
    print(f" The same at {mid}+ from the opening line")
    for label, part in (("November-December", d[month.isin([11, 12])]), ("January-February", d[month.isin([1, 2])]),
                        ("March-April", d[month.isin([3, 4])]),
                        ("both teams ranked outside our top 150", d[(d["home_rank"] > 150) & (d["away_rank"] > 150)])):
        record(label, part, ours, opening, actual, mid)
    both = d[d[close].notna() & d[opening].notna()]
    gap = both[ours] - both[opening]
    moved = np.sign(gap) * (both[close] - both[opening])
    print(" After the line opened, did it move toward our number?")
    for low in gaps[:5]:
        use = gap.abs() >= low
        print(f"  ours {low}+ from the opener: {int(use.sum()):6d} games; moved our way {np.mean(moved[use] > 0):.0%}, "
              f"against {np.mean(moved[use] < 0):.0%}, average {moved[use].mean():+.2f} points")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--seasons", type=int, nargs="+", default=[2023, 2024, 2025, 2026])
    parser.add_argument("--raw-dir", default=os.path.join(REPO, "data", "raw"))
    args = parser.parse_args()
    os.makedirs(args.cache, exist_ok=True)
    predictions = daily_predictions(args.cache, sorted(args.seasons))
    d = with_lines(predictions, args.seasons, args.raw_dir)
    print(f"{len(predictions)} games predicted in {sorted(args.seasons)}; {len(d)} of them have a line "
          f"({d.groupby('season').size().to_dict()})")
    if d.empty:
        return
    report(d, "spread")
    report(d, "total")


if __name__ == "__main__":
    main()

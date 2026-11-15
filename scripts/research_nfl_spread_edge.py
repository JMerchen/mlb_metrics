"""Does the NFL win model (v5, src/mlb_metrics/nfl_qb_elo.py) beat the
closing point spread? Tested on 25 seasons instead of the 8 the model was
built on, with the controls a skeptic would ask for.

Why (2026-11-15): v5 was built to pick winners and its settings were
chosen on win-probability log loss over 2019-2026. Against the closing
spread on those 8 seasons its bigger disagreements looked good but proved
nothing (54.8% of 250, t 1.5). The rating needs only scores, quarterback
box scores and past closing lines, all public back to 1999 - so 2002-2018
is 17 more seasons it has never seen, a clean test.

Our margin is the rating gap in points (config.NFL_QB_ELO_POINTS Elo
points to a point, the same scale the rating uses to read a closing
line); the "gap" is our margin minus the closing spread. Every game is
predicted from ratings as they stood before it, the game's own line is
never used, and the starters are the ones who started (which the closing
line also knew).

RESULT (2026-11-15; 6,636 games 2002-2026, regular season and playoffs):
  Taking our side of the closing spread at -110 (52.4% breaks even):
    gap 1+      4,309 bets  52.3%    -4.0 units
    gap 2+      2,564 bets  52.6%   +11.4
    gap 3+      1,293 bets  53.6%   +30.0
    gap 3.5+      885 bets  54.6%   +37.1   15 of 25 seasons   <- the rule
    gap 4+        591 bets  55.2%   +31.4
    gap 5+        289 bets  54.3%   +10.7
  By gap band: 0-1 49.8%, 1-2 51.9%, 2-3 51.6%, 3-4 52.3%, 4-5 56.0%,
  5+ 54.3%. Every game: 51.5% of 6,464.
  By era at 3.5+: 2002-10 51.4% of 370; 2011-18 56.7% of 275 (never seen
  when the settings were chosen); 2019-26 57.1% of 240. But the last four
  seasons, 2023-26, are 44-41.
  Real margin regressed on the closing line and our margin together: ours
  +0.315 (t 4.1); with a flexible curve in the line, t 3.8. In every era:
  t 2.0, 3.8, 1.6. So the rating knows something the closing line doesn't.

  Is it only "take the underdog"? No. 90% of the 3.5+ bets are underdogs,
  but every underdog blindly covered 51.0% of 6,445 and lost 164 units.
  Within underdogs our gap sorts them: we like the favorite 49.2% of
  1,832; like the dog by 0-1.5 50.9%; 1.5-3.5 51.3%; 3.5+ 55.0% of 795.
  With our margins stretched to be as wide as the line's (no built-in
  lean): 54.5% of 861.
  Is it quarterback news the line was slow on? No, the reverse: games
  where both teams start the quarterback who started their last game,
  55.6% of 567 (+34.4 units, 16 of 25 seasons; ours beyond the line t 3.8;
  by era 51.4% of 245, 58.5% of 195, 59.1% of 127; 2023-26 19-19); with a
  change, 52.8% of 318 (+2.7 units, t 1.4). A new starter is where a
  box-score rating knows least and the line most, so the rule skips those
  games.
  Is it one lucky setting? The rule's record moves with the settings
  (market pull 0 / 0.1 / 0.4 / 0.6: 53.6 / 53.9 / 53.3 / 52.7%; K 8 / 20:
  54.1 / 52.2%; quarterback weight 0 / 2 / 5: 52.2 / 55.0 / 53.8%), but
  the regression's t stays between 2.9 and 4.3 in all of them. Read the
  rule as "about 53-55%", not as 54.6% to the decimal.
  Moneyline instead of the points (2006 on, 688 of the bets): +6.8% a bet
  against the spread, -0.3% on the moneyline overall - but when our side
  was a small underdog (priced 33-45% to win, about +120 to +200) the
  moneyline made +20% a bet on 163 against +12% with the points, and when
  it was a big underdog (under 20%) the moneyline lost 34% a bet.
  That fits how moneylines are priced in general (5,202 games 2006-26, a
  unit on every side): underdogs priced 33-45% +1.3% a bet over 2,070
  bets, every other band loses, and the longest shots (under 20%) lose
  16.7% a bet.

WHAT THE SITE DOES WITH IT: a spread bet when our margin is
config.NFL_SPREAD_BET_MIN_GAP or more from the line and neither team has
changed quarterbacks since its last game (nfl_game_evaluation.spread_sides,
nfl_game_picks.quarterback_changes), with that record beside it.

Also here, --wind: unders in windy outdoor games against the closing
total, 1999-2026 (kickoff wind from the gamebook, so the bettor has a
forecast, not this exact number):
    10+ mph  1,882 games  54.9% under  +92.0 units  18 of 28 seasons
    13+ mph  1,003 games  56.1%        +71.8
    15+ mph    674 games  57.0%        +59.1
    under 10 3,283 games  48.8%
  10+ mph by era: 1999-2012 52.9%, 2013-19 57.5%, 2020-26 57.7%.

Usage:
    PYTHONPATH=src python3 scripts/research_nfl_spread_edge.py --cache DIR [--settings] [--wind]
(DIR gets nflverse's games.csv and the 1999-2015 weekly player stats; the
later seasons are read from data/raw/nfl.)
"""

import argparse
import glob
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from mlb_metrics import config, nfl_qb_elo  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
WEEKLY_URL = "https://github.com/nflverse/nflverse-data/releases/download/stats_player/stats_player_week_{season}.parquet"
FIRST_SCORED = 2002   # three seasons for the ratings to settle
WIN, ERAS = 100 / 110, ((2002, 2010), (2011, 2018), (2019, 2026))


def load(cache: str):
    """(games, weekly): every finished game since 1999 with its closing
    lines, and every player-week of stats."""
    os.makedirs(cache, exist_ok=True)
    games_path = os.path.join(cache, "games.csv")
    if not os.path.exists(games_path):
        urllib.request.urlretrieve(GAMES_URL, games_path)
    games = pd.read_csv(games_path)
    games = games[games["home_score"].notna()].copy()
    on_disk = sorted(glob.glob(os.path.join(REPO, config.NFL_RAW_DATA_DIR, "weekly_20*.parquet")))
    first_local = min(int(os.path.basename(f)[7:11]) for f in on_disk)
    frames = []
    for season in range(1999, first_local):
        path = os.path.join(cache, f"weekly_{season}.parquet")
        if not os.path.exists(path):
            urllib.request.urlretrieve(WEEKLY_URL.format(season=season), path)
        frames.append(pd.read_parquet(path))
    frames += [pd.read_parquet(f) for f in on_disk]
    weekly = pd.concat(frames, ignore_index=True).drop_duplicates(["season", "week", "team", "player_id"])
    return games, weekly


def replay(games, weekly, **settings) -> pd.DataFrame:
    """The games from FIRST_SCORED on with a closing spread, with v5's
    margin for the home team before each (our_margin), its gap to the
    spread (gap) and the result against the spread (cover: home margin
    minus the spread). `settings` overrides config values for the run."""
    saved = {name: getattr(config, name) for name in settings}
    for name, value in settings.items():
        setattr(config, name, value)
    try:
        _, gaps = nfl_qb_elo.rate(games, weekly, record=True)
    finally:
        for name, value in saved.items():
            setattr(config, name, value)
    d = games.merge(gaps, on="game_id")
    d = d[(d["season"] >= FIRST_SCORED) & d["spread_line"].notna()].copy()
    d["margin"] = d["home_score"] - d["away_score"]
    d["our_margin"] = d["qb_elo_gap"] / config.NFL_QB_ELO_POINTS
    d["gap"] = d["our_margin"] - d["spread_line"]      # spread_line: the home margin the line expects
    d["cover"] = d["margin"] - d["spread_line"]
    return d


def record(label, d, side=None, width=44) -> None:
    """Prints how betting `side` (positive = the home team; default our
    gap's side) against the spread did over the games in `d`."""
    d = d[d["cover"] != 0]
    side = d["gap"] if side is None else side.loc[d.index]
    won = (side > 0) == (d["cover"] > 0)
    units = np.where(won, WIN, -1.0)
    seasons = pd.Series(units, index=d.index).groupby(d["season"]).sum()
    n = len(d)
    print(f"  {label:{width}s} {n:5d} bets, won {won.mean():.1%}, {units.sum():+7.1f} units, "
          f"winning seasons {int((seasons > 0).sum())}/{len(seasons)}, z {(won.mean() - 0.5) / np.sqrt(0.25 / max(n, 1)):+.2f}")


def t_beyond_line(d, flexible=False) -> tuple:
    """(coefficient, t) of our margin when the real margin is regressed on
    the closing line and our margin together."""
    import statsmodels.api as sm
    X = pd.DataFrame({"line": d["spread_line"], "ours": d["our_margin"]})
    if flexible:
        X["line_cubed"] = d["spread_line"] ** 3 / 100
        X["line_root"] = np.sign(d["spread_line"]) * np.sqrt(d["spread_line"].abs())
    fit = sm.OLS(d["margin"], sm.add_constant(X)).fit(cov_type="HC1")
    return fit.params["ours"], fit.tvalues["ours"]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--settings", action="store_true", help="Also rerun under neighbouring settings.")
    parser.add_argument("--wind", action="store_true", help="Also print windy-game unders.")
    args = parser.parse_args()
    games, weekly = load(args.cache)
    d = replay(games, weekly)
    rule = config.NFL_SPREAD_BET_MIN_GAP
    print(f"{len(d)} games, {int(d['season'].min())}-{int(d['season'].max())}")

    print("\nOur side of the closing spread, by how far our margin is from it")
    record("every game", d)
    for low in (1, 2, 3, 3.5, 4, 5):
        record(f"gap {low}+", d[d["gap"].abs() >= low])
    for low, high in ((0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 99)):
        record(f"gap {low} to {high}", d[(d["gap"].abs() >= low) & (d["gap"].abs() < high)])
    bets = d[d["gap"].abs() >= rule]
    for first, last in ERAS + ((2023, 2026),):
        record(f"gap {rule}+ in {first}-{last}", bets[(bets["season"] >= first) & (bets["season"] <= last)])
    won = ((bets["gap"] > 0) == (bets["cover"] > 0))[bets["cover"] != 0]
    by = won.groupby(bets["season"]).agg(["sum", "count"])
    print("  by season: " + "  ".join(f"{int(s)}: {int(r['sum'])}-{int(r['count'] - r['sum'])}" for s, r in by.iterrows()))

    print("\nDoes our margin add to the closing line? (real margin regressed on both)")
    print("  all seasons: ours %+.3f (t %.2f); with a flexible curve in the line: %+.3f (t %.2f)"
          % (*t_beyond_line(d), *t_beyond_line(d, flexible=True)))
    for first, last in ERAS:
        print(f"  {first}-{last}: ours %+.3f (t %.2f)" % t_beyond_line(d[(d["season"] >= first) & (d["season"] <= last)]))

    print("\nControl: is it only the underdog?")
    dog = -np.sign(d["spread_line"])
    lined = d[d["spread_line"].abs() >= 0.5]
    record("every underdog, blind", lined, dog)
    likes_dog = lined["gap"] * dog.loc[lined.index]
    for low, high in ((-99, 0), (0, 1.5), (1.5, 3.5), (3.5, 99)):
        record(f"the underdog when we like it by {low} to {high}", lined[(likes_dog >= low) & (likes_dog < high)], dog)
    wide = d.assign(gap=d["our_margin"] * d["spread_line"].std() / d["our_margin"].std() - d["spread_line"])
    record(f"our margins as wide as the line's, gap {rule}+", wide[wide["gap"].abs() >= rule])

    print("\nControl: is it quarterback news?")
    last_starter, changed = {}, []
    for row in d.sort_values(["season", "week", "gameday", "game_id"]).itertuples():
        change = False
        for team, starter in ((row.home_team, row.home_qb_id), (row.away_team, row.away_qb_id)):
            change = change or (team in last_starter and last_starter[team] != starter)
            last_starter[team] = starter
        changed.append((row.game_id, change))
    changed = d["game_id"].map(dict(changed))
    same = bets[~changed.loc[bets.index]]
    record(f"gap {rule}+, same starters as last game", same)
    record(f"gap {rule}+, a quarterback change", bets[changed.loc[bets.index]])
    for first, last in ERAS + ((2023, 2026),):
        record(f"  same starters, {first}-{last}", same[(same["season"] >= first) & (same["season"] <= last)])
    print("  ours beyond the line, same starters: %+.3f (t %.2f); a change: %+.3f (t %.2f)"
          % (*t_beyond_line(d[~changed]), *t_beyond_line(d[changed])))
    won = ((same["gap"] > 0) == (same["cover"] > 0))[same["cover"] != 0]
    by = won.groupby(same["season"]).agg(["sum", "count"])
    print("  same starters by season: " + "  ".join(f"{int(s)}: {int(r['sum'])}-{int(r['count'] - r['sum'])}" for s, r in by.iterrows()))
    for low in (3, 4, 5):
        other = d[(d["gap"].abs() >= low) & ~changed]
        record(f"  same starters, gap {low}+", other)

    print("\nThe same bets on the moneyline instead (seasons with closing moneylines)")
    priced = bets[(bets["cover"] != 0) & bets["home_moneyline"].notna() & (bets["margin"] != 0)]
    line = np.where(priced["gap"] > 0, priced["home_moneyline"], priced["away_moneyline"]).astype(float)
    pays = np.where(line > 0, line / 100, 100 / np.abs(line))
    straight = np.where((priced["gap"] > 0) == (priced["margin"] > 0), pays, -1.0)
    points = np.where((priced["gap"] > 0) == (priced["cover"] > 0), WIN, -1.0)
    implied = 1 / (1 + pays)
    print(f"  all {len(priced)}: the points {points.mean():+.3f} a bet, the moneyline {straight.mean():+.3f}")
    for low, high, label in ((0, 0.2, "under 20%"), (0.2, 0.33, "20-33%"), (0.33, 0.45, "33-45%"), (0.45, 1, "45%+")):
        use = (implied >= low) & (implied < high)
        print(f"  our side priced {label:10s} {use.sum():4d} bets: the points {points[use].mean():+.3f}, the moneyline {straight[use].mean():+.3f}")
    sides = pd.concat([
        pd.DataFrame({"line": games[f"{side}_moneyline"], "won": games["home_score"] * sign > games["away_score"] * sign,
                      "tie": games["home_score"] == games["away_score"]})
        for side, sign in (("home", 1), ("away", -1))])
    sides = sides[sides["line"].notna() & ~sides["tie"] & (games["game_type"].reindex(sides.index) == "REG")]
    pays = np.where(sides["line"] > 0, sides["line"] / 100, 100 / sides["line"].abs())
    implied, profit = 1 / (1 + pays), np.where(sides["won"], pays, -1.0)
    print("  a unit on every side at the closing moneyline, by its price:")
    for low, high, label in ((0, 0.2, "under 20%"), (0.2, 0.33, "20-33%"), (0.33, 0.45, "33-45%"), (0.45, 0.55, "near even"),
                             (0.55, 0.67, "55-67%"), (0.67, 0.8, "67-80%"), (0.8, 1.01, "80%+")):
        use = (implied >= low) & (implied < high)
        print(f"    {label:10s} {use.sum():5d} bets, {profit[use].mean():+.3f} a bet")

    if args.settings:
        print("\nNeighbouring settings")
        for label, settings in (
            ("market pull 0", {"NFL_QB_ELO_MARKET_PULL": 0.0}), ("market pull 0.1", {"NFL_QB_ELO_MARKET_PULL": 0.1}),
            ("market pull 0.4", {"NFL_QB_ELO_MARKET_PULL": 0.4}), ("market pull 0.6", {"NFL_QB_ELO_MARKET_PULL": 0.6}),
            ("K 8", {"NFL_QB_ELO_K": 8.0}), ("K 20", {"NFL_QB_ELO_K": 20.0}),
            ("quarterback weight 0", {"NFL_QB_ELO_QB_WEIGHT": 0.0}), ("quarterback weight 2", {"NFL_QB_ELO_QB_WEIGHT": 2.0}),
            ("quarterback weight 5", {"NFL_QB_ELO_QB_WEIGHT": 5.0}),
        ):
            other = replay(games, weekly, **settings)
            record(f"{label}, gap {rule}+", other[other["gap"].abs() >= rule])
            print("      ours beyond the line: %+.3f (t %.2f)" % t_beyond_line(other))

    if args.wind:
        print("\nUnders in windy outdoor games, against the closing total")
        out = games[games["total_line"].notna() & games["wind"].notna() & games["roof"].isin(["outdoors", "open"])].copy()
        out["points"] = out["home_score"] + out["away_score"]
        out = out[out["points"] != out["total_line"]]
        out["under"] = out["points"] < out["total_line"]
        for low in (10, 13, 15):
            for first, last in ((1999, 2012), (2013, 2019), (2020, 2026), (1999, 2026)):
                use = out[(out["wind"] >= low) & (out["season"] >= first) & (out["season"] <= last)]
                units = np.where(use["under"], WIN, -1.0)
                seasons = pd.Series(units, index=use.index).groupby(use["season"]).sum()
                print(f"  {low}+ mph, {first}-{last}: {len(use):5d} games, under {use['under'].mean():.1%}, {units.sum():+6.1f} units, "
                      f"winning seasons {int((seasons > 0).sum())}/{len(seasons)}")
        calm = out[out["wind"] < 10]
        print(f"  under 10 mph: {len(calm)} games, under {calm['under'].mean():.1%}")


if __name__ == "__main__":
    main()

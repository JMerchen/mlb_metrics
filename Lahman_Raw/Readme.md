# Lahman_Raw

Manually-refreshed source data for the Age Curves page
(`docs/age-curves.html`). Contains `People.csv`, `Batting.csv`, and
`Pitching.csv` from Lahman's Baseball Database, downloaded by hand from
SABR's mirror: <https://sabr.app.box.com/s/y1prhc795jk8zvmelfd3jq7tl389y6cd>

## Why a manually-committed CSV instead of an automated fetch

Two automated options were tried first and both failed:

- `pybaseball.lahman` downloads a zip from
  `https://github.com/chadwickbureau/baseballdatabank/archive/master.zip`
  at call time. That download started failing with `zipfile.BadZipFile` -
  reproduced in real GitHub Actions CI with full internet access, so it's
  an upstream break, not a network-restricted-environment issue.
- The `lahman` PyPI package needs no network fetch (data ships in the
  wheel), but it turned out to be a single abandoned release from 2022
  with data frozen at the 2020 season.

Downloading directly from SABR and committing the CSVs sidesteps both
problems, at the cost of needing a manual refresh instead of an automated
one.

## Refreshing (roughly once a season, after Lahman adds the completed year)

1. Open the SABR Box link above and download `People.csv`, `Batting.csv`,
   and `Pitching.csv` (or the full database zip and extract just those
   three).
2. Replace the files in this folder with the new versions.
3. Run `python scripts/fetch_lahman.py` to convert them to
   `data/raw/lahman/*.parquet` (what the rest of the codebase actually
   reads).
4. Run `python scripts/build_age_curves.py` to rebuild the Age Curves
   page's data, and `python scripts/backtest_age_curve.py` to re-validate
   the projection method and update the numbers in `config.py`/`README.md`
   if they changed meaningfully.

# mlb_metrics

Daily updated hitter/pitcher/team metrics (WAVE, PAVE, Game_Hit_Probability,
team Strength/Confidence) built from rolling windows of Statcast data, published
to `docs/` as a GitHub Pages dashboard.

## Project layout

- `src/mlb_metrics/` - the pipeline: `config.py` (window lengths and blend
  weights), `helpers.py` (event classifiers), `data.py` (Statcast fetch/raw
  persistence), `hitters.py`, `pitchers.py`, `teams.py` (metric computation),
  `pipeline.py` (orchestration + CLI), `predictions.py`/`evaluation.py`/
  `git_backtest.py` (hitter-pick backtesting - see below), `schedule.py`
  (MLB Stats API - probable pitchers, game schedules, final scores),
  `matchup.py` (batter-level opponent-pitching blend), `game_picks.py`/
  `game_predictions.py`/`game_evaluation.py` (Automated Game Picks - see
  below).
- `scripts/wave.py` - daily pipeline entrypoint (`python scripts/wave.py`).
- `scripts/run_backtest.py` - backtest entrypoint (`python scripts/run_backtest.py`).
- `data/raw/` - persisted raw Statcast pulls, one parquet file per season,
  committed daily alongside the output CSVs so history accumulates run over run.
- `data/predictions/predictions.csv` - append-only log of every daily hitter
  pick (date, player, predicted probability, realized actual_hit once
  known). `data/predictions/game_predictions.csv` is the analogous log for
  Automated Game Picks, keyed on `game_pk` instead of a player id.
- `docs/data/` - the published `wave.csv` / `pave.csv` / `confidence.csv`
  consumed by the `docs/` dashboard, plus `backtest_summary.csv` (see below).
  `confidence.csv` includes `Bullpen_PAVE_PLUS`
  (and `Bullpen_BAA`/`Bullpen_Power_A`/`Bullpen_HR_Per`/`Bullpen_AtBats`): PAVE
  computed only from each team's relief appearances, not its starters - a hitter
  typically sees the bullpen for 1-3 of their 3-5 at-bats in a game, so evaluating
  a matchup against "the starter's PAVE" alone ignores most of the at-bats.
- `tests/` - pytest suite covering the event classifiers, the window-blend
  formulas (WAVE, PAVE, Bullpen_PAVE, Game_Hit_Probability), and the
  backtesting pipeline, against hand-computed expected values and synthetic
  end-to-end scenarios.

## Backtesting

Every daily `scripts/wave.py` run now also logs that day's top picks to
`data/predictions/predictions.csv` (before the games happen) and resolves
any previous day's pending picks against the newly-fetched data - this is
what finally answers whether WAVE/Game_Hit_Probability actually predicts
hits, which nothing in the project did before.

`scripts/run_backtest.py` additionally replays the ~40+ days of `wave.csv`
already sitting in git history through the same pick-selection logic (see
`git_backtest.py`), so there's a backtestable dataset immediately rather than
waiting weeks for new predictions to accumulate. It resolves against
`data/raw/` and, with `--fetch-missing`, against Statcast directly for any
date not already persisted - then writes `docs/data/backtest_summary.csv`
(hit rate by pick rank, Brier score, log loss, calibration) via
`evaluation.py`. A `Backtest` GitHub Actions workflow (`workflow_dispatch`)
runs this with real network access and commits the results.

Every pick is qualified by `BACKTEST_MIN_PLATE_APPEARANCES` (config.py,
default 30) before ranking - without it, a batter with a handful of at-bats
and one lucky hit can show a probability of 1.0 and dominate the picks on
pure sample-size noise.

## Beat the Streak Tracker (dashboard)

The dashboard's top section simulates actually playing Beat the Streak,
following its real rules rather than a simplified per-day win/loss model
(see `evaluation.streak_progression()`):

- A batter is only "recommended" if `predicted_probability` clears
  `DAILY_PICK_MIN_PROBABILITY` (config.py, default 0.80) - a day can
  surface 0, 1, or up to `DAILY_PICK_MAX` (2) picks depending on how many
  clear the bar, not a fixed count regardless of matchup quality.
- A pick with an at-bat and no hit resets the streak to 0.
- A pick with zero at-bats that day (rained out, DNP, not in the lineup)
  is neutral - it neither advances nor resets the streak.
- Otherwise the streak increases by however many picks got a hit (0, 1,
  or 2).

Distinguishing "confirmed zero at-bats" from "outcome not known yet"
needed an `at_bats` field per pick (`predictions.py`) - a row only
resolves once its date falls within the coverage of the fetched/persisted
Statcast data, not just once that specific batter shows up in it.

This reads `docs/data/beat_the_streak_picks.csv` and
`beat_the_streak_summary.csv`, written by `evaluation.build_beat_the_streak_export()`
after every daily run and every backtest run - so the picks that make up the
streak are always the ones that were actually recommended *at the time*
(from `data/predictions/predictions.csv`), not recomputed with hindsight.

## Lineup awareness

Backtesting found ~30% of logged top-5 picks had zero at-bats on the day
they were picked - the batter wasn't even in the lineup. Two independent
fixes, both applied in `predictions.select_picks`:

- **Batting-order consistency** (`data.assign_batting_order`, `lineup.py`) -
  derived entirely from Statcast data already persisted, no new dependency,
  works retroactively. A batter only qualifies if their average batting-order
  slot over their current team's last `LINEUP_WINDOW_GAMES` games is in the
  top half (`LINEUP_TOP_HALF_MAX_SLOT`) *and* they've actually started at
  that rate (`LINEUP_MIN_START_RATE`) - together these exclude both a bench
  player's hot week and a recent call-up without a real track record. A
  mid-season trade resets the window to the batter's new team only.
- **Probable-pitcher matchup blending** (`schedule.py`, `matchup.py`) - a new
  dependency (`MLB-StatsAPI`, since `pybaseball` has no schedule/lineup
  support at all) fetches today's probable starters and blends each batter's
  `Game_Hit_Probability` with their opponent's probable starter's
  `PAVE_PLUS` and `Bullpen_PAVE_PLUS`, weighted by assumed at-bat share, into
  a separate `Matchup_Hit_Probability` metric - logged alongside the
  original via the existing multi-metric comparison rather than replacing
  it, since it's an unvalidated first pass that should be backtested before
  ever becoming the "recommended" metric. Also adds a "team is actually
  playing today" qualifier (an off day currently produces zero picks under
  either metric, not a stale recommendation).

Confirmed starting lineups (batting order 1-9 for *today's* game) are
explicitly out of scope for now - they post only ~2-4 hours before first
pitch, too late for the pipeline's 8am ET run regardless of data source, and
would need a second later run plus a decision about revising an
already-logged morning pick.

## Pick qualification: probability vs. Game_Hit_Probability

`hitters.py` computes two independent estimates of "will this batter get a
hit in their next game": `probability` (a binomial-style model applied to
the at-bat-level WAVE rate, assuming `WAVE_TRIALS_PER_GAME` trials per game)
and `Game_Hit_Probability` (the directly observed, independently
recency-blended rate of games with >=1 hit - no binomial modeling). When
these two diverge, it's informative, not noise: a high `Game_Hit_Probability`
with a low `probability` is a batter who barely gets a hit most games (a lot
of 1-for-4/5s); a high `probability` with a low `Game_Hit_Probability` is
boom-or-bust (occasional multi-hit games mixed with a lot of 0-fors). Neither
pattern is the "reliable to get a hit today" signal a pick is supposed to
represent.

`predictions.select_picks` requires *both* columns to clear
`config.HITTER_MIN_PROBABILITY` (0.7) before a hitter qualifies as a
candidate at all, and by default ranks the qualified pool by `Approach`
(`Game_Hit_Probability * probability`, already computed in
`hitters.assemble_hitters`) rather than `Game_Hit_Probability` alone - a
`rank_metric` parameter picks *which* qualified hitters get chosen without
changing what `predicted_probability`/`metric` report (still
`Game_Hit_Probability`, so `DAILY_PICK_MIN_PROBABILITY` and the rest of the
Beat the Streak tracking logic are unaffected). This was empirically
validated, not just theorized: a 42-day git-history replay backtest (the
same technique `git_backtest.py` uses to reconstruct picks) found that
`Game_Hit_Probability`-only ranking scored a 0.284 Brier score on resolved
picks - effectively a coin flip - with the picks that actually cleared
`DAILY_PICK_MIN_PROBABILITY` hitting only 55% of the time. Adding the joint
qualifier and ranking by `Approach` improved the Brier score to 0.260 and
raised that same recommended-pick hit rate to 65%, without reducing pick
coverage (still at least one pick on all 42 backtested days). Matchup
quality (`Matchup_Hit_Probability`) still applies on top of this qualifier
when today's schedule is available - it isn't a substitute for
`probability`/`Game_Hit_Probability` both being solid, but a further
tiebreaker among hitters who already are.

`Automated Game Picks` (below) uses a much smaller resolved sample (69
backtested games as of this writing) - not enough to safely recalibrate its
`GAME_PICK_MIN_PROBABILITY` threshold or ranking without overfitting to
noise (its per-bin calibration isn't monotonic at that sample size). That
recalibration should wait until more games accumulate.

## Automated Game Picks (dashboard)

A second, independent dashboard section predicts a winner for each of
today's games (not hitters) from six team-level signals: each team's
Pythagorean strength (`pyth_Strength`), Pythagorean confidence
(`pyth_Confidence`), suppression resistance (`suppression_resistance`), and
true power (`true_power`) - all from `confidence.csv` - adjusted by the
specific pitching (probable starter's `PAVE_PLUS` blended with
`Bullpen_PAVE_PLUS`) each team is projected to face, via the same
clip-then-blend logic `matchup.py` uses for hitters
(`matchup.clip_and_blend_pitching_quality`, shared by both).

- A game is only "picked" if the favored side's win probability clears
  `GAME_PICK_MIN_PROBABILITY` (config.py, default 0.58 - much lower than the
  hitter picks' 0.80, since single-game MLB win probabilities are
  compressed near 50/50 even for real favorites) - a day can surface 0 or
  more picks depending on how much separation the model sees, never a
  forced pick every game.
- `home_win_probability = home_rating / (home_rating + away_rating)`, where
  each team's rating is its own offensive composite (equal-weighted blend
  of the four signals above, `GAME_PICK_COMPOSITE_WEIGHTS`) multiplied by
  the *opposing* team's pitching-weakness multiplier. This is a simple
  ratio, not log5 - these composites aren't calibrated win percentages, so
  a ratio is the honestly-explainable choice rather than borrowing false
  precision from a formula built for a different kind of input.
- Tracked with a plain win/loss accuracy and a simple consecutive-correct
  streak (`game_evaluation.build_game_picks_export`) - not Beat the
  Streak's reset-on-any-miss multi-pick rule, since that's MLB's specific
  hitter-streak game mechanic and doesn't apply to independent game-by-game
  picks.

This reads `docs/data/game_picks_picks.csv` and `game_picks_summary.csv`,
written after every daily run from `data/predictions/game_predictions.csv`
(a separate log from the hitter-pick `predictions.csv`, keyed on `game_pk`
instead of a player id). Resolution uses final scores fetched via
`schedule.fetch_game_results`, not Statcast.

**Historical backtest** (`scripts/run_game_picks_backtest.py`,
`game_picks_backtest.py`): unlike probable-pitcher/live-schedule data
(never persisted - see `Matchup_Hit_Probability`'s equivalent limitation
above), schedule/game *outcomes* are already fully reconstructable from
data this project already has: `confidence.csv`/`pave.csv` have been
committed daily (same commit as `wave.csv`) since the project started, in
exactly the as-of-that-date snapshot shape the model needs, and Statcast
itself already contains each game's real `game_pk`, actual starting
pitchers, and actual final score - no separate schedule-API history
needed. This replays the last N days (default 40) of `confidence.csv`/
`pave.csv` git history through `game_picks.compute_game_win_probabilities`,
using the *actual* starter in place of the announced "probable" one (a
reasonable stand-in in hindsight - the two agree the vast majority of the
time) and resolves every pick immediately against the real final score,
since a backtested pick's outcome is already known at reconstruction time
(no pending state, unlike a live pick). Writes into the same
`data/predictions/game_predictions.csv` log and
`docs/data/game_picks_picks.csv`/`game_picks_summary.csv` exports used by
the live daily system, exactly like `git_backtest.py` does for hitter
picks - so the dashboard shows one continuous history, not two separate
tracks.

This is a first-pass, unvalidated blend (see `game_picks.py`'s module
docstring), meant to be watched and compared against reality before being
trusted, not treated as ground truth.

## Running

```
pip install -r requirements.txt
python scripts/wave.py                        # today, using games through yesterday
python scripts/wave.py --as-of-date 2026-06-15 # re-run against a past date
```

`--as-of-date` only ever uses games strictly before that date, so a run can
never be computed from a game that hasn't happened yet (or is still in
progress) - this is also what lets the pipeline be re-run against any past
date once enough raw history has accumulated in `data/raw/`.

## Tests

```
pip install -r requirements-dev.txt
pytest
```

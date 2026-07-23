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
- `scripts/evaluate_current_model.py` - read-only backtest report for the
  currently-live hitter-pick and game-pick models (see "Model versioning"
  below); doesn't write to `data/predictions/` or `docs/data/`.
- `scripts/fetch_lahman.py` / `scripts/build_age_curves.py` /
  `scripts/backtest_age_curve.py` - the Age Curves page's data pipeline
  (see "Age Curves" below); occasional-cadence, not part of the daily
  pipeline.
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

- A batter is only "recommended" if a blended score - the mean of whichever
  of `predicted_probability` (Game_Hit_Probability), `probability`, and
  `Matchup_Hit_Probability` are available for that pick (see
  `evaluation._combined_probability`) - clears `DAILY_PICK_MIN_PROBABILITY`
  (config.py, default 0.77). A day can surface 0, 1, or up to
  `DAILY_PICK_MAX` (2) picks depending on how many clear the bar, not a
  fixed count regardless of matchup quality. This used to gate on
  Game_Hit_Probability alone at 0.80, which ignored the other two signals
  and produced zero-pick days whenever GHP landed just under the bar even
  with a strong matchup; blending all three (and lowering the bar to match,
  since a mean of two-or-three probabilities runs lower than GHP alone -
  see config.py's docstring for the backtest that picked 0.77) fixes that
  without loosening quality, empirically validated via a 42-day
  git-history-replay backtest. A zero-pick day still gets an explicit
  `"no_pick"` row in the export (see `build_beat_the_streak_export`) rather
  than being silently absent - the dashboard shows "No pick for `<date>`"
  instead of falling back to whatever earlier day last had one, which would
  otherwise look like a stale/broken pipeline.
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

### Model versioning

`predictions.csv` is append-only - once a pick is logged, its
`predicted_probability`/which player was picked is frozen forever (only the
outcome fills in later). That means a selection-logic change (e.g. the
qualifier/ranking work above) never visibly moves the all-time accuracy
numbers until enough new-logic days outweigh the old ones, which can take a
long time. Every row is tagged with `model_version`
(`config.HITTER_MODEL_VERSION`, bumped whenever `select_picks`'s
qualifier/ranking logic meaningfully changes; git-history-reconstructed rows
from `git_backtest.py` are tagged `"legacy"` instead, since they don't
reflect current live logic) - `evaluation.summarize()`/
`build_beat_the_streak_export()` both take an optional `model_version` filter
so a recalibration's real forward performance is directly checkable
(`evaluation.summarize(log, model_version=config.HITTER_MODEL_VERSION)`)
instead of being diluted by history. `docs/data/beat_the_streak_summary_by_version.csv`
carries this same split (`all_time` plus the current version) for the
dashboard/anyone reading the CSVs directly. Game picks use the identical
pattern (`config.GAME_PICK_MODEL_VERSION`, `game_evaluation.build_game_picks_export`'s
own `model_version` filter).

Even with `model_version` tagging, a change made today still won't show up
in *live* picks until tomorrow's run - today's picks were already logged
before the change existed, and the append-only log correctly refuses to
rewrite them. `scripts/evaluate_current_model.py` answers "how does the
model I'd ship today actually perform" immediately instead of waiting:
it backtests the currently-live hitter-pick logic
(`git_backtest.reconstruct_historical_picks`, replaying `wave.csv` git
history through today's `select_picks()`) and the currently-live game-pick
logic (`game_picks_backtest.reconstruct_historical_game_picks_from_persisted`,
which recomputes `confidence.csv`/`pave.csv` fresh from persisted Statcast
per replayed date rather than replaying old git-committed snapshots, so a
signal added since - e.g. `Power_A_PLUS` - isn't silently skipped over the
way it would be replaying old commits), resolves both against real
outcomes, and prints an accuracy/Brier-score report. It's read-only - it
never writes to `data/predictions/` or `docs/data/`.

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
  support at all) fetches today's probable starters and, via a log5
  (odds-ratio) combination of the batter's own AB-level `WAVE` rate with the
  opposing starter's/bullpen's raw `PAVE`/`Bullpen_PAVE`, produces a
  `Matchup_Hit_Probability` for that specific matchup (see "Pick
  qualification" below for the full formula and why it actually drives picks
  now, not just a logged-alongside comparison). Also adds a "team is
  actually playing today" qualifier (an off day currently produces zero
  picks, not a stale recommendation).

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

### Matchup quality: probability, Game_Hit_Probability, and the opposing pitching

A good matchup is now checked just as much as a hitter's own form, not
logged as an unused side metric. When today's probable starters are known,
`matchup.compute_matchup_hit_probability` combines a batter's own AB-level
`WAVE` rate with the opposing pitching's AB-level rate they're actually
projected to face - `PAVE`/`Bullpen_PAVE` (the real batting-average-against
scale, not the league-normalized `PAVE_PLUS` ratio `game_picks.py`'s
team-level model uses) blended by assumed at-bat share
(`MATCHUP_STARTER_AB_SHARE`/`MATCHUP_BULLPEN_AB_SHARE`) - via log5 (the
odds-ratio method, a generalization of Bill James' log5 formula for
combining two rates against a shared league baseline into the rate expected
specifically between them). The resulting matchup AB-level rate converts to
a per-game `Matchup_Hit_Probability` via the same binomial-trials formula
`probability` itself uses, so all three land on the same 0-1 scale.

`predictions.select_picks`'s joint qualifier (see above) extends
automatically to `Matchup_Hit_Probability` whenever it's present on the
table passed in - a hitter needs `probability`, `Game_Hit_Probability`,
*and* a good matchup all above `HITTER_MIN_PROBABILITY`, not just the first
two. `pipeline.run` ranks that qualified pool by `Matchup_Approach`
(`Approach * Matchup_Hit_Probability`) on days matchup data is available,
falling back to `Approach` alone (no matchup qualifier at all) if the
schedule fetch fails - the same resilience pattern as the fetch itself.
`predicted_probability`/`metric` logged still reflect `Game_Hit_Probability`
either way, so `DAILY_PICK_MIN_PROBABILITY`'s calibration is unaffected.

`WAVE`/`PAVE`/`Bullpen_PAVE` are all AB-level and reliably reproducible from
persisted Statcast at any past as-of-date, which made a 30-day backtest
possible even without git-history CSVs (raw `PAVE`/`WAVE` were never
persisted before this change). Replaying the actual `predictions.select_picks`
logic (the full joint qualifier, ranked by `Matchup_Approach`) against the
"recommended" subset that actually counts toward Beat the Streak
(rank<=2, `Game_Hit_Probability`>=0.80) raised the hit rate from 65.8%
(n=38, no matchup) to 75.0% (n=32, with matchup) - a real, meaningful
improvement over 29-30 days of history.

That backtest was only possible after fixing a real, pre-existing bug this
investigation surfaced: `data.assign_game_ids` used to reconstruct game
boundaries from scratch via an at_bat_number-reset counter, which silently
assumed one real game's rows were contiguous in the table - nothing
guaranteed that (`persist_raw_statcast` only sorts by `game_date`, and MLB
plays ~15 games in parallel most days, so their at-bat rows interleave in
whatever order Statcast/pybaseball delivered them). Confirmed on real data:
one sample batter's 96 real games were fragmented into 233 different
`game_id` values, deflating any from-scratch `Game_Hit_Probability`
reconstruction by roughly 40%. Fixed by grouping directly on Statcast's own
`game_pk` (MLB's real, already-unique game identifier - the same one
`game_picks_backtest.py` already used directly) instead of reconstructing
game identity at all; `game_id` itself stays a dense, date-ordered integer
(not raw `game_pk`, which isn't chronologically monotonic) so downstream
"latest game" logic (`teams.py`, `lineup.py`) is unaffected. Confirmed
against a real historical `wave.csv` commit: reconstructed
`Game_Hit_Probability` now matches the committed values exactly.

### Platoon and park awareness

Two further adjustments to `Matchup_Hit_Probability`, on top of the base
batter-vs-starter blend above:

- **Platoon.** A batter's `WAVE` used to be a single hand-blended rate
  regardless of whether today's probable starter throws left- or
  right-handed - a real platoon split (common) got averaged away.
  `pitchers.compute_pitcher_throws` exposes each pitcher's own throwing
  hand (`Throws` in `pave.csv`, constant per pitcher), and
  `matchup._platoon_wave` now picks the batter's `WAVE_L`/`WAVE_R` (now
  also in `wave.csv`) to match, falling back to the blended `WAVE` when
  `Throws` is unknown (unannounced starter) or absent (old snapshots).
- **Park.** `teams.compute_park_factors` computes `Park_Factor` per team's
  home venue (each team's home games are treated as one park, since
  Statcast's pitch-level data doesn't carry a separate ballpark id) -
  combined runs/game at that venue relative to the across-all-parks
  average, the same PAVE_PLUS-style ratio convention (mean 1.0) used
  elsewhere. `matchup.py` looks up the actual venue for today's game (the
  home team, via `schedule_df`'s `is_home` - a road batter's *own* park is
  irrelevant, only the venue they're actually playing in matters) and
  scales the matchup AB rate by it, clipped to `MATCHUP_PARK_FACTOR_CLIP`
  and dialed by `MATCHUP_PARK_FACTOR_WEIGHT`.

Both were validated together via a 15-date persisted-Statcast backtest
(recomputing `wave.csv`/`pave.csv`/`confidence.csv` fresh per date, the
same technique `game_picks_backtest.py`'s persisted variant uses - neither
signal exists in old git-committed snapshots, so a git-history replay
can't see them): on the full qualified-candidate pool (n=43-52 resolved
picks per variant), platoon and park together raised the hit rate from
58% (neither) to 65% and cut the Brier score from 0.286 to 0.244 - the
best of every combination tried (platoon alone: 62%/0.261; park alone:
57%/0.293, roughly a wash by itself). See
`config.MATCHUP_PARK_FACTOR_WEIGHT`'s docstring for the full numbers,
including the smaller `rank<=2` "recommended" subset (n=11-18 - too small
to trust on its own, and noisier in the opposite direction on this
particular window). Revisit once more dates accumulate.

## Automated Game Picks (dashboard)

A second, independent dashboard section predicts a winner for each of
today's games (not hitters) from six team-level signals: each team's
Pythagorean strength (`pyth_Strength`), Pythagorean confidence
(`pyth_Confidence`), suppression resistance (`suppression_resistance`), and
true power (`true_power`) - all from `confidence.csv` - adjusted by the
specific pitching each team is projected to face, via the same
clip-then-blend logic `matchup.py` uses for hitters
(`matchup.clip_and_blend_pitching_quality`, shared by both).

Pitching quality itself blends two complementary signals from the probable
starter and that team's bullpen: `PAVE_PLUS` (hit-rate against) and
`Power_A_PLUS` (total-bases-allowed rate against - a run-prevention/
ERA-like signal computed the same recency-windowed way as PAVE, see
`pitchers.py`; deliberately not raw ERA, which isn't computed from this
project's own Statcast data). `PAVE_PLUS` alone treats every hit allowed
identically, so a pitcher who mostly allows singles and one who mostly
allows home runs could look the same despite very different run-prevention
profiles - `config.GAME_PICK_SUSCEPTIBILITY_WEIGHT` (default 0.5, an equal
blend) controls the split, empirically validated via a 35-day
persisted-Statcast backtest: the 0.5/0.5 blend beat pure `PAVE_PLUS` on both
accuracy (56.3% vs 54.3%) and Brier score (0.2493 vs 0.2517).

- A game is only "picked" if the favored side's win probability clears
  `GAME_PICK_MIN_PROBABILITY` (config.py, default 0.58 - much lower than the
  hitter picks' 0.77, since single-game MLB win probabilities are
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

## Probable Pitchers (dashboard)

A small daily list of today's games with each probable starter, their
throwing hand (`Throws`), and `PAVE`/`PAVE_PLUS`/`Power_A_PLUS`
(`schedule.build_probable_pitchers_table`),
written to `docs/data/probable_pitchers.csv` alongside the matchup blend
(same resilience as the rest of Part B - a failed/empty schedule fetch means
no file that day, not a stale one). A team with an unannounced or
unmatched starter still gets a row with blank pitcher fields, so the day's
full slate stays visible rather than silently missing teams.

## Age Curves (`docs/age-curves.html`, exploratory - separate page)

A standalone page, entirely separate from the daily pick pipeline: pick a
current hitter or pitcher and see how their season stat line compares to
historical players at the same age, plus a projection for next season
built from what those historical comparables actually did the following
year.

Uses Lahman's Baseball Database (1871-2025), read from CSVs committed
under `Lahman_Raw/` at the repo root (`lahman_data.py`) - downloaded by
hand from [SABR's mirror](https://sabr.app.box.com/s/y1prhc795jk8zvmelfd3jq7tl389y6cd)
and refreshed manually, roughly once a season (see `Lahman_Raw/Readme.md`
for the refresh steps). Two automated alternatives were tried first and
both failed: pybaseball's own built-in `lahman` submodule downloads a zip
from the baseballdatabank GitHub repo at call time, and that download
started failing (`zipfile.BadZipFile`, reproduced in real CI with full
internet access - an upstream break, not a local network issue); the
`lahman` PyPI package needs no network fetch but turned out to be a single
abandoned release from 2022 with data frozen at the 2020 season. A
manually-refreshed, committed CSV sidesteps both failure modes.

A different data source than the rest of this project (Statcast), needed
because Lahman's historical seasons have no Statcast-derived signal
(WAVE/PAVE) to compare against. A current player is
put on the same stat basis - season `AVG`/`OBP`/`SLG`/`OPS` for hitters,
`K9`/`BB9`/`HR9`/`FIP` for pitchers (`traditional_stats.py`, reusing
`helpers.py`'s event classifiers, not recency-windowed like WAVE) - so
they're comparable to a single historical season. `key_mlbam` (this
project's own identity) bridges to Lahman's `playerID` via a two-hop
crosswalk: `chadwick_register()`'s `key_bbref` column matches Lahman
People's `bbrefID` (`lahman_data.build_crosswalk`).

Pitcher metrics deliberately **exclude ERA**: this project already avoids
ERA for its live pick models (`Power_A_PLUS` - ERA is too dependent on
defense/sequencing/inherited runners to isolate a pitcher's own
performance) and applies the same reasoning here. `K9`/`BB9`/`HR9` are the
three defense-independent "own stuff" component rates (mirroring
`AVG`/`OBP`/`SLG`'s role on the hitter side); `FIP` combines those same
three components into one number (mirroring `OPS`), using a fixed
constant (`config.AGE_CURVE_FIP_CONSTANT = 3.10`, a pure additive shift
that doesn't affect comparable-search distances or projections) rather
than a real per-season constant, which would need ERA/runs data.

The page computes a **separate curve/projection per metric**
(`config.AGE_CURVE_HITTER_METRICS`: `AVG`, `OBP`, `SLG`, `OPS`;
`config.AGE_CURVE_PITCHER_METRICS`: `K9`, `BB9`, `HR9`, `FIP` - all
selectable on the page) rather than one blended number - power (`SLG`)
typically peaks earlier and declines faster than plate discipline (`OBP`),
while contact rate (`AVG`) tends to be more stable, and pitching
components age differently from each other too, so collapsing them into
one composite would hide that. A player's "closest historical comparables"
can therefore be a genuinely different set of players depending on which
metric is selected (e.g. their power comps vs. their contact comps).

`age_curve.py`'s method (v1, deliberately simple - see its module
docstring), run independently for each metric: find the
`AGE_CURVE_K_NEIGHBORS` (25) historical same-position seasons within
`AGE_CURVE_AGE_WINDOW` (1) year of the current player's age with the
closest value on that metric (a hand-implemented nearest-neighbor sort -
no scikit-learn dependency, matching this project's pattern of
implementing its own stats rather than pulling in an ML library for one
call), then look at what those comparables actually did on that same
metric the *following* season. Comparables with no next season on record
(retired, hurt, released, or fell below `AGE_CURVE_MIN_AB`/`AGE_CURVE_MIN_IP`)
are excluded from the projection and that excluded fraction is reported
alongside it - survivorship bias made visible, not hidden. The projection
is a range (mean plus 25th/75th percentile), not a false-precision single
number.

The chart plots the player's own **real trajectory**, not just their
current season: for a player who already has Lahman-tracked history
(anyone who debuted before Lahman's most recent completed season), every
one of their own past ages on record is plotted as an actual point,
connected into a real multi-year line, with their current season and a
distinctly-marked projected next-season point at the end
(`age_curve_player_history.csv`, built by matching each current player's
own crosswalked Lahman `playerID` back into their own historical rows -
their real career arc, not a comparable's). A player with no Lahman
history yet (a very recent debut) only has their current + projected
points, same as before.

Known v1 simplifications (documented, not silently ignored): each metric's
comparable search is **independent** (ranked by that one stat alone, not a
joint multi-metric similarity - a future "one holistic comparable set,
several metric views" mode is a possible follow-up); **no era/park
adjustment** - comparing raw rates across very different offensive eras
(e.g. the high-offense late 1990s vs. a lower-average modern era) or
ballparks is a real limitation of this first pass.

`scripts/fetch_lahman.py` (converts `Lahman_Raw/*.csv` to
`data/raw/lahman/*.parquet`) and `scripts/build_age_curves.py` (writes
`docs/data/age_curve_projections.csv`/`age_curve_comparables.csv`/
`age_curve_league.csv`/`age_curve_player_history.csv`) run on their own
**occasional cadence** (weekly, via
`.github/workflows/age_curves_update.yml` - also manually triggerable) -
deliberately a separate workflow from `daily_update.yml`, since historical
comparables don't move day to day. `scripts/backtest_age_curve.py`
validates the projection method itself against a sample of real past
player-seasons (each with a known real next season to check against),
using only comparable data available at or before that test season's own
year - the same no-lookahead discipline as `git_backtest.py`/
`game_picks_backtest.py` elsewhere in this project.

**Validated against real data**: 500 sampled real player-seasons from
2010-2019 (out of 32,158 qualified historical hitter-seasons / 28,127
qualified historical pitcher-seasons, 1871-2025). Every hitter metric
beat the naive "always guess the sample mean" baseline, with a real
positive correlation between projected and actual next-season value -
strongest for the power/contact-combination metrics:

| metric | MAE | naive baseline MAE | correlation | n scored |
|---|---|---|---|---|
| AVG | 0.0245 | 0.0254 | 0.352 | 355/500 |
| OBP | 0.0274 | 0.0284 | 0.429 | 355/500 |
| SLG | 0.0533 | 0.0582 | 0.473 | 355/500 |
| OPS | 0.0756 | 0.0789 | 0.446 | 355/500 |

(See `config.py`'s `AGE_CURVE_AGE_WINDOW` docstring for the full
methodology note - the 145/500 unscored seasons had no comparable with a
resolvable next season and are reported, not hidden.)

Pitcher metrics, same methodology: `K9` shows a strong signal (the
strongest of any Age Curves metric, hitter or pitcher - strikeout rate is
largely an "own stuff" skill that persists year to year); `BB9` and `FIP`
both clearly beat their baselines; `HR9` is reported honestly as a wash -
it correlates with next-season `HR9` (year-to-year home-run rate is
notoriously volatile, driven by batted-ball luck/park effects/defense as
much as pitcher skill) but its MAE is statistically indistinguishable
from just guessing the sample mean, so treat `HR9` projections on this
page as a weak signal, not a strong one:

| metric | MAE | naive baseline MAE | correlation | n scored |
|---|---|---|---|---|
| K9 | 1.1476 | 1.5878 | 0.746 | 296/500 |
| BB9 | 0.6277 | 0.7776 | 0.616 | 296/500 |
| HR9 | 0.3312 | 0.3264 | 0.321 | 296/500 |
| FIP | 0.5907 | 0.6606 | 0.440 | 296/500 |

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

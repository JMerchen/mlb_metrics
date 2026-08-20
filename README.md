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

- A pick is graded "recommended" if a blended score - the mean of whichever
  of `predicted_probability` (Game_Hit_Probability), `probability`, and
  `Matchup_Hit_Probability` are available for that pick (see
  `evaluation._combined_probability`) - clears `DAILY_PICK_MIN_PROBABILITY`
  (config.py, default 0.77), else "speculative". This used to gate on
  Game_Hit_Probability alone at 0.80, which ignored the other two signals
  and produced zero-pick days whenever GHP landed just under the bar even
  with a strong matchup; blending all three (and lowering the bar to match,
  since a mean of two-or-three probabilities runs lower than GHP alone -
  see config.py's docstring for the backtest that picked 0.77) fixed that,
  empirically validated via a 42-day git-history-replay backtest.
- **The dashboard's top DAILY_PICK_MAX (2) candidates are ALWAYS shown,
  each individually graded** (`evaluation.graded_daily_picks`) - real user
  feedback: that 42-day replay never carried real (non-NaN)
  `Matchup_Hit_Probability` (it wasn't persisted to git history at the
  time), so once live runs started actually carrying it most days, the
  blended mean ran measurably lower than the replay ever exercised - a real
  5-day-straight stretch (2026-08-11 through 2026-08-15) landed the
  top-ranked candidate's combined probability at 0.71-0.77 every single
  day, just under the bar, leaving the dashboard blank despite real,
  already-qualified candidates existing all five days. Rather than
  re-chase a moving threshold, `DAILY_PICK_MIN_PROBABILITY`'s role changed
  instead of its value: it's no longer "the bar a day needs to clear to
  show anything" - it's now the recommended/speculative grade boundary,
  and the dashboard always shows its real top candidates, honestly graded,
  never blank on a weak-slate day. **Only "recommended"-grade picks still
  count toward the tracked streak/day_survival_rate** (`streak_progression`/
  `_recommended_picks`, unchanged) - a "speculative" day is still a real
  no-op for the streak, exactly like a zero-pick day was before. A date
  with literally NOTHING logged (a true off day, not a weak one) still
  gets the explicit `"no_pick"` row in the export rather than silently
  vanishing.
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

- **Recently-played eligibility** (`hitters.compute_last_game_dates`,
  `config.HITTER_MAX_DAYS_SINCE_LAST_GAME`, default 5 days) - a hitter's
  career-long PA total and season-long WAVE/Game_Hit_Probability rates stay
  high even after a week or more hurt/benched, so without this an inactive
  player could still surface as an official pick. `predictions.select_picks`
  excludes any hitter whose most recent completed game is more than the
  threshold before the pick date (column-gated, so old wave.csv snapshots
  without `Last_Game_Date` are unaffected). This is an eligibility/data-
  hygiene gate, not a new predictive signal - it doesn't touch ranking, only
  who's eligible to be ranked at all - so unlike the trained hit-probability
  model (still explicitly NOT wired into live picks) it didn't need its own
  backtest, the same reasoning that applies to the batting-order/lineup
  gates above. Real-data check on 2026-07-28 excluded 213 of 602 hitters
  (season-long stragglers - traded, released, injured) while correctly
  keeping actually-active players eligible.

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

### Real bug fixed: PAVE excluded strikeouts from the AB denominator

Surfaced by a direct, specific complaint: a hitter's DFS matchup that day
was against Jacob Misiorowski, a dominant strikeout starter, and neither
`Matchup_Hit_Probability` nor the DFS pitcher projection treated that
matchup as tough at all. `pitchers._pave_rate` converts a hits-per-plate-
appearance rate into a hits-per-at-bat rate by excluding non-at-bat
events from the denominator - but the original formula excluded
strikeouts from that denominator ALONGSIDE walks/HBP
(`helpers.is_strikeout_walk_hbp`). A strikeout **is** an official at-bat;
only a walk or HBP isn't. Excluding it too shrinks the effective
denominator for every strikeout a pitcher records, which **inflates**
the computed hit-rate for exactly the pitchers who rack up the most
strikeouts - the more swings-and-misses a pitcher generates, the worse
the old formula made him look. Fixed by excluding only walks/HBP
(`helpers.is_non_at_bat_event`), leaving strikeouts in the AB count.

Confirmed against Misiorowski's real 2026 Statcast log (439 batters
faced: 173 K, 28 BB, 8 HBP, 61 hits): the old formula gave a full-season
PAVE of 0.265 (PAVE_PLUS ~0.93 - barely better than average); the fixed
formula gives 61/(439-28-8) = 0.151 - genuinely elite, matching his real
Cy-Young-caliber season.

**Re-backtested on the same real 20-date sample, same dates, before vs.
after the fix** (`dfs_backtest.backtest_dfs_projections` - the pitcher
side is the most directly affected, since `Expected_H_Allowed = PAVE *
Expected_IP * DFS_BATTERS_FACED_PER_INNING`):

| signal | MAE before | MAE after | naive-baseline MAE | correlation before | correlation after |
|---|---|---|---|---|---|
| `Expected_H_Allowed` vs. `Actual_H` | 2.6252 | **1.8032** | 1.8395 | 0.262 | 0.299 |
| `DK_Points_Pitcher` (combined) | 6.8840 | 6.8312 | 7.1776 | 0.327 | 0.332 |
| `DK_Points_Hitter` (heuristic) | 4.7161 | 4.7156 | 4.6879 | 0.010 | 0.010 |

`Expected_H_Allowed` is the headline result: its MAE was worse than the
naive baseline before this fix (a previously-documented, unresolved weak
signal - see "Pitchers" below) and now **beats** the baseline for the
first time, a direct, real confirmation that the bug fix - not just the
Misiorowski anecdote - measurably improved a signal this project had
already flagged as broken. `DK_Points_Pitcher`'s combined MAE improves
more modestly (it also blends the FIP-based ER estimate, which PAVE
doesn't touch). `DK_Points_Hitter`'s heuristic number is essentially
unchanged - expected, since that heuristic's correlation was already
near zero for unrelated, previously-documented reasons (the
`compute_matchup_adjustment` ratio - see "Hitters" below), and hitters'
live projection is served by the ML model, not this heuristic, anyway.

**What this fix reaches, project-wide**: every consumer of
`PAVE`/`PAVE_PLUS`/`Bullpen_PAVE` - `matchup.py`'s
`Matchup_Hit_Probability` (both the hitter matchup ratio above and its
Boom-Adjusted/Value_Score downstream uses), `game_picks.py`'s
susceptibility signal (blended against `Power_A_PLUS`, see
`GAME_PICK_SUSCEPTIBILITY_WEIGHT`), and `dfs.py`'s `Expected_H_Allowed`.
The DFS hitter/pitcher ML models (`dfs_ml.py`) were retrained after this
fix since `starter_PAVE`/`Bullpen_PAVE`/`Expected_H_Allowed` are direct
input features - see "Machine learning follow-up" below for the
retrained numbers.

**Automated Game Picks re-checked too, same 20-date sample, before vs.
after** (`game_picks_backtest.reconstruct_historical_game_picks_from_persisted`,
also a fresh recompute, not a git-history replay): accuracy actually went
from 57.9% to 56.1% (Brier 0.2470 -> 0.2486) - a small move in the WRONG
direction. On n=57 resolved games that's exactly one game's outcome
flipping, well within noise for a sample this size - not treated as a
real regression, but reported plainly rather than omitted because it
didn't confirm the hypothesis. The likely reason it barely moved at all
either way: `game_picks.py` blends `PAVE_PLUS` (a ratio re-normalized to
mean 1.0 across that day's qualified pitcher pool), not the raw PAVE
rate the DFS pitcher signals use directly - renormalization absorbs most
of a uniform formula shift, so this signal was always going to be far
less sensitive to the fix than `Expected_H_Allowed`'s raw AB-rate
calculation. `GAME_PICK_SUSCEPTIBILITY_WEIGHT` (0.5) was left unchanged;
one game on n=57 isn't grounds to recalibrate a weight that was itself
chosen from a real backtest.

**What could NOT be cleanly re-verified**: the 65.8%->75.0% Beat the
Streak hit-rate uplift reported just above, and the original
`Ceiling_DK_Points`/`Boom_Rate` capture-rate backtests, replay
git-history-committed CSV snapshots (`docs/data/wave.csv`/`confidence.csv`
at past commits) that were computed under the OLD, buggy PAVE - rewriting
that committed history to "fix" it retroactively isn't something this
project does. Those numbers stand as historically accurate for what was
live at the time; the corrected signal's real effect on live picks will
show up naturally as new daily data accumulates under the fixed formula
going forward, the same way any other model change's live impact is
observed.

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

### Real batted-ball-quality signal (`hitters.compute_quality_of_contact`)

Every windowed hitter signal up to this point (`WAVE`, `Game_Hit_Probability`,
`Consistency`, `Approach`, the extended DK rates) is an outcome-RATE
statistic - hit/walk/RBI rates blended over recent games. None of them
measure contact QUALITY directly: how hard/well the ball was actually hit,
independent of whether it happened to find a fielder. The raw Statcast
columns for that (`launch_speed`, `estimated_ba_using_speedangle`,
`estimated_woba_using_speedangle`, `launch_speed_angle`) were already
present on every real Statcast row, just never used.

`compute_quality_of_contact(dt)` computes real recency-windowed
`Exit_Velo` (mean `launch_speed`), `Barrel_Rate` (share of batted balls
with `launch_speed_angle == 6` - Statcast's own real "barrel" bucket, not
an approximation reconstructed from the raw formula), `xBA` (mean
`estimated_ba_using_speedangle`), and `xwOBA` (mean
`estimated_woba_using_speedangle`), computed over real batted-ball rows
only (`helpers.is_batted_ball` filters to `type == "X"` - a strikeout/
walk/HBP has no batted-ball data and would otherwise silently bias every
rate toward 0/NaN). Blended by throws-side using the exact same
`_blend_windows`/`_side_window_agg` machinery `WAVE`/`WTB`/the extended DK
rates already use, reusing `config.WHOPS_WTB_WINDOWS`. Merged into
`wave.csv` and into `dfs_ml.HITTER_FEATURE_COLUMNS` (both the DK-points
regressor and the hit-probability classifier), so both models can learn
from contact quality directly instead of only ever seeing outcome rates.

**A real production bug found via a live retrain, not a synthetic test**:
`helpers.is_barrel` originally did `(launch_speed_angle == 6).astype(int)`.
Every synthetic test fixture supplied clean, non-null values, so `pytest`
passed cleanly (541 tests) - but the real persisted
`data/raw/statcast_2026.parquet`'s `launch_speed_angle` column has genuine
null values on some real tracked batted balls, and comparing a pandas
nullable-dtype column via `==` against those null rows returns `pd.NA`
(not `False`, per three-valued comparison logic), which `.astype(int)`
can't cast - `scripts/train_dfs_ml_models.py` crashed for real on the
first retrain attempt (GitHub Actions run 32280494858). Fixed with
`.fillna(False)` before casting, so "no tracked quality bucket" reads as
"not a barrel" - a real, honest 0, matching every other classifier's
missing-data convention in `helpers.py`. The retrain that followed the
fix succeeded end to end (run 32281354568) - see the significance report
above and the DFS ML retrain table below for what these four columns
actually contributed once real data was behind them.

### Real pitch-type-specific platoon matchup (`pitchers.compute_pitch_arsenal`, `hitters.compute_pitch_family_rates`, `matchup._pitch_arsenal_multiplier`)

A pitcher's real fastball/breaking/offspeed pitch mix (Statcast's own
`pitch_type` column) was sitting unused in the same already-persisted raw
data as the batted-ball-quality columns above - never used as a matchup
signal beyond the existing handedness platoon adjustment.
`helpers.pitch_type_family` groups every real `pitch_type` code into the
same three-bucket scheme Baseball Savant's own pitch-arsenal pages use.
`pitchers.compute_pitch_arsenal` computes a pitcher's real windowed USAGE
mix (what share of their actual pitches are fastball/breaking/offspeed -
needs every real pitch thrown, not just PA-ending ones, so it's the one
function in this project built on `pipeline.build_all_pitch_events` rather
than `data.completed_events`). `hitters.compute_pitch_family_rates`
computes the batter-side complement: WAVE split by the PA-ending pitch's
family instead of by pitcher handedness. Both are merged into
`dfs_ml.HITTER_FEATURE_COLUMNS` (`Fastball_WAVE`/`Breaking_WAVE`/
`Offspeed_WAVE` for the batter, `starter_fastball_rate`/
`starter_breaking_rate`/`starter_offspeed_rate` for today's probable
starter) so the ML models can learn from this matchup directly, and
`matchup._pitch_arsenal_multiplier` blends them into `Matchup_Hit_Probability`
the same clip-then-weight-dial way `_park_factor_multiplier` does - ships
at `config.MATCHUP_PITCH_ARSENAL_WEIGHT = 0.0` (informational-only, exact
no-op) until a real historical-reconstruction backtest earns a nonzero
default, same "ship conservatively" precedent `PITCHER_MATCHUP_OFFENSE_WEIGHT`
already established - that backtest is a genuinely separate undertaking
from the ML-feature validation below and hasn't been run yet.

**Two real production bugs found via live retrains, not synthetic
tests** (both GitHub Actions runs against real persisted data, neither
caught by the 559-560 passing synthetic-fixture tests at the time):
1. `scripts/train_dfs_ml_models.py` crashed with a `TypeError` deep
   inside `matchup.compute_matchup_hit_probability`'s final `.clip(0, 1)`
   (run 32288246152). Root cause: `_pitch_arsenal_multiplier`'s
   `vs_league.replace(0, pd.NA)` silently upcast a float64 Series to
   `object` dtype - every arithmetic op downstream then ran through
   Python's own operators instead of numpy's, and on a real row
   `(1 - matchup_ab_rate) ** 3.5` produced a genuine Python `complex`
   value (Python's `**` does that for a negative base and a non-integer
   exponent; numpy's vectorized power would have just given `nan`) -
   `.clip()` then couldn't compare a complex number against an int.
   Fixed by using plain float division instead (`vs_starter`/`vs_league`
   can only ever be a genuine 0/0, never x/0, since both are weighted
   sums of the exact same `Fastball_WAVE`/`Breaking_WAVE`/`Offspeed_WAVE`
   terms) - confirmed against the exact failing case afterward: a real
   float64 `1.0`, never an object-dtype value.
2. (See the batted-ball-quality section above for the earlier, separate
   `is_barrel`/`pd.NA`-vs-`np.nan` bug this project already hit once -
   the SAME underlying class of bug (a pandas nullable/NA value leaking
   into a numeric computation and breaking a downstream cast or op)
   resurfaced in genuinely different code here, which is itself worth
   noting: `pd.NA` in an otherwise-float computation is a real, recurring
   footgun in this codebase, not a one-off mistake.

**Real retrain results** (GitHub Actions run 32289285787, after both
fixes): individually significant (p<0.0001 each) - real signal, not
noise, same pattern the batted-ball-quality columns showed. Not
significant in the combined multivariate model (collinear with existing
WAVE/Approach signal - see the significance report above for the exact
numbers) and the starter's own `starter_fastball_rate`/
`starter_breaking_rate`/`starter_offspeed_rate` were never significant
either way. The full walk-forward hitter hit-probability model (all 28
features together) came out essentially unchanged from the immediately
prior run - adding these six columns didn't move the holdout numbers by
a meaningful margin, consistent with the significance report's finding
that they're collinear rather than independently predictive. Reported
honestly: this slice's real value so far is the two production bugs it
surfaced and fixed, not a measurable accuracy gain - the
`MATCHUP_PITCH_ARSENAL_WEIGHT` matchup multiplier remains unvalidated
and off (0.0) pending a real backtest.

### Real plate-discipline signal (`helpers.is_swing`/`is_whiff`/`is_out_of_zone`/`is_chase`, `hitters.compute_plate_discipline`)

The last raw signal this item set out to add: every pitch a hitter sees or
takes carries real swing/contact/zone information in Statcast's own
`description` and `zone` columns, already sitting unused in the same
persisted raw data the batted-ball-quality and pitch-arsenal work above
already tapped. `helpers.is_swing` classifies real Statcast `description`
codes into swing vs. take (bunts counted as swings - a documented
simplification, since bunts are ~0.25% of real pitches); `is_whiff`
classifies genuine swinging strikes (explicitly excluding `foul_tip`,
which Statcast scores as real contact, not a miss); `is_out_of_zone` uses
Statcast's own real `zone` 11-14 "chase quadrant" codes (its own
classification of the area just outside the 3x3 strike-zone grid, not
something derived from `plate_x`/`plate_z`/`sz_top`/`sz_bot`); `is_chase`
is the conjunction of the two. `pipeline.build_all_pitch_events` (already
built for the pitch-arsenal work above) was widened to carry
`description`/`zone` alongside `pitch_type`, and its blanket
`pitch_type`-null pre-filter was removed - `description`/`zone` are
populated on some rows where `pitch_type` is null, so a pre-filter keyed
only on `pitch_type` would have silently dropped real plate-discipline
data. `hitters.compute_plate_discipline` blends per-window `Whiff_Rate`
(swinging-strike rate on swings) and `Chase_Rate` (chase rate on
out-of-zone pitches) using the same rate-then-blend pattern every other
windowed signal in this codebase follows (`WAVE`, PAVE,
`compute_quality_of_contact`, `compute_pitch_arsenal`) - compute the RATE
per window first, then blend the windows' rates by weight, reusing
`config.WAVE_WINDOWS`. Merged into `wave.csv` and
`dfs_ml.HITTER_FEATURE_COLUMNS`.

**Real retrain results** (GitHub Actions run 32304349888, 2026-08-19 -
first clean run this item's three slices have had, no new production bug
surfaced): `Whiff_Rate` is significant both individually (coef -0.0980,
p<0.0001) and in the combined multivariate model (coef -0.1011,
p<0.0001) - the expected direction (a hitter who whiffs more often on
swings is less likely to get a hit) and, unlike most of the
batted-ball-quality/pitch-arsenal columns, it SURVIVES the multivariate
fit rather than collapsing to collinearity with existing WAVE/Approach
signal. `Chase_Rate` is only marginal individually (coef 0.0197,
p=0.0785) but reaches significance in the combined model (coef 0.0380,
p=0.0073) - both with a positive sign, which is counterintuitive (naively,
chasing more should predict fewer hits, not more) and worth flagging
honestly rather than glossing over; a plausible read is that `Chase_Rate`
is partly standing in for aggressive, high-contact-skill hitters once
`Whiff_Rate` is already in the model, but this hasn't been separately
tested. The full walk-forward hitter hit-probability model (now 30
features) improved slightly on this run: holdout log_loss=0.6802 (vs.
0.6817 the immediately prior run), ROC AUC=0.5726 (vs. 0.5679) - both
still beating naive-baseline (log_loss 0.6843) and the
`Game_Hit_Probability` heuristic (log_loss 0.7003, ROC AUC 0.5527) by a
real if modest margin. This is the first of item #1's three slices where
the new columns visibly moved the holdout numbers in the model's favor
rather than landing flat - a genuinely useful addition, not just informational.

### Hitter hit log (`data/predictions/hitter_hit_log.csv`, data asset - not yet a live signal)

Investigating why Beat the Streak's day-survival rate sits at ~50% found a
real gap: the only signal with enough logged pick history to test
(`Game_Hit_Probability`) showed no statistically significant relationship
with actual hit outcomes, and the sample was tiny (n=64 resolved picks) -
`predictions.csv` only logs the ~2 hitters/day that ever became an
official pick, not the full slate. A logistic regression fit against that
few, gated rows can't give trustworthy coefficients or significance
levels.

`dfs_backtest.assemble_hitter_hit_log` (run daily via
`scripts/build_hitter_hit_log.py`) builds the real training table instead:
one row per hitter per game, for **every** hitter with a game that date
(not just the ones that cleared the pick gates), with every feature
computed strictly before that game - `starter_PAVE`, `Bullpen_PAVE`,
`WAVE`/`WAVE_L`/`WAVE_R`, `Total_PA` (`PA_L + PA_R`),
`Game_Hit_Probability`, `Matchup_Hit_Probability`, `Park_Factor`, and the
rest of `dfs_ml.HITTER_FEATURE_COLUMNS` - plus a binary `Got_Hit` label
(did that batter record at least one hit that date, from real completed
Statcast events). Same no-lookahead recompute (`dfs_backtest._compute_date_outputs`)
and the same feature-building function (`dfs_ml.build_hitter_features`)
already used and tested for the DFS ML models, so this reuses trusted
machinery rather than hand-rolling a second merge path.

`scripts/build_hitter_hit_log.py` (no `--days` for a one-time full
historical backfill; `--days 10` for the daily incremental run
`daily_update.yml` uses) appends and dedupes on `(date, key_mlbam)` -
keep-last, so a freshly recomputed row always wins over a stale one and
re-running never creates duplicate rows.

This log is **purely a data asset today** - it feeds nothing live. Fitting
the actual logistic regression against it (and deciding whether it ever
replaces or augments `predictions.select_picks`'s probability gate) is a
follow-up once the log has accumulated real history.

### Fitting the predictive model: logistic regression vs. gradient boosting (`scripts/train_hitter_hit_model.py`)

Real answers to the "why is survival only ~50%, and which features are
actually significant" question, now that the hit log above fixes the
sample-size problem (n=26,955 hitter-games, PA-qualified, vs. the n=64
resolved picks the earlier ad hoc analysis had to work with). Two
deliverables from one script, both fit on rows filtered to
`Total_PA >= config.BACKTEST_MIN_PLATE_APPEARANCES` (the same gate
`predictions.select_picks` already applies before a hitter can be a pick
candidate - a handful of career plate appearances gives a mostly-noise
`WAVE`/`Game_Hit_Probability`, and letting that noise into the fit would
just drag down real coefficients):

**Significance report** (`statsmodels.Logit`, full history, real numbers
from 2026-08-19, after adding the batted-ball-quality columns, the
pitch-type-family columns, and the plate-discipline columns below):
individually (one univariate model per feature), `WAVE`, `WAVE_L`/`WAVE_R`,
`PA_L`/`PA_R`, `probability`, `Game_Hit_Probability`, `Consistency`,
`Approach`, `Expected_Bases`, `Expected_RBI`, `Exit_Velo`, `Barrel_Rate`,
`xBA`, `xwOBA`, `Fastball_WAVE`, `Breaking_WAVE`, `Offspeed_WAVE`,
`Whiff_Rate`, `Park_Factor`, and `Matchup_Hit_Probability` are all
significant (p<0.001) - notably including `Game_Hit_Probability` itself
(coef 0.2153, p<0.0001), the OPPOSITE of the much-earlier n=64 finding.
That earlier result wasn't wrong given its data - it was underpowered; a
64-pick sample simply can't reliably detect an effect of this size.
`Chase_Rate` is only marginal individually (coef 0.0197, p=0.0785).
`Expected_BB`, `Expected_HBP`, `starter_PAVE`, `Bullpen_PAVE`, `is_home`,
and `starter_fastball_rate`/`starter_breaking_rate`/`starter_offspeed_rate`
are not individually significant. In the combined (multivariate) model,
`probability`, `Game_Hit_Probability`, and `Consistency` show enormous
standard errors (~1,000,000+) and are unusable there - real, expected
multicollinearity, since `Consistency` (`Game_Hit_Probability -
probability`) and `Approach` (`Game_Hit_Probability * probability`) are
algebraically derived from those same two columns. `WAVE_L`, `WAVE_R`,
`PA_R`, `Park_Factor`, `Whiff_Rate` (coef -0.1011, p<0.0001), and
`Chase_Rate` (coef 0.0380, p=0.0073) remain significant once the others
are controlled for - `Whiff_Rate` in the expected direction (more whiffs,
fewer hits) and surviving the multivariate fit unlike most of the other
new columns; `Chase_Rate` significant but with a counterintuitive positive
sign, reported honestly rather than glossed over (see the plate-discipline
subsection above for a plausible read). **Reported honestly**:
`Exit_Velo`/`Barrel_Rate`/`xBA`/`xwOBA` and `Fastball_WAVE`/`Breaking_WAVE`/
`Offspeed_WAVE` are each individually significant on their own (real
signal, not noise), but none remain significant in the combined model
(p=0.48/0.49/0.93/0.45 for the first four, p=0.80/0.71/0.31 for the
pitch-family three) - they're substantially collinear with the existing
WAVE/Approach-family columns (a hitter who makes consistently hard
contact, or reliably ends ABs on fastballs successfully, also tends to
have a high recent WAVE), so this simple linear specification can't
cleanly credit them with independent lift once those columns are already
in the model. The starter's own `starter_fastball_rate`/
`starter_breaking_rate`/`starter_offspeed_rate` never reach significance
either way (individually or combined, p=0.31-0.98) - a genuine, useful
finding on its own, not a failure to ship - see the walk-forward model
result just below, which reflects all 30 columns together.

**Walk-forward-validated predictive model** (`ml_models.py`'s
`WalkForwardDateSplit`/nested-holdout machinery, same discipline as the
three live DFS ML models): quant-analytics item #2 ("model family") -
until this run, this was a single `sklearn.LogisticRegression` with no
real alternative ever tried. `scripts/train_hitter_hit_model.py` now
grid-searches TWO candidates on the same walk-forward CV -
`LogisticRegression` (`config.HITTER_HIT_LOGIT_C_GRID`) and
`HistGradientBoostingClassifier` (new `config.HITTER_HIT_GBM_PARAM_GRID`,
reusing `DFS_HITTER_GBM_PARAM_GRID`'s exact values) - and keeps whichever
wins by walk-forward CV `neg_log_loss`, mirroring
`train_dfs_ml_models.py`'s own Ridge-vs-GBM selection for
`DK_Points_Hitter`.

**Real run (2026-08-20, n=5,186 holdout, 30 features)**:
`LogisticRegression` `best_params={'C': 0.3}` `cv_score=-0.6754`;
`HistGradientBoostingClassifier`
`best_params={'learning_rate': 0.03, 'max_depth': 3, 'max_iter': 100,
'min_samples_leaf': 200}` `cv_score=-0.6747` - GBM wins the CV comparison
and is selected. On the untouched holdout: log_loss 0.6795 vs.
naive-baseline 0.6843 vs. the `Game_Hit_Probability` heuristic's 0.7003 -
beats both bars, saved. **Reported honestly, not a clean win**: compared
against the immediately-prior LogisticRegression-only run on the same
holdout window (log_loss 0.6802, ROC AUC 0.5726 - see the plate-discipline
section above), GBM's log_loss is marginally better (0.6795 vs. 0.6802)
but its ROC AUC is actually WORSE (0.5654 vs. 0.5726) and its accuracy is
essentially flat (0.5669 vs. 0.5673). Two different proper-scoring metrics
moved in opposite directions on the same holdout - a real, mixed result on
this modest-sized holdout, not a decisive verdict that gradient boosting
beats logistic regression here. Neither of these two runs' numbers is
directly comparable to the original 2026-07-29 run (log_loss
0.6757/0.6815/0.6901, ROC AUC 0.575/0.564) - the holdout window itself
moved forward with the season across all these runs, not just the model/
feature set. Saved to `config.HITTER_HIT_PROBABILITY_MODEL_PATH`
(`data/models/hitter_hit_probability_model.joblib`); `HITTER_MODEL_VERSION`
is unchanged - it tracks the selection/gating logic (v3/v4), not which
sklearn class produces `Model_Hit_Probability`, and a model-family swap
under the same artifact schema isn't a version-worthy event under this
project's own convention.

**Permutation-importance report** (`sklearn.inspection.permutation_importance`,
same untouched holdout, chosen over SHAP specifically to avoid adding a
new heavy dependency to a project that has never needed one beyond
scikit-learn/statsmodels): a real, model-agnostic sanity check on what the
winning GBM is actually learning from, computed independently of the
significance report's own Logit coefficients above. Top real features by
mean importance: `Game_Hit_Probability` (0.00245), `Consistency`
(0.00108), `PA_L` (0.00060), `PA_R` (0.00058), `Whiff_Rate` (0.00046),
`Offspeed_WAVE` (0.00018), `Park_Factor` (0.00016), `WAVE_R` (0.00014),
`Fastball_WAVE` (0.00013), `xBA` (0.00009) - every other feature's
importance rounds to ~0.0000-0.00002, several slightly negative (expected
sampling noise for a genuinely uninformative feature, not evidence of
active harm). **A real, honest divergence worth flagging**: `Whiff_Rate`
ranking 5th here roughly matches its own significance in the Logit above,
but `Matchup_Hit_Probability` (highly significant there, coef 0.1069,
p<0.0001) ranks near the very bottom here (0.00002), and `Chase_Rate`
(significant in the Logit's combined model) barely registers (0.00005).
GBM's tree splits absorb correlated signal differently than a linear
model's coefficients do - this isn't evidence either method is wrong, just
that "which features matter" genuinely depends on which model family is
asking, a real quant lesson this comparison surfaces rather than papering
over.

The model shortlists candidates for the official Beat the Streak picks
(`predictions.select_picks`, see the next section) rather than ranking
the whole pool directly - see `HITTER_MODEL_VERSION`'s `"v4-model-shortlist"`
docstring and the "Wiring the model into Our Picks" section below for why.
`.github/workflows/ml_training_update.yml`'s weekly retrain job now
includes `scripts/train_hitter_hit_model.py` alongside the three DFS/
age-curve models, so this artifact stays current instead of being frozen
at its original training date.

### Bayesian shrinkage for small-sample hitters (`helpers.shrink_rate`, `hitters.compute_wave`/`compute_game_hit_probability`)

Quant-analytics item #3 ("uncertainty quantification"), slice 1 of 3
(shrinkage → calibration → confidence intervals). Before this slice, a
15-PA rookie and a 500-PA veteran ran through the exact same `count / n`
division inside `compute_wave` and `compute_game_hit_probability` - the
only sample-size protection anywhere was a hard external gate
(`config.BACKTEST_MIN_PLATE_APPEARANCES`) that fully excludes a player
below the line and fully includes one above it, with nothing in between.
`helpers.shrink_rate` adds real Beta-Binomial empirical-Bayes shrinkage:
`(count + prior_strength * prior_rate) / (n + prior_strength)`, where
`prior_strength` is a real-unit pseudo-observation count (at-bats for
`compute_wave`, games for `compute_game_hit_probability`) - not an
abstract 0-1 weight - pulling a low-`n` player's rate toward a real
league-average rate (computed from the same already-loaded data, no new
fetch, no lookahead) while barely moving a high-`n` veteran's. As with
every other tunable weight in this codebase, `prior_strength=0.0` is the
exact null hypothesis - byte-for-byte identical to the unshrunk rate, not
an approximation of it - so both constants shipped at `0.0` until a real
backtest earned a nonzero default.

**Real backtest results** (`scripts/backtest_shrinkage.py`, dispatched via
`.github/workflows/debug_backtest_shrinkage.yml` against the full
persisted 2026 season - GitHub Actions run 32415595512, 2026-08-20):
swept `{0, 25, 50}` for each constant independently through the real
no-lookahead historical reconstruction (`dfs_backtest.assemble_hitter_hit_log`),
scored against real `Got_Hit` outcomes.

| strength | WAVE full-pop log_loss (n=37,744) | WAVE PA-gated log_loss (n=33,035) | Game_Hit_Probability full-pop log_loss | Game_Hit_Probability PA-gated log_loss |
|---|---|---|---|---|
| 0 (unshrunk) | 0.9468 | 0.6908 | 1.2024 | 0.6983 |
| 25 | 0.6831 | 0.6805 | 0.6776 | **0.6752** |
| 50 | 0.6821 | **0.6798** | 0.6786 | 0.6762 |

(Game_Hit_Probability's own best PA-gated value was strength=25 at
0.6752, edging out strength=50's 0.6762.) The **full unfiltered
population's** log_loss at `strength=0` is dramatically worse than the
PA-gated subset (0.9468 vs. 0.6908 for WAVE; 1.2024 vs. 0.6983 for
Game_Hit_Probability) - real evidence of exactly the problem this slice
targets, since that population includes the small-sample rows a hard PA
gate excludes from live picks today but that an unshrunk rate still
handles badly. On the PA-gated population - the real go/no-go bar, since
that's what live Beat the Streak picks are actually exposed to -
shrinkage was a clean win for both signals, so:

- `config.WAVE_SHRINKAGE_STRENGTH` shipped at **50.0** (log_loss 0.6798 vs.
  0.6908 unshrunk).
- `config.GAME_HIT_PROB_SHRINKAGE_STRENGTH` shipped at **25.0** (log_loss
  0.6752 vs. 0.6983 unshrunk).

`config.BACKTEST_MIN_PLATE_APPEARANCES`'s hard gate stays in place
unremoved - it still serves an "insufficient data to trust even a shrunk
estimate at all" role that shrinkage alone doesn't replace. Since these
are real, earned nonzero defaults that change `WAVE`/`Game_Hit_Probability`
values (and everything downstream: `Approach`, `Consistency`,
`Matchup_Hit_Probability`, `Model_Hit_Probability`), the hitter
hit-probability model was retrained afterward via the normal
`ml_training_update.yml` weekly job so its training features reflect the
shrunk values.

### Wiring the model into Our Picks (`pipeline.py`, `predictions.py`)

`pipeline.run()` resolves `predictions.select_picks`'s `rank_metric`
through a two-tier fallback, computed once per day: `Matchup_Approach`
(schedule/matchup data available) or `Approach` (no schedule at all).
Separately - independent of `rank_metric` - whenever the model artifact
loads and produces a real prediction for today's schedule
(`dfs_ml.predict_hitter_hit_probability` returns non-empty), its
`Model_Hit_Probability` column gets merged onto the pick pool and
`select_picks` uses that column's mere presence to narrow the
already-qualified pool down to a broad **shortlist** - its top
`config.HITTER_MODEL_SHORTLIST_SIZE` (10) candidates by
`Model_Hit_Probability` - BEFORE `rank_metric` ranks among that shortlist
and picks the final `top_n`. The model is a gate, not the ranker.

**This is a reversal of an earlier design (v3, `config.HITTER_MODEL_VERSION`)**
that let `Model_Hit_Probability` rank the WHOLE qualified pool directly,
gated on a probability threshold (`HITTER_MIN_MODEL_PROBABILITY`, since
removed). Real feedback after v3 shipped live (2026-08-05): a day it
surfaced a single hitter as the lone recommended pick and dropped another
hitter the user explicitly wanted, "because of their place in the
lineup" - a signal `Approach`/`Matchup_Approach` implicitly captures via
the `avg_batting_order`/`start_rate` qualifiers (a hitter who bats high
in an everyday lineup) that `Model_Hit_Probability` doesn't see directly
(it's not one of `dfs_ml.HITTER_FEATURE_COLUMNS`). The current design
(v4, `"v4-model-shortlist"`) keeps the model's original purpose - killing
pure hot-streak outliers by requiring a real matchup-aware model score at
all - while handing the final call back to the heuristic signal the model
doesn't capture. It also sidesteps `HITTER_MIN_MODEL_PROBABILITY`'s
calibration problem entirely: a rank-based cutoff needs no probability
threshold to derive or backtest.

`config.HITTER_MODEL_SHORTLIST_SIZE = 10` is an explicit user-specified
value, not backtest-derived like almost everything else in this file -
`scripts/backtest_selection_rule.py` (extended for this design, comparing
`heuristic_only` vs. `model_shortlist` variants head to head) should still
validate or inform retuning it, reported honestly either way, once that
real run is unblocked (see the script's own docstring for the sandbox
network limitation blocking it so far).

`predicted_probability`/`probability`/`Matchup_Hit_Probability`/
`Model_Hit_Probability` are all still logged to `predictions.csv` on
every pick regardless of whether the shortlist engaged that day, so
downstream evaluation/backtesting can always see the full picture.

Deliberately NOT bundled into this change (still, in v4 as in v3):
`evaluation.py`'s separate "recommended" gate
(`_combined_probability`/`config.DAILY_PICK_MIN_PROBABILITY`, which
decides which of the top-ranked candidates become the 0-2 picks shown as
"Our Picks" on the dashboard) still runs unchanged on top of whatever
`select_picks` returns - a deliberate fast-follow, not part of this
change (landing two new uncalibrated thresholds in one change would make
a good or bad outcome hard to attribute to either).

### Dashboard: Hit Streaks and Model Odds

The Beat the Streak section of the dashboard has three subtabs: **Our
Picks** (the official picks above - the model shortlists on days it's
available, per the wiring above, then `Approach`/`Matchup_Approach` picks
the final order), **Hit Streaks**, and **Model Odds**. The latter two
remain purely **informational** views alongside the official picks.

- **Hit Streaks** (`hitters.compute_current_hit_streaks`,
  `scripts/build_hit_streaks.py` → `docs/data/hit_streaks.csv`): each
  recently-active batter's real current consecutive-games-with-a-hit
  streak, counted from real completed Statcast events. A batter whose most
  recent game is more than `config.HIT_STREAK_RECENT_DAYS` (5) days old is
  excluded entirely, so an inactive/injured player's frozen streak doesn't
  crowd out who's actually hot right now. Never consulted by
  `predictions.select_picks` - informational only, same as before.
- **Model Odds** (`dfs_ml.predict_hitter_hit_probability`,
  `scripts/build_hitter_hit_predictions.py` → `docs/data/hitter_hit_predictions.csv`):
  today's PA-qualified hitters ranked by the trained hit-probability
  model's own predicted probability - the same model that gates the
  shortlist for Our Picks on days it's available, shown here
  independently for anyone who wants to see the model's own full ranking
  rather than just who it shortlisted.

Both scripts follow `build_dfs_rankings.py`'s resilience conventions
(missing input/model/failed schedule fetch leaves yesterday's output in
place rather than crashing or overwriting with nothing) and are wired into
`daily_update.yml` right after "Build DFS rankings."

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

- Every scheduled game is published (`game_predictions.select_game_picks`),
  flagged `above_threshold` if the favored side's win probability clears
  `GAME_PICK_MIN_PROBABILITY` (config.py, default 0.58 - much lower than the
  hitter picks' 0.77, since single-game MLB win probabilities are
  compressed near 50/50 even for real favorites) - the dashboard highlights
  the flagged games in Today's Picks rather than only ever showing them.
  `game_evaluation.build_game_picks_export`'s accuracy/streak numbers stay
  scoped to the `above_threshold` subset only - a below-threshold game's
  real outcome never moves those, exactly as before this change; only what
  gets *published* got broader.
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

### Game-pick logistic regression (`scripts/train_game_pick_model.py`, fit + report only)

Same two-phase pattern as the hitter-side model above, applied to the
team-level heuristic: `game_picks.build_game_features` exposes the ten raw,
unblended per-team/matchup ingredients (`home_composite`, `away_composite`,
`home`/`away_bullpen_pave_plus`, `home`/`away_bullpen_power_a_plus`,
`home`/`away_starter_pave_plus`, `home`/`away_starter_power_a_plus`) that
`compute_game_win_probabilities` combines into `home_win_probability`,
instead of only ever returning the already-blended ratio -
`compute_game_win_probabilities` itself is refactored to build on top of
`build_game_features` (a behavior-preserving change; all pre-existing
`test_game_picks.py` tests pass unmodified).
`game_picks_backtest.assemble_game_pick_log` replays every historical game
(no-lookahead, one row per `game_pk`, reusing the same as-of-date
`pipeline.compute_outputs` recompute `reconstruct_historical_game_picks_from_persisted`
uses) into a training table of those ten features, the heuristic's own
`home_win_probability` (carried through as a comparison column, not fed to
the model), and the real `Home_Won` outcome.

**Real numbers (2026-07-29, full persisted history, n=1,595 games across
121 dates)**: individually, only `home_starter_pave_plus` (p=0.0256) and
`home_starter_power_a_plus` (p=0.0335) are significant at p<0.05 -
`home_bullpen_pave_plus`/`home_bullpen_power_a_plus` are marginal
(p≈0.06); `home_composite`, `away_composite`, and all four away-side
pitching features are not significant. In the combined multivariate model
nothing clears p<0.05. On a 20-date holdout (n=264), the walk-forward
`LogisticRegression` essentially ties a coin flip (accuracy 0.500, ROC AUC
0.5035, log_loss 0.6937) and does not even beat the naive baseline
(0.6929), let alone the existing `home_win_probability` heuristic (log_loss
0.6874, ROC AUC 0.5603, clearly the stronger signal here) - **not saved**,
reported honestly per the same bar the hitter-side model had to clear.
Unlike the hitter side, this fit-and-report phase found no edge to
capture: the composite/bullpen signals `game_picks.py` already blends
carry real information the raw per-team features alone don't obviously
improve on with this data volume, at least not via a plain logistic
regression on today's feature set.

**Scope**: fit and report only, same as the hitter-side script - no
artifact was produced this run, and even if a future re-run (as more
history accumulates) does clear the bar, the design for actually using it
(nudging the heuristic's favored team's probability slightly toward the
model's pick, calibrated via its own backtest, mirroring how
`GAME_PICK_SUSCEPTIBILITY_WEIGHT` and `GAME_PICK_MIN_PROBABILITY` were both
calibrated rather than guessed) remains a separate, later decision.

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
constant (`config.FIP_CONSTANT = 3.10`, a pure additive shift
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
closest value on that metric (a hand-implemented nearest-neighbor sort,
matching this project's pattern of implementing its own stats rather than
pulling in an ML library for one call - unchanged even though
scikit-learn is now a real dependency for the HR9 follow-up below), then
look at what those comparables actually did on that same
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
both clearly beat their baselines; `HR9`'s single-dimension comparable
search *used to be* reported honestly as a wash - it correlated with
next-season `HR9` (year-to-year home-run rate is notoriously volatile,
driven by batted-ball luck/park effects/defense as much as pitcher skill)
but its MAE was statistically indistinguishable from just guessing the
sample mean. A real ML follow-up fixed that (see "Machine learning
follow-up" below) - `HR9` in the table is now the ML result, and the KNN
comparable search remains what every other metric here uses:

| metric | MAE | naive baseline MAE | correlation | n scored |
|---|---|---|---|---|
| K9 | 1.1476 | 1.5878 | 0.746 | 296/500 |
| BB9 | 0.6277 | 0.7776 | 0.616 | 296/500 |
| HR9 (KNN, superseded) | 0.3312 | 0.3264 | 0.321 | 296/500 |
| HR9 (ML, live) | 0.3190 | 0.3264 | 0.359 | 296/500 |
| FIP | 0.5907 | 0.6606 | 0.440 | 296/500 |

### Machine learning follow-up: HR9

`HR9` was the weakest signal in this project by a clear margin (README's
DFS section below has the other three), so it got a real ML attempt
(`age_curve_ml.py`, `scripts/train_age_curve_hr9_model.py`) rather than
further heuristic tuning: a gradient-boosting regression on all six
features together (`age`, `IP`, `K9`, `BB9`, `HR9`, `FIP`) instead of the
KNN path's single dimension (`HR9`'s own value). Trained ONLY on seasons
before 2010 (16,324 rows, 1871-2009) - a stricter, single-global-model
analogue of the KNN backtest's own no-lookahead discipline (year-blocked
cross-validation, `age_curve_ml.YearBlockedSplit`, picks hyperparameters
using only that pre-2010 pool) - then evaluated once on the exact same
500-season 2010-2019 holdout the KNN number above uses.

The honest expectation going in was that this might simply not be
fixable by any model built from a pitcher's own aggregate rate stats
(single-season HR9 is substantially batted-ball-luck/park/defense-driven).
That turned out to be too pessimistic: the model beat both the naive
baseline and the KNN search, and is now the live projection source for
`HR9` specifically - every other metric is untouched, still served by the
original KNN comparable search. See `config.py`'s Age Curves section for
the model's selected hyperparameters and `age_curve_ml.py`'s module
docstring for the full methodology.

## DFS Player Rankings (`docs/dfs.html`, `dfs.py`)

A standalone page ranking today's hitters and probable starting pitchers
by estimated DraftKings Classic MLB fantasy points. Originally a
**ranked list of good plays, not a salary-cap lineup optimizer** (no
salary data is ingested anywhere in this project, by design - the user's
original request was rankings/projections only). An "Optimal Lineup" tab
has since been added on top of these same projections - see "Optimal
Lineup" below for what changed and, critically, why its salaries are a
MODELED estimate, not real DraftKings prices.

DraftKings' scoring rules (`config.DFS_DK_*`) were confirmed live via web
search, not pulled from memory, with sources cited directly in
`config.py`:
[dknetwork.draftkings.com](https://dknetwork.draftkings.com/2020/05/29/beginner-mlb-dfs-scoring/)
and [draftkings.com/help/rules/2/59](https://www.draftkings.com/help/rules/2/59).
Hitters: single=3, double=5, triple=8, HR=10, run=2, RBI=2, BB=2, HBP=2,
SB=5, and (confirmed current as of July 2026) **no caught-stealing
penalty** - DraftKings removed it from the current ruleset. Pitchers:
2.25 points/inning (0.75/out), K=2, win=4, ER=-2, H=-0.6, BB=-0.6,
HBP=-0.6, plus rare complete-game/shutout/no-hitter bonuses.

### Hitters

DK pays non-linearly for hit type (a double isn't 2x a single's value: 5
!= 2*3), but this project only computes a linear `Expected_Bases` signal
(`hitters.compute_wtb`) - no per-player 1B/2B/3B/HR rate breakdown
exists. `DK_Points_Hitter_HitType` approximates the non-linear scoring
with a single calibrated "DK points per expected total base" constant,
`config.DFS_DK_POINTS_PER_TOTAL_BASE = 2.6998`, computed from **real**
Lahman batting 2015-2025 hit-type shares (not a guessed league average -
see the constant's docstring for the full arithmetic), applied to
`Expected_Bases` after scaling it for today's specific matchup via
`dfs.compute_matchup_adjustment` - a ratio of `Matchup_Hit_Probability`
(today's actual opposing pitcher, platoon+park adjusted) to the batter's
own blended `Game_Hit_Probability`.

**Widened beyond hit types (2026-07-26)**: `DK_Points_Hitter` also scores
walks, hit-by-pitch, and RBI on top of `DK_Points_Hitter_HitType`, via
`hitters.compute_extended_dk_rates` - the same recency-windowed,
platoon-blended architecture `compute_wave`/`compute_whops` already use
(`config.WHOPS_WTB_WINDOWS`), reused rather than inventing a new one.
- **BB/HBP**: trivial from the persisted Statcast `events` column -
  `helpers.is_walk_for_dk_scoring` (a walk OR an intentional walk;
  deliberately a separate classifier from the existing `helpers.is_walk`,
  which several already-validated signals - pitcher BB9, Age Curves BB9,
  the pitcher-side DFS backtest - depend on and which must NOT silently
  change scope) and `helpers.is_hit_by_pitch`.
- **RBI**: not a structured Statcast field, so `helpers.estimate_rbi`
  approximates it as `post_bat_score - bat_score` on the batter's own
  plate-appearance-ending row (both columns exist on every persisted row)
  - the standard Statcast RBI-approximation technique. Two accepted
  simplifications: a bases-loaded walk's forced-in run is credited as a
  "BB" event, not separately as an RBI (DK scores it as both; this
  doesn't), and this can't distinguish a batter's own go-ahead RBI from,
  e.g., a run that scored on the same play for an unrelated reason on rare
  multi-run plays. Clipped at 0 (an RBI can't be negative).

**Still excluded**: runs scored by the batter (would need real multi-PA
baserunner tracking across plate appearances within a game/inning - a
real but deferred v1.1 follow-up, not built) and stolen bases (Statcast
has no structured `events` value for a steal at all - it only appears in
the free-text `des` field, and is attributed to whoever was AT BAT during
the steal, not the runner - parsing that reliably would mean matching on
names, which conflicts with this project's numeric-ID-only join
convention everywhere else; explicitly scope-cut, not planned even for
v1.1).

**Backtest reality check**: widening the scored categories did NOT fix
`DK_Points_Hitter`'s underlying weak-signal problem - see "Machine
learning follow-up" below for the numbers and why the ML model, not this
heuristic, is what's actually live.

**Recency exclusion and position subtabs (2026-07-29)**: `scripts/build_dfs_rankings.py`
now drops any hitter whose most recent completed plate appearance
(`hitters.compute_last_game_dates`) is more than
`config.HITTER_MAX_DAYS_SINCE_LAST_GAME` (5) days before `--as-of-date` -
the same gate `predictions.select_picks` already applies to Beat the
Streak picks, reused directly. Without it, a hitter on a season-long
injured-list stay could still qualify on PA alone and keep getting
selected by the Optimal Lineup optimizer below (a real, live bug: Dansby
Swanson, injured for months, kept appearing). It also merges in each
hitter's DK Classic fielding slot (`dk_slot`, `roster_positions.py` - the
same eligibility lookup the optimizer already used) so the dashboard's
Hitters tab can offer per-position subtabs (All/C/1B/2B/3B/SS/OF)
alongside the existing full ranked list. A failed position-eligibility
fetch is non-fatal here (unlike a failed schedule fetch) - today's
rankings still publish, just with a blank `dk_slot` column and no
per-position filtering that day.

### Pitchers

Needed one new signal this project didn't have: `pitcher_form.py`'s
recency-windowed `K9`/`BB9`/`HR9`/innings-per-start, mirroring
`pitchers.compute_pave`'s windowing pattern but applied to start-level
data via its own separate `config.DFS_PITCHER_WINDOWS` (deliberately not
reusing `PAVE_WINDOWS`, since PAVE blends at-bat-level data while this
blends start-level data and needs wider windows to avoid tiny-sample
buckets). `DK_Points_Pitcher` combines expected innings, strikeouts,
walks, and hits allowed (the last via PAVE scaled by
`config.DFS_BATTERS_FACED_PER_INNING = 4.335`, computed from this
project's own persisted 2026 Statcast) plus an estimated earned-run
penalty. `Expected_ER` is derived from a windowed FIP
(`config.FIP_CONSTANT`, shared with Age Curves) rather than a real ER
signal - this project has a consistent house principle against modeling
ERA/earned runs directly, since they're defense/sequencing-dependent
rather than purely the pitcher's own skill (the same reasoning
`Power_A_PLUS` and Age Curves' pitcher metrics already use FIP over ERA
for).

**Explicitly excluded from v1**: Win (needs a win-probability estimate
too dependent on the pitcher's own team's offense to reasonably
approximate) and the rare discrete bonuses (complete game, shutout,
no-hitter). Relief pitchers are entirely out of scope - a meaningful
"innings per appearance" number only exists for a starter, not a
variable-length bullpen outing. A pitcher needs
`config.DFS_PITCHER_MIN_STARTS` (3) recorded starts and to be a team's
announced probable starter today to be ranked at all - no neutral
fallback for an unannounced pitcher.

### Cadence and pipeline

Unlike Age Curves (weekly - historical comparables barely move day to
day), DFS rankings run **daily**, wired directly into
`daily_update.yml` right after the model step
(`scripts/build_dfs_rankings.py`, writing
`docs/data/dfs_hitters.csv`/`dfs_pitchers.csv`), since matchups and
probable starters change every day and a weekly cadence would serve
stale matchups six days out of seven. A failed probable-pitcher fetch or
an empty schedule leaves the previous day's files untouched rather than
overwriting them with nothing.

`dfs_backtest.py` validates the projections with the same no-lookahead
discipline as `game_picks_backtest.py`/`backtest_age_curve.py`: for each
of a sample of real past game dates, `pipeline.compute_outputs` is
recomputed fresh from an as-of-date slice of persisted Statcast (nothing
after that date is visible), and the resulting projections are compared
against that date's real outcomes.

**Validated against real data**: 15 real game dates in July 2026,
recomputed with no lookahead. Reported honestly, not softened - this is
the ORIGINAL heuristic-only backtest; three of these five signals have
since been superseded by validated ML models (see "Machine learning
follow-up" below), kept here for the historical record:

| signal | MAE | naive baseline MAE | correlation | n scored |
|---|---|---|---|---|
| `DK_Points_Hitter` (heuristic, hit-type only, superseded) | 3.7719 | 3.7496 | -0.004 | 4,156 |
| `Expected_IP` | 1.0257 | 1.0332 | 0.351 | 360 |
| `Expected_K` | 1.8971 | 1.9922 | 0.393 | 360 |
| `Expected_BB` (heuristic, superseded) | 1.0570 | 1.0380 | 0.182 | 360 |
| `Expected_H_Allowed` (heuristic, superseded) | 2.6545 | 1.7860 | 0.223 | 360 |
| `DK_Points_Pitcher` (combined, heuristic) | 6.9409 | 7.1970 | 0.306 | 360 |

The hitter ranking's real-world correlation was essentially zero -
`compute_matchup_adjustment`'s ratio (derived from hit-PROBABILITY
signals but applied multiplicatively to a TOTAL-BASES signal, flagged up
front as the single highest-risk modeling choice in this feature) did
**not** hold up at the single-game level. `Expected_H_Allowed`'s MAE was
actually worse than the naive baseline despite a positive correlation,
suggesting the PAVE-to-hits-allowed scaling was systematically off, not
just noisy. `Expected_IP`/`Expected_K` already beat their baselines and
were left alone - the ML follow-up below only targeted the three flagged
signals.

**Hitter scoring widened (2026-07-26)**: `DK_Points_Hitter` now also
scores BB/HBP/RBI, not just hit types (see `dfs.py`'s module docstring for
exactly what's included and why runs/stolen bases still aren't - no
reliable structured signal exists for either in the persisted Statcast
data). Re-running the same heuristic backtest against 20 recent game dates
with the widened scoring: MAE 4.7569 vs. 4.7457 naive-baseline MAE,
correlation 0.009 (n=5,430 scored). Widening the scored categories did
**not** fix the underlying weak-signal problem - the heuristic's
correlation is still essentially zero, because the flawed
`compute_matchup_adjustment` ratio still drives most of the point total
(`DK_Points_Hitter_HitType`) and the new BB/HBP/RBI terms are additive on
top of it, not a structural fix. The live default for `DK_Points_Hitter`
is the ML model below (trained directly on raw ingredients, bypassing
that ratio), not this heuristic.

### Machine learning follow-up: hitter and pitcher models

`DK_Points_Hitter`, `Expected_H_Allowed`, and `Expected_BB` each got a
real ML attempt (`dfs_ml.py`, `scripts/train_dfs_ml_models.py`) rather
than further heuristic tuning, once the numbers above showed they weren't
working. Trained on the FULL persisted 2026 Statcast history, with
walk-forward blocked cross-validation (`ml_models.WalkForwardDateSplit`)
for grid search and a final holdout (the most recent `config.ML_FINAL_HOLDOUT_DATES`
= 20 dates) the grid search never sees at all - the reported numbers below
come exclusively from that untouched holdout, refit on everything before
it, so they're a fair head-to-head against the original 15-20 date
heuristic numbers.

**Retrained 2026-07-26** after `DK_Points_Hitter`'s scoring widened to
include BB/HBP/RBI (see "Hitters" above) - the old model artifact was
deleted, not left in place, since its feature schema no longer matched
`dfs_ml.HITTER_FEATURE_COLUMNS`:

| signal | model | MAE | naive baseline MAE | heuristic MAE | correlation | n scored |
|---|---|---|---|---|---|---|
| `DK_Points_Hitter` | gradient boosting | 4.6716 | 4.7210 | 4.7569 | 0.162 | 5,642 |
| `Expected_H_Allowed` | Ridge (alpha=30) | 1.7385 | 1.8164 | 2.6241 | 0.302 | 468 |
| `Expected_BB` | Ridge (alpha=0.1) | 1.0233 | 1.0389 | 1.0744 | 0.120 | 468 |

`DK_Points_Hitter`'s MAE is on the wider post-BB/HBP/RBI point scale, so
not directly comparable in absolute terms to the pre-widening figure kept
in git history (3.6374) - but its correlation actually IMPROVED (0.162 vs.
0.145), a genuine gain, not just a like-for-like retrain.
`Expected_H_Allowed`/`Expected_BB`'s own inputs are unaffected by the
hitter-scoring widening - they were retrained only because more persisted
history had accumulated by this date; their small movement versus the
original run reflects that, not the widening itself.

All three beat both the naive baseline and their own prior heuristic, so
all three are now **live** (`dfs_ml.apply_ml_overrides`, called from
`scripts/build_dfs_rankings.py` every day) - a missing/not-yet-trained
model artifact falls back to the original heuristic automatically, so
nothing breaks if the weekly training workflow hasn't run yet. None of
these are strong signals - correlations of 0.12-0.30 mean there's still a
lot of unexplained variance - but each is a real, validated improvement
over both guessing and the prior heuristic, reported honestly rather than
oversold. The hitter model's biggest structural change: instead of
`compute_matchup_adjustment`'s flagged-risky ratio, it's trained directly
on the ratio's raw ingredients (the batter's own `WAVE`/
`Game_Hit_Probability`/`Consistency`/`Approach`, the opposing starter's
and bullpen's `PAVE`, `Park_Factor`, `is_home`, plus `Expected_BB`/
`Expected_HBP`/`Expected_RBI`) via `dfs_ml.build_hitter_features` -
letting the model find its own combination instead of inheriting the
multiplicative assumption that turned out not to hold up.

Model training runs weekly (`.github/workflows/ml_training_update.yml`,
same cadence rationale as Age Curves - re-fitting from full history is
too slow for the daily pipeline), committing `data/models/*.joblib`
alongside the code. Re-run `scripts/train_dfs_ml_models.py` (and update
the numbers above) after any change to the DFS feature set or windowing
constants.

**Retrained again 2026-07-26** after the PAVE bug fix above
(`starter_PAVE`/`Bullpen_PAVE` are direct hitter-model features;
`Expected_H_Allowed` - itself PAVE-derived - is a direct pitcher-model
feature), against the full persisted history (118 hitter dates / 103
pitcher dates):

| signal | model | MAE | naive baseline MAE | heuristic MAE | correlation | n scored |
|---|---|---|---|---|---|---|
| `DK_Points_Hitter` | gradient boosting | 4.6299 | 4.6648 | 4.7156 | 0.161 | 5,760 |
| `Expected_H_Allowed` | Ridge (alpha=30) | 1.7685 | 1.8395 | 1.8032 | 0.297 | 481 |
| `Expected_BB` | Ridge (alpha=0.1) | 1.0141 | 1.0310 | 1.0689 | 0.130 | 481 |

`DK_Points_Hitter`'s numbers are essentially unchanged from the prior
retrain (correlation 0.161 vs. 0.162) - expected, since PAVE isn't the
reason this signal's correlation is weak (see "Hitters" above).
`Expected_H_Allowed` is the interesting one: the PAVE fix alone already
turned its HEURISTIC from worse-than-baseline (MAE 2.6252 pre-fix) into
better-than-baseline (1.8032) on the identical dates - and this retrained
model improves on that fixed heuristic further still (1.7685), a real
additional gain stacked on top of the formula fix, not just the model
recovering ground the bug had cost it. All three again beat both the
naive baseline and their own (now-corrected) heuristic, so all three
stayed live with these updated weights.

**Retrained again 2026-08-19** after `dfs_ml.HITTER_FEATURE_COLUMNS`
widened to add `Exit_Velo`/`Barrel_Rate`/`xBA`/`xwOBA` (see "Real
batted-ball-quality signal" above), forcing a retire-and-retrain of the
two stale artifacts (`dfs_hitter_model.joblib`,
`hitter_hit_probability_model.joblib`) whose schema no longer matched:

| signal | model | MAE | naive baseline MAE | heuristic MAE | correlation | n scored |
|---|---|---|---|---|---|---|
| `DK_Points_Hitter` | gradient boosting | 4.3898 | 4.2960 | 4.3548 | 0.164 | 5,463 |
| `Expected_H_Allowed` | Ridge (alpha=30) | 1.7879 | 1.7959 | 1.9050 | 0.202 | 490 |
| `Expected_BB` | Ridge (alpha=0.1) | 0.9869 | 0.9790 | 0.9841 | 0.194 | 490 |

Reported honestly rather than silently kept as-is: `DK_Points_Hitter`'s
retrained model does **not** beat the naive baseline this run (4.3898 vs.
4.2960) - a reversal from the prior retrain above, where it beat both
bars. `Expected_BB` also misses the baseline (0.9869 vs. 0.9790), same as
several earlier runs. Both fall back to their heuristic automatically
(`ml_models.save_model`'s existing "beats baseline AND heuristic or don't
save" gate - see `dfs.py`'s docstring), so the live DFS Hitters tab keeps
serving the heuristic for these two exactly as it did before either model
ever existed - no crash, no silently-worse predictions shipped. Whether
this miss traces to the four new columns specifically, to a materially
different holdout window (the season has moved forward since the last
retrain, so the dates being fit/scored aren't the same), or plain
retrain-to-retrain noise on a signal whose correlation has hovered
0.16-0.30 the whole time isn't separable from this run alone - it isn't
evidence the new columns are bad, just that they didn't rescue this
particular model this time. `Expected_H_Allowed` (whose feature set
doesn't include the new columns at all) beat both bars again and stayed
live, essentially unaffected as expected. Next weekly retrain will show
whether `DK_Points_Hitter`/`Expected_BB` recover baseline on fresh data.

**Retrained again 2026-08-19** (same day, second run) after
`HITTER_FEATURE_COLUMNS` widened again to add `Fastball_WAVE`/
`Breaking_WAVE`/`Offspeed_WAVE`/`starter_fastball_rate`/
`starter_breaking_rate`/`starter_offspeed_rate` (see "Real
pitch-type-specific platoon matchup" above): `DK_Points_Hitter` MAE
4.3982 (vs. 4.3898 just above, still short of the 4.2960 naive baseline -
still not saved), `Expected_H_Allowed` and `Expected_BB` both essentially
unchanged (1.7879 and 0.9869, identical to just above - expected, since
neither's feature set includes any of the six new columns). No
meaningful movement either direction from the pitch-family columns on
this signal.

**Retrained again 2026-08-19** (same day, third run) after
`HITTER_FEATURE_COLUMNS` widened once more to add `Whiff_Rate`/
`Chase_Rate` (see "Real plate-discipline signal" above): `DK_Points_Hitter`
MAE 4.3806 (vs. 4.3982 just above, still short of the 4.2960 naive
baseline - still not saved, heuristic keeps serving), correlation 0.165
(n=5,463). `Expected_H_Allowed`/`Expected_BB` don't include `Whiff_Rate`/
`Chase_Rate` in their own feature sets and are unaffected by this run.
Consistent with the pattern in the immediately-prior two retrains: these
two hitter-side batted-ball/plate-discipline signals move the hit-
probability classifier (see above) but haven't yet rescued the separate
`DK_Points_Hitter` points-regression model off its naive baseline.

## Optimal Lineup (`docs/dfs.html`'s "Optimal Lineup" tab, `dfs_optimizer.py`)

**`Estimated_Salary` is NOT a real DraftKings price.** DraftKings has no
public API for contest salaries - there is no free, ToS-compliant way to
fetch real ones. Scraping DraftKings' site was ruled out (fragile, likely
against their Terms of Service) and so was a manual daily CSV upload;
instead, on the user's own explicit direction ("build your best guess at
pricing based on performance"), `Estimated_Salary` is a MODELED number
derived entirely from this project's own `DK_Points_Hitter`/
`DK_Points_Pitcher` projections. Never treat it as a real, submittable
DraftKings price, and never assume a lineup built here is a valid budget
on the real platform - a real DFS player could lose real money making
either assumption. This is disclaimed everywhere the number surfaces: the
column is always named `Estimated_Salary` (never bare `Salary`), a
red-bordered warning box sits directly above the Optimal Lineup tab's
table, the rendered column header itself reads "Est. Salary (NOT a real
DK price)" (not relying solely on the box above it), and every module
docstring involved (`estimated_salary.py`, `dfs_optimizer.py`,
`roster_positions.py`) restates it.

**DraftKings Classic MLB roster rules**, confirmed live via web search
(not memory) against
[draftkings.com/help/rules/mlb](https://www.draftkings.com/help/rules/mlb):
10 roster spots - 2 P, C, 1B, 2B, 3B, SS, 3 OF (no FLEX/UTIL slot) - and a
$50,000 salary cap (`config.DFS_ROSTER_SLOTS`/`DFS_SALARY_CAP`). Real DK
salaries always use a $2,000 floor and $100 increments (also confirmed via
that source, `config.DFS_ESTIMATED_SALARY_FLOOR`/`_ROUND_TO`) - the salary
*ceiling* (`DFS_ESTIMATED_SALARY_CEILING`) is NOT from an official DK
table (DraftKings doesn't publish one; prices float algorithmically) -
it's an anecdotal 2026 example from secondary sources, rounded outward.
This floor/increment-are-real-but-ceiling-is-anecdotal distinction is
itself part of the honesty requirement here.

**The formula** (`estimated_salary.py`): linear min-max scaling of a
player's own `DK_Points_Hitter`/`DK_Points_Pitcher` into
[`DFS_ESTIMATED_SALARY_FLOOR`, `DFS_ESTIMATED_SALARY_CEILING`], clipped at
both ends, rounded to the nearest $100.

**Salary $/point parity fix (2026-07-26)**: the original v1 scaled
hitters and pitchers into SEPARATE reference point ranges and SEPARATE
salary ceilings, each independently min-max-scaled to its own group. This
was a real bug, caught by inspecting a real optimizer output: two elite
pitchers alone consumed ~$22K of the $50K cap, leaving the other 8 hitter
slots filled with replacement-level bats. The root cause wasn't that
pitchers are really worth that much more - it's that pitcher DK scoring
naturally spans a much wider raw point range (~2.6-22.75, box-score
categories piling up over 6 innings) than hitters' hit-type-driven
scoring (~2.6-4.7 pre-widening), so independently rescaling each group to
its own ceiling made a pitcher's marginal DK point worth roughly 5x a
hitter's in salary terms - a scaling artifact, not real relative DFS
value. Fixed by collapsing both groups onto ONE shared reference point
range (`DFS_REFERENCE_MIN_POINTS`/`DFS_REFERENCE_MAX_POINTS`) and ONE
shared ceiling (`DFS_ESTIMATED_SALARY_CEILING`) - both
`compute_hitter_estimated_salary`/`compute_pitcher_estimated_salary` now
delegate to the same constants, guaranteeing the same dollars-per-point
rate regardless of position (`test_hitter_and_pitcher_salary_share_same_dollar_per_point_rate`
is a direct regression test for this). A pitcher still costs far more
than a hitter in practice - just because pitchers genuinely PROJECT far
more points, the real reason, not a second hidden rescaling on top of it.

**Acceptance check, against real production `DK_Points` from 2026-07-25**
(`docs/data/dfs_hitters.csv`/`dfs_pitchers.csv`): under the OLD
per-position scale, a hitter projecting exactly 8.0 points priced at
**$6,500** (pinned to the hitter ceiling - 8 points was near the top of
hitters' whole observed range) while a pitcher projecting the SAME 8.0
points priced at only **$4,400** (still cheap relative to pitchers' much
wider range) - the same real point total was worth 48% more salary
depending on which position scored it, purely a scaling artifact. Under
the NEW shared scale, both price identically at **$4,800**. The real
top-2-pitchers-eat-the-budget case also improved: the two highest-scoring
real pitchers that day (22.75 and 22.39 points) combined for $21,800
under the old scale vs. $20,100 under the new one.

The shared reference range is computed from a REAL 20-game-date backtest
sample (`dfs_backtest.backtest_dfs_projections`, the same no-lookahead
recompute every other backtest in this project uses), pooling both
position groups' `DK_Points` together - not guessed, and not a single
day's snapshot like the pre-fix v1 constants were: hitters'
`DK_Points_Hitter` ranged 0.1595-16.6757 (mean 4.827, n=5,430); pitchers'
`DK_Points_Pitcher` ranged 0.1207-25.1412 (mean 11.459, n=468). The
pitcher range fully spans the hitter range on both ends, so the shared
min/max are effectively the pitcher extremes - expected, not a bug, since
pitcher DK scoring genuinely spans a wider range even after the hitter
scoring widened. Under this shared scale, a real average hitter (~4.8
points) now prices around $3,700 and a real average pitcher (~11.5
points) around $6,100 - a plausible, non-degenerate spread. This is still
a single-signal (points only) linear model - real DK pricing also
reflects season-long track record, popularity/ownership effects, and
other signals this project doesn't compute - a deliberate v1
simplification, not hidden.

**Position eligibility** (`roster_positions.py`): a genuinely new data
source for this project - no fielding-position data existed anywhere
before this. Fetched via the MLB Stats API's `people` endpoint
(`statsapi.get("people", {"personIds": ...})`), queried directly by
`key_mlbam` (this project's own player id - no name-matching needed).
Restricted to a player's single PRIMARY position, NOT DraftKings' real
multi-position eligibility (which DK derives from recent multi-position
starts) - a player whose primary position has no DK Classic slot at all
(most commonly "DH", a common primary role for several everyday hitters)
is excluded outright, a real v1 gap, not hidden. The field paths used here
are based on the MLB-StatsAPI package's own implementation and public
documentation, not a live-confirmed response (this sandbox cannot reach
statsapi.mlb.com at all - the same restriction `schedule.py`'s probable-
pitcher fetch already documents) - `scripts/debug_statsapi_positions.py`
(run through the `Debug statsapi` GitHub Actions workflow, real network
access) is the same bootstrapping step `schedule.py`'s own field paths
went through before being trusted in production.

**Injured/inactive hitters (2026-07-29)**: the optimizer's input,
`docs/data/dfs_hitters.csv`, now excludes any hitter who hasn't played
recently (see "Recency exclusion" under DFS Player Rankings' Hitters
section above) - previously nothing here checked *current* activity, only
season-long PA/rates, so a hitter on a long injured-list stay kept getting
optimally selected (a real, live bug: Dansby Swanson). `dfs_hitters.csv`
also now carries its own `dk_slot` column (for the dashboard's Hitters
position subtabs) - `scripts/build_optimal_lineup.py` drops it on read,
before `build_player_pool`'s own independent eligibility fetch/merge runs,
so the two don't collide into `dk_slot_x`/`dk_slot_y`.

**The optimizer** (`dfs_optimizer.py`): an exact MILP (mixed-integer
linear program), solved via [PuLP](https://github.com/coin-or/pulp)'s
bundled CBC solver - a new dependency, added because this is a real
knapsack/assignment problem (choosing 3 of ~80-100 OF-eligible candidates
alone is already a ~117,000-combination choice, before the shared salary-
cap budget across every other slot is considered), where a greedy
best-points-per-dollar pick per slot does NOT guarantee the true optimum.
The slot groups happen to be disjoint in this v1 design (one primary
position per player), which would let a hand-rolled DP solve this without
a new dependency - PuLP was chosen anyway since it directly expresses the
real constraints without careful group-merge bookkeeping, and because it
survives the near-certain follow-up of real DK multi-position eligibility
(which breaks the disjoint-groups assumption a DP would depend on, while
ILP trivially generalizes). A two-way player (appearing in both the
hitter and pitcher pools) is constrained to be selected at most once
across both roles, so the optimizer can never fill two roster slots with
the same real person. Returns no lineup (not a crash) if the slate can't
fill every slot or the cheapest possible full roster exceeds the cap -
`scripts/build_optimal_lineup.py` leaves the previous day's
`optimal_lineup.csv` in place in that case, same resilience pattern as a
failed schedule fetch elsewhere in this project.

`scripts/build_optimal_lineup.py` runs daily (`daily_update.yml`,
immediately after `scripts/build_dfs_rankings.py`), writing
`docs/data/optimal_lineup.csv` (the 10 selected players) and
`docs/data/dfs_salary_pool.csv` (every eligible player considered, for
transparency into what the optimizer actually saw).

### Slate filtering (client-side, `docs/dfs_solver.js`, 2026-07-31)

Real DraftKings contests split today's full slate into sub-contests by
game start time (early/main/night) - the daily batch lineup above always
optimizes across every game today, so it's never actually the optimal
lineup for any single real contest. The Optimal Lineup tab now lets a
user pick which games are in their contest (checkboxes, sorted by start
time) and click Analyze to re-solve the lineup for just those games -
entirely in the browser, using `docs/data/dfs_salary_pool.csv` (already
published daily), no new backend or network call.

**Why a hand-rolled JS solver, not a WASM/JS MILP library**: PapaParse is
the only third-party JS dependency anywhere on this site, matching its
no-build-step philosophy. `dfs_optimizer.py`'s own module docstring
already argues the problem is exactly solvable without PuLP: since every
pool row has exactly one `dk_slot`, the DraftKings roster slots are
disjoint groups, so this is a bounded multiple-choice knapsack per slot
merged over one shared salary budget - and since `Estimated_Salary` is
always a multiple of $100 (`config.DFS_ESTIMATED_SALARY_ROUND_TO`), the
budget dimension has only 501 discrete states at the $50,000 cap. A
per-slot dynamic-programming table (exact, not a heuristic - see
`docs/dfs_solver.js`'s own module docstring for the full algorithm)
merged via a knapsack-of-knapsacks solves a real slate in single-digit
milliseconds. Verified exact against the Python MILP directly: a randomized parity fuzz
test (`tests/test_dfs_solver_js.py`, 31 cases: 20 random pools, 10 with a
`min_salary` floor, 1 with a two-way-player collision) compares
`docs/dfs_solver.js`'s `solveOptimalLineupDP` against
`dfs_optimizer.solve_optimal_lineup` on the same random pools (including a
`min_salary` floor and a two-way-player collision) and asserts the two
exact solvers agree on both feasibility and the achievable objective
total - plus a direct production check: "select every game, objective
mean" reproduces the real `optimal_lineup.csv` exactly on live data.

The one thing the DP can't express natively is `dfs_optimizer.py`'s
cross-group "at most once per player" constraint for a true two-way
player (present in both the hitter and pitcher pools). Handled by solving
once per combination of "which single role stays available" when
duplicates exist (still exact, since each combination is itself an exact
solve and the real pool essentially never has more than one two-way
player) - more than 3 simultaneous duplicates (should not happen in
practice) falls back to keeping each duplicated player's higher-value row
and flags the result as approximate rather than an exponential blow-up.
**Known v1 limitation, matching `roster_positions.py`'s own documented
gap**: this DP depends on the disjoint-slot-groups property - if this
project ever implements real DraftKings multi-position eligibility (a
player legally fillable at more than one slot), the Python MILP already
generalizes to that case for free, while this DP would need to be
revisited.

Each game's real start time (`game_datetime`, threaded from
`schedule.normalize_schedule` through `dfs.py`'s
`HITTER_DFS_COLUMNS`/`PITCHER_DFS_COLUMNS` into `dfs_salary_pool.csv`) is
the raw statsapi `gameDate` field - confirmed live via the `Debug
statsapi` GitHub Actions workflow (run 30661418977, 2026-07-31): a bare,
unconditional sibling key of `gamePk`/`teams`/`status` on every raw game
dict, no extra hydrate needed, same as every other field path
`schedule.py` already trusts in production.

### Ceiling / volatility signal (`dfs_ceiling.py`)

GPP (tournament) DFS lineups are won by boom/spike-game players, not
players who reliably score near their own mean - a real, well-known DFS
strategy concept the user raised directly ("the winner won't have players
getting 5 points across their lineup, they're more likely to have lucked
into players averaging 15 or so"). Every signal in this project up to
this point was a MEAN projection; nothing measured upside.
`Ceiling_DK_Points` is the `config.DFS_CEILING_PERCENTILE`-th percentile
(90th) of a player's own REAL historical modeled DK points
(`dfs_backtest.compute_actual_hitter_dk_points`/
`compute_actual_pitcher_dk_points` applied per real game date they
played) - a genuine outcome-history signal, not a new projection model. A
player with fewer than `config.DFS_CEILING_MIN_GAMES` (10) real scored
games falls back to the group-wide (all hitters', or all pitchers')
percentile instead of a noisy small-sample per-player number
(`Ceiling_Source` marks which applies). Merged into `dfs_hitters.csv`/
`dfs_pitchers.csv` as an ADDITIONAL informational column, never replacing
`DK_Points_Hitter`/`DK_Points_Pitcher`.

**Backtested before trusting it** (`dfs_ceiling.backtest_ceiling_signal`,
same 20-date no-lookahead sample as the heuristic DFS backtest,
`Ceiling_DK_Points` computed from ONLY history strictly before each test
date): does ranking by ceiling actually surface real boom-game days
better than ranking by the existing mean projection? Measured as a
"capture rate" - of the player-days that ACTUALLY landed in that date's
real top decile, what fraction were ALSO top-decile by `Ceiling_DK_Points`
going in, vs. by the mean projection instead:

| player type | correlation | real top-decile days | ceiling capture rate | mean-projection capture rate |
|---|---|---|---|---|
| Hitters | 0.127 | 543 (n=5,430) | 19.3% | 10.1% |
| Pitchers | 0.313 | 47 (n=468) | 23.4% | 27.7% |

For hitters, ceiling-ranking is genuinely ~2x better than the mean
projection at flagging real boom days in advance (the mean projection is
barely above the ~10% base rate - `DK_Points_Hitter` has almost no power
to predict WHICH day a hitter booms). For pitchers, there's no clear
edge - the mean projection actually did slightly better on this (small,
n=47) sample. **Reported as a mixed result, not oversold**: this is a
validated upside signal for hitters specifically, not a blanket win.

Because of that mixed result, `Ceiling_DK_Points` ships
**informational-only** - the Optimal Lineup optimizer still defaults to
maximizing the mean projection. Pass `--objective ceiling` to
`scripts/build_optimal_lineup.py` (`dfs_optimizer.solve_optimal_lineup`'s
new `objective_column` parameter) to build a ceiling-maximizing lineup
instead - opt-in, not the default, until real backtest evidence supports
defaulting to it (which it does not yet, especially for pitchers). A
player with no `Ceiling_DK_Points` at all (no real scored history yet)
falls back to their own mean projection in the optimizer's pool, so
they're never silently unselectable under the ceiling objective.

**Explicit v1 scope cuts**: no real DK multi-position eligibility (single
primary position only - full-time DH players are excluded entirely); no
multi-lineup generation, no GPP-vs-cash-game strategy, no ownership-
projection/game-theory diversification (one single "best" lineup per
day); no live salary scraping or manual upload path (both explicitly
rejected in favor of the modeled-estimate approach); relief pitchers
remain out of scope (inherited from `dfs.py`'s own probable-starters-only
restriction). Salary-accuracy backtesting is also explicitly out of scope
- there's no real DraftKings price to validate `Estimated_Salary` against,
unlike every other backtested signal in this project.

### Boom-Adjusted DK Points (`dfs_ceiling.py`)

The user's own framing, verbatim: "player A scores 5 points literally
every night... player B sometimes scores 15, sometimes 5, sometimes 20,
and even sometimes 0, and he thus averages 4.8 - I'd rather have player B
than player A," followed by an explicit rejection of both extremes: "I
don't want straight boom or bust nor straight mean. Neither is
optimizing." Pure `Ceiling_DK_Points` (a single 90th-percentile game) is
exactly the "straight boom" case it rejects - it credits raw upside
regardless of how OFTEN a player actually reaches it. The plain mean is
the other rejected extreme.

`Boom_Adjusted_DK_Points = mean_projection + k * Upside_Deviation`, where
`Upside_Deviation` is a player's own real historical UPSIDE-ONLY
semi-deviation (`dfs_ceiling.compute_upside_deviation`): `sqrt(mean((x -
their_own_historical_mean).clip(lower=0) ** 2))` over ALL their real
games, not just the boom ones. Below-average games contribute exactly
0 - this is deliberately a semi-deviation, not plain standard deviation,
because plain stdev would reward bust-heavy inconsistency exactly as much
as boom-heavy inconsistency, the opposite of the point. Dividing by the
FULL game count (not just the boom subset) means a player who booms
rarely scores lower than one who booms often at similar magnitude -
real DFS value tracks boom FREQUENCY, not just size, which is exactly
what distinguished player B from a player who got lucky once.

**k was backtested, not guessed** (`dfs_ceiling.backtest_boom_adjusted_signal`,
same 20-date no-lookahead sample and capture-rate methodology as
`Ceiling_DK_Points`'s own backtest, run once per candidate k):

| player type | k=0.0 (mean) | k=1.0 (chosen, hitters) | k=2.0 (grid max) |
|---|---|---|---|
| Hitters correlation | 0.009 | 0.044 | 0.061 |
| Hitters capture rate | 10.1% | 12.5% | 13.3% |
| Pitchers correlation | 0.337 | 0.339 (k=0.75 peak) | 0.328 |
| Pitchers capture rate | 27.7% | 27.7% | 25.5% |

For hitters, correlation and capture rate rose monotonically across the
ENTIRE tested grid (0.0 through 2.0) without ever turning over - the grid
didn't localize a true peak. The grid-max k=2.0 was deliberately NOT
chosen anyway: a real check of magnitude (mean historical
`Upside_Deviation` 4.417 vs. mean `DK_Points_Hitter` 5.185) shows that at
k=2.0 the volatility term outweighs the mean term by nearly 2:1 - which
just recreates pure ceiling-chasing under a different name, the opposite
of what was asked for. **k=1.0** was chosen instead: it keeps the two
terms roughly balanced (mean contributes ~54% of a typical score,
deviation ~46%) while still capturing a real, validated improvement over
the plain mean (capture rate +24% relative). For pitchers, correlation
stayed flat and capture rate bounced non-monotonically across the whole
grid with no k reliably beating the k=0 baseline - the same conclusion
`Ceiling_DK_Points` reached for pitchers. **k=1.0 for hitters, k=0.0
(plain mean) for pitchers** (`config.DFS_BOOM_ADJUSTED_K_HITTER`/
`_PITCHER`).

Same shipping posture as `Ceiling_DK_Points`: an ADDITIONAL informational
column, not a default. Pass `--objective boom` to
`scripts/build_optimal_lineup.py` to build a lineup that maximizes
`Boom_Adjusted_DK_Points` instead of the plain mean - a genuine middle
ground between the "mean" and "ceiling" objectives, backed by the
balance/backtest reasoning above rather than picked arbitrarily.

### Matchup-aware boom score (`Matchup_Boom_Score`, hitters only)

Real evidence from an actual DraftKings contest surfaced the next gap:
the winning lineup spent $17,500 on its 2 pitchers and second place
$18,100, vs. this project's own suggested lineup spending over $21,000 -
and even after recalibrating the salary model with real backtest data
(see "Salary $/point parity fix" above), the optimizer's mean-projection
total for a real slate came out to only ~81 points, far below what
winning lineups actually need. The user's own accounting of what a
winning lineup needs: pitchers contribute roughly 50-60 points combined
(achievable from mean projections alone); the other ~100 points need to
come from 8 position players, and since only a handful of hitters
project anywhere near the ~12-point average that would require, **a few
specific hitters need to boom** - both the real 1st- and 2nd-place
lineups had 3 players combine for roughly 60 points between them. The
ask: predict WHICH hitters are most likely to boom today, based on their
matchup, not just who has boomed historically.

`Ceiling_DK_Points` and `Boom_Adjusted_DK_Points` are both matchup-BLIND
on their volatility side - only the mean term reads today's opponent.
`Matchup_Boom_Score` fixes that:

- `compute_boom_threshold`: a single FIXED, GROUP-WIDE point value (the
  `config.DFS_CEILING_PERCENTILE`-th percentile pooled across every real
  hitter-game, not each player's own percentile). A per-player threshold
  would be tautological here - by construction, roughly 10% of ANY
  player's own games clear their OWN 90th percentile, so every player's
  "boom rate" against their own bar would converge to about the same
  number regardless of how boom-prone they really are.
- `compute_boom_rate`: each player's real historical rate of clearing
  that SHARED bar - now players genuinely differ (a boom-prone hitter
  clears it far more than a steady one). Small-sample players (fewer
  than `config.DFS_CEILING_MIN_GAMES` real games) fall back to the
  group-wide clear rate.
- `compute_matchup_boom_score`: `Boom_Rate * Matchup_Ratio` - today's
  real opposing-pitcher/park-adjusted matchup signal
  (`dfs.compute_matchup_adjustment`'s output, already computed as part
  of `DK_Points_Hitter`) scales the player's own boom frequency up or
  down for today specifically. This is a RANKING signal, not a
  calibrated probability, despite "boom" in the name - never treat it as
  P(boom) in a literal sense.

**Backtested** (`dfs_ceiling.backtest_matchup_boom_signal`, same 20-date
no-lookahead sample, hitters only, n=5,430, 689 real boom-days against an
average no-lookahead threshold of ~14.0 points) - capture rate of those
689 real boom days, i.e. what fraction were flagged in advance by each
signal's own top decile:

| signal | capture rate |
|---|---|
| Mean projection (`DK_Points_Hitter`) alone | 10.6% |
| `Matchup_Boom_Score` (boom rate x today's matchup) | 14.1% |
| `Boom_Rate` alone (no matchup adjustment at all) | **17.3%** |

**Honest, somewhat surprising result: the matchup adjustment made it
WORSE, not better.** `Boom_Rate` alone - a player's plain historical
frequency of clearing the shared bar, with no read on today's opponent at
all - beat the matchup-multiplied version by a real margin (17.3% vs.
14.1%), and both comfortably beat the mean projection. Multiplying by
`Matchup_Ratio` added noise rather than signal here - consistent with
`Matchup_Ratio` already being flagged elsewhere in this project (see
`dfs.py`'s module docstring) as the single highest-risk modeling choice
in the whole DFS feature set, and already SUPERSEDED by an ML model for
the main hitter projection specifically because the heuristic ratio
didn't hold up. The same weak signal dragging down `DK_Points_Hitter`
drags down `Matchup_Boom_Score` too when multiplied in.

Because `Boom_Rate` is the real, validated win here, it's exposed as its
own column (not just an internal ingredient of `Matchup_Boom_Score`) -
use `Boom_Rate` to identify genuinely boom-prone hitters; treat
`Matchup_Boom_Score` as informational/exploratory only, not as an
improvement over it. This is reported plainly rather than reframed as a
win, matching how this project has always handled a backtest that didn't
confirm the hypothesis (see Age Curves HR9's KNN-vs-ML precedent,
Ceiling/Boom-Adjusted's pitcher non-results above).

**Re-checked after the PAVE bug fix above** (`Matchup_Ratio` reads
`Matchup_Hit_Probability`, a direct PAVE consumer; `Boom_Rate` itself is
historical-outcome-only and unaffected): n=5,547/675 real boom-days (a
slightly larger accumulated sample than the original 5,430/689, not a
strict same-day A/B) - mean projection 10.8%, `Matchup_Boom_Score` 14.5%,
`Boom_Rate` alone 17.6%. The ordering conclusion is UNCHANGED - `Boom_Rate`
alone still clearly beats `Matchup_Boom_Score` - with a small, consistent
uptick across all three from the corrected matchup input. The fix didn't
overturn this finding; `Matchup_Ratio` remains a weak signal for
reasons unrelated to the PAVE bug (see above).

**Hitters only** - pitchers have no `Matchup_Ratio` analog in this
project, and the earlier Ceiling/Boom-Adjusted backtests already found no
real upside-signal edge for pitchers, so this wasn't extended to a
foundation that had already shown no signal. Not wired into the
optimizer's `--objective` flag either (a full-lineup objective needs a
value for every roster slot including pitchers, which this doesn't
have) - it's a pure informational column for identifying which hitters to
prioritize, matching how it was actually asked for.

### Opponent offense adjustment (`Opponent_Offense_Ratio`, pitchers only, `pitcher_matchup.py`, 2026-07-31)

A second, distinct matchup gap, this one user-observed rather than
backtest-discovered: "matchup doesn't seem to be taken into account as
much as I'd like. It's choosing the same players for the most part
despite who they're facing. Does [a pitcher] have great boom potential
for the price? Yes. Will he boom off of [a tough offense]? Almost
certainly no." Checking the actual code confirmed the gap was real and,
unlike `Matchup_Boom_Score` above, genuinely untested: `dfs.
compute_pitcher_dk_points` had **no opponent-quality signal anywhere**,
not even in the base mean projection - `K9`/`BB9`/`HR9`/`IP_per_start`
are all windowed averages of the pitcher's OWN recent form. A pitcher's
projection was identical whether facing a last-place offense or a
first-place one.

`teams.compute_offensive_edge` already computes real, rolling
bases-scored-per-game per team (`team_bases_pg`), but its own
`offensive_edge`/`true_power` outputs net that out against whichever
opponent a team's OWN most recent game happened to be against - not
today's actual matchup - so both are contaminated for this use.
`team_bases_pg` itself has no opponent term at all and is now exposed
directly from `compute_offensive_edge` as the correct building block.

`pitcher_matchup.py`:

- `compute_opponent_offense_ratio`: today's opponent's `team_bases_pg`
  divided by the league average, blended toward a neutral 1.0 by a
  `weight` and clipped to `config.PITCHER_MATCHUP_OFFENSE_CLIP` (so one
  extreme-outlier offense can't blow up a projection). `weight=0.0`
  returns exactly 1.0 for every row - the built-in null hypothesis that
  reproduces today's unadjusted heuristic exactly, not an approximation
  of it.
- `attach_opponent_offense` / `compute_opponent_adjusted_pitcher_points`:
  scales `Expected_H_Allowed`/`Expected_ER` by the ratio and recomputes
  `DK_Points_Pitcher` from the scaled components. `Expected_K`/
  `Expected_BB`/`Expected_IP` are left unadjusted for v1 - a tougher
  offense plausibly affects those too, but there's no existing
  per-opponent signal for either yet, and scaling the two categories a
  bases-scored rate most directly predicts is the smallest, most
  defensible first step.

**Backtested** (`pitcher_matchup.backtest_pitcher_matchup_signal`, real
persisted Statcast, `--days 90` - the pitcher sample is much smaller than
the hitter-side backtests above, dozens of probable starters per date vs.
thousands of plate appearances, so this needed a longer window than the
20-date default used elsewhere in this project). n=2,081 real
pitcher-days, correlation and MAE of the adjusted `DK_Points_Pitcher`
against that date's REAL `Actual_DK_Points_Modeled`, per weight in
`config.PITCHER_MATCHUP_WEIGHT_GRID`:

| weight | correlation | MAE |
|---|---|---|
| 0.00 (baseline, no adjustment) | 0.3397 | 6.6736 |
| 0.25 | 0.3407 | 6.6658 |
| 0.50 | 0.3414 | 6.6612 |
| 0.75 | 0.3416 | 6.6605 |
| 1.00 (full, unblended ratio) | **0.3417** | **6.6600** |

**Honest result: the right direction, too small a magnitude.** Unlike
`Matchup_Boom_Score` above (which went backwards), this improves
correlation and lowers MAE monotonically across the entire grid - genuine
evidence the adjustment points the right way. But the size of the win is
tiny: weight=1.0 vs. weight=0.0 is only a 0.6% relative correlation
improvement and a 0.2% relative MAE improvement, nowhere near the real
margins that justified this project's other nonzero defaults (e.g.
`Boom_Adjusted_DK_Points`' `k=1.0` needed a 24% relative capture-rate
improvement over `k=0.0` to be chosen). At n=2,081 this small a gap isn't
distinguishable from noise with real confidence.

`config.PITCHER_MATCHUP_OFFENSE_WEIGHT` stays **0.0** for exactly that
reason - `DK_Points_Pitcher` is unchanged from before this module.
`Opponent_Offense_Ratio` still ships as an **informational-only column**
on the pitcher table (`scripts/build_dfs_rankings.py` calls
`attach_opponent_offense` with the weight-0.0 default, which is a no-op
ratio of exactly 1.0 today, but keeps the mechanism ready to flip on
without a code change if a future, larger-sample backtest clears the
bar) rather than being silently dropped - the DIRECTION is real evidence
worth showing, even though the MAGNITUDE doesn't clear this project's
bar for changing the live projection.

### Value_Score: "stars, not superstars" (`dfs_optimizer.py --objective value`)

None of the fixes above (salary parity, `Boom_Adjusted_DK_Points`,
`Matchup_Boom_Score`) actually solved the real complaint. Real evidence,
reported directly: on an actual slate, `mean`/`ceiling`/`boom` all
independently converged on the SAME two most expensive pitchers, spending
$19,900-$21,200 of the $50,000 cap on pitching and leaving barely enough
room for 8 undifferentiated floor-priced hitters. The user's diagnosis,
verified against the numbers rather than taken on faith: `Estimated_Salary`'s
fixed $2,000 floor (see "Salary $/point parity fix" above) is a much
smaller fraction of an elite player's price than a replacement-level
player's, so ANY high scorer - regardless of position - gets a
structurally better AVERAGE dollars-per-point rate purely from that floor
dilution, even though the parity fix already equalized the MARGINAL rate.
An objective that maximizes raw point totals under a budget will always
rationally chase that average-rate advantage and overpay for the 1-2
biggest scorers, leaving nothing to build a real roster - a mix of
reliable "consistent" floor plays and genuine "boom" upside plays
("consistent players carry their own, boom players pick up slack," in the
user's own framing) - with what's left. This is a roster-CONSTRUCTION
problem, not a single-day boom-prediction problem, so it's a different
question from the capture-rate backtests above - **implemented per
explicit user direction ("implement no matter the test"), not validated
against a capture-rate backtest the way `Ceiling_DK_Points`/`Boom_Rate`
were.**

`Value_Score = boom-adjusted points / (Estimated_Salary / 1000)` -
directly rewards being UNDERPRICED relative to real upside (a "star")
over being already fully priced-in (a "superstar"), computed for BOTH
hitters (reusing their own validated `Boom_Adjusted_DK_Points`, k=1.0) and
pitchers (a separate boom-adjusted computation using
`config.DFS_VALUE_BOOM_K_PITCHER = 1.0` - deliberately NOT the same as
`DFS_BOOM_ADJUSTED_K_PITCHER`'s validated 0.0, which answers the
different, narrower question of predicting which day a pitcher booms).

**A second real flaw, found via the same honest sanity check that shipped
this feature**: maximizing a per-dollar RATIO carries no pressure to
spend near the cap. Tested against real production data
(`docs/data/dfs_salary_pool.csv` recomputed with real `Upside_Deviation`):
`objective=value` with no floor picked a full legal lineup for only
$39,700 of the $50,000 cap - total `DK_Points` fell from 81.38 (mean
objective, same slate) to 56.31, because leaving $10,300 unspent doesn't
cost the ratio objective anything. Fixed with a `min_salary` floor
constraint on the same MILP (`solve_optimal_lineup`'s new `min_salary`
parameter: `sum(Estimated_Salary) >= min_salary`), set to
`config.DFS_VALUE_MIN_SALARY_FRACTION` (0.85) of the cap - chosen by
testing three fractions against the same real pool:

| min_salary fraction | total salary spent | total DK_Points | pitcher salary |
|---|---|---|---|
| 0.85 ($42,500) | $43,900 | 67.99 | $16,900 |
| 0.90 ($45,000) | $46,000 | 73.66 | $19,000 |
| 0.95 ($47,500) | $47,500 | 77.79 | $19,900 |

0.95 defeats the whole point - pitcher spend creeps back to the same
$19,900 this feature exists to avoid. 0.85 keeps pitcher spend near the
real winning-lineup range ($17,500-$18,100 in the contest that originally
motivated this work) while still using most of the budget - the best
balance of the three tested, not a formally optimized value.
`scripts/build_optimal_lineup.py` passes this floor only for
`--objective value`; `mean`/`ceiling`/`boom` are unchanged (no floor).

**`value` is now the daily production default** (`daily_update.yml` runs
`python scripts/build_optimal_lineup.py --objective value`), unlike
`ceiling`/`boom` which stay opt-in-only behind the flag. This is a
deliberate exception to this project's usual "don't default to an
unvalidated signal" rule: `mean`'s real-world failure mode (chronic
pitcher overspend) is exactly what motivated this whole feature, so
leaving `value` opt-in would mean the live "Optimal Lineup" tab kept
showing the broken behavior even after building the fix.
`dfs_optimizer.solve_optimal_lineup`'s own function default stays
`"DK_Points"` (mean) for library/test backward compatibility - only the
daily workflow's invocation changed.

## NFL DFS (`docs/nfl.html`, in progress)

An NFL analog of the MLB DFS pipeline above, mirroring its two core
ideas - rolling windows (not season averages) for player form, and
matchup analysis (opponent defense quality) - adapted for DraftKings NFL
Classic contests. Built incrementally; this section fills in with real
numbers/behavior as each phase lands.

Data source is `nflreadpy` (the actively-maintained nflverse successor to
the now-deprecated `nfl_data_py` - see its GitHub README for the
deprecation notice). Real field names/shapes referenced anywhere in this
project's NFL modules are confirmed live via
`scripts/debug_nfl_data.py`/`.github/workflows/debug_nfl_data.yml`, not
assumed - see `nfl_data.py`'s module docstring for the specific run cited.

Raw per-season tables (`weekly`, `schedules`, `injuries`,
`rosters_weekly`, `team_stats`, `snap_counts`) persist to
`data/raw/nfl/*.parquet` via `nfl_data.py`, one file per table per season
- the current season's file is overwritten wholesale on each fetch
(nflverse retroactively corrects stats, so an append-and-dedupe pattern
isn't safe), while completed historical seasons are fetched once via
`scripts/fetch_nfl_historical.py` (`config.NFL_HISTORICAL_SEASONS`) and
left alone. `snap_counts` is keyed by a real Pro-Football-Reference id
(`pfr_player_id`), a different id space than every other table's GSIS
`player_id` - crossed over via `rosters_weekly`'s own real `gsis_id`/
`pfr_id` columns wherever it's used (see `nfl_bestball.compute_player_snap_share`).

Player form is a **games-back rolling blend** (`nfl_passing.py` for QBs,
`nfl_rush_rec.py` for RB/WR/TE together, since both share DK's FLEX pool)
across `config.NFL_QB_WINDOWS`/`NFL_SKILL_WINDOWS` - games-back, not
day-count, unlike the MLB pipeline's `WAVE_WINDOWS` (see
`config.NFL_QB_WINDOWS`'s docstring for why NFL's bye weeks make a
calendar-count window the wrong shape). **Window weights are an
unvalidated first-pass placeholder** - no NFL backtest exists yet to
calibrate them against (see the Backtesting phase below, not yet built).

Matchup adjustment (`nfl_teams.py`/`nfl_matchup.py`) is a direct
structural port of `pitcher_matchup.py`'s opponent-offense ratio, applied
in the opposite direction: a QB/RB/WR/TE's projection is scaled by the
opponent DEFENSE's real recent pass/rush-yards-allowed rate
(`nfl_teams.compute_defense_rolling_rates`, derived from `weekly`'s own
`opponent_team` column - no separate schedule join needed for "what was
allowed"), looked up via that player's **upcoming** opponent from the
current week's schedule (`nfl_matchup.attach_matchup_adjustment`) -
never a defense's own last-played opponent, the same leakage bug class
`teams.compute_offensive_edge` had to be split apart to avoid for MLB.
Like `PITCHER_MATCHUP_OFFENSE_WEIGHT`, `config.NFL_MATCHUP_WEIGHT`
**defaults to 0.0** (informational-only) until a real NFL backtest earns
a nonzero live weight.

DK scoring (`nfl_dfs.py` for QB/RB/WR/TE, `nfl_dst.py` for DST) uses
DraftKings' real NFL Classic scoring rules, confirmed live via web
search against DraftKings' own published rules (not from memory):
0.04 pt/passing yard, 4 pts/passing TD, -1 pt/interception thrown, 0.1
pt/rushing or receiving yard, 6 pts/rushing or receiving TD, 1 pt/
reception (full PPR), -1 pt/fumble lost, 2 pts/2-point conversion, +3
bonus for 100+ rushing OR receiving yards in a game (scored separately -
a player can clear both), +3 bonus for 300+ passing yards in a game.
DST: 1 pt/sack, 2 pts/interception, 2 pts/fumble recovery, 2 pts/safety,
2 pts/blocked kick, 6 pts/defensive or return TD, plus a real non-linear
points-allowed bucket table (0 pts allowed = +10, down to 35+ pts
allowed = -4). DK NFL Classic uses a 9-slot roster (QB/RB/RB/WR/WR/WR/
TE/FLEX/DST) and a $50,000 salary cap - same real cap MLB Classic uses,
confirmed independently, not assumed just because MLB happens to match.

The 100+/300+ yardage bonuses are step functions, not linear in a
windowed mean - v1 scores them as an expected value (each player's own
historical rate of clearing the threshold, blended across the same
games-back windows as the rest of their projection), the same pattern
`hitters.compute_wave` already uses to turn a hit rate into a
probability. Flagged unvalidated pending a real NFL backtest (Phase 7).

`nfl_estimated_salary.py` is a direct structural port of
`estimated_salary.py` - same shared reference-point-range/floor/ceiling/
$100-round-to approach, same "never a real DraftKings price, always
`Estimated_Salary`" disclaimer. Reference range is computed from real
2025-season DK_Points_QB/DK_Points_Skill/DK_Points_DST output, not
guessed (see `config.py`'s NFL Estimated Salary section for the exact
numbers).

`nfl_roster_positions.py` maps `position` directly to a DK Classic slot
(QB/RB/WR/TE) - simpler than MLB's `roster_positions.py`, which needs its
own MLB Stats API fetch since Statcast carries no fielding-position data
at all; nflreadpy's tables already carry `position` directly. Every
RB/WR/TE gets TWO eligibility rows (their own slot AND `"FLEX"`) so
Phase 6's optimizer can handle FLEX with zero new constraint types.

The FLEX-slot optimizer needs almost no new logic. `nfl_dfs_optimizer.py`
reuses `dfs_optimizer.solve_optimal_lineup` UNMODIFIED - its existing
"cap any duplicated `key_mlbam` group at <= 1 selected row" MILP
constraint (originally built for the rare MLB two-way-player edge case)
now handles FLEX for real, since every RB/WR/TE gets two pool rows (own
slot + `"FLEX"`, same identity) from `nfl_roster_positions.py`. The
client-side `docs/nfl_dfs_solver.js` (a JS dynamic-programming
re-implementation, mirroring `docs/dfs_solver.js`) can't express that
same cross-group constraint directly, so it instead solves the ordinary
NO-FLEX problem exactly 3 times - once per hypothesis of which position
absorbs the extra slot (`{RB:3,WR:3,TE:1}` / `{RB:2,WR:4,TE:1}` /
`{RB:2,WR:3,TE:2}`) - and keeps the best, an exact (not approximate)
reduction since DK scores a FLEX RB identically to an RB-slot RB. Both
solvers verified against the same hand-constructed pool with a known-
by-construction optimum (`tests/test_nfl_dfs_optimizer.py` and
`docs/nfl_dfs_solver.test.js`), wired into CI alongside the MLB solver's
own Node tests.

**Real backtest results** (`nfl_dfs_backtest.py`, `scripts/backtest_nfl_dfs_rankings.py`,
no-lookahead, against the full real backfilled 2016-2025 history in
`data/raw/nfl/`): reported honestly either way, same standard every other
signal in this project is held to.

- **QB**: MAE 6.66 vs. a naive "always predict the sample mean" baseline's
  7.82 (14.8% better), correlation 0.514, n=6,358.
- **Skill (RB/WR/TE)**: MAE 4.68 vs. naive baseline's 6.18 (24.2% better),
  correlation 0.585, n=50,988.
- **DST**: MAE 4.72 vs. naive baseline's 4.63 - **worse than naive**,
  correlation 0.101 (essentially no signal), n=5,490.

QB/Skill also beat a simpler "flat, unweighted season-average" heuristic
(real but modest margins) - real evidence the RECENCY-WEIGHTING mechanism
itself adds value, not just "having a player-form signal at all." This
validates Phase 2's windowing MECHANISM; it does NOT validate the
specific window weight VALUES (still an unrecalibrated placeholder - see
`config.NFL_QB_WINDOWS`'s docstring).

**DST is an honest negative result** - `nfl_dst.py`'s points-allowed-
bucket-via-windowed-mean approximation doesn't beat guessing. `DST_Points`
still ships (the optimizer/roster need it structurally), but should be
treated as unvalidated/weak, not a trustworthy signal - see
`config.py`'s NFL Backtesting section for the full numbers and reasoning.

Not yet built: the weekly-cadence pipeline/workflows for the in-season
DFS Optimal Lineup subtab (needs live-season data this project doesn't
have yet).

### Bestball Draft Strategy (`docs/nfl.html`'s Preseason subtab, `nfl_bestball.py`)

**Its own visual theme.** `docs/style.css` is shared across every page, but
`docs/nfl.html` overrides the brand accent tokens (`--accent`,
`--accent-strong`, `--on-accent`) in its own inline `<style>` block -
football green instead of the baseball-orange every other page uses - and
swaps the header/favicon brand mark from the baseball-seam glyph to a
football-and-laces glyph. Scoped to this one file only (the shared
stylesheet and every other page's markup are untouched), so it's a purely
additive, page-local override rather than a new theming system.

Real preseason bestball drafts don't need a forward-looking weekly
projection - they need "how much value/risk did this player represent
LAST season, as a whole." `nfl_bestball.py` answers that with a real,
realized season total rather than a rolling-window projection: it reuses
`nfl_dfs_backtest.py`'s `compute_actual_qb_dk_points`/
`compute_actual_skill_dk_points` (already-validated real full-PPR DK
scoring from real box scores, see the backtest numbers above) summed
across every real regular-season week, for QB/RB/WR/TE (DST excluded -
bestball drafting doesn't need DST optimization the way weekly DFS does).

**Games played is the injury-history signal** - deliberately simple and
honest, not a full medical history: real games played vs. that player's
team's real regular-season game count (16 games through 2020, 17 from
2021 on - always read from the real schedule, never hardcoded), plus the
same figure for the prior season for a cheap repeat-injury-risk read.
`dk_points_per_game` and `games_missed` are kept as separate columns
rather than blended into one "draft score" - there's no backtestable
ground truth for what the right health/talent tradeoff weighting would
even be, so this project's "show raw signals honestly, don't blend
without a real backtest to justify it" standard applies here too.

**Real season-total offensive snap share** (`season_snap_share`,
`nfl_bestball.compute_player_snap_share`, from `nfl_data.fetch_snap_counts`
- real Pro-Football-Reference-sourced per-game snap counts via
`nflreadpy.load_snap_counts`) is also published on the rankings table -
a real playing-time-ROLE signal, distinct from games played (availability)
and points-per-game (production rate). This is a real SEASON-TOTAL share
(a player's real total offensive snaps that season, divided by their
team's real total offensive snaps that season - both summed across every
real game, not just the games the player themselves appeared in), not a
per-game average - a per-game average would let a real one-or-two-game
emergency spot start at a high per-game rate look identical to a real
every-week starter, exactly the small-sample problem a playing-time
signal needs to avoid. A team's real total offensive snaps for one game
is read as the max real `offense_snaps` among that team's players that
game (confirmed live: some player, almost always an O-line starter,
plays exactly 100% of a team's real offensive snaps in 541/544, ~99.4%,
of real 2025 team-games). It's crosswalked from the snap data's own
`pfr_player_id` join key to this project's `player_id`/GSIS id via
`fetch_rosters_weekly`'s real `gsis_id`/`pfr_id` columns (confirmed live
to cover 382 of 383, ~99.7%, of a real 2025 qualified population); a
player missing that crosswalk gets a real `NaN`, never a fabricated 0%.

**Real round-split scoring** (`r1_dk_points`/`r2_r4_dk_points`,
`nfl_bestball.build_bestball_rankings`) breaks the real season total into
the two real stages DraftKings Best Ball Mania tournaments actually run
on - confirmed live via web search against Establish The Run and 4for4:
a real "Round 1" spanning weeks 1-14 (cumulative real points across
those weeks decide who advances out of each real 12-team draft pod - the
top 2), followed by three real single-week knockout rounds at weeks 15,
16, and 17 ("Rounds 2-4", reported here as one combined real sum, not
split further). Both are real per-range sums of the exact same DK
scoring formula `dk_points_total` uses (`config.
NFL_BESTBALL_ROUND1_END_WEEK = 14`), not a different metric or a
simulated pod result - this project has no draft-pod/ADP data, so it
can't simulate real tournament advancement, only report how a player's
real production actually split across the two real stages. The
dashboard's Bestball table shows `R1 Score`/`R2-R4 Score` plus a real
season total at the end of the row, in place of the redundant `Games`
column (already covered by `Missed`/`Missed (Prior Yr)`) and the
`Season Snap %` column (kept in the underlying CSV for the position-
scarcity qualifier above, just not surfaced in this table).

**Real rank columns and live player search.** `build_bestball_rankings`
adds `overall_rank` (1-based, across every real position together) and
`position_rank` (the same real idea, computed separately within each
position, via pandas' own `rank(method="first")` so ties get consecutive
real ranks rather than an invented tie-break) - both surfaced as the
leading `Rank`/`Pos Rank` columns on the dashboard table, so a real
"QB12"/"WR3" read doesn't require counting rows by hand. A player search
box above the table filters by real player name as you type (`oninput`,
no button/Enter needed), combined with the existing position tabs.

The `Pos Rank` column also shows each player's real `points_z_score` in
parentheses (e.g. `1 (+2.8σ)`) - how many real standard deviations their
own real `dk_points_total` sits from their position's real,
outlier-excluded core mean (`compute_position_scarcity`'s own
`mean_dk_points`/`std_dk_points`, reused directly rather than a second
statistical basis). Deliberately the real SEASON-TOTAL z-score, not a
Round-1-only one - `position_rank` itself comes from `points_above_
replacement` (season-total based, further scaled by `necessity_ratio`),
so pairing it with a season-total z-score keeps the parenthetical
answering the same real question the rank does, rather than quietly
mixing in a different one. Computed for every real player in the table,
not just the snap-share/points-floor qualified ones (an honest "how far
from a typical qualified starter" read even for players who didn't clear
the bar); a real std of 0 or NaN (fewer than 2 real core players at that
position) gives a real NaN `points_z_score`, shown as a bare rank with no
parenthetical rather than a fabricated number.

**Real draft strategy is now baked directly into the ranking, not left
as a separate table to cross-reference.** Real user feedback: raw
`dk_points_total` alone was a misleading overall order - Josh Allen
ranked #4 overall on real 2025 points (389.6), well above where real
DraftKings ADP would ever put him, because QB is a real flat/deep
position (real CV 0.28 - see Draft Strategy below) where a QB drafted
many rounds later still scores close to a QB1's total. `overall_rank`/
`position_rank` are now computed from a new `points_above_replacement`
column instead of raw `dk_points_total`: each player's real
`dk_points_total` minus their OWN position's real roster-depth-derived
points floor (`compute_draftable_points_floor` - the same real floor
`compute_position_scarcity`'s qualifier already uses, see below) - a
real "how much do you actually lose by waiting on this position"
question, the same logic real ADP reflects. A flat position's floor
sits close to its ceiling (real 2025 QB: mean 260.2, floor 145.5 - not
much daylight), so even a QB1's raw points overstate their real value;
a scarce position's floor sits far below its stars (real 2025 RB: floor
43.6 vs. McCaffrey's 428.6), so real RB/WR production keeps most of its
raw-points rank. `position_rank` lands on the identical order either
way (subtracting one position's own constant floor can't reorder
players within that position), but `overall_rank` genuinely changes -
real 2025 effect: Josh Allen drops from raw-points rank #4 to real
`overall_rank` 13 (still `position_rank` 1, still the real #1 QB - the
adjustment is about cross-position value, not within-position ranking).
That's a real, substantial, directionally-correct move, though it lands
short of the ~25 the user's own recollection of DK ADP suggested -
expected, since real ADP also prices in things this project doesn't
model (injury risk, name value, rookie hype), and this metric is
independently derived from real roster-depth math, not fit to reproduce
a specific external number. `points_above_replacement` itself is also
published as a `Value` column for transparency. A position whose real
player pool doesn't reach its real roster-depth pool size gets a real
floor of 0 (not a fabricated exclusion) - falls back to ranking on raw
`dk_points_total`, same as before this feature existed.

Built via `scripts/build_nfl_bestball_rankings.py`
(`.github/workflows/build_nfl_bestball_rankings.yml`, `workflow_dispatch`
only) - a manual/one-time build, not a daily/weekly cron, since real
last-season stats don't change and this isn't meant to be a live-updated
feed.

**Position Scarcity** (`docs/data/nfl_position_scarcity.csv`,
`nfl_bestball.compute_position_scarcity`): a "how many difference-makers
exist at this position, and how many replacement-level guys" read for
draft strategy, not just a flat rank list. Per position (QB/RB/WR/TE):
the real total player pool size, how many of those players cleared a
real season-total-snap-share qualifier (`config.
NFL_BESTBALL_SCARCITY_MIN_SNAP_SHARE = 0.3`, "played at least 30% of
your team's total real offensive snaps this season" - a simple, honest,
not-backtest-derived first-pass default, deliberately lower than a
"majority of snaps" bar; see below for why), and a bell-curve breakdown
of those qualified players by how many standard deviations they sit
from their own position's mean (the standard empirical-rule bands - 1-2
SD, 2-3 SD, beyond 3 SD in both directions - with the central "within 1
SD" band further split into real quarter-SD slices, since most
draft-relevant players land there and one wide bucket would hide real
shape right where it matters most).

**This qualifier has gone through two real fixes, both driven by real
data, not guesswork:**

1. It used to be `games_played >= 8`, changed after real 2025 data
   showed it let in players with almost no real offensive role. A real
   return specialist (Jamal Agnew) had `games_played=11` - clearing the
   old bar easily - but played exactly 1 real offensive snap (0.18%
   share) all season and scored 0 real points; several similar
   special-teamers/inactive-but-rostered players sat right in the same
   "qualified" population as true starters. `games_played` only
   requires ANY real stat row that week (even a single special-teams
   play), not a meaningful offensive role - this is exactly why the
   table's mean and std were once suspiciously close together (real
   2025 WR: mean 99.2, std 83.1 under the old rule - std nearly as
   large as the mean, a genuine red flag).
2. The snap-share qualifier itself was then a real PER-GAME average
   (`avg_offense_pct`), which fixed the special-teamer problem but
   introduced a real, different small-sample problem in the opposite
   direction: a real one-or-two-game emergency spot start at a high
   per-game rate would qualify just as easily as a real every-week
   starter. Real example: Raiders backup QB Aidan O'Connell's real
   single-game rate was 82% across his exactly-1 real game played -
   comfortably clearing even a 50% per-game bar. `compute_player_snap_share`
   now computes a real SEASON-TOTAL share instead (real total snaps
   that season / team's real total offensive snaps that season, both
   summed over every real game, not just games the player appeared in)
   - the same O'Connell drops to a real 5.1% season share once measured
   against the Raiders' real full-season offensive-play total, and is
   correctly excluded.

Because a real season-total share runs meaningfully lower than a
per-game rate even for genuinely relevant players (real 2025 example:
Jaylen Warren and TreVeyon Henderson, both clearly real, draftable
committee-role RBs with 200+ real season DK points, sit at real season
shares of 0.47/0.46 - not the 0.5+ a "majority of snaps" framing might
suggest), the qualifier's threshold was lowered from 0.5 to 0.3
alongside this fix - see `config.NFL_BESTBALL_SCARCITY_MIN_SNAP_SHARE`'s
own docstring for the full real-data reasoning.

**A real production floor is ALSO required, layered on top of (not
instead of) the snap-share qualifier** (`nfl_bestball.
compute_draftable_points_floor`, `config.NFL_BESTBALL_DRAFTABLE_POOL_SIZE`).
Real user feedback: even at a real 50%+ snap-share bar, a position's
qualified-pool mean still didn't reflect "players who'd actually get
drafted" - real complementary-role players can clear a real snap-share
bar (a genuine role/health signal) while producing very little (a
separate value question snap share doesn't answer). The floor is set at
the real Nth-ranked player's own real `dk_points_total`, N derived from
real DraftKings Best Ball Mania roster-depth math - confirmed live via
web search against DraftKings' own published rules and major fantasy
outlets: real 12-team draft pods, 20 real roster spots/team, with real
published strategy guidance on typical per-team position counts (QB
2-3, RB 5-7 [6 the consistently-cited number], WR 6-8, TE 1-2) - the
midpoint of each real range, times the real 12-team pod size, gives a
real "how many players at this position does a typical pool actually
draft" depth: `{"QB": 30, "RB": 72, "WR": 84, "TE": 18}`. A position
with fewer real players than its pool size gets no real floor (real
data doesn't reach that deep) rather than a fabricated one.

**Real statistical outliers are excluded from the mean/std before the
bell curve is built, via Tukey's IQR-fence rule** (`config.
NFL_BESTBALL_SCARCITY_IQR_MULTIPLIER = 1.5`, the textbook convention,
not tuned for this project - chosen specifically because it doesn't
need an already-computed mean/std as an input, unlike a z-score rule,
which would be circular for exactly this problem). Those excluded
outliers still appear in the bell curve buckets (almost always in the
extreme bands, exactly where a real outlier belongs) - they're excluded
from the summary statistics, not from the table. Removing them does NOT
eliminate all spread, honestly: real production among a real qualified
population genuinely ranges from replacement-level to true
difference-makers, and that remaining spread is real, not leftover
contamination.

Real 2025 numbers as of this writing (via `python
scripts/build_nfl_bestball_rankings.py`), under BOTH the season-total
snap-share qualifier AND the roster-depth-based production floor - note
how much closer the WR/TE means now sit to real "would actually be
drafted" territory than under snap share alone (WR mean 127.8 -> 157.4,
TE 90.6 -> 172.8):

| Position | Total players | Qualified | Points floor | Outliers excluded | Mean DK pts | Std DK pts | Coefficient of variation |
|---|---|---|---|---|---|---|---|
| QB | 81 | 30 | 145.5 | 0 | 260.2 | 72.4 | 0.28 |
| RB | 151 | 46 | 43.6 | 1 | 198.4 | 82.4 | 0.42 |
| WR | 240 | 81 | 75.1 | 2 | 157.4 | 62.5 | 0.40 |
| TE | 136 | 17 | 130.8 | 1 | 172.8 | 23.9 | 0.14 |

Two honest real observations worth flagging:

- **RB's real points floor (43.6) is surprisingly low** and barely
  changed RB's qualified count (46, same as snap share alone) - real DK
  Best Ball roster-construction strategy deliberately drafts deep at RB
  (handcuffs/dart-throws for real injury insurance, a well-documented
  real strategy given the position's real injury volatility), so a
  rank-72 real cutoff lands on real committee/replacement-tier value,
  not a bug.
- **TE's qualified pool shrank the most** (68 -> 17) - a real reflection
  of how few NFL tight ends are real receiving weapons versus real
  in-line blockers who still play real meaningful snap shares without
  much real receiving production.

**Draft Strategy** (`docs/data/nfl_draft_strategy_takeaways.csv`,
`nfl_bestball.compute_draft_strategy_takeaways`): a real, numbers-driven
answer to "does this season's real spread argue for prioritizing this
position early in a draft, or waiting" - the direct question the bell
curve above raises but doesn't answer by itself. Ranks each position's
`coefficient_of_variation` (std/mean of its own real, outlier-excluded
core group) **relative to the other real positions this season**, not
against an invented absolute cutoff - "high" or "low" dispersion only
means something compared to the other real positions in the same real
season's data. Positions at or above the real median CV are read as the
more top-heavy/scarcer positions this season (a bigger real gap between
a difference-maker and a typical qualified player, so grabbing a proven
top-tier player there early carries more relative value); positions
below the median CV are read as flatter/deeper (real production is more
interchangeable, so it's generally safer to wait and spend an early
pick at a scarcer position instead).

This directly answers a natural question like "if QBs are all within 2
SD of the mean, should a TE be prioritized instead" - yes, exactly when
the other position's own real CV this season ranks higher (scarcer)
than QB's. Real 2025 read, under the corrected qualifier (season-total
snap share AND a real production floor): RB is now the SCARCEST
position (CV 0.42, real ranking #1 of 4), followed by WR (0.40), then
QB (0.28), with TE the flattest/deepest by a wide margin (CV 0.14) -
the fourth real read this table has produced across the qualifier's
three real fixes (originally QB flattest under games-played, then QB
scarcest under per-game average, then TE scarcest under season-total
share alone, now RB scarcest with the production floor added). Each
flip is itself a real illustration of how much the qualifier's
definition matters for this kind of relative comparison - not evidence
any one read was "wrong" so much as that a population contaminated with
the wrong players (special-teamers, short high-rate samples, or
low-value complementary role players) produces a real but misleading
CV, and only a population filtered on both real role AND real value
reflects players a real bestball drafter would actually be choosing
among.

**Position Necessity** (`docs/data/nfl_position_necessity.csv`,
`nfl_bestball.compute_position_necessity`): a real, distinct question from
the CV-based takeaway above - not "how spread out is production within
this position" but "how many of each position do we actually want on our
own 20-round roster, compared to how many real players are actually good
enough to draft this season." Compares real per-team roster-construction
targets (`config.NFL_BESTBALL_ROSTER_TARGET = {"QB": (2,3), "RB": (5,7),
"WR": (6,8), "TE": (1,2)}`, the same real DK Best Ball Mania guidance
`NFL_BESTBALL_DRAFTABLE_POOL_SIZE` is now DERIVED from, not independently
hardcoded) times a real 12-team pod (`config.NFL_BESTBALL_DRAFT_POD_SIZE
= 12`) against the position-scarcity table's own real `qualified_players`
count - real demand vs. real supply. Real 2025 read:

| Position | Roster target/team | Pod demand | Available (qualified) | Necessity ratio | Read |
|---|---|---|---|---|---|
| QB | 2-3 | 30 | 30 | 1.00 | roughly balanced |
| RB | 5-7 | 72 | 46 | 1.57 | real shortage |
| WR | 6-8 | 84 | 81 | 1.04 | roughly balanced |
| TE | 1-2 | 18 | 17 | 1.06 | roughly balanced |

RB is the one real standout: a 12-team pod wants 72 real RBs across its
20-round drafts, but only 46 real RBs actually clear this season's real
role+value bar - real bestball RB strategy (drafting deep/handcuffs for
real injury insurance, discussed above) means a real chunk of "the RBs a
pod drafts" are dart-throws, not genuinely startable difference-makers.

**This necessity_ratio is ALSO baked directly into `points_above_
replacement`/`overall_rank` in `nfl_bestball_rankings.csv`, not left as a
separate table to cross-reference only** - `build_bestball_rankings`
multiplies each player's VOR-adjusted value by their own position's real
necessity_ratio (both the points-floor subtraction and this multiplier
are per-position CONSTANTS, so `position_rank` is unaffected - only
`overall_rank` moves further). Real, honest compounding effect worth
flagging: since RB was already the position VOR rewarded most (see
above), stacking a real 1.57x necessity multiplier on top pushes RB even
further ahead - the real top 10 overall-ranked players are now ALL RBs
(previously a mix with WRs and Josh Allen in the top 10). Josh Allen's
`overall_rank` moves from 13 (VOR alone) to 26 (VOR + necessity) - QB's
necessity_ratio is right at 1.00 (roughly balanced), so this move is
entirely RB pulling further away, not QB being pushed down further. The
`read`/threshold cutoffs (>=1.1 shortage, <=0.9 surplus) are a simple,
honest first pass - NOT backtest-derived, easy to revisit once more real
seasons accumulate.

### Draft Assistant ("My Draft" - real, live-draft advice)

Everything above is a static preseason snapshot - it has no notion of an
in-progress draft. Real user ask: given a real current roster (e.g. 3 RBs,
2 WRs already drafted) and the pick you're at right now, what should you
actually do next? Three real gaps stood between the static tables above
and that ask, all addressed here: (1) real draft-state tracking - what's
on your roster, what pick you're at; (2) real market-consensus data to
judge "is this pick a reach"; (3) real snake-draft "picks until your next
turn" arithmetic.

**Real ECR data, not raw ADP.** Every raw ADP-provider domain (Fantasy
Football Calculator, FantasyPros itself, MyFantasyLeague) is blocked by
this sandbox's network egress policy, and this project has no HTTP client
dependency (see `requirements.txt`) to reach one directly regardless.
Instead, `nfl_ff_rankings.py` uses `nflreadpy.load_ff_rankings()`/
`load_ff_playerids()` (already-installed dependency, zero new packages) -
real FantasyPros Expert Consensus Rank (ECR) data, confirmed live via
`scripts/debug_nfl_data.py` (`.github/workflows/debug_nfl_data.yml`, run
31513644704, 2026-08-11) before any parsing code was written against
assumed field names, per this project's established data-confirmation
discipline. Filtered to real `ecr_type == "bo"` ("best-ball overall" - one
of nine real confirmed slices, the one that matches this project's
bestball focus directly and is already cross-position-comparable, like a
real draft pick number). Crosswalked from FantasyPros' own real player id
to this project's `player_id` (GSIS id) via `load_ff_playerids()`'s real
`fantasypros_id`/`gsis_id` columns - real confirmed coverage ~70.5%
(2211/3134 real 2025-rostered players), meaningfully lower than the 99.7%
`pfr_id` crosswalk `compute_player_snap_share` uses, so a real chunk of
players legitimately has no ECR match (absent from the output, never
fabricated). Unlike every other table on this page, ECR is a real
CURRENT-moment snapshot, not a per-season historical stat - fetched LIVE
at build time by `scripts/build_nfl_bestball_rankings.py` (not from
persisted parquet), and resiliently skipped (never a build failure) if
the real fetch fails that run.

**Roster gap and real necessity - buildable from data this project
already had.** `docs/nfl_draft_assistant.js` (real `node --test`
coverage, `docs/nfl_draft_assistant.test.js`, added to `ci.yml` - the
same real-test-coverage bar `docs/nfl_dfs_solver.js` already holds):
`computeRosterGap` compares your real entered roster against
`config.NFL_BESTBALL_ROSTER_TARGET`'s real per-team ranges (mirrored as a
JS constant, same "duplicate the real constant rather than share a
Python/JS module" convention `nfl_dfs_solver.js` already uses), reading
each position as real `need` (below the real min), `optional` (within
range), or `full` (at/above the real max). Combined with each position's
already-published real `necessity_ratio` (Position Necessity, above) -
every column a real, separate, transparent number, no invented composite
"take this player" score.

**No prediction of who's still on the board.** An earlier version of this
tool tried to guess a real "best remaining" player per position by
filtering `nflBestball` against ECR - real user feedback caught this
guessing at pick 59 that Josh Allen and Christian McCaffrey were still
available, when a real draft board would obviously have taken them long
before. The tool has no view of a live draft board and can't know what
OTHER teams have taken, so it stopped trying to guess. Instead, a second
search-and-add list ("Players I'm Considering For This Pick") lets you
name the specific players YOU'RE weighing right now, and the table
assesses only those real players: their real `points_above_replacement`
value, whether their position is a real roster need for you, the
position's real `necessity_ratio`, and a real reach/value read (below).
Drafting a considered player (adding them to My Roster) automatically
drops them from the considering list - they're no longer a decision to
weigh, they're on your team.

**Reach/value read and snake-draft math - the two pieces that needed real
ECR data.** `computeReachValueRead(ecr, ecrSd, pickNumber)`: a real,
honest comparison - your entered pick number against a player's real ECR
± their own real expert-rank standard deviation (no invented probability
model; both numbers come straight from the real confirmed `load_ff_rankings`
columns), applied per considered player rather than as a filter over the
whole player pool. A player with no real ECR match simply shows no read
(never a fabricated one), and is NOT excluded from consideration - you
told the tool you're weighing them, so it assesses what it honestly can.
`computeSnakeDraftPicksUntilNextTurn(draftSlot, podSize, pickNumber)`:
real snake-draft arithmetic (12-team pod, alternating direction each
round) from an entered real draft slot (1-12) and pick number - both
optional inputs; roster gap/necessity/considered-player assessment always
work without them.

**Entirely client-side, no account/server.** Both search-and-add lists
(My Roster, Players I'm Considering For This Pick) reuse the
already-loaded `nflBestball` array (no new CSV for search itself),
rendered as clickable candidate buttons (mirrors `docs/app.js`'s
`searchPlayer`/`showPlayer` pattern) with removable player cards (reusing
the existing `.removePitcher` button styling, laid out via the same
`.todaysPicks` flex-wrap container `docs/app.js` already uses elsewhere).
Your roster, considered-players list, draft slot, and pick number persist
to this browser's `localStorage` only (key `nflDraftAssistant.v1`) - a
public static GitHub Pages site with no per-user backend - with a "Clear
My Draft" button to reset.

**Preseason Notes** (`docs/data/nfl_draft_notes.csv`): a one-time,
hand-curated list of real training-camp storylines specifically for
players drafted in the April 2026 NFL Draft (depth-chart battles,
landing-spot opportunity, injury setbacks) from trusted sources (ESPN,
Yahoo Sports, SI, Fox Sports, NFL.com, team sites), each a short
paraphrase in original words with a link back to the source - never
reproduced article text. This list deliberately targets rookies: the
bestball rankings above are built entirely from real last-season stats,
so they have nothing to say about a player with no NFL track record
yet - this is where that gap gets filled in, once, by hand. This is
genuinely NOT automated - this codebase has zero precedent for fetching
external web content (no scraping/RSS/WebFetch anywhere), and the NFL
page's own design philosophy is "no new network call, same-origin CSV
fetches only."

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

When `--as-of-date` isn't passed, "today" is `schedule.today_local()` -
Arizona time (fixed UTC-7, no DST), not the running machine's own local
time. `daily_update.yml`'s GitHub Actions runner is UTC, so a manual
`workflow_dispatch` run late in the Arizona evening (e.g. 8:45pm AZ =
3:45am UTC, already past UTC midnight) would otherwise publish tomorrow's
slate a day early for an Arizona audience. Same reasoning applies to
`scripts/build_dfs_rankings.py`/`build_hitter_hit_predictions.py`'s own
`--as-of-date` defaults.

## Tests

```
pip install -r requirements-dev.txt
pytest
```

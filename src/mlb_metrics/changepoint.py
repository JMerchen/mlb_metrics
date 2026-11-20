"""Spotting the game a player's role or skill really changed: Bayesian
online changepoint detection.

Why (2026-11-20): every "recent form" number on the site blends fixed
windows - a share of the carries over the last four games and the season,
a batting average over the season and the last 81, 30 and 10 days. A fixed
window is wrong in both directions. When nothing has changed it throws
away good data to chase noise; when a role really changes (a back loses
the job, a receiver returns as the third option) it takes the window's
whole length to catch up.

Bayesian online changepoint detection (Adams and MacKay 2007) keeps, after
every game, a probability for each possible "run length": how many games
ago the player's current level began. Each game either extends every run
(if it looks like that run's level) or starts a new one (always possible,
with a small prior chance - the `hazard`). The estimate of his level is
then the average over runs, each weighted by its probability: all of his
history when nothing has changed, only the games since the break when one
is clear, and an honest mixture while it is unsure. No window to choose.

Two kinds of series:

  rates    k successes in n chances each game (carries out of the team's,
           hits out of at-bats): rate_runs. Team carries are not
           independent coin flips - a back's share swings with the game
           script - so the counts are `tempered`: multiplied by a number
           under 1 that says how many independent chances a game is really
           worth. Hazard and temper are fitted to data, not guessed.
  levels   a measurement each game (a pitcher's average fastball speed):
           level_runs, with a spread learned as it goes.
"""

import numpy as np
from scipy.special import betaln, gammaln

MAX_RUN = 60


def _normalize(log_weights: np.ndarray) -> np.ndarray:
    top = log_weights.max()
    w = np.exp(log_weights - top)
    return w / w.sum()


def rate_runs(successes, trials, hazard: float, temper: float = 1.0, prior_mean: float = 0.2,
              prior_strength: float = 2.0) -> dict:
    """After each game of a rate series: {mean, changed, run}.

    mean[t]     the estimate of the rate after game t (so the prediction
                for game t + 1): each run's Beta estimate weighted by the
                run's probability.
    changed[t]  the probability the current level began within the last 3
                games - "his role just changed".
    run[t]      the most probable number of games the current level has
                lasted.

    hazard: the prior chance, each game, that the level changes. temper:
    how many independent chances one counted chance is worth. A new level
    starts from a prior worth `prior_strength` (tempered) chances, centred
    on the player's own average so far (`prior_mean` before he has any)."""
    k = np.asarray(successes, dtype=float) * temper
    n = np.asarray(trials, dtype=float) * temper
    steps = len(k)
    mean, changed, run = np.empty(steps), np.empty(steps), np.empty(steps, dtype=int)
    a = np.array([prior_mean * prior_strength])
    b = np.array([(1.0 - prior_mean) * prior_strength])
    log_w = np.array([0.0])
    seen_k = seen_n = 0.0
    for t in range(steps):
        # How well each run predicted this game (a Beta-Binomial, which
        # takes the tempered, non-whole counts as they are).
        predictive = (gammaln(n[t] + 1) - gammaln(k[t] + 1) - gammaln(n[t] - k[t] + 1)
                      + betaln(a + k[t], b + n[t] - k[t]) - betaln(a, b))
        grown = log_w + predictive + np.log1p(-hazard)
        fresh = np.logaddexp.reduce(log_w + predictive) + np.log(hazard)
        seen_k, seen_n = seen_k + k[t], seen_n + n[t]
        # A new level starts from the player's own average so far, itself
        # pulled toward prior_mean by the same weak prior.
        centre = (seen_k + prior_mean * prior_strength) / (seen_n + prior_strength)
        centre = min(0.98, max(0.02, centre))
        # The new run starts from the prior; it has not seen this game.
        a = np.concatenate([[centre * prior_strength], a + k[t]])[:MAX_RUN]
        b = np.concatenate([[(1.0 - centre) * prior_strength], b + n[t] - k[t]])[:MAX_RUN]
        log_w = np.concatenate([[fresh], grown])[:MAX_RUN]
        w = _normalize(log_w)
        log_w = np.log(np.maximum(w, 1e-300))
        mean[t] = float(np.dot(w, a / (a + b)))
        changed[t] = float(w[:4].sum()) if t >= 4 else 0.0
        run[t] = int(w.argmax())
    return {"mean": mean, "changed": changed, "run": run}


def level_runs(values, hazard: float, prior_mean: float | None = None, prior_weight: float = 0.5,
               spread: float | None = None) -> dict:
    """After each game of a measured series: {mean, changed, run}, as
    rate_runs. Each run's level and spread are learned together (a normal
    model with both unknown); `spread` is a starting guess for the size of
    game-to-game noise (the series' own first few values if left out)."""
    x = np.asarray(values, dtype=float)
    steps = len(x)
    if steps == 0:
        return {"mean": np.array([]), "changed": np.array([]), "run": np.array([], dtype=int)}
    m0 = float(x[0]) if prior_mean is None else prior_mean
    s0 = float(np.std(x[:5])) or 1.0 if spread is None else spread
    alpha0, beta0 = 2.0, 2.0 * s0 * s0 / 2.0
    mu, kappa = np.array([m0]), np.array([prior_weight])
    alpha, beta = np.array([alpha0]), np.array([beta0])
    log_w = np.array([0.0])
    mean, changed, run = np.empty(steps), np.empty(steps), np.empty(steps, dtype=int)
    for t in range(steps):
        # Student-t predictive of each run for this value.
        scale2 = beta * (kappa + 1.0) / (alpha * kappa)
        dof = 2.0 * alpha
        predictive = (gammaln((dof + 1) / 2) - gammaln(dof / 2) - 0.5 * np.log(dof * np.pi * scale2)
                      - (dof + 1) / 2 * np.log1p((x[t] - mu) ** 2 / (dof * scale2)))
        grown = log_w + predictive + np.log1p(-hazard)
        fresh = np.logaddexp.reduce(log_w + predictive) + np.log(hazard)
        new_mu = (kappa * mu + x[t]) / (kappa + 1.0)
        new_beta = beta + kappa * (x[t] - mu) ** 2 / (2.0 * (kappa + 1.0))
        centre = float(np.mean(x[: t + 1]))
        mu = np.concatenate([[centre], new_mu])[:MAX_RUN]
        kappa = np.concatenate([[prior_weight], kappa + 1.0])[:MAX_RUN]
        alpha = np.concatenate([[alpha0], alpha + 0.5])[:MAX_RUN]
        beta = np.concatenate([[beta0], new_beta])[:MAX_RUN]
        log_w = np.concatenate([[fresh], grown])[:MAX_RUN]
        w = _normalize(log_w)
        log_w = np.log(np.maximum(w, 1e-300))
        mean[t] = float(np.dot(w, mu))
        changed[t] = float(w[:4].sum()) if t >= 4 else 0.0
        run[t] = int(w.argmax())
    return {"mean": mean, "changed": changed, "run": run}

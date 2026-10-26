# NFL game-pick settings grid search

1152 of 1152 combinations scored (validation: walk-forward 2018-2024, lower log loss is better; C re-picked per combination).

Today's settings: validation log loss 0.6480, rank 638 of 1152.

## Top 15

| rank | combo | team_strength_windows | pythagorean_exponent | confidence_sos_weight | qb_windows | qb_continuity_weight | C | log loss | vs today | accuracy |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 1135 | [[5, 0.4], [10, 0.3], [null, 0.3]] | 3.0 | 0.15 | [[null, 0.4], [8, 0.3], [4, 0.3]] | 1.0 | 0.03 | 0.6404 | -0.0076 | 0.643 |
| 2 | 1134 | [[5, 0.4], [10, 0.3], [null, 0.3]] | 3.0 | 0.15 | [[null, 0.4], [8, 0.3], [4, 0.3]] | 0.75 | 0.03 | 0.6404 | -0.0076 | 0.643 |
| 3 | 1132 | [[5, 0.4], [10, 0.3], [null, 0.3]] | 3.0 | 0.15 | [[null, 0.4], [8, 0.3], [4, 0.3]] | 0.5 | 0.03 | 0.6404 | -0.0076 | 0.643 |
| 4 | 1133 | [[5, 0.4], [10, 0.3], [null, 0.3]] | 3.0 | 0.15 | [[null, 0.4], [8, 0.3], [4, 0.3]] | 0.25 | 0.03 | 0.6404 | -0.0076 | 0.643 |
| 5 | 1110 | [[5, 0.4], [10, 0.3], [null, 0.3]] | 3.0 | 0.3 | [[null, 0.4], [8, 0.3], [4, 0.3]] | 0.75 | 0.03 | 0.6404 | -0.0076 | 0.642 |
| 6 | 1108 | [[5, 0.4], [10, 0.3], [null, 0.3]] | 3.0 | 0.3 | [[null, 0.4], [8, 0.3], [4, 0.3]] | 0.5 | 0.03 | 0.6404 | -0.0076 | 0.642 |
| 7 | 1109 | [[5, 0.4], [10, 0.3], [null, 0.3]] | 3.0 | 0.3 | [[null, 0.4], [8, 0.3], [4, 0.3]] | 0.25 | 0.03 | 0.6404 | -0.0076 | 0.642 |
| 8 | 1111 | [[5, 0.4], [10, 0.3], [null, 0.3]] | 3.0 | 0.3 | [[null, 0.4], [8, 0.3], [4, 0.3]] | 1.0 | 0.03 | 0.6404 | -0.0076 | 0.642 |
| 9 | 1087 | [[5, 0.4], [10, 0.3], [null, 0.3]] | 2.7 | 0.15 | [[null, 0.4], [8, 0.3], [4, 0.3]] | 1.0 | 0.03 | 0.6405 | -0.0075 | 0.643 |
| 10 | 1086 | [[5, 0.4], [10, 0.3], [null, 0.3]] | 2.7 | 0.15 | [[null, 0.4], [8, 0.3], [4, 0.3]] | 0.75 | 0.03 | 0.6405 | -0.0075 | 0.643 |
| 11 | 1085 | [[5, 0.4], [10, 0.3], [null, 0.3]] | 2.7 | 0.15 | [[null, 0.4], [8, 0.3], [4, 0.3]] | 0.25 | 0.03 | 0.6405 | -0.0075 | 0.643 |
| 12 | 1084 | [[5, 0.4], [10, 0.3], [null, 0.3]] | 2.7 | 0.15 | [[null, 0.4], [8, 0.3], [4, 0.3]] | 0.5 | 0.03 | 0.6405 | -0.0075 | 0.643 |
| 13 | 1061 | [[5, 0.4], [10, 0.3], [null, 0.3]] | 2.7 | 0.3 | [[null, 0.4], [8, 0.3], [4, 0.3]] | 0.25 | 0.03 | 0.6406 | -0.0074 | 0.642 |
| 14 | 1062 | [[5, 0.4], [10, 0.3], [null, 0.3]] | 2.7 | 0.3 | [[null, 0.4], [8, 0.3], [4, 0.3]] | 0.75 | 0.03 | 0.6406 | -0.0074 | 0.642 |
| 15 | 1063 | [[5, 0.4], [10, 0.3], [null, 0.3]] | 2.7 | 0.3 | [[null, 0.4], [8, 0.3], [4, 0.3]] | 1.0 | 0.03 | 0.6406 | -0.0074 | 0.642 |

## Each setting, averaged over every combination of the others

**NFL_TEAM_STRENGTH_WINDOWS** (today: `[[3, 0.5], [7, 0.3], [null, 0.2]]`)

| value | mean | best | vs grand mean | combos |
|---|---|---|---|---|
| `[[5, 0.4], [10, 0.3], [null, 0.3]]` | 0.6410 | 0.6404 | -0.0063 | 192 |
| `[[3, 0.35], [7, 0.35], [null, 0.3]]` | 0.6445 | 0.6438 | -0.0028 | 192 |
| `[[4, 0.5], [8, 0.3], [null, 0.2]]` | 0.6450 | 0.6444 | -0.0024 | 192 |
| `[[3, 0.5], [7, 0.3], [null, 0.2]]` | 0.6482 | 0.6475 | +0.0009 | 192 |
| `[[3, 0.65], [7, 0.25], [null, 0.1]]` | 0.6525 | 0.6518 | +0.0051 | 192 |
| `[[2, 0.5], [6, 0.3], [null, 0.2]]` | 0.6528 | 0.6523 | +0.0055 | 192 |

**NFL_PYTHAGOREAN_EXPONENT** (today: `2.37`)

| value | mean | best | vs grand mean | combos |
|---|---|---|---|---|
| `3.0` | 0.6471 | 0.6404 | -0.0002 | 288 |
| `2.7` | 0.6472 | 0.6405 | -0.0001 | 288 |
| `2.37` | 0.6474 | 0.6407 | +0.0000 | 288 |
| `2.0` | 0.6475 | 0.6409 | +0.0002 | 288 |

**NFL_CONFIDENCE_SOS_WEIGHT** (today: `0.3`)

| value | mean | best | vs grand mean | combos |
|---|---|---|---|---|
| `0.3` | 0.6471 | 0.6404 | -0.0002 | 288 |
| `0.15` | 0.6473 | 0.6404 | -0.0001 | 288 |
| `0.5` | 0.6473 | 0.6408 | -0.0000 | 288 |
| `0.0` | 0.6476 | 0.6406 | +0.0003 | 288 |

**NFL_QB_WINDOWS** (today: `[[null, 0.2], [8, 0.3], [4, 0.5]]`)

| value | mean | best | vs grand mean | combos |
|---|---|---|---|---|
| `[[null, 0.4], [8, 0.3], [4, 0.3]]` | 0.6471 | 0.6404 | -0.0002 | 384 |
| `[[null, 0.2], [8, 0.3], [4, 0.5]]` | 0.6474 | 0.6406 | +0.0000 | 384 |
| `[[null, 0.1], [8, 0.3], [4, 0.6]]` | 0.6475 | 0.6407 | +0.0002 | 384 |

**NFL_QB_CONTINUITY_WEIGHT** (today: `0.5`)

| value | mean | best | vs grand mean | combos |
|---|---|---|---|---|
| `0.25` | 0.6473 | 0.6404 | +0.0000 | 288 |
| `0.5` | 0.6473 | 0.6404 | +0.0000 | 288 |
| `0.75` | 0.6473 | 0.6404 | +0.0000 | 288 |
| `1.0` | 0.6473 | 0.6404 | +0.0000 | 288 |

## Interactions (RMS departure of each pair's cells from additive, log loss)

| pair | RMS |
|---|---|
| team_strength_windows x confidence_sos_weight | 0.00017 |
| team_strength_windows x pythagorean_exponent | 0.00004 |
| team_strength_windows x qb_windows | 0.00002 |
| pythagorean_exponent x confidence_sos_weight | 0.00002 |
| confidence_sos_weight x qb_windows | 0.00000 |
| pythagorean_exponent x qb_windows | 0.00000 |
| qb_windows x qb_continuity_weight | 0.00000 |
| team_strength_windows x qb_continuity_weight | 0.00000 |
| confidence_sos_weight x qb_continuity_weight | 0.00000 |
| pythagorean_exponent x qb_continuity_weight | 0.00000 |

## 2025 holdout

Best combination 1135 changes: NFL_TEAM_STRENGTH_WINDOWS -> `[[5, 0.4], [10, 0.3], [null, 0.3]]`, NFL_PYTHAGOREAN_EXPONENT -> `3.0`, NFL_CONFIDENCE_SOS_WEIGHT -> `0.15`, NFL_QB_WINDOWS -> `[[null, 0.4], [8, 0.3], [4, 0.3]]`, NFL_QB_CONTINUITY_WEIGHT -> `1.0`

272 games: log loss today 0.6272, best 0.6320 (+0.0048); accuracy 0.614 vs 0.621; paired t-test p = 0.486.

Ship it only if 2025 improves too (lower log loss, ideally p < 0.05).

## Epochs: gradient boosting's validation log loss by number of trees (today's settings)

| trees | learning rate 0.03 | learning rate 0.1 |
|---|---|---|
| 10 | 0.6775 | 0.6593 |
| 25 | 0.6643 | 0.6513 |
| 50 | 0.6552 | 0.6553 |
| 100 | 0.6520 | 0.6679 |
| 200 | 0.6574 | 0.6905 |
| 400 | 0.6715 | 0.7251 |

Logistic regression (today's model type) on the same seasons: 0.6480.

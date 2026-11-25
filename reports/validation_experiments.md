# Validation experiments

Written by `scripts/03_train.py`. Models are trained on the train split and compared on the validation split; the test split is not used.

**How configs are compared.** Different models (and different C values) put their scores on different scales: a strongly regularised model rarely scores above 0.5 even when it ranks the right tags first. Judging everything at one fixed threshold would therefore measure score scale, not quality. So each config is judged at its own best global threshold (0.05 to 0.95) and tag limit (1 to 5) on validation - the same search `04_tune_thresholds.py` uses - and the config with the highest `tuned_micro_f1` is chosen. `best_threshold` and `best_max_tags` show where that best was found; macro-F1, precision and recall are measured there too. P@3 ignores thresholds. `micro_f1_at_0.5` (fixed threshold 0.5, up to 5 tags) is kept only for reference.

## Chosen configs

| model | chosen config | tuned_micro_f1 | best_threshold | best_max_tags | macro_f1 | micro_precision | micro_recall | p_at_3 | micro_f1_at_0.5 |
|---|---|---|---|---|---|---|---|---|---|
| keyword | keywords from taxonomy (no training) | 0.555 | 0.400 | 2 | 0.452 | 0.581 | 0.531 | 0.425 | 0.523 |
| tfidf | title_weight=0, C=16 | 0.688 | 0.150 | 5 | 0.588 | 0.712 | 0.665 | 0.508 | 0.565 |
| semantic | pooling=body_mean, C=4 | 0.667 | 0.250 | 4 | 0.569 | 0.693 | 0.642 | 0.484 | 0.601 |
| hybrid | weight_tfidf=0.7 | 0.695 | 0.200 | 4 | 0.585 | 0.752 | 0.646 | 0.512 | 0.569 |

## TF-IDF grid (title weight x C)

Title weight 0 means the title is ignored; higher values make title words count more. See `figures/title_weight.png`.

| title_weight | C | tuned_micro_f1 | best_threshold | best_max_tags | macro_f1 | micro_precision | micro_recall | p_at_3 | micro_f1_at_0.5 | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 4 | 0.679 | 0.150 | 3 | 0.577 | 0.680 | 0.678 | 0.500 | 0.496 | 17.200 |
| 0 | 16 | 0.688 | 0.150 | 5 | 0.588 | 0.712 | 0.665 | 0.508 | 0.565 | 19.500 |
| 0 | 64 | 0.685 | 0.150 | 5 | 0.581 | 0.750 | 0.631 | 0.509 | 0.589 | 18.700 |
| 1 | 4 | 0.680 | 0.150 | 3 | 0.578 | 0.708 | 0.654 | 0.489 | 0.518 | 17.200 |
| 1 | 16 | 0.679 | 0.150 | 3 | 0.578 | 0.747 | 0.623 | 0.494 | 0.559 | 18.000 |
| 1 | 64 | 0.682 | 0.100 | 3 | 0.593 | 0.722 | 0.645 | 0.493 | 0.588 | 18.200 |
| 2 | 4 | 0.613 | 0.150 | 4 | 0.506 | 0.631 | 0.596 | 0.442 | 0.515 | 17.800 |
| 2 | 16 | 0.609 | 0.100 | 3 | 0.513 | 0.611 | 0.606 | 0.442 | 0.549 | 18.400 |
| 2 | 64 | 0.607 | 0.100 | 3 | 0.507 | 0.640 | 0.578 | 0.442 | 0.556 | 19.100 |
| 3 | 4 | 0.565 | 0.200 | 5 | 0.459 | 0.632 | 0.511 | 0.406 | 0.513 | 18.600 |
| 3 | 16 | 0.559 | 0.150 | 4 | 0.461 | 0.595 | 0.527 | 0.406 | 0.531 | 18.900 |
| 3 | 64 | 0.557 | 0.100 | 5 | 0.464 | 0.567 | 0.547 | 0.403 | 0.536 | 19.600 |

## Semantic model (pooling x C)

title_only uses only the title vector; body_mean averages the body chunk vectors; the title+ options join the title vector with the averaged (mean) or strongest (max) body values. See `figures/pooling.png`.

| pooling | C | tuned_micro_f1 | best_threshold | best_max_tags | macro_f1 | micro_precision | micro_recall | p_at_3 | micro_f1_at_0.5 | seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| title_only | 4 | 0.522 | 0.250 | 5 | 0.432 | 0.553 | 0.495 | 0.383 | 0.485 | 1.200 |
| title_only | 16 | 0.501 | 0.300 | 3 | 0.413 | 0.531 | 0.473 | 0.369 | 0.477 | 1.300 |
| title_only | 64 | 0.482 | 0.400 | 2 | 0.385 | 0.560 | 0.422 | 0.356 | 0.467 | 1.300 |
| body_mean | 4 | 0.667 | 0.250 | 4 | 0.569 | 0.693 | 0.642 | 0.484 | 0.601 | 1.000 |
| body_mean | 16 | 0.658 | 0.250 | 4 | 0.576 | 0.663 | 0.654 | 0.487 | 0.627 | 1.100 |
| body_mean | 64 | 0.645 | 0.300 | 3 | 0.566 | 0.667 | 0.625 | 0.478 | 0.627 | 1.200 |
| title+body_mean | 4 | 0.652 | 0.250 | 4 | 0.567 | 0.668 | 0.637 | 0.482 | 0.615 | 1.100 |
| title+body_mean | 16 | 0.634 | 0.250 | 2 | 0.550 | 0.692 | 0.585 | 0.466 | 0.623 | 1.200 |
| title+body_mean | 64 | 0.603 | 0.350 | 5 | 0.524 | 0.614 | 0.592 | 0.445 | 0.598 | 1.300 |
| title+body_max | 4 | 0.650 | 0.300 | 5 | 0.557 | 0.704 | 0.603 | 0.477 | 0.611 | 1.200 |
| title+body_max | 16 | 0.629 | 0.350 | 5 | 0.546 | 0.677 | 0.588 | 0.456 | 0.613 | 1.200 |
| title+body_max | 64 | 0.596 | 0.350 | 3 | 0.518 | 0.622 | 0.572 | 0.441 | 0.591 | 1.300 |

## Hybrid (weight given to TF-IDF, using the best TF-IDF and semantic configs)

| weight_tfidf | tuned_micro_f1 | best_threshold | best_max_tags | macro_f1 | micro_precision | micro_recall | p_at_3 | micro_f1_at_0.5 |
|---|---|---|---|---|---|---|---|---|
| 0.300 | 0.683 | 0.200 | 3 | 0.595 | 0.691 | 0.675 | 0.499 | 0.590 |
| 0.500 | 0.691 | 0.200 | 4 | 0.592 | 0.716 | 0.668 | 0.508 | 0.582 |
| 0.700 | 0.695 | 0.200 | 4 | 0.585 | 0.752 | 0.646 | 0.512 | 0.569 |

Total time: 248 seconds.

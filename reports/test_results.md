# Test results

Written by `scripts/05_evaluate.py`: the first and only use of the test split (the newest 705 posts). Each model uses its tuned threshold and tag limit. 

- review_coverage_pct: posts NOT sent to an editor for review (higher = less manual work)
- auto_micro_f1: micro-F1 on the posts that were auto-suggested only
- ms_per_post: median time of one engine.suggest call (cleaning included), 200 calls

| model | label | micro_precision | micro_recall | micro_f1 | macro_f1 | p_at_3 | r_at_3 | review_coverage_pct | auto_micro_f1 | ms_per_post |
|---|---|---|---|---|---|---|---|---|---|---|
| keyword | Keyword baseline | 0.649 | 0.558 | 0.600 | 0.520 | 0.435 | 0.687 | 95.200 | 0.614 | 16.000 |
| tfidf | TF-IDF + Logistic Regression | 0.679 | 0.753 | 0.714 | 0.649 | 0.532 | 0.821 | 98.200 | 0.719 | 10.800 |
| semantic | Semantic (MiniLM embeddings) | 0.663 | 0.689 | 0.675 | 0.612 | 0.525 | 0.809 | 96.700 | 0.684 | 161.700 |
| hybrid | Hybrid (TF-IDF + Semantic) | 0.701 | 0.740 | 0.719 | 0.651 | 0.538 | 0.828 | 97.300 | 0.727 | 172.700 |

Default model: `hybrid`. Rule: pick the model with the best validation micro-F1 (hybrid); if another model is within 0.01 micro-F1 of it and faster per post, pick the fastest such model. Chosen: hybrid. Validation numbers: keyword: micro-F1 0.616, 9.1 ms/post; tfidf: micro-F1 0.703, 4.9 ms/post; semantic: micro-F1 0.679, 151.7 ms/post; hybrid: micro-F1 0.723, 149.1 ms/post.

## Macro-F1 by tag frequency (thirds by training count)

| model | common | medium | rare |
|---|---|---|---|
| hybrid | 0.753 | 0.566 | 0.619 |
| keyword | 0.624 | 0.425 | 0.497 |
| semantic | 0.703 | 0.534 | 0.586 |
| tfidf | 0.748 | 0.545 | 0.640 |

## Short drafts (title + first N words of the body)

| model | draft | micro_f1 | macro_f1 | p_at_3 | review_pct |
|---|---|---|---|---|---|
| keyword | full | 0.600 | 0.520 | 0.435 | 4.800 |
| keyword | first_40_words | 0.377 | 0.262 | 0.336 | 52.100 |
| keyword | first_10_words | 0.320 | 0.193 | 0.294 | 98.000 |
| tfidf | full | 0.714 | 0.649 | 0.532 | 1.800 |
| tfidf | first_40_words | 0.506 | 0.401 | 0.384 | 23.000 |
| tfidf | first_10_words | 0.357 | 0.270 | 0.300 | 98.000 |
| semantic | full | 0.675 | 0.612 | 0.525 | 3.300 |
| semantic | first_40_words | 0.509 | 0.440 | 0.401 | 4.100 |
| semantic | first_10_words | 0.363 | 0.299 | 0.284 | 97.600 |
| hybrid | full | 0.719 | 0.651 | 0.538 | 2.700 |
| hybrid | first_40_words | 0.540 | 0.471 | 0.406 | 8.700 |
| hybrid | first_10_words | 0.377 | 0.298 | 0.305 | 97.600 |

See also: test_per_tag.csv, similar_tag_confusions.csv, test_predictions.csv, error_examples.md, figures/model_comparison.png, figures/test_by_frequency.png.

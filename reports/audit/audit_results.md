# Label audit results

Posts audited: 40 (random test posts, seed 42).

| Labels | Precision | Recall | F1 |
|---|---|---|---|
| original (author tags) | 0.704 | 0.750 | 0.726 |
| corrected (after audit) | 0.765 | 0.765 | 0.765 |

- Posts with at least one missing tag: 7 of 40
- Model 'false positives' (suggested but not in the author's tags): 24; judged correct by the reviewer: 5 (21%)
- Unknown tag names typed by the reviewer (ignored): none

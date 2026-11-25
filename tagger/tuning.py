"""Search for the best global threshold and tag limit (shared by scripts 03 and 04)."""

import numpy as np
from sklearn.metrics import f1_score

from tagger.suggest import apply_settings

THRESHOLDS = [round(t, 2) for t in np.arange(0.05, 0.951, 0.05)]
MAX_TAGS_OPTIONS = [1, 2, 3, 4, 5]


def micro_f1(Y, Y_pred):
    return f1_score(Y, Y_pred, average="micro", zero_division=0)


def tune_global(scores, n_words, Y, tags, min_words):
    """Best (micro-F1, threshold, max_tags) over every threshold x max_tags pair.

    Models put their scores on different scales, so comparing them at one fixed threshold
    is unfair; comparing each at its own best threshold is not.
    """
    best = (-1, None, None)
    for threshold in THRESHOLDS:
        for max_tags in MAX_TAGS_OPTIONS:
            Y_pred, _ = apply_settings(scores, tags, {"threshold": threshold, "max_tags": max_tags}, n_words, min_words)
            f1 = micro_f1(Y, Y_pred)
            if f1 > best[0]:
                best = (f1, threshold, max_tags)
    return best

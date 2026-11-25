"""Scores for multi-label tagging: precision, recall, F1 and "top 3" ranking quality."""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support


def evaluate(Y_true, Y_pred, scores):
    """Compare true and predicted 0/1 label matrices [posts x tags].

    micro = count every (post, tag) decision together (common tags weigh more).
    macro = score each tag on its own, then average (every tag weighs the same).
    P@3 / R@3 look only at the 3 highest-scoring tags per post, ignoring thresholds.
    """
    Y_true = np.asarray(Y_true)
    Y_pred = np.asarray(Y_pred)
    scores = np.asarray(scores)
    result = {}
    for average in ["micro", "macro"]:
        p, r, f, _ = precision_recall_fscore_support(Y_true, Y_pred, average=average, zero_division=0)
        result[f"{average}_precision"] = float(p)
        result[f"{average}_recall"] = float(r)
        result[f"{average}_f1"] = float(f)

    # The 3 best-scoring tags of each post ("stable" keeps the earlier tag on ties).
    top3 = np.argsort(-scores, axis=1, kind="stable")[:, :3]
    hits = np.take_along_axis(Y_true, top3, axis=1).sum(axis=1)
    n_true = Y_true.sum(axis=1)
    result["p_at_3"] = float(np.mean(hits / 3))
    has_tags = n_true > 0               # recall is undefined for a post with no true tags
    result["r_at_3"] = float(np.mean(hits[has_tags] / n_true[has_tags])) if has_tags.any() else 0.0
    return result


def per_tag_report(Y_true, Y_pred, tags):
    """One row per tag: how many true posts it has (support), precision, recall and F1."""
    p, r, f, support = precision_recall_fscore_support(
        np.asarray(Y_true), np.asarray(Y_pred), average=None, zero_division=0)
    return pd.DataFrame({"tag": tags, "support": support, "precision": p, "recall": r, "f1": f})

"""Turn a row of model scores into the tags we suggest, and decide when an editor must review.

This is the one place where thresholds are applied, so the app, the tuning script
and the test evaluation all make exactly the same decisions.
"""

from dataclasses import asdict, dataclass

import numpy as np

MAX_CANDIDATES = 5          # the editor always sees the 5 best-scoring tags

REVIEW_REASON_TEXT = {
    "empty": "The post has no text, so no tags can be suggested.",
    "too_short": "The post is very short, so the suggestions are unreliable. Please check them.",
    "low_confidence": "No tag scored above its threshold. Please choose tags from the candidates.",
}


@dataclass
class TagScore:
    tag: str
    score: float
    threshold: float
    snippet: str | None = None   # the sentence that best supports this tag (filled by the engine)


@dataclass
class Suggestion:
    tags: list[TagScore]          # tags that passed their threshold (the actual suggestion)
    candidates: list[TagScore]    # top 5 by score, shown to the editor either way
    needs_review: bool
    review_reason: str | None     # "empty", "too_short", "low_confidence" or None
    model: str
    n_words: int
    title_clean: str
    body_clean: str

    def tag_names(self):
        return [t.tag for t in self.tags]

    def to_dict(self):
        result = asdict(self)
        result["review_reason_text"] = REVIEW_REASON_TEXT.get(self.review_reason)
        return result


def suggest_from_scores(score_row, tag_names, model_settings, n_words, min_words):
    """Apply thresholds to one post's scores.

    Returns (tags, candidates, needs_review, review_reason).
    Only names from tag_names (the approved taxonomy) can ever come out.
    """
    if n_words == 0:
        return [], [], True, "empty"

    score_row = np.asarray(score_row, dtype=float)
    if len(score_row) != len(tag_names):
        raise ValueError(f"got {len(score_row)} scores for {len(tag_names)} tags")
    global_threshold = model_settings.get("threshold", 0.5)
    max_tags = model_settings.get("max_tags", MAX_CANDIDATES)
    per_tag = model_settings.get("per_tag_thresholds") or {}

    best_first = np.argsort(-score_row, kind="stable")[:MAX_CANDIDATES]
    candidates = []
    for i in best_first:
        tag = tag_names[i]
        candidates.append(TagScore(tag, float(score_row[i]), float(per_tag.get(tag, global_threshold))))
    tags = [c for c in candidates if c.score >= c.threshold][:max_tags]

    if n_words < min_words:
        return tags, candidates, True, "too_short"
    if not tags:
        return tags, candidates, True, "low_confidence"
    return tags, candidates, False, None


def apply_settings(scores, tag_names, model_settings, n_words_list, min_words):
    """Run suggest_from_scores on every row of a score matrix.

    Returns (Y_pred 0/1 matrix, list of review reasons with None for auto-suggested posts).
    """
    Y_pred = np.zeros(np.shape(scores), dtype=int)
    reasons = []
    column = {tag: j for j, tag in enumerate(tag_names)}
    for i, row in enumerate(scores):
        tags, _, _, reason = suggest_from_scores(row, tag_names, model_settings, n_words_list[i], min_words)
        for t in tags:
            Y_pred[i, column[t.tag]] = 1
        reasons.append(reason)
    return Y_pred, reasons

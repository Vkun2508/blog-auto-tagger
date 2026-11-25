from tagger.suggest import REVIEW_REASON_TEXT, apply_settings, suggest_from_scores

TAGS = ["A", "B", "C", "D", "E", "F", "G"]
SETTINGS = {"threshold": 0.5, "max_tags": 2, "per_tag_thresholds": {}}


def test_only_approved_tags_limited_counts():
    scores = [0.9, 0.8, 0.7, 0.6, 0.2, 0.1, 0.05]
    tags, candidates, needs_review, reason = suggest_from_scores(scores, TAGS, SETTINGS, n_words=100, min_words=30)
    assert [t.tag for t in tags] == ["A", "B"]          # ranked, at most max_tags
    assert len(candidates) == 5
    assert all(c.tag in TAGS for c in candidates)
    assert needs_review is False and reason is None


def test_empty_post_has_no_tags():
    tags, candidates, needs_review, reason = suggest_from_scores([0.9] * 7, TAGS, SETTINGS, n_words=0, min_words=30)
    assert tags == [] and candidates == []
    assert needs_review and reason == "empty"


def test_short_post_still_gets_tags_but_review():
    tags, candidates, needs_review, reason = suggest_from_scores([0.9] + [0.1] * 6, TAGS, SETTINGS, 5, 30)
    assert [t.tag for t in tags] == ["A"]
    assert len(candidates) == 5
    assert needs_review and reason == "too_short"


def test_nothing_passes_low_confidence():
    tags, candidates, needs_review, reason = suggest_from_scores([0.3] * 7, TAGS, SETTINGS, 100, 30)
    assert tags == [] and len(candidates) == 5
    assert needs_review and reason == "low_confidence"
    assert reason in REVIEW_REASON_TEXT


def test_per_tag_threshold_honoured():
    settings = {"threshold": 0.5, "max_tags": 3, "per_tag_thresholds": {"A": 0.95, "C": 0.2}}
    scores = [0.9, 0.6, 0.3, 0.1, 0.1, 0.1, 0.1]
    tags, _, _, _ = suggest_from_scores(scores, TAGS, settings, 100, 30)
    assert [t.tag for t in tags] == ["B", "C"]         # A needs 0.95, C only 0.2


def test_apply_settings_matches_single_post_logic():
    scores = [[0.9, 0.8, 0.7, 0.1, 0.1, 0.1, 0.1], [0.1] * 7]
    Y_pred, reasons = apply_settings(scores, TAGS, SETTINGS, [100, 100], 30)
    assert Y_pred.tolist() == [[1, 1, 0, 0, 0, 0, 0], [0] * 7]
    assert reasons == [None, "low_confidence"]

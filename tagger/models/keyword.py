"""Keyword baseline: no learning, just counts each tag's keywords (from the taxonomy) in the post."""

import json
import math
import re

import numpy as np

from tagger.models.base import TagModel
from tagger.taxonomy import load_taxonomy

MAX_BODY_HITS = 10      # a post repeating "python" 200 times is not 20x more Python than one saying it 10 times


def keyword_pattern(keywords):
    """Build one regex that finds any of the keywords as whole words.

    Lowercase keywords match any case; keywords with a capital letter match that exact case,
    so "AI" does not fire on "said" and "in Go" does not fire on "go to".
    """
    any_case = sorted((k for k in keywords if k == k.lower()), key=len, reverse=True)
    exact_case = sorted((k for k in keywords if k != k.lower()), key=len, reverse=True)
    options = []
    if any_case:
        options.append("(?i:" + "|".join(re.escape(k) for k in any_case) + ")")
    options += [re.escape(k) for k in exact_case]
    # (?<!\w) and (?!\w) mean "no letter, digit or _ right before/after": a whole-word match.
    return re.compile(r"(?<!\w)(?:" + "|".join(options) + r")(?!\w)")


def build_tag_patterns(taxonomy):
    """One compiled regex per tag, built once and reused for every post."""
    return {tag: keyword_pattern(taxonomy.keywords[tag]) for tag in taxonomy.tags}


def count_keyword_hits(pattern, text):
    """How many keyword occurrences are in the text."""
    return len(pattern.findall(text or ""))


class KeywordModel(TagModel):
    name = "keyword"

    def __init__(self, taxonomy=None):
        self.taxonomy = taxonomy or load_taxonomy()
        self.patterns = build_tag_patterns(self.taxonomy)

    def fit(self, titles, bodies, Y):
        return self              # nothing to learn: the keywords come from the taxonomy

    def score_post(self, title, body):
        row = []
        for tag in self.taxonomy.tags:
            pattern = self.patterns[tag]
            title_hits = len({m.lower() for m in pattern.findall(title or "")})   # distinct keywords
            body_hits = min(count_keyword_hits(pattern, body), MAX_BODY_HITS)
            # A title mention counts 3x a body mention; 1 - exp(-x) turns any count into 0..1.
            row.append(1 - math.exp(-(3 * title_hits + body_hits) / 4))
        return row

    def predict_scores(self, titles, bodies):
        rows = [self.score_post(t, b) for t, b in zip(titles, bodies)]
        return np.array(rows, dtype=np.float32).reshape(len(rows), len(self.taxonomy.tags))

    def save(self, models_dir):
        config = {"model": "keyword", "note": "No learned weights: keywords come from taxonomy/taxonomy.yaml."}
        (models_dir / "keyword.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, models_dir):
        return cls()

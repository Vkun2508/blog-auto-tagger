"""Hybrid model: a weighted average of the TF-IDF and semantic scores.

TF-IDF is good at exact words ("kubectl", "pandas"); the semantic model is good at meaning
when the words differ. Averaging lets each cover the other's blind spots.
"""

import json

from tagger.models.base import TagModel
from tagger.models.semantic import SemanticModel
from tagger.models.tfidf import TfidfModel


class HybridModel(TagModel):
    name = "hybrid"

    def __init__(self, weight_tfidf=0.5, tfidf_model=None, semantic_model=None):
        self.weight_tfidf = weight_tfidf
        self.tfidf_model = tfidf_model or TfidfModel()
        self.semantic_model = semantic_model or SemanticModel()

    def fit(self, titles, bodies, Y, embeddings=None):
        self.tfidf_model.fit(titles, bodies, Y)
        self.semantic_model.fit(titles, bodies, Y, embeddings=embeddings)
        return self

    def predict_scores(self, titles, bodies, embeddings=None):
        tfidf_scores = self.tfidf_model.predict_scores(titles, bodies)
        semantic_scores = self.semantic_model.predict_scores(titles, bodies, embeddings=embeddings)
        return self.weight_tfidf * tfidf_scores + (1 - self.weight_tfidf) * semantic_scores

    def save(self, models_dir):
        # The two parts are saved by their own save(); this file only says how to mix them.
        config = {"model": "hybrid", "weight_tfidf": self.weight_tfidf,
                  "components": ["tfidf.joblib", "semantic.joblib"]}
        (models_dir / "hybrid.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, models_dir):
        config = json.loads((models_dir / "hybrid.json").read_text(encoding="utf-8"))
        return cls(config["weight_tfidf"], TfidfModel.load(models_dir), SemanticModel.load(models_dir))

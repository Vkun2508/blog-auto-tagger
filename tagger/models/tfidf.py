"""TF-IDF + Logistic Regression: learns which words and word pairs predict each tag."""

import joblib
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from tagger.models.base import TagModel


def make_vectorizer(max_features):
    # Single words and word pairs; sublinear_tf stops a word repeated 50 times from dominating.
    return TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2,
                           max_features=max_features, stop_words="english", dtype=np.float32)


class TfidfModel(TagModel):
    name = "tfidf"

    def __init__(self, title_weight=1.0, C=4.0):
        self.title_weight = title_weight   # how much louder title words are than body words
        self.C = C                         # higher C = weaker regularisation
        self.title_vectorizer = make_vectorizer(20000)
        self.body_vectorizer = make_vectorizer(50000)
        # One yes/no classifier per tag; liblinear is fast for sparse text features.
        self.classifier = OneVsRestClassifier(LogisticRegression(C=C, solver="liblinear"))

    def features(self, titles, bodies):
        X_title = self.title_vectorizer.transform(titles)
        X_body = self.body_vectorizer.transform(bodies)
        return hstack([self.title_weight * X_title, X_body]).tocsr()

    def fit(self, titles, bodies, Y):
        self.title_vectorizer.fit(titles)
        self.body_vectorizer.fit(bodies)
        self.classifier.fit(self.features(titles, bodies), np.asarray(Y))
        return self

    def predict_scores(self, titles, bodies):
        return self.classifier.predict_proba(self.features(titles, bodies)).astype(np.float32)

    def save(self, models_dir):
        for vectorizer in [self.title_vectorizer, self.body_vectorizer]:
            # stop_words_ lists every word pair that was cut; it is only for inspection and is huge.
            if hasattr(vectorizer, "stop_words_"):
                del vectorizer.stop_words_
        for estimator in self.classifier.estimators_:
            # float32 halves the file size; the lost precision is far below anything that changes a score.
            estimator.coef_ = estimator.coef_.astype(np.float32)
        joblib.dump({"title_weight": self.title_weight, "C": self.C,
                     "title_vectorizer": self.title_vectorizer, "body_vectorizer": self.body_vectorizer,
                     "classifier": self.classifier},
                    models_dir / "tfidf.joblib", compress=3)

    @classmethod
    def load(cls, models_dir):
        saved = joblib.load(models_dir / "tfidf.joblib")
        model = cls(saved["title_weight"], saved["C"])
        model.title_vectorizer = saved["title_vectorizer"]
        model.body_vectorizer = saved["body_vectorizer"]
        model.classifier = saved["classifier"]
        return model

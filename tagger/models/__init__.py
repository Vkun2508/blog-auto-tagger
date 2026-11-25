"""The four tagging models and how to load a saved one by name."""

from pathlib import Path

MODEL_NAMES = ["keyword", "tfidf", "semantic", "hybrid"]

MODEL_LABELS = {
    "keyword": "Keyword baseline",
    "tfidf": "TF-IDF + Logistic Regression",
    "semantic": "Semantic (MiniLM embeddings)",
    "hybrid": "Hybrid (TF-IDF + Semantic)",
}

# Files in models/ that must exist before a model can be loaded.
MODEL_FILES = {
    "keyword": ["keyword.json"],
    "tfidf": ["tfidf.joblib"],
    "semantic": ["semantic.joblib"],
    "hybrid": ["hybrid.json", "tfidf.joblib", "semantic.joblib"],
}


def load_model(name, models_dir):
    """Load a saved model by its name ("keyword", "tfidf", "semantic" or "hybrid")."""
    # Imports happen here so that, for example, using the keyword model never loads scikit-learn.
    if name == "keyword":
        from tagger.models.keyword import KeywordModel as model_class
    elif name == "tfidf":
        from tagger.models.tfidf import TfidfModel as model_class
    elif name == "semantic":
        from tagger.models.semantic import SemanticModel as model_class
    elif name == "hybrid":
        from tagger.models.hybrid import HybridModel as model_class
    else:
        raise ValueError(f"unknown model '{name}'; choose one of {MODEL_NAMES}")
    return model_class.load(Path(models_dir))

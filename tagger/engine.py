"""TaggingEngine: the one object the app and the scripts use to suggest tags for a post."""

import json
from functools import lru_cache
from pathlib import Path

from tagger import PROJECT_DIR
from tagger.cleaning import clean_post, split_sentences, word_count
from tagger.models import MODEL_FILES, MODEL_LABELS, MODEL_NAMES, load_model
from tagger.models.keyword import build_tag_patterns, count_keyword_hits
from tagger.suggest import Suggestion, suggest_from_scores
from tagger.taxonomy import load_taxonomy

DEFAULT_MODELS_DIR = PROJECT_DIR / "models"
MAX_SNIPPET_CHARS = 240
NEEDS_ENCODER = {"semantic", "hybrid"}


@lru_cache(maxsize=1)
def encoder_library_works():
    """True if sentence-transformers can be imported (checked once, it takes a few seconds)."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except Exception:
        return False


class TaggingEngine:
    def __init__(self, models_dir, taxonomy, settings):
        self.models_dir = Path(models_dir)
        self.taxonomy = taxonomy
        self.settings = settings
        self.default_model = settings["default_model"]
        self._models = {}                                  # loaded on first use, then kept
        self._patterns = build_tag_patterns(taxonomy)      # keyword regexes, for snippets

    @classmethod
    def load(cls, models_dir=None):
        models_dir = Path(models_dir or DEFAULT_MODELS_DIR)
        settings_path = models_dir / "settings.json"
        if not settings_path.exists():
            raise FileNotFoundError(f"{settings_path} not found: run scripts/03_train.py and "
                                    "scripts/04_tune_thresholds.py first")
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        return cls(models_dir, load_taxonomy(), settings)

    @property
    def available_models(self):
        """Models whose files exist (and, for semantic/hybrid, whose library can be imported)."""
        names = []
        for name in MODEL_NAMES:
            files_exist = all((self.models_dir / f).exists() for f in MODEL_FILES[name])
            tuned = name in self.settings.get("models", {})
            if files_exist and tuned and (name not in NEEDS_ENCODER or encoder_library_works()):
                names.append(name)
        return names

    def model_label(self, name):
        return MODEL_LABELS.get(name, name)

    def get_model(self, name):
        if name not in self.available_models:
            raise ValueError(f"model '{name}' is not available; available: {self.available_models}")
        if name not in self._models:
            self._models[name] = load_model(name, self.models_dir)
        return self._models[name]

    def supporting_snippet(self, body_clean, title_clean, tag):
        """The sentence with the most keyword matches for the tag (earliest on ties), or None."""
        pattern = self._patterns[tag]
        best_sentence, best_hits = None, 0
        for sentence in split_sentences(title_clean + "\n" + body_clean):
            hits = count_keyword_hits(pattern, sentence)
            if hits > best_hits:                 # ">" not ">=", so the earliest sentence wins ties
                best_sentence, best_hits = sentence, hits
        if best_sentence and len(best_sentence) > MAX_SNIPPET_CHARS:
            best_sentence = best_sentence[:MAX_SNIPPET_CHARS - 3] + "..."
        return best_sentence

    def build_suggestion(self, score_row, name, title_clean, body_clean):
        n_words = word_count(title_clean) + word_count(body_clean)
        tags, candidates, needs_review, reason = suggest_from_scores(
            score_row, self.taxonomy.tags, self.settings["models"][name], n_words, self.settings["min_words"])
        for candidate in candidates:     # tags are the same objects, so they get snippets too
            candidate.snippet = self.supporting_snippet(body_clean, title_clean, candidate.tag)
        return Suggestion(tags, candidates, needs_review, reason, name, n_words, title_clean, body_clean)

    def suggest(self, title, body, model=None):
        """Suggest tags for one post. Accepts plain text, Markdown or HTML (it cleans the input)."""
        return self.suggest_many([(None, title, body)], model)[0][1]

    def suggest_many(self, items, model=None):
        """items = [(post_id, title, body), ...]. Scores all posts in one batch."""
        name = model or self.default_model
        if name not in self.available_models:
            raise ValueError(f"model '{name}' is not available; available: {self.available_models}")
        cleaned =[clean_post(title, body) for _, title, body in items]
        # Posts with no words at all are never sent to the model ("empty" review instead).
        to_score = [i for i, (t, b) in enumerate(cleaned) if word_count(t) + word_count(b) > 0]
        scores = {}
        if to_score:
            model_object = self.get_model(name)
            rows = model_object.predict_scores([cleaned[i][0] for i in to_score], [cleaned[i][1] for i in to_score])
            scores = dict(zip(to_score, rows))
        results = []
        for i, (post_id, _, _) in enumerate(items):
            title_clean, body_clean = cleaned[i]
            if i in scores:
                suggestion = self.build_suggestion(scores[i], name, title_clean, body_clean)
            else:
                suggestion = Suggestion([], [], True, "empty", name, 0, title_clean, body_clean)
            results.append((post_id, suggestion))
        return results

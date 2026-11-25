"""Semantic model: a small pre-trained sentence encoder (MiniLM) turns text into meaning vectors,
then one logistic regression per tag learns which vectors belong to which tag.

sentence-transformers (and torch) are imported only when the encoder is first needed,
so the rest of the project works on a machine without them.
"""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from tagger import PROJECT_DIR
from tagger.models.base import TagModel

ENCODER_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_WORDS = 150        # MiniLM reads at most 256 word pieces, so long bodies are cut into chunks
MAX_CHUNKS = 8           # at most 8 chunks per post keeps encoding time reasonable
BATCH_SIZE = 64
POOLINGS = ["title_only", "body_mean", "title+body_mean", "title+body_max"]
CACHE_DIR = PROJECT_DIR / "data" / "cache"

_encoders = {}           # encoder name -> loaded encoder (loading takes seconds, so do it once)


def get_encoder(encoder_name=ENCODER_NAME):
    if encoder_name not in _encoders:
        from sentence_transformers import SentenceTransformer
        _encoders[encoder_name] = SentenceTransformer(encoder_name, device="cpu")
    return _encoders[encoder_name]


def split_into_chunks(body, chunk_words=CHUNK_WORDS, max_chunks=MAX_CHUNKS):
    """Cut a body into pieces of about chunk_words words; keep at most max_chunks spread evenly."""
    words = (body or "").split()
    chunks = [" ".join(words[i:i + chunk_words]) for i in range(0, len(words), chunk_words)]
    if len(chunks) > max_chunks:
        # Evenly spread picks (start, middle, end) represent the whole post better than the first 8.
        keep = np.linspace(0, len(chunks) - 1, max_chunks).round().astype(int)
        chunks = [chunks[i] for i in keep]
    return chunks


def embed_posts(titles, bodies, encoder_name=ENCODER_NAME, chunk_words=CHUNK_WORDS, max_chunks=MAX_CHUNKS):
    """Encode posts. Returns {"title", "body_mean", "body_max"}, each an array [posts x 384]."""
    encoder = get_encoder(encoder_name)
    title_vectors = encoder.encode(list(titles), batch_size=BATCH_SIZE, normalize_embeddings=True)

    # Encode every chunk of every post in one go (much faster than post by post).
    all_chunks, owner = [], []
    for post_number, body in enumerate(bodies):
        for chunk in split_into_chunks(body, chunk_words, max_chunks):
            all_chunks.append(chunk)
            owner.append(post_number)
    owner = np.array(owner, dtype=int)
    chunk_vectors = encoder.encode(all_chunks, batch_size=BATCH_SIZE, normalize_embeddings=True,
                                   show_progress_bar=len(all_chunks) > 1000)

    size = title_vectors.shape[1]
    body_mean = np.zeros((len(titles), size), dtype=np.float32)   # an empty body stays all zeros
    body_max = np.zeros((len(titles), size), dtype=np.float32)
    for post_number in range(len(titles)):
        rows = chunk_vectors[owner == post_number] if len(all_chunks) else []
        if len(rows):
            body_mean[post_number] = rows.mean(axis=0)
            body_max[post_number] = rows.max(axis=0)
    return {"title": title_vectors.astype(np.float32), "body_mean": body_mean, "body_max": body_max}


def cached_embeddings(split, ids, titles, bodies):
    """Embeddings for one data split, saved to data/cache so experiments do not re-encode.

    The cache is reused only if it holds the same post ids made with the same settings.
    """
    path = CACHE_DIR / f"embeddings_{split}.npz"
    settings = np.array([ENCODER_NAME, str(CHUNK_WORDS), str(MAX_CHUNKS)])
    if path.exists():
        saved = np.load(path)
        if np.array_equal(saved["ids"], np.asarray(ids)) and np.array_equal(saved["settings"], settings):
            return {"title": saved["title"], "body_mean": saved["body_mean"], "body_max": saved["body_max"]}
    print(f"   encoding {len(ids)} {split} posts with {ENCODER_NAME} (cached afterwards)...")
    embeddings = embed_posts(titles, bodies)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(path, ids=np.asarray(ids), settings=settings, **embeddings)
    return embeddings


def pool_features(embeddings, pooling):
    """Build the classifier input from the embeddings, as chosen by `pooling`."""
    if pooling == "title_only":
        return embeddings["title"]
    if pooling == "body_mean":
        return embeddings["body_mean"]
    if pooling == "title+body_mean":
        return np.hstack([embeddings["title"], embeddings["body_mean"]])
    if pooling == "title+body_max":
        return np.hstack([embeddings["title"], embeddings["body_max"]])
    raise ValueError(f"unknown pooling '{pooling}'; choose one of {POOLINGS}")


class SemanticModel(TagModel):
    name = "semantic"

    def __init__(self, pooling="title+body_mean", C=4.0, encoder_name=ENCODER_NAME,
                 chunk_words=CHUNK_WORDS, max_chunks=MAX_CHUNKS):
        self.pooling = pooling
        self.C = C
        self.encoder_name = encoder_name
        self.chunk_words = chunk_words
        self.max_chunks = max_chunks
        self.head = OneVsRestClassifier(LogisticRegression(C=C, max_iter=2000))

    def embed(self, titles, bodies, embeddings):
        """Use precomputed embeddings when given (from the cache), else encode now."""
        if embeddings is None:
            embeddings = embed_posts(titles, bodies, self.encoder_name, self.chunk_words, self.max_chunks)
        return pool_features(embeddings, self.pooling)

    def fit(self, titles, bodies, Y, embeddings=None):
        self.head.fit(self.embed(titles, bodies, embeddings), np.asarray(Y))
        return self

    def predict_scores(self, titles, bodies, embeddings=None):
        return self.head.predict_proba(self.embed(titles, bodies, embeddings)).astype(np.float32)

    def save(self, models_dir):
        # Only the small classifier head and settings are saved; the encoder is downloaded by name.
        joblib.dump({"head": self.head, "encoder_name": self.encoder_name, "chunk_words": self.chunk_words,
                     "max_chunks": self.max_chunks, "pooling": self.pooling, "C": self.C},
                    models_dir / "semantic.joblib", compress=3)

    @classmethod
    def load(cls, models_dir):
        saved = joblib.load(models_dir / "semantic.joblib")
        model = cls(saved["pooling"], saved["C"], saved["encoder_name"], saved["chunk_words"], saved["max_chunks"])
        model.head = saved["head"]
        return model

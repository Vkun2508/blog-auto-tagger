"""Read the prepared dataset (data/processed/posts.jsonl) for the training and evaluation scripts."""

import json

import numpy as np

from tagger import PROJECT_DIR

POSTS_FILE = PROJECT_DIR / "data" / "processed" / "posts.jsonl"


def load_posts(split=None):
    """All prepared posts, or only those of one split ("train", "val" or "test")."""
    if not POSTS_FILE.exists():
        raise FileNotFoundError(f"{POSTS_FILE} not found: run scripts/02_prepare_data.py first")
    with open(POSTS_FILE, encoding="utf-8") as f:
        posts = [json.loads(line) for line in f if line.strip()]
    return [p for p in posts if split is None or p["split"] == split]


def label_matrix(posts, tags):
    """0/1 matrix [posts x tags]: 1 where the post has that approved tag."""
    Y = np.zeros((len(posts), len(tags)), dtype=int)
    column = {tag: j for j, tag in enumerate(tags)}
    for i, post in enumerate(posts):
        for tag in post["tags"]:
            Y[i, column[tag]] = 1
    return Y


def predict_split(model, posts, split):
    """Scores for a whole split. Semantic and hybrid models reuse the cached embeddings."""
    titles = [p["title"] for p in posts]
    bodies = [p["body"] for p in posts]
    if model.name in ("semantic", "hybrid"):
        from tagger.models.semantic import cached_embeddings
        embeddings = cached_embeddings(split, [p["id"] for p in posts], titles, bodies)
        return model.predict_scores(titles, bodies, embeddings=embeddings)
    return model.predict_scores(titles, bodies)

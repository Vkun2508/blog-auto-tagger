"""Session 2: clean the raw posts, turn raw tags into approved labels, find near-duplicates
and split the data by date into train / validation / test.

Run (about 1 minute):   python scripts/02_prepare_data.py

Writes:
    data/processed/posts.jsonl      one cleaned, labelled post per line (not uploaded to GitHub)
    reports/tag_counts.csv          posts per approved tag and split
    reports/dataset_summary.md      what was cleaned, dropped, merged and split
    reports/figures/tag_counts.png  bar chart of tag counts by split
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

PROJECT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT))          # lets this script import the tagger package

from tagger.charts import COLORS, new_figure, save_figure  # noqa: E402
from tagger.cleaning import clean_post, word_count  # noqa: E402
from tagger.taxonomy import load_taxonomy  # noqa: E402

RAW_DIR = PROJECT / "data" / "raw"
POSTS_FILE = PROJECT / "data" / "processed" / "posts.jsonl"
REPORTS = PROJECT / "reports"
FIGURES = REPORTS / "figures"

DUPLICATE_SIMILARITY = 0.90     # cosine similarity of body TF-IDF vectors that counts as "the same post"
TRAIN_SHARE, VAL_SHARE = 0.70, 0.15
SPLITS = ["train", "val", "test"]


def load_raw_posts():
    posts = []
    for path in sorted(RAW_DIR.glob("*.json"), key=lambda p: int(p.stem)):
        posts.append(json.loads(path.read_text(encoding="utf-8")))
    return posts


def normalize_title(title):
    """Lowercase, no punctuation, single spaces: "Hello, World!" and "hello world" become equal."""
    return " ".join(re.sub(r"[^\w\s]", " ", title.lower()).split())


class UnionFind:
    """Keeps track of which posts belong to the same duplicate group.

    union(a, b) puts a and b in one group; find(a) returns the group's representative.
    """

    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, a):
        while self.parent[a] != a:
            self.parent[a] = self.parent[self.parent[a]]    # shortcut the path to speed up later finds
            a = self.parent[a]
        return a

    def union(self, a, b):
        root_a, root_b = self.find(a), self.find(b)
        if root_a != root_b:
            self.parent[max(root_a, root_b)] = min(root_a, root_b)


def link_same_value(groups, values, edges, reason):
    """Join posts that share the same (non-empty) value, e.g. the same normalized title."""
    first_with_value = {}
    for i, value in enumerate(values):
        if not value:
            continue
        if value in first_with_value:
            groups.union(first_with_value[value], i)
            edges.append((first_with_value[value], i, reason))
        else:
            first_with_value[value] = i


def link_similar_bodies(groups, bodies, edges):
    """Join posts whose cleaned bodies are almost the same text (cosine >= 0.90)."""
    vectors = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(bodies)
    block = 500                                   # compare 500 posts at a time to keep memory small
    for start in range(0, len(bodies), block):
        similarity = (vectors[start:start + block] @ vectors.T).toarray()
        for row, col in np.argwhere(similarity >= DUPLICATE_SIMILARITY):
            i = start + row
            if i < col:                           # each pair once, and never a post with itself
                groups.union(i, col)
                edges.append((i, col, "similar body"))


def assign_splits(posts):
    """Split by duplicate group in date order: oldest 70% of posts train, next 15% val, newest 15% test.

    A whole group always goes to one split, so a copy of a test post can never be in training.
    """
    group_posts = {}
    for i, post in enumerate(posts):
        group_posts.setdefault(post["group_id"], []).append(i)
    ordered = sorted(group_posts.values(),
                     key=lambda members: (min(posts[i]["published_at"] for i in members),
                                          min(posts[i]["id"] for i in members)))
    total, seen = len(posts), 0
    for members in ordered:
        if seen < TRAIN_SHARE * total:
            split = "train"
        elif seen < (TRAIN_SHARE + VAL_SHARE) * total:
            split = "val"
        else:
            split = "test"
        for i in members:
            posts[i]["split"] = split
        seen += len(members)


def tag_count_table(posts, taxonomy):
    alias_use = Counter(raw.strip().lower() for p in posts for raw in p["raw_tags"])
    rows = []
    for tag in taxonomy.tags:
        counts = {split: sum(1 for p in posts if p["split"] == split and tag in p["tags"]) for split in SPLITS}
        total = sum(counts.values())
        used = [f"{alias} ({alias_use[alias]})" for alias in taxonomy.aliases[tag] if alias_use[alias]]
        rows.append({"tag": tag, "total": total, **counts,
                     "too_few": counts["test"] < 20 or total < 80, "aliases_merged": "; ".join(used)})
    return pd.DataFrame(rows)


def draw_tag_counts(table):
    fig, ax = new_figure(9, 8)
    table = table.sort_values("total")
    left = np.zeros(len(table))
    for split, color in zip(SPLITS, COLORS):
        ax.barh(table["tag"], table[split], left=left, color=color, label=split,
                edgecolor="white", linewidth=1)
        left += table[split].to_numpy()
    ax.set_xlabel("Posts")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, loc="lower right")
    save_figure(fig, FIGURES / "tag_counts.png", "Posts per approved tag, by split")


def write_summary(raw_posts, posts, n_unlabelled, edges, table, taxonomy):
    generic = set(taxonomy.dropped_generic)
    lines = ["# Dataset summary", "", "Written by `scripts/02_prepare_data.py`.", "",
             "## Posts", "",
             f"- Raw posts downloaded: {len(raw_posts)}",
             f"- Dropped because none of their raw tags maps to an approved tag: {n_unlabelled}",
             f"- Posts kept: {len(posts)}", "",
             "| Split | Posts | Share | First post | Last post |", "|---|---|---|---|---|"]
    for split in SPLITS:
        dates = sorted(p["published_at"] for p in posts if p["split"] == split)
        lines.append(f"| {split} | {len(dates)} | {len(dates) / len(posts):.1%} | {dates[0][:10]} | {dates[-1][:10]} |")

    group_sizes = Counter(p["group_id"] for p in posts)
    duplicate_groups = [g for g, size in group_sizes.items() if size > 1]
    reasons = Counter(reason for _, _, reason in edges)
    lines += ["", "## Near-duplicates", "",
              "Posts are joined into one group if they have the same normalized title, the same "
              f"canonical URL, or bodies with TF-IDF cosine similarity >= {DUPLICATE_SIMILARITY}. "
              "A group is never split across train/val/test.", "",
              f"- Duplicate groups (2 or more posts): {len(duplicate_groups)}",
              f"- Posts inside those groups: {sum(group_sizes[g] for g in duplicate_groups)}",
              f"- Links found: same title {reasons['same title']}, same canonical URL "
              f"{reasons['same canonical URL']}, similar body {reasons['similar body']}", "",
              "Example pairs:", ""]
    for a, b, reason in edges[:3]:
        lines.append(f"- ({reason}) \"{posts[a]['title']}\"  /  \"{posts[b]['title']}\"")

    labels_per_post = Counter(len(p["tags"]) for p in posts)
    lines += ["", "## Approved labels per post", "", "| Labels | Posts |", "|---|---|"]
    lines += [f"| {n} | {labels_per_post[n]} |" for n in sorted(labels_per_post)]
    lines.append(f"\nAverage labels per post: {np.mean([len(p['tags']) for p in posts]):.2f}")

    generic_use = Counter(t for p in raw_posts for t in {r.strip().lower() for r in p["tags"]} if t in generic)
    with_generic = sum(1 for p in raw_posts if any(r.strip().lower() in generic for r in p["tags"]))
    lines += ["", "## Generic tags dropped", "",
              f"{with_generic / len(raw_posts):.1%} of raw posts ({with_generic} of {len(raw_posts)}) used at "
              "least one of their 4 tag slots for a generic tag.", "",
              "| Generic tag | Posts using it |", "|---|---|"]
    lines += [f"| {tag} | {count} |" for tag, count in generic_use.most_common()]

    few = table[table["too_few"]]["tag"].tolist()
    lines += ["", "## Tag counts", "", "See `reports/tag_counts.csv` and `reports/figures/tag_counts.png`.",
              f"Tags flagged too_few (test < 20 or total < 80): {', '.join(few) or 'none'}", ""]
    (REPORTS / "dataset_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    taxonomy = load_taxonomy()
    raw_posts = load_raw_posts()
    print(f"Loaded {len(raw_posts)} raw posts")

    # Step 1: clean text and map raw tags to approved tags; keep posts with at least one approved tag.
    posts, n_unlabelled = [], 0
    for raw in raw_posts:
        tags = taxonomy.map_raw_tags(raw["tags"])
        if not tags:
            n_unlabelled += 1
            continue
        title, body = clean_post(raw["title"], raw["body_markdown"])
        posts.append({"id": raw["id"], "title": title, "body": body, "url": raw["url"],
                      "published_at": raw["published_at"], "raw_tags": raw["tags"], "tags": tags,
                      "canonical_url": raw.get("canonical_url") or "",
                      "n_words": word_count(title) + word_count(body)})
    print(f"Cleaned {len(posts)} posts; dropped {n_unlabelled} with no approved tag")

    # Step 2: near-duplicate groups.
    groups, edges = UnionFind(len(posts)), []
    link_same_value(groups, [normalize_title(p["title"]) for p in posts], edges, "same title")
    link_same_value(groups, [p["canonical_url"].strip().rstrip("/").lower() for p in posts], edges,
                    "same canonical URL")
    link_similar_bodies(groups, [p["body"] for p in posts], edges)
    for i, post in enumerate(posts):
        post["group_id"] = int(posts[groups.find(i)]["id"])    # a group is named after its first post's id
    print(f"Found {len(edges)} duplicate links, {len(set(p['group_id'] for p in posts))} groups")

    # Step 3: date-ordered split by group.
    assign_splits(posts)
    for split in SPLITS:
        print(f"   {split}: {sum(1 for p in posts if p['split'] == split)} posts")

    # Step 4: write everything.
    POSTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(POSTS_FILE, "w", encoding="utf-8") as f:
        for p in posts:
            row = {k: p[k] for k in ["id", "title", "body", "url", "published_at", "raw_tags", "tags",
                                     "group_id", "split", "n_words"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {POSTS_FILE}")

    REPORTS.mkdir(exist_ok=True)
    table = tag_count_table(posts, taxonomy)
    table.to_csv(REPORTS / "tag_counts.csv", index=False)
    write_summary(raw_posts, posts, n_unlabelled, edges, table, taxonomy)
    draw_tag_counts(table)
    print("Wrote reports/tag_counts.csv and reports/dataset_summary.md")


if __name__ == "__main__":
    main()

"""Session 1: download blog posts from DEV.to using its free public API.

Quick test (about 20 seconds):   python scripts/01_download_devto.py --test
Full download (about 1.5 hours): python scripts/01_download_devto.py

It is safe to stop (Ctrl+C) and run again: posts already saved are skipped.
"""

import csv
import json
import sys
import time
from pathlib import Path

import requests

API = "https://dev.to/api/articles"
HEADERS = {"User-Agent": "blog-auto-tagger-student-project"}

# Where things get saved. Path(__file__) is this script; .parent.parent is the project folder.
PROJECT = Path(__file__).parent.parent
RAW_DIR = PROJECT / "data" / "raw"          # one JSON file per post (not uploaded to GitHub)
IDS_FILE = PROJECT / "data" / "article_ids.csv"  # the list of post IDs (uploaded to GitHub)

# DEV.to tags we search for. Several of them mean the same topic (ai / llm,
# security / cybersecurity). We merge those into approved tags in Session 2.
SEED_TAGS = [
    "python", "javascript", "typescript", "react", "api", "security", "cybersecurity",
    "machinelearning", "ai", "llm", "datascience", "devops", "docker", "kubernetes",
    "aws", "cloud", "database", "sql", "postgres", "java", "go", "rust",
    "mobile", "android", "flutter", "git", "github", "testing", "career",
]
PAGES_PER_TAG = 2      # each page has up to 100 posts
SECONDS_BETWEEN_REQUESTS = 1.0   # be polite: about one request per second


def get_json(url, params=None):
    """Ask the API for a URL and return the answer as Python data.

    If the API says "too many requests" (code 429), wait and try again.
    Returns None if the post no longer exists or the API keeps failing.
    """
    for attempt in range(5):
        time.sleep(SECONDS_BETWEEN_REQUESTS)
        try:
            response = requests.get(url, params=params, headers=HEADERS, timeout=30)
        except requests.RequestException as error:
            print(f"   network problem ({error}); retrying...")
            continue
        if response.status_code == 200:
            return response.json()
        if response.status_code == 404:      # post was deleted
            return None
        if response.status_code == 429:      # we went too fast
            wait = 30 * (attempt + 1)
            print(f"   API asked us to slow down; waiting {wait} seconds...")
            time.sleep(wait)
            continue
        print(f"   unexpected answer {response.status_code}; retrying...")
    return None


def list_post_ids(tag, pages):
    """Return the IDs of the most popular posts of the last year for one tag."""
    ids = []
    for page in range(1, pages + 1):
        # top=365 means "most popular in the last 365 days"
        posts = get_json(API, {"tag": tag, "top": 365, "per_page": 100, "page": page})
        if not posts:
            break
        for post in posts:
            ids.append(post["id"])
    return ids


def download_post(post_id):
    """Download one full post and save only the fields we need. Returns True if saved."""
    post = get_json(f"{API}/{post_id}")
    if post is None:
        return False
    tags = post.get("tags") or []
    if isinstance(tags, str):                # the API sometimes sends "a, b" instead of a list
        tags = [t.strip() for t in tags.split(",") if t.strip()]
    wanted = {
        "id": post["id"],
        "title": post.get("title", ""),
        "body_markdown": post.get("body_markdown", ""),
        "tags": tags,
        "published_at": post.get("published_at", ""),
        "url": post.get("url", ""),
        "canonical_url": post.get("canonical_url", ""),
        "author": (post.get("user") or {}).get("username", ""),
    }
    (RAW_DIR / f"{post_id}.json").write_text(json.dumps(wanted), encoding="utf-8")
    return True


def main():
    test_mode = "--test" in sys.argv
    seed_tags = SEED_TAGS[:2] if test_mode else SEED_TAGS
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: collect post IDs for every tag. A set keeps each ID only once.
    all_ids = set()
    for tag in seed_tags:
        ids = list_post_ids(tag, pages=1 if test_mode else PAGES_PER_TAG)
        if test_mode:
            ids = ids[:5]
        all_ids.update(ids)
        print(f"tag '{tag}': {len(ids)} posts found, {len(all_ids)} unique so far")

    # Step 2: download each post, skipping ones we already have.
    todo = [i for i in sorted(all_ids) if not (RAW_DIR / f"{i}.json").exists()]
    print(f"\n{len(all_ids) - len(todo)} already downloaded, {len(todo)} to download")
    for number, post_id in enumerate(todo, start=1):
        saved = download_post(post_id)
        if number % 25 == 0 or number == len(todo) or not saved:
            status = "" if saved else f" (post {post_id} skipped)"
            print(f"   {number}/{len(todo)} done{status}")

    # Step 3: write the list of every post we have, so others can rebuild the same dataset.
    saved_ids = sorted(int(p.stem) for p in RAW_DIR.glob("*.json"))
    if not test_mode:
        with open(IDS_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["article_id"])
            for post_id in saved_ids:
                writer.writerow([post_id])
    print(f"\nFinished. {len(saved_ids)} posts are saved in {RAW_DIR}")


if __name__ == "__main__":
    main()

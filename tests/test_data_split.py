"""Checks on the real prepared dataset (skipped until scripts/02_prepare_data.py has run)."""

from collections import defaultdict

import pytest

from tagger.dataset import POSTS_FILE, load_posts

if not POSTS_FILE.exists():
    pytest.skip("data/processed/posts.jsonl not built yet", allow_module_level=True)

SPLIT_ORDER = ["train", "val", "test"]


def test_no_group_spans_two_splits():
    splits_of_group = defaultdict(set)
    for post in load_posts():
        splits_of_group[post["group_id"]].add(post["split"])
    assert all(len(splits) == 1 for splits in splits_of_group.values())


def test_splits_are_in_date_order_by_group():
    group_date, group_split = {}, {}
    for post in load_posts():
        g = post["group_id"]
        group_date[g] = min(group_date.get(g, post["published_at"]), post["published_at"])
        group_split[g] = post["split"]
    for earlier, later in zip(SPLIT_ORDER, SPLIT_ORDER[1:]):
        latest_earlier = max(d for g, d in group_date.items() if group_split[g] == earlier)
        first_later = min(d for g, d in group_date.items() if group_split[g] == later)
        assert latest_earlier <= first_later


def test_every_post_has_an_approved_tag():
    assert all(post["tags"] for post in load_posts())

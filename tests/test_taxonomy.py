import pytest

from tagger.taxonomy import load_taxonomy


def test_loads_22_tags():
    taxonomy = load_taxonomy()
    assert len(taxonomy.tags) == 22
    assert taxonomy.tags[0] == "Python"
    assert "webdev" in taxonomy.dropped_generic


def test_aliases_merge_into_one_tag():
    taxonomy = load_taxonomy()
    assert taxonomy.map_raw_tags(["ML"]) == ["Machine Learning"]
    assert taxonomy.map_raw_tags([" machinelearning ", "ml"]) == ["Machine Learning"]


def test_unknown_and_generic_tags_dropped_and_order_fixed():
    taxonomy = load_taxonomy()
    result = taxonomy.map_raw_tags(["webdev", "notatag", "docker", "python", "beginners"])
    assert result == ["Python", "Docker & Kubernetes"]      # taxonomy order, not input order
    assert taxonomy.map_raw_tags([]) == []
    assert taxonomy.map_raw_tags(None) == []


def test_default_taxonomy_is_read_once():
    assert load_taxonomy() is load_taxonomy()


def test_duplicate_alias_raises(tmp_path):
    yaml_text = """
tags:
  - name: A
    aliases: [shared]
  - name: B
    aliases: [shared]
dropped_generic: []
"""
    path = tmp_path / "taxonomy.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_taxonomy(path)

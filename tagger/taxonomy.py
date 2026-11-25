"""Load the approved tag taxonomy (taxonomy/taxonomy.yaml) and map raw author tags onto it."""

from dataclasses import dataclass
from functools import lru_cache

import yaml

from tagger import PROJECT_DIR

DEFAULT_TAXONOMY_PATH = PROJECT_DIR / "taxonomy" / "taxonomy.yaml"


@dataclass
class Taxonomy:
    tags: list[str]                  # approved tag names, in the fixed column order
    definitions: dict                # tag -> definition sentence
    examples: dict                   # tag -> example titles
    aliases: dict[str, list]         # tag -> raw DEV.to tags that merge into it
    keywords: dict[str, list]        # tag -> words the keyword baseline looks for
    alias_to_tag: dict[str, str]     # raw tag -> approved tag (the reverse of aliases)
    dropped_generic: list            # raw tags we deliberately ignore

    def map_raw_tags(self, raw_tags):
        """Turn messy raw tags into approved tags (unknown and generic tags are dropped).

        The result has no duplicates and follows the taxonomy order, so the same
        raw tags always give the same list.
        """
        found = set()
        for raw in raw_tags or []:
            tag = self.alias_to_tag.get(str(raw).strip().lower())
            if tag:
                found.add(tag)
        return [tag for tag in self.tags if tag in found]


def load_taxonomy(path=None):
    """Read the taxonomy YAML file. Raises ValueError if one alias points to two tags.

    The default file is read once and then reused (callers must not change the returned object).
    """
    if path is None:
        return load_default_taxonomy()
    return read_taxonomy_file(path)


@lru_cache(maxsize=1)
def load_default_taxonomy():
    # Cached because feedback records and models ask for the taxonomy many times.
    return read_taxonomy_file(DEFAULT_TAXONOMY_PATH)


def read_taxonomy_file(path):
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    tags, definitions, examples, aliases, keywords = [], {}, {}, {}, {}
    alias_to_tag = {}
    for entry in data["tags"]:
        name = entry["name"]
        tags.append(name)
        definitions[name] = entry.get("definition", "")
        examples[name] = list(entry.get("examples") or [])
        aliases[name] = [str(a).strip().lower() for a in entry.get("aliases") or []]
        keywords[name] = [str(k) for k in entry.get("keywords") or []]
        for alias in aliases[name]:
            # An alias that merges into two different tags would make labels ambiguous.
            if alias in alias_to_tag and alias_to_tag[alias] != name:
                raise ValueError(f"alias '{alias}' maps to both '{alias_to_tag[alias]}' and '{name}'")
            alias_to_tag[alias] = name

    dropped = [str(t).strip().lower() for t in data.get("dropped_generic") or []]
    return Taxonomy(tags, definitions, examples, aliases, keywords, alias_to_tag, dropped)

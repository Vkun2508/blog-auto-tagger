"""Session 0: a Python refresher using our project's own idea (tagging blog posts).

Run it with:   python scripts/00_python_refresher.py
Read each section, then look at what it printed.
"""

# --- 1. Variables: a name that holds a value -------------------------------
title = "Securing a Python API with tokens"
word_count = 850
print("1. Title:", title, "| words:", word_count)

# --- 2. Lists: several values in order -------------------------------------
# One post can have SEVERAL tags. That is what "multi-label" means.
tags = ["Python", "API Development", "Application Security"]
print("2. This post has", len(tags), "tags. The first is", tags[0])

# --- 3. Dictionaries: look up a value by a key ------------------------------
# An "alias map" turns the messy tags authors type into our approved tags.
alias_map = {
    "py": "Python",
    "python3": "Python",
    "ml": "Machine Learning",
    "machinelearning": "Machine Learning",
}
print("3. The alias 'ml' means:", alias_map["ml"])


# --- 4. Functions: a reusable recipe ----------------------------------------
def normalize_tag(raw_tag):
    """Turn a raw tag into an approved tag, or None if we don't know it."""
    cleaned = raw_tag.lower().strip()  # "  ML " -> "ml"
    return alias_map.get(cleaned)      # .get() gives None instead of crashing


print("4. normalize_tag('  ML ') ->", normalize_tag("  ML "))
print("   normalize_tag('cooking') ->", normalize_tag("cooking"))

# --- 5. Loops and if: do something for each item ----------------------------
raw_tags = ["py", "ML", "cooking", "python3"]
approved = []
for raw in raw_tags:
    tag = normalize_tag(raw)
    if tag is not None and tag not in approved:  # skip unknowns and repeats
        approved.append(tag)
print("5. Raw tags", raw_tags, "became approved tags", approved)

# --- 6. A tiny "keyword tagger": the simplest version of our whole project --
post_text = "In this tutorial we build a REST API in Python and train an ML model."
suggested = []
for alias, tag in alias_map.items():
    if alias in post_text.lower().split() and tag not in suggested:
        suggested.append(tag)
print("6. Keyword tagger suggests:", suggested)
print("   (It missed 'Python' because the text says 'python', not 'py'.")
print("    Fixing weaknesses like this is why we will build smarter models.)")

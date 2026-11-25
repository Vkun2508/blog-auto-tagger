"""Suggest tags from the command line.

    python -m tagger --title "My post title" --body "The post text..."
    python -m tagger --file samples/my_post.md --model tfidf
"""

import argparse
import sys
from pathlib import Path

from tagger.engine import TaggingEngine
from tagger.suggest import REVIEW_REASON_TEXT

BAR_WIDTH = 20


def read_post_file(path):
    """Title = the first Markdown heading, or the file name if there is none."""
    text = Path(path).read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip(), text
    return Path(path).stem, text


def main():
    # Windows consoles may not support every character in a post title; replace those instead of crashing.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Suggest approved tags for a blog post.")
    parser.add_argument("--title", default="")
    parser.add_argument("--body", default="")
    parser.add_argument("--file", help="a .md, .html or .txt file to tag instead of --title/--body")
    parser.add_argument("--model", help="keyword, tfidf, semantic or hybrid (default: the tuned best)")
    args = parser.parse_args()

    title, body = (read_post_file(args.file) if args.file else (args.title, args.body))
    engine = TaggingEngine.load()
    suggestion = engine.suggest(title, body, model=args.model)

    print(f"Model: {engine.model_label(suggestion.model)} ({suggestion.model})")
    print(f"Words: {suggestion.n_words}\n")
    print("Candidates (score, threshold):")
    for c in suggestion.candidates:
        bar = "#" * round(c.score * BAR_WIDTH)
        passed = "  <- suggested" if c in suggestion.tags else ""
        print(f"  {c.tag:<22} {bar:<{BAR_WIDTH}} {c.score:.2f} (>= {c.threshold:.2f}){passed}")
    print(f"\nSuggested tags: {', '.join(suggestion.tag_names()) or '(none)'}")
    if suggestion.needs_review:
        print(f"Needs review: yes ({suggestion.review_reason}) - {REVIEW_REASON_TEXT[suggestion.review_reason]}")
    else:
        print("Needs review: no")


if __name__ == "__main__":
    main()

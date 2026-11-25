"""Turn raw post text (HTML, Markdown or plain text) into clean readable text.

Why clean at all? Models should learn from what the post is about, not from
menus, share buttons, links or the author's own hashtag list (that last one
would leak the answer into the input).
"""

import re
import warnings

from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

# BeautifulSoup warns when a short text looks like a file name or URL; that is harmless here.
warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

# Page parts that are never article content.
JUNK_ELEMENTS = ["nav", "header", "footer", "aside", "script", "style", "form",
                 "iframe", "noscript", "svg", "button"]
# Elements that end a line of text, so words from two paragraphs do not glue together.
BLOCK_ELEMENTS = ["p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6",
                  "section", "article", "blockquote", "pre", "table", "ul", "ol"]

HTML_TAG = re.compile(r"</?(html|body|head|div|p|nav|header|footer|article|section|span|a|ul|ol|li"
                      r"|h[1-6]|br|img|table|tr|td|script|style|iframe|pre|code|figure)\b", re.I)
MARKDOWN_SIGN = re.compile(r"^\s{0,3}(#{1,6}\s|```|[-*+]\s|>\s|\d+\.\s)|\]\(|\*\*|__", re.M)

FRONT_MATTER_MAX_LINES = 60

CODE_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.S)
MAX_CODE_LINES = 30
LIQUID_TAG = re.compile(r"\{%.*?%\}", re.S)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
BARE_URL = re.compile(r"(https?://|www\.)\S+")
HEADING_MARK = re.compile(r"^\s{0,3}#{1,6}\s+", re.M)
LIST_OR_QUOTE_MARK = re.compile(r"^\s*([-*+>]|\d+\.)\s+", re.M)
EMPHASIS = re.compile(r"\*\*|__|~~|`")
SINGLE_EMPHASIS = re.compile(r"(?<!\w)[*_]([^*_\n]+)[*_](?!\w)")     # *word* or _word_

# Lines that are about the author or the blog, not the topic (matched case-insensitively).
BOILERPLATE = re.compile(
    # "subscribe" alone would also hit code such as observable.subscribe(), so only these forms count.
    r"\bfollow me\b|\bsubscribe (to|for) (my|our)\b|^\W*subscribe\W*$|\boriginally published (at|on)\b|\bthanks for reading\b"
    r"|\bbuy me a coffee\b|\bconnect with me\b|\bshare this\b|\blike and share\b"
    r"|\bif you enjoyed\b|\bnewsletter\b|\bpart \d+ of\b", re.I)
# Only short lines are dropped: a long paragraph that happens to say "newsletter" is real content.
BOILERPLATE_MAX_WORDS = 40
# A line that only names social sites (what is left of "[Twitter](...) | [GitHub](...)").
SOCIAL_ONLY = re.compile(
    r"^(\W*(twitter|x|linkedin|github|instagram|youtube|facebook|mastodon|bluesky|threads|dev\.to)\W*)+$", re.I)
# Tag widgets: "#python #ai #webdev" or "Tags: python, ai". Removing them stops label leakage.
HASHTAG_LINE = re.compile(r"^(#[\w+.\-]+[\s,]*)+$")
TAGS_LINE = re.compile(r"^tags?\s*:\s*[#\w\s,.+\-]*$", re.I)


def detect_format(text):
    """Guess whether text is "html", "markdown" or "plain"."""
    if not text:
        return "plain"
    # Any Markdown sign wins: Markdown posts often embed HTML, and markdown_to_text cleans that HTML too.
    if MARKDOWN_SIGN.search(text):
        return "markdown"
    if HTML_TAG.search(text):
        return "html"
    return "plain"


def strip_front_matter(text):
    """Drop a leading YAML block (--- title: ... tags: ... ---) that some posts start with.

    It holds the author's tags and metadata, not content. If no closing --- appears within
    the first 60 lines, the opening --- is probably a horizontal rule, so nothing is removed.
    """
    lines = text.lstrip().split("\n")
    if lines[0].rstrip() != "---":
        return text
    for i in range(1, min(len(lines), FRONT_MATTER_MAX_LINES)):
        if lines[i].rstrip() == "---":
            return "\n".join(lines[i + 1:])
    return text


def html_to_text(text):
    """Strip HTML with BeautifulSoup: drop page furniture, keep the readable text."""
    soup = BeautifulSoup(text, "html.parser")
    for element in soup.find_all(JUNK_ELEMENTS):
        element.decompose()
    for element in soup.find_all(BLOCK_ELEMENTS):
        element.append("\n")
    return soup.get_text()


def shorten_code_block(match):
    """Keep the fence's language word and the first 30 code lines (long code is mostly noise)."""
    words = match.group(1).split()
    language = words[0] if words else ""
    code_lines = match.group(2).split("\n")[:MAX_CODE_LINES]
    return "\n" + language + "\n" + "\n".join(code_lines) + "\n"


def markdown_to_text(text):
    """Remove Markdown syntax, DEV.to liquid tags, images and URLs; keep link words."""
    text = CODE_FENCE.sub(shorten_code_block, text)
    text = LIQUID_TAG.sub(" ", text)
    text = HTML_COMMENT.sub(" ", text)
    if HTML_TAG.search(text):              # Markdown posts may contain raw HTML blocks
        text = html_to_text(text)
    text = IMAGE.sub(" ", text)
    text = LINK.sub(r"\1", text)
    text = BARE_URL.sub(" ", text)
    text = HEADING_MARK.sub("", text)
    text = LIST_OR_QUOTE_MARK.sub("", text)
    text = EMPHASIS.sub("", text)
    text = SINGLE_EMPHASIS.sub(r"\1", text)
    return text


def is_unwanted_line(line):
    """True for boilerplate, social-link and tag-widget lines."""
    if HASHTAG_LINE.match(line) or TAGS_LINE.match(line) or SOCIAL_ONLY.match(line):
        return True
    return len(line.split()) <= BOILERPLATE_MAX_WORDS and bool(BOILERPLATE.search(line))


def clean_lines(text):
    """Collapse spaces, drop empty, symbol-only and unwanted lines; keep one line per paragraph."""
    kept = []
    for line in text.split("\n"):
        line = " ".join(line.split())
        # A line with no letters or digits (e.g. "---", "| --- |", "}") carries no meaning.
        if not re.search(r"[^\W_]", line) or is_unwanted_line(line):
            continue
        kept.append(line)
    return "\n".join(kept)


def clean_text(text):
    """Clean any post text. Never fails: empty or None input gives ""."""
    if not text:
        return ""
    text = str(text).replace("\r\n", "\n").replace("\r", "\n")
    text = strip_front_matter(text)
    text_format = detect_format(text)
    if text_format == "html":
        text = html_to_text(text)
    elif text_format == "markdown":
        text = markdown_to_text(text)
    return clean_lines(text)


def clean_post(title, body):
    """Clean a title and body. Titles only get whitespace tidied (they are short and plain)."""
    title_clean = " ".join(str(title or "").split())
    return title_clean, clean_text(body)


def word_count(text):
    """Number of words separated by whitespace."""
    return len(str(text or "").split())


def split_sentences(text):
    """Split text into sentences: every line, then at . ! or ? followed by a space."""
    sentences = []
    for line in str(text or "").split("\n"):
        for part in re.split(r"(?<=[.!?])\s+", line):
            if part.strip():
                sentences.append(part.strip())
    return sentences

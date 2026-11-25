from tagger.cleaning import clean_post, clean_text, detect_format, split_sentences, word_count


def test_html_page_furniture_removed_article_kept():
    html = """<html><body>
    <nav>Home | Blog | About</nav>
    <header>My Site Header</header>
    <article><h1>Docker tips</h1><p>Use multi-stage builds to shrink images.</p></article>
    <script>alert('tracking')</script>
    <footer>Copyright 2025</footer>
    </body></html>"""
    assert detect_format(html) == "html"
    text = clean_text(html)
    assert "multi-stage builds" in text
    assert "Docker tips" in text
    for junk in ["Home | Blog", "My Site Header", "alert", "Copyright"]:
        assert junk not in text


def test_markdown_liquid_images_urls_removed_link_text_kept():
    markdown = ("## Intro\n"
                "{% embed https://example.com/video %}\n"
                "![diagram](https://example.com/a.png)\n"
                "Read the [official docs](https://docs.python.org) for more.\n"
                "See https://example.com/page too.\n")
    text = clean_text(markdown)
    assert "Intro" in text and "##" not in text
    assert "embed" not in text and "{%" not in text
    assert "diagram" not in text and ".png" not in text
    assert "official docs" in text
    assert "http" not in text


def test_hashtag_widget_and_tags_lines_removed():
    text = clean_text("Real sentence about Python.\n#python #ai #webdev\nTags: python, ai, webdev\n")
    assert text == "Real sentence about Python."


def test_boilerplate_lines_removed():
    text = clean_text("Useful content here.\nThanks for reading! Follow me on Twitter.\n"
                      "*Originally published at myblog.com*\nBuy me a coffee\n")
    assert text == "Useful content here."


def test_code_that_says_subscribe_is_kept():
    assert clean_text("source.subscribe(handle_event)") == "source.subscribe(handle_event)"
    assert clean_text("## Next\nRead *my series* on AWS.") == "Next\nRead my series on AWS."


def test_code_fence_truncated_to_30_lines():
    code = "\n".join(f"line_{i} = {i}" for i in range(50))
    text = clean_text(f"Some intro text.\n```python\n{code}\n```\nAfter the code.")
    assert "python" in text
    assert "line_29 = 29" in text
    assert "line_30" not in text
    assert "After the code." in text


def test_front_matter_removed():
    post = ("\n---\ntitle: My post\npublished: true\ntags: python, ai\n---\n\n"
            "## Setup\nInstall the package first.")
    text = clean_text(post)
    assert "title:" not in text and "published:" not in text and "python, ai" not in text
    assert text == "Setup\nInstall the package first."


def test_horizontal_rule_mid_text_not_truncated():
    post = "## Part one\nFirst paragraph.\n\n---\n\nSecond paragraph.\n\n---\n\nThird paragraph."
    text = clean_text(post)
    assert "First paragraph." in text and "Second paragraph." in text and "Third paragraph." in text


def test_markdown_with_html_blocks_is_cleaned_as_markdown():
    post = ('<div class="intro"><img src="https://example.com/logo.png"></div>\n'
            "<div>Intro text in HTML.</div>\n"
            "![diagram](https://example.com/a.png)\n"
            "This is **bold** advice about [caching](https://example.com/cache).\n")
    assert detect_format(post) == "markdown"
    text = clean_text(post)
    assert "Intro text in HTML." in text
    assert "This is bold advice about caching." in text
    assert "diagram" not in text and "http" not in text and "*" not in text and "<" not in text


def test_empty_and_none_inputs():
    assert clean_text(None) == ""
    assert clean_text("") == ""
    assert clean_post(None, None) == ("", "")
    assert word_count(None) == 0


def test_plain_text_passes_through():
    assert clean_text("Just   a plain\n\n sentence.") == "Just a plain\nsentence."


def test_split_sentences():
    assert split_sentences("One. Two!\nThree?") == ["One.", "Two!", "Three?"]

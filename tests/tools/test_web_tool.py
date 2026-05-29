"""Tests for WebTool (offline; HTML-to-text extraction)."""

from __future__ import annotations

from darwin.tools.web import WebTool, _extract_text


def test_html_to_text_strips_tags_and_collapses_whitespace() -> None:
    body = b"""
    <html><head><title>Test page</title>
    <style>body{color:red}</style></head>
    <body>
      <h1>Heading</h1>
      <p>First   paragraph    with   extra   whitespace.</p>

      <p>Second paragraph.</p>
      <script>console.log('this should be dropped');</script>
    </body></html>
    """
    text, title = _extract_text(body)
    assert title == "Test page"
    assert "Heading" in text
    assert "First paragraph with extra whitespace." in text
    assert "console.log" not in text
    # script and style content must be excluded.
    assert "color:red" not in text


def test_html_entities_are_unescaped() -> None:
    body = b"<p>5 &gt; 3 &amp;&amp; 1 &lt; 2</p>"
    text, _ = _extract_text(body)
    assert "5 > 3 && 1 < 2" in text


def test_scheme_other_than_http_https_rejected() -> None:
    web = WebTool()
    result = web.execute({"url": "file:///etc/passwd"})
    assert not result.success
    assert "scheme" in result.error.lower()


def test_empty_url_rejected() -> None:
    web = WebTool()
    result = web.execute({"url": ""})
    assert not result.success
    assert "empty" in result.error.lower()

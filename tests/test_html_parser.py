\
from src.html_parser import extract_segments, rebuild_html


def test_extract_and_rebuild_roundtrip():
    html = """
    <html><body>
      <h1>Title</h1>
      <p>Hello <em>world</em>.</p>
      <p>Second paragraph</p>
    </body></html>
    """
    segs, meta = extract_segments(html)
    assert len(segs) == 3
    # Fake translation
    for s in segs:
        s["html_translated"] = s["html"].replace("Hello", "Bonjour")
    rebuilt = rebuild_html(html, segs)
    assert "Bonjour" in rebuilt

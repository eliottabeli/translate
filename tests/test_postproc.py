\
from src.postproc import compare_html_structure


def test_compare_html_structure_detects_attr_change():
    src = '<p>See <a href="https://example.com">link</a>.</p>'
    trans = '<p>Voir <a href="https://evil.com">lien</a>.</p>'
    ok, issues = compare_html_structure(src, trans)
    assert not ok
    assert any("href" in x for x in issues)

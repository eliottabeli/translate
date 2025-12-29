\
from src.translator import DummyTranslator, translate_segments


def test_translate_segments_dummy():
    segments = [
        {"id": "seg_0000", "tag": "p", "html": "Hello <em>world</em>.", "text": "Hello world."},
        {"id": "seg_0001", "tag": "p", "html": "Second.", "text": "Second."},
    ]
    tr = DummyTranslator()
    out = translate_segments(segments, translator=tr, glossary_map={}, max_chars_per_chunk=9999, max_segments_per_chunk=8)
    assert len(out) == 2
    assert "html_translated" in out[0]

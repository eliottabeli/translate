from src.translator import (
    DummyTranslator,
    TranslationMemory,
    mask_glossary_terms,
    translate_segments,
    unmask_glossary_terms,
)


def test_translate_segments_dummy():
    segments = [
        {"id": "seg_0000", "tag": "p", "html": "Hello <em>world</em>.", "text": "Hello world."},
        {"id": "seg_0001", "tag": "p", "html": "Second.", "text": "Second."},
    ]
    tr = DummyTranslator()
    out = translate_segments(segments, translator=tr, glossary_map={}, max_chars_per_chunk=9999, max_segments_per_chunk=8)
    assert len(out) == 2
    assert "html_translated" in out[0]


def test_translate_segments_preserves_numbers_and_structure():
    segments = [
        {"id": "seg_0000", "tag": "p", "html": "In 1492, Columbus sailed.", "text": "In 1492, Columbus sailed."}
    ]
    tr = DummyTranslator()
    out = translate_segments(
        segments,
        translator=tr,
        glossary_map={},
        max_chars_per_chunk=1000,
        max_segments_per_chunk=1,
        structure_guard=True,
        mask_numbers_dates=True,
    )
    assert out[0]["html_translated"].startswith("In 1492")
    assert "1492" in out[0]["text_translated"]


def test_mask_glossary_terms_tracks_source_and_restores():
    html = "The manor lord visited the manor."
    masked, placeholders = mask_glossary_terms(html, {"manor": "domaine"})
    assert masked != html
    placeholder_data = next(iter(placeholders.values()))
    assert placeholder_data["source"] == "manor"
    restored = unmask_glossary_terms(masked, placeholders)
    assert "domaine" in restored


def test_translation_memory_normalization_and_dedup():
    tm = TranslationMemory(max_segments=5, max_terms=5, storage_path=None)
    tm.add_segment("Feudal tenure.", "Tenure féodale")
    tm.add_segment("Feudal tenure.", "Tenure féodale")
    assert len(tm.segment_bank) == 1
    match = tm.fuzzy_lookup("feudal tenure", threshold=0.8)
    assert match is not None
    assert match.get("tgt") == "Tenure féodale"

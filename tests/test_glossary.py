import json

from src.glossary import TermCandidate, _parse_glossary_response, _fallback_proposals


def test_parse_glossary_response_aligns_out_of_order_items():
    batch = [TermCandidate(term="england", ngram=1, count=1), TermCandidate(term="church", ngram=1, count=1)]
    payload = json.dumps(
        [
            {"term": "church", "fr_preferred": "église"},
            {"term": "england", "fr_preferred": "Angleterre"},
        ]
    )

    parsed = _parse_glossary_response(payload, batch)

    assert [p["term"] for p in parsed] == ["england", "church"]
    assert parsed[0]["fr_preferred"] == "Angleterre"
    assert parsed[1]["fr_preferred"] == "église"


def test_parse_glossary_response_handles_ndjson_lines():
    batch = [TermCandidate(term="england", ngram=1, count=1), TermCandidate(term="church", ngram=1, count=1)]
    payload = "\n".join(
        [
            json.dumps({"term": "england", "fr_preferred": "Angleterre"}),
            json.dumps({"term": "church", "fr_preferred": "église"}),
        ]
    )

    parsed = _parse_glossary_response(payload, batch)

    assert [p["fr_preferred"] for p in parsed] == ["Angleterre", "église"]


def test_fallback_used_when_parsing_fails():
    batch = [TermCandidate(term="england", ngram=1, count=1)]
    payload = "not json"

    parsed = _parse_glossary_response(payload, batch)
    assert parsed == []

    fallback = _fallback_proposals(batch, payload)
    assert fallback[0]["register_note"].startswith("Réponse non JSON")

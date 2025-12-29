\
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup

# Fallback tiny stopword list if nltk stopwords aren't available.
_FALLBACK_EN_STOPWORDS = {
    "the","a","an","and","or","of","to","in","on","for","with","as","by","at","from",
    "is","are","was","were","be","been","being","this","that","these","those","it","its",
    "their","his","her","they","them","we","you","i","he","she","not","no","yes","but",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")


def _get_stopwords(lang: str = "english") -> set[str]:
    try:
        import nltk
        from nltk.corpus import stopwords
        try:
            return set(stopwords.words(lang))
        except LookupError:
            nltk.download("stopwords", quiet=True)
            return set(stopwords.words(lang))
    except Exception:
        return set(_FALLBACK_EN_STOPWORDS)


def extract_glossary_candidates(
    segments: List[Dict[str, Any]],
    top_n: int = 150,
    min_len: int = 4,
    include_from_tags: Optional[List[str]] = None,
    stopwords_lang: str = "english",
) -> List[Dict[str, Any]]:
    """
    Extract candidate specialized terms from segments.

    Heuristics:
      - text inside inline tags (em/i/strong) -> high priority
      - frequent non-stopword terms (simple unigram frequency)
    """
    include_from_tags = include_from_tags or ["em", "i", "strong"]
    stop = _get_stopwords(stopwords_lang)

    tagged_terms = Counter()
    freq_terms = Counter()

    for seg in segments:
        html = seg.get("html", "")
        text = seg.get("text", "")
        soup = BeautifulSoup(f"<div>{html}</div>", "html.parser")

        # Extract inline-tagged terms
        for tname in include_from_tags:
            for node in soup.find_all(tname):
                term = node.get_text(" ", strip=True)
                term = term.strip()
                if len(term) >= min_len:
                    tagged_terms[term] += 1

        # Frequency terms (unigrams)
        for w in _WORD_RE.findall(text):
            lw = w.lower()
            if lw in stop:
                continue
            if len(lw) < min_len:
                continue
            freq_terms[lw] += 1

    # Build ranked list: tagged terms first, then frequent terms
    candidates: List[Dict[str, Any]] = []

    for term, cnt in tagged_terms.most_common(top_n):
        candidates.append(
            {
                "term": term,
                "count": cnt,
                "source": "inline_tag",
                "translation": "",
                "notes": "",
            }
        )

    # Add frequent terms not already present (case-insensitive)
    existing_lower = {c["term"].lower() for c in candidates}
    for term, cnt in freq_terms.most_common(top_n):
        if term.lower() in existing_lower:
            continue
        candidates.append(
            {
                "term": term,
                "count": cnt,
                "source": "frequency",
                "translation": "",
                "notes": "",
            }
        )
        if len(candidates) >= top_n:
            break

    return candidates


def glossary_to_map(glossary_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    """
    Convert glossary rows to a mapping {source_term: forced_translation}.
    Ignores empty translations.
    """
    mapping: Dict[str, str] = {}
    for row in glossary_rows:
        src = str(row.get("term", "")).strip()
        tgt = str(row.get("translation", "")).strip()
        if src and tgt:
            mapping[src] = tgt
    return mapping

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from bs4 import BeautifulSoup
import importlib.util
import os
import json

# Fallback tiny stopword list if nltk stopwords aren't available.
_FALLBACK_EN_STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "of",
    "to",
    "in",
    "on",
    "for",
    "with",
    "as",
    "by",
    "at",
    "from",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "their",
    "his",
    "her",
    "they",
    "them",
    "we",
    "you",
    "i",
    "he",
    "she",
    "not",
    "no",
    "yes",
    "but",
}

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")
_ROMAN_RE = re.compile(r"\b[MDCLXVI]{1,4}\b", re.IGNORECASE)


@dataclass
class TermCandidate:
    term: str
    ngram: int
    count: int
    contexts: List[str] = field(default_factory=list)
    signals: Dict[str, int] = field(default_factory=dict)
    domain_score: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "term": self.term,
            "ngram": self.ngram,
            "count": self.count,
            "contexts": self.contexts,
            "signals": self.signals,
            "domain_score": round(self.domain_score, 4),
        }


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


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _iter_ngrams(tokens: List[str], n: int) -> Iterable[Tuple[str, ...]]:
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i : i + n])


def _capture_definition_heads(text: str) -> List[str]:
    heads: List[str] = []
    patterns = [
        r"^(?P<head>[A-Za-z][A-Za-z\s\-'’]{2,80})\s*[—\-:]\s+",
        r"^(?P<head>An?\s+[A-Za-z][A-Za-z\s\-'’]{1,80})\s+is\s",
        r"^(?P<head>[A-Za-z][A-Za-z\s\-'’]{1,80})\s+is\s",
        r"^(?P<head>[A-Za-z][A-Za-z\s\-'’]{1,80})\s+refers\s+to\s",
    ]
    for pat in patterns:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            head = m.group("head").strip()
            heads.append(head)
    return heads


def _is_temporal_entity(candidate: str) -> bool:
    if _ROMAN_RE.search(candidate) and "c" in candidate.lower():
        return True
    if re.search(r"\b(?:1[0-9]|[5-9])(?:th|st|nd|rd)?\s*century\b", candidate, flags=re.IGNORECASE):
        return True
    if re.search(r"\b(?:1[0-9]|[5-9])\s*\-\s*(?:1[0-9]|[5-9])\s*c\b", candidate, flags=re.IGNORECASE):
        return True
    return False


def extract_term_candidates(
    segments: List[Dict[str, Any]],
    top_n: int = 400,
    min_len: int = 3,
    stopwords_lang: str = "english",
    max_contexts: int = 5,
) -> List[TermCandidate]:
    """
    Multi-signal candidate extraction for glossary building.

    Signals used:
      - n-gram frequency (1-4 grams)
      - emphasis/typography (em, strong, italics)
      - titles & list bullets
      - definition patterns (X — … / X: … / X is …)
      - temporal/roman numerals for century-style entities
      - co-occurrence graph to boost clusters
    """

    stop = _get_stopwords(stopwords_lang)

    counts: Counter[str] = Counter()
    signals: Dict[str, Counter[str]] = defaultdict(Counter)
    contexts: Dict[str, List[str]] = defaultdict(list)
    cooc: Counter[Tuple[str, str]] = Counter()

    for seg in segments:
        html = seg.get("html", "")
        text = seg.get("text", "")
        soup = BeautifulSoup(f"<div>{html}</div>", "html.parser")
        tokens = [t for t in _tokenize(text) if len(t) >= min_len and t not in stop]

        # n-grams 1..4
        for n in range(1, 5):
            for gram in _iter_ngrams(tokens, n):
                term = " ".join(gram)
                counts[term] += 1
                if len(contexts[term]) < max_contexts:
                    contexts[term].append(text[:240])

        # typography signals
        for tname in ("em", "i", "strong"):
            for node in soup.find_all(tname):
                term_text = node.get_text(" ", strip=True)
                if len(term_text) >= min_len:
                    signals[term_text][f"in_{tname}"] += 1
                    counts[term_text] += 1
                    if len(contexts[term_text]) < max_contexts:
                        contexts[term_text].append(text[:240])

        if seg.get("tag") in {"h1", "h2", "h3", "title"}:
            for head in _tokenize(text):
                if head not in stop and len(head) >= min_len:
                    signals[head]["in_title"] += 1
                    counts[head] += 1

        if seg.get("tag") == "li":
            for candidate in _tokenize(text):
                if candidate not in stop and len(candidate) >= min_len:
                    signals[candidate]["in_list"] += 1

        # definition patterns
        for head in _capture_definition_heads(text):
            signals[head]["in_definition_pattern"] += 1
            counts[head] += 1
            if len(contexts[head]) < max_contexts:
                contexts[head].append(text[:240])

        # temporal markers
        for token in tokens:
            if _is_temporal_entity(token):
                signals[token]["temporal"] += 1

        # co-occurrence graph (within a block)
        uniq = sorted(set(tokens))
        for i, t1 in enumerate(uniq):
            for t2 in uniq[i + 1 :]:
                cooc[(t1, t2)] += 1

    # derive relatedness boost
    related_boost: Dict[str, float] = defaultdict(float)
    for (t1, t2), cnt in cooc.items():
        related_boost[t1] += math.log1p(cnt)
        related_boost[t2] += math.log1p(cnt)

    candidates: List[TermCandidate] = []
    for term, cnt in counts.most_common():
        if len(term) < min_len:
            continue
        ngram = len(term.split())
        sigs = dict(signals.get(term, {}))
        sigs.setdefault("frequency", cnt)
        domain_score = _compute_domain_score(cnt, sigs, related_boost.get(term, 0.0))
        candidates.append(
            TermCandidate(
                term=term,
                ngram=ngram,
                count=cnt,
                contexts=contexts.get(term, [])[:max_contexts],
                signals=sigs,
                domain_score=domain_score,
            )
        )

    candidates = sorted(candidates, key=lambda c: c.domain_score, reverse=True)
    return candidates[:top_n]


def _compute_domain_score(freq: int, sigs: Dict[str, int], relatedness: float) -> float:
    score = 0.0
    score += math.log1p(freq) * 0.35
    score += math.log1p(sigs.get("in_definition_pattern", 0)) * 0.25
    score += math.log1p(sigs.get("in_strong", 0) + sigs.get("in_em", 0) + sigs.get("in_i", 0)) * 0.15
    score += math.log1p(sigs.get("in_title", 0)) * 0.1
    score += math.log1p(relatedness) * 0.15
    return min(1.0, score)


def glossary_to_map(glossary_rows: List[Dict[str, Any]]) -> Dict[str, str]:
    """Convert glossary rows to a mapping {source_term: forced_translation}.

    Empty translations are ignored.
    """

    mapping: Dict[str, str] = {}
    for row in glossary_rows:
        src = str(row.get("term", "")).strip()
        tgt = str(row.get("translation", "")).strip()
        if src and tgt:
            mapping[src] = tgt
    return mapping


def compute_translation_instability(translated_segments: List[Dict[str, Any]], terms: List[str]) -> Dict[str, int]:
    """Measure how many distinct translations appear for each source term.

    If no translation is present yet, returns an empty dict.
    """

    variants: Dict[str, set[str]] = defaultdict(set)
    for seg in translated_segments:
        src = seg.get("text", "")
        tgt = seg.get("text_translated", "")
        if not tgt:
            continue
        for term in terms:
            if term.lower() in src.lower():
                variants[term].add(tgt)
    return {term: len(vset) for term, vset in variants.items()}


def draft_glossary_rows(candidates: List[TermCandidate], top_n: int = 200) -> List[Dict[str, Any]]:
    rows = []
    for cand in candidates[:top_n]:
        rows.append(
            {
                "term": cand.term,
                "count": cand.count,
                "signals": cand.signals,
                "domain_score": round(cand.domain_score, 4),
                "translation": "",
                "notes": "",
            }
        )
    return rows


def generate_translation_proposals(
    candidates: List[TermCandidate],
    style_guide: Optional[Dict[str, Any]] = None,
    top_n: int = 300,
    k: int = 3,
    model: str = "gpt-4.1-mini",
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Generate glossary translation proposals with justifications.

    Falls back to heuristic placeholders when OpenAI is not configured.
    """

    if not _has_openai() or not (api_key or os.getenv("OPENAI_API_KEY")):
        return [
            {
                "term": cand.term,
                "fr_preferred": cand.term,
                "fr_alternatives": [],
                "pos": "", 
                "definition_fr_short": "", 
                "register_note": "Fournir une traduction manuelle (clé API manquante).",
                "do_not_translate": False,
                "justification": "Placeholder faute de configuration API.",
            }
            for cand in candidates[:top_n]
        ]

    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    style_info = json.dumps(style_guide or {}, ensure_ascii=False)
    proposals: List[Dict[str, Any]] = []

    for batch_start in range(0, min(top_n, len(candidates)), 20):
        batch = candidates[batch_start : batch_start + 20]
        prompt = {
            "role": "user",
            "content": (
                "Vous êtes un traducteur universitaire. Pour chaque terme, proposez une traduction préférée, "
                "des alternatives utiles, la catégorie grammaticale (pos), une définition FR courte, une note de registre, "
                "et si le terme doit être conservé tel quel (do_not_translate). Répondez en JSON ligne par ligne.\n"
                f"Style guide: {style_info}\n"
                f"Nombre de propositions demandées par terme: {k}.\n"
                "Termes: " + "; ".join(c.term for c in batch)
            ),
        }
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": "Répondez uniquement en JSON valide, sans texte hors JSON."}, prompt],
            temperature=0.2,
            max_tokens=800,
        )
        content = resp.choices[0].message.content or ""
        try:
            data = json.loads(content)
            if isinstance(data, list):
                proposals.extend(data)
            elif isinstance(data, dict):
                proposals.append(data)
        except json.JSONDecodeError:
            for cand in batch:
                proposals.append(
                    {
                        "term": cand.term,
                        "fr_preferred": cand.term,
                        "fr_alternatives": [],
                        "pos": "",
                        "definition_fr_short": "",
                        "register_note": "Réponse non JSON : relire manuellement.",
                        "do_not_translate": False,
                        "justification": content[:180],
                    }
                )
    return proposals


def _has_openai() -> bool:
    return importlib.util.find_spec("openai") is not None

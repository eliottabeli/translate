from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, Tag

from .utils import compact_whitespace


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_CITATION_RE = re.compile(r"\[[0-9]+\]|\([a-zA-Z]\)")


def _pluralize(term: str) -> list[str]:
    if term.endswith("y") and len(term) > 2:
        return [term[:-1] + "ies"]
    if term.endswith(("s", "x", "sh", "ch")):
        return [term + "es"]
    return [term + "s"]


def _glossary_variants(term: str) -> list[str]:
    base = [term, term.lower(), term.capitalize()]
    variants = list(base)
    for cand in base:
        if " " not in cand and "-" not in cand:
            variants.extend(_pluralize(cand))
    if "-" in term:
        variants.append(term.replace("-", " "))
    seen = set()
    ordered: list[str] = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            ordered.append(v)
    return ordered


def apply_glossary_to_fragment(html_fragment: str, glossary_map: Dict[str, str]) -> str:
    """
    Apply a forced glossary mapping to a HTML fragment by replacing terms in *text nodes only*.
    Does NOT touch attributes.
    """
    if not glossary_map:
        return html_fragment

    # Replace longer keys first to avoid partial overlaps and handle multi-word phrases
    items: list[tuple[str, str]] = []
    for src, tgt in glossary_map.items():
        for variant in _glossary_variants(src):
            items.append((variant, tgt))
    items = sorted(items, key=lambda kv: len(kv[0]), reverse=True)

    soup = BeautifulSoup(f"<div>{html_fragment}</div>", "html.parser")
    for node in soup.find_all(string=True):
        if isinstance(node, Tag):
            continue
        parent = getattr(node, "parent", None)
        if parent and parent.name in ("script", "style"):
            continue
        txt = str(node)
        new_txt = txt
        for src, tgt in items:
            if not src:
                continue
            new_txt = _replace_term(new_txt, src, tgt)
            new_txt = _replace_term(new_txt, src.capitalize(), tgt.capitalize())
            new_txt = _replace_term(new_txt, src.upper(), tgt.upper())
        if new_txt != txt:
            node.replace_with(new_txt)

    return "".join(str(x) for x in soup.div.contents)


def _replace_term(text: str, src: str, tgt: str) -> str:
    # If multi-word, do a simple replace (case-sensitive)
    if " " in src or "-" in src:
        return text.replace(src, tgt)

    # Word boundary replace for single tokens
    pattern = re.compile(rf"\b{re.escape(src)}\b")
    return pattern.sub(tgt, text)


def compare_html_structure(
    original_html: str,
    translated_html: str,
    ignore_attrs: Optional[set[str]] = None,
) -> Tuple[bool, List[str]]:
    """
    Compare tag structure + attribute values.
    Returns: (ok, issues)
    """
    ignore_attrs = ignore_attrs or {"alt"}  # alt can be translated optionally
    issues: List[str] = []

    o = BeautifulSoup(original_html, "html.parser")
    t = BeautifulSoup(translated_html, "html.parser")

    o_tags = o.find_all(True)
    t_tags = t.find_all(True)

    if len(o_tags) != len(t_tags):
        issues.append(f"Different number of tags: src={len(o_tags)} vs trans={len(t_tags)}")
        return False, issues

    for i, (ot, tt) in enumerate(zip(o_tags, t_tags)):
        if ot.name != tt.name:
            issues.append(f"Tag mismatch at index {i}: src=<{ot.name}> vs trans=<{tt.name}>")
            continue

        # Compare attributes
        o_attrs = dict(ot.attrs)
        t_attrs = dict(tt.attrs)

        # Normalize class lists
        if "class" in o_attrs:
            o_attrs["class"] = sorted(list(o_attrs["class"]))
        if "class" in t_attrs:
            t_attrs["class"] = sorted(list(t_attrs["class"]))

        # Remove ignored attrs
        for a in list(o_attrs.keys()):
            if a in ignore_attrs:
                o_attrs.pop(a, None)
        for a in list(t_attrs.keys()):
            if a in ignore_attrs:
                t_attrs.pop(a, None)

        if set(o_attrs.keys()) != set(t_attrs.keys()):
            issues.append(
                f"Attribute keys differ at tag {i} <{ot.name}>: "
                f"src={sorted(o_attrs.keys())} vs trans={sorted(t_attrs.keys())}"
            )
            continue

        for k in o_attrs.keys():
            if o_attrs[k] != t_attrs[k]:
                issues.append(
                    f"Attribute value changed at tag {i} <{ot.name}> attr='{k}': src={o_attrs[k]} vs trans={t_attrs[k]}"
                )

    ok = len(issues) == 0
    return ok, issues


def numbers_and_citations_check(source_text: str, translated_text: str) -> List[str]:
    """
    Simple consistency check:
      - numbers should be preserved as raw digit tokens
      - citations like [1], (a) should be preserved
    Returns list of warnings.
    """
    warnings: List[str] = []
    src_nums = Counter(_NUMBER_RE.findall(source_text))
    tr_nums = Counter(_NUMBER_RE.findall(translated_text))
    if src_nums != tr_nums:
        warnings.append(f"Numbers changed: src={dict(src_nums)} vs trans={dict(tr_nums)}")

    src_cit = Counter(_CITATION_RE.findall(source_text))
    tr_cit = Counter(_CITATION_RE.findall(translated_text))
    if src_cit != tr_cit:
        warnings.append(f"Citations changed: src={dict(src_cit)} vs trans={dict(tr_cit)}")

    return warnings


def terminology_report(translated_segments: List[Dict[str, Any]], glossary_map: Dict[str, str]) -> List[Dict[str, Any]]:
    """Return rows highlighting glossary deviations across translated text."""

    rows: List[Dict[str, Any]] = []
    for term_en, term_fr in glossary_map.items():
        mismatches: List[Dict[str, Any]] = []
        for seg in translated_segments:
            src = seg.get("text", "")
            tgt = seg.get("text_translated", "")
            if not tgt:
                continue
            if term_en.lower() in src.lower() and term_fr not in tgt:
                mismatches.append({"segment_id": seg.get("id"), "translation_found": tgt})
        if mismatches:
            rows.append(
                {
                    "term_en": term_en,
                    "expected_fr": term_fr,
                    "occurrences": len(mismatches),
                    "segments": mismatches,
                }
            )
    return rows


def auto_correct_terminology(
    translated_segments: List[Dict[str, Any]],
    glossary_map: Dict[str, str],
    report_rows: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Apply glossary substitutions to segments flagged by the terminology report."""

    if not glossary_map:
        return []

    index = {seg.get("id"): seg for seg in translated_segments}
    corrections: List[Dict[str, Any]] = []
    report_rows = report_rows or terminology_report(translated_segments, glossary_map)

    for row in report_rows:
        term_en = row.get("term_en")
        expected = row.get("expected_fr", "")
        for occurrence in row.get("segments", []):
            seg_id = occurrence.get("segment_id")
            seg = index.get(seg_id)
            if not seg:
                continue
            before_html = seg.get("html_translated", "")
            patched_html = apply_glossary_to_fragment(before_html, glossary_map)
            if patched_html != before_html:
                seg["html_translated"] = patched_html
                seg["text_translated"] = BeautifulSoup(f"<div>{patched_html}</div>", "html.parser").get_text(
                    " ",
                    strip=True,
                )
                corrections.append(
                    {
                        "kind": "terminology_correction",
                        "segment_id": seg_id,
                        "term_en": term_en,
                        "expected_fr": expected,
                        "before": before_html,
                        "after": patched_html,
                    }
                )
    return corrections


def evaluate_register(
    translated_segments: List[Dict[str, Any]],
    anglicisms: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Heuristic register evaluation: anglicism density + ASCII-only token ratio."""

    anglicisms = anglicisms or [
        "actually",
        "eventually",
        "overall",
        "throughout",
        "regarding",
        "regardless",
    ]

    total_tokens = 0
    ascii_tokens = 0
    anglicism_hits = 0
    sample_hits: List[Dict[str, Any]] = []

    for seg in translated_segments:
        txt = seg.get("text_translated", "")
        tokens = re.findall(r"[A-Za-zÀ-ÿ'-]+", txt)
        total_tokens += len(tokens)
        ascii_tokens += len([t for t in tokens if re.fullmatch(r"[A-Za-z'-]+", t)])
        for ang in anglicisms:
            if ang.lower() in txt.lower():
                anglicism_hits += 1
                sample_hits.append({"segment_id": seg.get("id"), "term": ang, "text": txt[:160]})

    ascii_ratio = (ascii_tokens / total_tokens) if total_tokens else 0.0
    anglicism_ratio = (anglicism_hits / max(len(translated_segments), 1))
    score = max(0.0, 1.0 - (ascii_ratio * 0.4 + anglicism_ratio * 0.6))

    return {
        "total_segments": len(translated_segments),
        "total_tokens": total_tokens,
        "ascii_ratio": round(ascii_ratio, 4),
        "anglicism_ratio": round(anglicism_ratio, 4),
        "register_score": round(score, 4),
        "samples": sample_hits,
    }


def learn_collocations(translated_segments: List[Dict[str, Any]], top_n: int = 25, min_count: int = 2) -> List[str]:
    """Extract frequent French bigrams/trigrams to feed the revision prompt."""

    grams = Counter()
    for seg in translated_segments:
        txt = seg.get("text_translated", "")
        tokens = [t.lower() for t in re.findall(r"[a-zàâçéèêëîïôûùüÿñæœ'-]+", txt)]
        for n in (2, 3):
            for i in range(len(tokens) - n + 1):
                grams[" ".join(tokens[i : i + n])] += 1
    common = [g for g, c in grams.most_common() if c >= min_count]
    return common[:top_n]

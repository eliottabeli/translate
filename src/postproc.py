\
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from bs4 import BeautifulSoup, Tag

from .utils import compact_whitespace


_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)?")
_CITATION_RE = re.compile(r"\[[0-9]+\]|\([a-zA-Z]\)")


def apply_glossary_to_fragment(html_fragment: str, glossary_map: Dict[str, str]) -> str:
    """
    Apply a forced glossary mapping to a HTML fragment by replacing terms in *text nodes only*.
    Does NOT touch attributes.
    """
    if not glossary_map:
        return html_fragment

    # Replace longer keys first to avoid partial overlaps
    items = sorted(glossary_map.items(), key=lambda kv: len(kv[0]), reverse=True)

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

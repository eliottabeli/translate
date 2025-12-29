\
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

from bs4 import BeautifulSoup, Tag, NavigableString

from .utils import sha1_text, compact_whitespace


DEFAULT_BLOCK_TAGS = ["p", "h1", "h2", "h3", "li", "figcaption", "blockquote"]


def _inner_html(tag: Tag) -> str:
    """Return inner HTML of a tag (children only), preserving inline tags."""
    return "".join(str(x) for x in tag.contents)


def extract_segments(
    html_text: str,
    block_tags: Optional[List[str]] = None,
    include_empty: bool = False,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Parse HTML and extract "block" segments.

    Returns:
      segments: [{id, tag, html, text}]
      meta: {source_sha1, n_segments, block_tags}
    """
    block_tags = block_tags or DEFAULT_BLOCK_TAGS
    soup = BeautifulSoup(html_text, "html.parser")

    segments: List[Dict[str, Any]] = []
    seg_id = 0
    section_stack: List[Dict[str, Any]] = []
    section_counter = 0
    list_ids: Dict[int, str] = {}
    list_counter = 0

    for tag in soup.find_all(block_tags):
        inner = _inner_html(tag)
        plain = compact_whitespace(tag.get_text(" ", strip=True))

        if not include_empty and plain == "" and inner.strip() == "":
            continue

        # Track heading hierarchy for contextual chunking
        if tag.name and tag.name.startswith("h") and len(tag.name) == 2 and tag.name[1].isdigit():
            level = int(tag.name[1])
            while section_stack and section_stack[-1]["level"] >= level:
                section_stack.pop()
            section_counter += 1
            section_stack.append(
                {
                    "id": f"sec_{section_counter:04d}",
                    "title": plain,
                    "level": level,
                }
            )

        section_id = section_stack[-1]["id"] if section_stack else "sec_root"
        section_titles = [s["title"] for s in section_stack]
        section_levels = [s["level"] for s in section_stack]

        list_parent = tag.find_parent(["ul", "ol"])
        list_id = None
        if list_parent:
            key = id(list_parent)
            if key not in list_ids:
                list_counter += 1
                list_ids[key] = f"list_{list_counter:04d}"
            list_id = list_ids[key]

        context_group = section_id
        if list_id:
            context_group = f"{section_id}:{list_id}"

        segments.append(
            {
                "id": f"seg_{seg_id:04d}",
                "tag": tag.name,
                "html": inner,
                "text": plain,
                "section_id": section_id,
                "section_titles": section_titles,
                "section_levels": section_levels,
                "context_group": context_group,
                "list_id": list_id,
            }
        )
        seg_id += 1

    meta = {
        "source_sha1": sha1_text(html_text),
        "n_segments": len(segments),
        "block_tags": block_tags,
        "sections": section_counter,
    }
    return segments, meta


def rebuild_html(
    original_html: str,
    translated_segments: List[Dict[str, Any]],
    block_tags: Optional[List[str]] = None,
) -> str:
    """
    Rebuild a translated HTML document by replacing each extracted block's inner HTML
    with the translated version, in the same order.

    Important: This assumes `translated_segments` corresponds 1:1 to blocks found in `original_html`
    under the same `block_tags` extraction logic (same order, same count).
    """
    block_tags = block_tags or DEFAULT_BLOCK_TAGS
    soup = BeautifulSoup(original_html, "html.parser")

    blocks = soup.find_all(block_tags)
    if len(blocks) != len(translated_segments):
        raise ValueError(
            f"Segment count mismatch: original blocks={len(blocks)} vs translated_segments={len(translated_segments)}. "
            "Ensure you're rebuilding from the same source HTML and same segmentation settings."
        )

    for idx, (tag, seg) in enumerate(zip(blocks, translated_segments)):
        new_inner_html = seg.get("html_translated") or seg.get("html") or ""
        # Replace contents
        tag.clear()
        frag = BeautifulSoup(new_inner_html, "html.parser")

        # Append fragment contents
        for node in list(frag.contents):
            tag.append(node)

    return str(soup)

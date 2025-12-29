from __future__ import annotations

import difflib
import os
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

from .postproc import compare_html_structure, numbers_and_citations_check
from .utils import compact_whitespace


def difflib_ratio(a: str, b: str) -> float:
    return difflib.SequenceMatcher(a=a, b=b).ratio()


def run_basic_checks_on_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add QA fields per segment:
      - qa_ok_structure
      - qa_structure_issues
      - qa_num_cit_warnings
    """
    out = []
    for seg in segments:
        src_wrapped = f'<{seg["tag"]}>{seg.get("html","")}</{seg["tag"]}>'
        tr_html = seg.get("html_translated", "")
        tr_wrapped = f'<{seg["tag"]}>{tr_html}</{seg["tag"]}>'

        ok, issues = compare_html_structure(src_wrapped, tr_wrapped)
        warnings = numbers_and_citations_check(seg.get("text",""), seg.get("text_translated",""))

        seg2 = dict(seg)
        seg2["qa_ok_structure"] = ok
        seg2["qa_structure_issues"] = issues
        seg2["qa_num_cit_warnings"] = warnings
        out.append(seg2)
    return out


# --- Optional back-translation with OpenAI (requires keys) ---

_SYSTEM_PROMPT_BACKTRANSLATE = """\
You are a professional translation engine. You WILL translate text from French into English while PRESERVING all HTML tags and attributes exactly as they are.
You MUST NOT modify attribute values (href, src, id, class, data-*, style).
Do not add or remove tags. Return only the translated HTML fragment (no explanations)."""

_USER_PROMPT_BACKTRANSLATE = """\
Translate the following HTML fragment from French to English.
Preserve tags and attributes exactly; translate only the human-readable text nodes.
Return only HTML.

HTML:
{html}
"""


def back_translate_html_openai(html_fragment: str, api_key: str, model: str = "gpt-4.1-mini") -> str:
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError("openai package missing. pip install openai>=1.0.0") from e

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT_BACKTRANSLATE},
            {"role": "user", "content": _USER_PROMPT_BACKTRANSLATE.format(html=html_fragment)},
        ],
        temperature=0.0,
        max_tokens=2000,
    )
    return (resp.choices[0].message.content or "").strip()


def run_backtranslation_qa_openai(
    segments: List[Dict[str, Any]],
    api_key: Optional[str] = None,
    model: str = "gpt-4.1-mini",
    min_ratio: float = 0.65,
) -> List[Dict[str, Any]]:
    """
    Back-translate translated HTML -> English and compare to source English text using difflib ratio.
    Adds:
      - qa_backtranslation_ratio
      - qa_backtranslation_flag
    """
    api_key = api_key or os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY for back-translation QA.")

    out = []
    for seg in segments:
        tr_html = seg.get("html_translated", "")
        wrapped = f'<{seg["tag"]}>{tr_html}</{seg["tag"]}>'
        bt_html = back_translate_html_openai(wrapped, api_key=api_key, model=model)

        # Compare plain text
        src_text = compact_whitespace(seg.get("text", ""))
        bt_text = compact_whitespace(BeautifulSoup(bt_html, "html.parser").get_text(" ", strip=True))

        ratio = difflib_ratio(src_text, bt_text)
        seg2 = dict(seg)
        seg2["qa_backtranslation_ratio"] = ratio
        seg2["qa_backtranslation_flag"] = ratio < min_ratio
        out.append(seg2)

    return out

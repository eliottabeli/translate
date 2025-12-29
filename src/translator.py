\
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol

from bs4 import BeautifulSoup

from .utils import chunk_segments, compact_whitespace


SYSTEM_PROMPT_HTML_TRANSLATION = """\
You are a professional translation engine. You WILL translate text from English into French while PRESERVING all HTML tags and attributes exactly as they are.
You MUST NOT modify attribute values (href, src, id, class, data-*, style) except to translate only textual alt attribute content if requested.
Keep inline tags (<em>, <strong>, <sup>, <sub>, <a>, <span>, <img>, <figcaption>, etc.) and their positions unchanged.
Do not add or remove tags. Return only the translated HTML fragment (no explanations)."""

USER_PROMPT_TEMPLATE = """\
Translate the following HTML fragment from English to French.
Preserve tags and attributes exactly; translate only the human-readable text nodes.
If a text node contains references like [1], (a), or numbers, keep them unchanged.
Use French typographic rules (espace avant : ; guillemets « »).
If the text contains specialized terminology, prefer consistent translations (apply the supplied glossary).
Glossary (JSON): {glossary_json}

HTML:
{html}
"""


class BaseTranslator(Protocol):
    def translate_html(self, html_fragment: str, glossary: Optional[Dict[str, str]] = None) -> str:
        ...


@dataclass
class OpenAIConfig:
    model: str = "gpt-4.1-mini"
    temperature: float = 0.1
    max_output_tokens: int = 2000


class OpenAITranslator:
    """
    OpenAI translator that uses a strict prompt to preserve HTML.

    Requires:
      - `openai` python package
      - OPENAI_API_KEY in env or provided.
    """

    def __init__(self, api_key: Optional[str] = None, cfg: Optional[OpenAIConfig] = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("Missing OPENAI_API_KEY. Put it in .env or environment variables.")
        self.cfg = cfg or OpenAIConfig()

        try:
            from openai import OpenAI  # type: ignore
        except Exception as e:
            raise RuntimeError("openai package not installed or incompatible. pip install openai>=1.0.0") from e

        self._client = OpenAI(api_key=self.api_key)

    def translate_html(self, html_fragment: str, glossary: Optional[Dict[str, str]] = None) -> str:
        glossary_json = json.dumps(glossary or {}, ensure_ascii=False)

        user_prompt = USER_PROMPT_TEMPLATE.format(glossary_json=glossary_json, html=html_fragment)

        # Prefer chat.completions for compatibility
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_HTML_TRANSLATION},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_output_tokens,
        )

        content = resp.choices[0].message.content or ""
        return _strip_code_fences(content).strip()


class DeepLTranslator:
    """
    DeepL translator with tag_handling='html'.

    Requires:
      - `deepl` python package
      - DEEPL_AUTH_KEY in env or provided.
    """

    def __init__(self, auth_key: Optional[str] = None, formality: str = "more", preserve_formatting: bool = True):
        self.auth_key = auth_key or os.getenv("DEEPL_AUTH_KEY", "")
        if not self.auth_key:
            raise RuntimeError("Missing DEEPL_AUTH_KEY. Put it in .env or environment variables.")
        self.formality = formality
        self.preserve_formatting = preserve_formatting

        try:
            import deepl  # type: ignore
        except Exception as e:
            raise RuntimeError("deepl package not installed. pip install deepl>=1.16.0") from e
        self._deepl = deepl.Translator(self.auth_key)

    def translate_html(self, html_fragment: str, glossary: Optional[Dict[str, str]] = None) -> str:
        # DeepL has glossary feature but it's more complex (requires glossary creation in account).
        # Here we just translate; glossary can be applied in post-processing.
        result = self._deepl.translate_text(
            html_fragment,
            source_lang="EN",
            target_lang="FR",
            tag_handling="html",
            preserve_formatting=self.preserve_formatting,
            formality=self.formality,
        )
        return str(result)


class DummyTranslator:
    """Offline translator for testing/dev. Does not translate; just marks content."""
    def translate_html(self, html_fragment: str, glossary: Optional[Dict[str, str]] = None) -> str:
        # Very naive: pretend it's translated by adding a prefix to plain text nodes (not tags)
        soup = BeautifulSoup(f"<div>{html_fragment}</div>", "html.parser")
        for node in soup.find_all(string=True):
            if node.parent.name in ("script", "style"):
                continue
            txt = str(node)
            if txt.strip():
                node.replace_with(txt.replace(txt, f"{txt}"))
        return "".join(str(x) for x in soup.div.contents)


def _strip_code_fences(s: str) -> str:
    # Remove ```html ... ``` or ``` ... ```
    fence = re.compile(r"^\s*```(?:html)?\s*([\s\S]*?)\s*```\s*$", re.IGNORECASE)
    m = fence.match(s.strip())
    return m.group(1) if m else s


def translate_segments(
    segments: List[Dict[str, Any]],
    translator: BaseTranslator,
    glossary_map: Optional[Dict[str, str]] = None,
    max_chars_per_chunk: int = 9000,
    max_segments_per_chunk: int = 8,
) -> List[Dict[str, Any]]:
    """
    Translate segments, in chunks, while preserving per-segment boundaries.

    We build a chunk HTML with explicit block tags and a `data-seg-id` attribute,
    translate it, then split back into per-segment inner HTML.
    """
    out: List[Dict[str, Any]] = [dict(seg) for seg in segments]

    id_to_index = {seg["id"]: i for i, seg in enumerate(out)}

    for chunk in chunk_segments(out, max_chars_per_chunk=max_chars_per_chunk, max_segments_per_chunk=max_segments_per_chunk):
        chunk_html = "<div>" + "".join(
            f'<{seg["tag"]} data-seg-id="{seg["id"]}">{seg["html"]}</{seg["tag"]}>'
            for seg in chunk
        ) + "</div>"

        translated_chunk = translator.translate_html(chunk_html, glossary=glossary_map or {})
        translated_chunk = translated_chunk.strip()

        # Parse result
        soup = BeautifulSoup(translated_chunk, "html.parser")
        container = soup.find("div") or soup

        extracted = container.find_all(attrs={"data-seg-id": True})
        if extracted and len(extracted) == len(chunk):
            for el in extracted:
                seg_id = el.get("data-seg-id", "")
                idx = id_to_index.get(seg_id)
                if idx is None:
                    continue
                inner = "".join(str(x) for x in el.contents)
                out[idx]["html_translated"] = inner
                out[idx]["text_translated"] = compact_whitespace(el.get_text(" ", strip=True))
        else:
            # Fallback: try to match by tag order (less robust if model changes structure)
            els = []
            for seg in chunk:
                els.append(container.find(seg["tag"]))
                if els[-1] is not None:
                    # remove so next find gets next one
                    els[-1].extract()
            if any(e is None for e in els):
                raise ValueError("Unable to split translated chunk back into segments. The model likely altered structure.")
            for seg, el in zip(chunk, els):
                idx = id_to_index[seg["id"]]
                inner = "".join(str(x) for x in el.contents)
                out[idx]["html_translated"] = inner
                out[idx]["text_translated"] = compact_whitespace(el.get_text(" ", strip=True))

    return out

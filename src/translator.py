from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Pattern, Protocol, Tuple

from bs4 import BeautifulSoup, Comment, NavigableString

from .postproc import compare_html_structure
from .utils import chunk_segments_contextual, compact_whitespace, sha1_text


SYSTEM_PROMPT_HTML_TRANSLATION = """\
You are a professional translation engine. You WILL translate text from English into French while PRESERVING all HTML tags and attributes exactly as they are.
You MUST NOT modify attribute values (href, src, id, class, data-*, style) except to translate only textual alt attribute content if requested.
Keep inline tags (<em>, <strong>, <sup>, <sub>, <a>, <span>, <img>, <figcaption>, etc.) and their positions unchanged.
Do not add or remove tags. Return only the translated HTML fragment (no explanations)."""

USER_PROMPT_TEMPLATE = """\
Translate the following HTML fragment from English to French.
Preserve tags and attributes exactly; translate only the human-readable text nodes.
If a text node contains references like [1], (a), or numbers, keep them unchanged.
Apply the glossary and keep terminology stable. Use the following global context and style guidance for an academic register.

Global context: {global_context}
Style guide: {style_guide}
Recent term bank: {term_bank}
Glossary (JSON): {glossary_json}

HTML:
{html}
"""

REVISION_PROMPT_TEMPLATE = """\
You are revising a French translation of an English academic history manual. Polish the French while keeping facts, dates, numbers, citations, and HTML tags identical.
Improve academic register, avoid literal calques, prefer natural collocations (e.g., « vis-à-vis de », « à l’égard de », « relève de ») and enforce glossary choices.
Do not add or remove HTML tags or attributes. Never change numbers, dates, references [1] or (a).

Global context: {global_context}
Style guide: {style_guide}
Glossary (JSON): {glossary_json}
Preferred collocations: {preferred_collocations}
Recent term bank: {term_bank}

SOURCE HTML:
{source_html}

FRENCH DRAFT:
{translated_html}
"""


class AuditTrail:
    """Lightweight audit collector for prompt hashes and pipeline decisions."""

    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def record(self, kind: str, payload: Dict[str, Any]) -> None:
        entry = {"kind": kind, **payload}
        self.records.append(entry)

    def extend(self, records: List[Dict[str, Any]]) -> None:
        for rec in records:
            if isinstance(rec, dict) and rec.get("kind"):
                self.records.append(rec)

    def as_list(self) -> List[Dict[str, Any]]:
        return list(self.records)


class BaseTranslator(Protocol):
    def translate_html(
        self,
        html_fragment: str,
        glossary: Optional[Dict[str, str]] = None,
        style_guide: Optional[Dict[str, Any]] = None,
        global_context: str = "",
        term_bank: Optional[List[str]] = None,
    ) -> str:
        ...


@dataclass
class OpenAIConfig:
    model: str = "gpt-4.1-mini"
    temperature: float = 0.1
    max_output_tokens: int = 2000


@dataclass
class StyleGuide:
    register: str = "académique, neutre, précis"
    conventions: List[str] = None  # type: ignore[assignment]
    preferences: List[str] = None  # type: ignore[assignment]
    context: str = "manuel universitaire d’histoire médiévale"

    def as_prompt(self) -> str:
        conv = "; ".join(self.conventions or [])
        pref = "; ".join(self.preferences or [])
        return f"registre={self.register}; conventions={conv}; préférences={pref}; contexte={self.context}"


def _pluralize(term: str) -> List[str]:
    variants = []
    if term.endswith("y") and len(term) > 2:
        variants.append(term[:-1] + "ies")
    elif term.endswith("s") or term.endswith("x") or term.endswith("sh") or term.endswith("ch"):
        variants.append(term + "es")
    else:
        variants.append(term + "s")
    return variants


def _glossary_variants(term: str) -> List[str]:
    base = [term, term.lower(), term.capitalize()]
    variants = list(base)
    for candidate in base:
        if " " not in candidate and "-" not in candidate:
            variants.extend(_pluralize(candidate))
    if "-" in term:
        variants.append(term.replace("-", " "))
    return list(dict.fromkeys(v for v in variants if v))


_STYLE_GUIDE_ALLOWED_KEYS = {"register", "conventions", "preferences", "context"}


def _validate_style_guide_dict(data: Dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise ValueError("Le style guide doit être un objet JSON.")
    extra = set(data.keys()) - _STYLE_GUIDE_ALLOWED_KEYS
    if extra:
        raise ValueError(f"Clés inconnues dans le style guide: {', '.join(sorted(extra))}")
    for key in ["register", "context"]:
        if key not in data or not isinstance(data[key], str) or len(data[key].strip()) < 3:
            raise ValueError(f"Champ requis manquant ou vide: {key}")
    if "conventions" not in data or not isinstance(data["conventions"], list) or not data["conventions"]:
        raise ValueError("Le style guide doit contenir au moins une convention.")
    if not all(isinstance(item, str) and len(item.strip()) >= 3 for item in data["conventions"]):
        raise ValueError("Toutes les conventions doivent être des chaînes non vides.")
    prefs = data.get("preferences", []) or []
    if not isinstance(prefs, list) or not all(isinstance(item, str) and len(item.strip()) >= 3 for item in prefs):
        raise ValueError("Les préférences doivent être une liste de chaînes non vides.")


def _merge_style_guides(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged: Dict[str, Any] = {
        "register": base.get("register", "académique, neutre, précis"),
        "context": base.get("context", "manuel universitaire d’histoire médiévale"),
        "conventions": list(base.get("conventions", []) or []),
        "preferences": list(base.get("preferences", []) or []),
    }

    if override.get("register"):
        merged["register"] = override["register"]
    if override.get("context"):
        merged["context"] = override["context"]

    def _merge_list(key: str) -> List[str]:
        seen: set[str] = set()
        out: List[str] = []
        for val in (base.get(key, []) or []) + (override.get(key, []) or []):
            if not isinstance(val, str):
                continue
            val_clean = val.strip()
            if val_clean and val_clean not in seen:
                seen.add(val_clean)
                out.append(val_clean)
        return out

    merged["conventions"] = _merge_list("conventions")
    merged["preferences"] = _merge_list("preferences")
    return merged


def _load_json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    except FileNotFoundError:
        return None


DEFAULT_STYLE_GUIDE_FALLBACK: Dict[str, Any] = {
    "register": "académique, neutre, précis",
    "context": "manuel universitaire d’histoire médiévale et de la féodalité",
    "conventions": [
        "Siècles en chiffres romains suivis de 'e siècle' (ex : XIIe siècle)",
        "Ponctuation française : guillemets « » et espace fine insécable avant : ; ? !",
        "Préserver la numérotation, les dates et les citations telles quelles",
        "Utiliser des tournures analytiques sobres (éviter les calques anglicisés)",
    ],
    "preferences": [
        "Éviter le sujet impersonnel 'on' lorsque le sujet est connu",
        "Favoriser des collocations académiques comme 'à l'égard de', 'vis-à-vis de', 'relève de', 'renvoie à'",
        "Stabiliser la terminologie technique sur l'ensemble du document",
        "Favoriser les nominalisations modérées lorsque cela clarifie l'argument",
    ],
}


def load_style_guide(
    style_guide_path: Optional[str],
    default_path: Optional[str] = None,
    schema_path: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> StyleGuide:
    """Load, merge, and validate a style guide against a strict schema."""

    default_data = _load_json(default_path) or DEFAULT_STYLE_GUIDE_FALLBACK
    custom_data = _load_json(style_guide_path) or {}
    merged = _merge_style_guides(default_data, custom_data)

    schema = _load_json(schema_path)
    if schema_path and schema is None and logger:
        logger.warning("Schéma de style introuvable à %s, utilisation du validateur interne.", schema_path)
    _validate_style_guide_dict(merged)

    if logger:
        logger.info(
            "Style guide chargé (base=%s, override=%s, conventions=%s, préférences=%s)",
            default_path or "fallback",
            style_guide_path or "(none)",
            len(merged.get("conventions", [])),
            len(merged.get("preferences", [])),
        )

    return StyleGuide(
        register=merged.get("register", "académique, neutre, précis"),
        conventions=merged.get("conventions", []),
        preferences=merged.get("preferences", []),
        context=merged.get("context", "manuel universitaire d’histoire médiévale"),
    )


def build_glossary_variant_map(glossary_map: Optional[Dict[str, str]]) -> Dict[str, str]:
    """Return a variant->translation map to stabilize glossary application (pre/post)."""

    if not glossary_map:
        return {}
    variants: Dict[str, str] = {}
    for term, translation in glossary_map.items():
        for variant in _glossary_variants(term):
            variants[variant] = translation
    return variants


def _compile_term_pattern(term: str) -> re.Pattern[str]:
    if " " in term or "-" in term:
        return re.compile(re.escape(term), flags=re.IGNORECASE)
    return re.compile(rf"\b{re.escape(term)}\b", flags=re.IGNORECASE)


_NUMBER_TOKEN_RE = re.compile(r"\d{1,4}(?:[.,]\d+)?")
_DATE_RANGE_RE = re.compile(r"\b\d{1,4}(?:[/.-]\d{1,2}){1,2}\b")
_CITATION_TOKEN_RE = re.compile(r"\[[0-9]+\]|\([a-zA-Z]\)")


def _mask_tokens(text: str, pattern: re.Pattern[str], prefix: str, counter: List[int]) -> Tuple[str, Dict[str, str]]:
    placeholders: Dict[str, str] = {}

    def _repl(match: re.Match[str]) -> str:  # noqa: ANN001 - inline replacement
        placeholder = f"¤{prefix}{counter[0]}¤"
        placeholders[placeholder] = match.group(0)
        counter[0] += 1
        return placeholder

    return pattern.sub(_repl, text), placeholders


def mask_numbers_dates_citations(html_fragment: str) -> Tuple[str, Dict[str, str]]:
    """Mask numbers/dates/citations in text nodes to prevent model drift."""

    soup = BeautifulSoup(f"<div>{html_fragment}</div>", "html.parser")
    placeholders: Dict[str, str] = {}
    counter = [0]

    for node in list(soup.descendants):
        if not isinstance(node, NavigableString) or isinstance(node, Comment):
            continue
        parent = node.parent
        if parent and parent.name in ("script", "style"):
            continue
        text = str(node)
        updated, nums = _mask_tokens(text, _DATE_RANGE_RE, "DATE", counter)
        placeholders.update(nums)
        updated, nums = _mask_tokens(updated, _NUMBER_TOKEN_RE, "NUM", counter)
        placeholders.update(nums)
        updated, nums = _mask_tokens(updated, _CITATION_TOKEN_RE, "CIT", counter)
        placeholders.update(nums)
        if updated != text:
            node.replace_with(updated)

    return "".join(str(x) for x in soup.div.contents), placeholders


def unmask_numbers_dates_citations(html_fragment: str, placeholders: Dict[str, str]) -> str:
    updated = html_fragment
    for token, original in placeholders.items():
        updated = updated.replace(token, original)
    return updated


def _precompile_glossary_patterns(glossary_variants: Dict[str, str]) -> List[Tuple[Pattern[str], str, str]]:
    items = sorted(glossary_variants.items(), key=lambda kv: len(kv[0]), reverse=True)
    compiled: List[Tuple[Pattern[str], str, str]] = []
    for src, tgt in items:
        compiled.append((_compile_term_pattern(src), src, tgt))
    return compiled


def mask_glossary_terms(html_fragment: str, glossary_variants: Dict[str, str]) -> Tuple[str, Dict[str, Dict[str, str]]]:
    """Mask glossary terms in text nodes with placeholders before sending to a model."""

    if not glossary_variants:
        return html_fragment, {}

    soup = BeautifulSoup(f"<div>{html_fragment}</div>", "html.parser")
    placeholders: Dict[str, Dict[str, str]] = {}
    compiled = _precompile_glossary_patterns(glossary_variants)
    counter = 0

    for node in list(soup.descendants):
        if not isinstance(node, NavigableString) or isinstance(node, Comment):
            continue
        parent = node.parent
        if parent and parent.name in ("script", "style"):
            continue
        text = str(node)
        updated = text

        for pattern, src, tgt in compiled:

            def _repl(match: re.Match[str]) -> str:  # noqa: ANN001 - inline for regex
                nonlocal counter
                placeholder = f"¤GLOSS{counter}¤"
                placeholders[placeholder] = {
                    "replacement": tgt,
                    "source": src,
                    "matched": match.group(0),
                }
                counter += 1
                return placeholder

            updated = pattern.sub(_repl, updated)

        if updated != text:
            node.replace_with(updated)

    return "".join(str(x) for x in soup.div.contents), placeholders


def unmask_glossary_terms(html_fragment: str, placeholders: Dict[str, Dict[str, str]]) -> str:
    if not placeholders:
        return html_fragment
    updated = html_fragment
    for token, replacement in placeholders.items():
        if isinstance(replacement, dict):
            rep = replacement.get("replacement") or replacement.get("tgt") or ""
        else:
            rep = str(replacement)
        updated = updated.replace(token, rep)
    return updated


def _normalize_tm_text(text: str) -> str:
    """Linguistically light normalization for TM scoring."""

    lowered = text.lower().strip()
    lowered = re.sub(r"[\s\u00A0]+", " ", lowered)
    lowered = re.sub(r"[\W_]+", " ", lowered, flags=re.UNICODE)
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


class TranslationMemory:
    def __init__(
        self,
        max_segments: int = 200,
        max_terms: int = 200,
        storage_path: Optional[str | Path] = None,
        fuzzy_threshold: float = 0.86,
    ):
        self.segment_bank: List[Dict[str, str]] = []
        self.term_bank: List[Tuple[str, str]] = []
        self.variant_map: Dict[str, set[str]] = {}
        self.max_segments = max_segments
        self.max_terms = max_terms
        self.storage_path = Path(storage_path) if storage_path else None
        self.fuzzy_threshold = fuzzy_threshold
        self._lock = threading.Lock()
        self._loaded_ids: set[str] = set()

        if self.storage_path and self.storage_path.exists():
            self._load()

    def _load(self) -> None:
        if not self.storage_path:
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not rec.get("src") or not rec.get("tgt"):
                        continue
                    rec_id = rec.get("id") or sha1_text(rec["src"])
                    if rec_id in self._loaded_ids:
                        continue
                    rec["id"] = rec_id
                    rec["norm_src"] = rec.get("norm_src") or _normalize_tm_text(rec["src"])
                    self.segment_bank.append(rec)
                    self._loaded_ids.add(rec_id)
        except FileNotFoundError:
            return
        self.segment_bank = self.segment_bank[-self.max_segments :]

    def _persist(self, record: Dict[str, str]) -> None:
        if not self.storage_path:
            return
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def add_segment(self, source: str, target: str, target_html: str = "", logger: Optional[logging.Logger] = None) -> None:
        if not source or not target:
            return
        rec_id = sha1_text(source)
        record = {
            "src": source,
            "norm_src": _normalize_tm_text(source),
            "tgt": target,
            "tgt_html": target_html,
            "id": rec_id,
        }
        with self._lock:
            variants = self.variant_map.setdefault(rec_id, set())
            if target in variants:
                return
            self.segment_bank = [rec for rec in self.segment_bank if rec.get("id") != rec_id]
            self.segment_bank.append(record)
            self.segment_bank = self.segment_bank[-self.max_segments :]
            self._loaded_ids.add(rec_id)
            self._persist(record)
            variants.add(target)
            if logger and len(variants) > 1:
                logger.warning(
                    "Variant translations detected for segment hash %s: %s",
                    rec_id,
                    "; ".join(sorted(variants)),
                )

    def add_term(self, term: str, translation: str) -> None:
        if not term or not translation:
            return
        with self._lock:
            self.term_bank.append((term, translation))
            self.term_bank = self.term_bank[-self.max_terms :]

    def recent_terms(self, n: int = 30) -> List[str]:
        with self._lock:
            return [f"{src} -> {tgt}" for src, tgt in self.term_bank[-n:]]

    def fuzzy_lookup(self, source: str, threshold: Optional[float] = None) -> Optional[Dict[str, Any]]:
        if not source:
            return None
        best: Tuple[float, Optional[Dict[str, Any]]] = (0.0, None)
        th = threshold if threshold is not None else self.fuzzy_threshold
        norm_source = _normalize_tm_text(source)
        with self._lock:
            for rec in self.segment_bank:
                candidate_norm = rec.get("norm_src") or _normalize_tm_text(rec.get("src", ""))
                char_ratio = SequenceMatcher(None, norm_source, candidate_norm).ratio()
                token_ratio = SequenceMatcher(None, norm_source.split(), candidate_norm.split()).ratio()
                ratio = (char_ratio + token_ratio) / 2
                if ratio > best[0]:
                    best = (ratio, rec)
        if best[0] >= th and best[1]:
            rec = dict(best[1])
            rec["score"] = best[0]
            return rec
        return None


class RateLimiter:
    """Simple thread-safe rate limiter (requests per minute)."""

    def __init__(self, requests_per_minute: Optional[int] = None):
        self.requests_per_minute = requests_per_minute or 0
        self.interval = 60.0 / self.requests_per_minute if self.requests_per_minute > 0 else 0.0
        self._lock = threading.Lock()
        self._last_ts = 0.0

    def wait(self) -> None:
        if self.interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            delta = now - self._last_ts
            if delta < self.interval:
                time.sleep(self.interval - delta)
            self._last_ts = time.monotonic()


class MissingApiKeyError(RuntimeError):
    """Raised when a required provider API key is missing."""


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
            raise MissingApiKeyError(
                "OPENAI_API_KEY manquant : définissez la variable d'environnement ou ajoutez-la à votre .env."
            )
        self.cfg = cfg or OpenAIConfig()

        from openai import OpenAI  # type: ignore

        self._client = OpenAI(api_key=self.api_key)

    def translate_html(
        self,
        html_fragment: str,
        glossary: Optional[Dict[str, str]] = None,
        style_guide: Optional[Dict[str, Any]] = None,
        global_context: str = "",
        term_bank: Optional[List[str]] = None,
    ) -> str:
        glossary_json = json.dumps(glossary or {}, ensure_ascii=False)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            glossary_json=glossary_json,
            html=html_fragment,
            global_context=global_context,
            style_guide=json.dumps(style_guide or {}, ensure_ascii=False),
            term_bank="; ".join(term_bank or []),
        )

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
            raise MissingApiKeyError(
                "DEEPL_AUTH_KEY manquant : définissez la variable d'environnement ou ajoutez-la à votre .env."
            )
        self.formality = formality
        self.preserve_formatting = preserve_formatting

        import deepl  # type: ignore

        self._deepl = deepl.Translator(self.auth_key)

    def translate_html(
        self,
        html_fragment: str,
        glossary: Optional[Dict[str, str]] = None,
        style_guide: Optional[Dict[str, Any]] = None,
        global_context: str = "",
        term_bank: Optional[List[str]] = None,
    ) -> str:
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

    def translate_html(
        self,
        html_fragment: str,
        glossary: Optional[Dict[str, str]] = None,
        style_guide: Optional[Dict[str, Any]] = None,
        global_context: str = "",
        term_bank: Optional[List[str]] = None,
    ) -> str:
        soup = BeautifulSoup(f"<div>{html_fragment}</div>", "html.parser")
        for node in soup.find_all(string=True):
            if node.parent.name in ("script", "style"):
                continue
            txt = str(node)
            if txt.strip():
                node.replace_with(txt.replace(txt, f"{txt}"))
        return "".join(str(x) for x in soup.div.contents)


class OpenAIRevisor(OpenAITranslator):
    """Second-pass academic reviser using OpenAI."""

    def revise_html(
        self,
        source_html: str,
        translated_html: str,
        glossary: Optional[Dict[str, str]] = None,
        style_guide: Optional[Dict[str, Any]] = None,
        global_context: str = "",
        term_bank: Optional[List[str]] = None,
        preferred_collocations: Optional[List[str]] = None,
    ) -> str:
        glossary_json = json.dumps(glossary or {}, ensure_ascii=False)
        prompt = REVISION_PROMPT_TEMPLATE.format(
            global_context=global_context,
            style_guide=json.dumps(style_guide or {}, ensure_ascii=False),
            glossary_json=glossary_json,
            preferred_collocations=", ".join(preferred_collocations or []),
            term_bank="; ".join(term_bank or []),
            source_html=source_html,
            translated_html=translated_html,
        )
        resp = self._client.chat.completions.create(
            model=self.cfg.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_HTML_TRANSLATION},
                {"role": "user", "content": prompt},
            ],
            temperature=self.cfg.temperature,
            max_tokens=self.cfg.max_output_tokens,
        )
        content = resp.choices[0].message.content or ""
        return _strip_code_fences(content).strip()


def _strip_code_fences(s: str) -> str:
    fence = re.compile(r"^\s*```(?:html)?\s*([\s\S]*?)\s*```\s*$", re.IGNORECASE)
    m = fence.match(s.strip())
    return m.group(1) if m else s


def _guard_chunk_html(chunk: List[Dict[str, Any]]) -> str:
    """Wrap a list of segments with sentinels and guard attributes for stricter validation."""

    guarded = [
        f'<!--SEG_START:{seg["id"]}-->'
        f'<{seg["tag"]} data-seg-id="{seg["id"]}" data-guard="seg">{seg["html"]}</{seg["tag"]}>'
        f'<!--SEG_END:{seg["id"]}-->'
        for seg in chunk
    ]
    return "<div data-guard=\"container\">" + "".join(guarded) + "</div>"


def _validate_guarded_chunk(html: str, chunk: List[Dict[str, Any]]) -> None:
    """Ensure the guarded chunk preserves expected data markers before/after model calls."""

    soup = BeautifulSoup(html, "html.parser")
    container = soup.find("div") or soup
    extracted = container.find_all(attrs={"data-seg-id": True})
    if len(extracted) != len(chunk):
        raise ValueError(
            f"Guard validation failed: expected {len(chunk)} segments, found {len(extracted)} in guarded HTML."
        )
    expected_tags = [seg["tag"] for seg in chunk]
    actual_tags = [el.name for el in extracted]
    if expected_tags != actual_tags:
        raise ValueError(
            "Guard validation failed: tag sequence changed before/after translation call."
        )


def translate_segments(
    segments: List[Dict[str, Any]],
    translator: BaseTranslator,
    glossary_map: Optional[Dict[str, str]] = None,
    style_guide: Optional[StyleGuide] = None,
    global_context: str = "",
    translation_memory: Optional[TranslationMemory] = None,
    max_chars_per_chunk: int = 9000,
    max_segments_per_chunk: int = 8,
    parallel_workers: int = 1,
    structure_guard: bool = True,
    max_retries: int = 2,
    retry_backoff: float = 2.0,
    rate_limiter: Optional[RateLimiter] = None,
    mask_numbers_dates: bool = True,
    audit: Optional[AuditTrail] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Translate segments, in contextual chunks, preserving structure."""

    out: List[Dict[str, Any]] = [dict(seg) for seg in segments]
    id_to_index = {seg["id"]: i for i, seg in enumerate(out)}
    style_prompt = style_guide.as_prompt() if style_guide else ""
    tm = translation_memory or TranslationMemory()
    glossary_variants = build_glossary_variant_map(glossary_map)
    pending: List[Dict[str, Any]] = []
    for seg in out:
        match = tm.fuzzy_lookup(seg.get("text", ""))
        if match:
            seg["html_translated"] = match.get("tgt_html") or seg.get("html_translated") or match.get("tgt", "")
            seg["text_translated"] = match.get("tgt") or seg.get("text_translated") or ""
            seg["tm_match"] = {"id": match.get("id"), "score": match.get("score")}
            continue
        pending.append(seg)

    chunks: List[List[Dict[str, Any]]] = list(
        chunk_segments_contextual(
            pending,
            max_chars_per_chunk=max_chars_per_chunk,
            max_segments_per_chunk=max_segments_per_chunk,
        )
    )

    def _process_chunk(chunk: List[Dict[str, Any]]) -> List[Tuple[str, str, str]]:
        chunk_html = _guard_chunk_html(chunk) if structure_guard else "<div>" + "".join(
            f'<{seg["tag"]} data-seg-id="{seg["id"]}">{seg["html"]}</{seg["tag"]}>'
            for seg in chunk
        ) + "</div>"

        chunk_html, glossary_placeholders = mask_glossary_terms(chunk_html, glossary_variants)
        number_placeholders: Dict[str, str] = {}
        if mask_numbers_dates:
            chunk_html, number_placeholders = mask_numbers_dates_citations(chunk_html)

        if structure_guard:
            _validate_guarded_chunk(chunk_html, chunk)

        if audit:
            audit.record(
                "translation_prompt",
                {
                    "chunk_size": len(chunk),
                    "chunk_hash": sha1_text(chunk_html),
                    "chars": len(chunk_html),
                    "glossary_tokens": len(glossary_placeholders),
                    "number_tokens": len(number_placeholders),
                },
            )

        translated_chunk = ""
        attempts = max(1, max_retries + 1)
        for attempt in range(attempts):
            if rate_limiter:
                rate_limiter.wait()
            try:
                candidate = translator.translate_html(
                    chunk_html,
                    glossary=glossary_map or {},
                    style_guide={"style": style_prompt},
                    global_context=global_context,
                    term_bank=tm.recent_terms(),
                ).strip()

                candidate = unmask_numbers_dates_citations(candidate, number_placeholders)
                candidate = unmask_glossary_terms(candidate, glossary_placeholders)

                if structure_guard:
                    ok, issues = compare_html_structure(chunk_html, candidate)
                    if not ok:
                        raise ValueError(
                            f"HTML structure drift detected: {'; '.join(issues)}"
                        )
                    _validate_guarded_chunk(candidate, chunk)

                translated_chunk = candidate
                break
            except Exception as exc:  # noqa: PERF203 - retries intentionally broad
                if logger:
                    logger.warning(
                        "Chunk %s attempt %s/%s failed: %s",
                        [seg.get("id") for seg in chunk],
                        attempt + 1,
                        attempts,
                        exc,
                    )
                if attempt < attempts - 1:
                    time.sleep(retry_backoff * (2**attempt))
                else:
                    raise

        soup = BeautifulSoup(translated_chunk or chunk_html, "html.parser")
        container = soup.find("div") or soup

        extracted = container.find_all(attrs={"data-seg-id": True})
        updates: List[Tuple[str, str, str]] = []
        if extracted and len(extracted) == len(chunk):
            for el in extracted:
                seg_id = el.get("data-seg-id", "")
                inner = "".join(str(x) for x in el.contents)
                text = compact_whitespace(el.get_text(" ", strip=True))
                updates.append((seg_id, inner, text))
                tm.add_segment(
                    next((s.get("text", "") for s in chunk if s["id"] == seg_id), ""),
                    text,
                    target_html=inner,
                    logger=logger,
                )
        else:
            els = []
            for seg in chunk:
                els.append(container.find(seg["tag"]))
                if els[-1] is not None:
                    els[-1].extract()
            if any(e is None for e in els):
                raise ValueError("Unable to split translated chunk back into segments. The model likely altered structure.")
            for seg, el in zip(chunk, els):
                inner = "".join(str(x) for x in el.contents)
                text = compact_whitespace(el.get_text(" ", strip=True))
                updates.append((seg["id"], inner, text))
                tm.add_segment(seg.get("text", ""), text, target_html=inner, logger=logger)
        return updates

    if parallel_workers <= 1:
        for chunk in chunks:
            for seg_id, inner, text in _process_chunk(chunk):
                idx = id_to_index.get(seg_id)
                if idx is None:
                    continue
                out[idx]["html_translated"] = inner
                out[idx]["text_translated"] = text
    else:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(_process_chunk, chunk) for chunk in chunks]
            for future in as_completed(futures):
                updates = future.result()
                for seg_id, inner, text in updates:
                    idx = id_to_index.get(seg_id)
                    if idx is None:
                        continue
                    out[idx]["html_translated"] = inner
                    out[idx]["text_translated"] = text

    return out


def revise_segments(
    segments: List[Dict[str, Any]],
    revisor: OpenAIRevisor,
    glossary_map: Optional[Dict[str, str]] = None,
    style_guide: Optional[StyleGuide] = None,
    global_context: str = "",
    translation_memory: Optional[TranslationMemory] = None,
    preferred_collocations: Optional[List[str]] = None,
    parallel_workers: int = 1,
    structure_guard: bool = True,
    max_retries: int = 2,
    retry_backoff: float = 2.0,
    rate_limiter: Optional[RateLimiter] = None,
    mask_numbers_dates: bool = True,
    audit: Optional[AuditTrail] = None,
    logger: Optional[logging.Logger] = None,
) -> List[Dict[str, Any]]:
    """Second pass over already translated segments to polish academic style."""

    out: List[Dict[str, Any]] = [dict(seg) for seg in segments]
    id_to_index = {seg["id"]: i for i, seg in enumerate(out)}
    style_prompt = style_guide.as_prompt() if style_guide else ""
    tm = translation_memory or TranslationMemory()
    glossary_variants = build_glossary_variant_map(glossary_map)

    def _revise(seg: Dict[str, Any]) -> Tuple[str, str, str]:
        src_html = f'<{seg["tag"]}>{seg.get("html", "")}</{seg["tag"]}>'
        tr_html = f'<{seg["tag"]}>{seg.get("html_translated", "")}</{seg["tag"]}>'
        guarded_html = tr_html
        if structure_guard:
            guarded_html = f'<div data-guard="container">{tr_html}</div>'

        masked_html, glossary_placeholders = mask_glossary_terms(guarded_html, glossary_variants)
        number_placeholders: Dict[str, str] = {}
        if mask_numbers_dates:
            masked_html, number_placeholders = mask_numbers_dates_citations(masked_html)

        if audit:
            audit.record(
                "revision_prompt",
                {
                    "segment_id": seg.get("id"),
                    "hash": sha1_text(masked_html),
                    "glossary_tokens": len(glossary_placeholders),
                    "number_tokens": len(number_placeholders),
                },
            )

        revised = ""
        attempts = max(1, max_retries + 1)
        for attempt in range(attempts):
            if rate_limiter:
                rate_limiter.wait()
            try:
                candidate = revisor.revise_html(
                    source_html=src_html,
                    translated_html=masked_html,
                    glossary=glossary_map or {},
                    style_guide={"style": style_prompt},
                    global_context=global_context,
                    term_bank=tm.recent_terms(),
                    preferred_collocations=preferred_collocations,
                )

                candidate = unmask_numbers_dates_citations(candidate, number_placeholders)
                candidate = unmask_glossary_terms(candidate, glossary_placeholders)
                if structure_guard:
                    ok, issues = compare_html_structure(guarded_html, candidate)
                    if not ok:
                        raise ValueError(
                            f"HTML structure drift detected during revision: {'; '.join(issues)}"
                        )
                revised = candidate
                break
            except Exception as exc:  # noqa: PERF203 - retries intentionally broad
                if logger:
                    logger.warning(
                        "Revision of segment %s attempt %s/%s failed: %s",
                        seg.get("id"),
                        attempt + 1,
                        attempts,
                        exc,
                    )
                if attempt < attempts - 1:
                    time.sleep(retry_backoff * (2**attempt))
                else:
                    raise

        soup = BeautifulSoup(revised or tr_html, "html.parser")
        container = soup.find(seg["tag"]) or soup
        inner = "".join(str(x) for x in container.contents)
        text = compact_whitespace(container.get_text(" ", strip=True))
        tm.add_segment(seg.get("text", ""), text, target_html=inner, logger=logger)
        return seg["id"], inner, text

    if parallel_workers <= 1:
        for seg in out:
            seg_id, inner, text = _revise(seg)
            idx = id_to_index.get(seg_id)
            if idx is None:
                continue
            out[idx]["html_translated"] = inner
            out[idx]["text_translated"] = text
    else:
        with ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(_revise, seg) for seg in out]
            for future in as_completed(futures):
                seg_id, inner, text = future.result()
                idx = id_to_index.get(seg_id)
                if idx is None:
                    continue
                out[idx]["html_translated"] = inner
                out[idx]["text_translated"] = text
    return out

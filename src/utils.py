from __future__ import annotations

import hashlib
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Dict, Any, Optional


_LIGATURES = {
    "ﬀ": "ff",
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
}

_WEIRD_SPACES_PATTERN = re.compile(r"[\u00A0\u2000-\u200B\u202F\u205F\u3000]")


def setup_logger(log_dir: str | Path, name: str = "streamlit-ocr-translator") -> logging.Logger:
    """Create a simple file+console logger."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers (e.g., Streamlit reruns)
    if logger.handlers:
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_dir / "app.log", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def normalize_ocr_text(text: str) -> str:
    """
    Basic cleanup helpers for OCR-ish artifacts:
    - ligatures (ﬁ -> fi, etc.)
    - non-breaking/odd spaces -> normal space
    - collapse duplicated spaces
    - normalize newlines
    """
    for k, v in _LIGATURES.items():
        text = text.replace(k, v)
    text = _WEIRD_SPACES_PATTERN.sub(" ", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Fix "space before punctuation" issues can be done post-translation (FR rules)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def safe_mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass(frozen=True)
class Chunk:
    """A translation chunk: a group of segments."""
    chunk_id: str
    segment_ids: List[str]
    html: str


def chunk_segments(
    segments: List[Dict[str, Any]],
    max_chars_per_chunk: int = 9000,
    max_segments_per_chunk: int = 8,
) -> Iterator[List[Dict[str, Any]]]:
    """
    Group segments into chunks constrained by:
    - maximum characters
    - maximum number of segments
    """
    buf: List[Dict[str, Any]] = []
    buf_chars = 0

    for seg in segments:
        seg_html = seg.get("html", "")
        seg_len = len(seg_html)

        # If single segment exceeds max_chars, yield it alone.
        if not buf and seg_len >= max_chars_per_chunk:
            yield [seg]
            continue

        if (buf and (buf_chars + seg_len > max_chars_per_chunk)) or (len(buf) >= max_segments_per_chunk):
            yield buf
            buf = []
            buf_chars = 0

        buf.append(seg)
        buf_chars += seg_len

    if buf:
        yield buf


def chunk_segments_contextual(
    segments: List[Dict[str, Any]],
    max_chars_per_chunk: int = 9000,
    max_segments_per_chunk: int = 8,
) -> Iterator[List[Dict[str, Any]]]:
    """Chunk segments by logical context groups (sections/lists) then apply size caps."""

    if not segments:
        return iter(())

    def _context_key(seg: Dict[str, Any]) -> str:
        return seg.get("context_group") or seg.get("section_id") or "__default__"

    grouped: List[List[Dict[str, Any]]] = []
    current_key: Optional[str] = None
    buf: List[Dict[str, Any]] = []

    for seg in segments:
        key = _context_key(seg)
        if current_key is None:
            current_key = key
        if key != current_key:
            if buf:
                grouped.append(buf)
            buf = [seg]
            current_key = key
        else:
            buf.append(seg)

    if buf:
        grouped.append(buf)

    for group in grouped:
        # Within a context group, still respect size constraints.
        cursor: List[Dict[str, Any]] = []
        cursor_chars = 0
        for seg in group:
            seg_len = len(seg.get("html", ""))
            if (cursor and (cursor_chars + seg_len > max_chars_per_chunk)) or (
                len(cursor) >= max_segments_per_chunk
            ):
                yield cursor
                cursor = []
                cursor_chars = 0

            cursor.append(seg)
            cursor_chars += seg_len

        if cursor:
            yield cursor


def build_context_windows(
    segments: List[Dict[str, Any]],
    window: int = 1,
) -> Dict[str, Dict[str, str]]:
    """
    Build a light context window for each segment with the previous/next textual neighbors.

    The window is inclusive of up to ``window`` segments before and after the current
    index. This is meant to be injected into prompts so models see immediate context
    without changing chunking boundaries.
    """

    if window <= 0 or not segments:
        return {}

    context_map: Dict[str, Dict[str, str]] = {}
    texts = [compact_whitespace(seg.get("text", "")) for seg in segments]

    for i, seg in enumerate(segments):
        prev_idx_start = max(0, i - window)
        next_idx_end = min(len(segments), i + window + 1)
        previous = [t for t in texts[prev_idx_start:i] if t]
        following = [t for t in texts[i + 1 : next_idx_end] if t]
        context_map[seg.get("id", str(i))] = {
            "previous": " | ".join(previous[-window:]),
            "next": " | ".join(following[:window]),
        }

    return context_map


def strip_control_chars(s: str) -> str:
    # Keep \n and \t
    return re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", s)


def compact_whitespace(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()

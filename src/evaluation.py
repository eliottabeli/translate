from __future__ import annotations

import importlib
import os
import random
from statistics import mean
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

from .postproc import terminology_report
from .qa import back_translate_html_openai
from .utils import compact_whitespace


def _select_long_sample(
    segments: List[Dict[str, Any]],
    min_chars: int = 240,
    max_samples: int = 20,
) -> List[Dict[str, Any]]:
    long_segments = [seg for seg in segments if len(seg.get("text_translated", "")) >= min_chars]
    if len(long_segments) <= max_samples:
        return long_segments
    return random.sample(long_segments, k=max_samples)


def _compute_bleu_via_backtranslation(
    sample: List[Dict[str, Any]],
    api_key: Optional[str],
    model: str,
) -> Dict[str, Any]:
    if importlib.util.find_spec("sacrebleu") is None:  # pragma: no cover - optional dependency
        return {"metric": "bleu", "score": None, "warning": "sacrebleu missing"}

    sacrebleu = importlib.import_module("sacrebleu")  # type: ignore

    if not api_key:
        return {"metric": "bleu", "score": None, "warning": "OPENAI_API_KEY absent for back-translation"}

    hypotheses: List[str] = []
    references: List[str] = []
    for seg in sample:
        tr_html = seg.get("html_translated", "")
        wrapped = f'<{seg.get("tag", "div")}>{tr_html}</{seg.get("tag", "div")}>'
        bt_html = back_translate_html_openai(wrapped, api_key=api_key, model=model)
        hypotheses.append(compact_whitespace(BeautifulSoup(bt_html, "html.parser").get_text(" ", strip=True)))
        references.append(compact_whitespace(seg.get("text", "")))

    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    return {"metric": "bleu", "score": bleu.score, "samples": len(hypotheses)}


def _compute_comet_qe(sample: List[Dict[str, Any]], model_name: str = "Unbabel/wmt22-cometkiwi-da") -> Dict[str, Any]:
    if importlib.util.find_spec("comet") is None:  # pragma: no cover - optional dependency
        return {"metric": "comet", "score": None, "warning": "COMET not available"}

    from comet import download_model, load_from_checkpoint  # type: ignore

    data = [
        {"src": compact_whitespace(seg.get("text", "")), "mt": compact_whitespace(seg.get("text_translated", ""))}
        for seg in sample
        if seg.get("text") and seg.get("text_translated")
    ]
    if not data:
        return {"metric": "comet", "score": None, "warning": "No data to score."}

    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    outputs = model.predict(data, batch_size=min(8, len(data)), gpus=0)
    sys_score = outputs.get("system_score") if isinstance(outputs, dict) else None
    seg_scores = outputs.get("scores") if isinstance(outputs, dict) else None
    return {
        "metric": "comet",
        "score": float(sys_score) if sys_score is not None else None,
        "mean_segment_score": float(mean(seg_scores)) if seg_scores else None,
        "samples": len(data),
    }


def _style_glossary_check(
    sample: List[Dict[str, Any]],
    glossary_map: Dict[str, str],
    style_conventions: Optional[List[str]],
) -> Dict[str, Any]:
    term_report = terminology_report(sample, glossary_map) if glossary_map else []
    total_terms = len(glossary_map)
    missing_terms = sum(row.get("occurrences", 0) > 0 for row in term_report)
    coverage = None if total_terms == 0 else 1 - (missing_terms / total_terms)
    style_flags = []
    for convention in style_conventions or []:
        style_flags.append({"convention": convention, "status": "reminder"})
    return {
        "metric": "style_glossary",
        "glossary_coverage": coverage,
        "missing_glossary_terms": missing_terms,
        "style_conventions": style_flags,
    }


def run_automatic_evaluation(
    translated: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    glossary_map: Dict[str, str],
    style_guide,
    logger,
) -> Dict[str, Any]:
    ev_cfg = cfg.get("evaluation", {})
    if not ev_cfg.get("enabled", False):
        logger.info("   Evaluation disabled (evaluation.enabled=false).")
        return {}

    sample = _select_long_sample(
        translated,
        min_chars=int(ev_cfg.get("min_chars", 240)),
        max_samples=int(ev_cfg.get("max_samples", 20)),
    )
    if not sample:
        logger.warning("   Evaluation skipped: no long segments found.")
        return {"warning": "no_long_segments"}

    logger.info("   Evaluating %s segments for BLEU/COMET/checklists", len(sample))
    openai_model = ev_cfg.get("backtranslation_model", "gpt-4.1-mini")
    metrics: List[Dict[str, Any]] = []
    metrics.append(
        _compute_bleu_via_backtranslation(sample, api_key=os.getenv("OPENAI_API_KEY", ""), model=openai_model)
    )
    metrics.append(_compute_comet_qe(sample, model_name=ev_cfg.get("comet_model", "Unbabel/wmt22-cometkiwi-da")))
    metrics.append(_style_glossary_check(sample, glossary_map, getattr(style_guide, "conventions", [])))

    return {
        "samples": len(sample),
        "metrics": metrics,
    }

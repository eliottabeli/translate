from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from bs4 import BeautifulSoup

from src.utils import setup_logger
from src import storage
from src.html_parser import extract_segments, rebuild_html
from src.images_ocr import ocr_images_folder
from src.glossary import (
    extract_term_candidates,
    glossary_to_map,
    draft_glossary_rows,
    generate_translation_proposals,
)
from src.translator import (
    OpenAITranslator,
    OpenAIConfig,
    OpenAIRevisor,
    DeepLTranslator,
    DummyTranslator,
    MissingApiKeyError,
    translate_segments,
    revise_segments,
    load_style_guide,
    TranslationMemory,
    RateLimiter,
    AuditTrail,
)
from src.postproc import (
    apply_glossary_to_fragment,
    compare_html_structure,
    numbers_and_citations_check,
    learn_collocations,
    terminology_report,
    auto_correct_terminology,
    evaluate_register,
)


def segment_document(
    html_text: str,
    cfg: Dict[str, Any],
    paths: Dict[str, Any],
    logger,
    audit: AuditTrail,
    resume: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    segments_path = Path(paths["segments_json"])
    meta_path = Path(paths["segments_meta_json"])
    if resume and segments_path.exists() and meta_path.exists():
        logger.info("   Reusing existing segments from disk.")
        segments = storage.read_json(segments_path)
        meta = storage.read_json(meta_path)
    else:
        segments, meta = extract_segments(
            html_text,
            block_tags=cfg["segmentation"].get("block_tags"),
            include_empty=cfg["segmentation"].get("include_empty", False),
        )
        storage.write_json(segments_path, segments)
        storage.write_json(meta_path, meta)
        logger.info(f"   Extracted {len(segments)} segments.")
    audit.record("segmentation", {"segments": len(segments)})
    return segments, meta


def run_ocr(cfg: Dict[str, Any], paths: Dict[str, Any], logger, audit: AuditTrail) -> None:
    if not cfg.get("ocr", {}).get("enabled", False):
        logger.info("   OCR disabled (ocr.enabled=false).")
        return
    ocr_cfg = cfg["ocr"]
    ocr_results = ocr_images_folder(
        images_dir=paths["raw_images_dir"],
        lang=ocr_cfg.get("lang", "eng"),
        tesseract_cmd=ocr_cfg.get("tesseract_cmd", ""),
        only_if_filename_matches=ocr_cfg.get("only_if_filename_matches", ""),
    )
    storage.write_json(paths["images_ocr_json"], ocr_results)
    audit.record("ocr", {"images": len(ocr_results)})
    logger.info(f"   OCR done for {len(ocr_results)} images.")


def prepare_style_and_glossary(
    cfg: Dict[str, Any],
    paths: Dict[str, Any],
    segments: List[Dict[str, Any]],
    logger,
    audit: AuditTrail,
) -> Tuple[Any, Dict[str, str]]:
    style_guide = load_style_guide(
        paths.get("style_guide"),
        default_path=paths.get("style_guide_default"),
        schema_path=paths.get("style_guide_schema"),
        logger=logger,
    )
    audit.record(
        "style_guide",
        {
            "source": paths.get("style_guide") or "(none)",
            "merged_conventions": len(style_guide.conventions or []),
            "merged_preferences": len(style_guide.preferences or []),
        },
    )

    if not cfg.get("glossary", {}).get("enabled", True):
        logger.info("   Glossary disabled (glossary.enabled=false)")
        return style_guide, {}

    glossary_path = Path(paths["glossary_json"])
    if glossary_path.exists():
        glossary_rows = storage.read_json(glossary_path)
        logger.info(f"   Using existing glossary: {glossary_path}")
    else:
        candidates = extract_term_candidates(
            segments,
            top_n=cfg["glossary"].get("top_n", 400),
            min_len=cfg["glossary"].get("min_len", 3),
            stopwords_lang=cfg["glossary"].get("stopwords_lang", "english"),
        )
        storage.write_json(paths["term_candidates_json"], [c.as_dict() for c in candidates])
        glossary_rows = draft_glossary_rows(candidates, top_n=cfg["glossary"].get("draft_top_n", 250))
        storage.write_json(paths["glossary_draft_json"], glossary_rows)
        proposals_cfg = cfg["glossary"].get("proposals", {})
        if proposals_cfg.get("enabled", True):
            proposals = generate_translation_proposals(
                candidates,
                style_guide=style_guide.__dict__,
                top_n=proposals_cfg.get("top_n", 300),
                k=proposals_cfg.get("k", 3),
            )
            storage.write_json(paths.get("glossary_proposals_json", "data/segments/glossary_proposals.json"), proposals)
            logger.info(
                "   Draft glossary and proposals written to %s / %s",
                paths["glossary_draft_json"],
                paths.get("glossary_proposals_json", "data/segments/glossary_proposals.json"),
            )
    glossary_map = glossary_to_map(glossary_rows)
    storage.write_json(paths["glossary_json"], glossary_rows)
    logger.info(f"   Glossary size: {len(glossary_map)} terms.")
    audit.record("glossary", {"terms": len(glossary_map)})
    return style_guide, glossary_map


def build_translator(provider: str, cfg: Dict[str, Any]) -> Any:
    provider = provider.lower()
    if provider == "openai":
        ocfg = cfg["translation"].get("openai", {})
        return OpenAITranslator(
            cfg=OpenAIConfig(
                model=ocfg.get("model", "gpt-4.1-mini"),
                temperature=float(ocfg.get("temperature", 0.1)),
                max_output_tokens=int(ocfg.get("max_output_tokens", 2000)),
            )
        )
    if provider == "deepl":
        dcfg = cfg["translation"].get("deepl", {})
        return DeepLTranslator(
            formality=dcfg.get("formality", "more"),
            preserve_formatting=bool(dcfg.get("preserve_formatting", True)),
        )
    if provider == "dummy":
        return DummyTranslator()
    raise ValueError(f"Unknown translation provider: {provider}")


def translate_pipeline(
    segments: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    paths: Dict[str, Any],
    glossary_map: Dict[str, str],
    style_guide,
    audit: AuditTrail,
    logger,
) -> Tuple[List[Dict[str, Any]], TranslationMemory]:
    provider = cfg["translation"].get("provider", "openai").lower()
    try:
        translator = build_translator(provider, cfg)
    except MissingApiKeyError as exc:
        logger.error(str(exc))
        raise
    chcfg = cfg["translation"].get("chunking", {})
    sched_cfg = cfg["translation"].get("scheduling", {})
    guard_cfg = cfg["translation"].get("structure_guard", {})
    tm_cfg = cfg["translation"].get("translation_memory", {})
    safety_cfg = cfg["translation"].get("safety", {})
    limiter = RateLimiter(int(sched_cfg.get("requests_per_minute", 0)))
    tm = TranslationMemory(
        max_segments=int(tm_cfg.get("max_segments", 2000)),
        max_terms=int(tm_cfg.get("max_terms", 400)),
        storage_path=tm_cfg.get("path") or paths.get("translation_memory"),
        fuzzy_threshold=float(tm_cfg.get("fuzzy_threshold", 0.86)),
    )

    translated = translate_segments(
        segments,
        translator=translator,
        glossary_map=glossary_map,
        style_guide=style_guide,
        global_context=style_guide.context,
        translation_memory=tm,
        max_chars_per_chunk=int(chcfg.get("max_chars_per_chunk", 9000)),
        max_segments_per_chunk=int(chcfg.get("max_segments_per_chunk", 8)),
        parallel_workers=int(cfg["translation"].get("parallel_workers", 1)),
        structure_guard=bool(guard_cfg.get("enabled", True)),
        max_retries=int(sched_cfg.get("max_retries", 2)),
        retry_backoff=float(sched_cfg.get("retry_backoff_seconds", 2.0)),
        rate_limiter=limiter,
        mask_numbers_dates=bool(safety_cfg.get("mask_numbers_dates", True)),
        audit=audit,
        logger=logger,
    )
    storage.write_json(paths.get("translations_json", "data/segments/translations.json"), translated)
    return translated, tm


def run_revision(
    translated: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    paths: Dict[str, Any],
    glossary_map: Dict[str, str],
    style_guide,
    tm: TranslationMemory,
    collocations: List[str],
    provider: str,
    audit: AuditTrail,
    logger,
) -> List[Dict[str, Any]]:
    if not (cfg.get("revision", {}).get("enabled", True) and provider == "openai"):
        logger.info("   Revision disabled or unsupported provider.")
        return translated

    rcfg = cfg.get("revision", {}).get("openai", {})
    revisor = OpenAIRevisor(
        cfg=OpenAIConfig(
            model=rcfg.get("model", "gpt-4.1-mini"),
            temperature=float(rcfg.get("temperature", 0.1)),
            max_output_tokens=int(rcfg.get("max_output_tokens", 2000)),
        )
    )
    rev_sched = cfg.get("revision", {}).get("scheduling", {})
    rev_guard = cfg.get("revision", {}).get("structure_guard", {})
    rev_limiter = RateLimiter(int(rev_sched.get("requests_per_minute", 0)))
    rev_safety = cfg.get("revision", {}).get("safety", {})
    revised = revise_segments(
        translated,
        revisor=revisor,
        glossary_map=glossary_map,
        style_guide=style_guide,
        global_context=style_guide.context,
        translation_memory=tm,
        preferred_collocations=collocations,
        parallel_workers=int(cfg.get("revision", {}).get("parallel_workers", 1)),
        structure_guard=bool(rev_guard.get("enabled", True)),
        max_retries=int(rev_sched.get("max_retries", 2)),
        retry_backoff=float(rev_sched.get("retry_backoff_seconds", 2.0)),
        rate_limiter=rev_limiter,
        mask_numbers_dates=bool(rev_safety.get("mask_numbers_dates", True)),
        audit=audit,
        logger=logger,
    )
    storage.write_json(paths["translations_v2_json"], revised)
    return revised


def postprocess_segments(
    translated: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    glossary_map: Dict[str, str],
    logger,
) -> List[Dict[str, Any]]:
    post = cfg.get("postproc", {})
    for seg in translated:
        tr_html = seg.get("html_translated", "")
        if post.get("apply_glossary", True) and glossary_map:
            tr_html = apply_glossary_to_fragment(tr_html, glossary_map)
            seg["html_translated"] = tr_html
            seg["text_translated"] = BeautifulSoup(f"<div>{tr_html}</div>", "html.parser").get_text(" ", strip=True)

        if post.get("verify_html_structure", True):
            src_wrapped = f'<{seg["tag"]}>{seg.get("html", "")}</{seg["tag"]}>'
            tr_wrapped = f'<{seg["tag"]}>{seg.get("html_translated", "")}</{seg["tag"]}>'
            ok, issues = compare_html_structure(src_wrapped, tr_wrapped)
            seg["check_structure_ok"] = ok
            seg["check_structure_issues"] = issues

        if post.get("verify_numbers_dates", True):
            warnings = numbers_and_citations_check(seg.get("text", ""), seg.get("text_translated", ""))
            seg["check_numbers_citations"] = warnings

    return translated


def run_qa_reports(
    translated: List[Dict[str, Any]],
    cfg: Dict[str, Any],
    paths: Dict[str, Any],
    glossary_map: Dict[str, str],
    audit: AuditTrail,
    logger,
) -> None:
    qa_cfg = cfg.get("qa", {})
    report_rows: List[Dict[str, Any]] = []
    if qa_cfg.get("terminology_report", True) and glossary_map:
        report_rows = terminology_report(translated, glossary_map)
        storage.write_json(paths["terminology_report"], report_rows)
        logger.info(f"   Terminology report: {paths['terminology_report']}")

    if qa_cfg.get("auto_correct_terminology", True) and glossary_map and report_rows:
        corrections = auto_correct_terminology(translated, glossary_map, report_rows)
        if corrections:
            storage.write_json(
                paths.get("terminology_corrections", paths["terminology_report"] + ".patch.json"),
                corrections,
            )
            audit.extend(corrections)
            logger.info("   Applied %s terminology corrections", len(corrections))
            report_rows = terminology_report(translated, glossary_map)
            storage.write_json(paths["terminology_report"], report_rows)

    if qa_cfg.get("register_report", True):
        register_report = evaluate_register(translated)
        storage.write_json(paths.get("register_report", "logs/register_report.json"), register_report)
        audit.record("register", register_report)


def export_html(html_text: str, translated: List[Dict[str, Any]], cfg: Dict[str, Any], paths: Dict[str, Any], logger) -> None:
    translated_html = rebuild_html(html_text, translated, block_tags=cfg["segmentation"].get("block_tags"))
    exports_dir = Path(paths.get("exports_dir", "data/exports"))
    exports_dir.mkdir(parents=True, exist_ok=True)
    out_html = exports_dir / "translated_document.html"
    storage.write_text(out_html, translated_html)
    logger.info(f"   Exported: {out_html}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Full translation pipeline (HTML + images).")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    args = parser.parse_args()

    load_dotenv()

    cfg = storage.read_json(args.config)
    paths = cfg["paths"]
    resume = bool(cfg.get("pipeline", {}).get("resume", True))

    logger = setup_logger(paths.get("logs_dir", "logs"))
    audit = AuditTrail()

    raw_html_path = Path(paths["raw_html"])
    html_text = storage.read_text(raw_html_path)

    try:
        logger.info("1) Segmentation…")
        segments, _ = segment_document(html_text, cfg, paths, logger, audit, resume=resume)
    except Exception:
        logger.exception("Segmentation failed.")
        raise SystemExit(1)

    try:
        logger.info("2) OCR images…")
        run_ocr(cfg, paths, logger, audit)
    except Exception:
        logger.exception("OCR step failed.")
        raise SystemExit(1)

    logger.info("3) Glossary extraction…")
    try:
        style_guide, glossary_map = prepare_style_and_glossary(cfg, paths, segments, logger, audit)
    except Exception:
        logger.exception("Glossary extraction failed.")
        raise SystemExit(1)

    if cfg.get("glossary", {}).get("only_extract", False):
        logger.info("Glossary extraction only. Exiting.")
        return

    try:
        translated, tm = translate_pipeline(segments, cfg, paths, glossary_map, style_guide, audit, logger)
    except MissingApiKeyError:
        raise SystemExit(1)
    except Exception:
        logger.exception("Translation failed.")
        raise SystemExit(1)

    logger.info("5) Collocation learning for revision…")
    collocations = learn_collocations(translated)
    audit.record("collocations", {"count": len(collocations)})

    logger.info("6) Revision pass 2 (academic polish)…")
    try:
        provider = cfg["translation"].get("provider", "openai").lower()
        translated = run_revision(
            translated,
            cfg,
            paths,
            glossary_map,
            style_guide,
            tm,
            collocations,
            provider,
            audit,
            logger,
        )
    except Exception:
        logger.exception("Revision failed.")
        raise SystemExit(1)

    try:
        logger.info("7) Post-processing + checks…")
        translated = postprocess_segments(translated, cfg, glossary_map, logger)
        storage.write_json(paths["translations_json"], translated)
    except Exception:
        logger.exception("Post-processing failed.")
        raise SystemExit(1)

    qa_cfg = cfg.get("qa", {})
    if qa_cfg.get("enabled", True):
        try:
            logger.info("8) QA reports…")
            run_qa_reports(translated, cfg, paths, glossary_map, audit, logger)
        except Exception:
            logger.exception("QA stage failed.")
            raise SystemExit(1)
    else:
        logger.info("8) QA reports skipped (qa.enabled=false).")

    audit_path = paths.get("audit_report", "logs/audit.json")
    storage.write_json(audit_path, audit.as_list())
    logger.info(f"   Audit trail saved to: {audit_path}")

    try:
        logger.info("9) Rebuild final HTML…")
        export_html(html_text, translated, cfg, paths, logger)
    except Exception:
        logger.exception("Export failed.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()

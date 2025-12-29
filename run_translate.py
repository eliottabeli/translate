\
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from bs4 import BeautifulSoup

from src.utils import setup_logger
from src import storage
from src.html_parser import extract_segments, rebuild_html
from src.images_ocr import ocr_images_folder
from src.glossary import extract_glossary_candidates, glossary_to_map
from src.translator import (
    OpenAITranslator,
    OpenAIConfig,
    DeepLTranslator,
    DummyTranslator,
    translate_segments,
)
from src.postproc import apply_glossary_to_fragment, compare_html_structure, numbers_and_citations_check


def main() -> None:
    parser = argparse.ArgumentParser(description="Full translation pipeline (HTML + images).")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config.json")
    args = parser.parse_args()

    load_dotenv()

    cfg = storage.read_json(args.config)
    paths = cfg["paths"]

    logger = setup_logger(paths.get("logs_dir", "logs"))

    raw_html_path = Path(paths["raw_html"])
    html_text = storage.read_text(raw_html_path)

    logger.info("1) Segmentation…")
    segments, meta = extract_segments(
        html_text,
        block_tags=cfg["segmentation"].get("block_tags"),
        include_empty=cfg["segmentation"].get("include_empty", False),
    )
    storage.write_json(paths["segments_json"], segments)
    storage.write_json(paths["segments_meta_json"], meta)
    logger.info(f"   Extracted {len(segments)} segments.")

    logger.info("2) OCR images…")
    if cfg.get("ocr", {}).get("enabled", False):
        ocr_cfg = cfg["ocr"]
        ocr_results = ocr_images_folder(
            images_dir=paths["raw_images_dir"],
            lang=ocr_cfg.get("lang", "eng"),
            tesseract_cmd=ocr_cfg.get("tesseract_cmd", ""),
            only_if_filename_matches=ocr_cfg.get("only_if_filename_matches", ""),
        )
        storage.write_json(paths["images_ocr_json"], ocr_results)
        logger.info(f"   OCR done for {len(ocr_results)} images.")
    else:
        logger.info("   OCR disabled (ocr.enabled=false).")

    logger.info("3) Glossary extraction…")
    glossary_map = {}
    if cfg.get("glossary", {}).get("enabled", True):
        glossary_path = Path(paths["glossary_json"])
        if glossary_path.exists():
            glossary_rows = storage.read_json(glossary_path)
            logger.info(f"   Using existing glossary: {glossary_path}")
        else:
            glossary_rows = extract_glossary_candidates(
                segments,
                top_n=cfg["glossary"].get("top_n", 150),
                min_len=cfg["glossary"].get("min_len", 4),
                include_from_tags=cfg["glossary"].get("include_from_tags", ["em", "i", "strong"]),
                stopwords_lang=cfg["glossary"].get("stopwords_lang", "english"),
            )
            storage.write_json(glossary_path, glossary_rows)
            logger.info(f"   Glossary candidates written to: {glossary_path}")
        glossary_map = glossary_to_map(glossary_rows)
        logger.info(f"   Glossary forced translations: {len(glossary_map)}")
    else:
        logger.info("   Glossary disabled (glossary.enabled=false).")

    logger.info("4) Translation…")
    if not cfg.get("translation", {}).get("enabled", True):
        logger.info("   Translation disabled (translation.enabled=false). Exiting.")
        return

    provider = cfg["translation"].get("provider", "openai").lower()
    translator = None
    if provider == "openai":
        ocfg = cfg["translation"].get("openai", {})
        translator = OpenAITranslator(cfg=OpenAIConfig(
            model=ocfg.get("model", "gpt-4.1-mini"),
            temperature=float(ocfg.get("temperature", 0.1)),
            max_output_tokens=int(ocfg.get("max_output_tokens", 2000)),
        ))
    elif provider == "deepl":
        dcfg = cfg["translation"].get("deepl", {})
        translator = DeepLTranslator(
            formality=dcfg.get("formality", "more"),
            preserve_formatting=bool(dcfg.get("preserve_formatting", True)),
        )
    elif provider == "dummy":
        translator = DummyTranslator()
    else:
        raise ValueError(f"Unknown translation provider: {provider}")

    chcfg = cfg["translation"].get("chunking", {})
    translated = translate_segments(
        segments,
        translator=translator,
        glossary_map=glossary_map,
        max_chars_per_chunk=int(chcfg.get("max_chars_per_chunk", 9000)),
        max_segments_per_chunk=int(chcfg.get("max_segments_per_chunk", 8)),
    )

    logger.info("5) Post-processing + checks…")
    post = cfg.get("postproc", {})
    for seg in translated:
        tr_html = seg.get("html_translated", "")
        if post.get("apply_glossary", True) and glossary_map:
            tr_html = apply_glossary_to_fragment(tr_html, glossary_map)
            seg["html_translated"] = tr_html
            seg["text_translated"] = BeautifulSoup(f"<div>{tr_html}</div>", "html.parser").get_text(" ", strip=True)

        if post.get("verify_html_structure", True):
            src_wrapped = f'<{seg["tag"]}>{seg.get("html","")}</{seg["tag"]}>'
            tr_wrapped = f'<{seg["tag"]}>{seg.get("html_translated","")}</{seg["tag"]}>'
            ok, issues = compare_html_structure(src_wrapped, tr_wrapped)
            seg["check_structure_ok"] = ok
            seg["check_structure_issues"] = issues

        if post.get("verify_numbers_dates", True):
            warnings = numbers_and_citations_check(seg.get("text",""), seg.get("text_translated",""))
            seg["check_numbers_citations"] = warnings

    storage.write_json(paths["translations_json"], translated)

    logger.info("6) Rebuild final HTML…")
    translated_html = rebuild_html(html_text, translated, block_tags=cfg["segmentation"].get("block_tags"))
    exports_dir = Path(paths.get("exports_dir", "data/exports"))
    exports_dir.mkdir(parents=True, exist_ok=True)
    out_html = exports_dir / "translated_document.html"
    storage.write_text(out_html, translated_html)
    logger.info(f"   Exported: {out_html}")


if __name__ == "__main__":
    main()

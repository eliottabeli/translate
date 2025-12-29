\
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv

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
from src.qa import run_basic_checks_on_segments


st.set_page_config(page_title="Streamlit OCR Translator", layout="wide")
load_dotenv()


def load_config(path: str) -> Dict[str, Any]:
    return storage.read_json(path)


def main() -> None:
    st.title("streamlit-ocr-translator")
    st.caption("Pipeline : HTML + images → segments → OCR (optionnel) → glossaire → traduction → QA → export HTML")

    with st.sidebar:
        st.header("Configuration")
        config_path = st.text_input("Chemin config.json", value="config.json")
        if st.button("Charger config"):
            st.session_state["cfg"] = load_config(config_path)

        if "cfg" not in st.session_state:
            # fallback
            if Path("config.json").exists():
                st.session_state["cfg"] = load_config("config.json")
            else:
                st.session_state["cfg"] = load_config("config.example.json")

        cfg = st.session_state["cfg"]
        st.json(cfg, expanded=False)

    cfg = st.session_state["cfg"]
    paths = cfg["paths"]

    logger = setup_logger(paths.get("logs_dir", "logs"))

    cols = st.columns(3)
    with cols[0]:
        st.subheader("1) Segmentation")
        if st.button("Construire segments"):
            html_text = storage.read_text(paths["raw_html"])
            segs, meta = extract_segments(
                html_text,
                block_tags=cfg["segmentation"].get("block_tags"),
                include_empty=cfg["segmentation"].get("include_empty", False),
            )
            storage.write_json(paths["segments_json"], segs)
            storage.write_json(paths["segments_meta_json"], meta)
            st.success(f"{len(segs)} segments écrits → {paths['segments_json']}")

    with cols[1]:
        st.subheader("2) OCR images")
        ocr_enabled = cfg.get("ocr", {}).get("enabled", False)
        st.write(f"OCR activé: **{ocr_enabled}**")
        if st.button("Lancer OCR"):
            if not ocr_enabled:
                st.warning("Activez ocr.enabled=true dans config.json")
            else:
                ocr_cfg = cfg["ocr"]
                res = ocr_images_folder(
                    images_dir=paths["raw_images_dir"],
                    lang=ocr_cfg.get("lang", "eng"),
                    tesseract_cmd=ocr_cfg.get("tesseract_cmd", ""),
                    only_if_filename_matches=ocr_cfg.get("only_if_filename_matches", ""),
                )
                storage.write_json(paths["images_ocr_json"], res)
                st.success(f"OCR terminé ({len(res)} images) → {paths['images_ocr_json']}")

    with cols[2]:
        st.subheader("3) Glossaire")
        if st.button("Extraire candidats glossaire"):
            segs = storage.read_json(paths["segments_json"])
            gcfg = cfg.get("glossary", {})
            glossary_rows = extract_glossary_candidates(
                segs,
                top_n=gcfg.get("top_n", 150),
                min_len=gcfg.get("min_len", 4),
                include_from_tags=gcfg.get("include_from_tags", ["em", "i", "strong"]),
                stopwords_lang=gcfg.get("stopwords_lang", "english"),
            )
            storage.write_json(paths["glossary_json"], glossary_rows)
            st.success(f"Glossaire écrit → {paths['glossary_json']}")

    st.divider()

    st.subheader("4) Traduction + post-traitement")
    provider = cfg.get("translation", {}).get("provider", "openai").lower()
    st.write(f"Provider configuré: **{provider}**")

    colT1, colT2 = st.columns([1, 2])
    with colT1:
        if st.button("Traduire segments"):
            segs = storage.read_json(paths["segments_json"])
            glossary_rows = storage.read_json(paths["glossary_json"]) if Path(paths["glossary_json"]).exists() else []
            glossary_map = glossary_to_map(glossary_rows)

            # Instantiate translator
            if provider == "openai":
                ocfg = cfg["translation"].get("openai", {})
                tr = OpenAITranslator(cfg=OpenAIConfig(
                    model=ocfg.get("model", "gpt-4.1-mini"),
                    temperature=float(ocfg.get("temperature", 0.1)),
                    max_output_tokens=int(ocfg.get("max_output_tokens", 2000)),
                ))
            elif provider == "deepl":
                dcfg = cfg["translation"].get("deepl", {})
                tr = DeepLTranslator(
                    formality=dcfg.get("formality", "more"),
                    preserve_formatting=bool(dcfg.get("preserve_formatting", True)),
                )
            elif provider == "dummy":
                tr = DummyTranslator()
            else:
                st.error(f"Provider inconnu: {provider}")
                return

            chcfg = cfg["translation"].get("chunking", {})
            translated = translate_segments(
                segs,
                translator=tr,
                glossary_map=glossary_map,
                max_chars_per_chunk=int(chcfg.get("max_chars_per_chunk", 9000)),
                max_segments_per_chunk=int(chcfg.get("max_segments_per_chunk", 8)),
            )

            # Postproc
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
            st.success(f"Traductions écrites → {paths['translations_json']}")

    with colT2:
        st.write("Édition manuelle (post-édition)")
        if Path(paths["translations_json"]).exists():
            translated = storage.read_json(paths["translations_json"])
            df = pd.DataFrame(translated)
            # Keep a compact view
            view_cols = [c for c in ["id", "tag", "text", "text_translated", "html_translated"] if c in df.columns]
            edited = st.data_editor(df[view_cols], num_rows="dynamic", use_container_width=True, key="editor")
            if st.button("Sauvegarder modifications"):
                # Merge back edits into full rows
                edited_map = {row["id"]: row for row in edited.to_dict(orient="records")}
                for row in translated:
                    if row["id"] in edited_map:
                        row["text_translated"] = edited_map[row["id"]].get("text_translated", row.get("text_translated",""))
                        row["html_translated"] = edited_map[row["id"]].get("html_translated", row.get("html_translated",""))
                storage.write_json(paths["translations_json"], translated)
                st.success("Modifications sauvegardées.")
        else:
            st.info("Aucune traduction trouvée. Lancez la traduction d'abord.")

    st.divider()

    st.subheader("5) QA & export")

    colQ1, colQ2 = st.columns(2)
    with colQ1:
        if st.button("QA basique (structure + nombres + citations)"):
            if not Path(paths["translations_json"]).exists():
                st.warning("Aucune traduction à vérifier.")
            else:
                translated = storage.read_json(paths["translations_json"])
                qa = run_basic_checks_on_segments(translated)
                storage.write_json(paths["translations_json"], qa)
                n_bad = sum(1 for s in qa if not s.get("qa_ok_structure", True))
                st.success(f"QA terminée. Segments avec souci structure: {n_bad}")

    with colQ2:
        if st.button("Exporter HTML final"):
            if not Path(paths["translations_json"]).exists():
                st.warning("Aucune traduction à exporter.")
            else:
                html_text = storage.read_text(paths["raw_html"])
                translated = storage.read_json(paths["translations_json"])
                out_html = rebuild_html(html_text, translated, block_tags=cfg["segmentation"].get("block_tags"))
                exports_dir = Path(paths.get("exports_dir", "data/exports"))
                exports_dir.mkdir(parents=True, exist_ok=True)
                out_path = exports_dir / "translated_document.html"
                storage.write_text(out_path, out_html)
                st.success(f"HTML exporté → {out_path}")
                st.download_button("Télécharger translated_document.html", data=out_html.encode("utf-8"), file_name="translated_document.html", mime="text/html")


if __name__ == "__main__":
    main()

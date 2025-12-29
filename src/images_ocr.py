from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, Optional

from .utils import normalize_ocr_text


def ocr_images_folder(
    images_dir: str | Path,
    lang: str = "eng",
    tesseract_cmd: str = "",
    only_if_filename_matches: str = "",
) -> Dict[str, str]:
    """
    OCR all images in a folder. Returns a dict: {filename: extracted_text}.

    Requirements:
      - tesseract installed on the system
      - python deps: pillow, pytesseract
    """
    try:
        from PIL import Image
        import pytesseract
    except Exception as e:
        raise RuntimeError(
            "OCR dependencies missing. Please install pillow+pytesseract and ensure Tesseract is installed."
        ) from e

    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    if tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd

    pattern = re.compile(only_if_filename_matches) if only_if_filename_matches else None

    results: Dict[str, str] = {}
    for fn in sorted(os.listdir(images_dir)):
        if not fn.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp")):
            continue
        if pattern and not pattern.search(fn):
            continue

        path = images_dir / fn
        try:
            text = pytesseract.image_to_string(Image.open(path), lang=lang)
        except Exception as e:
            text = f"[OCR_ERROR] {e}"

        results[fn] = normalize_ocr_text(text)

    return results

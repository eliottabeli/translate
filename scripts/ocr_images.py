\
from __future__ import annotations

import argparse

from src import storage
from src.images_ocr import ocr_images_folder


def main() -> None:
    parser = argparse.ArgumentParser(description="OCR all images in a folder.")
    parser.add_argument("--images", required=True, help="Images folder")
    parser.add_argument("--out", required=True, help="Output JSON file")
    parser.add_argument("--lang", default="eng", help="Tesseract language code (default: eng)")
    parser.add_argument("--tesseract-cmd", default="", help="Path to tesseract binary if needed")
    parser.add_argument("--only-if-filename-matches", default="", help="Regex: OCR only matching filenames")
    args = parser.parse_args()

    results = ocr_images_folder(
        images_dir=args.images,
        lang=args.lang,
        tesseract_cmd=args.tesseract_cmd,
        only_if_filename_matches=args.only_if_filename_matches,
    )
    storage.write_json(args.out, results)
    print(f"Wrote OCR for {len(results)} images to {args.out}")


if __name__ == "__main__":
    main()

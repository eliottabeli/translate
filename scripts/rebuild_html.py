from __future__ import annotations

import argparse
from pathlib import Path

from src import storage
from src.html_parser import rebuild_html


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild HTML from translated segments.")
    parser.add_argument("--html", required=True, help="Path to source HTML")
    parser.add_argument("--segments", required=True, help="Path to translated segments JSON (segments_translated.json)")
    parser.add_argument("--out", required=True, help="Output translated HTML")
    args = parser.parse_args()

    html_path = Path(args.html)
    segs_path = Path(args.segments)

    if not html_path.exists():
        raise SystemExit(f"Source HTML not found: {html_path}")
    if not segs_path.exists():
        raise SystemExit(f"Segments file not found: {segs_path}")

    html_text = storage.read_text(html_path)
    segs = storage.read_json(segs_path)

    out_html = rebuild_html(html_text, segs)
    storage.write_text(args.out, out_html)

    print(f"Wrote translated HTML to {args.out}")


if __name__ == "__main__":
    main()

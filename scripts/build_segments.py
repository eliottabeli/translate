\
from __future__ import annotations

import argparse
from pathlib import Path

from src import storage
from src.html_parser import extract_segments


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract block segments from an HTML file.")
    parser.add_argument("--html", required=True, help="Path to source HTML")
    parser.add_argument("--out", required=True, help="Output segments.json")
    parser.add_argument("--meta-out", default="", help="Optional output segments_meta.json")
    args = parser.parse_args()

    html_text = storage.read_text(args.html)
    segments, meta = extract_segments(html_text)

    storage.write_json(args.out, segments)
    if args.meta_out:
        storage.write_json(args.meta_out, meta)

    print(f"Wrote {len(segments)} segments to {args.out}")


if __name__ == "__main__":
    main()

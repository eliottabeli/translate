from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def _ensure_exists(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def read_text(path: str | Path, encoding: str = "utf-8") -> str:
    p = _ensure_exists(Path(path))
    return p.read_text(encoding=encoding)


def write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding=encoding)


def read_json(path: str | Path) -> Any:
    p = _ensure_exists(Path(path))
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | Path, obj: Any, indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def write_segments_csv(path: str | Path, rows: List[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(p, index=False, encoding="utf-8")


def read_segments_csv(path: str | Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    return df.to_dict(orient="records")

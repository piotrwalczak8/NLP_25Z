# src/utils/config_utils.py
import json
from pathlib import Path
from typing import Dict, Any

def load_config(path: str = "config/config_IMDB.json") -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with p.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    return cfg

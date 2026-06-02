"""IO helpers: yaml/json/parquet read/write, hashing, path resolution."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

def project_root() -> Path:
    """Return the project root (the funding_arb directory)."""
    # src/utils/io.py -> src/utils -> src -> funding_arb
    return Path(__file__).resolve().parents[2]


def config_dir() -> Path:
    return project_root() / "config"


def data_dir() -> Path:
    return project_root() / "data"


def raw_dir() -> Path:
    p = data_dir() / "raw"
    p.mkdir(parents=True, exist_ok=True)
    return p


def normalized_dir() -> Path:
    p = data_dir() / "normalized"
    p.mkdir(parents=True, exist_ok=True)
    return p


def outputs_dir() -> Path:
    p = project_root() / "outputs"
    p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------------- #
# YAML / JSON
# --------------------------------------------------------------------------- #

def load_yaml(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _yaml_safe(obj: Any) -> Any:
    """Coerce non-yaml-native types to strings recursively."""
    import datetime as _dt
    if isinstance(obj, dict):
        return {str(k): _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_yaml_safe(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, (_dt.datetime, _dt.date, _dt.timedelta)):
        return str(obj)
    # pandas.Timestamp, numpy scalars, etc.
    return str(obj)


def dump_yaml(obj: Any, path: Path | str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(_yaml_safe(obj), f, sort_keys=False)


def load_json(path: Path | str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(obj: Any, path: Path | str, *, indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, default=str)


# --------------------------------------------------------------------------- #
# Parquet
# --------------------------------------------------------------------------- #

def write_parquet(df: pd.DataFrame, path: Path | str) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False)
    return p


def read_parquet(path: Path | str) -> pd.DataFrame:
    return pd.read_parquet(path)


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #

def hash_payload(obj: Any) -> str:
    """Stable SHA1 of an arbitrary JSON-serializable payload."""
    raw = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()

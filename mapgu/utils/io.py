"""Shared I/O helpers: CSV writing, NPZ saving, YAML config saving, and small numeric utilities."""

from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional

import numpy as np


# ── Numeric helpers ────────────────────────────────────────────────────────────

def _ms(xs) -> tuple:
    """Return (mean, std) of *xs*, or (nan, nan) when *xs* is empty."""
    xs = list(xs)
    return (float(np.mean(xs)), float(np.std(xs))) if xs else (float("nan"), float("nan"))


def _fmt_eps(eps: float) -> str:
    """Format an epsilon value for use in file/directory names."""
    return f"{float(eps):.12g}"


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ── NPZ persistence ────────────────────────────────────────────────────────────

def save_npz(path: str, X: np.ndarray, y: np.ndarray) -> None:
    """Save feature matrix and label vector as a compressed NPZ file."""
    _ensure_dir(os.path.dirname(path) or ".")
    np.savez_compressed(path, X=X.astype(np.float32), y=y.astype(np.int64))


# ── CSV helpers ────────────────────────────────────────────────────────────────

def save_metrics_csv(path: str, metrics: Dict[str, tuple], metadata: Optional[Dict[str, Any]] = None) -> None:
    """Write ``{metric_name: (mean, std)}`` to a CSV.

    When *metadata* is provided, each metadata key is written as an extra column
    and repeated on every metric row. This is useful for carrying experiment
    settings such as FT/Post epochs into per-phase result CSVs.
    """
    _ensure_dir(os.path.dirname(path) or ".")
    metadata = dict(metadata or {})
    fieldnames = list(metadata.keys()) + ["Metric", "Mean", "Std"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for k, (mean_val, std_val) in metrics.items():
            row = dict(metadata)
            row.update({"Metric": k, "Mean": mean_val, "Std": std_val})
            w.writerow(row)


def save_summary_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    """Write a list of summary-row dicts to a CSV.

    Field names are inferred from the first row, so all dicts must share the
    same key set (or a superset thereof).
    """
    if not rows:
        return
    _ensure_dir(os.path.dirname(path) or ".")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def append_runtime_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    """Append runtime rows to a cumulative runtimes CSV.

    Each row must share the same keys.  The header is written only when the file
    does not yet exist (or is empty), so multiple calls accumulate all results in
    one place.

    Expected keys per row::

        Method | Param | Forget Ratio | Phase | N Runs | Mean (s) | Std (s)

    Optional metadata columns are normalized automatically:

        FT Epochs | Post Epochs
    """
    if not rows:
        return
    _ensure_dir(os.path.dirname(path) or ".")
    fieldnames = [
        "Method",
        "Param",
        "Forget Ratio",
        "Phase",
        "N Runs",
        "Mean (s)",
        "Std (s)",
        "FT Epochs",
        "Post Epochs",
    ]

    def _normalize(row: Dict[str, Any]) -> Dict[str, Any]:
        return {key: row.get(key, "") for key in fieldnames}

    normalized_rows = [_normalize(row) for row in rows]
    existing_rows: List[Dict[str, Any]] = []
    rewrite_with_header = not os.path.isfile(path) or os.path.getsize(path) == 0

    if not rewrite_with_header:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = list(reader.fieldnames or [])
            existing_rows = [_normalize(row) for row in reader]
        if existing_fieldnames != fieldnames:
            rewrite_with_header = True

    mode = "w" if rewrite_with_header else "a"
    with open(path, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if rewrite_with_header:
            w.writeheader()
            if existing_rows:
                w.writerows(existing_rows)
        w.writerows(normalized_rows)


def save_config_yaml(path: str, config: Dict[str, Any]) -> None:
    """Write a flat-ish config dict to a YAML file (no PyYAML dependency).

    Supports values that are str, int, float, bool, None, or lists of those.
    Nested dicts are written as sub-mappings with a blank line separator.
    """
    _ensure_dir(os.path.dirname(path) or ".")

    def _scalar(v: Any) -> str:
        if v is None:
            return "null"
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            # Use repr to preserve full precision; avoid 'inf'/'nan' confusion
            s = repr(v)
            return s
        # strings: quote if empty or contains special YAML chars
        s = str(v)
        if not s or any(c in s for c in ":{}[]|>&*!,'\"#%@`\n\r"):
            return "'" + s.replace("'", "''") + "'"
        return s

    def _value(v: Any) -> str:
        if isinstance(v, (list, tuple)):
            items = ", ".join(_scalar(i) for i in v)
            return f"[{items}]"
        if isinstance(v, dict):
            return ""  # handled below as sub-mapping
        return _scalar(v)

    lines: List[str] = []
    for key, val in config.items():
        if isinstance(val, dict):
            lines.append(f"{key}:")
            for sk, sv in val.items():
                lines.append(f"  {sk}: {_value(sv)}")
        else:
            lines.append(f"{key}: {_value(val)}")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def log_metrics_table(logger, metrics: Dict[str, tuple], title: str) -> None:
    """Log a metrics dict as a formatted table via *logger*."""
    logger.info("=" * 60)
    logger.info("METRICS — %s", title.upper())
    logger.info("=" * 60)
    logger.info("%-24s %-12s %-12s", "Metric", "Mean", "Std")
    logger.info("-" * 60)
    for k, (mean_val, std_val) in metrics.items():
        logger.info("%-24s %-12.4f %-12.4f", k, mean_val, std_val)
    logger.info("=" * 60)

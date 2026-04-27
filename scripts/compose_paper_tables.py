"""MAPGU paper — LaTeX table composer.

Tables generated
----------------
  1   Dataset statistics                         (static)
  2   Method overview                            (static)
  3   Training hyperparameters                   (static)
  4   Utility (Bef/Aft/Δ) + MIA (AUC/TPR@1%) — merged, per model: mlp / xgboost / resnet18
  6a  Head-to-head unlearning efficiency
  6b  Phase-by-phase runtime breakdown
  7   FT-epoch sensitivity  (CIFAR-10 / ResNet-18 — MAPGU_ε + Certified-SP)
  8   Forget ratio sensitivity  (Adult / MLP)

Usage
-----
  # All tables → results/paper_tables/
  python scripts/compose_paper_tables.py

  # Single table to stdout
  python scripts/compose_paper_tables.py --table 4

  # Custom output directory
  python scripts/compose_paper_tables.py --out paper/tables/

  # Custom results directory
  python scripts/compose_paper_tables.py --results_dir /path/to/results
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))

# ---------------------------------------------------------------------------
# Configuration — edit here or override via CLI
# ---------------------------------------------------------------------------
FORGET_RATIO    = 0.05
FT_EPOCHS       = 5
NUM_SHARDS      = 5
SEED            = 7

KANON_K         = 30
DP_EPS          = 1.0
CERTIFIED_SP_EPS        = 1
CERTIFIED_SP_DELTA      = "1e-5"

# CIFAR-10 uses k=10 for k-anonymity (latent-feature clustering);
# all tabular datasets use k=30.
KANON_K_PER_DATASET: Dict[str, int] = {
    "adult": 30, "heart": 30, "credit": 30, "cifar10": 10,
}

DATASETS_TABULAR = ["adult", "heart", "credit"]
DATASETS_VISION  = ["cifar10"]
MODELS_TABULAR   = ["mlp", "xgboost"]
MODEL_VISION     = "resnet18"

DATASET_LABEL = {
    "adult":   "Adult",
    "heart":   "Heart",
    "credit":  "Credit",
    "cifar10": "CIFAR-10",
}
# True → metric is accuracy (%); False → metric is AUC (%)
DATASET_IS_ACC = {
    "adult": True, "heart": True, "credit": False, "cifar10": True,
}
METRIC_LABEL = {
    "adult": "Acc.", "heart": "Acc.", "credit": "AUC", "cifar10": "Acc.",
}

# Sensitivity study subdir for Table 8 (Adult / MLP — forget ratio)
FORGET_RATIO_SENSITIVITY_SUBDIR = "sensitivity_studies/adult_mlp_forget_ratio"
FORGET_RATIO_VALUES = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90]
# Sensitivity study subdirs for Table 7 (CIFAR-10 / ResNet-18 — FT epochs)
FT_EPOCHS_MAPGU_SUBDIR  = "sensitivity_studies/cifar10_resnet18_eps_ft_epochs_onecycle_lr5e-2"
FT_EPOCHS_CERTSP_SUBDIR = "sensitivity_studies/cifar10_resnet18_certified_sp_post_epochs_onecycle_lr5e-2"
FT_EPOCH_VALUES         = [3, 5, 10, 20, 30, 50]

MISSING = r"\text{---}"

# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _fmt_fr(fr: float) -> str:
    return str(float(fr))


def _fmt_eps(eps: float) -> str:
    return str(float(eps))


def _certified_sp_fr_str(fr: float) -> str:
    return f"{fr:.4f}".rstrip("0").rstrip(".")


def load_metrics(path: str) -> Dict[str, Tuple[float, float]]:
    """Read a Metric/Mean/Std CSV → {metric: (mean, std)}.  Returns {} if file missing."""
    if not os.path.isfile(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {
            row["Metric"]: (float(row["Mean"]), float(row["Std"]))
            for row in csv.DictReader(f)
            if row.get("Metric")
        }


def load_runtimes(dataset: str, model: str, results_dir: str) -> pd.DataFrame:
    path = os.path.join(results_dir, dataset, f"{model}_runtimes.csv")
    if not os.path.isfile(path):
        return pd.DataFrame(
            columns=["Method", "Param", "Forget Ratio", "Phase", "N Runs", "Mean (s)", "Std (s)"]
        )
    return pd.read_csv(path)


def get_rt(
    df: pd.DataFrame,
    method: str,
    param: str,
    fr: float,
    phases: List[str],
) -> Tuple[float, float]:
    """Sum mean runtimes for the listed phases.  Returns (total_mean, pooled_std).

    Some runtimes CSVs contain duplicate rows for the same (method, param, fr, phase)
    from multiple experiment runs appended together.  We keep only the *last* row per
    phase (most recent / canonical run) before summing across phases.
    """
    if df.empty:
        return math.nan, math.nan
    mask = (
        (df["Method"] == method)
        & (df["Param"] == param)
        & (df["Forget Ratio"].astype(float).round(6) == round(float(fr), 6))
        & (df["Phase"].isin(phases))
    )
    sub = df[mask]
    if sub.empty:
        return math.nan, math.nan
    # Deduplicate: keep the last occurrence of each phase (canonical run)
    sub = sub.groupby("Phase", as_index=False).last()
    return float(sub["Mean (s)"].sum()), float(np.sqrt((sub["Std (s)"] ** 2).sum()))


def find_certified_sp_csv(dataset: str, model: str, fr: float, seed: int, results_dir: str) -> Optional[str]:
    """Locate the CERTIFIED_SP summary CSV for a given dataset/model.

    Tries (in order):
      1. <model>_certified_sp_summary.csv          (tabular experiments)
      2. <model>_certified_sp_*_summary.csv        (vision experiments, e.g. post_epochs=5)
    """
    base = os.path.join(results_dir, dataset)
    # Preferred: canonical summary written by tabular CERTIFIED_SP runner
    canonical = os.path.join(base, f"{model}_certified_sp_summary.csv")
    if os.path.isfile(canonical):
        return canonical
    # Fallback: vision CERTIFIED_SP summary (e.g. resnet18_certified_sp_eps=1_post_epochs=5_summary.csv)
    matches = sorted(glob.glob(os.path.join(base, f"{model}_certified_sp_*_summary.csv")))
    return matches[-1] if matches else None


def load_certified_sp_post_metrics(certified_sp_csv: Optional[str], fr: float) -> Dict[str, Tuple[float, float]]:
    """Read the *post* phase rows from a CERTIFIED_SP summary CSV.

    CERTIFIED_SP summary files (both tabular and vision) have columns:
        Experiment, Phase, [Post Epochs,] Param, Forget Ratio, Metric, Mean, Std, ...
    Metric names carry unit suffixes like " (%)" or " (s)"; those are stripped before
    mapping to the standard keys used by the rest of this script.
    """
    if not certified_sp_csv or not os.path.isfile(certified_sp_csv):
        return {}

    out: Dict[str, Tuple[float, float]] = {}
    with open(certified_sp_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("Phase", "").strip() != "post":
                continue
            try:
                if abs(float(row.get("Forget Ratio", "nan")) - fr) > 1e-6:
                    continue
            except (ValueError, TypeError):
                continue
            raw_metric = row.get("Metric", "").strip()
            # Strip trailing unit annotation " (%)" or " (s)"
            metric = raw_metric.replace(" (%)", "").replace(" (s)", "").strip()
            try:
                mean_val = float(row.get("Mean", "nan"))
                std_val  = float(row.get("Std",  "nan"))
            except (ValueError, TypeError):
                continue
            if metric in ("Test Accuracy", "Test AUC"):
                out[metric] = (mean_val, std_val)
            elif metric == "MIA AUC":
                out["MIA AUC (RMIA)"] = (mean_val, std_val)
            elif metric == "MIA TPR@1%FPR":
                out["MIA TPR@1%FPR (RMIA)"] = (mean_val, std_val)
    return out


# Keep the old name as a thin shim used by Table 7 (forget-ratio sensitivity).
def load_certified_sp_metrics(certified_sp_csv: Optional[str]) -> Dict[str, Tuple[float, float]]:
    return load_certified_sp_post_metrics(certified_sp_csv, FORGET_RATIO)


def mia_auc(metrics: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    """Return MIA AUC, trying the (RMIA) suffixed key first."""
    for key in ("MIA AUC (RMIA)", "MIA AUC"):
        if key in metrics:
            return metrics[key]
    return math.nan, math.nan


def mia_tpr(metrics: Dict[str, Tuple[float, float]]) -> Tuple[float, float]:
    for key in ("MIA TPR@1%FPR (RMIA)", "MIA TPR@1%FPR"):
        if key in metrics:
            return metrics[key]
    return math.nan, math.nan


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _val(
    mean: float,
    std: float = math.nan,
    *,
    prec: int = 2,
    show_std: bool = True,
    show_sign: bool = False,
) -> str:
    """Plain-text cell value (no math-mode wrapper).  Used by runtime tables."""
    if math.isnan(mean):
        return MISSING
    prefix = "+" if (show_sign and mean > 0) else ""
    s = f"{prefix}{mean:.{prec}f}"
    if show_std and not math.isnan(std):
        s += rf" \scriptscriptstyle\pm {std:.{prec}f}"
    return s


def _mval(
    mean: float,
    std: float = math.nan,
    *,
    prec: int = 2,
    show_std: bool = True,
) -> str:
    """Math-mode cell value: ``$mean {\scriptscriptstyle\pm} std$``.

    Used by quality tables to match the formatting requested in the paper style.
    """
    if math.isnan(mean):
        return MISSING
    s = f"{mean:.{prec}f}"
    if show_std and not math.isnan(std):
        s = rf"{s} {{\scriptscriptstyle\pm}} {std:.{prec}f}"
    return f"${s}$"


def _bold(s: str) -> str:
    return rf"\textbf{{{s}}}"


def _uline(s: str) -> str:
    return rf"\underline{{{s}}}"


def _apply_best(
    vals: List[float],
    fmts: List[str],
    higher_is_better: bool,
) -> List[str]:
    """Bold the best, underline the second-best value in a column.

    Only non-NaN, non-MISSING entries are considered.
    """
    indexed = [
        (v, i) for i, v in enumerate(vals)
        if not math.isnan(v) and fmts[i] != MISSING
    ]
    if not indexed:
        return fmts
    indexed.sort(key=lambda x: x[0], reverse=higher_is_better)
    out = list(fmts)
    if len(indexed) >= 1:
        out[indexed[0][1]] = _bold(fmts[indexed[0][1]])
    if len(indexed) >= 2:
        out[indexed[1][1]] = _uline(fmts[indexed[1][1]])
    return out


def _row(*cols: str) -> str:
    return " & ".join(cols) + r" \\"


def _hline() -> str:
    return r"\hline"


def _mark_best(
    values: List[Optional[float]],
    formatted: List[str],
    higher_is_better: bool,
) -> List[str]:
    """Bold best, underline second-best (NaN ignored)."""
    valid = [(v, i) for i, v in enumerate(values) if v is not None and not math.isnan(v)]
    if not valid:
        return formatted
    valid.sort(key=lambda x: x[0], reverse=higher_is_better)
    result = list(formatted)
    if len(valid) >= 1:
        result[valid[0][1]] = _bold(formatted[valid[0][1]])
    if len(valid) >= 2:
        result[valid[1][1]] = _uline(formatted[valid[1][1]])
    return result


def _latex_table(
    body_lines: List[str],
    caption: str,
    label: str,
    colspec: str,
    wide: bool = False,
    resizebox: bool = False,
    tabcolsep: Optional[str] = None,
) -> str:
    env = "table*" if wide else "table"
    lines = [
        rf"\begin{{{env}}}[htbp]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
    ]
    if tabcolsep is not None:
        lines.append(rf"\setlength{{\tabcolsep}}{{{tabcolsep}}}")
    if resizebox:
        lines.append(r"\resizebox{\textwidth}{!}{%")
    lines += [
        rf"\begin{{tabular}}{{{colspec}}}",
        r"\toprule",
    ]
    lines += body_lines
    lines += [r"\bottomrule", r"\end{tabular}"]
    if resizebox:
        lines.append(r"}%")
    lines.append(rf"\end{{{env}}}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Table 1 — Dataset statistics  (static)
# ---------------------------------------------------------------------------

def build_table_1() -> str:
    body = [
        r"Dataset & Domain & Train & Test & Features & Classes & Forget (5\%) \\",
        r"\midrule",
        r"Adult Income  & Tabular & 32{,}561 & 16{,}281 & 108$^\dagger$ & 2 & 1{,}628 \\",
        r"Heart Disease & Tabular & 56{,}000 &  14{,}000 & 11 & 2 & 2{,}800 \\",
        r"Credit        & Tabular & 112{,}000 & 28{,}000 & 10 & 2 & 5{,}600 \\",
        r"CIFAR-10      & Vision  & 50{,}000 & 10{,}000 & $3{\times}32{\times}32$ & 10 & 2{,}500 \\",
        r"\multicolumn{7}{l}{\scriptsize $^\dagger$ After one-hot encoding of categorical features.}",
    ]
    return _latex_table(
        body,
        caption="Dataset statistics used in all experiments.",
        label="tab:datasets",
        colspec=r"l l r r r r r",
    )


# ---------------------------------------------------------------------------
# Table 2 — Method overview  (static)
# ---------------------------------------------------------------------------

def build_table_2() -> str:
    body = [
        r"Method & Full name & Core idea & Unlearning cost \\",
        r"\midrule",
        r"Retrain & Full retrain on $\mathcal{D}_r$"
        r" & Exact gold standard; retrain from scratch on retain set"
        r" & Full training on $|\mathcal{D}_r|$ \\",
        r"\addlinespace",
        r"SISA & Sharded Isolated Sliced Aggregated"
        r" & Trains on disjoint shards with checkpoints; only affected shards are retrained"
        r" & Retrain 1 shard ($\approx 1/S$ of data) \\",
        r"\addlinespace",
        r"MAPGU$_k$ (Ours) & $k$-Anonymity unlearning"
        r" & Trains on $k$-anonymous data; forgets by fine-tuning on $\mathcal{D}_r$"
        r" & Fine-tune on $\mathcal{D}_r$ \\",
        r"\addlinespace",
        r"MAPGU$_\varepsilon$ (Ours) & DP unlearning"
        r" & Trains on differentially private synthetic data; forgets by fine-tuning"
        r" & Fine-tune on $\mathcal{D}_r$ \\",
        r"\addlinespace",
        r"CERTIFIED_SP & Privacy-Aware Bayesian Inference"
        r" & Injects calibrated Gaussian noise during an unlearning pass"
        r" & Noisy gradient descent $+$ post fine-tune \\",
    ]
    return _latex_table(
        body,
        caption="Overview of unlearning methods compared in this work.",
        label="tab:methods",
        colspec=r"l l p{3.5cm} p{3cm}",
        wide=True,
    )


# ---------------------------------------------------------------------------
# Table 3 — Training hyperparameters  (static)
# ---------------------------------------------------------------------------

def build_table_3() -> str:
    body = [
        r"Dataset & Model & Optimizer & LR & Scheduler & Epochs & Batch & WD \\",
        r"\midrule",
        r"Adult  & MLP     & Adam & $10^{-2}$ & Cosine & 50  & 256 & $10^{-4}$ \\",
        r"Heart  & MLP     & Adam & $10^{-2}$ & Cosine & 50  & 256 & $10^{-5}$ \\",
        r"Credit & MLP     & Adam & $10^{-2}$ & Cosine & 50  & 256 & $10^{-5}$ \\",
        r"\midrule",
        r"Adult  & XGBoost & ---  & 0.5       & ---    & 100 trees & --- & $\lambda{=}5$ \\",
        r"Heart  & XGBoost & ---  & 0.5       & ---    & 100 trees & --- & $\lambda{=}5$ \\",
        r"Credit & XGBoost & ---  & 0.5       & ---    & 200 trees & --- & $\lambda{=}5$ \\",
        r"\midrule",
        r"CIFAR-10 & ResNet-18 & SGD & $10^{-1}$ & Cosine & 100 & 256 & $5{\times}10^{-4}$ \\",
    ]
    return _latex_table(
        body,
        caption=(
            r"Training hyperparameters. WD = weight decay. "
            r"Fine-tuning steps use the same optimizer/scheduler with $\texttt{ft\_epochs}=5$."
        ),
        label="tab:hyperparams",
        colspec=r"l l l r l r r r",
        wide=True,
    )


def _get_method_rows(
    dataset: str,
    model: str,
    results_dir: str,
    fr: float = FORGET_RATIO,
    ft: int   = FT_EPOCHS,
    S: int    = NUM_SHARDS,
    k: int    = KANON_K,
    eps: float = DP_EPS,
    seed: int  = SEED,
) -> List[Tuple[str, Dict, Dict]]:
    """
    Returns (display_label, before_metrics, after_metrics) for each method on
    (dataset, model).
    """
    fr_s  = _fmt_fr(fr)
    eps_s = _fmt_eps(eps)
    d     = results_dir

    def _p(fname: str) -> str:
        return os.path.join(d, dataset, fname)

    rows: List[Tuple[str, Dict, Dict]] = []

    # ── Retrain (oracle) ──────────────────────────────────────────────────
    before = load_metrics(_p(f"{model}_m_d_fr={fr_s}.csv"))
    after  = load_metrics(_p(f"{model}_mret_dret_fr={fr_s}.csv"))
    rows.append(("Retrain", before, after))

    # ── SISA ─────────────────────────────────────────────────────────────
    before = load_metrics(_p(f"{model}_sisa_m_d_shards={S}_fr={fr_s}.csv"))
    after  = load_metrics(_p(f"{model}_sisa_mret_dret_shards={S}_fr={fr_s}.csv"))
    rows.append((rf"SISA ($S$={S})", before, after))

    # ── MAPGU_k (k-anonymity) ──────────────────────────────────────────────
    k_ds = KANON_K_PER_DATASET.get(dataset, k)
    before = load_metrics(_p(f"{model}_mk={k_ds}_d_fr={fr_s}_epochs={ft}.csv"))
    after  = load_metrics(_p(f"{model}_mk={k_ds}_dret_fr={fr_s}_epochs={ft}.csv"))
    rows.append((rf"MAPGU ($k$={k_ds})", before, after))

    # ── MAPGU_eps (DP) ─────────────────────────────────────────────────────
    before = load_metrics(_p(f"{model}_mdpd_eps={eps_s}_fr={fr_s}_epochs={ft}.csv"))
    after  = load_metrics(_p(f"{model}_mdpret_eps={eps_s}_fr={fr_s}_epochs={ft}.csv"))
    eps_disp = int(eps) if eps == int(eps) else eps
    rows.append((rf"MAPGU ($\varepsilon$={eps_disp})", before, after))

    # ── Certified-SP (CERTIFIED_SP) ───────────────────────────────────────────────
    if model != "xgboost":
        certified_sp_csv = find_certified_sp_csv(dataset, model, fr, seed, d)
        before   = load_metrics(_p(f"{model}_m_d_fr={fr_s}.csv"))  # same init as retrain
        after    = load_certified_sp_post_metrics(certified_sp_csv, fr)
        rows.append((
            rf"Certified-SP ($\varepsilon$={CERTIFIED_SP_EPS}, $\delta$={CERTIFIED_SP_DELTA})",
            before, after,
        ))
    else:
        rows.append(("Certified-SP", {}, {}))

    return rows


# ---------------------------------------------------------------------------
# Table 4a — Quality: tabular models (MLP + XGBoost combined)
# ---------------------------------------------------------------------------
# Layout: Dataset × Method rows;  MLP | XGBoost column groups (6 metrics each).
# 3-level header; bold best / underline second-best per metric × dataset × model.
# Requires: \usepackage{multirow,booktabs} in the preamble.
# Higher-is-better: Utility Bef./Aft. ↑   Lower-is-better: MIA AUC, TPR@1% ↓

# Metric slots order: (bef_util, aft_util, bef_mia, aft_mia, bef_tpr, aft_tpr)
_METRIC_HIGHER = [True, True, False, False, False, False]


def _collect_ds_model(
    ds: str,
    model: str,
    results_dir: str,
    fr: float, ft: int, S: int, k: int, eps: float, seed: int,
) -> List[tuple]:
    """Return [(label, bef_util, aft_util, bef_mia, aft_mia, bef_tpr, aft_tpr), ...]
    where each metric field is a (mean, std) tuple."""
    method_rows = _get_method_rows(ds, model, results_dir, fr, ft, S, k, eps, seed)
    entries = []
    for label, before, after in method_rows:
        if label == "Certified-SP" and model == "xgboost":
            continue
        if ds == "credit":
            bef_util = before.get("Test AUC", before.get("Test Accuracy", (math.nan, math.nan)))
            aft_util = after.get("Test AUC", after.get("Test Accuracy", (math.nan, math.nan)))
        else:
            bef_util = before.get("Test Accuracy", (math.nan, math.nan))
            aft_util = after.get("Test Accuracy", (math.nan, math.nan))
        entries.append((
            label,
            bef_util,           aft_util,
            mia_auc(before),    mia_auc(after),
            mia_tpr(before),    mia_tpr(after),
        ))
    return entries


def _format_and_mark(entries: List[tuple]) -> List[List[str]]:
    """Format all cells then apply bold/underline per metric column."""
    N = 6
    # fmts[method_idx][metric_idx]
    fmts = [
        [_mval(entries[mi][1 + si][0], entries[mi][1 + si][1]) for si in range(N)]
        for mi in range(len(entries))
    ]
    for si in range(N):
        vals    = [entries[mi][1 + si][0] for mi in range(len(entries))]
        col_fmt = [fmts[mi][si]           for mi in range(len(entries))]
        marked  = _apply_best(vals, col_fmt, _METRIC_HIGHER[si])
        for mi in range(len(entries)):
            fmts[mi][si] = marked[mi]
    return fmts


def build_table_quality_tabular(
    results_dir: str,
    datasets: Optional[List[str]] = None,
    fr: float  = FORGET_RATIO,
    ft: int    = FT_EPOCHS,
    S: int     = NUM_SHARDS,
    k: int     = KANON_K,
    eps: float = DP_EPS,
    seed: int  = SEED,
) -> str:
    if datasets is None:
        datasets = DATASETS_TABULAR

    MODELS       = ["mlp", "xgboost"]
    MODEL_LABELS = {"mlp": "MLP", "xgboost": "XGBoost"}
    N_MET        = 6   # metrics per model group

    # ── collect ────────────────────────────────────────────────────────────
    data: Dict[str, Dict[str, List[tuple]]] = {ds: {} for ds in datasets}
    method_labels: List[str] = []
    for ds in datasets:
        for model in MODELS:
            entries = _collect_ds_model(ds, model, results_dir, fr, ft, S, k, eps, seed)
            data[ds][model] = entries
            if not method_labels and model == "mlp":
                method_labels = [e[0] for e in entries]
    n_meth = len(method_labels)  # rows per dataset block (MLP method count)

    # ── colspec ────────────────────────────────────────────────────────────
    # Dataset | Method | 6×MLP | 6×XGBoost
    colspec = "l l" + (" r" * N_MET) * len(MODELS)

    # ── 3-level header ─────────────────────────────────────────────────────
    # Level 1: model group spans
    model_spans = " & ".join(
        rf"\multicolumn{{{N_MET}}}{{c}}{{{MODEL_LABELS[m]}}}" for m in MODELS
    )
    cmidrule1 = "".join(
        rf"\cmidrule(lr){{{3 + i * N_MET}-{2 + (i + 1) * N_MET}}}"
        for i in range(len(MODELS))
    )
    h1 = (_row(r"\multirow{3}{*}{Dataset}", r"\multirow{3}{*}{Method}", model_spans)
          + "\n" + cmidrule1)

    # Level 2: Utility / MIA AUC / TPR sub-groups
    grp_span = (r"\multicolumn{2}{c}{Utility\,$\uparrow$} & "
                r"\multicolumn{2}{c}{MIA~AUC\,$\downarrow$} & "
                r"\multicolumn{2}{c}{TPR@1\%\,$\downarrow$}")
    grp_all = " & ".join([grp_span] * len(MODELS))
    cmidrule2 = "".join(
        rf"\cmidrule(lr){{{3 + j * 2 + i * N_MET}-{4 + j * 2 + i * N_MET}}}"
        for i in range(len(MODELS))
        for j in range(3)
    )
    h2 = _row("", "", grp_all) + "\n" + cmidrule2

    # Level 3: Bef./Aft. for each pair
    pair_row  = "Bef. & Aft. & Bef. & Aft. & Bef. & Aft."
    pair_all  = " & ".join([pair_row] * len(MODELS))
    h3 = _row("", "", pair_all)

    header_lines = [h1, h2, h3, r"\midrule"]

    # ── data rows ──────────────────────────────────────────────────────────
    data_lines: List[str] = []
    for di, ds in enumerate(datasets):
        fmt_by_model = {m: _format_and_mark(data[ds][m]) for m in MODELS}
        n_xgb = len(fmt_by_model["xgboost"])

        for mi, label in enumerate(method_labels):
            ds_cell = (rf"\multirow{{{n_meth}}}{{*}}{{{DATASET_LABEL[ds]}}}"
                       if mi == 0 else "")
            cells = [ds_cell, label]
            for model in MODELS:
                fmts = fmt_by_model[model]
                if mi < len(fmts):
                    cells.extend(fmts[mi])
                else:
                    cells.extend([MISSING] * N_MET)
            data_lines.append(_row(*cells))

        if di < len(datasets) - 1:
            data_lines.append(r"\midrule")

    # ── caption ────────────────────────────────────────────────────────────
    caption = (
        rf"Utility and MIA resistance before/after unlearning "
        rf"(tabular models, forget ratio~$= {int(fr * 100)}\%$, "
        r"mean~$\pm$~std over 5~runs).  "
        r"Bef./Aft.~= before/after unlearning.  "
        r"Utility is accuracy~(\%) for Adult and Heart, AUC~(\%) for Credit.  "
        r"MIA~AUC and TPR@1\%~FPR from the RMIA offline attack; "
        r"MIA~AUC~$\approx 50\%$ after unlearning indicates successful forgetting.  "
        r"\textbf{Bold} = best, \underline{underline} = second best "
        r"per metric per dataset per model."
    )
    return _latex_table(
        header_lines + data_lines,
        caption=caption,
        label="tab:quality_tabular",
        colspec=colspec,
        wide=True,
        resizebox=True,
    )


# ---------------------------------------------------------------------------
# Table 4b — Quality: vision model (ResNet-18, CIFAR-10)
# ---------------------------------------------------------------------------
# Layout: Method rows; Bef./Aft. column pairs for Utility, MIA AUC, TPR@1%.
# 2-level header; bold best / underline second-best per metric.

def build_table_quality(
    model: str,
    results_dir: str,
    datasets: Optional[List[str]] = None,
    fr: float  = FORGET_RATIO,
    ft: int    = FT_EPOCHS,
    S: int     = NUM_SHARDS,
    k: int     = KANON_K,
    eps: float = DP_EPS,
    seed: int  = SEED,
) -> str:
    """Single-model quality table (used for ResNet-18 / CIFAR-10).

    Layout: Dataset × Method rows; Bef./Aft. column pairs for Utility,
    MIA AUC, TPR@1%.  2-level header; bold best / underline second-best.
    """
    if datasets is None:
        datasets = DATASETS_VISION

    # ── collect ──────────────────────────────────────────────────────────
    data_by_ds: Dict[str, List[tuple]] = {}
    method_labels: List[str] = []
    for ds in datasets:
        entries = _collect_ds_model(ds, model, results_dir, fr, ft, S, k, eps, seed)
        data_by_ds[ds] = entries
        if not method_labels:
            method_labels = [e[0] for e in entries]

    n_meth = len(method_labels)

    # ── colspec: Dataset | Method | Bef.U Aft.U  Bef.M Aft.M  Bef.T Aft.T ──
    colspec = "l l" + " r" * 6

    # ── 2-level header ────────────────────────────────────────────────────
    grp_span = (r"\multicolumn{2}{c}{Utility\,$\uparrow$} & "
                r"\multicolumn{2}{c}{MIA~AUC\,$\downarrow$} & "
                r"\multicolumn{2}{c}{TPR@1\%\,$\downarrow$}")
    cmidrule = r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}"
    h1 = (_row(r"\multirow{2}{*}{Dataset}", r"\multirow{2}{*}{Method}", grp_span)
          + "\n" + cmidrule)
    h2 = _row("", "", "Bef. & Aft. & Bef. & Aft. & Bef. & Aft.")
    header_lines = [h1, h2, r"\midrule"]

    # ── data rows ─────────────────────────────────────────────────────────
    data_lines: List[str] = []
    for di, ds in enumerate(datasets):
        entries = data_by_ds[ds]
        fmts    = _format_and_mark(entries)   # applies bold/underline per metric

        for mi, label in enumerate(method_labels):
            ds_cell = (rf"\multirow{{{n_meth}}}{{*}}{{{DATASET_LABEL[ds]}}}"
                       if mi == 0 else "")
            cells = [ds_cell, label] + fmts[mi]
            data_lines.append(_row(*cells))

        if di < len(datasets) - 1:
            data_lines.append(r"\midrule")

    # ── caption ──────────────────────────────────────────────────────────
    model_label = {"mlp": "MLP", "xgboost": "XGBoost", "resnet18": "ResNet-18"}.get(model, model)
    util_note = (r"Utility is accuracy~(\%) for Adult and Heart, AUC~(\%) for Credit.  "
                 if "credit" in (datasets or []) else "")
    caption = (
        rf"Utility and MIA resistance before/after unlearning ({model_label}, "
        rf"forget ratio~$= {int(fr * 100)}\%$, mean~$\pm$~std over 5~runs).  "
        r"Bef./Aft.~= before/after unlearning.  "
        + util_note +
        r"MIA~AUC and TPR@1\%~FPR from the RMIA offline attack; "
        r"MIA~AUC~$\approx 50\%$ after unlearning indicates successful forgetting.  "
        r"\textbf{Bold} = best, \underline{underline} = second best per metric."
    )
    return _latex_table(
        header_lines + data_lines,
        caption=caption,
        label=f"tab:quality_{model}",
        colspec=colspec,
        wide=False,
        resizebox=False,
    )


# ---------------------------------------------------------------------------
# Table 6a — Head-to-head unlearning efficiency
# Two model-family groups: Neural Network (MLP tabular + ResNet-18 CIFAR-10)
#                          XGBoost (tabular only).
# Each group has its own Avg. speedup column.
# ---------------------------------------------------------------------------

def build_table_efficiency_headtohead(
    results_dir: str,
    fr: float  = FORGET_RATIO,
    S: int     = NUM_SHARDS,
    k: int     = KANON_K,
    eps: float = DP_EPS,
) -> str:
    nn_datasets  = DATASETS_TABULAR + DATASETS_VISION   # Adult MLP, Heart MLP, Credit MLP, CIFAR-10
    xgb_datasets = DATASETS_TABULAR                     # Adult XGB, Heart XGB, Credit XGB
    nn_model_map  = {"adult": "mlp",     "heart": "mlp",     "credit": "mlp",     "cifar10": "resnet18"}
    xgb_model_map = {"adult": "xgboost", "heart": "xgboost", "credit": "xgboost"}

    # colspec: Method | 4×NN | NN_spdup | 3×XGB | XGB_spdup
    colspec = "l rrrr r rrrr"
    eps_disp = int(eps) if eps == int(eps) else eps

    # ── shared helpers ────────────────────────────────────────────────────
    def _retrain_time_from_summary(ds: str, model: str) -> float:
        csv_path = os.path.join(results_dir, ds, f"{model}_baseline_retrain_summary.csv")
        if not os.path.isfile(csv_path):
            return math.nan
        try:
            df = pd.read_csv(csv_path)
            row = df[
                (df.get("Experiment", pd.Series(dtype=str)).str.lower() == "retrain") &
                (df["Metric"].str.contains("Retraining Time", case=False, na=False))
            ]
            if row.empty:
                row = df[df["Metric"].str.contains("Retraining Time", case=False, na=False)]
            if not row.empty:
                return float(row.iloc[0]["Mean"])
        except Exception:
            pass
        return math.nan

    def _retrain_std_from_summary(ds: str, model: str) -> float:
        csv_path = os.path.join(results_dir, ds, f"{model}_baseline_retrain_summary.csv")
        if not os.path.isfile(csv_path):
            return math.nan
        try:
            df = pd.read_csv(csv_path)
            row = df[
                (df.get("Experiment", pd.Series(dtype=str)).str.lower() == "retrain") &
                (df["Metric"].str.contains("Retraining Time", case=False, na=False))
            ]
            if row.empty:
                row = df[df["Metric"].str.contains("Retraining Time", case=False, na=False)]
            if not row.empty and "Std" in row.columns:
                return float(row.iloc[0]["Std"])
        except Exception:
            pass
        return math.nan

    def _sisa_time_from_summary(ds: str, model: str, S_val: int) -> Tuple[float, float]:
        """Return (mean, std) for SISA shard retrain from {model}_sisa_shards={S}_summary.csv."""
        for fname in [f"{model}_sisa_retrain_summary.csv",
                      f"{model}_sisa_shards={S_val}_summary.csv"]:
            csv_path = os.path.join(results_dir, ds, fname)
            if not os.path.isfile(csv_path):
                continue
            try:
                df = pd.read_csv(csv_path)
                row = df[df["Metric"].str.contains("Retraining Time", case=False, na=False)]
                if not row.empty:
                    mean_val = float(row.iloc[0]["Mean"])
                    std_val  = float(row.iloc[0]["Std"]) if "Std" in row.columns else math.nan
                    return mean_val, std_val
            except Exception:
                pass
        return math.nan, math.nan

    def _get_time(ds: str, model: str, method: str, param: str,
                  phases: List[str]) -> Tuple[float, float]:
        rt = load_runtimes(ds, model, results_dir)
        t_m, t_s = get_rt(rt, method, param, fr, phases)
        if math.isnan(t_m) and method == "baseline":
            t_m = _retrain_time_from_summary(ds, model)
            t_s = _retrain_std_from_summary(ds, model)
        if math.isnan(t_m) and method == "sisa":
            t_m, t_s = _sisa_time_from_summary(ds, model, S)
        # Fallback for certified_sp: vision runtimes use compound param "eps=X|post_epochs=N"
        if math.isnan(t_m) and method == "certified_sp" and not rt.empty:
            fr_r = round(float(fr), 6)
            sub = rt[
                (rt["Method"] == "certified_sp")
                & (rt["Param"].str.startswith(f"eps={CERTIFIED_SP_EPS}"))
                & (rt["Forget Ratio"].astype(float).round(6) == fr_r)
                & (rt["Phase"].isin(phases))
            ]
            if not sub.empty:
                sub = sub.groupby("Phase", as_index=False).last()
                t_m = float(sub["Mean (s)"].sum())
                t_s = float(np.sqrt((sub["Std (s)"] ** 2).sum()))
        return t_m, t_s

    # ── retrain times for speedup denominators ────────────────────────────
    retrain_nn:  Dict[str, float] = {}
    retrain_xgb: Dict[str, float] = {}
    for ds in nn_datasets:
        t, _ = _get_time(ds, nn_model_map[ds], "baseline", "-", ["retrain_Dr"])
        retrain_nn[ds] = t
    for ds in xgb_datasets:
        t, _ = _get_time(ds, xgb_model_map[ds], "baseline", "-", ["retrain_Dr"])
        retrain_xgb[ds] = t

    # ── method specs (label, method_key, param, phases) ──────────────────
    def _specs_for(ds: str, model_map: Dict[str, str]) -> List[tuple]:
        k_ds = KANON_K_PER_DATASET.get(ds, k)
        return [
            ("Retrain",
             "baseline", "-", ["retrain_Dr"]),
            (rf"SISA ($S$={S})",
             "sisa", f"S={S}", ["retrain_affected_shard_Dr"]),
            (rf"MAPGU ($k$={k_ds})",
             "kanon", f"k={k_ds}", ["ft_Mk_Dr"]),
            (rf"MAPGU ($\varepsilon$={eps_disp})",
             "dp", f"eps={_fmt_eps(eps)}", ["ft_Meps_Dr"]),
            (rf"Certified-SP ($\varepsilon$={CERTIFIED_SP_EPS}, $\delta$={CERTIFIED_SP_DELTA})",
             "certified_sp", f"eps={CERTIFIED_SP_EPS}", ["unlearn_noisy", "post_ft"]),
        ]

    method_labels = [s[0] for s in _specs_for(nn_datasets[0], nn_model_map)]
    n_meth = len(method_labels)

    # ── header ────────────────────────────────────────────────────────────
    nn_ds_hdr  = " & ".join(DATASET_LABEL[ds] for ds in nn_datasets)
    xgb_ds_hdr = " & ".join(DATASET_LABEL[ds] for ds in xgb_datasets)
    # Row 1: group spans (cols 2-6 = NN, cols 7-10 = XGB)
    h1 = (_row("",
               rf"\multicolumn{{5}}{{c}}{{Neural Network}}",
               rf"\multicolumn{{4}}{{c}}{{XGBoost}}")
          + "\n"
          + r"\cmidrule(lr){2-6}\cmidrule(lr){7-10}")
    # Row 2: column names
    h2 = _row("Method",
              nn_ds_hdr, "Avg.~speedup",
              xgb_ds_hdr, "Avg.~speedup")
    header_lines = [h1, h2, r"\midrule"]

    # ── data rows ─────────────────────────────────────────────────────────
    data_lines: List[str] = []
    for mi, label in enumerate(method_labels):
        nn_cells:  List[str] = []
        xgb_cells: List[str] = []
        nn_speedups:  List[float] = []
        xgb_speedups: List[float] = []

        # NN columns
        for ds in nn_datasets:
            model = nn_model_map[ds]
            _, method, param, phases = _specs_for(ds, nn_model_map)[mi]
            t_m, t_s = _get_time(ds, model, method, param, phases)
            nn_cells.append(_mval(t_m, t_s))
            if not math.isnan(t_m) and not math.isnan(retrain_nn[ds]) and t_m > 0:
                nn_speedups.append(retrain_nn[ds] / t_m)

        # XGB columns (Certified-SP not applicable; sub-second → 3 decimal places)
        is_certsp = (mi == n_meth - 1)
        for ds in xgb_datasets:
            if is_certsp:
                xgb_cells.append(MISSING)
            else:
                model = xgb_model_map[ds]
                _, method, param, phases = _specs_for(ds, xgb_model_map)[mi]
                t_m, t_s = _get_time(ds, model, method, param, phases)
                xgb_cells.append(_mval(t_m, t_s, prec=3))
                if not math.isnan(t_m) and not math.isnan(retrain_xgb[ds]) and t_m > 0:
                    xgb_speedups.append(retrain_xgb[ds] / t_m)

        nn_sp  = float(np.mean(nn_speedups))  if nn_speedups  else math.nan
        xgb_sp = float(np.mean(xgb_speedups)) if xgb_speedups else math.nan
        nn_sp_cell  = f"${nn_sp:.2f}\\times$"  if not math.isnan(nn_sp)  else MISSING
        xgb_sp_cell = f"${xgb_sp:.2f}\\times$" if not math.isnan(xgb_sp) else MISSING

        row_cells = [label] + nn_cells + [nn_sp_cell] + xgb_cells + [xgb_sp_cell]
        data_lines.append(_row(*row_cells))
        data_lines.append(r"\addlinespace")

    caption = (
        rf"Head-to-head unlearning wall time (seconds, mean~$\pm$~std over 5~runs, "
        rf"forget ratio~$= {int(fr * 100)}\%$).  "
        r"Speedup is relative to Retrain within each model family.  "
        r"Certified-SP time covers the noisy unlearning pass and post fine-tuning only "
        r"(baseline training excluded).  "
        r"XGBoost unlearning times are sub-second."
    )
    return _latex_table(
        header_lines + data_lines,
        caption=caption,
        label="tab:efficiency_headtohead",
        colspec=colspec,
        wide=True,
        tabcolsep="4pt",
    )


# ---------------------------------------------------------------------------
# Table 6b — Phase-by-phase runtime breakdown
# ---------------------------------------------------------------------------

def build_table_efficiency_phases(
    results_dir: str,
    fr: float  = FORGET_RATIO,
    k: int     = KANON_K,
    eps: float = DP_EPS,
) -> str:
    # 7-column layout (fixed throughout all sections):
    #   Dataset | c1 | c2 | c3 | c4 | c5 | Total
    # k-anon : Prep | Train Mk | FT D | FT Dr | [—]  | Total
    # DP     : Embed | DP synth | Train Mε | FT D | FT Dr | Total
    # Cert-SP: Train base | Noisy unl | Post FT | [—] | [—] | Total
    N_COLS = 7
    all_datasets = DATASETS_TABULAR + DATASETS_VISION
    all_model_map = {
        "adult": "mlp", "heart": "mlp", "credit": "mlp", "cifar10": "resnet18",
    }
    eps_disp = int(eps) if eps == int(eps) else eps
    eps_s    = _fmt_eps(eps)

    def _certified_sp_phase_rt(rt: pd.DataFrame, phase: str) -> Tuple[float, float]:
        """get_rt for certified_sp, falling back to compound-param match for vision."""
        m, s = get_rt(rt, "certified_sp", f"eps={CERTIFIED_SP_EPS}", fr, [phase])
        if math.isnan(m) and not rt.empty:
            fr_r = round(float(fr), 6)
            sub = rt[
                (rt["Method"] == "certified_sp")
                & (rt["Param"].str.startswith(f"eps={CERTIFIED_SP_EPS}"))
                & (rt["Forget Ratio"].astype(float).round(6) == fr_r)
                & (rt["Phase"] == phase)
            ]
            if not sub.empty:
                row = sub.iloc[-1]
                m, s = float(row["Mean (s)"]), float(row["Std (s)"])
        return m, s

    def _section_header(title: str) -> List[str]:
        """Blank line + full-width italic title + thin rule."""
        return [
            r"\addlinespace[2pt]",
            rf"\multicolumn{{{N_COLS}}}{{l}}{{\textit{{{title}}}}} \\",
            r"\midrule",
        ]

    body_lines: List[str] = []

    # ── MAPGU (k-anon) ─────────────────────────────────────────────────────
    body_lines += _section_header(
        rf"MAPGU ($k$): k-anon prep $\to$ train $M_k$ $\to$ "
        rf"fine-tune on $\mathcal{{D}}$ $\to$ fine-tune on $\mathcal{{D}}_r$"
    )
    body_lines.append(_row(
        "Dataset",
        r"Prep~(s)", r"Train~$M_k$~(s)",
        r"FT on $\mathcal{D}$~(s)", r"FT on $\mathcal{D}_r$~(s)",
        "", r"Total~(s)"
    ))
    body_lines.append(r"\midrule")
    for ds in DATASETS_TABULAR:
        model = all_model_map[ds]
        rt    = load_runtimes(ds, model, results_dir)
        k_ds  = KANON_K_PER_DATASET.get(ds, k)
        cells = [DATASET_LABEL[ds]]
        total = 0.0
        for ph in ["kanon_prep", "train_Mk", "ft_Mk_D", "ft_Mk_Dr"]:
            m, s = get_rt(rt, "kanon", f"k={k_ds}", fr, [ph])
            cells.append(_mval(m, s))
            if not math.isnan(m):
                total += m
        cells.append("")
        cells.append(_mval(total) if total > 0 else MISSING)
        body_lines.append(_row(*cells))

    # ── MAPGU (DP) ─────────────────────────────────────────────────────────
    body_lines += _section_header(
        rf"MAPGU ($\varepsilon$={eps_disp}): [embed]$^\dagger$ $\to$ DP synthesis $\to$ "
        rf"train $M_\varepsilon$ $\to$ fine-tune on $\mathcal{{D}}$ $\to$ fine-tune on $\mathcal{{D}}_r$"
    )
    body_lines.append(_row(
        "Dataset",
        r"Embed~(s)$^\dagger$", r"DP synth~(s)", r"Train~$M_\varepsilon$~(s)",
        r"FT on $\mathcal{D}$~(s)", r"FT on $\mathcal{D}_r$~(s)", r"Total~(s)"
    ))
    body_lines.append(r"\midrule")
    for ds in all_datasets:
        model   = all_model_map[ds]
        rt      = load_runtimes(ds, model, results_dir)
        has_emb = (ds == "adult")
        embed_m, _ = get_rt(rt, "dp", f"eps={eps_s}", fr, ["adult_embed"])
        cells = [DATASET_LABEL[ds], _mval(embed_m) if has_emb else MISSING]
        total = embed_m if (has_emb and not math.isnan(embed_m)) else 0.0
        for ph in ["dp_generate", "train_Meps", "ft_Meps_D", "ft_Meps_Dr"]:
            m, s = get_rt(rt, "dp", f"eps={eps_s}", fr, [ph])
            cells.append(_mval(m, s))
            if not math.isnan(m):
                total += m
        cells.append(_mval(total) if total > 0 else MISSING)
        body_lines.append(_row(*cells))

    # ── Certified-SP ──────────────────────────────────────────────────────
    body_lines += _section_header(
        rf"Certified-SP ($\varepsilon$={CERTIFIED_SP_EPS}, $\delta$={CERTIFIED_SP_DELTA}): "
        rf"train baseline $\to$ noisy unlearning $\to$ post fine-tuning"
    )
    body_lines.append(_row(
        "Dataset",
        r"Train baseline~(s)", r"Noisy unlearn~(s)", r"Post FT~(s)",
        "", "", r"Total~(s)"
    ))
    body_lines.append(r"\midrule")
    for ds in all_datasets:
        model = all_model_map[ds]
        rt    = load_runtimes(ds, model, results_dir)
        cells = [DATASET_LABEL[ds]]
        total = 0.0
        for ph in ["train_baseline", "unlearn_noisy", "post_ft"]:
            m, s = _certified_sp_phase_rt(rt, ph)
            cells.append(_mval(m, s))
            if not math.isnan(m):
                total += m
        cells.append("")
        cells.append("")
        cells.append(_mval(total) if total > 0 else MISSING)
        body_lines.append(_row(*cells))

    body_lines.append(
        rf"\multicolumn{{{N_COLS}}}{{l}}{{\scriptsize "
        rf"$^\dagger$ Adult embedding is a one-time pre-computation; "
        rf"amortised over all unlearning requests.}} \\"
    )

    caption = (
        r"Phase-by-phase runtime breakdown for multi-step unlearning methods "
        rf"(forget ratio~$= {int(fr * 100)}\%$, mean~$\pm$~std over 5~runs).  "
        r"Certified-SP baseline training is included here to show total pipeline cost "
        r"but excluded from the head-to-head speedup comparison "
        r"(Table~\ref{tab:efficiency_headtohead})."
    )
    return _latex_table(
        body_lines,
        caption=caption,
        label="tab:efficiency_phases",
        colspec=r"l r r r r r r",
        wide=True,
    )


# ---------------------------------------------------------------------------
# Table 7 — Forget ratio sensitivity
# ---------------------------------------------------------------------------

def build_table_forget_ratio(
    results_dir: str,
    sensitivity_subdir: str = FORGET_RATIO_SENSITIVITY_SUBDIR,
    fr_values: Optional[List[float]] = None,
    ft: int   = FT_EPOCHS,
    S: int    = NUM_SHARDS,
    k: int    = KANON_K,
    eps: float = DP_EPS,
    seed: int  = SEED,
) -> str:
    if fr_values is None:
        fr_values = FORGET_RATIO_VALUES

    sens_dir = os.path.join(results_dir, sensitivity_subdir)
    dataset  = "adult"
    model    = "mlp"

    METHOD_SPECS = [
        ("Retrain",
         lambda fr_s: f"{model}_mret_dret_fr={fr_s}.csv"),
        (rf"SISA ($S$={S})",
         lambda fr_s: f"{model}_sisa_mret_dret_shards={S}_fr={fr_s}.csv"),
        (rf"MAPGU$_k$ ($k$={k})",
         lambda fr_s: f"{model}_mk={k}_dret_fr={fr_s}_epochs={ft}.csv"),
        (rf"MAPGU$_\varepsilon$ ($\varepsilon$={int(eps) if eps == int(eps) else eps})",
         lambda fr_s: f"{model}_mdpret_eps={_fmt_eps(eps)}_fr={fr_s}_epochs={ft}.csv"),
        ("CERTIFIED_SP", None),  # handled via glob
    ]

    fr_labels = [rf"{int(round(fr * 100))}\%" for fr in fr_values]

    # colspec: method | fr1 (acc, mia) | fr2 ... each fr = 2 cols
    colspec = "l" + "".join(f" r r" for _ in fr_values)
    fr_header = " & ".join(
        rf"\multicolumn{{2}}{{c}}{{{lbl}}}" for lbl in fr_labels
    )
    sub_header = " & ".join(r"Acc. & MIA" for _ in fr_values)

    header_lines = [
        _row("Method", fr_header),
        _row("", sub_header),
        r"\midrule",
    ]

    data_lines: List[str] = []
    for m_label, path_fn in METHOD_SPECS:
        cells: List[str] = [m_label]
        for fr in fr_values:
            fr_s = _fmt_fr(fr)
            if path_fn is not None:
                fname = path_fn(fr_s)
                metrics = load_metrics(os.path.join(sens_dir, dataset, fname))
            else:
                # CERTIFIED_SP
                certified_sp_csv = find_certified_sp_csv(dataset, model, fr, seed, os.path.join(sens_dir))
                metrics  = load_certified_sp_post_metrics(certified_sp_csv, fr)

            acc_v, acc_s = metrics.get("Test Accuracy", (math.nan, math.nan))  # Adult uses Accuracy
            auc_v, auc_s = mia_auc(metrics)
            cells.append(_val(acc_v, acc_s))
            cells.append(_val(auc_v, auc_s))

        data_lines.append(_row(*cells))
        data_lines.append(r"\addlinespace")

    caption = (
        r"Forget ratio sensitivity on Adult / MLP.  "
        r"Acc.\ = test accuracy (\%); MIA = RMIA attack AUC (\%).  "
        r"MIA~$\approx 50\%$ indicates successful forgetting."
    )
    return _latex_table(
        header_lines + data_lines,
        caption=caption,
        label="tab:forget_ratio",
        colspec=colspec,
        wide=True,
    )


# ---------------------------------------------------------------------------
# Table 7 — FT-epoch sensitivity (CIFAR-10 / ResNet-18)
# ---------------------------------------------------------------------------

def build_table_ft_epochs(
    results_dir: str,
    mapgu_subdir: str = FT_EPOCHS_MAPGU_SUBDIR,
    certsp_subdir: str = FT_EPOCHS_CERTSP_SUBDIR,
    ft_values: Optional[List[int]] = None,
    eps: float = DP_EPS,
    fr: float = FORGET_RATIO,
) -> str:
    """Table 7: FT-epoch sensitivity for CIFAR-10 / ResNet-18 (MAPGU_ε + Certified-SP)."""
    if ft_values is None:
        ft_values = FT_EPOCH_VALUES

    eps_s    = _fmt_eps(eps)
    eps_disp = int(eps) if eps == int(eps) else eps
    fr_s     = _fmt_fr(fr)

    mapgu_dir  = os.path.join(results_dir, mapgu_subdir,  "cifar10")
    certsp_dir = os.path.join(results_dir, certsp_subdir, "cifar10")

    epoch_labels = [str(n) for n in ft_values]

    colspec    = "l" + " r r" * len(ft_values)
    epoch_hdr  = " & ".join(rf"\multicolumn{{2}}{{c}}{{{lbl}}}" for lbl in epoch_labels)
    sub_header = " & ".join(r"Acc. & MIA" for _ in ft_values)

    header_lines = [
        _row("Method", epoch_hdr),
        _row("", sub_header),
        r"\midrule",
    ]

    data_lines: List[str] = []

    # MAPGU_ε row
    cells: List[str] = [rf"MAPGU$_\varepsilon$ ($\varepsilon$={eps_disp})"]
    for n in ft_values:
        fname   = f"resnet18_mdpret_eps={eps_s}_fr={fr_s}_epochs={n}.csv"
        metrics = load_metrics(os.path.join(mapgu_dir, fname))
        acc_v, acc_s = metrics.get("Test Accuracy", (math.nan, math.nan))
        auc_v, auc_s = mia_auc(metrics)
        cells.append(_val(acc_v, acc_s))
        cells.append(_val(auc_v, auc_s))
    data_lines.append(_row(*cells))
    data_lines.append(r"\addlinespace")

    # Certified-SP row
    cells = [rf"Certified-SP ($\varepsilon$={CERTIFIED_SP_EPS}, $\delta$={CERTIFIED_SP_DELTA})"]
    for n in ft_values:
        csv_path = os.path.join(
            certsp_dir,
            f"resnet18_certified_sp_eps={CERTIFIED_SP_EPS}_post_epochs={n}_summary.csv",
        )
        metrics  = load_certified_sp_post_metrics(csv_path, fr)
        acc_v, acc_s = metrics.get("Test Accuracy", (math.nan, math.nan))
        auc_v, auc_s = mia_auc(metrics)
        cells.append(_val(acc_v, acc_s))
        cells.append(_val(auc_v, auc_s))
    data_lines.append(_row(*cells))
    data_lines.append(r"\addlinespace")

    caption = (
        rf"Fine-tuning epoch sensitivity on CIFAR-10 / ResNet-18 "
        rf"(forget ratio~$= {int(fr * 100)}\%$, $\varepsilon$={eps_disp}).  "
        r"Acc.\ = test accuracy (\%); MIA = RMIA attack AUC (\%).  "
        r"MIA~$\approx 50\%$ indicates successful forgetting."
    )
    return _latex_table(
        header_lines + data_lines,
        caption=caption,
        label="tab:ft_epochs",
        colspec=colspec,
        wide=True,
    )


# ---------------------------------------------------------------------------
# Write all tables
# ---------------------------------------------------------------------------

ALL_TABLE_IDS = ["1", "2", "3", "4", "6a", "6b", "7", "8"]


def write_all_tables(
    out_dir: str,
    results_dir: str,
    table_sel: str = "all",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    def _save(name: str, content: str) -> None:
        path = os.path.join(out_dir, f"{name}.tex")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"  Wrote {path}")

    def _run(tid: str) -> None:
        print(f"Building Table {tid}...")
        if tid == "1":
            _save("table_1_datasets",        build_table_1())
        elif tid == "2":
            _save("table_2_methods",         build_table_2())
        elif tid == "3":
            _save("table_3_hyperparams",     build_table_3())
        elif tid == "4":
            for model in MODELS_TABULAR:
                _save(f"table_4_quality_{model}",
                      build_table_quality(model, results_dir, DATASETS_TABULAR))
            _save("table_4_quality_resnet18",
                  build_table_quality(MODEL_VISION, results_dir))
        elif tid == "6a":
            _save("table_6a_efficiency_headtohead",
                  build_table_efficiency_headtohead(results_dir))
        elif tid == "6b":
            _save("table_6b_efficiency_phases",
                  build_table_efficiency_phases(results_dir))
        elif tid == "7":
            _save("table_7_ft_epochs",
                  build_table_ft_epochs(results_dir))
        elif tid == "8":
            _save("table_8_forget_ratio",
                  build_table_forget_ratio(results_dir))

    ids = ALL_TABLE_IDS if table_sel == "all" else [table_sel]
    for tid in ids:
        _run(tid)

    # Also write a single combined file
    if table_sel == "all":
        combined = os.path.join(out_dir, "paper_tables.tex")
        with open(combined, "w", encoding="utf-8") as fout:
            for fname in sorted(os.listdir(out_dir)):
                if fname.endswith(".tex") and fname != "paper_tables.tex":
                    with open(os.path.join(out_dir, fname), encoding="utf-8") as fin:
                        fout.write(f"% {'='*60}\n% {fname}\n% {'='*60}\n\n")
                        fout.write(fin.read())
                        fout.write("\n\n")
        print(f"  Combined → {combined}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Compose LaTeX tables for the MAPGU paper.")
    p.add_argument(
        "--table", default="all",
        choices=["all"] + ALL_TABLE_IDS,
        help="Which table to generate (default: all)",
    )
    p.add_argument(
        "--out", default=os.path.join(_ROOT, "results", "paper_tables"),
        help="Output directory for .tex files",
    )
    p.add_argument(
        "--results_dir", default=os.path.join(_ROOT, "results"),
        help="Root results directory",
    )
    args = p.parse_args()

    print(f"Results dir : {args.results_dir}")
    print(f"Output dir  : {args.out}")
    print()
    write_all_tables(args.out, args.results_dir, args.table)


if __name__ == "__main__":
    main()

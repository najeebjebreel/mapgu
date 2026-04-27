"""DP dataset generators: Laplace + ExponentialCategorical for tabular data, DP-Pix for CIFAR-10."""

from __future__ import annotations

import os
import time
import pickle
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from diffprivlib.mechanisms import Laplace, ExponentialCategorical

from mapgu.utils import get_logger
from mapgu.config import ADULT_COLUMNS, ADULT_NUM_COLS as ADULT_NUMERIC, ADULT_LABEL
from mapgu.data.privacy.dp_pix import dp_pix

logger = get_logger(__name__)


# -----------------------------
# Shared runtime logging
# -----------------------------

@dataclass
class RuntimeRow:
    dataset: str
    epsilon: float
    n_rows: int
    n_cols: int
    m_perturbed: int
    eps_attr: float
    laplace_time_sec: float
    utility_time_sec: float
    total_time_sec: float
    output_path: str


def append_runtime_csv(runtime_csv: str, row: Union[RuntimeRow, Dict[str, Any]]) -> None:
    os.makedirs(os.path.dirname(runtime_csv) or ".", exist_ok=True)
    if isinstance(row, RuntimeRow):
        df_row = pd.DataFrame([asdict(row)])
    else:
        df_row = pd.DataFrame([row])
    if os.path.exists(runtime_csv) and os.path.getsize(runtime_csv) > 0:
        df_row.to_csv(runtime_csv, mode="a", header=False, index=False)
    else:
        df_row.to_csv(runtime_csv, mode="w", header=True, index=False)


def _exists_nonempty_file(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _dir_has_files(path: str) -> bool:
    if not os.path.isdir(path):
        return False
    for _, _, files in os.walk(path):
        if files:
            return True
    return False


# -----------------------------
# Shared Laplace helper
# -----------------------------

def laplace_perturb_column(values: pd.Series, epsilon: float) -> np.ndarray:
    """
    Laplace mechanism with sensitivity = (max-min) on observed column.
    Clip to observed bounds (post-processing; DP-safe).
    Returns float array.
    """
    a = float(values.min())
    b = float(values.max())
    sensitivity = (b - a) if (b > a) else 1.0
    mech = Laplace(epsilon=float(epsilon), sensitivity=sensitivity)

    noisy = values.astype(float).apply(lambda x: mech.randomise(float(x))).to_numpy(dtype=float)
    noisy = np.clip(noisy, a, b)
    return noisy


# -----------------------------
# ADULT
# -----------------------------

def generate_dp_adult(
    in_path: str,
    out_dir: str,
    utilities_pkl: str,
    eps_list: List[float],
    seed: int = 7,
    runtime_csv: Optional[str] = None,
    *,
    skip_existing: bool = False,
    return_runtime_rows: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """
    Adult:
    - Drop fnlwgt, education
    - Do NOT perturb label (income)
    - Laplace for numeric
    - ExponentialCategorical for categorical using utilities.pkl
    - Split epsilon equally across perturbed attributes

    Returns:
      - If return_runtime_rows=True: list of dict rows (one per eps)
      - Else: None
    """
    os.makedirs(out_dir, exist_ok=True)
    np.random.seed(seed)

    data = pd.read_csv(in_path, names=ADULT_COLUMNS, sep=r" *, *", engine="python", na_values="?")
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)
    data = data.drop(columns=["fnlwgt", "education"])

    with open(utilities_pkl, "rb") as f:
        utility_dict: Dict[str, Any] = pickle.load(f)

    numeric_cols = ADULT_NUMERIC
    label_col = ADULT_LABEL
    cat_cols_all = [c for c in data.columns if c not in numeric_cols + [label_col]]
    cat_cols = [c for c in cat_cols_all if c in utility_dict]  # only those with utilities

    missing_utils = sorted(set(cat_cols_all) - set(cat_cols))
    if missing_utils:
        logger.warning("[adult] Missing utilities (will NOT perturb these and NOT count in eps split): %s",
                       ", ".join(missing_utils))

    perturbed_cols = numeric_cols + cat_cols
    m = len(perturbed_cols)
    if m == 0:
        raise ValueError("[adult] No columns selected for DP perturbation.")

    rows: List[Dict[str, Any]] = []

    for eps in eps_list:
        out_path = os.path.join(out_dir, f"dp_adult_eps={eps}.csv")
        if skip_existing and _exists_nonempty_file(out_path):
            logger.info("[adult eps=%s] [skip] exists: %s", eps, out_path)
            # still record a row (with times = 0) to keep bookkeeping consistent
            row = RuntimeRow(
                dataset="adult",
                epsilon=float(eps),
                n_rows=int(len(data)),
                n_cols=int(data.shape[1]),
                m_perturbed=int(m),
                eps_attr=float(eps) / m,
                laplace_time_sec=0.0,
                utility_time_sec=0.0,
                total_time_sec=0.0,
                output_path=out_path,
            )
            if runtime_csv:
                append_runtime_csv(runtime_csv, row)
            rows.append(asdict(row))
            continue

        t_total0 = time.time()
        data_copy = data.copy()
        eps_attr = float(eps) / m

        # ---- Numeric (Laplace) ----
        t0 = time.time()
        for col in numeric_cols:
            noisy = laplace_perturb_column(data_copy[col], eps_attr)
            data_copy[col] = noisy  # keep continuous
        lap_t = time.time() - t0

        # ---- Categorical (ExponentialCategorical) ----
        t0 = time.time()
        for col in cat_cols:
            utility_list = utility_dict[col]
            norm_list = [[str(a), str(b), float(u)] for (a, b, u) in utility_list]
            mech = ExponentialCategorical(epsilon=eps_attr, utility_list=norm_list)
            data_copy[col] = data_copy[col].apply(lambda x: mech.randomise(str(x)))
        util_t = time.time() - t0

        data_copy.to_csv(out_path, index=False)

        total_t = time.time() - t_total0
        logger.info("[adult eps=%s] saved %s | eps_attr=%.6g | lap=%.2fs util=%.2fs total=%.2fs | m=%d",
                    eps, out_path, eps_attr, lap_t, util_t, total_t, m)

        row = RuntimeRow(
            dataset="adult",
            epsilon=float(eps),
            n_rows=int(len(data_copy)),
            n_cols=int(data_copy.shape[1]),
            m_perturbed=int(m),
            eps_attr=float(eps_attr),
            laplace_time_sec=float(lap_t),
            utility_time_sec=float(util_t),
            total_time_sec=float(total_t),
            output_path=out_path,
        )

        if runtime_csv:
            append_runtime_csv(runtime_csv, row)
        rows.append(asdict(row))

    return rows if return_runtime_rows else None


# -----------------------------
# HEART / CREDIT (Laplace-only)
# -----------------------------

def generate_dp_laplace_only(
    dataset_name: str,
    in_path: str,
    out_dir: str,
    label_col: str,
    eps_list: List[float],
    runtime_csv: Optional[str] = None,
    *,
    skip_existing: bool = False,
    return_runtime_rows: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """
    Laplace-only DP for tabular datasets. Does epsilon split across perturbed columns.

    Returns list of runtime rows if return_runtime_rows=True.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    if dataset_name == "heart":
        data = pd.read_csv(in_path, sep=";")
        if "id" in data.columns:
            data = data.drop(columns=["id"])
    else:
        data = pd.read_csv(in_path)
        if "Unnamed: 0" in data.columns:
            data = data.drop(columns=["Unnamed: 0"])

    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    cols_to_perturb = [c for c in data.columns if c != label_col]
    m = len(cols_to_perturb)
    if m == 0:
        raise ValueError(f"[{dataset_name}] No columns to perturb (label_col={label_col}).")

    for eps in eps_list:
        out_path = os.path.join(out_dir, f"dp_{dataset_name}_eps={eps}.csv")
        if skip_existing and _exists_nonempty_file(out_path):
            logger.info("[%s eps=%s] [skip] exists: %s", dataset_name, eps, out_path)
            row = RuntimeRow(
                dataset=dataset_name,
                epsilon=float(eps),
                n_rows=int(len(data)),
                n_cols=int(data.shape[1]),
                m_perturbed=int(m),
                eps_attr=float(eps) / m,
                laplace_time_sec=0.0,
                utility_time_sec=0.0,
                total_time_sec=0.0,
                output_path=out_path,
            )
            if runtime_csv:
                append_runtime_csv(runtime_csv, row)
            rows.append(asdict(row))
            continue

        t_total0 = time.time()
        data_copy = data.copy()

        eps_attr = float(eps) / m
        t0 = time.time()
        for col in cols_to_perturb:
            noisy = laplace_perturb_column(data_copy[col], eps_attr)
            if pd.api.types.is_integer_dtype(data_copy[col].dtype):
                data_copy[col] = np.rint(noisy).astype(data_copy[col].dtype)
            else:
                data_copy[col] = noisy.astype(data_copy[col].dtype)
        lap_t = time.time() - t0

        data_copy.to_csv(out_path, index=False)
        total_t = time.time() - t_total0

        logger.info("[%s eps=%s] saved %s | eps_attr=%.6g | lap=%.2fs util=0.00s total=%.2fs | m=%d",
                    dataset_name, eps, out_path, eps_attr, lap_t, total_t, m)

        row = RuntimeRow(
            dataset=dataset_name,
            epsilon=float(eps),
            n_rows=int(len(data_copy)),
            n_cols=int(data_copy.shape[1]),
            m_perturbed=int(m),
            eps_attr=float(eps_attr),
            laplace_time_sec=float(lap_t),
            utility_time_sec=0.0,
            total_time_sec=float(total_t),
            output_path=out_path,
        )
        if runtime_csv:
            append_runtime_csv(runtime_csv, row)
        rows.append(asdict(row))

    return rows if return_runtime_rows else None


# -----------------------------
# CIFAR-10 (DP-Pix)
# -----------------------------

def _load_cifar10_trainset(data_root: str):
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader

    transform = transforms.Compose([transforms.ToTensor()])
    trainset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform)
    trainloader = DataLoader(trainset, batch_size=1, shuffle=False, num_workers=0)
    return trainset, trainloader


def generate_dp_cifar10_dppix(
    out_dir: str,
    eps_list: List[float],
    m: int,
    block_size: int,
    data_root: str,
    runtime_csv: Optional[str] = None,
    *,
    skip_existing: bool = False,
    return_runtime_rows: bool = False,
) -> Optional[List[Dict[str, Any]]]:
    """
    Saves DP-Pix perturbed CIFAR-10 as a compressed NPZ file:
      {out_dir}/eps={eps}/dp_cifar10.npz  (keys: X uint8 [N,3,32,32], y int64 [N])

    Returns list of runtime rows if return_runtime_rows=True.
    """
    os.makedirs(out_dir, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    trainset, trainloader = _load_cifar10_trainset(data_root)
    labels = np.array(trainset.targets, dtype=np.int64)

    for eps in eps_list:
        eps_tag = f"{float(eps):.12g}"
        eps_dir = os.path.join(out_dir, f"eps={eps_tag}")
        npz_path = os.path.join(eps_dir, "dp_cifar10.npz")

        if skip_existing and os.path.isfile(npz_path):
            logger.info("[cifar10 eps=%s] [skip] exists: %s", eps, npz_path)
            row = RuntimeRow(
                dataset="cifar10_dppix",
                epsilon=float(eps),
                n_rows=int(len(trainset)),
                n_cols=0,
                m_perturbed=int(m),
                eps_attr=float(eps),
                laplace_time_sec=0.0,
                utility_time_sec=0.0,
                total_time_sec=0.0,
                output_path=npz_path,
            )
            if runtime_csv:
                append_runtime_csv(runtime_csv, row)
            rows.append(asdict(row))
            continue

        t_total0 = time.time()
        os.makedirs(eps_dir, exist_ok=True)

        images_list = []
        for images, _ in tqdm(trainloader, total=len(trainloader), desc=f"cifar eps={eps}"):
            image = images[0]  # [3, H, W] float32
            dp_image = dp_pix(image, block_size, m, float(eps))
            img_uint8 = (dp_image.clamp(0, 1) * 255).byte().numpy()  # [3, 32, 32] uint8
            images_list.append(img_uint8)

        X = np.stack(images_list, axis=0)  # [N, 3, 32, 32] uint8
        np.savez_compressed(npz_path, X=X, y=labels)

        total_t = time.time() - t_total0
        logger.info("[cifar10 eps=%s] saved %s | total=%.2fs", eps, npz_path, total_t)

        row = RuntimeRow(
            dataset="cifar10_dppix",
            epsilon=float(eps),
            n_rows=int(len(trainset)),
            n_cols=0,
            m_perturbed=int(m),
            eps_attr=float(eps),
            laplace_time_sec=0.0,
            utility_time_sec=0.0,
            total_time_sec=float(total_t),
            output_path=npz_path,
        )
        if runtime_csv:
            append_runtime_csv(runtime_csv, row)
        rows.append(asdict(row))

    return rows if return_runtime_rows else None

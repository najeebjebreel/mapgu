#!/usr/bin/env python3
"""
Prepare privacy-protected datasets OFFLINE (one-time) for MAPGU.

Replaces scripts/prepare_private_data.py with mapgu-package imports.

Usage (as module):
  python -m scripts.prepare_data kanon --dataset adult --k-values 30 --skip-existing
  python -m scripts.prepare_data dp   --dataset adult --eps 1 --skip-existing
"""

from __future__ import annotations

import os
import csv
import time
import argparse
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

import numpy as np

from mapgu.utils import get_logger
from mapgu.config import DATA_DIR, DP_DIR, KANON_DIR, ADULT_EMB_RT_CSV

from mapgu.data.loaders import DatasetLoader
from mapgu.models import build_cluster_repr_onehot, build_cluster_repr_tabnet
from mapgu.data.privacy.kanon import mdav_clusters, probabilistic_k_anonymize_by_permutation

logger = get_logger(__name__)


# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------
def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _write_csv(path: str, header: List[str], row: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerow(row)


def _fmt_eps(eps: float) -> str:
    return f"{float(eps):.12g}"


def save_npz(path: str, X: np.ndarray, y: np.ndarray) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    np.savez_compressed(path, X=X.astype(np.float32), y=y.astype(np.int64))


def _exists_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def load_adult_embedding_offline_times() -> Tuple[float, float]:
    """
    Returns (embedding_offline_sec, utility_offline_sec) from:
      data/adult/embeddings_runtimes.csv
    If missing/unreadable: returns (0.0, 0.0).
    """
    if not _exists_nonempty(ADULT_EMB_RT_CSV):
        return 0.0, 0.0

    try:
        with open(ADULT_EMB_RT_CSV, "r", newline="") as f:
            r = csv.DictReader(f)
            rows = list(r)
        if not rows:
            return 0.0, 0.0
        row = rows[-1]
        emb = float(row.get("embedding", 0.0))
        util = float(row.get("utility", 0.0))
        return emb, util
    except Exception:
        return 0.0, 0.0


# --------------------------------------------------------------------------------------
# Runtime schemas
# --------------------------------------------------------------------------------------
@dataclass
class KAnonRuntimeRow:
    dataset: str
    seed: int
    cluster_repr: str
    k: int
    n_train: int

    onehot_time_sec: float              # Adult+onehot only (transform raw->onehot)
    repr_time_sec: float                # time to build clustering repr (onehot/tabnet)
    embedding_offline_sec: float        # Adult+tabnet only, loaded from embeddings_runtimes.csv
    utility_offline_sec: float          # Adult+tabnet only, loaded from embeddings_runtimes.csv

    anon_time_sec: float                # MDAV + swapping
    total_time_sec: float


@dataclass
class DPRuntimeRow:
    dataset: str
    seed: int
    epsilon: float

    embedding_offline_sec: float        # Adult only, loaded from embeddings_runtimes.csv
    utility_offline_sec: float          # Adult only, loaded from embeddings_runtimes.csv

    anon_time_sec: float                # DP generation time (row["total_time_sec"] from generator)
    total_time_sec: float               # embedding_offline + utility_offline + anon
    output_path: str


def save_kanon_runtimes_csv(out_dir: str, r: KAnonRuntimeRow) -> str:
    path = os.path.join(out_dir, "runtimes.csv")
    row = asdict(r)
    _write_csv(path, header=list(row.keys()), row=row)
    return path


def save_dp_runtimes_csv(out_dir: str, r: DPRuntimeRow) -> str:
    path = os.path.join(out_dir, "runtimes.csv")
    row = asdict(r)
    _write_csv(path, header=list(row.keys()), row=row)
    return path


# --------------------------------------------------------------------------------------
# K-ANON PREP
# --------------------------------------------------------------------------------------
def prepare_kanon(
    dataset: str,
    k_values: List[int],
    cluster_repr: str,
    seed: int,
    embeddings_pkl: Optional[str],
    skip_existing: bool,
    perm_type: str = "rowwise",
    cifar_root: Optional[str] = None,
) -> None:
    """
    Output:
      data/k_anon_data/<dataset>/cluster=<repr>/seed=<seed>/k=<k>/train_anon.npz
      data/k_anon_data/<dataset>/cluster=<repr>/seed=<seed>/k=<k>/runtimes.csv
    """
    preproc = None
    X_train_raw_df = None

    if dataset == "adult":
        X_train, y_train, _X_test, _y_test, preproc, X_train_raw_df, _X_test_raw_df = DatasetLoader.load_adult(return_raw=True)
    elif dataset == "credit":
        X_train, y_train, _X_test, _y_test, preproc = DatasetLoader.load_credit()
    elif dataset == "heart":
        X_train, y_train, _X_test, _y_test, preproc = DatasetLoader.load_heart()
    elif dataset == "cifar10":
        import torch
        from torchvision.datasets import CIFAR10
        from mapgu.data.privacy.cifar_kanon import build_cifar_kanon_data
        from mapgu.config import CIFAR_MEAN, CIFAR_STD

        _root = cifar_root if cifar_root is not None else DATA_DIR
        cifar_train = CIFAR10(root=_root, train=True, download=True)
        raw = np.asarray(cifar_train.data, dtype=np.float32) / 255.0       # (N, H, W, C)
        raw = raw.transpose(0, 3, 1, 2)                                      # (N, C, H, W)
        _mean = np.array(CIFAR_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
        _std  = np.array(CIFAR_STD,  dtype=np.float32).reshape(1, 3, 1, 1)
        images_norm = (raw - _mean) / _std
        y_arr = np.asarray(cifar_train.targets, dtype=np.int64)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for k in k_values:
            out_dir = os.path.join(KANON_DIR, "cifar10", "cluster=latent", f"seed={seed}", f"k={int(k)}")
            out_npz = os.path.join(out_dir, "train_anon.npz")
            out_rt  = os.path.join(out_dir, "runtimes.csv")

            if skip_existing and _exists_nonempty(out_npz) and os.path.exists(out_rt):
                logger.info("[kanon][skip] exists: %s", out_npz)
                continue

            _ensure_dir(out_dir)

            t0 = time.time()
            images_anon, y_anon, feat_summary, anon_summary = build_cifar_kanon_data(
                images_norm=images_norm.copy(),
                y=y_arr.copy(),
                k=int(k),
                seed=int(seed),
                device=device,
            )
            anon_time = time.time() - t0

            save_npz(out_npz, images_anon, y_anon)

            rt_path = save_kanon_runtimes_csv(
                out_dir,
                KAnonRuntimeRow(
                    dataset="cifar10",
                    seed=int(seed),
                    cluster_repr="latent",
                    k=int(k),
                    n_train=int(images_norm.shape[0]),
                    onehot_time_sec=0.0,
                    repr_time_sec=float(anon_time),
                    embedding_offline_sec=0.0,
                    utility_offline_sec=0.0,
                    anon_time_sec=float(anon_time),
                    total_time_sec=float(anon_time),
                ),
            )

            logger.info("[kanon] cifar10 k=%d repr=latent | %s | %s | anon=%.2fs | %s | %s",
                        k, feat_summary, anon_summary, anon_time, out_npz, rt_path)
        return
    else:
        raise ValueError("k-anonymity preparation supported only for: adult/credit/heart/cifar10")

    n_train = int(X_train.shape[0])

    # offline adult embedding/util times (only meaningful for adult+tabnet)
    emb_offline, util_offline = load_adult_embedding_offline_times()

    for k in k_values:
        out_dir = os.path.join(
            KANON_DIR,
            dataset,
            f"cluster={cluster_repr}",
            f"seed={seed}",
            f"k={int(k)}",
        )
        out_npz = os.path.join(out_dir, "train_anon.npz")
        out_rt = os.path.join(out_dir, "runtimes.csv")

        if skip_existing and _exists_nonempty(out_npz) and os.path.exists(out_rt):
            logger.info("[kanon][skip] exists: %s", out_npz)
            continue

        _ensure_dir(out_dir)

        onehot_time = 0.0
        repr_time = 0.0
        embedding_offline_sec = 0.0
        utility_offline_sec = 0.0

        # 1) Build clustering representation (and time it)
        if cluster_repr == "onehot":
            # For Adult: time actual one-hot pipeline transform from raw (so it's a real cost)
            if dataset == "adult":
                if preproc is None or X_train_raw_df is None:
                    raise RuntimeError("Adult preprocessor/raw DF missing; cannot time onehot transform.")
                t0 = time.time()
                X_repr = preproc.transform(X_train_raw_df)
                onehot_time = time.time() - t0
                repr_time = float(onehot_time)  # repr is the onehot transform here
            else:
                # credit/heart have no categorical onehot stage in your baseline; treat as 0
                t0 = time.time()
                X_repr = build_cluster_repr_onehot(X_train)
                repr_time = time.time() - t0
                onehot_time = 0.0

        elif cluster_repr == "tabnet":
            if dataset != "adult":
                raise ValueError("cluster-repr tabnet supported only for Adult.")
            if embeddings_pkl is None or (not os.path.exists(embeddings_pkl)):
                raise ValueError("--embeddings-pkl is required for --cluster-repr tabnet and must exist.")
            if X_train_raw_df is None:
                raise RuntimeError("Adult raw DF missing; cannot build tabnet repr.")

            # offline costs must come from embeddings_runtimes.csv
            embedding_offline_sec = float(emb_offline)
            utility_offline_sec = float(util_offline)

            # repr build time (using embeddings.pkl) is still useful to record separately
            t0 = time.time()
            X_repr = build_cluster_repr_tabnet(X_train_raw_df=X_train_raw_df, embeddings_pkl_path=embeddings_pkl)
            repr_time = time.time() - t0

        else:
            raise ValueError(f"Unknown cluster_repr={cluster_repr}")

        # 2) MDAV + swapping (features only), y unchanged
        t1 = time.time()

        clusters = mdav_clusters(
            X=np.array(X_repr, copy=True),
            y=np.array(y_train, copy=True),
            k=int(k),
            show_progress=True,
        )

        X_anon, y_anon = probabilistic_k_anonymize_by_permutation(
            X=np.array(X_train, copy=True),
            y=np.array(y_train, copy=True),
            clusters=clusters,
            seed=int(seed),
            protect_mask=None,
            perm_type=str(perm_type),
        )

        if not np.array_equal(y_anon, y_train):
            raise RuntimeError("probabilistic_k_anonymize_by_permutation changed y! This must not happen.")

        anon_time = time.time() - t1

        total_time = float(
            embedding_offline_sec
            + utility_offline_sec
            + repr_time
            + anon_time
        )

        save_npz(out_npz, X_anon, y_anon)

        rt_path = save_kanon_runtimes_csv(
            out_dir,
            KAnonRuntimeRow(
                dataset=dataset,
                seed=int(seed),
                cluster_repr=str(cluster_repr),
                k=int(k),
                n_train=n_train,
                onehot_time_sec=float(onehot_time),
                repr_time_sec=float(repr_time),
                embedding_offline_sec=float(embedding_offline_sec),
                utility_offline_sec=float(utility_offline_sec),
                anon_time_sec=float(anon_time),
                total_time_sec=float(total_time),
            ),
        )

        if cluster_repr == "onehot":
            detail = f"onehot={onehot_time:.2f}s"
        else:
            detail = f"embed_offline={embedding_offline_sec:.2f}s util_offline={utility_offline_sec:.2f}s repr={repr_time:.2f}s"

        logger.info("[kanon] %s k=%d repr=%s | %s anon=%.2fs total=%.2fs | %s | %s",
                    dataset, k, cluster_repr, detail, anon_time, total_time, out_npz, rt_path)


# --------------------------------------------------------------------------------------
# DP PREP (PER-EPS RUNTIMES IN OUTPUT FOLDERS, LOADING OFFLINE EMB/UTIL TIMES)
# --------------------------------------------------------------------------------------
def prepare_dp(
    dataset: str,
    eps_values: List[float],
    seed: int,
    skip_existing: bool,
    adult_in_path: str,
    adult_utilities_pkl: str,
    heart_in_path: str,
    credit_in_path: str,
    cifar_root: str,
    cifar_m: int,
    cifar_block: int,
) -> None:
    """
    Adult/Credit/Heart:
      data/dp_data/<dataset>/eps=<eps>/dp_<dataset>.csv
      data/dp_data/<dataset>/eps=<eps>/runtimes.csv

    CIFAR10:
      data/dp_data/cifar10/eps=<eps>/dp_cifar10.npz
      + data/dp_data/cifar10/eps=<eps>/runtimes.csv
    """
    _ensure_dir(DP_DIR)

    # offline adult embedding/util times (used only for adult dp)
    emb_offline, util_offline = load_adult_embedding_offline_times()

    for eps in eps_values:
        eps_str = _fmt_eps(eps)

        if dataset == "adult":
            from mapgu.data.privacy.dp import generate_dp_adult

            out_dir = os.path.join(DP_DIR, "adult", f"eps={eps_str}")
            out_csv = os.path.join(out_dir, "dp_adult.csv")
            _ensure_dir(out_dir)

            # run generator and get its measured runtime row
            rows = generate_dp_adult(
                in_path=adult_in_path,
                out_dir=out_dir,
                utilities_pkl=adult_utilities_pkl,
                eps_list=[float(eps)],
                seed=int(seed),
                runtime_csv=None,
                skip_existing=skip_existing,
                return_runtime_rows=True,
            ) or []

            # generator writes dp_adult_eps=<eps>.csv -> rename to dp_adult.csv
            tmp_path = os.path.join(out_dir, f"dp_adult_eps={eps_str}.csv")
            if os.path.exists(tmp_path):
                if os.path.exists(out_csv):
                    os.remove(out_csv)
                os.replace(tmp_path, out_csv)

            # DP generation runtime comes from row["total_time_sec"]
            anon_time = float(rows[-1]["total_time_sec"]) if rows else 0.0

            total_time = float(emb_offline + util_offline + anon_time)

            save_dp_runtimes_csv(
                out_dir,
                DPRuntimeRow(
                    dataset="adult",
                    seed=int(seed),
                    epsilon=float(eps),
                    embedding_offline_sec=float(emb_offline),
                    utility_offline_sec=float(util_offline),
                    anon_time_sec=float(anon_time),
                    total_time_sec=float(total_time),
                    output_path=out_csv,
                ),
            )

            logger.info("[dp] adult eps=%s | embed_offline=%.2fs util_offline=%.2fs anon=%.2fs total=%.2fs | %s | %s",
                        eps_str, emb_offline, util_offline, anon_time, total_time,
                        out_csv, os.path.join(out_dir, "runtimes.csv"))

        elif dataset == "credit":
            from mapgu.data.privacy.dp import generate_dp_laplace_only

            out_dir = os.path.join(DP_DIR, "credit", f"eps={eps_str}")
            out_csv = os.path.join(out_dir, "dp_credit.csv")
            _ensure_dir(out_dir)

            rows = generate_dp_laplace_only(
                dataset_name="credit",
                in_path=credit_in_path,
                out_dir=out_dir,
                label_col="SeriousDlqin2yrs",
                eps_list=[float(eps)],
                runtime_csv=None,
                skip_existing=skip_existing,
                return_runtime_rows=True,
            ) or []

            tmp_path = os.path.join(out_dir, f"dp_credit_eps={eps_str}.csv")
            if os.path.exists(tmp_path):
                if os.path.exists(out_csv):
                    os.remove(out_csv)
                os.replace(tmp_path, out_csv)

            anon_time = float(rows[-1]["total_time_sec"]) if rows else 0.0
            total_time = float(anon_time)

            save_dp_runtimes_csv(
                out_dir,
                DPRuntimeRow(
                    dataset="credit",
                    seed=int(seed),
                    epsilon=float(eps),
                    embedding_offline_sec=0.0,
                    utility_offline_sec=0.0,
                    anon_time_sec=float(anon_time),
                    total_time_sec=float(total_time),
                    output_path=out_csv,
                ),
            )

            logger.info("[dp] credit eps=%s | anon=%.2fs total=%.2fs | %s | %s",
                        eps_str, anon_time, total_time, out_csv, os.path.join(out_dir, "runtimes.csv"))

        elif dataset == "heart":
            from mapgu.data.privacy.dp import generate_dp_laplace_only

            out_dir = os.path.join(DP_DIR, "heart", f"eps={eps_str}")
            out_csv = os.path.join(out_dir, "dp_heart.csv")
            _ensure_dir(out_dir)

            rows = generate_dp_laplace_only(
                dataset_name="heart",
                in_path=heart_in_path,
                out_dir=out_dir,
                label_col="cardio",
                eps_list=[float(eps)],
                runtime_csv=None,
                skip_existing=skip_existing,
                return_runtime_rows=True,
            ) or []

            tmp_path = os.path.join(out_dir, f"dp_heart_eps={eps_str}.csv")
            if os.path.exists(tmp_path):
                if os.path.exists(out_csv):
                    os.remove(out_csv)
                os.replace(tmp_path, out_csv)

            anon_time = float(rows[-1]["total_time_sec"]) if rows else 0.0
            total_time = float(anon_time)

            save_dp_runtimes_csv(
                out_dir,
                DPRuntimeRow(
                    dataset="heart",
                    seed=int(seed),
                    epsilon=float(eps),
                    embedding_offline_sec=0.0,
                    utility_offline_sec=0.0,
                    anon_time_sec=float(anon_time),
                    total_time_sec=float(total_time),
                    output_path=out_csv,
                ),
            )

            logger.info("[dp] heart eps=%s | anon=%.2fs total=%.2fs | %s | %s",
                        eps_str, anon_time, total_time, out_csv, os.path.join(out_dir, "runtimes.csv"))

        elif dataset == "cifar10":
            from mapgu.data.privacy.dp import generate_dp_cifar10_dppix

            out_base = os.path.join(DP_DIR, "cifar10")
            eps_dir = os.path.join(out_base, f"eps={eps_str}")

            rows = generate_dp_cifar10_dppix(
                out_dir=out_base,
                eps_list=[float(eps)],
                m=int(cifar_m),
                block_size=int(cifar_block),
                data_root=cifar_root,
                runtime_csv=None,
                skip_existing=skip_existing,
                return_runtime_rows=True,
            ) or []
            anon_time = float(rows[-1]["total_time_sec"]) if rows else 0.0

            _ensure_dir(eps_dir)
            save_dp_runtimes_csv(
                eps_dir,
                DPRuntimeRow(
                    dataset="cifar10_dppix",
                    seed=int(seed),
                    epsilon=float(eps),
                    embedding_offline_sec=0.0,
                    utility_offline_sec=0.0,
                    anon_time_sec=float(anon_time),
                    total_time_sec=float(anon_time),
                    output_path=eps_dir,
                ),
            )

            logger.info("[dp] cifar10 eps=%s | anon=%.2fs total=%.2fs | %s | %s",
                        eps_str, anon_time, anon_time, eps_dir, os.path.join(eps_dir, "runtimes.csv"))

        else:
            raise ValueError("dp dataset must be one of: adult, credit, heart, cifar10")


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser("Prepare privacy-preserved data (dp + probabilistic k-anon)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # k-anon
    k = sub.add_parser("kanon")
    k.add_argument("--dataset", choices=["adult", "credit", "heart"], required=True)
    k.add_argument("--k-values", nargs="+", type=int, default=[30])
    k.add_argument("--cluster-repr", choices=["onehot", "tabnet"], default="onehot")
    k.add_argument("--perm-type", choices=["rowwise", "colwise"], default="rowwise",
                   help="QI permutation strategy: 'rowwise' (default) or 'colwise'")
    k.add_argument("--embeddings-pkl", default=None)
    k.add_argument("--seed", type=int, default=7)
    k.add_argument("--skip-existing", action="store_true")

    # dp
    d = sub.add_parser("dp")
    d.add_argument("--dataset", choices=["adult", "credit", "heart", "cifar10"], required=True)
    d.add_argument("--eps", nargs="+", type=float, default=[1.0])
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--skip-existing", action="store_true")

    # dp paths
    d.add_argument("--adult-in-path", default=os.path.join(DATA_DIR, "adult", "adult.data"))
    d.add_argument("--adult-utilities-pkl", default=os.path.join(DATA_DIR, "adult", "utilities.pkl"))
    d.add_argument("--heart-in-path", default=os.path.join(DATA_DIR, "heart", "cardio_train.csv"))
    d.add_argument("--credit-in-path", default=os.path.join(DATA_DIR, "credit", "cs-training.csv"))
    d.add_argument("--cifar-root", default=os.path.join(DATA_DIR))
    d.add_argument("--cifar-m", type=int, default=16)
    d.add_argument("--cifar-block", type=int, default=4)

    args = p.parse_args()

    if args.cmd == "kanon":
        if args.cluster_repr == "tabnet":
            if args.dataset != "adult":
                raise SystemExit("tabnet clustering repr only supported for adult.")
            if args.embeddings_pkl is None or not os.path.exists(args.embeddings_pkl):
                raise SystemExit("Provide valid --embeddings-pkl (e.g., data/adult/embeddings.pkl).")
            # embeddings_runtimes.csv is optional but strongly expected
            if not _exists_nonempty(ADULT_EMB_RT_CSV):
                logger.warning("Missing %s. embed_offline/util_offline will be 0.0", ADULT_EMB_RT_CSV)

        prepare_kanon(
            dataset=args.dataset,
            k_values=args.k_values,
            cluster_repr=args.cluster_repr,
            seed=args.seed,
            embeddings_pkl=args.embeddings_pkl,
            skip_existing=args.skip_existing,
            perm_type=args.perm_type,
        )

    elif args.cmd == "dp":
        if args.dataset == "adult":
            if not _exists_nonempty(ADULT_EMB_RT_CSV):
                logger.warning("Missing %s. embed_offline/util_offline will be 0.0", ADULT_EMB_RT_CSV)

        prepare_dp(
            dataset=args.dataset,
            eps_values=args.eps,
            seed=args.seed,
            skip_existing=args.skip_existing,
            adult_in_path=args.adult_in_path,
            adult_utilities_pkl=args.adult_utilities_pkl,
            heart_in_path=args.heart_in_path,
            credit_in_path=args.credit_in_path,
            cifar_root=args.cifar_root,
            cifar_m=args.cifar_m,
            cifar_block=args.cifar_block,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Build TabNet-based category embeddings and utilities for Adult dataset.

Replaces scripts/build_adult_embeddings.py with eupg-package imports.

Always writes a simple runtime CSV inside the dataset folder (overwritten each run):
  data/adult/embeddings_runtimes.csv

Schema (3 columns):
  embedding,utility,total

Usage examples (run as module):
  python -m scripts.build_embeddings build \
    --in-path data/adult/adult.data \
    --out-embeddings data/adult/embeddings.pkl \
    --out-utilities data/adult/utilities.pkl

  # Force retrain + recompute
  python -m scripts.build_embeddings build --force

  # Skip numeric utilities
  python -m scripts.build_embeddings build --no-numeric-utils

  # Visualize one attribute (PCA)
  python -m scripts.build_embeddings visualize \
    --embeddings data/adult/embeddings.pkl \
    --col marital-status \
    --out data/adult/marital-status_pca.png
"""

from __future__ import annotations

import argparse
import csv
import os
import pickle
import random
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.pretraining import TabNetPretrainer

from eupg.utils import get_logger

logger = get_logger(__name__)

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder


# Optional plotting only when needed
def _lazy_import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")  # safe for servers/headless
    import matplotlib.pyplot as plt
    return plt


# ----------------------------
# Data + preprocessing
# ----------------------------
ADULT_COLUMNS = [
    "age", "workClass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income",
]

NUMERIC_COLS = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CAT_COLS = ["workClass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
LABEL_COL = "income"


def load_adult(in_path: str) -> pd.DataFrame:
    df = pd.read_csv(
        in_path,
        names=ADULT_COLUMNS,
        sep=r" *, *",
        engine="python",
        na_values="?",
    )
    df.dropna(inplace=True)
    df.drop(["fnlwgt", "education"], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def encode_categoricals(
    df: pd.DataFrame,
    cat_cols: List[str],
    label_col: str,
) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder], Dict[str, Dict[int, str]]]:
    """
    Label-encode categoricals and label (income). Returns:
      - df with encoded columns
      - encoders dict
      - mapping_dict[col][code] = original_string
    """
    encoders: Dict[str, LabelEncoder] = {}
    mapping_dict: Dict[str, Dict[int, str]] = {c: {} for c in cat_cols + [label_col]}

    for col in cat_cols + [label_col]:
        le = LabelEncoder()
        df[col] = df[col].astype(str).fillna("VV_likely")
        df[col] = le.fit_transform(df[col].values)
        encoders[col] = le
        for i, cls in enumerate(le.classes_):
            mapping_dict[col][i] = cls

    return df, encoders, mapping_dict


def make_train_valid_split(X_all: np.ndarray, valid_frac: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(X_all)
    m = int(valid_frac * n)
    idxs = np.arange(n)
    rng = np.random.default_rng(seed)
    rng.shuffle(idxs)
    X_valid = X_all[idxs[:m]]
    X_train = X_all[idxs[m:]]
    return X_train, X_valid


# ----------------------------
# TabNet training + extraction
# ----------------------------
def train_tabnet_pretrainer(
    X_train: np.ndarray,
    X_valid: np.ndarray,
    cat_idxs: List[int],
    cat_dims: List[int],
    cat_emb_dim: int,
    max_epochs: int,
    seed: int,
    verbose: int,
) -> TabNetPretrainer:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    model = TabNetPretrainer(
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        cat_emb_dim=cat_emb_dim,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2),
        mask_type="entmax",
        n_shared_decoder=1,
        n_indep_decoder=1,
        verbose=verbose,
    )

    model.fit(
        X_train=X_train,
        eval_set=[X_valid],
        max_epochs=max_epochs if not os.getenv("CI", False) else 2,
        patience=10,
        batch_size=2048,
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False,
        pretraining_ratio=0.5,
    )
    return model


def extract_attribute_embeddings(
    unsupervised_model: TabNetPretrainer,
    cat_cols: List[str],
    label_col: str,
    mapping_dict: Dict[str, Dict[int, str]],
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Returns attribute_embedding[col][original_value_string] = embedding_vector
    Only for CAT_COLS (excludes label and numerics).
    """
    net = unsupervised_model.network
    if not hasattr(net, "embedder") or not hasattr(net.embedder, "embeddings"):
        raise RuntimeError("Could not find TabNet embedder embeddings. Check pytorch-tabnet version.")

    # Order corresponds to order of cat_idxs passed at init time.
    # Here we expect it to be cat_cols + [label_col] (same order we pass).
    catcol_to_embmat: Dict[str, np.ndarray] = {}
    for col, emb_layer in zip(cat_cols + [label_col], net.embedder.embeddings):
        catcol_to_embmat[col] = emb_layer.weight.detach().cpu().numpy()

    attribute_embedding: Dict[str, Dict[str, np.ndarray]] = {col: {} for col in cat_cols}
    for col in cat_cols:
        emb_mat = catcol_to_embmat[col]
        for code, orig in mapping_dict[col].items():
            attribute_embedding[col][orig] = emb_mat[int(code)].copy()

    return attribute_embedding


# ----------------------------
# Utilities computation
# ----------------------------
def compute_categorical_utilities(
    attribute_embedding: Dict[str, Dict[str, np.ndarray]],
    cat_cols: List[str],
) -> Dict[str, List[List[Any]]]:
    utility_dict: Dict[str, List[List[Any]]] = {}
    for col in cat_cols:
        vals = list(attribute_embedding[col].keys())
        util_list: List[List[Any]] = []

        # pre-stack and cosine sim matrix
        E = np.stack([attribute_embedding[col][v] for v in vals], axis=0)  # (k, d)
        norms = np.linalg.norm(E, axis=1, keepdims=True) + 1e-12
        En = E / norms
        S = En @ En.T  # cosine in [-1, 1]

        k = len(vals)
        for i in range(k):
            for j in range(i + 1, k):
                u = (float(S[i, j]) + 1.0) / 2.0  # map [-1,1] -> [0,1]
                util_list.append([vals[i], vals[j], float(u)])

        utility_dict[col] = util_list
    return utility_dict


def compute_numeric_utilities(df: pd.DataFrame, numeric_cols: List[str]) -> Dict[str, List[List[float]]]:
    utility_dict: Dict[str, List[List[float]]] = {}
    for col in numeric_cols:
        uniq = np.unique(df[col].astype(float).values)
        max_a = float(np.max(uniq)) if len(uniq) > 0 else 0.0
        denom = max_a if max_a > 0 else 1.0

        util_list: List[List[float]] = []
        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                u = (denom - abs(float(uniq[i]) - float(uniq[j]))) / denom
                util_list.append([float(uniq[i]), float(uniq[j]), float(u)])

        utility_dict[col] = util_list
    return utility_dict


# ----------------------------
# Persistence helpers
# ----------------------------
def save_pickle(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: str) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def exists_nonempty(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


# ----------------------------
# Simple runtime CSV (always overwrite)
# ----------------------------
def _dataset_dir_from_in_path(in_path: str) -> str:
    # In this repo this is typically "data/adult/adult.data" -> "data/adult"
    return os.path.dirname(os.path.abspath(in_path)) or "."


def write_simple_runtime_csv(dataset_dir: str, embedding_sec: float, utility_sec: float, total_sec: float) -> str:
    """
    Writes (overwrites) {dataset_dir}/embeddings_runtimes.csv with header:
      embedding,utility,total
    """
    os.makedirs(dataset_dir, exist_ok=True)
    out_csv = os.path.join(dataset_dir, "embeddings_runtimes.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["embedding", "utility", "total"])
        w.writerow([f"{embedding_sec:.6f}", f"{utility_sec:.6f}", f"{total_sec:.6f}"])
    return out_csv


# ----------------------------
# Main pipeline
# ----------------------------
def build_adult_tabnet_utilities(
    in_path: str,
    out_embeddings_path: str,
    out_utilities_path: str,
    seed: int,
    cat_emb_dim: int,
    max_epochs: int,
    valid_frac: float,
    include_numeric_utils: bool,
    force: bool,
    verbose: int,
) -> Dict[str, Any]:
    """
    Build and cache:
      - embeddings.pkl: attribute_embedding[col][category_string] -> np.ndarray
      - utilities.pkl:  utility_dict[col] -> list of [val_i, val_j, utility]

    Always writes a simple runtime CSV (overwritten) in the dataset folder:
      <dataset_dir>/embeddings_runtimes.csv

    Times:
      embedding_sec = time spent to (train TabNet + extract + save embeddings)
                      OR 0 if embeddings loaded from cache
      utility_sec   = time spent to compute + save utilities
                      OR 0 if utilities existed and not recomputed
      total_sec     = full wall-clock runtime of this call
    """
    t_total0 = time.time()

    status = "built"
    did_train = False
    did_util = False

    # seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    # Load data (always; needed for util numeric and for label encoders if retraining)
    logger.info("[1/6] Loading Adult dataset...")
    df = load_adult(in_path)

    logger.info("[2/6] Encoding categoricals...")
    df_enc, encoders, mapping_dict = encode_categoricals(df.copy(), CAT_COLS, LABEL_COL)

    features = NUMERIC_COLS + CAT_COLS + [LABEL_COL]
    X_all = df_enc[features].values

    cat_idxs = [features.index(c) for c in CAT_COLS + [LABEL_COL]]
    cat_dims = [len(encoders[c].classes_) for c in CAT_COLS + [LABEL_COL]]

    # ------------------
    # Embeddings stage timing
    # ------------------
    embedding_sec = 0.0
    attribute_embedding = None

    # Try cache
    if (not force) and exists_nonempty(out_embeddings_path):
        logger.info("[cache] Loading embeddings from %s", out_embeddings_path)
        attribute_embedding = load_pickle(out_embeddings_path)

    if attribute_embedding is None:
        t_emb0 = time.time()

        logger.info("[3/6] Creating train/valid split...")
        X_train, X_valid = make_train_valid_split(X_all, valid_frac=valid_frac, seed=seed)

        logger.info("[4/6] Training TabNetPretrainer (unsupervised)...")
        unsup = train_tabnet_pretrainer(
            X_train=X_train,
            X_valid=X_valid,
            cat_idxs=cat_idxs,
            cat_dims=cat_dims,
            cat_emb_dim=cat_emb_dim,
            max_epochs=max_epochs,
            seed=seed,
            verbose=verbose,
        )
        did_train = True

        logger.info("[5/6] Extracting and saving attribute embeddings...")
        attribute_embedding = extract_attribute_embeddings(
            unsupervised_model=unsup,
            cat_cols=CAT_COLS,
            label_col=LABEL_COL,
            mapping_dict=mapping_dict,
        )
        save_pickle(attribute_embedding, out_embeddings_path)
        logger.info("Saved embeddings to: %s", out_embeddings_path)

        embedding_sec = time.time() - t_emb0

    # ------------------
    # Utilities stage timing
    # ------------------
    utility_sec = 0.0
    if (not force) and exists_nonempty(out_utilities_path):
        logger.info("[cache] Utilities already exist at %s (use --force to recompute).", out_utilities_path)
    else:
        t_u0 = time.time()

        logger.info("[6/6] Computing utilities...")
        utility_dict = compute_categorical_utilities(attribute_embedding, CAT_COLS)

        if include_numeric_utils:
            num_utils = compute_numeric_utilities(df, NUMERIC_COLS)
            utility_dict.update(num_utils)

        save_pickle(utility_dict, out_utilities_path)
        did_util = True
        logger.info("Saved utilities to: %s", out_utilities_path)

        utility_sec = time.time() - t_u0

    total_sec = time.time() - t_total0
    logger.info("Done. Total runtime: %.2fs", total_sec)

    # Always overwrite a simple runtime CSV in the dataset folder
    dataset_dir = _dataset_dir_from_in_path(in_path)
    rt_csv = write_simple_runtime_csv(dataset_dir, embedding_sec, utility_sec, total_sec)
    logger.info("Saved runtimes to: %s", rt_csv)

    return {
        "status": status if (did_train or did_util) else "skipped",
        "embeddings": out_embeddings_path,
        "utilities": out_utilities_path,
        "embedding_sec": float(embedding_sec),
        "utility_sec": float(utility_sec),
        "total_sec": float(total_sec),
        "trained_tabnet": bool(did_train),
        "computed_utilities": bool(did_util),
        "runtime_csv": rt_csv,
    }


# ----------------------------
# Visualization
# ----------------------------
def visualize_embeddings_pca(
    embeddings_path: str,
    col: str,
    out_path: str | None,
    figsize: Tuple[int, int] = (9, 7),
) -> None:
    attribute_embedding = load_pickle(embeddings_path)
    if col not in attribute_embedding:
        raise ValueError(f"Column '{col}' not in embeddings. Available: {list(attribute_embedding.keys())}")

    embs_dict = attribute_embedding[col]
    labels = list(embs_dict.keys())
    E = np.stack([embs_dict[k] for k in labels], axis=0)

    pca = PCA(n_components=2)
    R = pca.fit_transform(E)

    plt = _lazy_import_matplotlib()
    plt.figure(figsize=figsize)
    plt.scatter(R[:, 0], R[:, 1], c="brown")

    for i, name in enumerate(labels):
        plt.annotate(name, (R[i, 0], R[i, 1]), fontsize=8)

    plt.xlabel("PC 1")
    plt.ylabel("PC 2")
    plt.title(f"{col} embeddings (PCA)")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200)
        logger.info("Saved plot to: %s", out_path)
    else:
        plt.show()


# ----------------------------
# CLI
# ----------------------------
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Adult TabNet embeddings + utilities builder")
    sub = p.add_subparsers(dest="command", required=True)

    p_build = sub.add_parser("build", help="Train TabNet (if needed) and save embeddings/utilities")
    p_build.add_argument("--in-path", default="data/adult/adult.data")
    p_build.add_argument("--out-embeddings", default="data/adult/embeddings.pkl")
    p_build.add_argument("--out-utilities", default="data/adult/utilities.pkl")
    p_build.add_argument("--seed", type=int, default=0)
    p_build.add_argument("--cat-emb-dim", type=int, default=10)
    p_build.add_argument("--max-epochs", type=int, default=200)
    p_build.add_argument("--valid-frac", type=float, default=0.1)
    p_build.add_argument("--no-numeric-utils", action="store_true", help="Do not compute numeric utilities")
    p_build.add_argument("--force", action="store_true", help="Force retrain/recompute even if cache exists")
    p_build.add_argument("--verbose", type=int, default=10)

    p_vis = sub.add_parser("visualize", help="PCA visualization for one categorical attribute")
    p_vis.add_argument("--embeddings", default="data/adult/embeddings.pkl")
    p_vis.add_argument("--col", required=True, help="e.g., marital-status, occupation, race, ...")
    p_vis.add_argument("--out", default=None, help="Output PNG path; if omitted, will try to show()")

    return p


def main():
    args = make_parser().parse_args()

    if args.command == "build":
        build_adult_tabnet_utilities(
            in_path=args.in_path,
            out_embeddings_path=args.out_embeddings,
            out_utilities_path=args.out_utilities,
            seed=args.seed,
            cat_emb_dim=args.cat_emb_dim,
            max_epochs=args.max_epochs,
            valid_frac=args.valid_frac,
            include_numeric_utils=(not args.no_numeric_utils),
            force=args.force,
            verbose=args.verbose,
        )

    elif args.command == "visualize":
        visualize_embeddings_pca(
            embeddings_path=args.embeddings,
            col=args.col,
            out_path=args.out,
        )


if __name__ == "__main__":
    main()

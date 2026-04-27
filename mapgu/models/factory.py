"""Model factory: get_model() for vision, ModelFactory for tabular/vision, and seeding utilities."""

from __future__ import annotations

import functools
import os
import random
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from mapgu.config import ADULT_CAT_COLS, ADULT_NUM_COLS
from mapgu.models.architectures import MLPModel
from mapgu.utils import get_logger

logger = get_logger(__name__)


# -----------------------------------------------------------------------------
# Reproducibility
# -----------------------------------------------------------------------------
def seed_everything(seed: int = 7, deterministic: bool = False) -> None:
    np.random.seed(int(seed))
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    if deterministic:
        # Required for fully deterministic CUDA ops (cuBLAS, cuDNN)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass


# -----------------------------------------------------------------------------
# Clustering-only repr helpers (kept for API compatibility)
# -----------------------------------------------------------------------------

def build_cluster_repr_onehot(X_train_transformed: np.ndarray) -> np.ndarray:
    return X_train_transformed


def build_cluster_repr_tabnet(
    X_train_raw_df: pd.DataFrame,
    embeddings_pkl_path: str,
    cat_cols=ADULT_CAT_COLS,
    num_cols=ADULT_NUM_COLS,
) -> np.ndarray:
    import pickle

    with open(embeddings_pkl_path, "rb") as f:
        emb_dict = pickle.load(f)  # {col: {value: vec}}

    X_num = X_train_raw_df[num_cols].astype(float).to_numpy()
    X_num = StandardScaler().fit_transform(X_num).astype(np.float32)

    emb_blocks = []
    for col in cat_cols:
        if col not in emb_dict:
            raise KeyError(f"Column '{col}' not found in embeddings.pkl keys={list(emb_dict.keys())[:10]}...")
        col_map = emb_dict[col]
        mean_vec = np.mean(np.stack(list(col_map.values()), axis=0), axis=0).astype(np.float32)

        dim = int(mean_vec.shape[0])
        block = np.zeros((len(X_train_raw_df), dim), dtype=np.float32)

        vals = X_train_raw_df[col].astype(str).to_numpy()
        for i, v in enumerate(vals):
            block[i] = col_map.get(v, mean_vec)

        emb_blocks.append(block)

    X_cat = np.concatenate(emb_blocks, axis=1) if emb_blocks else np.zeros((len(X_train_raw_df), 0), dtype=np.float32)
    return np.concatenate([X_num, X_cat], axis=1)


# -----------------------------------------------------------------------------
# XGBoost GPU support probing + safe params
# -----------------------------------------------------------------------------

@functools.lru_cache(maxsize=None)
def _xgb_supports_gpu_hist() -> bool:
    try:
        X = np.random.randn(32, 4).astype(np.float32)
        y = (np.random.rand(32) > 0.5).astype(np.int32)
        clf = XGBClassifier(
            n_estimators=1,
            max_depth=2,
            tree_method="gpu_hist",
            eval_metric="logloss",
            verbosity=0,
        )
        clf.fit(X, y)
        return True
    except Exception:
        return False


def _xgb_common_params(prefer_gpu: bool = True) -> dict:
    common = dict(
        reg_lambda=5,
        learning_rate=0.5,
        colsample_bytree=0.9,
        eval_metric="logloss",
        n_jobs=os.cpu_count() or 1,
    )
    use_gpu = bool(prefer_gpu) and torch.cuda.is_available() and _xgb_supports_gpu_hist()
    if use_gpu:
        common.update(dict(tree_method="gpu_hist", predictor="gpu_predictor"))
    else:
        common.update(dict(tree_method="hist"))
    return common


# -----------------------------------------------------------------------------
# Vision model factory (timm-based)
# -----------------------------------------------------------------------------

def get_model(architecture: str = 'resnet18', num_classes: int = 10) -> nn.Module:
    """
    Create a model. Tabular: 'mlp'. Vision: any timm name ('resnet18', 'densenet', etc.).
    CIFAR-10 stem optimizations (3x3 conv, no initial pool) are applied automatically.
    """
    if architecture == 'mlp':
        model = MLPModel(input_size=256, hidden_size=256, output_size=num_classes)
    elif architecture == 'densenet':
        model = timm.create_model("densenet121", num_classes=num_classes, in_chans=3)
    elif architecture == 'resnet18':
        model = timm.create_model(architecture, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    # EfficientNet-style stem
    if hasattr(model, 'conv_stem'):
        model.conv_stem = nn.Conv2d(
            3, model.conv_stem.out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )

    # DenseNet CIFAR stem (timm DenseNet modules)
    if hasattr(model, 'features'):
        features = model.features
        if hasattr(features, 'conv0') and isinstance(features.conv0, nn.Conv2d):
            features.conv0 = nn.Conv2d(
                3, features.conv0.out_channels,
                kernel_size=3, stride=1, padding=1, bias=False,
            )
        if hasattr(features, 'pool0'):
            features.pool0 = nn.Identity()

    # ResNet CIFAR stem: replace 7x7 + maxpool with 3x3
    if hasattr(model, 'conv1'):
        model.conv1 = nn.Conv2d(
            3, model.conv1.out_channels,
            kernel_size=3, stride=1, padding=1, bias=False,
        )
        if hasattr(model, 'maxpool'):
            model.maxpool = nn.Identity()

    return model


# -----------------------------------------------------------------------------
# Model factory (tabular + vision)
# -----------------------------------------------------------------------------

class ModelFactory:
    @staticmethod
    def create_model(
        model_type: str,
        dataset: str,
        num_features: Optional[int] = None,
        num_classes: Optional[int] = None,
        xgb_n_estimators: Optional[int] = None,
        xgb_max_depth: Optional[int] = None,
        xgb_lr: Optional[float] = None,
        xgb_reg_lambda: Optional[float] = None,
    ):
        if model_type == "mlp":
            hidden_size = 256 if dataset == "credit" else 128
            return MLPModel(num_features, hidden_size, num_classes)

        if model_type == "xgboost":
            common = _xgb_common_params(prefer_gpu=True)
            if xgb_lr is not None:
                common["learning_rate"] = float(xgb_lr)
            if xgb_reg_lambda is not None:
                common["reg_lambda"] = float(xgb_reg_lambda)

            if dataset == "adult":
                return XGBClassifier(
                    max_depth=int(xgb_max_depth) if xgb_max_depth is not None else 10,
                    n_estimators=int(xgb_n_estimators) if xgb_n_estimators is not None else 300,
                    **common,
                )
            if dataset == "credit":
                return XGBClassifier(
                    max_depth=int(xgb_max_depth) if xgb_max_depth is not None else 9,
                    n_estimators=int(xgb_n_estimators) if xgb_n_estimators is not None else 200,
                    **common,
                )
            if dataset == "heart":
                return XGBClassifier(
                    max_depth=int(xgb_max_depth) if xgb_max_depth is not None else 7,
                    n_estimators=int(xgb_n_estimators) if xgb_n_estimators is not None else 200,
                    **common,
                )

        if model_type == "densenet":
            return get_model("densenet", num_classes=num_classes)

        if model_type == "resnet18":
            return get_model("resnet18", num_classes=num_classes)

        raise ValueError(f"Unknown model type: {model_type}")

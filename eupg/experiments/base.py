"""Base PrivacyBenchmark class shared by all experiment runners."""
from __future__ import annotations

import copy
import os
import random
import time
from typing import Dict, Optional, Tuple, List, Any, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Subset, TensorDataset

from eupg.config import DATA_DIR, DEFAULT_SEED, MIA_RESAMPLES, MIA_EVAL_CAP, RESULTS_DIR
from eupg.data.loaders import DatasetLoader
from eupg.models.factory import ModelFactory, seed_everything, get_model
from eupg.training.trainer import train_model
from eupg.evaluation.metrics import accuracy as _accuracy, auc_score as _auc_score, compute_attack_components
from eupg.evaluation.attacks import mia_attack, tf_attack, rmia_attack, lira_scaled_logit_score
from eupg.utils import get_logger, save_metrics_csv, save_summary_csv, save_config_yaml, log_metrics_table, _ms, append_runtime_rows

logger = get_logger(__name__)



# -----------------------------------------------------------------------------#
# Optional clustering-only repr helpers (kept for API compatibility)
# -----------------------------------------------------------------------------#
ADULT_CAT_COLS = ["workClass", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
ADULT_NUM_COLS = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]


def build_cluster_repr_onehot(X_train_transformed: np.ndarray) -> np.ndarray:
    return X_train_transformed


def build_cluster_repr_tabnet(
    X_train_raw_df,
    embeddings_pkl_path: str,
    cat_cols=ADULT_CAT_COLS,
    num_cols=ADULT_NUM_COLS,
) -> np.ndarray:
    import pickle
    from sklearn.preprocessing import StandardScaler as _StandardScaler

    with open(embeddings_pkl_path, "rb") as f:
        emb_dict = pickle.load(f)

    X_num = X_train_raw_df[num_cols].astype(float).to_numpy()
    X_num = _StandardScaler().fit_transform(X_num).astype(np.float32)

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


# -----------------------------------------------------------------------------#
# Benchmark
# -----------------------------------------------------------------------------#
class PrivacyBenchmark:
    def __init__(
        self,
        dataset: str,
        model_type: str,
        forget_ratios: List[float],
        n_repeat: int = 3,
        max_epochs: int = 100,
        results_subdir: Optional[str] = None,
        kanon_cluster_repr: str = "onehot",
        kanon_perm_type: str = "rowwise",
        embeddings_pkl: Optional[str] = None,
        seed: int = DEFAULT_SEED,
        cifar_download: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        deterministic: bool = False,
        optimizer_name: str = "auto",
        scheduler_name: str = "none",
        ft_optimizer_name: Optional[str] = None,
        ft_scheduler_name: Optional[str] = None,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        adam_beta1: float = 0.9,
        adam_beta2: float = 0.999,
        adam_eps: float = 1e-8,
        scheduler_step_size: int = 30,
        scheduler_gamma: float = 0.1,
        onecycle_pct_start: float = 0.3,
        epoch_metrics: bool = False,
        batch_size: Optional[int] = None,
        lr: Optional[float] = None,
        xgb_n_estimators: Optional[int] = None,
        xgb_max_depth: Optional[int] = None,
        xgb_lr: Optional[float] = None,
        xgb_reg_lambda: Optional[float] = None,
        mia_attacks: Optional[List[str]] = None,
        rmia_n_ref: int = 1,
        use_amp: Optional[bool] = None,
    ):
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        # Auto-enable AMP for neural-net models on CUDA when the caller didn't
        # explicitly opt in or out.  XGBoost ignores use_amp entirely.
        _is_nn = model_type in {"mlp", "densenet", "resnet18"}
        if use_amp is None:
            self.use_amp = bool(torch.cuda.is_available() and _is_nn)
        else:
            self.use_amp = bool(use_amp)

        self.dataset = dataset
        self.model_type = model_type
        self.forget_ratios = list(forget_ratios)
        self.n_repeat = int(n_repeat)
        self.max_epochs = int(max_epochs)
        self.results_subdir = str(results_subdir).strip() if results_subdir is not None else None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = OneHotEncoder(sparse_output=False, categories="auto")

        self.kanon_cluster_repr = kanon_cluster_repr
        self.kanon_perm_type = str(kanon_perm_type)
        self.embeddings_pkl = embeddings_pkl

        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.cifar_download = bool(cifar_download)
        self.optimizer_name = str(optimizer_name).lower()
        self.scheduler_name = str(scheduler_name).lower()
        self.ft_optimizer_name = str(ft_optimizer_name).lower() if ft_optimizer_name is not None else self.optimizer_name
        self.ft_scheduler_name = str(ft_scheduler_name).lower() if ft_scheduler_name is not None else self.scheduler_name
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)
        self.adam_beta1 = float(adam_beta1)
        self.adam_beta2 = float(adam_beta2)
        self.adam_eps = float(adam_eps)
        self.scheduler_step_size = int(scheduler_step_size)
        self.scheduler_gamma = float(scheduler_gamma)
        self.onecycle_pct_start = float(onecycle_pct_start)
        self.epoch_metrics = bool(epoch_metrics)
        self.batch_size_override = int(batch_size) if batch_size is not None else None
        self.lr_override = float(lr) if lr is not None else None
        self.xgb_n_estimators = int(xgb_n_estimators) if xgb_n_estimators is not None else None
        self.xgb_max_depth = int(xgb_max_depth) if xgb_max_depth is not None else None
        self.xgb_lr = float(xgb_lr) if xgb_lr is not None else None
        self.xgb_reg_lambda = float(xgb_reg_lambda) if xgb_reg_lambda is not None else None
        _valid = {'loss', 'scaled_logit', 'rmia'}
        self.mia_attacks = list(mia_attacks) if mia_attacks else ['loss']
        unknown = set(self.mia_attacks) - _valid
        if unknown:
            raise ValueError(f"Unknown mia_attacks: {unknown}. Valid: {_valid}")
        self.rmia_n_ref = max(1, int(rmia_n_ref))
        self._ref_model_cache: Dict[Tuple[float, int], List[nn.Module]] = {}

        # Split cache: ensures baseline + retrain reuse same split for same repeat r
        self._split_cache: Dict[Tuple[float, int], Tuple[np.ndarray, np.ndarray]] = {}

        # Seed global RNG before data loading and model init so weights are reproducible
        seed_everything(self.seed, deterministic=self.deterministic)

        # Load dataset
        self._load_data(cifar_download=cifar_download)

        # Setup model
        self._setup_model()

        # Results dir
        results_root = os.path.normpath(RESULTS_DIR)
        if self.results_subdir:
            if os.path.isabs(self.results_subdir):
                raise ValueError("--results_subdir must be a relative subdirectory under results/")
            candidate_root = os.path.normpath(os.path.join(results_root, self.results_subdir))
            if os.path.commonpath([candidate_root, results_root]) != results_root:
                raise ValueError("--results_subdir must stay within the results/ directory")
            results_root = candidate_root
        self.results_root = results_root
        self.dataset_results_dir = os.path.join(self.results_root, dataset)
        os.makedirs(self.dataset_results_dir, exist_ok=True)

        # Cache CIFAR full loaders (avoid recreating them)
        self._cifar_full_train_loader: Optional[DataLoader] = None
        self._cifar_test_loader: Optional[DataLoader] = None
        if not self.is_tabular:
            self._cifar_full_train_loader = self._make_cifar_loader(self.trainset, shuffle=True)
            self._cifar_test_loader = self._make_cifar_loader(self.testset, shuffle=False)

    def _config_identity(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "model": self.model_type,
            "seed": int(self.seed),
            "n_repeat": int(self.n_repeat),
            "results_subdir": self.results_subdir,
            "forget_ratios": [float(fr) for fr in self.forget_ratios],
        }

    def _config_training(self) -> Dict[str, Any]:
        return {
            "max_epochs": int(self.max_epochs) if getattr(self, "max_epochs", None) is not None else None,
            "batch_size": int(self.batch_size) if getattr(self, "batch_size", None) is not None else None,
            "optimizer": getattr(self, "optimizer_name", None),
            "scheduler": getattr(self, "scheduler_name", None),
            "ft_optimizer": getattr(self, "ft_optimizer_name", None),
            "ft_scheduler": getattr(self, "ft_scheduler_name", None),
            "lr": float(self.lr) if getattr(self, "lr", None) is not None else None,
            "momentum": float(self.momentum) if getattr(self, "momentum", None) is not None else None,
            "weight_decay": float(self.weight_decay) if getattr(self, "weight_decay", None) is not None else None,
            "adam_beta1": float(self.adam_beta1) if getattr(self, "adam_beta1", None) is not None else None,
            "adam_beta2": float(self.adam_beta2) if getattr(self, "adam_beta2", None) is not None else None,
            "adam_eps": float(self.adam_eps) if getattr(self, "adam_eps", None) is not None else None,
            "scheduler_step_size": int(self.scheduler_step_size) if getattr(self, "scheduler_step_size", None) is not None else None,
            "scheduler_gamma": float(self.scheduler_gamma) if getattr(self, "scheduler_gamma", None) is not None else None,
            "onecycle_pct_start": float(self.onecycle_pct_start) if getattr(self, "onecycle_pct_start", None) is not None else None,
            "epoch_metrics": bool(self.epoch_metrics) if getattr(self, "epoch_metrics", None) is not None else None,
            "use_amp": bool(self.use_amp) if getattr(self, "use_amp", None) is not None else None,
        }

    def _config_runtime(self) -> Dict[str, Any]:
        return {
            "num_workers": int(self.num_workers) if getattr(self, "num_workers", None) is not None else None,
            "pin_memory": bool(self.pin_memory) if getattr(self, "pin_memory", None) is not None else None,
            "deterministic": bool(self.deterministic) if getattr(self, "deterministic", None) is not None else None,
            "cifar_download": bool(self.cifar_download) if getattr(self, "cifar_download", None) is not None else None,
        }

    def _config_xgboost(self) -> Dict[str, Any]:
        return {
            "n_estimators": int(self.xgb_n_estimators) if getattr(self, "xgb_n_estimators", None) is not None else None,
            "max_depth": int(self.xgb_max_depth) if getattr(self, "xgb_max_depth", None) is not None else None,
            "learning_rate": float(self.xgb_lr) if getattr(self, "xgb_lr", None) is not None else None,
            "reg_lambda": float(self.xgb_reg_lambda) if getattr(self, "xgb_reg_lambda", None) is not None else None,
        }

    def _config_mia(self) -> Dict[str, Any]:
        return {
            "attacks": list(self.mia_attacks) if getattr(self, "mia_attacks", None) is not None else None,
            "rmia_n_ref": int(self.rmia_n_ref) if getattr(self, "rmia_n_ref", None) is not None else None,
            "mia_resamples": int(self.mia_resamples) if getattr(self, "mia_resamples", None) is not None else None,
            "mia_eval_cap": int(self.mia_eval_cap) if getattr(self, "mia_eval_cap", None) is not None else None,
        }

    # -----------------------------
    # TrainResult unwrapping (FIX)
    # -----------------------------
    def _unwrap_trained_model(self, train_output: Any) -> nn.Module:
        """
        train.train_model returns TrainResult in your train.py.
        This function accepts either:
          - nn.Module (old behavior), or
          - TrainResult-like object with .model
        and returns an nn.Module on self.device.
        """
        if isinstance(train_output, nn.Module):
            return train_output.to(self.device)

        model = getattr(train_output, "model", None)
        if not isinstance(model, nn.Module):
            raise TypeError(
                "train_model output is neither nn.Module nor TrainResult-like with .model. "
                f"Got type={type(train_output)}"
            )
        return model.to(self.device)

    # -----------------------------
    # Loader helpers
    # -----------------------------
    def _loader_kwargs(self, *, shuffle: bool) -> dict:
        return dict(
            batch_size=self.batch_size,
            shuffle=bool(shuffle),
            num_workers=int(self.num_workers),
            pin_memory=bool(self.pin_memory),
            persistent_workers=bool(self.num_workers > 0),
        )

    def _make_cifar_loader(self, ds, *, shuffle: bool) -> DataLoader:
        return DataLoader(ds, **self._loader_kwargs(shuffle=shuffle))

    def _create_data_loaders(self) -> None:
        self.train_dataset = TensorDataset(
            torch.tensor(self.X_train, dtype=torch.float32),
            torch.tensor(self.y_train, dtype=torch.int64),
        )
        self.test_dataset = TensorDataset(
            torch.tensor(self.X_test, dtype=torch.float32),
            torch.tensor(self.y_test, dtype=torch.int64),
        )
        self.train_loader = DataLoader(self.train_dataset, **self._loader_kwargs(shuffle=True))
        self.test_loader = DataLoader(self.test_dataset, **self._loader_kwargs(shuffle=False))

    # -----------------------------
    # Data loading
    # -----------------------------
    def _load_data(self, cifar_download: bool = True) -> None:
        self.X_train_raw_df = None
        self.X_test_raw_df = None

        if self.dataset == "adult":
            out = DatasetLoader.load_adult(return_raw=True)
            (
                self.X_train,
                self.y_train,
                self.X_test,
                self.y_test,
                self.preprocessor,
                self.X_train_raw_df,
                self.X_test_raw_df,
            ) = out
            self.batch_size = 512
            self.is_tabular = True

        elif self.dataset == "credit":
            self.X_train, self.y_train, self.X_test, self.y_test, self.preprocessor = DatasetLoader.load_credit()
            self.batch_size = 256
            self.is_tabular = True

        elif self.dataset == "heart":
            self.X_train, self.y_train, self.X_test, self.y_test, self.preprocessor = DatasetLoader.load_heart()
            self.batch_size = 512
            self.is_tabular = True

        elif self.dataset == "cifar10":
            self.trainset, self.testset = DatasetLoader.load_cifar10(download=cifar_download)
            self.y_train = self.trainset.targets
            self.batch_size = 128
            self.is_tabular = False

        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        if self.is_tabular:
            self.num_features = int(self.X_train.shape[1])
            self.num_classes = int(len(set(self.y_train)))
            if self.batch_size_override is not None:
                self.batch_size = int(self.batch_size_override)
            # Fit encoder once so _compute_mia can use transform() without refitting each call.
            self.encoder.fit(self.y_train.reshape(-1, 1))
            self._create_data_loaders()
        else:
            self.num_features = None
            self.num_classes = 10
            if self.batch_size_override is not None:
                self.batch_size = int(self.batch_size_override)

    # -----------------------------
    # Model / criterion setup
    # -----------------------------
    def _setup_model(self) -> None:
        self.initial_model = ModelFactory.create_model(
            self.model_type,
            self.dataset,
            self.num_features,
            self.num_classes,
            xgb_n_estimators=self.xgb_n_estimators,
            xgb_max_depth=self.xgb_max_depth,
            xgb_lr=self.xgb_lr,
            xgb_reg_lambda=self.xgb_reg_lambda,
        )
        # Cache initial weights on CPU so _fresh_model() avoids deep-copying the whole object graph.
        if isinstance(self.initial_model, nn.Module):
            self._initial_state_dict: Optional[dict] = {
                k: v.cpu().clone() for k, v in self.initial_model.state_dict().items()
            }
        else:
            self._initial_state_dict = None

        if self.model_type == "mlp":
            self.lr = float(self.lr_override) if self.lr_override is not None else (1e-3 if self.dataset == "credit" else 1e-2)
            self.criterion = nn.CrossEntropyLoss()

        elif self.model_type in ["densenet", "resnet18"]:
            self.lr = float(self.lr_override) if self.lr_override is not None else 0.1
            self.criterion = nn.CrossEntropyLoss()

    def _fresh_model(self):
        """Return a fresh model with the initial weights.

        For nn.Module models this is faster than deepcopy because it only
        copies tensors (via load_state_dict), not the full Python object graph.
        For non-nn.Module (XGBoost) deepcopy is the correct path.
        """
        if self._initial_state_dict is not None:
            m = ModelFactory.create_model(
                self.model_type,
                self.dataset,
                self.num_features,
                self.num_classes,
                xgb_n_estimators=self.xgb_n_estimators,
                xgb_max_depth=self.xgb_max_depth,
                xgb_lr=self.xgb_lr,
                xgb_reg_lambda=self.xgb_reg_lambda,
            )
            m.load_state_dict(self._initial_state_dict)
            return m
        return copy.deepcopy(self.initial_model)

    def _resolve_optimizer_name(self, name: str) -> str:
        n = str(name).lower()
        if n == "auto":
            return "adam" if self.model_type == "mlp" else "sgd"
        if n not in {"adam", "sgd"}:
            raise ValueError(f"Unsupported optimizer={name}. Use one of: auto, adam, sgd.")
        return n

    def _build_torch_optimizer(self, model: nn.Module, lr: float, *, optimizer_name: str) -> optim.Optimizer:
        opt_name = self._resolve_optimizer_name(optimizer_name)
        if opt_name == "adam":
            return optim.Adam(
                model.parameters(),
                lr=float(lr),
                weight_decay=float(self.weight_decay),
                betas=(float(self.adam_beta1), float(self.adam_beta2)),
                eps=float(self.adam_eps),
                foreach=False,
            )
        return optim.SGD(
            model.parameters(),
            lr=float(lr),
            momentum=float(self.momentum),
            weight_decay=float(self.weight_decay),
        )

    def _build_torch_scheduler(
        self,
        optimizer: optim.Optimizer,
        *,
        scheduler_name: str,
        epochs: int,
        steps_per_epoch: int,
        max_lr: float,
    ) -> Tuple[Optional[optim.lr_scheduler._LRScheduler], bool]:
        name = str(scheduler_name).lower()
        if name in {"none", "constant"}:
            return None, True
        if name == "step":
            return optim.lr_scheduler.StepLR(
                optimizer,
                step_size=max(1, int(self.scheduler_step_size)),
                gamma=float(self.scheduler_gamma),
            ), False
        if name == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(epochs))), False
        if name == "onecycle":
            total_steps = max(1, int(epochs) * max(1, int(steps_per_epoch)))
            return optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(max_lr),
                total_steps=total_steps,
                pct_start=float(self.onecycle_pct_start),
                anneal_strategy="cos",
                cycle_momentum=False,
            ), True
        raise ValueError("Unsupported scheduler. Use one of: none, step, cosine, onecycle.")

    # -----------------------------
    # Split (seed + run_id), cached
    # -----------------------------
    def _get_split_indices(self, forget_ratio: float, run_id: int) -> Tuple[np.ndarray, np.ndarray]:
        key = (float(forget_ratio), int(run_id))
        if key in self._split_cache:
            return self._split_cache[key]

        n = len(self.y_train) if self.is_tabular else len(self.trainset)
        m = int(n * float(forget_ratio))
        idxs = np.arange(n)

        rng = np.random.default_rng(int(self.seed) + int(run_id))
        rng.shuffle(idxs)

        forget_idxs = idxs[:m]
        retain_idxs = idxs[m:]

        self._split_cache[key] = (retain_idxs, forget_idxs)
        return retain_idxs, forget_idxs

    def _split_forget_retain(self, forget_ratio: float, run_id: int) -> Tuple[Any, Any, Any, Any, DataLoader, DataLoader, int]:
        retain_idxs, forget_idxs = self._get_split_indices(forget_ratio, run_id)
        m = int(len(forget_idxs))

        if self.is_tabular:
            X_retain = self.X_train[retain_idxs]
            y_retain = self.y_train[retain_idxs]
            X_forget = self.X_train[forget_idxs]
            y_forget = self.y_train[forget_idxs]

            retain_ds = TensorDataset(torch.tensor(X_retain, dtype=torch.float32), torch.tensor(y_retain, dtype=torch.int64))
            forget_ds = TensorDataset(torch.tensor(X_forget, dtype=torch.float32), torch.tensor(y_forget, dtype=torch.int64))

            retain_loader = DataLoader(retain_ds, **self._loader_kwargs(shuffle=True))
            forget_loader = DataLoader(forget_ds, **self._loader_kwargs(shuffle=False))
            return X_retain, y_retain, X_forget, y_forget, retain_loader, forget_loader, m

        # CIFAR-10
        retain_set = Subset(self.trainset, retain_idxs)
        forget_set = Subset(self.trainset, forget_idxs)

        retain_loader = DataLoader(retain_set, **self._loader_kwargs(shuffle=True))
        forget_loader = DataLoader(forget_set, **self._loader_kwargs(shuffle=False))
        return None, None, None, None, retain_loader, forget_loader, m

    # -----------------------------
    # Metrics wrappers (tolerate utils signatures with/without device kwarg)
    # -----------------------------
    def _acc(self, model, loader) -> float:
        try:
            return float(_accuracy(model, loader, device=self.device))
        except TypeError:
            return float(_accuracy(model, loader))

    def _auc(self, model, loader) -> float:
        try:
            return float(_auc_score(model, loader, device=self.device))
        except TypeError:
            return float(_auc_score(model, loader))

    def _extract_true_class_probs(self, model: nn.Module, loader: DataLoader) -> np.ndarray:
        """Return softmax P(y_true | x) for each sample in loader."""
        model.eval()
        out = []
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)
                p = torch.softmax(model(x), dim=1)
                out.append(p[torch.arange(len(y)), y].cpu().numpy())
        return np.concatenate(out)

    def _get_reference_models(self, retain_loader: DataLoader, forget_ratio: float, run_id: int) -> List[nn.Module]:
        """Train (or retrieve cached) reference models on D_remain for RMIA.

        Trains self.rmia_n_ref independent models with different seeds.
        Each uses a unique seed: self.seed + run_id * 10000 + i.
        """
        key = (float(forget_ratio), int(run_id))
        cached = self._ref_model_cache.get(key, [])
        if len(cached) == self.rmia_n_ref:
            return cached

        n_to_train = self.rmia_n_ref - len(cached)
        logger.info(
            f"  Training {n_to_train} RMIA reference model(s) on D_remain "
            f"(fr={forget_ratio}, run={run_id}, total={self.rmia_n_ref})..."
        )
        test_loader = self.test_loader if self.is_tabular else self._cifar_test_loader
        assert test_loader is not None

        models = list(cached)
        for i in range(len(cached), self.rmia_n_ref):
            ref_seed = int(self.seed) + int(run_id) * 10000 + i
            seed_everything(ref_seed, deterministic=self.deterministic)
            ref_model = self._fresh_model()
            optimizer = self._build_torch_optimizer(ref_model, self.lr, optimizer_name=self.optimizer_name)
            scheduler, sched_step_per_batch = self._build_torch_scheduler(
                optimizer,
                scheduler_name=self.scheduler_name,
                epochs=int(self.max_epochs),
                steps_per_epoch=len(retain_loader),
                max_lr=float(self.lr),
            )
            train_out = train_model(
                ref_model, retain_loader, test_loader, self.criterion, optimizer, self.max_epochs,
                device=self.device,
                show_progress=False,
                report_each_epoch=False,
                scheduler=scheduler,
                scheduler_step_per_batch=sched_step_per_batch,
                use_amp=self.use_amp,
            )
            ref_model = self._unwrap_trained_model(train_out)
            ref_model.eval()
            models.append(ref_model)
            logger.info(f"  Reference model {i+1}/{self.rmia_n_ref} trained.")

        self._ref_model_cache[key] = models
        return models

    @staticmethod
    def _extract_true_class_probs_predict_proba(model, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Return P(y_true | x) for each sample from a predict_proba-compatible model."""
        proba = np.asarray(model.predict_proba(X), dtype=np.float64)
        return proba[np.arange(len(y)), np.asarray(y, dtype=int)]

    def _get_reference_models_predict_proba(
        self, X_retain: np.ndarray, y_retain: np.ndarray, forget_ratio: float, run_id: int
    ) -> list:
        """Train or retrieve RMIA reference models for predict_proba-compatible estimators."""
        key = (float(forget_ratio), int(run_id), "xgb")
        cached = self._ref_model_cache.get(key, [])
        if len(cached) == self.rmia_n_ref:
            return cached

        n_to_train = self.rmia_n_ref - len(cached)
        logger.info(
            f"  Training {n_to_train} predict_proba RMIA reference model(s) on D_remain "
            f"(fr={forget_ratio}, run={run_id}, total={self.rmia_n_ref})..."
        )

        models = list(cached)
        for i in range(len(cached), self.rmia_n_ref):
            ref_seed = int(self.seed) + int(run_id) * 10000 + i
            ref_model = copy.deepcopy(self.initial_model)  # XGBoost path: deepcopy is correct
            if hasattr(ref_model, "set_params"):
                ref_model.set_params(random_state=ref_seed)
            ref_model.fit(X_retain, y_retain)
            models.append(ref_model)
            logger.info(f"  Predict_proba reference model {i+1}/{self.rmia_n_ref} trained.")

        self._ref_model_cache[key] = models
        return models

    def _evaluate_model(self, model, is_pytorch: bool) -> Tuple[float, float]:
        if is_pytorch:
            model.eval()
            if self.dataset == "credit":
                train_m = self._auc(model, self.train_loader)
                test_m = self._auc(model, self.test_loader)
            else:
                if self.is_tabular:
                    train_m = self._acc(model, self.train_loader)
                    test_m = self._acc(model, self.test_loader)
                else:
                    train_m = self._acc(model, self._cifar_full_train_loader)  # type: ignore[arg-type]
                    test_m = self._acc(model, self._cifar_test_loader)         # type: ignore[arg-type]
            return float(train_m), float(test_m)

        # XGBoost
        if self.dataset == "credit":
            return (
                float(roc_auc_score(self.y_train, model.predict_proba(self.X_train)[:, 1])),
                float(roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])),
            )
        return (
            float(accuracy_score(self.y_train, model.predict(self.X_train))),
            float(accuracy_score(self.y_test, model.predict(self.X_test))),
        )

    # -----------------------------
    # MIA
    # -----------------------------
    def _compute_mia(
        self,
        model,
        forget_loader: DataLoader,
        m: int,
        run_id: int,
        is_pytorch: bool,
        retain_loader: Optional[DataLoader] = None,
        forget_ratio: Optional[float] = None,
        X_retain: Optional[np.ndarray] = None,
        y_retain: Optional[np.ndarray] = None,
    ) -> Dict[str, Tuple[float, float]]:
        """
        Run selected MIA attacks (self.mia_attacks) and return results per attack.

        Members   = forget set (Df)
        Non-members = random subset of test set, size = min(m, n_test)

        Returns dict: {attack_name: (auc, tpr_at_1pct_fpr)}
        attack_name in {'loss', 'scaled_logit', 'rmia'}
        """
        rng = np.random.default_rng(int(self.seed) + int(run_id))
        results: Dict[str, Tuple[float, float]] = {}

        if not is_pytorch:
            # XGBoost / sklearn path
            test_preds   = np.asarray(model.predict_proba(self.X_test))
            forget_preds = np.asarray(model.predict_proba(self.X_forget))

            y_test_oh   = self.encoder.transform(self.y_test.reshape(-1, 1))
            y_forget_oh = self.encoder.transform(self.y_forget.reshape(-1, 1))

            # Vectorized cross-entropy per sample (replaces O(n) Python loop)
            loss_test   = -np.sum(y_test_oh   * np.log(np.clip(test_preds,   1e-15, 1.0)), axis=1)
            loss_forget = -np.sum(y_forget_oh * np.log(np.clip(forget_preds, 1e-15, 1.0)), axis=1)

            n_test = len(self.y_test)
            m_eff  = int(min(int(m), n_test))
            if m_eff < 2:
                return {'loss': (0.5, 0.01)}

            rand_idxs = rng.choice(n_test, size=m_eff, replace=False)
            _xgb_forget_idx = None
            if len(self.y_forget) != m_eff:
                _xgb_forget_idx = rng.choice(len(self.y_forget), size=m_eff, replace=False)
                forget_preds = forget_preds[_xgb_forget_idx]
                loss_forget  = loss_forget[_xgb_forget_idx]
                y_forget_eff = np.asarray(self.y_forget)[_xgb_forget_idx]
            else:
                y_forget_eff = np.asarray(self.y_forget)

            attack_result = tf_attack(
                logits_train=forget_preds,
                logits_test=test_preds[rand_idxs],
                loss_train=loss_forget,
                loss_test=loss_test[rand_idxs],
                train_labels=y_forget_eff,
                test_labels=np.asarray(self.y_test)[rand_idxs],
                run_extended=False,
            )
            results['loss'] = (float(attack_result.get_yeom_auc()), float(attack_result.get_yeom_tpr_at_fpr()))

            # ── RMIA for sklearn/XGBoost ─────────────────────────────────────
            if 'rmia' in self.mia_attacks:
                if X_retain is None or y_retain is None or forget_ratio is None:
                    logger.warning("RMIA requested for a predict_proba model but X_retain/y_retain not provided; skipping.")
                    results['rmia'] = (0.5, 0.01)
                else:
                    from eupg.evaluation.attacks import rmia_attack as _rmia_attack
                    ref_models = self._get_reference_models_predict_proba(X_retain, y_retain, forget_ratio, run_id)

                    y_test_eff  = np.asarray(self.y_test)[rand_idxs]
                    y_forget_np = np.asarray(self.y_forget)
                    if _xgb_forget_idx is not None:
                        y_forget_eff_rmia = y_forget_np[_xgb_forget_idx]
                        X_forget_eff      = self.X_forget[_xgb_forget_idx]
                    else:
                        y_forget_eff_rmia = y_forget_np
                        X_forget_eff      = self.X_forget
                    X_test_eff = self.X_test[rand_idxs]

                    # Average P_ref across reference models
                    ref_m_stack  = np.stack([
                        self._extract_true_class_probs_predict_proba(rm, X_forget_eff, y_forget_eff_rmia)
                        for rm in ref_models
                    ], axis=0)
                    ref_nm_stack = np.stack([
                        self._extract_true_class_probs_predict_proba(rm, X_test_eff, y_test_eff)
                        for rm in ref_models
                    ], axis=0)
                    probs_ref_m  = ref_m_stack.mean(axis=0)
                    probs_ref_nm = ref_nm_stack.mean(axis=0)

                    # True-class probs from target
                    probs_tgt_m  = self._extract_true_class_probs_predict_proba(model, X_forget_eff, y_forget_eff_rmia)
                    probs_tgt_nm = self._extract_true_class_probs_predict_proba(model, X_test_eff,   y_test_eff)

                    auc, tpr = _rmia_attack(probs_tgt_m, probs_ref_m, probs_tgt_nm, probs_ref_nm)
                    results['rmia'] = (float(auc), float(tpr))

            return results

        # --- PyTorch path ---
        test_loader = self.test_loader if self.is_tabular else self._cifar_test_loader
        assert test_loader is not None

        logits_test,   loss_test,   test_labels   = compute_attack_components(model, test_loader)
        logits_forget, loss_forget, forget_labels = compute_attack_components(model, forget_loader)

        test_labels   = np.asarray(test_labels)
        forget_labels = np.asarray(forget_labels)
        n_test = len(test_labels)
        m_eff  = int(min(int(m), n_test))
        if m_eff < 2:
            return {a: (0.5, 0.01) for a in self.mia_attacks}

        rand_idxs = rng.choice(n_test, size=m_eff, replace=False)

        logits_test   = np.asarray(logits_test)[rand_idxs]
        loss_test     = np.asarray(loss_test).reshape(-1)[rand_idxs]
        test_labels   = test_labels[rand_idxs]

        logits_forget = np.asarray(logits_forget)
        loss_forget   = np.asarray(loss_forget).reshape(-1)

        _forget_idx_mem = None  # track for RMIA
        if len(forget_labels) != m_eff:
            idx_mem = rng.choice(len(forget_labels), size=m_eff, replace=False)
            _forget_idx_mem = idx_mem
            logits_forget = logits_forget[idx_mem]
            loss_forget   = loss_forget[idx_mem]
            forget_labels = forget_labels[idx_mem]

        # ── loss (Yeom threshold) ────────────────────────────────────────────────
        if 'loss' in self.mia_attacks:
            attack_result = tf_attack(
                logits_forget, logits_test, loss_forget, loss_test,
                forget_labels, test_labels, run_extended=False,
            )
            results['loss'] = (float(attack_result.get_yeom_auc()), float(attack_result.get_yeom_tpr_at_fpr()))

        # ── scaled_logit (LiRA single-model) ────────────────────────────────────
        if 'scaled_logit' in self.mia_attacks:
            from eupg.evaluation.attacks import lira_scaled_logit_score, _safe_auc_and_adv, _tpr_at_fpr
            scores_m  = lira_scaled_logit_score(logits_forget, forget_labels)
            scores_nm = lira_scaled_logit_score(logits_test,   test_labels)
            y_true  = np.concatenate([np.ones(m_eff, dtype=int), np.zeros(m_eff, dtype=int)])
            scores  = np.concatenate([scores_m, scores_nm])
            auc = _safe_auc_and_adv(y_true, scores)[0]
            tpr = _tpr_at_fpr(y_true, scores)
            results['scaled_logit'] = (float(auc), float(tpr))

        # ── RMIA (Zarifzadeh et al., 2024) ──────────────────────────────────────
        if 'rmia' in self.mia_attacks:
            if retain_loader is None or forget_ratio is None:
                logger.warning("RMIA requested but retain_loader/forget_ratio not provided — skipping.")
                results['rmia'] = (0.5, 0.01)
            else:
                from eupg.evaluation.attacks import rmia_attack as _rmia_attack, _to_probs

                ref_models = self._get_reference_models(retain_loader, forget_ratio, run_id)

                # Average P_ref across all reference models (reduces variance)
                ref_m_stack  = np.stack([
                    (self._extract_true_class_probs(rm, forget_loader)[_forget_idx_mem]
                     if _forget_idx_mem is not None
                     else self._extract_true_class_probs(rm, forget_loader))
                    for rm in ref_models
                ], axis=0)  # (n_ref, m_eff)
                ref_nm_stack = np.stack([
                    self._extract_true_class_probs(rm, test_loader)[rand_idxs]
                    for rm in ref_models
                ], axis=0)  # (n_ref, m_eff)

                probs_ref_m  = ref_m_stack.mean(axis=0)
                probs_ref_nm = ref_nm_stack.mean(axis=0)

                # True-class probs from target model (from logits already computed)
                probs_tgt_m  = _to_probs(logits_forget)[np.arange(m_eff), forget_labels]
                probs_tgt_nm = _to_probs(logits_test  )[np.arange(m_eff), test_labels  ]

                auc, tpr = _rmia_attack(probs_tgt_m, probs_ref_m, probs_tgt_nm, probs_ref_nm)
                results['rmia'] = (float(auc), float(tpr))

        return results

    # -----------------------------
    # I/O
    # -----------------------------
    @staticmethod
    def _print_metrics(metrics_dict: Dict[str, Tuple[float, float]], experiment_type: str) -> None:
        log_metrics_table(logger, metrics_dict, experiment_type)

    @staticmethod
    def _save_results(
        filename: str,
        metrics_dict: Dict[str, Tuple[float, float]],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        save_metrics_csv(filename, metrics_dict, metadata=metadata)

    # -----------------------------
    # Main run
    # -----------------------------
    def run_baseline(self) -> None:
        logger.info("Running BASELINE experiments...")
        summary_rows: List[Dict[str, Any]] = []

        for forget_ratio in self.forget_ratios:
            logger.info(f"Forget ratio: {forget_ratio}")

            # --------------------------
            # Baseline: Train on full data
            # --------------------------
            train_ms, test_ms, runtimes = [], [], []
            mia_results_all: Dict[str, List[float]] = {a: [] for a in self.mia_attacks}
            mia_tprs_all:    Dict[str, List[float]] = {a: [] for a in self.mia_attacks}

            for r in range(self.n_repeat):
                seed_everything(self.seed + int(r), deterministic=self.deterministic)
                logger.info(f"  Baseline run {r+1}/{self.n_repeat}")
                torch.cuda.empty_cache()

                X_retain_bl: Optional[np.ndarray] = None
                y_retain_bl: Optional[np.ndarray] = None
                if self.is_tabular:
                    X_retain_bl, y_retain_bl, self.X_forget, self.y_forget, retain_loader, forget_loader, m = self._split_forget_retain(
                        forget_ratio, run_id=r
                    )
                else:
                    _, _, _, _, retain_loader, forget_loader, m = self._split_forget_retain(forget_ratio, run_id=r)

                model = self._fresh_model()

                t0 = time.time()
                if self.model_type in ["mlp", "densenet", "resnet18"]:
                    optimizer = self._build_torch_optimizer(model, self.lr, optimizer_name=self.optimizer_name)

                    train_loader = self.train_loader if self.is_tabular else self._cifar_full_train_loader
                    test_loader = self.test_loader if self.is_tabular else self._cifar_test_loader
                    assert train_loader is not None and test_loader is not None
                    scheduler, sched_step_per_batch = self._build_torch_scheduler(
                        optimizer,
                        scheduler_name=self.scheduler_name,
                        epochs=int(self.max_epochs),
                        steps_per_epoch=len(train_loader),
                        max_lr=float(self.lr),
                    )

                    train_out = train_model(
                        model,
                        train_loader,
                        test_loader,
                        self.criterion,
                        optimizer,
                        self.max_epochs,
                        device=self.device,
                        verbose_epoch=max(1, int(self.max_epochs / 10)),
                        scheduler=scheduler,
                        scheduler_step_per_batch=sched_step_per_batch,
                        metric_fn=(self._auc if self.dataset == "credit" else self._acc),
                        metric_name=("auc" if self.dataset == "credit" else "acc"),
                        report_each_epoch=bool(self.epoch_metrics),
                        use_amp=self.use_amp,
                    )
                    model = self._unwrap_trained_model(train_out)  # FIX
                else:
                    if hasattr(model, "set_params"):
                        model.set_params(random_state=int(self.seed) + int(r))
                    model.fit(self.X_train, self.y_train)

                t1 = time.time()
                runtimes.append(t1 - t0)

                is_pytorch = (self.model_type != "xgboost")
                tr, te = self._evaluate_model(model, is_pytorch=is_pytorch)
                train_ms.append(100.0 * tr)
                test_ms.append(100.0 * te)

                mia_res = self._compute_mia(model, forget_loader, m, run_id=r, is_pytorch=is_pytorch,
                                           retain_loader=retain_loader, forget_ratio=forget_ratio,
                                           X_retain=X_retain_bl, y_retain=y_retain_bl)
                for atk in self.mia_attacks:
                    a, t = mia_res.get(atk, (0.5, 0.01))
                    mia_results_all[atk].append(100.0 * a)
                    mia_tprs_all[atk].append(100.0 * t)

                mia_summary = " | ".join(
                    f"MIA({atk}) AUC={mia_results_all[atk][-1]:.2f}% TPR={mia_tprs_all[atk][-1]:.2f}%"
                    for atk in self.mia_attacks
                )
                logger.info(f"    Train={tr*100:.2f}% | Test={te*100:.2f}% | {mia_summary} | Time={t1-t0:.2f}s")

            _ATK_LABEL = {'loss': 'Loss', 'scaled_logit': 'ScaledLogit', 'rmia': 'RMIA'}
            metrics_out: Dict[str, Tuple[float, float]] = {
                "Training Time": (float(np.mean(runtimes)), float(np.std(runtimes))),
                "Train Accuracy": (float(np.mean(train_ms)), float(np.std(train_ms))),
                "Test Accuracy": (float(np.mean(test_ms)), float(np.std(test_ms))),
            }
            for atk in self.mia_attacks:
                lbl = _ATK_LABEL.get(atk, atk)
                metrics_out[f"MIA AUC ({lbl})"]       = (float(np.mean(mia_results_all[atk])), float(np.std(mia_results_all[atk])))
                metrics_out[f"MIA TPR@1%FPR ({lbl})"] = (float(np.mean(mia_tprs_all[atk])),    float(np.std(mia_tprs_all[atk])))
            self._print_metrics(metrics_out, f"Baseline (forget_ratio={forget_ratio})")
            out1 = os.path.join(self.dataset_results_dir, f"{self.model_type}_m_d_fr={forget_ratio}.csv")
            self._save_results(out1, metrics_out)
            logger.info(f"Baseline results saved to {out1}")
            _rt = os.path.join(self.dataset_results_dir, f"{self.model_type}_runtimes.csv")
            append_runtime_rows(_rt, [{"Method": "baseline", "Param": "-", "Forget Ratio": float(forget_ratio), "Phase": "train_D",
                                       "N Runs": len(runtimes), "Mean (s)": metrics_out["Training Time"][0], "Std (s)": metrics_out["Training Time"][1]}])
            for metric, (mean_val, std_val) in metrics_out.items():
                summary_rows.append(
                    {
                        "Experiment": "baseline",
                        "Forget Ratio": float(forget_ratio),
                        "Metric": str(metric),
                        "Mean": float(mean_val),
                        "Std": float(std_val),
                        "Mean±Std": f"{float(mean_val):.4f} ± {float(std_val):.4f}",
                    }
                )

            # --------------------------
            # Retrain: Train only on retain set
            # --------------------------
            retain_ms, forget_ms, test_ms2, runtimes2 = [], [], [], []
            mia_results_all2: Dict[str, List[float]] = {a: [] for a in self.mia_attacks}
            mia_tprs_all2:    Dict[str, List[float]] = {a: [] for a in self.mia_attacks}

            for r in range(self.n_repeat):
                seed_everything(self.seed + int(r), deterministic=self.deterministic)
                logger.info(f"  Retrain run {r+1}/{self.n_repeat}")
                torch.cuda.empty_cache()

                if self.is_tabular:
                    X_retain, y_retain, self.X_forget, self.y_forget, retain_loader, forget_loader, m = self._split_forget_retain(
                        forget_ratio, run_id=r
                    )
                else:
                    X_retain, y_retain, _, _, retain_loader, forget_loader, m = self._split_forget_retain(forget_ratio, run_id=r)

                model = self._fresh_model()

                t0 = time.time()
                if self.model_type in ["mlp", "densenet", "resnet18"]:
                    optimizer = self._build_torch_optimizer(model, self.lr, optimizer_name=self.ft_optimizer_name)
                    test_loader = self.test_loader if self.is_tabular else self._cifar_test_loader
                    assert test_loader is not None
                    scheduler, sched_step_per_batch = self._build_torch_scheduler(
                        optimizer,
                        scheduler_name=self.ft_scheduler_name,
                        epochs=int(self.max_epochs),
                        steps_per_epoch=len(retain_loader),
                        max_lr=float(self.lr),
                    )

                    train_out = train_model(
                        model,
                        retain_loader,
                        test_loader,
                        self.criterion,
                        optimizer,
                        self.max_epochs,
                        device=self.device,
                        verbose_epoch=max(1, int(self.max_epochs / 10)),
                        scheduler=scheduler,
                        scheduler_step_per_batch=sched_step_per_batch,
                        metric_fn=(self._auc if self.dataset == "credit" else self._acc),
                        metric_name=("auc" if self.dataset == "credit" else "acc"),
                        report_each_epoch=bool(self.epoch_metrics),
                        use_amp=self.use_amp,
                    )
                    model = self._unwrap_trained_model(train_out)  # FIX
                else:
                    if hasattr(model, "set_params"):
                        model.set_params(random_state=int(self.seed) + int(r))
                    model.fit(X_retain, y_retain)

                t1 = time.time()
                runtimes2.append(t1 - t0)

                is_pytorch = (self.model_type != "xgboost")

                if is_pytorch:
                    if self.dataset == "credit":
                        retain_m = self._auc(model, retain_loader)
                        forget_m = self._auc(model, forget_loader)
                        test_m = self._auc(model, self.test_loader)
                    else:
                        retain_m = self._acc(model, retain_loader)
                        forget_m = self._acc(model, forget_loader)
                        test_loader = self.test_loader if self.is_tabular else self._cifar_test_loader
                        assert test_loader is not None
                        test_m = self._acc(model, test_loader)
                else:
                    if self.dataset == "credit":
                        retain_m = roc_auc_score(y_retain, model.predict_proba(X_retain)[:, 1])
                        forget_m = roc_auc_score(self.y_forget, model.predict_proba(self.X_forget)[:, 1])
                        test_m = roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])
                    else:
                        retain_m = accuracy_score(y_retain, model.predict(X_retain))
                        forget_m = accuracy_score(self.y_forget, model.predict(self.X_forget))
                        test_m = accuracy_score(self.y_test, model.predict(self.X_test))

                retain_ms.append(100.0 * float(retain_m))
                forget_ms.append(100.0 * float(forget_m))
                test_ms2.append(100.0 * float(test_m))

                mia_res2 = self._compute_mia(model, forget_loader, m, run_id=r, is_pytorch=is_pytorch,
                                            retain_loader=retain_loader, forget_ratio=forget_ratio,
                                            X_retain=(X_retain if not is_pytorch else None),
                                            y_retain=(y_retain if not is_pytorch else None))
                for atk in self.mia_attacks:
                    a, t = mia_res2.get(atk, (0.5, 0.01))
                    mia_results_all2[atk].append(100.0 * a)
                    mia_tprs_all2[atk].append(100.0 * t)

                mia_summary2 = " | ".join(
                    f"MIA({atk}) AUC={mia_results_all2[atk][-1]:.2f}% TPR={mia_tprs_all2[atk][-1]:.2f}%"
                    for atk in self.mia_attacks
                )
                logger.info(
                    f"    Retain={float(retain_m)*100:.2f}% | Forget={float(forget_m)*100:.2f}% | Test={float(test_m)*100:.2f}% | "
                    f"{mia_summary2} | Time={t1-t0:.2f}s"
                )

            _ATK_LABEL = {'loss': 'Loss', 'scaled_logit': 'ScaledLogit', 'rmia': 'RMIA'}
            metrics_out: Dict[str, Tuple[float, float]] = {
                "Retraining Time": (float(np.mean(runtimes2)), float(np.std(runtimes2))),
                "Retain Accuracy": (float(np.mean(retain_ms)), float(np.std(retain_ms))),
                "Forget Accuracy": (float(np.mean(forget_ms)), float(np.std(forget_ms))),
                "Test Accuracy": (float(np.mean(test_ms2)), float(np.std(test_ms2))),
            }
            for atk in self.mia_attacks:
                lbl = _ATK_LABEL.get(atk, atk)
                metrics_out[f"MIA AUC ({lbl})"]       = (float(np.mean(mia_results_all2[atk])), float(np.std(mia_results_all2[atk])))
                metrics_out[f"MIA TPR@1%FPR ({lbl})"] = (float(np.mean(mia_tprs_all2[atk])),    float(np.std(mia_tprs_all2[atk])))
            self._print_metrics(metrics_out, f"Retrain (forget_ratio={forget_ratio})")
            out2 = os.path.join(self.dataset_results_dir, f"{self.model_type}_mret_dret_fr={forget_ratio}.csv")
            self._save_results(out2, metrics_out)
            logger.info(f"Retrain results saved to {out2}")
            _rt = os.path.join(self.dataset_results_dir, f"{self.model_type}_runtimes.csv")
            append_runtime_rows(_rt, [{"Method": "baseline", "Param": "-", "Forget Ratio": float(forget_ratio), "Phase": "retrain_Dr",
                                       "N Runs": len(runtimes2), "Mean (s)": metrics_out["Retraining Time"][0], "Std (s)": metrics_out["Retraining Time"][1]}])
            for metric, (mean_val, std_val) in metrics_out.items():
                summary_rows.append(
                    {
                        "Experiment": "retrain",
                        "Forget Ratio": float(forget_ratio),
                        "Metric": str(metric),
                        "Mean": float(mean_val),
                        "Std": float(std_val),
                        "Mean±Std": f"{float(mean_val):.4f} ± {float(std_val):.4f}",
                    }
                )

        if summary_rows:
            logger.info("=" * 80)
            logger.info("FINAL SUMMARY (mean ± std across runs)")
            logger.info("=" * 80)
            logger.info(f"{'Exp':<10} {'FR':<8} {'Metric':<20} {'Mean±Std':<20}")
            logger.info("-" * 80)
            for row in summary_rows:
                logger.info(
                    f"{str(row['Experiment']):<10} {float(row['Forget Ratio']):<8.4f} "
                    f"{str(row['Metric']):<20} {str(row['Mean±Std']):<20}"
                )
            summary_path = os.path.join(self.dataset_results_dir, f"{self.model_type}_baseline_retrain_summary.csv")
            save_summary_csv(summary_path, summary_rows)
            logger.info(f"Saved overall baseline/retrain summary to {summary_path}")

            cfg_path = os.path.join(self.dataset_results_dir, f"{self.model_type}_baseline_config.yaml")
            cfg = self._config_identity()
            cfg.update({
                "training": self._config_training(),
                "retrain": {
                    "epochs": int(self.max_epochs) if getattr(self, "max_epochs", None) is not None else None,
                    "optimizer": getattr(self, "ft_optimizer_name", None),
                    "scheduler": getattr(self, "ft_scheduler_name", None),
                    "lr": float(self.lr) if getattr(self, "lr", None) is not None else None,
                },
                "runtime": self._config_runtime(),
                "xgboost": self._config_xgboost(),
                "mia": self._config_mia(),
            })
            save_config_yaml(cfg_path, cfg)
            logger.info(f"Saved baseline config to {cfg_path}")

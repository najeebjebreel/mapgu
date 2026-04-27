"""Differential-privacy experiments: 3-phase train/fine-tune/retain loop."""
from __future__ import annotations

import copy
import csv
import glob
import json
import os
import time
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics as sk_metrics
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import DataLoader, Subset, TensorDataset
from tqdm import tqdm
from torchvision import transforms

from mapgu.config import DATA_DIR, DP_DIR, KANON_DIR, CIFAR_MEAN, CIFAR_STD, ADULT_EMB_RT_CSV
from mapgu.experiments.base import PrivacyBenchmark, build_cluster_repr_onehot, build_cluster_repr_tabnet
from mapgu.models.factory import seed_everything
from mapgu.evaluation.attacks import tf_attack
from mapgu.data.privacy.kanon import mdav_clusters, probabilistic_k_anonymize_by_permutation
from mapgu.evaluation.metrics import accuracy, auc_score, compute_attack_components
from mapgu.training.trainer import train_model
from mapgu.utils import get_logger, save_metrics_csv, save_summary_csv, save_config_yaml, log_metrics_table, _ms, _fmt_eps, _ensure_dir, append_runtime_rows

_LOGGER_PRIVACY = get_logger("mapgu.experiments.privacy")
_LOGGER_DP = get_logger("mapgu.experiments.dp")
_LOGGER_KANON = get_logger("mapgu.experiments.kanon")
logger = _LOGGER_PRIVACY


@contextmanager
def _log_namespace(ns: str):
    global logger
    prev = logger
    if ns == "dp":
        logger = _LOGGER_DP
    elif ns == "kanon":
        logger = _LOGGER_KANON
    else:
        logger = _LOGGER_PRIVACY
    try:
        yield
    finally:
        logger = prev


# --------------------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------------------
def _append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    _ensure_dir(os.path.dirname(path) or ".")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _read_single_row_csv(path: str) -> Dict[str, Any]:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return {}
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        rows = list(r)
    return rows[-1] if rows else {}


def load_npz_train(path: str) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    return z["X"], z["y"]


def _dp_dataset_dir(dataset: str) -> str:
    return "GiveMeSomeCredit" if dataset == "credit" else dataset


def _dp_eps_dir(dataset: str, eps: float) -> str:
    return os.path.join(DP_DIR, _dp_dataset_dir(dataset), f"eps={_fmt_eps(eps)}")


def _resolve_dp_tabular_csv(dataset: str, eps: float) -> str:
    eps_dir = _dp_eps_dir(dataset, eps)
    base_name = {
        "adult": "dp_adult",
        "credit": "dp_credit",
        "heart": "dp_heart",
    }[dataset]
    canonical_path = os.path.join(eps_dir, f"{base_name}.csv")
    if os.path.isfile(canonical_path):
        return canonical_path

    legacy_matches = sorted(glob.glob(os.path.join(eps_dir, f"{base_name}_eps=*.csv")))
    if legacy_matches:
        return legacy_matches[0]

    return canonical_path


def lookup_prep_time_dp(dataset: str, eps: float, seed: int, *, m: int = 16, b: int = 4) -> float:
    rt_path = os.path.join(_dp_eps_dir(dataset, eps), "runtimes.csv")
    row = _read_single_row_csv(rt_path)
    if not row:
        return 0.0
    for key in ("total_time_sec", "prep_total_time_sec", "prep_time_sec"):
        v = row.get(key, "")
        if str(v).strip():
            try:
                return float(v)
            except Exception:
                pass
    return 0.0


def lookup_prep_time_kanon(dataset: str, k: int, cluster_repr: str, seed: int) -> float:
    rt_path = os.path.join(
        KANON_DIR, dataset, f"cluster={cluster_repr}", f"seed={seed}", f"k={int(k)}", "runtimes.csv"
    )
    row = _read_single_row_csv(rt_path)
    if not row:
        return 0.0
    for key in ("total_time_sec", "prep_total_time_sec", "prep_time_sec"):
        v = row.get(key, "")
        if str(v).strip():
            try:
                return float(v)
            except Exception:
                pass
    return 0.0


def lookup_adult_embed_time() -> float:
    """Return the one-time Adult embedding generation time (0.0 if not recorded yet).

    build_embeddings.py writes columns: embedding, utility, total.
    We report `total` (= embedding + utility) to match Table X in the paper.
    """
    row = _read_single_row_csv(ADULT_EMB_RT_CSV)
    if not row:
        return 0.0
    # Schema written by build_embeddings.py: embedding, utility, total
    for key in ("total", "embedding", "total_time_sec", "embed_time_sec"):
        v = row.get(key, "")
        if str(v).strip():
            try:
                return float(v)
            except Exception:
                pass
    return 0.0


# --------------------------------------------------------------------------------------
# Torch training helpers (fast + safe)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def _eval_loss(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    was_training = model.training
    model.eval()
    total = 0.0
    n = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        total += float(criterion(out, y).item())
        n += 1
    if was_training:
        model.train()
    return total / max(n, 1)


def _train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    *,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
    verbose_epoch: int = 10,
    patience: Optional[int] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    scheduler_step_per_batch: bool = True,
    metric_fn: Optional[Callable[[nn.Module, DataLoader], float]] = None,
    metric_name: str = "acc",
    show_epoch_metrics: bool = False,
) -> nn.Module:
    model.to(device)
    model.train()

    epoch_iter = tqdm(range(int(epochs)), desc="Epochs", leave=False)
    for ep in epoch_iter:
        running_loss = 0.0
        n_batches = 0
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            if scheduler is not None and bool(scheduler_step_per_batch):
                scheduler.step()
            running_loss += float(loss.item())
            n_batches += 1

        train_loss = running_loss / max(n_batches, 1)
        if scheduler is not None and (not bool(scheduler_step_per_batch)):
            scheduler.step()
        test_loss = None
        train_metric = None
        test_metric = None
        if show_epoch_metrics and val_loader is not None:
            test_loss = _eval_loss(model, val_loader, criterion, device)
            train_metric = float(metric_fn(model, train_loader)) if metric_fn is not None else None
            test_metric = float(metric_fn(model, val_loader)) if metric_fn is not None else None
            if train_metric is not None and test_metric is not None:
                logger.info(
                    f"[epoch {ep + 1:03d}/{int(epochs)}] "
                    f"train_loss={train_loss:.4f} test_loss={test_loss:.4f} "
                    f"train_{metric_name}={train_metric:.4f} test_{metric_name}={test_metric:.4f}"
                )
            else:
                logger.info(f"[epoch {ep + 1:03d}/{int(epochs)}] train_loss={train_loss:.4f} test_loss={test_loss:.4f}")

        if val_loader is not None and (ep == 0 or (ep + 1) % max(1, int(verbose_epoch)) == 0):
            v = float(test_loss) if test_loss is not None else _eval_loss(model, val_loader, criterion, device)
            epoch_iter.set_postfix(train_loss=f"{train_loss:.4f}", val_loss=f"{v:.4f}")
        else:
            epoch_iter.set_postfix(train_loss=f"{train_loss:.4f}")

    return model


# --------------------------------------------------------------------------------------
# Main experiments class (contains both DP and k-anon)
# --------------------------------------------------------------------------------------
class PrivacyExperiments(PrivacyBenchmark):
    """
    Loads prepared private datasets and runs 3-phase training + stable MIA.
    """

    def __init__(
        self,
        *args,
        mia_resamples: int = 10,
        mia_eval_cap: int = 5000,
        lr: Optional[float] = None,
        ft_lr: Optional[float] = None,
        private_optimizer: Optional[str] = None,
        private_scheduler: Optional[str] = None,
        ft_optimizer: Optional[str] = None,
        ft_scheduler: Optional[str] = None,
        kanon_mode: str = "prepared",
        kanon_seed: Optional[int] = None,
        epoch_metrics: bool = False,
        resume: bool = False,
        progress_path: Optional[str] = None,
        max_configs: Optional[int] = None,
        num_workers: int = 0,
        pin_memory: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.mia_resamples = int(mia_resamples)
        self.mia_eval_cap = int(mia_eval_cap)
        if lr is not None and float(lr) <= 0:
            raise ValueError("--lr must be > 0")
        if ft_lr is not None and float(ft_lr) <= 0:
            raise ValueError("--ft_lr must be > 0")

        default_lr = float(getattr(self, "lr", 1e-2))
        self.private_lr = float(lr) if lr is not None else default_lr
        self.ft_lr = float(ft_lr) if ft_lr is not None else float(self.private_lr)
        self.private_optimizer = str(private_optimizer).lower() if private_optimizer is not None else str(self.optimizer_name)
        self.private_scheduler = str(private_scheduler).lower() if private_scheduler is not None else str(self.scheduler_name)
        self.ft_optimizer = str(ft_optimizer).lower() if ft_optimizer is not None else str(self.ft_optimizer_name)
        self.ft_scheduler = str(ft_scheduler).lower() if ft_scheduler is not None else str(self.ft_scheduler_name)
        self.lr = float(self.private_lr)
        self.epoch_metrics = bool(epoch_metrics)
        if kanon_mode not in {"prepared", "regenerate"}:
            raise ValueError("--kanon_mode must be one of: prepared, regenerate")
        self.kanon_mode = str(kanon_mode)
        self.kanon_seed = int(kanon_seed) if kanon_seed is not None else int(self.seed)

        self.resume = bool(resume)
        self.progress_path = progress_path
        self.max_configs = max_configs
        self._configs_executed = 0

        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self._summary_rows: List[Dict[str, Any]] = []

        self._split_cache: Dict[Tuple[float, int], Tuple[np.ndarray, np.ndarray]] = {}

        self._torch_test_loader_cached = self._torch_test_loader()

    def _append_summary_rows(
        self,
        *,
        experiment: str,
        phase: str,
        forget_ratio: float,
        param_name: str,
        param_value: Any,
        metrics_dict: Dict[str, Tuple[float, float]],
        ft_epochs: int,
    ) -> None:
        for metric, (mean_val, std_val) in metrics_dict.items():
            self._summary_rows.append(
                {
                    "Experiment": str(experiment),
                    "Phase": str(phase),
                    "Forget Ratio": float(forget_ratio),
                    "Param Name": str(param_name),
                    "Param Value": str(param_value),
                    "FT Epochs": int(ft_epochs),
                    "Metric": str(metric),
                    "Mean": float(mean_val),
                    "Std": float(std_val),
                    "Mean±Std": f"{float(mean_val):.4f} ± {float(std_val):.4f}",
                }
            )

    def save_and_print_summary(self) -> None:
        if not self._summary_rows:
            return
        logger.info("=" * 80)
        logger.info("FINAL SUMMARY (mean ± std across runs)")
        logger.info("=" * 80)
        logger.info(f"{'Exp':<12} {'Phase':<22} {'Param':<14} {'FT':<6} {'FR':<8} {'Metric':<18} {'Mean±Std':<20}")
        logger.info("-" * 80)
        for row in self._summary_rows:
            param = f"{row['Param Name']}={row['Param Value']}"
            logger.info(
                f"{str(row['Experiment']):<12} {str(row['Phase']):<22} {param:<14} "
                f"{int(row['FT Epochs']):<6d} {float(row['Forget Ratio']):<8.4f} "
                f"{str(row['Metric']):<18} {str(row['Mean±Std']):<20}"
            )
        out_path = os.path.join(self.dataset_results_dir, f"{self.model_type}_privacy_summary.csv")
        save_summary_csv(out_path, self._summary_rows)
        logger.info(f"Saved overall privacy summary to {out_path}")

    def _build_kanon_loader(self, *, k: int, seed: int) -> Tuple[DataLoader, float]:
        t0 = time.time()

        if not self.is_tabular:
            # CIFAR-10 path: cluster in ResNet-18 latent space, permute pixels
            from mapgu.data.privacy.cifar_kanon import build_cifar_kanon_data
            # trainset.data is (N, H, W, C) uint8
            raw = np.asarray(self.trainset.data, dtype=np.float32) / 255.0  # (N, H, W, C)
            raw = raw.transpose(0, 3, 1, 2)                                  # (N, C, H, W)
            cifar_mean = np.array(CIFAR_MEAN, dtype=np.float32).reshape(1, 3, 1, 1)
            cifar_std  = np.array(CIFAR_STD,  dtype=np.float32).reshape(1, 3, 1, 1)
            images_norm = (raw - cifar_mean) / cifar_std
            y_arr = np.asarray(self.trainset.targets, dtype=np.int64)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            images_anon, y_anon, _, _ = build_cifar_kanon_data(
                images_norm=images_norm,
                y=y_arr,
                k=int(k),
                seed=int(seed),
                device=device,
                batch_size=self.batch_size,
            )
            prep_time = float(time.time() - t0)
            return self._torch_loader_from_numpy(images_anon, y_anon, shuffle=True), prep_time

        X_base = np.asarray(self.X_train)
        y_base = np.asarray(self.y_train)

        if self.kanon_cluster_repr == "onehot":
            X_repr = build_cluster_repr_onehot(X_base)
        elif self.kanon_cluster_repr == "tabnet":
            if self.dataset != "adult":
                raise ValueError("cluster_repr=tabnet is supported only for adult dataset.")
            if self.embeddings_pkl is None:
                raise ValueError("--embeddings_pkl is required for --kanon_cluster_repr tabnet.")
            if not hasattr(self, "X_train_raw_df") or self.X_train_raw_df is None:
                raise RuntimeError("Adult raw training dataframe is unavailable for tabnet clustering.")
            X_repr = build_cluster_repr_tabnet(X_train_raw_df=self.X_train_raw_df, embeddings_pkl_path=self.embeddings_pkl)
        else:
            raise ValueError(f"Unsupported kanon_cluster_repr={self.kanon_cluster_repr}")

        clusters = mdav_clusters(
            X=np.array(X_repr, copy=True),
            y=np.array(y_base, copy=True),
            k=int(k),
            show_progress=False,
        )
        X_anon, y_anon = probabilistic_k_anonymize_by_permutation(
            X=np.array(X_base, copy=True),
            y=np.array(y_base, copy=True),
            clusters=clusters,
            seed=int(seed),
            protect_mask=None,
            perm_type=str(getattr(self, "kanon_perm_type", "rowwise")),
        )
        prep_time = float(time.time() - t0)
        return self._torch_loader_from_numpy(X_anon, y_anon, shuffle=True), prep_time

    # -----------------------------
    # loaders
    # -----------------------------
    def _torch_test_loader(self) -> DataLoader:
        if self.is_tabular:
            return self.test_loader
        return DataLoader(
            self.testset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=bool(self.num_workers > 0),
        )

    def _torch_full_train_loader(self) -> DataLoader:
        if self.is_tabular:
            return self.train_loader
        return DataLoader(
            self.trainset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=bool(self.num_workers > 0),
        )

    def _torch_loader_from_numpy(self, X: np.ndarray, y: np.ndarray, *, shuffle: bool) -> DataLoader:
        ds = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64))
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=bool(shuffle),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=bool(self.num_workers > 0),
        )

    # -----------------------------
    # progress / resume helpers
    # -----------------------------
    def _log_progress(self, event: str, payload: Dict[str, Any]) -> None:
        if not self.progress_path:
            return
        _append_jsonl(self.progress_path, {"ts": time.time(), "event": event, **payload})

    def _maybe_stop_for_max_configs(self) -> bool:
        return (self.max_configs is not None) and (self._configs_executed >= int(self.max_configs))

    def _cfg_key(self, experiment_type: str, forget_ratio: float, k_or_eps: Any, ft_epochs: int) -> str:
        return (
            f"{experiment_type}|dataset={self.dataset}|model={self.model_type}|fr={forget_ratio}"
            f"|param={k_or_eps}|ft={ft_epochs}|repr={getattr(self, 'kanon_cluster_repr', None)}"
            f"|perm={getattr(self, 'kanon_perm_type', 'rowwise')}|seed={self.seed}"
        )

    def _expected_output_files(self, experiment_type: str, forget_ratio: float, k_or_eps: Any, ft_epochs: int) -> List[str]:
        base = self.dataset_results_dir
        if experiment_type == "k-anonymity":
            k = int(k_or_eps)
            return [
                os.path.join(base, f"{self.model_type}_mk={k}_dk_fr={forget_ratio}.csv"),
                os.path.join(base, f"{self.model_type}_mk={k}_d_fr={forget_ratio}_epochs={ft_epochs}.csv"),
                os.path.join(base, f"{self.model_type}_mk={k}_dret_fr={forget_ratio}_epochs={ft_epochs}.csv"),
            ]
        if experiment_type == "differential_privacy":
            eps = k_or_eps
            return [
                os.path.join(base, f"{self.model_type}_mdp_eps={eps}_fr={forget_ratio}.csv"),
                os.path.join(base, f"{self.model_type}_mdpd_eps={eps}_fr={forget_ratio}_epochs={ft_epochs}.csv"),
                os.path.join(base, f"{self.model_type}_mdpret_eps={eps}_fr={forget_ratio}_epochs={ft_epochs}.csv"),
            ]
        return []

    def _is_config_done(self, experiment_type: str, forget_ratio: float, k_or_eps: Any, ft_epochs: int) -> bool:
        files = self._expected_output_files(experiment_type, forget_ratio, k_or_eps, ft_epochs)
        return bool(files) and all(os.path.exists(p) and os.path.getsize(p) > 0 for p in files)

    # -----------------------------
    # per-repeat split caching
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

    def _split_forget_retain_per_run(self, forget_ratio: float, run_id: int):
        retain_idxs, forget_idxs = self._get_split_indices(forget_ratio, run_id)
        m = int(len(forget_idxs))

        if self.is_tabular:
            X_retain = self.X_train[retain_idxs]
            y_retain = self.y_train[retain_idxs]
            X_forget = self.X_train[forget_idxs]
            y_forget = self.y_train[forget_idxs]

            self.X_forget = X_forget
            self.y_forget = y_forget

            retain_loader = self._torch_loader_from_numpy(X_retain, y_retain, shuffle=True)
            forget_loader = self._torch_loader_from_numpy(X_forget, y_forget, shuffle=False)
            return X_retain, y_retain, retain_loader, forget_loader, m

        retain_set = Subset(self.trainset, retain_idxs)
        forget_set = Subset(self.trainset, forget_idxs)
        retain_loader = DataLoader(
            retain_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=bool(self.num_workers > 0),
        )
        forget_loader = DataLoader(
            forget_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=bool(self.num_workers > 0),
        )
        self.X_forget, self.y_forget = None, None
        return None, None, retain_loader, forget_loader, m

    # -----------------------------
    # evaluation helpers
    # -----------------------------
    def _torch_metric(self, model: nn.Module, loader: DataLoader) -> float:
        if self.dataset == "credit":
            return float(auc_score(model, loader, device=self.device))
        return float(accuracy(model, loader, device=self.device))

    def _evaluate_retain_forget_test(
        self, model: Any, *, retain_loader: DataLoader, forget_loader: DataLoader, is_torch_model: bool, X_retain=None, y_retain=None
    ) -> Tuple[float, float, float]:
        if is_torch_model:
            r = self._torch_metric(model, retain_loader)
            f = self._torch_metric(model, forget_loader)
            t = self._torch_metric(model, self._torch_test_loader_cached)
            return r, f, t

        assert X_retain is not None and y_retain is not None
        X_forget, y_forget = self.X_forget, self.y_forget
        if self.dataset == "credit":
            r = roc_auc_score(y_retain, model.predict_proba(X_retain)[:, 1])
            f = roc_auc_score(y_forget, model.predict_proba(X_forget)[:, 1])
            t = roc_auc_score(self.y_test, model.predict_proba(self.X_test)[:, 1])
            return float(r), float(f), float(t)

        r = accuracy_score(y_retain, model.predict(X_retain))
        f = accuracy_score(y_forget, model.predict(X_forget))
        t = accuracy_score(self.y_test, model.predict(self.X_test))
        return float(r), float(f), float(t)

    # -----------------------------
    # MIA (stable)
    # -----------------------------
    @staticmethod
    def _effective_m(m: int, n_test: int) -> int:
        m = int(m)
        n_test = int(n_test)
        if m <= 0 or n_test <= 0:
            return 0
        return min(m, n_test)

    def _mia_from_precomputed(
        self,
        *,
        logits_members: np.ndarray,
        loss_members: np.ndarray,
        labels_members: np.ndarray,
        logits_test: np.ndarray,
        loss_test: np.ndarray,
        labels_test: np.ndarray,
        rand_idxs: np.ndarray,
        rmia_probs_full: "Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]" = None,
    ) -> Dict[str, Tuple[float, float]]:
        """Run selected MIA attacks on pre-computed logits/loss and return dict of results.

        rmia_probs_full: (probs_tgt_m, probs_ref_m, probs_tgt_nm_full, probs_ref_nm_full)
        where *_full arrays cover the entire test set and are subsampled here by rand_idxs.
        """
        from mapgu.evaluation.attacks import lira_scaled_logit_score, _safe_auc_and_adv, _tpr_at_fpr, rmia_attack as _rmia_attack
        results: Dict[str, Tuple[float, float]] = {}
        mia_attacks = list(getattr(self, 'mia_attacks', ['loss']))

        # ── loss (Yeom threshold) ──────────────────────────────────────────────
        if 'loss' in mia_attacks:
            attack_result = tf_attack(
                logits_train=logits_members,
                logits_test=logits_test[rand_idxs],
                loss_train=loss_members,
                loss_test=loss_test[rand_idxs],
                train_labels=labels_members,
                test_labels=labels_test[rand_idxs],
                run_extended=False,
            )
            results['loss'] = (float(attack_result.get_yeom_auc()), float(attack_result.get_yeom_tpr_at_fpr()))

        # ── scaled_logit (LiRA single-model) ──────────────────────────────────
        if 'scaled_logit' in mia_attacks:
            m_eff = len(logits_members)
            scores_m  = lira_scaled_logit_score(logits_members,         labels_members)
            scores_nm = lira_scaled_logit_score(logits_test[rand_idxs], labels_test[rand_idxs])
            y_true = np.concatenate([np.ones(m_eff, dtype=int), np.zeros(m_eff, dtype=int)])
            scores = np.concatenate([scores_m, scores_nm])
            auc = _safe_auc_and_adv(y_true, scores)[0]
            tpr = _tpr_at_fpr(y_true, scores)
            results['scaled_logit'] = (float(auc), float(tpr))

        # ── rmia ──────────────────────────────────────────────────────────────
        if 'rmia' in mia_attacks:
            if rmia_probs_full is None:
                results['rmia'] = (0.5, 0.01)
            else:
                probs_tgt_m, probs_ref_m, probs_tgt_nm_full, probs_ref_nm_full = rmia_probs_full
                auc, tpr = _rmia_attack(
                    probs_tgt_m, probs_ref_m,
                    probs_tgt_nm_full[rand_idxs], probs_ref_nm_full[rand_idxs],
                )
                results['rmia'] = (float(auc), float(tpr))

        return results

    def _compute_mia_stable(self, model: Any, forget_loader: DataLoader, m: int, is_torch_model: bool, run_id: int,
                            retain_loader: Optional[DataLoader] = None, forget_ratio: Optional[float] = None,
                            X_retain: Optional[np.ndarray] = None, y_retain: Optional[np.ndarray] = None) -> Tuple[float, float]:
        """Run stable MIA with multiple resamples; returns (mean_auc, mean_tpr) for primary attack.

        Primary attack is the first entry in self.mia_attacks (default: 'loss').
        Additional attacks are computed but their aggregated mean is returned only for the primary.
        """
        R = max(1, int(self.mia_resamples) if self.mia_resamples is not None else 1)
        rng = np.random.default_rng(int(self.seed) + int(run_id) + 1337)
        mia_attacks = list(getattr(self, 'mia_attacks', ['loss']))
        primary_atk = mia_attacks[0]

        if is_torch_model:
            was_training = getattr(model, "training", False)
            model.eval()

            logits_test, loss_test, labels_test = compute_attack_components(model, self._torch_test_loader_cached)
            logits_mem, loss_mem, labels_mem = compute_attack_components(model, forget_loader)

            if was_training:
                model.train()

            logits_test = np.asarray(logits_test)
            loss_test = np.asarray(loss_test).reshape(-1)
            labels_test = np.asarray(labels_test)

            logits_mem = np.asarray(logits_mem)
            loss_mem = np.asarray(loss_mem).reshape(-1)
            labels_mem = np.asarray(labels_mem)

            n_test = int(len(labels_test))
            m_eff = self._effective_m(m=int(len(labels_mem)), n_test=n_test)
            if m_eff < 2:
                return 0.5, 0.01

            if int(len(labels_mem)) != m_eff:
                idx = rng.choice(int(len(labels_mem)), size=m_eff, replace=False)
                logits_mem = logits_mem[idx]
                loss_mem = loss_mem[idx]
                labels_mem = labels_mem[idx]

            # Precompute RMIA reference probs once (reused across R resamples)
            rmia_probs_full = None
            if 'rmia' in mia_attacks and retain_loader is not None and forget_ratio is not None:
                from mapgu.evaluation.attacks import _to_probs
                ref_models = self._get_reference_models(retain_loader, forget_ratio, run_id)
                probs_tgt_m      = _to_probs(logits_mem)[np.arange(m_eff), labels_mem]
                probs_tgt_nm_full = _to_probs(logits_test)[np.arange(n_test), labels_test]
                probs_ref_m      = np.stack([self._extract_true_class_probs(rm, forget_loader) for rm in ref_models]).mean(axis=0)
                if len(probs_ref_m) != m_eff:
                    probs_ref_m = probs_ref_m[idx]  # apply same member subsample
                probs_ref_nm_full = np.stack([self._extract_true_class_probs(rm, self._torch_test_loader_cached) for rm in ref_models]).mean(axis=0)
                rmia_probs_full = (probs_tgt_m, probs_ref_m, probs_tgt_nm_full, probs_ref_nm_full)

            aucs_per_atk: Dict[str, List[float]] = {a: [] for a in mia_attacks}
            tprs_per_atk: Dict[str, List[float]] = {a: [] for a in mia_attacks}
            for _ in range(R):
                ridx = rng.choice(n_test, size=m_eff, replace=False)
                res = self._mia_from_precomputed(
                    logits_members=logits_mem,
                    loss_members=loss_mem,
                    labels_members=labels_mem,
                    logits_test=logits_test,
                    loss_test=loss_test,
                    labels_test=labels_test,
                    rand_idxs=ridx,
                    rmia_probs_full=rmia_probs_full,
                )
                for atk in mia_attacks:
                    a, t = res.get(atk, (0.5, 0.01))
                    aucs_per_atk[atk].append(a)
                    tprs_per_atk[atk].append(t)
            return float(np.mean(aucs_per_atk[primary_atk])), float(np.mean(tprs_per_atk[primary_atk]))

        # xgboost branch
        test_preds = np.asarray(model.predict_proba(self.X_test))
        forget_preds = np.asarray(model.predict_proba(self.X_forget))

        y_test_one_hot = self.encoder.fit_transform(self.y_test.reshape(-1, 1))
        y_forget_one_hot = self.encoder.transform(self.y_forget.reshape(-1, 1))

        loss_test = np.array([sk_metrics.log_loss(y_test_one_hot[i], test_preds[i]) for i in range(len(self.y_test))], dtype=float)
        loss_forget = np.array([sk_metrics.log_loss(y_forget_one_hot[i], forget_preds[i]) for i in range(len(self.y_forget))], dtype=float)

        n_test = int(len(self.y_test))
        m_eff = self._effective_m(m=int(len(self.y_forget)), n_test=n_test)
        if m_eff < 2:
            return 0.5, 0.01

        if int(len(self.y_forget)) != m_eff:
            idx = rng.choice(int(len(self.y_forget)), size=m_eff, replace=False)
            forget_preds = forget_preds[idx]
            loss_forget = loss_forget[idx]
            y_forget = np.asarray(self.y_forget)[idx]
            X_forget_eff = self.X_forget[idx]
        else:
            y_forget = np.asarray(self.y_forget)
            X_forget_eff = self.X_forget

        y_test = np.asarray(self.y_test)

        # Precompute RMIA reference probs once for XGBoost
        xgb_rmia_probs_full = None
        if 'rmia' in mia_attacks and X_retain is not None and y_retain is not None and forget_ratio is not None:
            from mapgu.evaluation.attacks import rmia_attack as _rmia_attack
            ref_models = self._get_reference_models_predict_proba(X_retain, y_retain, forget_ratio, run_id)
            probs_tgt_m       = forget_preds[np.arange(m_eff), y_forget]
            probs_tgt_nm_full = test_preds[np.arange(n_test), y_test]
            probs_ref_m       = np.stack([self._extract_true_class_probs_predict_proba(rm, X_forget_eff, y_forget) for rm in ref_models]).mean(axis=0)
            probs_ref_nm_full = np.stack([self._extract_true_class_probs_predict_proba(rm, self.X_test, y_test) for rm in ref_models]).mean(axis=0)
            xgb_rmia_probs_full = (probs_tgt_m, probs_ref_m, probs_tgt_nm_full, probs_ref_nm_full)

        mia_attacks = list(getattr(self, 'mia_attacks', ['loss']))
        primary_atk = mia_attacks[0]
        aucs_per_atk: Dict[str, List[float]] = {a: [] for a in mia_attacks}
        tprs_per_atk: Dict[str, List[float]] = {a: [] for a in mia_attacks}
        for _ in range(R):
            ridx = rng.choice(n_test, size=m_eff, replace=False)
            if 'loss' in mia_attacks:
                attack_result = tf_attack(
                    logits_train=forget_preds,
                    logits_test=test_preds[ridx],
                    loss_train=loss_forget,
                    loss_test=loss_test[ridx],
                    train_labels=y_forget,
                    test_labels=y_test[ridx],
                    run_extended=False,
                )
                aucs_per_atk['loss'].append(float(attack_result.get_yeom_auc()))
                tprs_per_atk['loss'].append(float(attack_result.get_yeom_tpr_at_fpr()))
            if 'rmia' in mia_attacks:
                if xgb_rmia_probs_full is None:
                    aucs_per_atk['rmia'].append(0.5)
                    tprs_per_atk['rmia'].append(0.01)
                else:
                    pt_m, pr_m, pt_nm, pr_nm = xgb_rmia_probs_full
                    a, t = _rmia_attack(pt_m, pr_m, pt_nm[ridx], pr_nm[ridx])
                    aucs_per_atk['rmia'].append(a)
                    tprs_per_atk['rmia'].append(t)
        return float(np.mean(aucs_per_atk[primary_atk])), float(np.mean(tprs_per_atk[primary_atk]))

    # ----------------------------------------------------------------------------------
    # Prepared data loaders
    # ----------------------------------------------------------------------------------
    def _load_dp_data(self, eps: float) -> Optional[DataLoader]:
        try:
            if self.dataset == "adult":
                dp_path = _resolve_dp_tabular_csv("adult", eps)
                dp_data = pd.read_csv(dp_path, sep=r" *, *", engine="python", na_values="?")
                dp_data.dropna(inplace=True)
                dp_data["income"] = dp_data["income"].astype(str).apply(
                    lambda x: 0 if x.strip() in ["<=50K", "<=50K."] else 1
                )

                X = self.preprocessor.transform(dp_data.drop("income", axis=1))
                y = dp_data["income"].values
                return self._torch_loader_from_numpy(X, y, shuffle=True)

            if self.dataset == "credit":
                dp_path = _resolve_dp_tabular_csv("credit", eps)
                dp_data = pd.read_csv(dp_path)
                dp_data.dropna(inplace=True)
                y = dp_data["SeriousDlqin2yrs"].values
                X = self.preprocessor.transform(dp_data.drop("SeriousDlqin2yrs", axis=1).values)
                return self._torch_loader_from_numpy(X, y, shuffle=True)

            if self.dataset == "heart":
                dp_path = _resolve_dp_tabular_csv("heart", eps)
                dp_data = pd.read_csv(dp_path)
                dp_data.dropna(inplace=True)

                if "id" in dp_data.columns:
                    dp_data = dp_data.drop(columns=["id"])
                if "age" in dp_data.columns:
                    dp_data["age_years"] = dp_data["age"] / 365.25
                    dp_data = dp_data.drop(columns=["age"])

                y = dp_data["cardio"].values
                X = self.preprocessor.transform(dp_data.drop("cardio", axis=1))
                return self._torch_loader_from_numpy(X, y, shuffle=True)

            if self.dataset == "cifar10":
                npz_path = os.path.join(_dp_eps_dir("cifar10", eps), "dp_cifar10.npz")
                if not os.path.isfile(npz_path):
                    raise FileNotFoundError(npz_path)

                data = np.load(npz_path)
                X_np = data["X"].astype(np.float32) / 255.0  # [N, 3, 32, 32] in [0, 1]
                y_np = data["y"]
                mean = np.array(CIFAR_MEAN, dtype=np.float32)[:, None, None]
                std = np.array(CIFAR_STD, dtype=np.float32)[:, None, None]
                X_np = (X_np - mean) / std
                ds = TensorDataset(torch.from_numpy(X_np), torch.from_numpy(y_np))
                return DataLoader(
                    ds,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=self.pin_memory,
                )

            raise ValueError(f"Unknown dataset={self.dataset}")

        except FileNotFoundError:
            return None

    # ----------------------------------------------------------------------------------
    # Core 3-phase runner
    # ----------------------------------------------------------------------------------
    def _snapshot_phase1_model(self, model: Any, *, is_torch_model: bool) -> Any:
        if not is_torch_model:
            return copy.deepcopy(model)

        model_cpu = copy.deepcopy(model).to(torch.device("cpu"))
        return copy.deepcopy(model_cpu.state_dict())

    def _restore_phase1_model(self, payload: Any, *, is_torch_model: bool) -> Any:
        if not is_torch_model:
            return copy.deepcopy(payload)

        model = copy.deepcopy(self.initial_model)
        model.load_state_dict(copy.deepcopy(payload))
        return model.to(self.device)

    def _run_privacy_experiment(
        self,
        *,
        train_loader_private: DataLoader,
        ft_epochs: int,
        prep_time: float,
        private_loader_factory: Optional[Callable[[int], Tuple[DataLoader, float]]] = None,
        phase1_cache: Optional[Dict[int, Dict[str, Any]]] = None,
        experiment_type: str,
        k_or_eps: Any,
        forget_ratio: float,
    ) -> Dict[str, Any]:
        """
        Per repeat:
          Phase 1: train on private data (DP or k-anon)
          Phase 2: fine-tune on original full training data
          Phase 3: fine-tune on retain data
        """
        results = {
            "phase1": {"prep_times": [], "train_times": [], "total_times": [], "train_accs": [], "test_accs": [], "mia_aucs": [], "mia_tprs": []},
            "phase2": {"times": [], "train_accs": [], "test_accs": [], "mia_aucs": [], "mia_tprs": []},
            "phase3": {"times": [], "retain_accs": [], "forget_accs": [], "test_accs": [], "mia_aucs": [], "mia_tprs": []},
        }

        cfg = self._cfg_key(experiment_type, float(forget_ratio), k_or_eps, int(ft_epochs))

        for r in range(int(self.n_repeat)):
            seed_everything(int(self.seed) + r, deterministic=self.deterministic)
            torch.cuda.empty_cache()
            logger.info(f"\n  Run {r+1}/{self.n_repeat} (ft_epochs={int(ft_epochs)})")

            X_retain, y_retain, retain_loader, forget_loader, m = self._split_forget_retain_per_run(float(forget_ratio), run_id=r)
            is_torch_model = (self.model_type != "xgboost")
            phase1_cached = phase1_cache.get(int(r)) if phase1_cache is not None else None

            # -------------------------
            # Phase 1
            # -------------------------
            if phase1_cached is None:
                model_private = copy.deepcopy(self.initial_model)
                t0 = time.time()
                prep_time_run = float(prep_time)
                train_loader_private_run = train_loader_private
                if private_loader_factory is not None:
                    train_loader_private_run, prep_time_run = private_loader_factory(int(r))

                if is_torch_model:
                    model_private.to(self.device)
                    model_private.train()
                    opt = self._build_torch_optimizer(model_private, self.private_lr, optimizer_name=self.private_optimizer)
                    sched, sched_step_per_batch = self._build_torch_scheduler(
                        opt,
                        scheduler_name=self.private_scheduler,
                        epochs=int(self.max_epochs),
                        steps_per_epoch=len(train_loader_private_run),
                        max_lr=float(self.private_lr),
                    )

                    train_out = train_model(
                        model_private,
                        train_loader_private_run,
                        self._torch_test_loader_cached,
                        self.criterion,
                        opt,
                        int(self.max_epochs),
                        device=self.device,
                        verbose_epoch=max(1, int(self.max_epochs / 10)),
                        patience=None,
                        scheduler=sched,
                        scheduler_step_per_batch=sched_step_per_batch,
                        metric_fn=(self._auc if self.dataset == "credit" else self._acc),
                        metric_name=("auc" if self.dataset == "credit" else "acc"),
                        report_each_epoch=bool(self.epoch_metrics),
                        use_amp=self.use_amp,
                    )
                    model_private = self._unwrap_trained_model(train_out)
                else:
                    Xp, yp = self._extract_data_from_loader(train_loader_private_run)
                    if hasattr(model_private, "set_params"):
                        model_private.set_params(random_state=int(self.seed) + int(r))
                    model_private.fit(Xp, yp)

                phase1_train = time.time() - t0
                phase1_total = float(prep_time_run) + float(phase1_train)

                train_acc, test_acc = self._evaluate_model(model_private, is_torch_model)
                auc, tpr = self._compute_mia_stable(model_private, forget_loader, m, is_torch_model=is_torch_model, run_id=r,
                                                   retain_loader=retain_loader, forget_ratio=forget_ratio,
                                                   X_retain=X_retain if not is_torch_model else None,
                                                   y_retain=y_retain if not is_torch_model else None)
                if phase1_cache is not None:
                    phase1_cache[int(r)] = {
                        "model": self._snapshot_phase1_model(model_private, is_torch_model=is_torch_model),
                        "prep_time": float(prep_time_run),
                        "train_time": float(phase1_train),
                        "total_time": float(phase1_total),
                        "train_acc": float(train_acc),
                        "test_acc": float(test_acc),
                        "mia_auc": float(auc),
                        "mia_tpr": float(tpr),
                    }
            else:
                model_private = self._restore_phase1_model(phase1_cached["model"], is_torch_model=is_torch_model)
                prep_time_run = float(phase1_cached["prep_time"])
                phase1_train = float(phase1_cached["train_time"])
                phase1_total = float(phase1_cached["total_time"])
                train_acc = float(phase1_cached["train_acc"])
                test_acc = float(phase1_cached["test_acc"])
                auc = float(phase1_cached["mia_auc"])
                tpr = float(phase1_cached["mia_tpr"])

            results["phase1"]["prep_times"].append(float(prep_time_run))
            results["phase1"]["train_times"].append(float(phase1_train))
            results["phase1"]["total_times"].append(float(phase1_total))
            results["phase1"]["train_accs"].append(100.0 * float(train_acc))
            results["phase1"]["test_accs"].append(100.0 * float(test_acc))
            results["phase1"]["mia_aucs"].append(100.0 * float(auc))
            results["phase1"]["mia_tprs"].append(100.0 * float(tpr))

            phase1_prefix = "cached " if phase1_cached is not None else ""
            logger.info(f"    Phase1: {phase1_prefix}prep={prep_time_run:.2f}s train={phase1_train:.2f}s total={phase1_total:.2f}s "
                  f"train={train_acc*100:.2f} test={test_acc*100:.2f} mia_auc={auc*100:.2f} mia_tpr={tpr*100:.2f}")

            # -------------------------
            # Phase 2: fine-tune on full original train
            # -------------------------
            model_ft = copy.deepcopy(model_private)
            t0 = time.time()

            if is_torch_model:
                train_loader_full = self._torch_full_train_loader()
                opt = self._build_torch_optimizer(model_ft, self.ft_lr, optimizer_name=self.ft_optimizer)
                sched, sched_step_per_batch = self._build_torch_scheduler(
                    opt,
                    scheduler_name=self.ft_scheduler,
                    epochs=int(ft_epochs),
                    steps_per_epoch=len(train_loader_full),
                    max_lr=float(self.ft_lr),
                )

                train_out = train_model(
                    model_ft,
                    train_loader_full,
                    self._torch_test_loader_cached,
                    self.criterion,
                    opt,
                    int(ft_epochs),
                    device=self.device,
                    verbose_epoch=max(1, int(ft_epochs / 10)),
                    patience=None,
                    scheduler=sched,
                    scheduler_step_per_batch=sched_step_per_batch,
                    metric_fn=(self._auc if self.dataset == "credit" else self._acc),
                    metric_name=("auc" if self.dataset == "credit" else "acc"),
                    report_each_epoch=bool(self.epoch_metrics),
                    use_amp=self.use_amp,
                )
                model_ft = self._unwrap_trained_model(train_out)
            else:
                xgb_ft_lr = float(self.xgb_lr) if getattr(self, "xgb_lr", None) is not None else 0.5
                model_ft.set_params(
                    learning_rate=xgb_ft_lr,
                    n_estimators=int(ft_epochs),
                    random_state=int(self.seed) + int(r),
                )
                model_ft.fit(self.X_train, self.y_train, xgb_model=model_private)

            phase2_time = time.time() - t0
            train_acc, test_acc = self._evaluate_model(model_ft, is_torch_model)
            auc, tpr = self._compute_mia_stable(model_ft, forget_loader, m, is_torch_model=is_torch_model, run_id=r,
                                               retain_loader=retain_loader, forget_ratio=forget_ratio,
                                               X_retain=X_retain if not is_torch_model else None,
                                               y_retain=y_retain if not is_torch_model else None)

            results["phase2"]["times"].append(float(phase2_time))
            results["phase2"]["train_accs"].append(100.0 * float(train_acc))
            results["phase2"]["test_accs"].append(100.0 * float(test_acc))
            results["phase2"]["mia_aucs"].append(100.0 * float(auc))
            results["phase2"]["mia_tprs"].append(100.0 * float(tpr))

            logger.info(f"    Phase2: ft_epochs={int(ft_epochs)} time={phase2_time:.2f}s train={train_acc*100:.2f} test={test_acc*100:.2f} "
                  f"mia_auc={auc*100:.2f} mia_tpr={tpr*100:.2f}")

            # -------------------------
            # Phase 3: fine-tune on retain only (start from Phase 1 like your original)
            # -------------------------
            model_retain = copy.deepcopy(model_private)
            t0 = time.time()

            if is_torch_model:
                opt = self._build_torch_optimizer(model_retain, self.ft_lr, optimizer_name=self.ft_optimizer)
                sched, sched_step_per_batch = self._build_torch_scheduler(
                    opt,
                    scheduler_name=self.ft_scheduler,
                    epochs=int(ft_epochs),
                    steps_per_epoch=len(retain_loader),
                    max_lr=float(self.ft_lr),
                )

                train_out = train_model(
                    model_retain,
                    retain_loader,
                    self._torch_test_loader_cached,
                    self.criterion,
                    opt,
                    int(ft_epochs),
                    device=self.device,
                    verbose_epoch=max(1, int(ft_epochs / 10)),
                    patience=None,
                    scheduler=sched,
                    scheduler_step_per_batch=sched_step_per_batch,
                    metric_fn=(self._auc if self.dataset == "credit" else self._acc),
                    metric_name=("auc" if self.dataset == "credit" else "acc"),
                    report_each_epoch=bool(self.epoch_metrics),
                    use_amp=self.use_amp,
                )
                model_retain = self._unwrap_trained_model(train_out)
            else:
                xgb_ft_lr = float(self.xgb_lr) if getattr(self, "xgb_lr", None) is not None else 0.5
                model_retain.set_params(
                    learning_rate=xgb_ft_lr,
                    n_estimators=int(ft_epochs),
                    random_state=int(self.seed) + int(r),
                )
                model_retain.fit(X_retain, y_retain, xgb_model=model_private)

            phase3_time = time.time() - t0
            r_acc, f_acc, t_acc = self._evaluate_retain_forget_test(
                model_retain,
                retain_loader=retain_loader,
                forget_loader=forget_loader,
                is_torch_model=is_torch_model,
                X_retain=X_retain,
                y_retain=y_retain,
            )
            auc, tpr = self._compute_mia_stable(model_retain, forget_loader, m, is_torch_model=is_torch_model, run_id=r,
                                               retain_loader=retain_loader, forget_ratio=forget_ratio,
                                               X_retain=X_retain if not is_torch_model else None,
                                               y_retain=y_retain if not is_torch_model else None)

            results["phase3"]["times"].append(float(phase3_time))
            results["phase3"]["retain_accs"].append(100.0 * float(r_acc))
            results["phase3"]["forget_accs"].append(100.0 * float(f_acc))
            results["phase3"]["test_accs"].append(100.0 * float(t_acc))
            results["phase3"]["mia_aucs"].append(100.0 * float(auc))
            results["phase3"]["mia_tprs"].append(100.0 * float(tpr))

            logger.info(f"    Phase3: ft_epochs={int(ft_epochs)} time={phase3_time:.2f}s retain={r_acc*100:.2f} forget={f_acc*100:.2f} test={t_acc*100:.2f} "
                  f"mia_auc={auc*100:.2f} mia_tpr={tpr*100:.2f}")

        return results

    # ----------------------------------------------------------------------------------
    # Experiment front-ends
    # ----------------------------------------------------------------------------------
    def run_kanonymity(self, *, k_values: List[int], ft_epochs_list: List[int]) -> None:
        with _log_namespace("kanon"):
            cfg_path = os.path.join(self.dataset_results_dir, f"{self.model_type}_kanon_config.yaml")
            cfg = self._config_identity()
            cfg.update({
                "kanon": {
                    "k_values": [int(k) for k in k_values],
                    "kanon_mode": self.kanon_mode,
                    "cluster_repr": str(self.kanon_cluster_repr),
                    "perm_type": str(getattr(self, "kanon_perm_type", "rowwise")),
                    "embeddings_pkl": self.embeddings_pkl,
                    "kanon_seed": int(self.kanon_seed),
                },
                "training": self._config_training(),
                "private_training": {
                    "private_lr": float(self.private_lr),
                    "private_optimizer": str(self.private_optimizer),
                    "private_scheduler": str(self.private_scheduler),
                },
                "finetune": {
                    "ft_epochs": [int(e) for e in ft_epochs_list],
                    "ft_lr": float(self.ft_lr),
                    "ft_optimizer": str(self.ft_optimizer),
                    "ft_scheduler": str(self.ft_scheduler),
                },
                "runtime": self._config_runtime(),
                "xgboost": self._config_xgboost(),
                "mia": self._config_mia(),
            })
            save_config_yaml(cfg_path, cfg)
            logger.info(f"Saved kanon config to {cfg_path}")

            mode_label = "REGENERATE" if self.kanon_mode == "regenerate" else "LOAD prepared"
            logger.info(f"Running k-anonymity ({mode_label})")
            logger.info(f"cluster_repr={self.kanon_cluster_repr} seed={self.seed}")

            for fr in self.forget_ratios:
                for k in k_values:
                    phase1_cache: Dict[int, Dict[str, Any]] = {}
                    for ft_epochs in ft_epochs_list:
                        if self._maybe_stop_for_max_configs():
                            logger.info(f"[stop] reached --max_configs={self.max_configs}")
                            return

                        if self.resume and self._is_config_done("k-anonymity", float(fr), int(k), int(ft_epochs)):
                            cfg = self._cfg_key("k-anonymity", float(fr), int(k), int(ft_epochs))
                            logger.info(f"[resume][skip] {cfg}")
                            self._log_progress("skipped_done", {"cfg": cfg})
                            continue

                        anon_path = os.path.join(
                            KANON_DIR,
                            self.dataset,
                            f"cluster={self.kanon_cluster_repr}",
                            f"seed={self.seed}",
                            f"k={int(k)}",
                            "train_anon.npz",
                        )
                        if self.kanon_mode == "prepared" and (not os.path.exists(anon_path)):
                            cfg = self._cfg_key("k-anonymity", float(fr), int(k), int(ft_epochs))
                            logger.info(f"[kanon][skip] missing {anon_path}")
                            self._log_progress("skipped_missing", {"cfg": cfg, "missing": anon_path})
                            continue

                        if self.kanon_mode == "prepared":
                            Xk, yk = load_npz_train(anon_path)
                            train_loader_k = self._torch_loader_from_numpy(Xk, yk, shuffle=True)
                            prep_time = lookup_prep_time_kanon(self.dataset, int(k), self.kanon_cluster_repr, int(self.seed))
                            private_loader_factory = None
                        else:
                            # Placeholder loader — overridden per run by private_loader_factory
                            train_loader_k = (
                                self._torch_full_train_loader() if not self.is_tabular
                                else self._torch_loader_from_numpy(self.X_train, self.y_train, shuffle=True)
                            )
                            prep_time = 0.0
                            private_loader_factory = lambda run_id, kk=int(k): self._build_kanon_loader(
                                k=int(kk),
                                seed=int(self.kanon_seed) + int(run_id),
                            )

                        cfg = self._cfg_key("k-anonymity", float(fr), int(k), int(ft_epochs))
                        self._log_progress("started", {"cfg": cfg, "prep_time": prep_time})
                        self._configs_executed += 1

                        res = self._run_privacy_experiment(
                            train_loader_private=train_loader_k,
                            ft_epochs=int(ft_epochs),
                            prep_time=float(prep_time),
                            private_loader_factory=private_loader_factory,
                            phase1_cache=phase1_cache,
                            experiment_type="k-anonymity",
                            k_or_eps=int(k),
                            forget_ratio=float(fr),
                        )
                        self._save_kanonymity_results(int(k), float(fr), int(ft_epochs), res)
                        self._log_progress("completed", {"cfg": cfg})

            self.save_and_print_summary()

    def run_differential_privacy(self, *, eps_values: List[float], ft_epochs_list: List[int]) -> None:
        with _log_namespace("dp"):
            cfg_path = os.path.join(self.dataset_results_dir, f"{self.model_type}_dp_config.yaml")
            cfg = self._config_identity()
            cfg.update({
                "dp": {
                    "eps_values": [float(e) for e in eps_values],
                },
                "training": self._config_training(),
                "private_training": {
                    "private_lr": float(self.private_lr),
                    "private_optimizer": str(self.private_optimizer),
                    "private_scheduler": str(self.private_scheduler),
                },
                "finetune": {
                    "ft_epochs": [int(e) for e in ft_epochs_list],
                    "ft_lr": float(self.ft_lr),
                    "ft_optimizer": str(self.ft_optimizer),
                    "ft_scheduler": str(self.ft_scheduler),
                },
                "runtime": self._config_runtime(),
                "xgboost": self._config_xgboost(),
                "mia": self._config_mia(),
            })
            save_config_yaml(cfg_path, cfg)
            logger.info(f"Saved dp config to {cfg_path}")

            logger.info("Running differential privacy (LOAD prepared)")
            for fr in self.forget_ratios:
                for eps in eps_values:
                    train_loader_dp = self._load_dp_data(float(eps))
                    if train_loader_dp is None:
                        for ft_epochs in ft_epochs_list:
                            cfg = self._cfg_key("differential_privacy", float(fr), float(eps), int(ft_epochs))
                            logger.info(f"[dp][skip] missing prepared data for eps={eps}")
                            self._log_progress("skipped_missing", {"cfg": cfg, "missing": f"dp(eps={eps})"})
                        continue

                    prep_time = lookup_prep_time_dp(self.dataset, float(eps), int(self.seed))
                    phase1_cache: Dict[int, Dict[str, Any]] = {}
                    for ft_epochs in ft_epochs_list:
                        if self._maybe_stop_for_max_configs():
                            logger.info(f"[stop] reached --max_configs={self.max_configs}")
                            return

                        if self.resume and self._is_config_done("differential_privacy", float(fr), float(eps), int(ft_epochs)):
                            cfg = self._cfg_key("differential_privacy", float(fr), float(eps), int(ft_epochs))
                            logger.info(f"[resume][skip] {cfg}")
                            self._log_progress("skipped_done", {"cfg": cfg})
                            continue
                        cfg = self._cfg_key("differential_privacy", float(fr), float(eps), int(ft_epochs))

                        self._log_progress("started", {"cfg": cfg, "prep_time": prep_time})
                        self._configs_executed += 1

                        res = self._run_privacy_experiment(
                            train_loader_private=train_loader_dp,
                            ft_epochs=int(ft_epochs),
                            prep_time=float(prep_time),
                            phase1_cache=phase1_cache,
                            experiment_type="differential_privacy",
                            k_or_eps=float(eps),
                            forget_ratio=float(fr),
                        )
                        self._save_dp_results(float(eps), float(fr), int(ft_epochs), res)
                        self._log_progress("completed", {"cfg": cfg})

            self.save_and_print_summary()

    # ----------------------------------------------------------------------------------
    # Saving
    # ----------------------------------------------------------------------------------
    def _save_kanonymity_results(self, k: int, forget_ratio: float, ft_epochs: int, results: Dict[str, Any]) -> None:
        base_path = self.dataset_results_dir
        _ensure_dir(base_path)

        out1 = {
            "Prep Time": _ms(results["phase1"]["prep_times"]),
            "Training Time": _ms(results["phase1"]["train_times"]),
            "Total Time": _ms(results["phase1"]["total_times"]),
            "Train Accuracy": _ms(results["phase1"]["train_accs"]),
            "Test Accuracy": _ms(results["phase1"]["test_accs"]),
            "MIA AUC": _ms(results["phase1"]["mia_aucs"]),
            "MIA TPR@1%FPR": _ms(results["phase1"]["mia_tprs"]),
        }
        f1 = os.path.join(base_path, f"{self.model_type}_mk={k}_dk_fr={forget_ratio}.csv")
        self._save_results(f1, out1)
        self._append_summary_rows(
            experiment="kanonymity",
            phase="phase1_private_train",
            forget_ratio=float(forget_ratio),
            param_name="k",
            param_value=int(k),
            metrics_dict=out1,
            ft_epochs=int(ft_epochs),
        )

        out2 = {
            "Training Time": _ms(results["phase2"]["times"]),
            "Train Accuracy": _ms(results["phase2"]["train_accs"]),
            "Test Accuracy": _ms(results["phase2"]["test_accs"]),
            "MIA AUC": _ms(results["phase2"]["mia_aucs"]),
            "MIA TPR@1%FPR": _ms(results["phase2"]["mia_tprs"]),
        }
        f2 = os.path.join(base_path, f"{self.model_type}_mk={k}_d_fr={forget_ratio}_epochs={ft_epochs}.csv")
        self._save_results(f2, out2, metadata={"FT Epochs": int(ft_epochs)})
        self._append_summary_rows(
            experiment="kanonymity",
            phase="phase2_finetune_full",
            forget_ratio=float(forget_ratio),
            param_name="k",
            param_value=int(k),
            metrics_dict=out2,
            ft_epochs=int(ft_epochs),
        )

        out3 = {
            "Training Time": _ms(results["phase3"]["times"]),
            "Retain Accuracy": _ms(results["phase3"]["retain_accs"]),
            "Forget Accuracy": _ms(results["phase3"]["forget_accs"]),
            "Test Accuracy": _ms(results["phase3"]["test_accs"]),
            "MIA AUC": _ms(results["phase3"]["mia_aucs"]),
            "MIA TPR@1%FPR": _ms(results["phase3"]["mia_tprs"]),
        }
        f3 = os.path.join(base_path, f"{self.model_type}_mk={k}_dret_fr={forget_ratio}_epochs={ft_epochs}.csv")
        self._save_results(f3, out3, metadata={"FT Epochs": int(ft_epochs)})
        self._append_summary_rows(
            experiment="kanonymity",
            phase="phase3_finetune_retain",
            forget_ratio=float(forget_ratio),
            param_name="k",
            param_value=int(k),
            metrics_dict=out3,
            ft_epochs=int(ft_epochs),
        )

        _rt = os.path.join(base_path, f"{self.model_type}_runtimes.csv")
        _p = f"k={k}"
        _fr = float(forget_ratio)
        _n1 = len(results["phase1"]["prep_times"])
        _n2 = len(results["phase2"]["times"])
        _n3 = len(results["phase3"]["times"])
        append_runtime_rows(_rt, [
            {"Method": "kanon", "Param": _p, "Forget Ratio": _fr, "Phase": "kanon_prep",
             "N Runs": _n1, "Mean (s)": out1["Prep Time"][0],      "Std (s)": out1["Prep Time"][1]},
            {"Method": "kanon", "Param": _p, "Forget Ratio": _fr, "Phase": "train_Mk",
             "N Runs": _n1, "Mean (s)": out1["Training Time"][0],  "Std (s)": out1["Training Time"][1]},
            {"Method": "kanon", "Param": _p, "Forget Ratio": _fr, "Phase": "ft_Mk_D",
             "N Runs": _n2, "Mean (s)": out2["Training Time"][0],  "Std (s)": out2["Training Time"][1], "FT Epochs": int(ft_epochs)},
            {"Method": "kanon", "Param": _p, "Forget Ratio": _fr, "Phase": "ft_Mk_Dr",
             "N Runs": _n3, "Mean (s)": out3["Training Time"][0],  "Std (s)": out3["Training Time"][1], "FT Epochs": int(ft_epochs)},
        ])

    def _save_dp_results(self, eps: float, forget_ratio: float, ft_epochs: int, results: Dict[str, Any]) -> None:
        base_path = self.dataset_results_dir
        _ensure_dir(base_path)

        out1 = {
            "Prep Time": _ms(results["phase1"]["prep_times"]),
            "Training Time": _ms(results["phase1"]["train_times"]),
            "Total Time": _ms(results["phase1"]["total_times"]),
            "Train Accuracy": _ms(results["phase1"]["train_accs"]),
            "Test Accuracy": _ms(results["phase1"]["test_accs"]),
            "MIA AUC": _ms(results["phase1"]["mia_aucs"]),
            "MIA TPR@1%FPR": _ms(results["phase1"]["mia_tprs"]),
        }
        f1 = os.path.join(base_path, f"{self.model_type}_mdp_eps={eps}_fr={forget_ratio}.csv")
        self._save_results(f1, out1)
        self._append_summary_rows(
            experiment="dp",
            phase="phase1_private_train",
            forget_ratio=float(forget_ratio),
            param_name="eps",
            param_value=float(eps),
            metrics_dict=out1,
            ft_epochs=int(ft_epochs),
        )

        out2 = {
            "Training Time": _ms(results["phase2"]["times"]),
            "Train Accuracy": _ms(results["phase2"]["train_accs"]),
            "Test Accuracy": _ms(results["phase2"]["test_accs"]),
            "MIA AUC": _ms(results["phase2"]["mia_aucs"]),
            "MIA TPR@1%FPR": _ms(results["phase2"]["mia_tprs"]),
        }
        f2 = os.path.join(base_path, f"{self.model_type}_mdpd_eps={eps}_fr={forget_ratio}_epochs={ft_epochs}.csv")
        self._save_results(f2, out2, metadata={"FT Epochs": int(ft_epochs)})
        self._append_summary_rows(
            experiment="dp",
            phase="phase2_finetune_full",
            forget_ratio=float(forget_ratio),
            param_name="eps",
            param_value=float(eps),
            metrics_dict=out2,
            ft_epochs=int(ft_epochs),
        )

        out3 = {
            "Training Time": _ms(results["phase3"]["times"]),
            "Retain Accuracy": _ms(results["phase3"]["retain_accs"]),
            "Forget Accuracy": _ms(results["phase3"]["forget_accs"]),
            "Test Accuracy": _ms(results["phase3"]["test_accs"]),
            "MIA AUC": _ms(results["phase3"]["mia_aucs"]),
            "MIA TPR@1%FPR": _ms(results["phase3"]["mia_tprs"]),
        }
        f3 = os.path.join(base_path, f"{self.model_type}_mdpret_eps={eps}_fr={forget_ratio}_epochs={ft_epochs}.csv")
        self._save_results(f3, out3, metadata={"FT Epochs": int(ft_epochs)})
        self._append_summary_rows(
            experiment="dp",
            phase="phase3_finetune_retain",
            forget_ratio=float(forget_ratio),
            param_name="eps",
            param_value=float(eps),
            metrics_dict=out3,
            ft_epochs=int(ft_epochs),
        )

        _rt = os.path.join(base_path, f"{self.model_type}_runtimes.csv")
        _p = f"eps={eps}"
        _fr = float(forget_ratio)
        _n1 = len(results["phase1"]["prep_times"])
        _n2 = len(results["phase2"]["times"])
        _n3 = len(results["phase3"]["times"])
        rt_rows = [
            {"Method": "dp", "Param": _p, "Forget Ratio": _fr, "Phase": "dp_generate",
             "N Runs": _n1, "Mean (s)": out1["Prep Time"][0],     "Std (s)": out1["Prep Time"][1]},
            {"Method": "dp", "Param": _p, "Forget Ratio": _fr, "Phase": "train_Meps",
             "N Runs": _n1, "Mean (s)": out1["Training Time"][0], "Std (s)": out1["Training Time"][1]},
            {"Method": "dp", "Param": _p, "Forget Ratio": _fr, "Phase": "ft_Meps_D",
             "N Runs": _n2, "Mean (s)": out2["Training Time"][0], "Std (s)": out2["Training Time"][1], "FT Epochs": int(ft_epochs)},
            {"Method": "dp", "Param": _p, "Forget Ratio": _fr, "Phase": "ft_Meps_Dr",
             "N Runs": _n3, "Mean (s)": out3["Training Time"][0], "Std (s)": out3["Training Time"][1], "FT Epochs": int(ft_epochs)},
        ]
        # For Adult, prepend a one-time embedding row (same cost for every eps).
        if self.dataset == "adult":
            embed_t = lookup_adult_embed_time()
            rt_rows.insert(0, {"Method": "dp", "Param": _p, "Forget Ratio": _fr,
                                "Phase": "adult_embed", "N Runs": 1,
                                "Mean (s)": embed_t, "Std (s)": 0.0})
        append_runtime_rows(_rt, rt_rows)

    # ----------------------------------------------------------------------------------
    # Small utility: extract numpy from loader (used for xgboost on private data)
    # ----------------------------------------------------------------------------------
    @staticmethod
    def _extract_data_from_loader(loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        Xs, ys = [], []
        for x, y in loader:
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().numpy()
            if isinstance(y, torch.Tensor):
                y = y.detach().cpu().numpy()
            Xs.append(x)
            ys.append(y)
        return np.vstack(Xs), np.hstack(ys)

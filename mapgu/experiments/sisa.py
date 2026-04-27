"""SISA (Sharded, Isolated, Sliced, Aggregated) unlearning experiments."""
from __future__ import annotations

import argparse
import copy
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader, Subset

from mapgu.evaluation.attacks import tf_attack, lira_scaled_logit_score, _safe_auc_and_adv, _tpr_at_fpr, rmia_attack as _rmia_attack
from mapgu.experiments.base import PrivacyBenchmark
from mapgu.models.factory import seed_everything
from mapgu.training.trainer import train_model
from mapgu.utils import (
    log_metrics_table,
    save_metrics_csv,
    save_summary_csv,
    save_config_yaml,
    get_logger,
    append_runtime_rows,
)
from mapgu.evaluation.metrics import (
    accuracy_with_majority_voting,
    auc_score_with_majority_voting,
    compute_attack_components_sisa1,
    compute_attack_components_sisa2,
)

logger = get_logger(__name__)


def _mean_std(xs: List[float]) -> Tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    arr = np.asarray(xs, dtype=float)
    return float(arr.mean()), float(arr.std())


def _print_metrics(title: str, metrics: Dict[str, Tuple[float, float]]) -> None:
    log_metrics_table(logger, metrics, title)


def _cross_entropy_from_probs(probs: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(y, dtype=int).reshape(-1)
    p = np.clip(p, 1e-12, 1.0)
    p = p / np.sum(p, axis=1, keepdims=True)
    return -np.log(p[np.arange(len(y)), y])


@dataclass
class PaperDefaults:
    batch_size: int
    train_lr: float
    retrain_lr: float
    epochs: int
    xgb_n_estimators: int | None = None
    xgb_max_depth: int | None = None
    xgb_lr: float | None = None
    xgb_reg_lambda: float | None = None


def _paper_defaults(dataset: str, model: str) -> PaperDefaults:
    if dataset == "adult" and model == "mlp":
        return PaperDefaults(batch_size=512, train_lr=1e-2, retrain_lr=1e-2, epochs=100)
    if dataset == "heart" and model == "mlp":
        return PaperDefaults(batch_size=512, train_lr=1e-2, retrain_lr=1e-2, epochs=200)
    if dataset == "credit" and model == "mlp":
        return PaperDefaults(batch_size=256, train_lr=1e-3, retrain_lr=1e-3, epochs=200)
    if dataset == "adult" and model == "xgboost":
        return PaperDefaults(batch_size=512, train_lr=0.5, retrain_lr=0.5, epochs=100, xgb_n_estimators=300, xgb_max_depth=10, xgb_lr=0.5, xgb_reg_lambda=5)
    if dataset == "heart" and model == "xgboost":
        return PaperDefaults(batch_size=512, train_lr=0.5, retrain_lr=0.5, epochs=200, xgb_n_estimators=200, xgb_max_depth=7, xgb_lr=0.5, xgb_reg_lambda=5)
    if dataset == "credit" and model == "xgboost":
        return PaperDefaults(batch_size=256, train_lr=0.5, retrain_lr=0.5, epochs=200, xgb_n_estimators=200, xgb_max_depth=9, xgb_lr=0.5, xgb_reg_lambda=5)
    if dataset == "cifar10" and model in {"densenet", "resnet18"}:
        return PaperDefaults(batch_size=64, train_lr=1e-1, retrain_lr=1e-2, epochs=100)
    raise SystemExit(f"No paper defaults for dataset={dataset}, model={model}")


class SISAExperiments(PrivacyBenchmark):
    def __init__(
        self,
        *args,
        num_shards: int = 5,
        num_slices: int = 10,
        slice_epochs: int | None = None,
        ft_epochs: int | None = None,
        ft_lr: float | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_shards = int(num_shards)
        self.num_slices = int(num_slices)
        if self.num_slices <= 0:
            raise ValueError("num_slices must be >= 1")
        self.slice_epochs = int(slice_epochs) if slice_epochs is not None else None
        self.ft_epochs = int(ft_epochs) if ft_epochs is not None else int(self.max_epochs)
        self.ft_lr = float(ft_lr) if ft_lr is not None else (float(self.lr) if getattr(self, "lr", None) is not None else None)

    def _base_dataset(self):
        return self.train_dataset if self.is_tabular else self.trainset

    def _test_loader(self):
        return self.test_loader if self.is_tabular else self._cifar_test_loader

    def _full_train_loader(self):
        return self.train_loader if self.is_tabular else self._cifar_full_train_loader

    def _subset_loader(self, idxs: np.ndarray, *, shuffle: bool) -> DataLoader:
        ds = Subset(self._base_dataset(), [int(i) for i in idxs.tolist()])
        return DataLoader(ds, **self._loader_kwargs(shuffle=shuffle))

    def _epochs_per_slice(self, total_epochs: int) -> int:
        if self.slice_epochs is not None:
            return max(1, int(self.slice_epochs))
        return max(1, int(np.ceil((2.0 * float(total_epochs)) / (float(self.num_slices) + 1.0))))

    def _partition_shards(self, forget_ratio: float, run_id: int):
        n = len(self.y_train) if self.is_tabular else len(self.trainset)
        rng = np.random.default_rng(int(self.seed) + int(run_id))
        perm = np.arange(n)
        rng.shuffle(perm)

        shard_idxs = np.array_split(perm, self.num_shards)
        train_shard_loaders: List[DataLoader] = []
        retain_shard_loaders: List[DataLoader] = []
        forget_shard_loaders: List[DataLoader] = []
        retain_sets = []
        forget_sets = []
        train_shard_idxs: List[np.ndarray] = []
        retain_shard_idxs: List[np.ndarray] = []
        forget_shard_idxs: List[np.ndarray] = []
        shard_slices: List[List[np.ndarray]] = []

        for gidx in shard_idxs:
            gidx = np.asarray(gidx, dtype=int)
            local = gidx.copy()
            rng.shuffle(local)
            slices = [np.asarray(s, dtype=int) for s in np.array_split(local, self.num_slices)]
            m = int(len(local) * float(forget_ratio))
            forget_idx = local[:m]
            retain_idx = local[m:]

            train_shard_idxs.append(gidx)
            retain_shard_idxs.append(retain_idx)
            forget_shard_idxs.append(forget_idx)
            shard_slices.append(slices)

            train_shard_loaders.append(self._subset_loader(gidx, shuffle=True))
            retain_shard_loaders.append(self._subset_loader(retain_idx, shuffle=True))
            forget_shard_loaders.append(self._subset_loader(forget_idx, shuffle=False))

            retain_sets.append(Subset(self._base_dataset(), [int(i) for i in retain_idx.tolist()]))
            forget_sets.append(Subset(self._base_dataset(), [int(i) for i in forget_idx.tolist()]))

        retain_dataset = ConcatDataset(retain_sets)
        forget_dataset = ConcatDataset(forget_sets)
        retain_loader = DataLoader(retain_dataset, **self._loader_kwargs(shuffle=True))
        forget_loader = DataLoader(forget_dataset, **self._loader_kwargs(shuffle=False))
        return {
            "train_shard_loaders": train_shard_loaders,
            "retain_shard_loaders": retain_shard_loaders,
            "forget_shard_loaders": forget_shard_loaders,
            "retain_loader": retain_loader,
            "forget_loader": forget_loader,
            "m_forget": len(forget_dataset),
            "train_shard_idxs": train_shard_idxs,
            "retain_shard_idxs": retain_shard_idxs,
            "forget_shard_idxs": forget_shard_idxs,
            "shard_slices": shard_slices,
        }

    def _state_dict_cpu(self, model: torch.nn.Module) -> Dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    def _train_torch_shard_sliced_with_checkpoints(
        self,
        *,
        slices: List[np.ndarray],
        lr: float,
        total_epochs: int,
        optimizer_name: str,
        scheduler_name: str,
    ) -> Tuple[torch.nn.Module, List[Dict[str, torch.Tensor]]]:
        """
        Train one constituent model using SISA slicing:
          step i trains on union of slices[:i+1], statefully.
        Returns final model and checkpoints_after where index j is state after j slices.
        """
        model = copy.deepcopy(self.initial_model).to(self.device)
        checkpoints_after: List[Dict[str, torch.Tensor]] = [self._state_dict_cpu(model)]
        test_loader = self._test_loader()
        assert test_loader is not None
        ep_slice = self._epochs_per_slice(total_epochs)

        for i in range(len(slices)):
            cum_idx = np.concatenate(slices[: i + 1]) if i >= 0 else np.array([], dtype=int)
            loader = self._subset_loader(cum_idx, shuffle=True)
            optimizer = self._build_torch_optimizer(model, lr, optimizer_name=optimizer_name)
            scheduler, sched_step_per_batch = self._build_torch_scheduler(
                optimizer,
                scheduler_name=scheduler_name,
                epochs=int(ep_slice),
                steps_per_epoch=max(1, len(loader)),
                max_lr=float(lr),
            )
            out = train_model(
                model=model,
                train_loader=loader,
                val_loader=test_loader,
                criterion=self.criterion,
                optimizer=optimizer,
                max_epochs=int(ep_slice),
                device=self.device,
                verbose_epoch=max(1, int(ep_slice / 2)),
                patience=None,
                scheduler=scheduler,
                scheduler_step_per_batch=sched_step_per_batch,
                report_each_epoch=False,
                show_progress=False,
                use_amp=self.use_amp,
            )
            model = self._unwrap_trained_model(out)
            checkpoints_after.append(self._state_dict_cpu(model))
        return model, checkpoints_after

    def _retrain_torch_shard_from_affected_slice(
        self,
        *,
        slices: List[np.ndarray],
        forget_idx_set: set[int],
        checkpoints_after: List[Dict[str, torch.Tensor]],
        lr: float,
        total_epochs: int,
        optimizer_name: str,
        scheduler_name: str,
    ) -> torch.nn.Module:
        affected = [i for i, s in enumerate(slices) if any(int(x) in forget_idx_set for x in s.tolist())]
        if not affected:
            model = copy.deepcopy(self.initial_model).to(self.device)
            model.load_state_dict(checkpoints_after[-1])
            return model

        start = int(min(affected))
        model = copy.deepcopy(self.initial_model).to(self.device)
        model.load_state_dict(checkpoints_after[start])

        test_loader = self._test_loader()
        assert test_loader is not None
        ep_slice = self._epochs_per_slice(total_epochs)

        for i in range(start, len(slices)):
            cum_idx = np.concatenate(slices[: i + 1])
            keep = np.asarray([int(x) for x in cum_idx.tolist() if int(x) not in forget_idx_set], dtype=int)
            if keep.size == 0:
                continue
            loader = self._subset_loader(keep, shuffle=True)
            optimizer = self._build_torch_optimizer(model, lr, optimizer_name=optimizer_name)
            scheduler, sched_step_per_batch = self._build_torch_scheduler(
                optimizer,
                scheduler_name=scheduler_name,
                epochs=int(ep_slice),
                steps_per_epoch=max(1, len(loader)),
                max_lr=float(lr),
            )
            out = train_model(
                model=model,
                train_loader=loader,
                val_loader=test_loader,
                criterion=self.criterion,
                optimizer=optimizer,
                max_epochs=int(ep_slice),
                device=self.device,
                verbose_epoch=max(1, int(ep_slice / 2)),
                patience=None,
                scheduler=scheduler,
                scheduler_step_per_batch=sched_step_per_batch,
                report_each_epoch=False,
                show_progress=False,
                use_amp=self.use_amp,
            )
            model = self._unwrap_trained_model(out)
        return model

    def _train_sharded_xgb_models(self, shard_idxs: List[np.ndarray], *, run_id: int, n_estimators: int) -> List[Any]:
        models: List[Any] = []
        for sid, idx in enumerate(shard_idxs):
            model = copy.deepcopy(self.initial_model)
            if hasattr(model, "set_params"):
                model.set_params(
                    n_estimators=int(n_estimators),
                    random_state=int(self.seed) + int(run_id) * 1000 + int(sid),
                )
            model.fit(self.X_train[idx], self.y_train[idx])
            models.append(model)
        return models

    def _xgb_ensemble_proba(self, models: List[Any], X: np.ndarray) -> np.ndarray:
        probs = [np.asarray(m.predict_proba(X), dtype=np.float64) for m in models]
        return np.mean(np.stack(probs, axis=0), axis=0)

    def _xgb_metric(self, models: List[Any], X: np.ndarray, y: np.ndarray) -> float:
        probs = self._xgb_ensemble_proba(models, X)
        if self.dataset == "credit":
            return float(roc_auc_score(y, probs[:, 1]))
        pred = np.argmax(probs, axis=1)
        return float(accuracy_score(y, pred))

    def _sisa_ensemble_true_class_probs_torch(
        self, models: List[torch.nn.Module], loader: DataLoader
    ) -> np.ndarray:
        """Return ensemble-averaged P(y_true|x) for each sample in loader."""
        for m in models:
            m.eval()
        per_model: List[List[np.ndarray]] = [[] for _ in models]
        ys: List[np.ndarray] = []
        with torch.no_grad():
            for batch in loader:
                x, y = batch[0].to(self.device), batch[1]
                ys.append(y.numpy())
                for i, m in enumerate(models):
                    p = torch.softmax(m(x), dim=1).cpu().numpy()
                    per_model[i].append(p)
        y_all = np.concatenate(ys)
        # (n_models, n_samples, n_classes) → average over models
        avg_proba = np.stack([np.vstack(parts) for parts in per_model]).mean(axis=0)
        return avg_proba[np.arange(len(y_all)), y_all]

    def _sisa_mia_compute(
        self,
        logits_forget: np.ndarray,
        loss_forget: np.ndarray,
        forget_labels: np.ndarray,
        logits_test: np.ndarray,
        loss_test: np.ndarray,
        test_labels: np.ndarray,
        ridx: np.ndarray,
        rmia_probs: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = None,
    ) -> Tuple[float, float]:
        """Run mia_attacks on SISA ensemble logits; return (auc, tpr) for primary attack.

        rmia_probs: (probs_tgt_m, probs_ref_m, probs_tgt_nm, probs_ref_nm) pre-computed
        true-class probabilities for RMIA. If None and 'rmia' is requested, falls back to
        random baseline.
        """
        mia_attacks = list(getattr(self, 'mia_attacks', ['loss']))
        primary_atk = mia_attacks[0]
        m_eff = len(logits_forget)
        res: Dict[str, Tuple[float, float]] = {}

        # ── loss ──────────────────────────────────────────────────────────────
        if 'loss' in mia_attacks:
            ar = tf_attack(
                logits_train=logits_forget,
                logits_test=logits_test[ridx],
                loss_train=loss_forget,
                loss_test=loss_test[ridx],
                train_labels=forget_labels,
                test_labels=test_labels[ridx],
                run_extended=False,
            )
            res['loss'] = (float(ar.get_yeom_auc()), float(ar.get_yeom_tpr_at_fpr()))

        # ── scaled_logit ───────────────────────────────────────────────────────
        if 'scaled_logit' in mia_attacks:
            scores_m  = lira_scaled_logit_score(logits_forget,         forget_labels)
            scores_nm = lira_scaled_logit_score(logits_test[ridx],     test_labels[ridx])
            y_true = np.concatenate([np.ones(m_eff, dtype=int), np.zeros(m_eff, dtype=int)])
            scores = np.concatenate([scores_m, scores_nm])
            auc = _safe_auc_and_adv(y_true, scores)[0]
            tpr = _tpr_at_fpr(y_true, scores)
            res['scaled_logit'] = (float(auc), float(tpr))

        # ── rmia ──────────────────────────────────────────────────────────────
        if 'rmia' in mia_attacks:
            if rmia_probs is None:
                logger.warning("RMIA probs not provided — returning random baseline.")
                res['rmia'] = (0.5, 0.01)
            else:
                probs_tgt_m, probs_ref_m, probs_tgt_nm, probs_ref_nm = rmia_probs
                auc, tpr = _rmia_attack(probs_tgt_m, probs_ref_m, probs_tgt_nm, probs_ref_nm)
                res['rmia'] = (float(auc), float(tpr))

        a, t = res.get(primary_atk, (0.5, 0.01))
        return float(a), float(t)

    def _sisa_mia_torch_baseline(
        self,
        models: List[torch.nn.Module],
        forget_shard_loaders: List[DataLoader],
        forget_loader_combined: DataLoader,
        m: int,
        run_id: int,
        retain_loader: Optional[DataLoader] = None,
        forget_ratio: Optional[float] = None,
    ) -> Tuple[float, float]:
        test_loader = self._test_loader()
        assert test_loader is not None
        rng = np.random.default_rng(int(self.seed) + int(run_id) + 1777)
        test_n = len(self.test_dataset) if self.is_tabular else len(self.testset)
        ridx = rng.choice(test_n, size=min(m, test_n), replace=False)

        logits_test, loss_test, test_labels = compute_attack_components_sisa1(models, test_loader, device=self.device)
        logits_forget, loss_forget, forget_labels = compute_attack_components_sisa2(models, forget_shard_loaders, device=self.device)
        ridx = ridx[: len(logits_test)]

        rmia_probs = None
        if 'rmia' in getattr(self, 'mia_attacks', []) and retain_loader is not None and forget_ratio is not None:
            ref_models = self._get_reference_models(retain_loader, forget_ratio, run_id)
            probs_tgt_m  = self._sisa_ensemble_true_class_probs_torch(models, forget_loader_combined)
            probs_tgt_nm = self._sisa_ensemble_true_class_probs_torch(models, test_loader)[ridx]
            probs_ref_m  = np.stack([self._extract_true_class_probs(rm, forget_loader_combined) for rm in ref_models]).mean(axis=0)
            probs_ref_nm = np.stack([self._extract_true_class_probs(rm, test_loader)[ridx]       for rm in ref_models]).mean(axis=0)
            rmia_probs = (probs_tgt_m, probs_ref_m, probs_tgt_nm, probs_ref_nm)

        return self._sisa_mia_compute(
            np.asarray(logits_forget), np.asarray(loss_forget), np.asarray(forget_labels),
            np.asarray(logits_test), np.asarray(loss_test), np.asarray(test_labels), ridx,
            rmia_probs=rmia_probs,
        )

    def _sisa_mia_torch_retrain(
        self,
        models: List[torch.nn.Module],
        forget_loader: DataLoader,
        m: int,
        run_id: int,
        retain_loader: Optional[DataLoader] = None,
        forget_ratio: Optional[float] = None,
    ) -> Tuple[float, float]:
        test_loader = self._test_loader()
        assert test_loader is not None
        rng = np.random.default_rng(int(self.seed) + int(run_id) + 1777)
        test_n = len(self.test_dataset) if self.is_tabular else len(self.testset)
        ridx = rng.choice(test_n, size=min(m, test_n), replace=False)

        logits_test, loss_test, test_labels = compute_attack_components_sisa1(models, test_loader, device=self.device)
        logits_forget, loss_forget, forget_labels = compute_attack_components_sisa1(models, forget_loader, device=self.device)
        ridx = ridx[: len(logits_test)]

        rmia_probs = None
        if 'rmia' in getattr(self, 'mia_attacks', []) and retain_loader is not None and forget_ratio is not None:
            ref_models = self._get_reference_models(retain_loader, forget_ratio, run_id)
            probs_tgt_m  = self._sisa_ensemble_true_class_probs_torch(models, forget_loader)
            probs_tgt_nm = self._sisa_ensemble_true_class_probs_torch(models, test_loader)[ridx]
            probs_ref_m  = np.stack([self._extract_true_class_probs(rm, forget_loader)     for rm in ref_models]).mean(axis=0)
            probs_ref_nm = np.stack([self._extract_true_class_probs(rm, test_loader)[ridx] for rm in ref_models]).mean(axis=0)
            rmia_probs = (probs_tgt_m, probs_ref_m, probs_tgt_nm, probs_ref_nm)

        return self._sisa_mia_compute(
            np.asarray(logits_forget), np.asarray(loss_forget), np.asarray(forget_labels),
            np.asarray(logits_test), np.asarray(loss_test), np.asarray(test_labels), ridx,
            rmia_probs=rmia_probs,
        )

    def _sisa_mia_xgb_baseline(
        self,
        models: List[Any],
        forget_shard_idxs: List[np.ndarray],
        m: int,
        run_id: int,
        X_retain: Optional[np.ndarray] = None,
        y_retain: Optional[np.ndarray] = None,
        forget_ratio: Optional[float] = None,
    ) -> Tuple[float, float]:
        rng = np.random.default_rng(int(self.seed) + int(run_id) + 1777)
        test_n = len(self.y_test)
        m_eff = min(int(m), int(test_n))
        ridx = rng.choice(test_n, size=m_eff, replace=False)

        probs_test = self._xgb_ensemble_proba(models, self.X_test)
        loss_test = _cross_entropy_from_probs(probs_test, self.y_test)

        probs_forget_parts = []
        y_forget_parts = []
        for mdl, fidx in zip(models, forget_shard_idxs):
            if len(fidx) == 0:
                continue
            probs_forget_parts.append(np.asarray(mdl.predict_proba(self.X_train[fidx]), dtype=np.float64))
            y_forget_parts.append(self.y_train[fidx])
        probs_forget = np.vstack(probs_forget_parts)
        y_forget = np.hstack(y_forget_parts)
        loss_forget = _cross_entropy_from_probs(probs_forget, y_forget)

        rmia_probs = None
        if 'rmia' in getattr(self, 'mia_attacks', []) and X_retain is not None and y_retain is not None and forget_ratio is not None:
            ref_models = self._get_reference_models_predict_proba(X_retain, y_retain, forget_ratio, run_id)
            y_test_eff = np.asarray(self.y_test)[ridx]
            # Ensemble average P(y_true|x) from target shard models
            probs_tgt_m  = self._xgb_ensemble_proba(models, self.X_train[np.concatenate([fi for fi in forget_shard_idxs if len(fi) > 0])])[np.arange(len(y_forget)), y_forget]
            probs_tgt_nm = self._xgb_ensemble_proba(models, self.X_test)[ridx, y_test_eff]
            probs_ref_m  = np.stack([self._extract_true_class_probs_predict_proba(rm, self.X_train[np.concatenate([fi for fi in forget_shard_idxs if len(fi) > 0])], y_forget) for rm in ref_models]).mean(axis=0)
            probs_ref_nm = np.stack([self._extract_true_class_probs_predict_proba(rm, self.X_test[ridx], y_test_eff) for rm in ref_models]).mean(axis=0)
            rmia_probs = (probs_tgt_m, probs_ref_m, probs_tgt_nm, probs_ref_nm)

        return self._sisa_mia_compute(
            probs_forget, loss_forget, y_forget,
            probs_test, loss_test, self.y_test, ridx,
            rmia_probs=rmia_probs,
        )

    def _sisa_mia_xgb_retrain(
        self,
        models: List[Any],
        forget_idx_all: np.ndarray,
        m: int,
        run_id: int,
        X_retain: Optional[np.ndarray] = None,
        y_retain: Optional[np.ndarray] = None,
        forget_ratio: Optional[float] = None,
    ) -> Tuple[float, float]:
        rng = np.random.default_rng(int(self.seed) + int(run_id) + 1777)
        test_n = len(self.y_test)
        m_eff = min(int(m), int(test_n))
        ridx = rng.choice(test_n, size=m_eff, replace=False)

        probs_test = self._xgb_ensemble_proba(models, self.X_test)
        loss_test = _cross_entropy_from_probs(probs_test, self.y_test)

        X_forget = self.X_train[forget_idx_all]
        y_forget = self.y_train[forget_idx_all]
        probs_forget = self._xgb_ensemble_proba(models, X_forget)
        loss_forget = _cross_entropy_from_probs(probs_forget, y_forget)

        rmia_probs = None
        if 'rmia' in getattr(self, 'mia_attacks', []) and X_retain is not None and y_retain is not None and forget_ratio is not None:
            ref_models = self._get_reference_models_predict_proba(X_retain, y_retain, forget_ratio, run_id)
            y_test_eff = np.asarray(self.y_test)[ridx]
            probs_tgt_m  = probs_forget[np.arange(len(y_forget)), y_forget]
            probs_tgt_nm = probs_test[ridx, y_test_eff]
            probs_ref_m  = np.stack([self._extract_true_class_probs_predict_proba(rm, X_forget,       y_forget)   for rm in ref_models]).mean(axis=0)
            probs_ref_nm = np.stack([self._extract_true_class_probs_predict_proba(rm, self.X_test[ridx], y_test_eff) for rm in ref_models]).mean(axis=0)
            rmia_probs = (probs_tgt_m, probs_ref_m, probs_tgt_nm, probs_ref_nm)

        return self._sisa_mia_compute(
            probs_forget, loss_forget, y_forget,
            probs_test, loss_test, self.y_test, ridx,
            rmia_probs=rmia_probs,
        )

    def run_sisa(self) -> None:
        base_dir = self.dataset_results_dir
        os.makedirs(base_dir, exist_ok=True)
        logger.info(f"Running SISA with num_shards={self.num_shards}")

        summary_rows: List[Dict[str, Any]] = []
        for fr in self.forget_ratios:
            base_train, base_test, base_auc, base_tpr, base_t = [], [], [], [], []
            ret_acc, ret_forget, ret_test, ret_auc, ret_tpr, ret_t = [], [], [], [], [], []

            for r in range(int(self.n_repeat)):
                seed_everything(int(self.seed) + int(r), deterministic=self.deterministic)
                logger.info(f"  Run {r+1}/{self.n_repeat} (ft_epochs={int(self.ft_epochs)})")
                torch.cuda.empty_cache()
                part = self._partition_shards(float(fr), run_id=r)

                if self.model_type == "xgboost":
                    if r == 0:
                        logger.info("    Note: XGBoost uses shard-only SISA (no slicing), consistent with paper caveat for non-stateful tree learners.")
                    # Baseline
                    t0 = time.time()
                    models = self._train_sharded_xgb_models(
                        part["train_shard_idxs"],
                        run_id=r,
                        n_estimators=int(self.xgb_n_estimators) if self.xgb_n_estimators is not None else 200,
                    )
                    t1 = time.time()
                    base_t.append(t1 - t0)
                    tr = self._xgb_metric(models, self.X_train, self.y_train)
                    te = self._xgb_metric(models, self.X_test, self.y_test)
                    auc, tpr = self._sisa_mia_xgb_baseline(
                        models, part["forget_shard_idxs"], part["m_forget"], run_id=r,
                        X_retain=self.X_train[np.concatenate(part["retain_shard_idxs"])] if part["retain_shard_idxs"] else None,
                        y_retain=self.y_train[np.concatenate(part["retain_shard_idxs"])] if part["retain_shard_idxs"] else None,
                        forget_ratio=fr,
                    )
                    base_train.append(100.0 * tr)
                    base_test.append(100.0 * te)
                    base_auc.append(100.0 * auc)
                    base_tpr.append(100.0 * tpr)
                    logger.info(f"    Baseline: Train={tr*100:.2f}% | Test={te*100:.2f}% | MIA AUC={auc*100:.2f}% | MIA TPR@1%FPR={tpr*100:.2f}% | Time={t1-t0:.2f}s")

                    # Retrain
                    t0 = time.time()
                    models_r = self._train_sharded_xgb_models(
                        part["retain_shard_idxs"],
                        run_id=r,
                        n_estimators=int(self.xgb_n_estimators) if self.xgb_n_estimators is not None else 200,
                    )
                    t1 = time.time()
                    ret_t.append(t1 - t0)
                    retain_idx_all = np.concatenate(part["retain_shard_idxs"]) if part["retain_shard_idxs"] else np.array([], dtype=int)
                    forget_idx_all = np.concatenate(part["forget_shard_idxs"]) if part["forget_shard_idxs"] else np.array([], dtype=int)
                    rr = self._xgb_metric(models_r, self.X_train[retain_idx_all], self.y_train[retain_idx_all])
                    rf = self._xgb_metric(models_r, self.X_train[forget_idx_all], self.y_train[forget_idx_all])
                    rt = self._xgb_metric(models_r, self.X_test, self.y_test)
                    auc2, tpr2 = self._sisa_mia_xgb_retrain(
                        models_r, forget_idx_all, part["m_forget"], run_id=r,
                        X_retain=self.X_train[retain_idx_all] if len(retain_idx_all) > 0 else None,
                        y_retain=self.y_train[retain_idx_all] if len(retain_idx_all) > 0 else None,
                        forget_ratio=fr,
                    )
                    ret_acc.append(100.0 * rr)
                    ret_forget.append(100.0 * rf)
                    ret_test.append(100.0 * rt)
                    ret_auc.append(100.0 * auc2)
                    ret_tpr.append(100.0 * tpr2)
                    logger.info(f"    Retrain: ft_epochs={int(self.ft_epochs)} | Retain={rr*100:.2f}% | Forget={rf*100:.2f}% | Test={rt*100:.2f}% | MIA AUC={auc2*100:.2f}% | MIA TPR@1%FPR={tpr2*100:.2f}% | Time={t1-t0:.2f}s")
                else:
                    # Baseline
                    t0 = time.time()
                    models: List[torch.nn.Module] = []
                    shard_checkpoints: List[List[Dict[str, torch.Tensor]]] = []
                    for slices in part["shard_slices"]:
                        mdl, ckpts = self._train_torch_shard_sliced_with_checkpoints(
                            slices=slices,
                            lr=float(self.lr),
                            total_epochs=int(self.max_epochs),
                            optimizer_name=self.optimizer_name,
                            scheduler_name=self.scheduler_name,
                        )
                        models.append(mdl)
                        shard_checkpoints.append(ckpts)
                    t1 = time.time()
                    base_t.append(t1 - t0)

                    full_train_loader = self._full_train_loader()
                    test_loader = self._test_loader()
                    assert full_train_loader is not None and test_loader is not None
                    if self.dataset == "credit":
                        tr = auc_score_with_majority_voting(models, full_train_loader, device=self.device)
                        te = auc_score_with_majority_voting(models, test_loader, device=self.device)
                    else:
                        tr = accuracy_with_majority_voting(models, full_train_loader, device=self.device)
                        te = accuracy_with_majority_voting(models, test_loader, device=self.device)
                    auc, tpr = self._sisa_mia_torch_baseline(
                        models, part["forget_shard_loaders"], part["forget_loader"],
                        part["m_forget"], run_id=r,
                        retain_loader=part["retain_loader"], forget_ratio=fr,
                    )
                    base_train.append(100.0 * tr)
                    base_test.append(100.0 * te)
                    base_auc.append(100.0 * auc)
                    base_tpr.append(100.0 * tpr)
                    logger.info(f"    Baseline: Train={tr*100:.2f}% | Test={te*100:.2f}% | MIA AUC={auc*100:.2f}% | MIA TPR@1%FPR={tpr*100:.2f}% | Time={t1-t0:.2f}s")

                    # Retrain
                    t0 = time.time()
                    models_r: List[torch.nn.Module] = []
                    for sid, slices in enumerate(part["shard_slices"]):
                        forget_set = set(int(x) for x in np.asarray(part["forget_shard_idxs"][sid], dtype=int).tolist())
                        mdl_r = self._retrain_torch_shard_from_affected_slice(
                            slices=slices,
                            forget_idx_set=forget_set,
                            checkpoints_after=shard_checkpoints[sid],
                            lr=float(self.ft_lr),
                            total_epochs=int(self.ft_epochs),
                            optimizer_name=self.ft_optimizer_name,
                            scheduler_name=self.ft_scheduler_name,
                        )
                        models_r.append(mdl_r)
                    t1 = time.time()
                    ret_t.append(t1 - t0)

                    if self.dataset == "credit":
                        rr = auc_score_with_majority_voting(models_r, part["retain_loader"], device=self.device)
                        rf = auc_score_with_majority_voting(models_r, part["forget_loader"], device=self.device)
                        rt = auc_score_with_majority_voting(models_r, self._test_loader(), device=self.device)
                    else:
                        rr = accuracy_with_majority_voting(models_r, part["retain_loader"], device=self.device)
                        rf = accuracy_with_majority_voting(models_r, part["forget_loader"], device=self.device)
                        rt = accuracy_with_majority_voting(models_r, self._test_loader(), device=self.device)
                    auc2, tpr2 = self._sisa_mia_torch_retrain(
                        models_r, part["forget_loader"], part["m_forget"], run_id=r,
                        retain_loader=part["retain_loader"], forget_ratio=fr,
                    )
                    ret_acc.append(100.0 * rr)
                    ret_forget.append(100.0 * rf)
                    ret_test.append(100.0 * rt)
                    ret_auc.append(100.0 * auc2)
                    ret_tpr.append(100.0 * tpr2)
                    logger.info(f"    Retrain: ft_epochs={int(self.ft_epochs)} | Retain={rr*100:.2f}% | Forget={rf*100:.2f}% | Test={rt*100:.2f}% | MIA AUC={auc2*100:.2f}% | MIA TPR@1%FPR={tpr2*100:.2f}% | Time={t1-t0:.2f}s")

            _ATK_LABEL = {'loss': 'Loss', 'scaled_logit': 'ScaledLogit', 'rmia': 'RMIA'}
            _primary_atk = (self.mia_attacks[0] if hasattr(self, 'mia_attacks') and self.mia_attacks else 'loss')
            _primary_lbl = _ATK_LABEL.get(_primary_atk, _primary_atk)
            baseline_metrics = {
                "Training Time": _mean_std(base_t),
                "Train Accuracy": _mean_std(base_train),
                "Test Accuracy": _mean_std(base_test),
                f"MIA AUC ({_primary_lbl})": _mean_std(base_auc),
                f"MIA TPR@1%FPR ({_primary_lbl})": _mean_std(base_tpr),
            }
            retrain_metrics = {
                "Retraining Time": _mean_std(ret_t),
                "Retain Accuracy": _mean_std(ret_acc),
                "Forget Accuracy": _mean_std(ret_forget),
                "Test Accuracy": _mean_std(ret_test),
                f"MIA AUC ({_primary_lbl})": _mean_std(ret_auc),
                f"MIA TPR@1%FPR ({_primary_lbl})": _mean_std(ret_tpr),
            }

            _print_metrics(f"SISA BASELINE (shards={self.num_shards}, fr={fr})", baseline_metrics)
            _print_metrics(f"SISA RETRAIN (shards={self.num_shards}, fr={fr}, ft_epochs={int(self.ft_epochs)})", retrain_metrics)

            f1 = os.path.join(base_dir, f"{self.model_type}_sisa_m_d_shards={self.num_shards}_fr={fr}.csv")
            f2 = os.path.join(base_dir, f"{self.model_type}_sisa_mret_dret_shards={self.num_shards}_fr={fr}.csv")
            save_metrics_csv(f1, baseline_metrics, metadata={"FT Epochs": int(self.ft_epochs)})
            save_metrics_csv(f2, retrain_metrics, metadata={"FT Epochs": int(self.ft_epochs)})
            logger.info(f"Saved: {f1}")
            logger.info(f"Saved: {f2}")
            _rt = os.path.join(base_dir, f"{self.model_type}_runtimes.csv")
            append_runtime_rows(_rt, [
                {"Method": "sisa", "Param": f"S={self.num_shards}", "Forget Ratio": float(fr), "Phase": "train_shards_D",
                 "N Runs": len(base_t), "Mean (s)": baseline_metrics["Training Time"][0], "Std (s)": baseline_metrics["Training Time"][1], "FT Epochs": int(self.ft_epochs)},
                {"Method": "sisa", "Param": f"S={self.num_shards}", "Forget Ratio": float(fr), "Phase": "retrain_affected_shard_Dr",
                 "N Runs": len(ret_t), "Mean (s)": retrain_metrics["Retraining Time"][0], "Std (s)": retrain_metrics["Retraining Time"][1], "FT Epochs": int(self.ft_epochs)},
            ])

            for metric, (mean_val, std_val) in baseline_metrics.items():
                summary_rows.append(
                    {
                        "Experiment": "sisa_baseline",
                        "Forget Ratio": float(fr),
                        "FT Epochs": int(self.ft_epochs),
                        "Metric": str(metric),
                        "Mean": float(mean_val),
                        "Std": float(std_val),
                        "Mean±Std": f"{float(mean_val):.4f} ± {float(std_val):.4f}",
                    }
                )
            for metric, (mean_val, std_val) in retrain_metrics.items():
                summary_rows.append(
                    {
                        "Experiment": "sisa_retrain",
                        "Forget Ratio": float(fr),
                        "FT Epochs": int(self.ft_epochs),
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
            logger.info(f"{'Exp':<16} {'FR':<8} {'Metric':<22} {'Mean±Std':<20}")
            logger.info("-" * 80)
            for row in summary_rows:
                logger.info(
                    f"{str(row['Experiment']):<16} {float(row['Forget Ratio']):<8.4f} "
                    f"{str(row['Metric']):<22} {str(row['Mean±Std']):<20}"
                )
            summary_path = os.path.join(base_dir, f"{self.model_type}_sisa_shards={self.num_shards}_summary.csv")
            save_summary_csv(summary_path, summary_rows)
            logger.info(f"Saved overall SISA summary to {summary_path}")

            cfg_path = os.path.join(base_dir, f"{self.model_type}_sisa_shards={self.num_shards}_config.yaml")
            cfg = self._config_identity()
            cfg.update({
                "sisa": {
                    "num_shards": int(self.num_shards),
                    "num_slices": int(self.num_slices),
                    "slice_epochs": int(self.slice_epochs) if self.slice_epochs is not None else None,
                    "ft_epochs": int(self.ft_epochs),
                    "ft_lr": float(self.ft_lr) if getattr(self, "ft_lr", None) is not None else None,
                    "ft_optimizer": getattr(self, "ft_optimizer_name", None),
                    "ft_scheduler": getattr(self, "ft_scheduler_name", None),
                },
                "training": self._config_training(),
                "runtime": self._config_runtime(),
                "xgboost": self._config_xgboost(),
                "mia": self._config_mia(),
            })
            save_config_yaml(cfg_path, cfg)
            logger.info(f"Saved SISA config to {cfg_path}")


def main(argv: List[str] | None = None) -> None:
    p = argparse.ArgumentParser("SISA experiments")
    p.add_argument("--dataset", choices=["adult", "credit", "heart", "cifar10"], required=True)
    p.add_argument("--model", choices=["mlp", "xgboost", "densenet", "resnet18"], required=True)
    p.add_argument("--forget_ratios", nargs="+", type=float, default=[0.05])
    p.add_argument("--n_repeat", type=int, default=3)
    p.add_argument("--max_epochs", type=int, default=None)
    p.add_argument("--ft_epochs", type=int, default=None)
    p.add_argument("--num_shards", type=int, default=5)
    p.add_argument("--num_slices", type=int, default=10)
    p.add_argument("--slice_epochs", type=int, default=None, help="Per-slice epochs override; default uses paper time-equivalent formula.")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--results_subdir", type=str, default=None)
    p.add_argument("--optimizer", choices=["auto", "adam", "sgd"], default="auto")
    p.add_argument("--scheduler", choices=["none", "step", "cosine", "onecycle"], default="none")
    p.add_argument("--ft_optimizer", choices=["auto", "adam", "sgd"], default=None)
    p.add_argument("--ft_scheduler", choices=["none", "step", "cosine", "onecycle"], default=None)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--scheduler_step_size", type=int, default=30)
    p.add_argument("--scheduler_gamma", type=float, default=0.1)
    p.add_argument("--onecycle_pct_start", type=float, default=0.3)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--ft_lr", type=float, default=None)
    p.add_argument("--xgb_n_estimators", type=int, default=None)
    p.add_argument("--xgb_max_depth", type=int, default=None)
    p.add_argument("--xgb_lr", type=float, default=None)
    p.add_argument("--xgb_reg_lambda", type=float, default=None)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--cifar_download", action="store_true")
    p.add_argument("--deterministic", action="store_true")
    args = p.parse_args(argv)

    valid = {
        "adult": ["mlp", "xgboost"],
        "credit": ["mlp", "xgboost"],
        "heart": ["mlp", "xgboost"],
        "cifar10": ["densenet", "resnet18"],
    }
    if args.model not in valid[args.dataset]:
        raise SystemExit(f"Unsupported model={args.model} for dataset={args.dataset}. Valid: {valid[args.dataset]}")

    dflt = _paper_defaults(args.dataset, args.model)
    max_epochs = int(args.max_epochs) if args.max_epochs is not None else int(dflt.epochs)
    ft_epochs = int(args.ft_epochs) if args.ft_epochs is not None else int(max_epochs)
    batch_size = int(args.batch_size) if args.batch_size is not None else int(dflt.batch_size)
    lr = float(args.lr) if args.lr is not None else float(dflt.train_lr)
    ft_lr = float(args.ft_lr) if args.ft_lr is not None else float(dflt.retrain_lr)
    xgb_n_estimators = int(args.xgb_n_estimators) if args.xgb_n_estimators is not None else dflt.xgb_n_estimators
    xgb_max_depth = int(args.xgb_max_depth) if args.xgb_max_depth is not None else dflt.xgb_max_depth
    xgb_lr = float(args.xgb_lr) if args.xgb_lr is not None else dflt.xgb_lr
    xgb_reg_lambda = float(args.xgb_reg_lambda) if args.xgb_reg_lambda is not None else dflt.xgb_reg_lambda

    logger.info("Using paper-aligned defaults (override via CLI flags when needed):")
    logger.info(
        f"  dataset={args.dataset} model={args.model} batch_size={batch_size} "
        f"train_lr={lr} retrain_lr={ft_lr} train_epochs={max_epochs} retrain_epochs={ft_epochs}"
    )
    if args.model == "xgboost":
        logger.info(
            f"  xgb_n_estimators={xgb_n_estimators} xgb_max_depth={xgb_max_depth} "
            f"xgb_lr={xgb_lr} xgb_reg_lambda={xgb_reg_lambda}"
        )

    runner = SISAExperiments(
        dataset=args.dataset,
        model_type=args.model,
        forget_ratios=args.forget_ratios,
        n_repeat=args.n_repeat,
        max_epochs=max_epochs,
        results_subdir=args.results_subdir,
        ft_epochs=ft_epochs,
        ft_lr=ft_lr,
        seed=args.seed,
        num_shards=args.num_shards,
        num_slices=args.num_slices,
        slice_epochs=args.slice_epochs,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        ft_optimizer_name=args.ft_optimizer,
        ft_scheduler_name=args.ft_scheduler,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        onecycle_pct_start=args.onecycle_pct_start,
        batch_size=batch_size,
        lr=lr,
        xgb_n_estimators=xgb_n_estimators,
        xgb_max_depth=xgb_max_depth,
        xgb_lr=xgb_lr,
        xgb_reg_lambda=xgb_reg_lambda,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        cifar_download=bool(args.cifar_download),
        deterministic=bool(args.deterministic),
    )
    runner.run_sisa()


if __name__ == "__main__":
    main()

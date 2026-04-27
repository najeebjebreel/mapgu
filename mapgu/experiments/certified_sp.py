"""CERTIFIED_SP (Privacy-Aware Bayesian Inference) unlearning runner."""
from __future__ import annotations

import argparse
import copy
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from mapgu.experiments.base import PrivacyBenchmark
from mapgu.models.factory import seed_everything
from mapgu.experiments.dp import PrivacyExperiments
from mapgu.utils import get_logger, save_metrics_csv, save_summary_csv, save_config_yaml, log_metrics_table, _ms, _fmt_eps, _ensure_dir, append_runtime_rows
from mapgu.training.trainer import train_model
from mapgu.training.certified_sp_trainer import (
    train_model_certified_sp_unlearn,
    train_model_certified_sp_post_finetune,
    certified_sp_steps,
    _global_param_l2_norm,
)
logger = get_logger(__name__)


# -------------------------
# small I/O + stats helpers
# -------------------------
def _mean_std(v: List[float]) -> Tuple[float, float]:
    if not v:
        return 0.0, 0.0
    a = np.asarray(v, dtype=float)
    return float(a.mean()), float(a.std())


def _model_dimension(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))



# -------------------------
# main runner
# -------------------------
class CERTIFIED_SPRunner(PrivacyBenchmark):
    """CERTIFIED_SP unlearning runner. Uses the same optimizer/scheduler/epochs as the baseline experiment."""

    def __init__(self, *args, mia_resamples: int = 10, mia_eval_cap: int = 5000, **kwargs):
        super().__init__(*args, **kwargs)

        # Ensure criterion exists
        if not hasattr(self, "criterion") or self.criterion is None:
            self.criterion = nn.CrossEntropyLoss().to(self.device)

        self._mia_helper = PrivacyExperiments(
            dataset=self.dataset,
            model_type=self.model_type,
            forget_ratios=self.forget_ratios,
            n_repeat=self.n_repeat,
            max_epochs=self.max_epochs,
            kanon_cluster_repr=self.kanon_cluster_repr,
            embeddings_pkl=self.embeddings_pkl,
            mia_resamples=mia_resamples,
            mia_eval_cap=mia_eval_cap,
            resume=False,
            mia_attacks=list(getattr(self, 'mia_attacks', ['loss'])),
            rmia_n_ref=int(getattr(self, 'rmia_n_ref', 1)),
            results_subdir=getattr(self, "results_subdir", None),
        )

        # share loaded dataset references (so stable MIA works)
        self._mia_helper.is_tabular = self.is_tabular
        if self.is_tabular:
            self._mia_helper.X_train = self.X_train
            self._mia_helper.y_train = self.y_train
            self._mia_helper.X_test = self.X_test
            self._mia_helper.y_test = self.y_test
            self._mia_helper.train_loader = self.train_loader
            self._mia_helper.test_loader = self.test_loader
            self._mia_helper.preprocessor = self.preprocessor
            self._mia_helper.encoder = self.encoder
        else:
            self._mia_helper.trainset = self.trainset
            self._mia_helper.testset = self.testset
            self._mia_helper.batch_size = self.batch_size

        self._mia_helper.seed = self.seed
        self._mia_helper.device = self.device

    # -------------------------
    # loaders
    # -------------------------
    def _torch_test_loader(self) -> DataLoader:
        if self.is_tabular:
            return self.test_loader
        return DataLoader(self.testset, batch_size=self.batch_size, shuffle=False)

    def _torch_full_train_loader(self) -> DataLoader:
        if self.is_tabular:
            return self.train_loader
        return DataLoader(self.trainset, **self._loader_kwargs(shuffle=True))

    # -------------------------
    # optimizer / scheduler
    # -------------------------
    def _opt_adam(self, model: nn.Module, lr: float, wd: float, b1: float, b2: float, eps: float) -> optim.Optimizer:
        return optim.Adam(
            model.parameters(),
            lr=float(lr),
            weight_decay=float(wd),
            betas=(float(b1), float(b2)),
            eps=float(eps),
            foreach=False,
        )

    def _opt_sgd(self, model: nn.Module, lr: float, wd: float, momentum: float) -> optim.Optimizer:
        return optim.SGD(
            model.parameters(),
            lr=float(lr),
            momentum=float(momentum),
            weight_decay=float(wd),
        )

    def _sched_from_steps(self, opt: optim.Optimizer, schedule: str, total_steps: int, max_lr: float) -> Optional[Any]:
        schedule = str(schedule)
        if schedule == "constant":
            return None
        if schedule == "onecycle":
            return optim.lr_scheduler.OneCycleLR(
                opt,
                max_lr=float(max_lr),
                total_steps=int(total_steps),
                pct_start=0.3,
                anneal_strategy="cos",
                cycle_momentum=False,
            )
        if schedule == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(total_steps))
        raise ValueError(f"Unsupported schedule={schedule}")

    def _sched_from_epochs(self, opt: optim.Optimizer, schedule: str, epochs: int) -> Optional[Any]:
        schedule = str(schedule)
        if schedule == "constant":
            return None
        if schedule == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(epochs))
        raise ValueError(f"Unsupported epoch-based schedule={schedule}")

    @staticmethod
    def _resolve_certified_sp_optimizer_name(name: str, model_type: str) -> str:
        n = str(name).lower()
        if n == "auto":
            return "adam" if model_type == "mlp" else "sgd"
        if n not in {"adam", "sgd"}:
            raise ValueError("optimizer must be one of: auto, adam, sgd")
        return n

    # -------------------------
    # baseline training
    # -------------------------
    def _train_baseline_full(self, model: nn.Module) -> Tuple[nn.Module, float]:
        """Train on the full dataset using the same optimizer/scheduler/epochs as the baseline."""
        loader = self._torch_full_train_loader()
        opt = self._build_torch_optimizer(model, self.lr, optimizer_name=self.optimizer_name)
        scheduler, sched_step_per_batch = self._build_torch_scheduler(
            opt,
            scheduler_name=self.scheduler_name,
            epochs=int(self.max_epochs),
            steps_per_epoch=len(loader),
            max_lr=float(self.lr),
        )

        t0 = time.time()
        train_out = train_model(
            model,
            loader,
            self._torch_test_loader(),
            self.criterion,
            opt,
            int(self.max_epochs),
            device=self.device,
            verbose_epoch=max(1, int(self.max_epochs / 10)),
            scheduler=scheduler,
            scheduler_step_per_batch=sched_step_per_batch,
            metric_fn=(self._auc if self.dataset == "credit" else self._acc),
            metric_name=("auc" if self.dataset == "credit" else "acc"),
            report_each_epoch=bool(self.epoch_metrics),
            use_amp=self.use_amp,
        )
        model = self._unwrap_trained_model(train_out)
        return model, time.time() - t0

    # -------------------------
    # metrics helpers
    # -------------------------
    @torch.no_grad()
    def _eval_loader_loss(self, model: nn.Module, loader: DataLoader) -> float:
        was_training = model.training
        model.eval()
        total = 0.0
        n = 0
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            out = model(x)
            total += float(self.criterion(out, y).item())
            n += 1
        if was_training:
            model.train()
        return total / max(n, 1)

    def _metric_retain_forget_test(self, model: nn.Module, retain_loader: DataLoader, forget_loader: DataLoader) -> Tuple[float, float, float]:
        if self.dataset == "credit":
            from mapgu.evaluation.metrics import auc_score
            r = float(auc_score(model, retain_loader, device=self.device))
            f = float(auc_score(model, forget_loader, device=self.device))
            t = float(auc_score(model, self._torch_test_loader(), device=self.device))
        else:
            from mapgu.evaluation.metrics import accuracy
            r = float(accuracy(model, retain_loader, device=self.device))
            f = float(accuracy(model, forget_loader, device=self.device))
            t = float(accuracy(model, self._torch_test_loader(), device=self.device))
        return r, f, t

    def _metric_mia(self, model: nn.Module, forget_loader: DataLoader, m: int, run_id: int,
                    retain_loader: DataLoader = None, forget_ratio: float = None) -> Tuple[float, float]:
        """Run stable MIA via the helper PrivacyExperiments instance; returns (auc, tpr) for primary attack."""
        auc, tpr = self._mia_helper._compute_mia_stable(
            model, forget_loader, m, True, run_id=run_id,
            retain_loader=retain_loader, forget_ratio=forget_ratio,
        )
        return float(auc), float(tpr)

    # Public API
    def run_certified_sp(self, *args, **kwargs) -> None:
        _run_certified_sp_impl(self, *args, **kwargs)


# -------------------------
# main CERTIFIED_SP
# -------------------------
def _run_certified_sp_impl(
        self,
        *,
        forget_ratio: float,
        repeats: int,
        # parity
        unlearn_epochs: int,
        # CERTIFIED_SP params
        init_model_clip: float,
        init_model_clip_type: str,
        grad_clip: float,
        epsilon_renyi_target: float,
        delta: float,
        unlearn_lr: float,
        unlearn_weight_decay: float,
        noise_schedule: str,
        # Post phase
        post_epochs_list: List[int],
        post_steps: int,
        post_optimizer: str,
        post_lr_schedule: str,
        post_max_lr: float,
        post_weight_decay: float,
        post_unlearn_clip: Optional[float],
        # baseline init
        load_ckpt: Optional[str] = None,
        save_ckpt_dir: Optional[str] = None,
    ) -> None:
        if self.model_type not in ["mlp", "densenet", "resnet18"]:
            raise SystemExit("Supported: mlp (tabular), densenet/resnet18 (cifar10).")
        if float(unlearn_lr) <= 0:
            raise ValueError("certified_sp_lr must be > 0")

        init_model_clip_type = str(init_model_clip_type).lower()
        if init_model_clip_type not in ["clip", "clamp"]:
            raise ValueError("--certified_sp_init_model_clip_type must be clip/clamp")

        noise_schedule = str(noise_schedule)
        if noise_schedule not in ["constant", "decreasing"]:
            raise ValueError("certified_sp_noise_schedule must be constant/decreasing (train_certified_sp must support others explicitly)")

        post_lr_schedule = str(post_lr_schedule)
        if post_lr_schedule not in ["constant", "onecycle", "cosine"]:
            raise ValueError("post_lr_schedule must be constant/onecycle/cosine")

        post_epochs_values = [int(e) for e in post_epochs_list]
        if not post_epochs_values:
            raise ValueError("post_epochs_list must not be empty")
        if any(int(e) <= 0 for e in post_epochs_values):
            raise ValueError("All post_epochs values must be positive integers")
        post_epochs_values = sorted(set(post_epochs_values))
        if int(post_steps) >= 0 and len(post_epochs_values) > 1:
            raise ValueError("A post_epochs sweep requires --post_steps < 0 so steps can be derived from each epoch count")

        out_dir = self.dataset_results_dir
        _ensure_dir(out_dir)

        _eps_str = _fmt_eps(float(epsilon_renyi_target))
        _fr = float(forget_ratio)

        common_acc: Dict[str, List[float]] = {
            "baseline_time_s": [],
            "baseline_train_%": [],
            "baseline_test_%": [],
            "baseline_mia_auc_%": [],
            "baseline_mia_tpr_%": [],
            "baseline_param_l2": [],
            "model_dimension": [],
            "init_model_clip_eff": [],
            "unlearn_time_s": [],
            "unlearn_retain_%": [],
            "unlearn_forget_%": [],
            "unlearn_test_%": [],
            "unlearn_mia_auc_%": [],
            "unlearn_mia_tpr_%": [],
            "certified_sp_T": [],
            "noisy_steps": [],
            "sigma_mean": [],
            "eps_est": [],
        }
        
        def _new_post_acc() -> Dict[str, List[float]]:
            return {
                "post_time_s": [],
                "post_retain_%": [],
                "post_forget_%": [],
                "post_test_%": [],
                "post_mia_auc_%": [],
                "post_mia_tpr_%": [],
                "post_steps": [],
            }

        post_acc_by_epoch: Dict[int, Dict[str, List[float]]] = {
            int(ep): _new_post_acc() for ep in post_epochs_values
        }
        for r in range(int(repeats)):
            seed_everything(int(self.seed) + r, deterministic=self.deterministic)
            torch.cuda.empty_cache()
            logger.info(f"\n[CERTIFIED_SP] repeat {r+1}/{repeats} (forget_ratio={forget_ratio}, post_epochs={post_epochs_values})")

            if self.is_tabular:
                _, _, _, _, retain_loader, forget_loader, m = self._split_forget_retain(float(forget_ratio), run_id=r)
                forget_idxs = self._get_split_indices(float(forget_ratio), r)[1]
                self._mia_helper.X_forget = self.X_train[forget_idxs]
                self._mia_helper.y_forget = self.y_train[forget_idxs]
            else:
                _, _, _, _, retain_loader, forget_loader, m = self._split_forget_retain(float(forget_ratio), run_id=r)

            model_base = copy.deepcopy(self.initial_model)

            if load_ckpt:
                sd = torch.load(load_ckpt, map_location=self.device)
                model_base.load_state_dict(sd)
                baseline_time = 0.0
                logger.info(f"[baseline] loaded checkpoint: {load_ckpt}")
            else:
                model_base, baseline_time = self._train_baseline_full(model_base)
                if save_ckpt_dir:
                    _ensure_dir(save_ckpt_dir)
                    ckpt_path = os.path.join(save_ckpt_dir, f"{self.model_type}_certified_sp_repeat{r}.pt")
                    torch.save(model_base.state_dict(), ckpt_path)
                    logger.info(f"[baseline] saved checkpoint: {ckpt_path}")

            train_acc, test_acc = self._evaluate_model(model_base, is_pytorch=True)
            auc_b, tpr_b = self._metric_mia(model_base, forget_loader, m, run_id=r,
                                               retain_loader=retain_loader, forget_ratio=float(forget_ratio))
            base_norm = _global_param_l2_norm(model_base)
            dim = _model_dimension(model_base)

            logger.info(f"[baseline] Train={train_acc*100:.2f} Test={test_acc*100:.2f} MIA_AUC={auc_b*100:.2f} MIA_TPR={tpr_b*100:.2f} time={baseline_time:.2f}s")
            logger.info(f"[baseline] param_l2_norm={base_norm:.6f} model_dimension={dim}")

            imc = float(init_model_clip)
            if imc == 0.0:
                init_model_clip_eff = float(base_norm)
            elif imc < 0.0:
                init_model_clip_eff = float(-imc) * float(base_norm)
            else:
                init_model_clip_eff = imc

            T = certified_sp_steps(init_model_clip=float(init_model_clip_eff), grad_clip=float(grad_clip), lr=float(unlearn_lr), weight_decay=float(unlearn_weight_decay))

            steps_per_epoch = len(retain_loader)

            def noisy_optim_ctor(mdl: nn.Module) -> optim.Optimizer:
                return optim.SGD(mdl.parameters(), lr=float(unlearn_lr), momentum=0.0, weight_decay=0.0)

            def post_optim_ctor(mdl: nn.Module) -> optim.Optimizer:
                post_opt_name = CERTIFIED_SPRunner._resolve_certified_sp_optimizer_name(post_optimizer, self.model_type)
                if post_opt_name == "adam":
                    return optim.Adam(
                        mdl.parameters(),
                        lr=float(post_max_lr),
                        weight_decay=0.0,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        foreach=False,
                    )
                return optim.SGD(mdl.parameters(), lr=float(post_max_lr), momentum=0.0, weight_decay=0.0)

            def post_sched_ctor(opt: optim.Optimizer, total_steps: int):
                if post_lr_schedule == "constant":
                    return None
                if post_lr_schedule == "onecycle":
                    return optim.lr_scheduler.OneCycleLR(
                        opt, max_lr=float(post_max_lr), total_steps=int(total_steps),
                        pct_start=0.3, anneal_strategy="cos", cycle_momentum=False
                    )
                if post_lr_schedule == "cosine":
                    return optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(total_steps))
                raise ValueError(f"post_lr_schedule={post_lr_schedule} not supported")

            model_noisy = copy.deepcopy(model_base)
            model_noisy, info = train_model_certified_sp_unlearn(
                model_noisy,
                retain_loader,
                criterion=self.criterion,
                optimizer_ctor=noisy_optim_ctor,
                device=self.device,
                init_model_clip=float(init_model_clip_eff),
                grad_clip=float(grad_clip),
                epsilon_renyi_target=float(epsilon_renyi_target),
                delta=float(delta),
                lr=float(unlearn_lr),
                weight_decay=float(unlearn_weight_decay),
                noise_schedule=str(noise_schedule),
                init_model_clip_type=str(init_model_clip_type),
                model_dimension=int(dim),
                max_steps=None,
                max_epochs=10**9,
                post_steps=0,
                post_optimizer_ctor=None,
                post_lr_scheduler_ctor=None,
                post_weight_decay=float(post_weight_decay),
                post_unlearn_clip=float(post_unlearn_clip) if post_unlearn_clip is not None else None,
                seed=int(self.seed) + int(r),
                post_steps_per_epoch=int(steps_per_epoch),
                post_epoch_hook=None,
            )

            unlearn_time_s = float(info.get("unlearn_time_s", 0.0))

            if "state_dict_after_noisy" not in info:
                raise RuntimeError("train_certified_sp must return state_dict_after_noisy to evaluate unlearn-phase metrics.")
            model_after_noisy = copy.deepcopy(model_base)
            model_after_noisy.load_state_dict(info["state_dict_after_noisy"])
            model_after_noisy.to(self.device)

            un_retain, un_forget, un_test = self._metric_retain_forget_test(model_after_noisy, retain_loader, forget_loader)
            un_auc, un_tpr = self._metric_mia(model_after_noisy, forget_loader, m, run_id=r,
                                               retain_loader=retain_loader, forget_ratio=float(forget_ratio))

            common_acc["baseline_time_s"].append(float(baseline_time))
            common_acc["baseline_train_%"].append(100.0 * float(train_acc))
            common_acc["baseline_test_%"].append(100.0 * float(test_acc))
            common_acc["baseline_mia_auc_%"].append(100.0 * float(auc_b))
            common_acc["baseline_mia_tpr_%"].append(100.0 * float(tpr_b))
            common_acc["baseline_param_l2"].append(float(base_norm))
            common_acc["model_dimension"].append(float(dim))
            common_acc["init_model_clip_eff"].append(float(init_model_clip_eff))
            common_acc["unlearn_time_s"].append(float(unlearn_time_s))
            common_acc["unlearn_retain_%"].append(100.0 * float(un_retain))
            common_acc["unlearn_forget_%"].append(100.0 * float(un_forget))
            common_acc["unlearn_test_%"].append(100.0 * float(un_test))
            common_acc["unlearn_mia_auc_%"].append(100.0 * float(un_auc))
            common_acc["unlearn_mia_tpr_%"].append(100.0 * float(un_tpr))
            common_acc["certified_sp_T"].append(float(info.get("certified_sp_steps_T", T)))
            common_acc["noisy_steps"].append(float(info.get("noisy_steps_run", 0.0)))
            common_acc["sigma_mean"].append(float(info.get("sigma_mean", 0.0)))
            common_acc["eps_est"].append(float(info.get("eps_est", 0.0)))

            logger.info(
                f"[certified_sp][run {r+1:02d}/{repeats}] baseline_time={baseline_time:.2f}s "
                f"unlearn_time={unlearn_time_s:.2f}s "
                f"T={float(info.get('certified_sp_steps_T', T)):.0f} noisy_steps={float(info.get('noisy_steps_run', 0)):.0f} "
                f"eps\u2248{float(info.get('eps_est', 0.0)):.3f}"
            )
            logger.info(
                f"[unlearn] Retain={un_retain*100:.2f} Forget={un_forget*100:.2f} Test={un_test*100:.2f} "
                f"MIA_AUC={un_auc*100:.2f} MIA_TPR={un_tpr*100:.2f}"
            )

            noisy_state = copy.deepcopy(info["state_dict_after_noisy"])
            for post_epochs in post_epochs_values:
                post_steps_eff = int(post_steps) if int(post_steps) >= 0 else int(max(0, int(post_epochs) * int(steps_per_epoch)))
                post_epoch_rows: List[Dict[str, float]] = []

                def post_epoch_hook(epoch_idx_1based: int, model_now: nn.Module, elapsed_s: float) -> None:
                    r_m, f_m, t_m = self._metric_retain_forget_test(model_now, retain_loader, forget_loader)
                    post_epoch_rows.append(
                        {"epoch": float(epoch_idx_1based), "retain": 100.0 * float(r_m), "forget": 100.0 * float(f_m), "test": 100.0 * float(t_m), "time_s": float(elapsed_s)}
                    )

                model_post = copy.deepcopy(model_base)
                model_post.load_state_dict(noisy_state)
                model_post.to(self.device)
                model_post, post_info = train_model_certified_sp_post_finetune(
                    model_post,
                    retain_loader,
                    criterion=self.criterion,
                    post_optimizer_ctor=post_optim_ctor,
                    device=self.device,
                    post_steps=int(post_steps_eff),
                    post_weight_decay=float(post_weight_decay),
                    post_lr_scheduler_ctor=post_sched_ctor if post_lr_schedule != "constant" else None,
                    post_steps_per_epoch=int(steps_per_epoch),
                    post_epoch_hook=post_epoch_hook,
                    post_unlearn_clip=float(post_unlearn_clip) if post_unlearn_clip is not None else None,
                    show_progress=True,
                )

                post_time_s = float(post_info.get("post_time_s", 0.0))
                post_steps_run = float(post_info.get("post_steps_run", 0.0))
                total_time = float(baseline_time) + float(unlearn_time_s) + float(post_time_s)
                po_retain, po_forget, po_test = self._metric_retain_forget_test(model_post, retain_loader, forget_loader)
                po_auc, po_tpr = self._metric_mia(model_post, forget_loader, m, run_id=r,
                                                   retain_loader=retain_loader, forget_ratio=float(forget_ratio))

                logger.info(
                    f"[certified_sp][run {r+1:02d}/{repeats}][post_epochs={int(post_epochs)}] total_time={total_time:.2f}s "
                    f"unlearn_time={unlearn_time_s:.2f}s post_time={post_time_s:.2f}s "
                    f"T={float(info.get('certified_sp_steps_T', T)):.0f} noisy_steps={float(info.get('noisy_steps_run', 0)):.0f} "
                    f"post_steps={post_steps_run:.0f} eps\u2248{float(info.get('eps_est', 0.0)):.3f}"
                )
                logger.info(
                    f"[post]    Retain={po_retain*100:.2f} Forget={po_forget*100:.2f} Test={po_test*100:.2f} "
                    f"MIA_AUC={po_auc*100:.2f} MIA_TPR={po_tpr*100:.2f}"
                )

                if post_epoch_rows and bool(self.epoch_metrics):
                    logger.info(f"\n[post][per-epoch][post_epochs={int(post_epochs)}] (retain/forget/test in %, cumulative post time)")
                    for row in post_epoch_rows:
                        logger.info(f"  ep {int(row['epoch']):03d}: retain={row['retain']:.2f} forget={row['forget']:.2f} test={row['test']:.2f} time={row['time_s']:.2f}s")

                post_acc = post_acc_by_epoch[int(post_epochs)]
                post_acc["post_time_s"].append(float(post_time_s))
                post_acc["post_retain_%"].append(100.0 * float(po_retain))
                post_acc["post_forget_%"].append(100.0 * float(po_forget))
                post_acc["post_test_%"].append(100.0 * float(po_test))
                post_acc["post_mia_auc_%"].append(100.0 * float(po_auc))
                post_acc["post_mia_tpr_%"].append(100.0 * float(po_tpr))
                post_acc["post_steps"].append(float(post_steps_run))


        baseline_out = {
            "Training Time (s)": _ms(common_acc["baseline_time_s"]),
            "Train Accuracy (%)": _ms(common_acc["baseline_train_%"]),
            "Test Accuracy (%)": _ms(common_acc["baseline_test_%"]),
            "MIA AUC (%)": _ms(common_acc["baseline_mia_auc_%"]),
            "MIA TPR@1%FPR (%)": _ms(common_acc["baseline_mia_tpr_%"]),
        }
        unlearn_out = {
            "Unlearn Time (s)": _ms(common_acc["unlearn_time_s"]),
            "Retain Accuracy (%)": _ms(common_acc["unlearn_retain_%"]),
            "Forget Accuracy (%)": _ms(common_acc["unlearn_forget_%"]),
            "Test Accuracy (%)": _ms(common_acc["unlearn_test_%"]),
            "MIA AUC (%)": _ms(common_acc["unlearn_mia_auc_%"]),
            "MIA TPR@1%FPR (%)": _ms(common_acc["unlearn_mia_tpr_%"]),
        }
        bookkeeping_common_out = {
            "CERTIFIED_SP Steps T": _ms(common_acc["certified_sp_T"]),
            "CERTIFIED_SP Noisy Steps": _ms(common_acc["noisy_steps"]),
            "CERTIFIED_SP Sigma Mean": _ms(common_acc["sigma_mean"]),
            "CERTIFIED_SP eps_est": _ms(common_acc["eps_est"]),
            "init_model_clip_eff": _ms(common_acc["init_model_clip_eff"]),
            "Model Dimension": _ms(common_acc["model_dimension"]),
            "Baseline Param L2": _ms(common_acc["baseline_param_l2"]),
        }
        _rt = os.path.join(out_dir, f"{self.model_type}_runtimes.csv")
        _n = len(common_acc["baseline_time_s"])

        for post_epochs in post_epochs_values:
            post_acc = post_acc_by_epoch[int(post_epochs)]
            post_out = {
                "Post Time (s)": _ms(post_acc["post_time_s"]),
                "Retain Accuracy (%)": _ms(post_acc["post_retain_%"]),
                "Forget Accuracy (%)": _ms(post_acc["post_forget_%"]),
                "Test Accuracy (%)": _ms(post_acc["post_test_%"]),
                "MIA AUC (%)": _ms(post_acc["post_mia_auc_%"]),
                "MIA TPR@1%FPR (%)": _ms(post_acc["post_mia_tpr_%"]),
            }
            bookkeeping_out = dict(bookkeeping_common_out)
            bookkeeping_out["CERTIFIED_SP Post Steps"] = _ms(post_acc["post_steps"])

            _param = f"eps={_eps_str}|post_epochs={int(post_epochs)}"
            _suffix = f"_post_epochs={int(post_epochs)}"

            logger.info(f"\n==================== Phase Summary (mean ± std over repeats, post_epochs={int(post_epochs)}) ====================")
            log_metrics_table(logger, baseline_out, f"CERTIFIED_SP Baseline ({_param}, fr={_fr})")
            log_metrics_table(logger, unlearn_out, f"CERTIFIED_SP Unlearn  ({_param}, fr={_fr})")
            log_metrics_table(logger, post_out, f"CERTIFIED_SP Post     ({_param}, fr={_fr})")
            log_metrics_table(logger, bookkeeping_out, f"CERTIFIED_SP Bookkeeping ({_param}, fr={_fr})")

            f1 = os.path.join(out_dir, f"{self.model_type}_certified_sp_eps={_eps_str}_m_d_fr={_fr}{_suffix}.csv")
            f2 = os.path.join(out_dir, f"{self.model_type}_certified_sp_eps={_eps_str}_unlearn_fr={_fr}{_suffix}.csv")
            f3 = os.path.join(out_dir, f"{self.model_type}_certified_sp_eps={_eps_str}_post_fr={_fr}{_suffix}.csv")
            save_metrics_csv(f1, baseline_out, metadata={"Post Epochs": int(post_epochs)})
            save_metrics_csv(f2, unlearn_out, metadata={"Post Epochs": int(post_epochs)})
            save_metrics_csv(f3, post_out, metadata={"Post Epochs": int(post_epochs)})
            logger.info(f"[io] saved {f1}")
            logger.info(f"[io] saved {f2}")
            logger.info(f"[io] saved {f3}")

            summary_rows: List[Dict[str, Any]] = []
            for phase_label, metrics_dict in [
                ("baseline", baseline_out),
                ("unlearn", unlearn_out),
                ("post", post_out),
                ("bookkeeping", bookkeeping_out),
            ]:
                for metric, (mean_v, std_v) in metrics_dict.items():
                    summary_rows.append({
                        "Experiment": "certified_sp",
                        "Phase": phase_label,
                        "Param": f"eps={_eps_str}",
                        "Post Epochs": int(post_epochs),
                        "Forget Ratio": _fr,
                        "Metric": metric,
                        "Mean": round(mean_v, 6),
                        "Std": round(std_v, 6),
                        "Mean±Std": f"{mean_v:.4f}±{std_v:.4f}",
                    })
            summary_path = os.path.join(out_dir, f"{self.model_type}_certified_sp_eps={_eps_str}{_suffix}_summary.csv")
            save_summary_csv(summary_path, summary_rows)
            logger.info(f"[io] saved summary to {summary_path}")

            append_runtime_rows(_rt, [
                {"Method": "certified_sp", "Param": _param, "Forget Ratio": _fr, "Phase": "train_baseline",
                 "N Runs": _n, "Mean (s)": baseline_out["Training Time (s)"][0], "Std (s)": baseline_out["Training Time (s)"][1], "Post Epochs": int(post_epochs)},
                {"Method": "certified_sp", "Param": _param, "Forget Ratio": _fr, "Phase": "unlearn_noisy",
                 "N Runs": _n, "Mean (s)": unlearn_out["Unlearn Time (s)"][0], "Std (s)": unlearn_out["Unlearn Time (s)"][1], "Post Epochs": int(post_epochs)},
                {"Method": "certified_sp", "Param": _param, "Forget Ratio": _fr, "Phase": "post_ft",
                 "N Runs": _n, "Mean (s)": post_out["Post Time (s)"][0], "Std (s)": post_out["Post Time (s)"][1], "Post Epochs": int(post_epochs)},
            ])

            cfg_path = os.path.join(out_dir, f"{self.model_type}_certified_sp_eps={_eps_str}{_suffix}_config.yaml")
            cfg = self._config_identity()
            cfg.update({
                "forget_ratio": float(forget_ratio),
                "training": self._config_training(),
                "runtime": self._config_runtime(),
                "mia": self._config_mia(),
                "certified_sp_unlearn": {
                    "epsilon_renyi": float(epsilon_renyi_target),
                    "delta": float(delta),
                    "unlearn_lr": float(unlearn_lr),
                    "unlearn_weight_decay": float(unlearn_weight_decay),
                    "noise_schedule": str(noise_schedule),
                    "init_model_clip": float(init_model_clip),
                    "init_model_clip_type": str(init_model_clip_type),
                    "grad_clip": float(grad_clip),
                },
                "post_finetune": {
                    "post_optimizer": str(post_optimizer),
                    "post_lr_schedule": str(post_lr_schedule),
                    "post_max_lr": float(post_max_lr),
                    "post_weight_decay": float(post_weight_decay),
                    "post_epochs_selected": int(post_epochs),
                    "post_epochs_sweep": [int(e) for e in post_epochs_values],
                    "post_steps": int(post_steps),
                    "post_unlearn_clip": float(post_unlearn_clip) if post_unlearn_clip is not None else None,
                },
            })
            cfg["n_repeat"] = int(repeats)
            cfg["forget_ratios"] = [float(forget_ratio)]
            save_config_yaml(cfg_path, cfg)
            logger.info(f"[io] saved config to {cfg_path}")

        return


def _str_to_bool(v: str) -> bool:
    """Allow --flag true / --flag false in addition to plain --flag."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected (true/false), got: {v!r}")


def main(argv: Optional[List[str]] = None):
    p = argparse.ArgumentParser("CERTIFIED_SP runner (neural nets only)",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ── Identity ──────────────────────────────────────────────────────────────
    p.add_argument("--dataset", choices=["adult", "credit", "heart", "cifar10"], required=True)
    p.add_argument("--model", choices=["mlp", "densenet", "resnet18"], required=True)
    p.add_argument(
        "--forget_ratios",
        "--forget_ratio",
        nargs="+",
        type=float,
        default=[0.05],
        dest="forget_ratios",
        help="Forget ratios to evaluate. Multiple values are run sequentially.",
    )
    p.add_argument(
        "--n_repeat",
        "--repeats",
        type=int,
        default=3,
        dest="n_repeat",
        help="Number of repeats to run.",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--results_subdir", type=str, default=None)

    # ── Performance ───────────────────────────────────────────────────────────
    p.add_argument("--use_amp", type=_str_to_bool, default=False, metavar="{true,false}",
                   help="Enable automatic mixed precision (AMP) for faster GPU training")
    p.add_argument("--cifar_download", type=_str_to_bool, default=False, metavar="{true,false}")

    # ── MIA ───────────────────────────────────────────────────────────────────
    p.add_argument("--mia_resamples", type=int, default=10)
    p.add_argument("--mia_eval_cap", type=int, default=5000)

    # ── Checkpointing ─────────────────────────────────────────────────────────
    p.add_argument("--load_ckpt", type=str, default=None)
    p.add_argument("--save_ckpt_dir", type=str, default=None)

    # ── Baseline training (identical to baseline experiment) ──────────────────
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--optimizer", choices=["auto", "adam", "sgd"], default="auto")
    p.add_argument("--scheduler", choices=["none", "step", "cosine", "onecycle"], default="none")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--epoch_metrics", type=_str_to_bool, default=False, metavar="{true,false}",
                   help="Print per-epoch train/test loss and train/test metrics")

    # ── CERTIFIED_SP unlearn ──────────────────────────────────────────────────────────
    p.add_argument("--unlearn_epochs", type=int, default=50)
    p.add_argument("--certified_sp_init_model_clip", type=float, default=0.01)
    p.add_argument("--certified_sp_init_model_clip_type", choices=["clip", "clamp"], default="clip")
    p.add_argument("--certified_sp_grad_clip", type=float, default=10.0)
    p.add_argument("--certified_sp_epsilon_renyi_target", type=float, nargs='+', default=[1.0])
    p.add_argument("--certified_sp_delta", type=float, default=1e-5)
    p.add_argument("--certified_sp_lr", type=float, default=1e-3)
    p.add_argument("--certified_sp_weight_decay", type=float, default=10.0)
    p.add_argument("--certified_sp_noise_schedule", choices=["constant", "decreasing"], default="constant")

    # ── Post fine-tune ────────────────────────────────────────────────────────
    p.add_argument(
        "--post_epochs",
        nargs="+",
        type=int,
        default=[50],
        help="One or more post-unlearning fine-tuning epoch counts to evaluate.",
    )
    p.add_argument("--post_steps", type=int, default=-1)
    p.add_argument("--post_optimizer", choices=["auto", "adam", "sgd"], default="auto")
    p.add_argument("--post_lr_schedule", choices=["onecycle", "cosine", "constant", "none"], default="onecycle",
                   help="'none' disables scheduling and is treated as 'constant'")
    p.add_argument("--post_max_lr", type=float, default=0.1)
    p.add_argument("--post_weight_decay", type=float, default=5e-4)
    p.add_argument("--post_unlearn_clip", type=float, default=0.0)

    args = p.parse_args(argv)

    valid = {"adult": ["mlp"], "credit": ["mlp"], "heart": ["mlp"], "cifar10": ["densenet", "resnet18"]}
    if args.model not in valid[args.dataset]:
        raise SystemExit(f"Unsupported model={args.model} for dataset={args.dataset}. Valid: {valid[args.dataset]}")
    post_lr_schedule = "constant" if args.post_lr_schedule == "none" else str(args.post_lr_schedule)
    post_unlearn_clip = None if float(args.post_unlearn_clip) <= 0 else float(args.post_unlearn_clip)

    runner = CERTIFIED_SPRunner(
        dataset=args.dataset,
        model_type=args.model,
        forget_ratios=args.forget_ratios,
        n_repeat=args.n_repeat,
        max_epochs=args.max_epochs,
        results_subdir=args.results_subdir,
        seed=args.seed,
        optimizer_name=str(args.optimizer),
        scheduler_name=str(args.scheduler),
        lr=args.lr,
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        epoch_metrics=bool(args.epoch_metrics),
        cifar_download=bool(args.cifar_download),
        mia_resamples=args.mia_resamples,
        mia_eval_cap=args.mia_eval_cap,
        use_amp=bool(args.use_amp),
    )

    for forget_ratio in args.forget_ratios:
        for epsilon_renyi_target in args.certified_sp_epsilon_renyi_target:
            runner.run_certified_sp(
                forget_ratio=float(forget_ratio),
                repeats=int(args.n_repeat),
                unlearn_epochs=int(args.unlearn_epochs),
                init_model_clip=float(args.certified_sp_init_model_clip),
                init_model_clip_type=str(args.certified_sp_init_model_clip_type),
                grad_clip=float(args.certified_sp_grad_clip),
                epsilon_renyi_target=float(epsilon_renyi_target),
                delta=float(args.certified_sp_delta),
                unlearn_lr=float(args.certified_sp_lr),
                unlearn_weight_decay=float(args.certified_sp_weight_decay),
                noise_schedule=str(args.certified_sp_noise_schedule),
                post_epochs_list=list(map(int, args.post_epochs)),
                post_steps=int(args.post_steps),
                post_optimizer=str(args.post_optimizer),
                post_lr_schedule=post_lr_schedule,
                post_max_lr=float(args.post_max_lr),
                post_weight_decay=float(args.post_weight_decay),
                post_unlearn_clip=post_unlearn_clip,
                load_ckpt=args.load_ckpt,
                save_ckpt_dir=args.save_ckpt_dir,
            )


if __name__ == "__main__":
    main()

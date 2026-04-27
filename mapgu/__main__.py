#!/usr/bin/env python3
"""
MAPGU unified command-line interface.

Usage:
  python -m mapgu run   --method {baseline,dp,kanon,sisa,certified_sp} --dataset ... --model ... [args]
  python -m mapgu prepare {dp,kanon,embeddings} [args]

Examples:
  python -m mapgu run --method baseline --dataset adult --model mlp --n_repeat 5 --max_epochs 100
  python -m mapgu run --method dp --dataset adult --model mlp --eps_values 1 --ft_epochs 5
  python -m mapgu run --method kanon --dataset adult --model mlp --k_values 30 --ft_epochs 5
  python -m mapgu run --method sisa --dataset adult --model mlp --num_shards 5 --num_slices 10
  python -m mapgu run --method certified_sp --dataset adult --model mlp --forget_ratios 0.05 --n_repeat 5 --max_epochs 100
  python -m mapgu prepare embeddings --in-path data/adult/adult.data
  python -m mapgu prepare kanon --dataset adult --k-values 30 --skip-existing true
  python -m mapgu prepare dp --dataset adult --eps 1 --skip-existing true
"""

from __future__ import annotations

import argparse
import sys
from typing import List, Optional


def _str_to_bool(v: str) -> bool:
    """Allow --flag true / --flag false in addition to plain --flag."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes"):
        return True
    if v.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected (true/false), got: {v!r}")


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mapgu",
        description="MAPGU: Efficient Unlearning with Privacy Guarantees — unified runner",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # -----------------------------------------------------------------------
    # run subcommand
    # -----------------------------------------------------------------------
    run_p = sub.add_parser(
        "run",
        help="Run an experiment (baseline / dp / kanon / sisa / certified_sp)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required
    run_p.add_argument(
        "--method",
        choices=["baseline", "dp", "kanon", "sisa", "certified_sp"],
        required=True,
        help="Which experiment method to run",
    )
    run_p.add_argument(
        "--dataset",
        choices=["adult", "credit", "heart", "cifar10"],
        required=True,
    )
    run_p.add_argument(
        "--model",
        choices=["mlp", "xgboost", "densenet", "resnet18"],
        required=True,
    )
    run_p.add_argument(
        "--results_subdir",
        type=str,
        default=None,
        help="Optional subfolder under results/ for this run, e.g. sensitivity_studies/adult_xgboost.",
    )

    # --- Shared training args ---
    run_p.add_argument(
        "--forget_ratios",
        "--forget_ratio",
        nargs="+",
        type=float,
        default=[0.05],
        dest="forget_ratios",
        help="Forget ratios to evaluate. Multiple values are run sequentially.",
    )
    run_p.add_argument(
        "--n_repeat",
        "--repeats",
        type=int,
        default=3,
        dest="n_repeat",
        help="Number of repeats to run.",
    )
    run_p.add_argument("--max_epochs", type=int, default=None,
                       help="Training epochs (None = method default)")
    run_p.add_argument("--seed", type=int, default=7)
    run_p.add_argument("--batch_size", type=int, default=None)
    run_p.add_argument("--lr", type=float, default=None,
                       help="Learning rate for primary training phase")
    run_p.add_argument("--ft_lr", type=float, default=None,
                       help="Learning rate for fine-tuning / retrain phase")
    run_p.add_argument("--optimizer", choices=["auto", "adam", "sgd"], default="auto")
    run_p.add_argument("--scheduler",
                       choices=["none", "step", "cosine", "onecycle"], default="none")
    run_p.add_argument("--ft_optimizer", choices=["auto", "adam", "sgd"], default=None)
    run_p.add_argument("--ft_scheduler",
                       choices=["none", "step", "cosine", "onecycle"], default=None)
    run_p.add_argument("--momentum", type=float, default=0.9)
    run_p.add_argument("--weight_decay", type=float, default=0.0)
    run_p.add_argument("--scheduler_step_size", type=int, default=30)
    run_p.add_argument("--scheduler_gamma", type=float, default=0.1)
    run_p.add_argument("--onecycle_pct_start", type=float, default=0.3)
    run_p.add_argument("--epoch_metrics", type=_str_to_bool, default=False, metavar="{true,false}",
                       help="Print per-epoch train/test loss and metrics")

    # XGBoost overrides
    run_p.add_argument("--xgb_n_estimators", type=int, default=None)
    run_p.add_argument("--xgb_max_depth", type=int, default=None)
    run_p.add_argument("--xgb_lr", type=float, default=None)
    run_p.add_argument("--xgb_reg_lambda", type=float, default=None)

    # Performance knobs
    run_p.add_argument("--num_workers", type=int, default=4)
    run_p.add_argument("--pin_memory", type=_str_to_bool, default=False, metavar="{true,false}")
    run_p.add_argument("--deterministic", type=_str_to_bool, default=False, metavar="{true,false}")
    run_p.add_argument("--use_amp", type=_str_to_bool, default=False, metavar="{true,false}",
                       help="Enable automatic mixed precision (AMP) for faster GPU training")
    run_p.add_argument("--cifar_download", type=_str_to_bool, default=False, metavar="{true,false}",
                       help="Download CIFAR-10 if not present")

    # --- DP / kanon specific ---
    run_p.add_argument("--eps_values", nargs="+", type=float, default=[1.0],
                       help="Epsilon values for DP experiments")
    run_p.add_argument("--k_values", nargs="+", type=int, default=[30],
                       help="k values for k-anonymity experiments")
    run_p.add_argument(
        "--ft_epochs",
        "--ft_epochs_sisa",
        nargs="+",
        type=int,
        default=None,
        dest="ft_epochs",
        help="Fine-tune epoch counts. DP/k-anon accept a list; SISA accepts a single value. "
             "If omitted, method defaults apply.",
    )
    run_p.add_argument("--kanon_cluster_repr", choices=["onehot", "tabnet", "latent"], default="onehot")
    run_p.add_argument("--kanon_perm_type", "--perm-type", choices=["rowwise", "colwise"], default="rowwise",
                       dest="kanon_perm_type",
                       help="QI permutation strategy within each k-anon cluster: "
                            "'rowwise' (default) shuffles whole rows; "
                            "'colwise' shuffles each QI column independently")
    run_p.add_argument("--embeddings_pkl", type=str, default=None,
                       help="Path to embeddings.pkl (required for --kanon_cluster_repr tabnet)")
    run_p.add_argument("--kanon_mode", choices=["prepared", "regenerate"], default="prepared")
    run_p.add_argument("--kanon_seed", type=int, default=None,
                       help="Seed for k-anon regenerate mode")
    run_p.add_argument("--private_optimizer", choices=["auto", "adam", "sgd"], default=None)
    run_p.add_argument("--private_scheduler",
                       choices=["none", "step", "cosine", "onecycle"], default=None)

    # MIA evaluation
    run_p.add_argument("--mia_resamples", type=int, default=10)
    run_p.add_argument("--mia_eval_cap", type=int, default=5000)
    run_p.add_argument(
        "--mia_attacks",
        nargs="+",
        choices=["loss", "scaled_logit", "rmia"],
        default=["loss"],
        help="MIA attacks to run. 'loss'=Yeom threshold, 'scaled_logit'=LiRA single-model, 'rmia'=RMIA offline (requires reference model training). Default: loss.",
    )
    run_p.add_argument(
        "--rmia_n_ref",
        type=int,
        default=1,
        help="Number of reference models for RMIA (default: 1). More models give a stronger attack but multiply training cost.",
    )

    # Resume / progress
    run_p.add_argument("--resume", type=_str_to_bool, default=False, metavar="{true,false}",
                       help="Resume from a saved progress file")
    run_p.add_argument("--progress_path", type=str, default=None)
    run_p.add_argument("--max_configs", type=int, default=None)

    # --- SISA specific ---
    run_p.add_argument("--num_shards", type=int, default=5)
    run_p.add_argument("--num_slices", type=int, default=10)
    run_p.add_argument("--slice_epochs", type=int, default=None,
                       help="Per-slice epochs for SISA (default: paper formula)")
    # --- CERTIFIED_SP specific ---
    run_p.add_argument("--load_ckpt", type=str, default=None)
    run_p.add_argument("--save_ckpt_dir", type=str, default=None)

    # CERTIFIED_SP unlearn phase
    run_p.add_argument("--unlearn_epochs", type=int, default=50)
    run_p.add_argument("--certified_sp_init_model_clip", type=float, default=0.01)
    run_p.add_argument("--certified_sp_init_model_clip_type",
                       choices=["clip", "clamp"], default="clip")
    run_p.add_argument("--certified_sp_grad_clip", type=float, default=10.0)
    run_p.add_argument("--certified_sp_epsilon_renyi_target", type=float, nargs='+', default=[1.0])
    run_p.add_argument("--certified_sp_delta", type=float, default=1e-5)
    run_p.add_argument("--certified_sp_lr", type=float, default=1e-3)
    run_p.add_argument("--certified_sp_weight_decay", type=float, default=10.0)
    run_p.add_argument("--certified_sp_noise_schedule",
                       choices=["constant", "decreasing"], default="constant")

    # CERTIFIED_SP post-unlearn fine-tune
    run_p.add_argument(
        "--post_epochs",
        nargs="+",
        type=int,
        default=[50],
        help="One or more post-unlearning fine-tuning epoch counts to evaluate.",
    )
    run_p.add_argument("--post_steps", type=int, default=-1)
    run_p.add_argument("--post_optimizer", choices=["auto", "adam", "sgd"], default="auto")
    run_p.add_argument("--post_lr_schedule",
                       choices=["onecycle", "cosine", "constant", "none"], default="onecycle",
                       help="'none' disables scheduling and is treated as 'constant'")
    run_p.add_argument("--post_max_lr", type=float, default=0.1)
    run_p.add_argument("--post_weight_decay", type=float, default=5e-4)
    run_p.add_argument("--post_unlearn_clip", type=float, default=0.0)

    # -----------------------------------------------------------------------
    # prepare subcommand
    # -----------------------------------------------------------------------
    prep_p = sub.add_parser(
        "prepare",
        help="Prepare privacy-protected data offline (one-time, before running experiments)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    prep_sub = prep_p.add_subparsers(dest="prep_type", required=True)

    # prepare embeddings
    emb_p = prep_sub.add_parser(
        "embeddings",
        help="Build Adult TabNet embeddings + utilities (required for tabnet kanon repr and Adult DP)",
    )
    emb_p.add_argument("--in-path", default="data/adult/adult.data",
                       help="Path to adult.data raw CSV")
    emb_p.add_argument("--out-embeddings", default="data/adult/embeddings.pkl")
    emb_p.add_argument("--out-utilities", default="data/adult/utilities.pkl")
    emb_p.add_argument("--seed", type=int, default=0)
    emb_p.add_argument("--cat-emb-dim", type=int, default=10)
    emb_p.add_argument("--max-epochs", type=int, default=200)
    emb_p.add_argument("--valid-frac", type=float, default=0.1)
    emb_p.add_argument("--no-numeric-utils", type=_str_to_bool, default=False, metavar="{true,false}",
                       help="Skip numeric utility computation")
    emb_p.add_argument("--force", type=_str_to_bool, default=False, metavar="{true,false}",
                       help="Force retrain/recompute even if cache exists")
    emb_p.add_argument("--verbose", type=int, default=10)

    # prepare kanon
    kanon_p = prep_sub.add_parser(
        "kanon",
        help="Prepare k-anonymous training data (MDAV + probabilistic permutation)",
    )
    kanon_p.add_argument("--dataset", choices=["adult", "credit", "heart"], required=True)
    kanon_p.add_argument("--k-values", nargs="+", type=int, default=[30])
    kanon_p.add_argument("--cluster-repr", choices=["onehot", "tabnet"], default="onehot")
    kanon_p.add_argument("--embeddings-pkl", default=None,
                         help="Path to embeddings.pkl (required when --cluster-repr tabnet)")
    kanon_p.add_argument("--seed", type=int, default=7)
    kanon_p.add_argument("--skip-existing", type=_str_to_bool, default=False, metavar="{true,false}")

    # prepare dp
    dp_p = prep_sub.add_parser(
        "dp",
        help="Prepare differentially private training data",
    )
    dp_p.add_argument("--dataset", choices=["adult", "credit", "heart", "cifar10"], required=True)
    dp_p.add_argument("--eps", nargs="+", type=float, default=[1.0])
    dp_p.add_argument("--seed", type=int, default=7)
    dp_p.add_argument("--skip-existing", type=_str_to_bool, default=False, metavar="{true,false}")

    import os
    _data = os.path.join(os.path.abspath("."), "data")
    dp_p.add_argument("--adult-in-path",
                      default=os.path.join(_data, "adult", "adult.data"))
    dp_p.add_argument("--adult-utilities-pkl",
                      default=os.path.join(_data, "adult", "utilities.pkl"))
    dp_p.add_argument("--heart-in-path",
                      default=os.path.join(_data, "heart", "cardio_train.csv"))
    dp_p.add_argument("--credit-in-path",
                      default=os.path.join(_data, "GiveMeSomeCredit", "cs-training.csv"))
    dp_p.add_argument("--cifar-root", default=_data)
    dp_p.add_argument("--cifar-m", type=int, default=16)
    dp_p.add_argument("--cifar-block", type=int, default=4)

    return p


# ---------------------------------------------------------------------------
# Routing helpers
# ---------------------------------------------------------------------------

def _run_baseline(args: argparse.Namespace) -> None:
    from mapgu.experiments.base import PrivacyBenchmark

    max_epochs = args.max_epochs if args.max_epochs is not None else 10

    bench = PrivacyBenchmark(
        dataset=args.dataset,
        model_type=args.model,
        forget_ratios=args.forget_ratios,
        n_repeat=args.n_repeat,
        max_epochs=max_epochs,
        results_subdir=args.results_subdir,
        kanon_cluster_repr=args.kanon_cluster_repr,
        embeddings_pkl=args.embeddings_pkl,
        seed=args.seed,
        cifar_download=bool(args.cifar_download),
        num_workers=int(args.num_workers),
        pin_memory=bool(args.pin_memory),
        deterministic=bool(args.deterministic),
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        ft_optimizer_name=args.ft_optimizer,
        ft_scheduler_name=args.ft_scheduler,
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        scheduler_step_size=int(args.scheduler_step_size),
        scheduler_gamma=float(args.scheduler_gamma),
        onecycle_pct_start=float(args.onecycle_pct_start),
        epoch_metrics=bool(args.epoch_metrics),
        batch_size=args.batch_size,
        lr=args.lr,
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_lr=args.xgb_lr,
        xgb_reg_lambda=args.xgb_reg_lambda,
        mia_attacks=list(args.mia_attacks),
        rmia_n_ref=int(args.rmia_n_ref),
        use_amp=bool(args.use_amp),
    )
    bench.run_baseline()


def _run_privacy(args: argparse.Namespace, method: str) -> None:
    from mapgu.experiments.dp import PrivacyExperiments

    max_epochs = args.max_epochs if args.max_epochs is not None else 100
    ft_epochs = list(map(int, args.ft_epochs)) if args.ft_epochs is not None else [5]

    exp = PrivacyExperiments(
        dataset=args.dataset,
        model_type=args.model,
        forget_ratios=args.forget_ratios,
        n_repeat=args.n_repeat,
        max_epochs=max_epochs,
        results_subdir=args.results_subdir,
        lr=args.lr,
        ft_lr=args.ft_lr,
        private_optimizer=args.private_optimizer,
        private_scheduler=args.private_scheduler,
        ft_optimizer=args.ft_optimizer,
        ft_scheduler=args.ft_scheduler,
        kanon_cluster_repr=args.kanon_cluster_repr,
        kanon_perm_type=args.kanon_perm_type,
        embeddings_pkl=args.embeddings_pkl,
        kanon_mode=args.kanon_mode,
        kanon_seed=args.kanon_seed,
        epoch_metrics=bool(args.epoch_metrics),
        mia_resamples=args.mia_resamples,
        mia_eval_cap=args.mia_eval_cap,
        resume=args.resume,
        progress_path=args.progress_path,
        max_configs=args.max_configs,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        ft_optimizer_name=args.ft_optimizer,
        ft_scheduler_name=args.ft_scheduler,
        momentum=float(args.momentum),
        weight_decay=float(args.weight_decay),
        xgb_n_estimators=args.xgb_n_estimators,
        xgb_max_depth=args.xgb_max_depth,
        xgb_lr=args.xgb_lr,
        xgb_reg_lambda=args.xgb_reg_lambda,
        scheduler_step_size=int(args.scheduler_step_size),
        scheduler_gamma=float(args.scheduler_gamma),
        onecycle_pct_start=float(args.onecycle_pct_start),
        mia_attacks=list(args.mia_attacks),
        rmia_n_ref=int(args.rmia_n_ref),
        use_amp=bool(args.use_amp),
    )

    if method == "kanon":
        exp.run_kanonymity(
            k_values=list(map(int, args.k_values)),
            ft_epochs_list=ft_epochs,
        )
    elif method == "dp":
        exp.run_differential_privacy(
            eps_values=list(map(float, args.eps_values)),
            ft_epochs_list=ft_epochs,
        )


def _run_sisa(args: argparse.Namespace) -> None:
    from mapgu.experiments.sisa import SISAExperiments, _paper_defaults

    dflt = _paper_defaults(args.dataset, args.model)
    max_epochs = int(args.max_epochs) if args.max_epochs is not None else int(dflt.epochs)
    ft_epochs_val = int(args.ft_epochs[0]) if args.ft_epochs is not None else int(max_epochs)
    batch_size = int(args.batch_size) if args.batch_size is not None else int(dflt.batch_size)
    lr = float(args.lr) if args.lr is not None else float(dflt.train_lr)
    ft_lr = float(args.ft_lr) if args.ft_lr is not None else float(dflt.retrain_lr)
    xgb_n_estimators = int(args.xgb_n_estimators) if args.xgb_n_estimators is not None else dflt.xgb_n_estimators
    xgb_max_depth = int(args.xgb_max_depth) if args.xgb_max_depth is not None else dflt.xgb_max_depth
    xgb_lr = float(args.xgb_lr) if args.xgb_lr is not None else dflt.xgb_lr
    xgb_reg_lambda = float(args.xgb_reg_lambda) if args.xgb_reg_lambda is not None else dflt.xgb_reg_lambda

    runner = SISAExperiments(
        dataset=args.dataset,
        model_type=args.model,
        forget_ratios=args.forget_ratios,
        n_repeat=args.n_repeat,
        max_epochs=max_epochs,
        results_subdir=args.results_subdir,
        ft_epochs=ft_epochs_val,
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
        mia_attacks=list(args.mia_attacks),
        rmia_n_ref=int(args.rmia_n_ref),
        use_amp=bool(args.use_amp),
    )
    runner.run_sisa()


def _run_certified_sp(args: argparse.Namespace) -> None:
    from mapgu.experiments.certified_sp import CERTIFIED_SPRunner

    post_lr_schedule = "constant" if args.post_lr_schedule == "none" else str(args.post_lr_schedule)
    post_unlearn_clip = (
        None if float(args.post_unlearn_clip) <= 0 else float(args.post_unlearn_clip)
    )

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
        mia_attacks=list(args.mia_attacks),
        rmia_n_ref=int(args.rmia_n_ref),
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


# ---------------------------------------------------------------------------
# Validate model/dataset combinations
# ---------------------------------------------------------------------------

_VALID_COMBOS = {
    "adult":  ["mlp", "xgboost"],
    "credit": ["mlp", "xgboost"],
    "heart":  ["mlp", "xgboost"],
    "cifar10": ["densenet", "resnet18"],
}

_CERTIFIED_SP_VALID_COMBOS = {
    "adult":  ["mlp"],
    "credit": ["mlp"],
    "heart":  ["mlp"],
    "cifar10": ["densenet", "resnet18"],
}


def _validate_run_args(args: argparse.Namespace) -> None:
    combos = _CERTIFIED_SP_VALID_COMBOS if args.method == "certified_sp" else _VALID_COMBOS
    if args.model not in combos.get(args.dataset, []):
        raise SystemExit(
            f"Unsupported --model={args.model} for --dataset={args.dataset} "
            f"with --method={args.method}. Valid: {combos[args.dataset]}"
        )
    if args.method == "sisa" and args.ft_epochs is not None and len(args.ft_epochs) != 1:
        raise SystemExit("SISA expects a single value for --ft_epochs.")
    if args.method == "kanon" and args.dataset == "cifar10" and args.kanon_cluster_repr != "latent":
        raise SystemExit(
            "k-anonymity for cifar10 requires --kanon_cluster_repr latent "
            "(ResNet-18 features + pixel-wise permutation)."
        )
    if args.method in ("kanon", "baseline") and args.kanon_cluster_repr == "tabnet":
        if args.dataset != "adult":
            raise SystemExit("--kanon_cluster_repr tabnet is only meaningful for adult.")
        if args.embeddings_pkl is None:
            raise SystemExit(
                "Provide --embeddings_pkl (e.g., data/adult/embeddings.pkl) "
                "when using --kanon_cluster_repr tabnet."
            )


# ---------------------------------------------------------------------------
# prepare routing
# ---------------------------------------------------------------------------

def _prepare_embeddings(args: argparse.Namespace) -> None:
    from scripts.build_embeddings import build_adult_tabnet_utilities

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


def _prepare_kanon(args: argparse.Namespace) -> None:
    from scripts.prepare_data import prepare_kanon

    prepare_kanon(
        dataset=args.dataset,
        k_values=args.k_values,
        cluster_repr=args.cluster_repr,
        seed=args.seed,
        embeddings_pkl=args.embeddings_pkl,
        skip_existing=args.skip_existing,
    )


def _prepare_dp(args: argparse.Namespace) -> None:
    from scripts.prepare_data import prepare_dp

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


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        _validate_run_args(args)
        if args.method == "baseline":
            _run_baseline(args)
        elif args.method in ("dp", "kanon"):
            _run_privacy(args, method=args.method)
        elif args.method == "sisa":
            _run_sisa(args)
        elif args.method == "certified_sp":
            _run_certified_sp(args)

    elif args.command == "prepare":
        if args.prep_type == "embeddings":
            _prepare_embeddings(args)
        elif args.prep_type == "kanon":
            _prepare_kanon(args)
        elif args.prep_type == "dp":
            _prepare_dp(args)


if __name__ == "__main__":
    main()

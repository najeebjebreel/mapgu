"""CERTIFIED_SP (Privacy-Aware Bayesian Inference) unlearning trainer with noisy and post phases."""

# train_certified_sp.py
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm import tqdm

from mapgu.training.trainer import global_param_l2_norm as _global_param_l2_norm


# -------------------------
# small utils
# -------------------------
@torch.no_grad()
def _clip_model_l2_(model: torch.nn.Module, max_norm: float) -> float:
    if max_norm is None or max_norm <= 0:
        return 0.0
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        return 0.0

    device = params[0].device
    sq = torch.zeros((), device=device)
    for p in params:
        sq += p.detach().float().norm(2) ** 2
    norm = sq.sqrt()

    if norm > float(max_norm):
        scale = (float(max_norm) / (norm + 1e-12)).to(device=device)
        for p in params:
            p.mul_(scale)
    return float(norm.item())


@torch.no_grad()
def _clamp_model_coords_(model: torch.nn.Module, max_val: float) -> None:
    """Element-wise clamp each parameter to [-max_val, max_val] (coordinate-wise projection)."""
    c = float(max_val)
    if c <= 0:
        return
    for p in model.parameters():
        if p.requires_grad:
            p.clamp_(-c, c)


@torch.no_grad()
def _add_param_noise_(model: torch.nn.Module, sigma: float, *, generator: torch.Generator) -> None:
    sig = float(sigma)
    if sig <= 0.0:
        return
    for p in model.parameters():
        if not p.requires_grad:
            continue
        noise = torch.randn(p.shape, device=p.device, dtype=torch.float32, generator=generator)
        p.add_(noise.to(dtype=p.dtype) * sig)


@torch.no_grad()
def _add_wd_after_clip_(model: torch.nn.Module, weight_decay: float) -> None:
    """Implements: g <- g + wd * theta  (after grad clipping)."""
    wd = float(weight_decay)
    if wd == 0.0:
        return
    for p in model.parameters():
        if (not p.requires_grad) or (p.grad is None):
            continue
        p.grad.add_(p, alpha=wd)


def _assert_sgd_no_momentum_no_wd_(optim: torch.optim.Optimizer) -> None:
    for pg in optim.param_groups:
        if abs(float(pg.get("momentum", 0.0))) > 1e-12:
            raise ValueError("CERTIFIED_SP iteration requires SGD with momentum=0.")
        if abs(float(pg.get("weight_decay", 0.0))) > 1e-12:
            raise ValueError("Set optimizer weight_decay=0; CERTIFIED_SP applies wd manually after clipping.")


def certified_sp_steps(init_model_clip: float, grad_clip: float, lr: float, weight_decay: float) -> int:
    lr = float(lr)
    init_model_clip = float(init_model_clip)
    grad_clip = float(grad_clip)

    T = int(math.ceil(init_model_clip / (grad_clip * lr + 1e-12)))
    T = max(T, 1)

    wd = float(weight_decay) if weight_decay else 0.0
    if wd > 0:
        inside = wd * 2.0 * init_model_clip / (grad_clip + 1e-12)
        if inside > 0:
            smaller = int(math.ceil(math.log(inside + 1e-12) / (lr * wd + 1e-12)))
            if 0 < smaller < T:
                T = smaller

    return max(T, 1)


def _sigma_t_certified_sp(
    step: int,
    *,
    epsilon_renyi: float,
    init_model_clip: float,
    grad_clip: float,
    lr: float,
    weight_decay: float,
    T: int,
    noise_schedule: str = "constant",
    init_model_clip_type: str = "clip",
    model_dimension: Optional[int] = None,
) -> float:
    eps_r = float(epsilon_renyi)

    diam = 2.0 * float(init_model_clip)
    if str(init_model_clip_type).lower() == "clamp":
        if model_dimension is None or int(model_dimension) <= 0:
            raise ValueError("init_model_clip_type='clamp' requires a valid model_dimension.")
        diam *= math.sqrt(float(model_dimension))

    C = float(grad_clip)
    eta = float(lr)
    lam = float(weight_decay) if weight_decay else 0.0
    T = int(T)

    if lam > 0:
        decay = 1.0 - eta * lam
        common = decay**T
        grad_clip_term = (2.0 * C / lam) * (1.0 - common)
        var = (diam * common + grad_clip_term) ** 2

        if noise_schedule == "constant":
            numerator = eta * lam * (2.0 - eta * lam)
            denom = 2.0 * eps_r * (1.0 - (common**2))
            scale = numerator / (denom + 1e-12)
            var *= scale
            return float(math.sqrt(max(var, 0.0)))

        if noise_schedule == "decreasing":
            scale = 2.0 * eps_r * T * (decay ** (2.0 * (T - int(step) - 1)))
            return float(math.sqrt(max(var / (scale + 1e-12), 0.0)))

        raise ValueError(f"Unsupported noise_schedule={noise_schedule}")

    # weight_decay == 0 path: only the constant-schedule formula is defined by the paper.
    if noise_schedule != "constant":
        raise ValueError(
            f"noise_schedule='{noise_schedule}' with weight_decay=0 is not supported; "
            "the decreasing-schedule formula requires weight_decay > 0."
        )
    var = (diam + 2.0 * C * eta * T) ** 2
    return float(math.sqrt(max(var / (4.0 * eps_r * T + 1e-12), 0.0)))


def _renyi_to_eps(eps_renyi: float, delta: float) -> float:
    eps_r = float(eps_renyi)
    d = float(delta)
    # Same grid/minimization as before, but vectorized to avoid Python-loop overhead.
    a = torch.linspace(1.001, 1000.0, 10000, dtype=torch.float64)
    vals = eps_r - (math.log(d) + torch.log(a)) / (a - 1.0) + torch.log((a - 1.0) / a)
    return float(torch.min(vals).item())


def _post_l2_penalty(model: torch.nn.Module, weight_decay: float) -> torch.Tensor:
    wd = float(weight_decay)
    any_param = next((p for p in model.parameters() if p.requires_grad), None)
    if wd <= 0:
        return any_param.detach().new_zeros(()) if any_param is not None else torch.tensor(0.0)

    s = None
    for p in model.parameters():
        if not p.requires_grad:
            continue
        term = (p**2).sum()
        s = term if s is None else (s + term)

    if s is None:
        return any_param.detach().new_zeros(()) if any_param is not None else torch.tensor(0.0)

    return 0.5 * wd * s


def train_model_certified_sp_post_finetune(
    model: torch.nn.Module,
    retain_loader: DataLoader,
    *,
    criterion: torch.nn.Module,
    post_optimizer_ctor: Callable[[torch.nn.Module], torch.optim.Optimizer],
    device: torch.device,
    post_steps: int,
    post_weight_decay: float = 0.0,
    post_lr_scheduler_ctor: Optional[Callable[[torch.optim.Optimizer, int], object]] = None,
    post_steps_per_epoch: Optional[int] = None,
    post_epoch_hook: Optional[Callable[[int, torch.nn.Module, float], None]] = None,
    post_unlearn_clip: Optional[float] = None,
    show_progress: bool = True,
    non_blocking: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, float]]:
    """Run only the clean post-unlearning fine-tuning phase."""
    post_done = 0
    post_time_s = 0.0

    steps_per_epoch = int(post_steps_per_epoch) if post_steps_per_epoch is not None else len(retain_loader)
    steps_per_epoch = max(steps_per_epoch, 1)

    if int(post_steps) > 0:
        model.train()
        if post_unlearn_clip is not None and float(post_unlearn_clip) > 0:
            _clip_model_l2_(model, float(post_unlearn_clip))

        post_optim = post_optimizer_ctor(model)
        post_sched = post_lr_scheduler_ctor(post_optim, int(post_steps)) if post_lr_scheduler_ctor is not None else None

        post_t0 = time.time()
        hook_overhead_s = 0.0  # time spent in per-epoch evaluation hooks; excluded from post_time_s
        epoch_idx = 0
        steps_in_epoch = 0
        post_steps_i = int(post_steps)
        post_wd_f = float(post_weight_decay)
        post_epochs_total = max(1, math.ceil(post_steps_i / steps_per_epoch))
        post_pbar = tqdm(range(10**9), total=post_epochs_total, desc="CERTIFIED_SP Post", leave=False, disable=not show_progress)

        for _epoch in post_pbar:
            model.train()
            for x, y in retain_loader:
                if post_done >= post_steps_i:
                    break

                x = x.to(device, non_blocking=non_blocking)
                y = y.to(device, non_blocking=non_blocking)

                post_optim.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)

                if post_wd_f > 0.0:
                    loss = loss + _post_l2_penalty(model, post_wd_f)

                loss.backward()
                post_optim.step()
                if post_sched is not None:
                    post_sched.step()

                post_done += 1
                steps_in_epoch += 1

                if steps_in_epoch >= steps_per_epoch:
                    epoch_idx += 1
                    steps_in_epoch = 0
                    if post_epoch_hook is not None:
                        was_training = model.training
                        elapsed = time.time() - post_t0 - hook_overhead_s
                        _hook_t0 = time.time()
                        post_epoch_hook(int(epoch_idx), model, float(elapsed))
                        hook_overhead_s += time.time() - _hook_t0
                        if was_training:
                            model.train()

            post_pbar.set_postfix(steps=post_done)
            if post_done >= post_steps_i:
                break

        if steps_in_epoch > 0 and post_epoch_hook is not None:
            epoch_idx += 1
            was_training = model.training
            elapsed = time.time() - post_t0 - hook_overhead_s
            _hook_t0 = time.time()
            post_epoch_hook(int(epoch_idx), model, float(elapsed))
            hook_overhead_s += time.time() - _hook_t0
            if was_training:
                model.train()

        post_pbar.close()
        post_time_s = time.time() - post_t0 - hook_overhead_s

    return model, {
        "post_steps_run": float(post_done),
        "post_time_s": float(post_time_s),
    }


@dataclass
class CERTIFIED_SPInfo:
    init_norm_before_clip: float
    init_norm_after_clip: float
    certified_sp_steps_T: float
    noisy_steps_run: float
    post_steps_run: float
    sigma_mean: float
    sigma_last: float
    eps_renyi_target: float
    delta: float
    eps_est: float
    init_model_clip_type: float  # 0 clip, 1 clamp (kept for compatibility)
    model_dimension: float
    unlearn_time_s: float
    post_time_s: float
    state_dict_after_noisy: Dict[str, torch.Tensor]


def train_model_certified_sp_unlearn(
    model: torch.nn.Module,
    retain_loader: DataLoader,
    *,
    criterion: torch.nn.Module,
    optimizer_ctor: Callable[[torch.nn.Module], torch.optim.Optimizer],
    device: torch.device,
    # CERTIFIED_SP hyperparams
    init_model_clip: float,
    grad_clip: float,
    epsilon_renyi_target: float,
    delta: float,
    lr: float,
    weight_decay: float,
    noise_schedule: str = "constant",
    init_model_clip_type: str = "clip",
    model_dimension: Optional[int] = None,
    # budget
    max_steps: Optional[int] = None,
    max_epochs: int = 999999,
    # post phase
    post_steps: int = 0,
    post_optimizer_ctor: Optional[Callable[[torch.nn.Module], torch.optim.Optimizer]] = None,
    post_lr_scheduler_ctor: Optional[Callable[[torch.optim.Optimizer, int], object]] = None,
    post_weight_decay: float = 0.0,
    post_unlearn_clip: Optional[float] = None,
    # rng
    seed: int = 0,
    # per-epoch reporting for post
    post_steps_per_epoch: Optional[int] = None,
    post_epoch_hook: Optional[Callable[[int, torch.nn.Module, float], None]] = None,
    # perf knobs
    non_blocking: bool = True,
    show_progress: bool = True,
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    CERTIFIED_SP "iteration" unlearning:
      - Noisy phase: SGD (mom=0, wd=0 in optimizer), manual wd after clipping, then noise-addition.
      - Post phase: clean fine-tuning for a fixed number of steps with optional scheduler and L2 penalty.

    Optimizations / robustness:
      - Factor out wd update and hook guard (train/eval state restore).
      - Avoid repeated int()/float() conversions in inner loops.
      - Faster/cleaner bookkeeping with a dataclass -> dict at return.
    """
    device = torch.device(device)
    model.to(device)
    model.train()

    init_model_clip_f = float(init_model_clip)
    grad_clip_f = float(grad_clip)
    lr_f = float(lr)
    wd_f = float(weight_decay) if weight_decay else 0.0
    eps_r_f = float(epsilon_renyi_target)
    delta_f = float(delta)
    T_i = 0  # set after T is computed; used to avoid repeated int() in inner loops
    noise_schedule_s = str(noise_schedule)
    init_model_clip_type_s = str(init_model_clip_type)
    model_dimension_i = int(model_dimension) if model_dimension is not None else None

    # Unlearning time should include initial projection/clipping setup + noisy updates.
    unlearn_t0 = time.time()

    init_norm_before = _global_param_l2_norm(model)

    # Initial model projection: coordinate-wise clamp or global L2 clip
    if init_model_clip_type_s == "clamp":
        _clamp_model_coords_(model, init_model_clip_f)
    else:
        _clip_model_l2_(model, init_model_clip_f)
    init_norm_after = _global_param_l2_norm(model)

    T = certified_sp_steps(init_model_clip_f, grad_clip_f, lr_f, wd_f)
    T_i = int(T)
    target_steps = int(max_steps) if max_steps is not None else int(T)
    target_steps = max(target_steps, 1)

    gen = torch.Generator(device=device)
    gen.manual_seed(int(seed) + 12345)

    optim = optimizer_ctor(model)
    _assert_sgd_no_momentum_no_wd_(optim)

    sigma_sum = 0.0
    sigma_last = 0.0
    sigma_n = 0
    global_step = 0

    # --------------------
    # Noisy unlearning
    # --------------------
    noisy_pbar = tqdm(range(int(max_epochs)), desc="CERTIFIED_SP Noisy", leave=False, disable=not show_progress)

    for _epoch in noisy_pbar:
        model.train()
        for x, y in retain_loader:
            if global_step >= target_steps:
                break

            x = x.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)

            optim.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()

            # (1) clip gradients
            clip_grad_norm_(model.parameters(), max_norm=grad_clip_f)

            # (2) manual wd after clipping
            if wd_f != 0.0:
                _add_wd_after_clip_(model, wd_f)

            # (3) step
            optim.step()

            # (4) noise after update
            sigma_t = _sigma_t_certified_sp(
                global_step,
                epsilon_renyi=eps_r_f,
                init_model_clip=init_model_clip_f,
                grad_clip=grad_clip_f,
                lr=lr_f,
                weight_decay=wd_f,
                T=T_i,
                noise_schedule=noise_schedule_s,
                init_model_clip_type=init_model_clip_type_s,
                model_dimension=model_dimension_i,
            )
            _add_param_noise_(model, sigma_t, generator=gen)

            sigma_last = float(sigma_t)
            sigma_sum += float(sigma_t)
            sigma_n += 1

            global_step += 1

        noisy_pbar.set_postfix(steps=global_step, sigma=f"{sigma_last:.4f}")
        if global_step >= target_steps:
            break
    unlearn_time_s = time.time() - unlearn_t0

    # Snapshot (CPU) after noisy phase
    state_dict_after_noisy = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    eps_est = _renyi_to_eps(eps_r_f, delta_f)

    model, post_info = train_model_certified_sp_post_finetune(
        model,
        retain_loader,
        criterion=criterion,
        post_optimizer_ctor=post_optimizer_ctor if post_optimizer_ctor is not None else optimizer_ctor,
        device=device,
        post_steps=int(post_steps),
        post_weight_decay=float(post_weight_decay),
        post_lr_scheduler_ctor=post_lr_scheduler_ctor,
        post_steps_per_epoch=post_steps_per_epoch,
        post_epoch_hook=post_epoch_hook,
        post_unlearn_clip=post_unlearn_clip,
        show_progress=show_progress,
        non_blocking=non_blocking,
    )
    post_done = float(post_info["post_steps_run"])
    post_time_s = float(post_info["post_time_s"])

    info = CERTIFIED_SPInfo(
        init_norm_before_clip=float(init_norm_before),
        init_norm_after_clip=float(init_norm_after),
        certified_sp_steps_T=float(T),
        noisy_steps_run=float(global_step),
        post_steps_run=float(post_done),
        sigma_mean=float(sigma_sum / max(sigma_n, 1)),
        sigma_last=float(sigma_last),
        eps_renyi_target=float(eps_r_f),
        delta=float(delta_f),
        eps_est=float(eps_est),
        init_model_clip_type=0.0 if str(init_model_clip_type).lower() == "clip" else 1.0,
        model_dimension=float(model_dimension) if model_dimension is not None else 0.0,
        unlearn_time_s=float(unlearn_time_s),
        post_time_s=float(post_time_s),
        state_dict_after_noisy=state_dict_after_noisy,
    )

    return model, info.__dict__

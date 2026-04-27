"""CIFAR-10 k-anonymity via pretrained ResNet-18 latent-space MDAV clustering
and pixel-wise random permutation."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from mapgu.config import CIFAR_MEAN, CIFAR_STD
from mapgu.utils import get_logger

logger = get_logger(__name__)

_CIFAR_MEAN = np.array(CIFAR_MEAN, dtype=np.float32).reshape(3, 1, 1)
_CIFAR_STD  = np.array(CIFAR_STD,  dtype=np.float32).reshape(3, 1, 1)


def extract_features(
    images_norm: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Extract 512-d ResNet-18 (ImageNet pretrained) features from CIFAR-normalised images.

    Images are re-normalised from CIFAR statistics to ImageNet statistics before
    passing through the backbone (no fine-tuning; purely as a fixed feature extractor).

    Args:
        images_norm: (N, 3, H, W) float32 array normalised with CIFAR mean/std.
        device: torch device for inference.
        batch_size: inference mini-batch size.

    Returns:
        (N, 512) float32 feature matrix.
    """
    import timm

    model = timm.create_model("resnet18", pretrained=True, num_classes=0, global_pool="avg")
    model.eval().to(device)

    imagenet_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    imagenet_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    cifar_mean    = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
    cifar_std     = torch.tensor(CIFAR_STD,  device=device).view(1, 3, 1, 1)

    ds = TensorDataset(torch.tensor(images_norm, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    feats: List[np.ndarray] = []
    with torch.no_grad():
        for (batch,) in dl:
            batch = batch.to(device)
            # Undo CIFAR normalisation → redo ImageNet normalisation
            pixels = batch * cifar_std + cifar_mean
            pixels = (pixels - imagenet_mean) / imagenet_std
            feats.append(model(pixels).cpu().numpy())

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return np.concatenate(feats, axis=0)  # (N, 512)


def pixel_permute_clusters(
    images_raw: np.ndarray,
    clusters: List[np.ndarray],
    seed: int,
) -> np.ndarray:
    """Column-wise pixel permutation within each MDAV cluster.

    For each cluster of k images:
      - flatten each image to a 1-D vector of length D = C*H*W
      - for every pixel position d independently, shuffle the k values across images

    This breaks per-image pixel identity while preserving the marginal distribution
    of every pixel position within the cluster.

    Args:
        images_raw: (N, C, H, W) float32, values in approximately [0, 1].
        clusters: list of 1-D index arrays (original row indices per cluster).
        seed: RNG seed for reproducibility.

    Returns:
        (N, C, H, W) float32 anonymised images (same memory layout as input copy).
    """
    N, C, H, W = images_raw.shape
    D = C * H * W
    out = images_raw.copy()
    rng = np.random.default_rng(seed)

    for cluster_idxs in clusters:
        k_c = len(cluster_idxs)
        if k_c < 2:
            continue
        flat = out[cluster_idxs].reshape(k_c, D)                        # (k, D)
        perm_mat = np.argsort(rng.random((k_c, D)), axis=0)             # (k, D)
        flat[:] = flat[perm_mat, np.arange(D, dtype=np.int64)[None, :]]
        out[cluster_idxs] = flat.reshape(k_c, C, H, W)

    return out


def build_cifar_kanon_data(
    images_norm: np.ndarray,
    y: np.ndarray,
    k: int,
    seed: int,
    device: torch.device,
    batch_size: int = 256,
) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Build k-anonymous CIFAR-10 training data.

    Pipeline:
      1. Extract ResNet-18 latent features (512-d) from normalised images.
      2. Run MDAV clustering in feature space with cluster size k.
      3. Denormalise images, apply column-wise pixel permutation per cluster,
         re-normalise.

    Args:
        images_norm: (N, 3, H, W) float32, CIFAR-normalised (mean/std from config).
        y: (N,) int64 class labels (pass-through; not modified).
        k: minimum cluster size for MDAV.
        seed: RNG seed for pixel permutation.
        device: torch device for ResNet-18 inference.
        batch_size: inference mini-batch size.

    Returns:
        images_anon: (N, 3, H, W) float32, CIFAR-normalised k-anonymous images.
        y: (N,) int64 labels (unchanged).
        feat_summary: log string describing feature statistics.
        anon_summary: log string describing cluster statistics.
    """
    from mapgu.data.privacy.kanon import mdav_clusters

    logger.info(f"[cifar_kanon] Extracting ResNet-18 features for {len(images_norm)} images ...")
    feats = extract_features(images_norm, device=device, batch_size=batch_size)
    feat_summary = f"feats shape={feats.shape} mean={feats.mean():.3f} std={feats.std():.3f}"
    logger.info(f"[cifar_kanon] {feat_summary}")

    logger.info(f"[cifar_kanon] Running MDAV (k={k}) ...")
    clusters = mdav_clusters(
        X=feats.copy(),
        y=np.asarray(y, dtype=np.int64).copy(),
        k=int(k),
        show_progress=True,
    )
    sizes = [len(c) for c in clusters]
    anon_summary = f"clusters={len(clusters)} min_size={min(sizes)} max_size={max(sizes)}"
    logger.info(f"[cifar_kanon] {anon_summary}")

    # Denorm → permute → renorm (all in float32)
    images_raw = images_norm * _CIFAR_STD + _CIFAR_MEAN          # approx [0, 1]
    logger.info("[cifar_kanon] Applying pixel-wise permutation ...")
    images_raw_anon = pixel_permute_clusters(images_raw, clusters, seed=seed)
    images_anon = (images_raw_anon - _CIFAR_MEAN) / _CIFAR_STD

    return images_anon.astype(np.float32), np.asarray(y, dtype=np.int64), feat_summary, anon_summary

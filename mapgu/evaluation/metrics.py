"""Evaluation metrics: accuracy, AUC, F1, and attack component extraction for MIA."""

from __future__ import annotations

import torch
from torch import nn
import numpy as np
from sklearn.metrics import roc_auc_score


def _infer_device_from_model(model: torch.nn.Module) -> torch.device:
    p = next((p for p in model.parameters() if p is not None), None)
    return p.device if p is not None else torch.device("cpu")


def accuracy(net, loader, device: torch.device | None = None) -> float:
    """
    Return accuracy on a dataset given by the data loader.

    Safe behavior:
      - uses torch.no_grad()
      - temporarily switches to eval() and restores original mode
      - infers device from model if not given (avoids global DEVICE mismatches)
    """
    if device is None:
        device = _infer_device_from_model(net)

    was_training = net.training
    net.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            outputs = net(features)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    if was_training:
        net.train()
    return correct / max(total, 1)

def accuracy_with_majority_voting(nets, loader, device: torch.device | None = None):
    """Compute accuracy using majority voting across multiple models."""
    if device is None:
        device = _infer_device_from_model(nets[0])

    for net in nets:
        net.eval()
        net.to(device)

    correct = 0
    total = 0
    for features, targets in loader:
        features, targets = features.to(device), targets.to(device)
        all_predictions = []
        for net in nets:
            outputs = net(features)
            _, predicted = outputs.max(1)
            all_predictions.append(predicted.unsqueeze(0))  # Add batch dimension for concatenation

        # Stack predictions along a new dimension to form [num_models, batch_size]
        all_predictions = torch.cat(all_predictions, dim=0)

        # Use mode to find the most common prediction (majority vote) for each input
        # mode returns values and indices, where values are the modes (majority votes)
        majority_votes, _ = torch.mode(all_predictions, dim=0)

        total += targets.size(0)
        correct += majority_votes.eq(targets).sum().item()

    return correct / total



def auc_score(model, data_loader, device: torch.device | None = None) -> float:
    """
    ROC-AUC for binary classification.

    Safe behavior:
      - uses torch.no_grad()
      - temporarily switches to eval() and restores original mode
      - infers device from model if not given
    """
    if device is None:
        device = _infer_device_from_model(model)

    was_training = model.training
    model.eval()

    scores = []
    actuals = []

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(inputs)

            if logits.ndim == 1:
                batch_scores = logits
            else:
                if logits.size(1) == 2:
                    batch_scores = logits[:, 1] - logits[:, 0]
                elif logits.size(1) == 1:
                    batch_scores = logits[:, 0]
                else:
                    raise ValueError(f"auc_score expects binary logits (C=1 or C=2), got {tuple(logits.shape)}")

            scores.extend(batch_scores.detach().cpu().numpy().astype(np.float64))
            actuals.extend(labels.detach().cpu().numpy().astype(np.int64))

    if was_training:
        model.train()

    return float(roc_auc_score(actuals, scores))


def auc_score_with_majority_voting(nets, loader, device: torch.device | None = None) -> float:
    """
    ROC-AUC for an ensemble of PyTorch models.
    We average logits, then compute the binary score as in auc_score().
    """
    if device is None:
        device = _infer_device_from_model(nets[0])

    for net in nets:
        net.eval()
        net.to(device)

    scores = []
    actuals = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            targets = targets.to(device)

            # Average logits across models
            logits = None
            for net in nets:
                out = net(features)
                logits = out if logits is None else (logits + out)
            logits = logits / len(nets)

            if logits.ndim == 1:
                batch_scores = logits
            else:
                if logits.size(1) == 2:
                    batch_scores = logits[:, 1] - logits[:, 0]
                elif logits.size(1) == 1:
                    batch_scores = logits[:, 0]
                else:
                    raise ValueError(
                        f"auc_score_with_majority_voting expects binary logits (C=1 or C=2), got {tuple(logits.shape)}"
                    )

            scores.extend(batch_scores.detach().cpu().numpy().astype(np.float64))
            actuals.extend(targets.detach().cpu().numpy().astype(np.int64))

    return float(roc_auc_score(actuals, scores))



def compute_attack_components(net, loader, idxs=None, device: torch.device | None = None):
    if device is None:
        device = _infer_device_from_model(net)

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    all_logits = []
    all_labels = []

    was_training = net.training
    net.eval()

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            if idxs is not None:
                # idxs is a tuple of indices for (continuous, categorical) feature columns
                logits = net(features[:, idxs[0]], features[:, idxs[1]].long())
            else:
                logits = net(features)

            losses = criterion(logits, targets).detach().cpu().numpy()
            all_losses.extend(losses.tolist())
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(targets.detach().cpu().numpy())

    if was_training:
        net.train()

    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return np.array(all_logits), np.array(all_losses), np.array(all_labels)


def compute_attack_components_sisa1(nets, loader, device: torch.device | None = None):

    if device is None:
        device = _infer_device_from_model(nets[0])

    for net in nets:
        net.eval()
        net.to(device)

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            logits = None
            for net in nets:
                if logits is None:
                    logits = net(features)
                else:
                    logits += net(features)

            logits = logits / len(nets)
            losses = criterion(logits, targets).detach().cpu().numpy()

            all_losses.extend(losses.tolist())
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(targets.detach().cpu().numpy())

    # Concatenate logits and labels along the samples axis
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return np.array(all_logits), np.array(all_losses), np.array(all_labels)

def compute_attack_components_sisa2(nets, loaders, device: torch.device | None = None):

    if device is None:
        device = _infer_device_from_model(nets[0])

    for net in nets:
        net.eval()
        net.to(device)

    criterion = nn.CrossEntropyLoss(reduction="none")
    all_losses = []
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for net, loader in zip(nets, loaders):
            for features, targets in loader:
                features = features.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                logits = net(features)
                losses = criterion(logits, targets).detach().cpu().numpy()
                all_losses.extend(losses.tolist())

                all_logits.append(logits.detach().cpu().numpy())
                all_labels.append(targets.detach().cpu().numpy())

    # Concatenate logits and labels along the samples axis
    all_logits = np.concatenate(all_logits, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    return np.array(all_logits), np.array(all_losses), np.array(all_labels)


"""Membership Inference Attack (MIA) implementations: threshold, entropy, LiRA, and classifier attacks."""

from __future__ import annotations

import warnings
import numpy as np
from dataclasses import dataclass
from typing import Tuple
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from mapgu.utils import get_logger

logger = get_logger(__name__)


# --------------------------------------------------
# Helper metrics
# --------------------------------------------------

def attacker_advantage(y_true, scores):
    """Attacker advantage = max_t (TPR(t) - FPR(t))."""
    y_true = np.asarray(y_true).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    finite = np.isfinite(scores)
    if not finite.all():
        y_true, scores = y_true[finite], scores[finite]
    if len(np.unique(y_true)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y_true, scores)
    return float(np.max(tpr - fpr))


def _to_probs(scores_2d: np.ndarray) -> np.ndarray:
    """
    Accept either logits or already-normalised class probabilities.
    Returns a valid probability matrix with rows summing to 1.
    """
    x = np.asarray(scores_2d, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D scores array, got shape={x.shape}")

    row_sum = np.sum(x, axis=1, keepdims=True)
    looks_like_probs = (
        np.all(np.isfinite(x))
        and np.all(x >= -1e-12)
        and np.all(x <= 1.0 + 1e-12)
        and np.all(np.isfinite(row_sum))
        and np.allclose(row_sum, 1.0, atol=1e-4)
    )
    if looks_like_probs:
        p = np.clip(x, 1e-12, 1.0)
        p /= np.sum(p, axis=1, keepdims=True)
        return p

    shifted = x - np.nanmax(x, axis=1, keepdims=True)
    exp = np.exp(shifted)
    denom = np.nansum(exp, axis=1, keepdims=True)
    p = exp / denom
    p = np.clip(p, 1e-12, 1.0)
    p /= np.sum(p, axis=1, keepdims=True)
    return p


def entropy_from_logits(logits):
    """Per-sample entropy from logits OR probabilities (numerically stable)."""
    probs = _to_probs(logits)
    return -(probs * np.log(probs)).sum(axis=1)


def lira_scaled_logit_score(logits, labels):
    """
    Single-model LiRA scaled logit score (Carlini et al., 2022).
        score_i = log(p_correct_i + eps) - log(p_wrong_i + eps)
    Higher score => more likely member.
    """
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    probs  = _to_probs(logits)
    n = logits.shape[0]
    p_correct = probs[np.arange(n), labels]
    probs_zeroed = probs.copy()
    probs_zeroed[np.arange(n), labels] = 0.0
    p_wrong = probs_zeroed.sum(axis=1)
    eps = 1e-45
    return np.log(p_correct + eps) - np.log(p_wrong + eps)


def _cross_entropy_from_probs(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Compute per-sample cross-entropy from probability matrix and integer labels."""
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    n = len(labels)
    p_correct = probs[np.arange(n), labels]
    return -np.log(np.clip(p_correct, 1e-45, 1.0))


# --------------------------------------------------
# TF-Privacy-compatible wrappers
# --------------------------------------------------

@dataclass
class _SingleAttackResult:
    attack_name: str
    auc: float
    adv: float

    def get_auc(self) -> float:
        return float(self.auc)

    def get_attacker_advantage(self) -> float:
        return float(self.adv)


class AttackResults:
    """
    Minimal adapter mimicking the TF-Privacy AttackResults API:
      - get_result_with_max_auc().get_auc()
      - get_result_with_max_attacker_advantage().get_attacker_advantage()
    Also keeps a .results dict for debugging / inspection.
    """

    def __init__(self, results_dict):
        self.results = results_dict

    def get_result_with_max_auc(self) -> _SingleAttackResult:
        keys = list(self.results.keys())
        aucs = np.array([self.results[k].get("auc", np.nan) for k in keys], dtype=float)
        best = keys[int(np.nanargmax(aucs))] if np.isfinite(aucs).any() else keys[0]
        r = self.results[best]
        return _SingleAttackResult(best, float(r.get("auc", np.nan)), float(r.get("adv", np.nan)))

    def get_result_with_max_attacker_advantage(self) -> _SingleAttackResult:
        keys = list(self.results.keys())
        advs = np.array([self.results[k].get("adv", np.nan) for k in keys], dtype=float)
        best = keys[int(np.nanargmax(advs))] if np.isfinite(advs).any() else keys[0]
        r = self.results[best]
        return _SingleAttackResult(best, float(r.get("auc", np.nan)), float(r.get("adv", np.nan)))

    def get_max_tpr_at_fpr(self, target_fpr: float = 0.01) -> float:
        """Max TPR@FPR across all attacks (best-case attacker at fixed false-positive rate)."""
        vals = [float(self.results[k].get("tpr_at_1fpr", target_fpr)) for k in self.results]
        valid = [v for v in vals if np.isfinite(v)]
        return float(max(valid)) if valid else float(target_fpr)

    def get_yeom_auc(self) -> float:
        """AUC from the Yeom loss-threshold attack (primary MIA metric)."""
        return float(self.results.get("THRESHOLD_ATTACK", {}).get("auc", 0.5))

    def get_yeom_tpr_at_fpr(self, target_fpr: float = 0.01) -> float:
        """TPR@FPR from the Yeom loss-threshold attack."""
        return float(self.results.get("THRESHOLD_ATTACK", {}).get("tpr_at_1fpr", target_fpr))


# --------------------------------------------------
# Robust scoring helpers
# --------------------------------------------------

def _safe_auc_and_adv(y_true: np.ndarray, scores: np.ndarray):
    """
    Compute AUC and attacker advantage robustly.
    Falls back to random-attacker baseline (AUC=0.5, ADV=0.0) on degeneracy.
    """
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    finite = np.isfinite(scores)
    if not finite.all():
        return 0.5, 0.0, True, int((~finite).sum())
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0, False, 0
    try:
        auc = float(roc_auc_score(y_true, scores))
        adv = attacker_advantage(y_true, scores)
        return auc, adv, False, 0
    except Exception:
        return 0.5, 0.0, True, 0


def _tpr_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float = 0.01) -> float:
    """TPR at a fixed FPR operating point, interpolated from the ROC curve.
    Falls back to target_fpr (random-classifier baseline) on degeneracy."""
    y_true = np.asarray(y_true).reshape(-1).astype(int)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if not np.isfinite(scores).all() or len(np.unique(y_true)) < 2:
        return float(target_fpr)
    try:
        fpr, tpr, _ = roc_curve(y_true, scores)
        return float(np.interp(target_fpr, fpr, tpr))
    except Exception:
        return float(target_fpr)


def rmia_attack(
    probs_target_m: np.ndarray,
    probs_ref_m: np.ndarray,
    probs_target_nm: np.ndarray,
    probs_ref_nm: np.ndarray,
    gamma: float = 2.0,
) -> Tuple[float, float]:
    """RMIA offline attack (Zarifzadeh et al., ICML 2024).

    For each sample x, score = fraction of non-member population z where
    LR(x) / LR(z) > gamma, where LR(x) = P_target(y|x) / P_ref(y|x).

    Members = Df (forget set), non-members/population = Dtest.

    Parameters
    ----------
    probs_target_m  : (N_m,)  softmax P(y_true|x) from target model on members
    probs_ref_m     : (N_m,)  softmax P(y_true|x) from reference model on members
    probs_target_nm : (N_nm,) softmax P(y_true|x) from target model on non-members
    probs_ref_nm    : (N_nm,) softmax P(y_true|x) from reference model on non-members
    gamma           : likelihood-ratio threshold (default 2.0)

    Returns
    -------
    (auc, tpr_at_1pct_fpr)
    """
    eps = 1e-45
    lr_m  = np.asarray(probs_target_m,  dtype=np.float64) / np.maximum(np.asarray(probs_ref_m,  dtype=np.float64), eps)
    lr_nm = np.asarray(probs_target_nm, dtype=np.float64) / np.maximum(np.asarray(probs_ref_nm, dtype=np.float64), eps)

    # score(x) = mean_z [ lr(x) > gamma * lr(z) ]
    scores_m  = (lr_m[:, None]  > gamma * lr_nm[None, :]).mean(axis=1)
    scores_nm = (lr_nm[:, None] > gamma * lr_nm[None, :]).mean(axis=1)

    y_true = np.concatenate([np.ones(len(scores_m), dtype=int), np.zeros(len(scores_nm), dtype=int)])
    scores  = np.concatenate([scores_m, scores_nm])

    auc = _safe_auc_and_adv(y_true, scores)[0]
    tpr = _tpr_at_fpr(y_true, scores)
    return float(auc), float(tpr)


def _build_feature_matrix(
    loss_train, loss_test,
    logits_train, logits_test,
    train_labels=None, test_labels=None,
):
    """
    Assemble per-sample feature matrix for classifier-based attacks.

    Features (always present):
      [0] loss              — lower => member signal
      [1] entropy           — lower => member signal
      [2] max logit         — higher => member signal
      [3] logit margin      — top1 - top2, higher => member signal

    Optional (requires labels):
      [4] LiRA scaled logit — higher => member signal
    """
    def _per_split(logits, loss, labels=None):
        logits = np.asarray(logits, dtype=np.float64)
        loss   = np.asarray(loss,   dtype=np.float64).reshape(-1)
        ent    = entropy_from_logits(logits)
        sorted_l  = np.sort(logits, axis=1)
        max_logit = sorted_l[:, -1]
        margin    = (sorted_l[:, -1] - sorted_l[:, -2]) if logits.shape[1] >= 2 \
                    else np.zeros_like(max_logit)
        cols = [loss, ent, max_logit, margin]
        if labels is not None:
            cols.append(lira_scaled_logit_score(logits, labels))
        return np.column_stack(cols)

    has_labels = (train_labels is not None) and (test_labels is not None)
    X = np.vstack([
        _per_split(logits_train, loss_train, train_labels if has_labels else None),
        _per_split(logits_test,  loss_test,  test_labels  if has_labels else None),
    ])
    y = np.concatenate([
        np.ones(len(loss_train),  dtype=int),
        np.zeros(len(loss_test),  dtype=int),
    ])
    return X, y


def _classifier_attack(X, y, clf, n_splits: int = 5, random_state: int = 0):
    """
    Stratified k-fold cross-validated membership classifier.

    Out-of-fold predicted probabilities are collected across all folds and
    used to compute a single AUC / attacker advantage over the full dataset,
    matching standard MIA evaluation practice (no data leakage between folds).
    """
    X = X.copy().astype(np.float64)
    for col in range(X.shape[1]):
        bad = ~np.isfinite(X[:, col])
        if bad.any():
            med = np.nanmedian(X[:, col])
            X[bad, col] = med if np.isfinite(med) else 0.0

    y = np.asarray(y).reshape(-1).astype(int)
    min_class = int(np.min(np.bincount(y, minlength=2)))
    if min_class < 2:
        return 0.5, 0.0, False, 0, 0.01

    n_splits_eff = max(2, min(n_splits, min_class))
    oof_probs    = np.full(len(y), np.nan)
    cv = StratifiedKFold(n_splits=n_splits_eff, shuffle=True, random_state=random_state)

    for tr_idx, val_idx in cv.split(X, y):
        scaler = StandardScaler()
        X_tr   = scaler.fit_transform(X[tr_idx])
        X_val  = scaler.transform(X[val_idx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            clf.fit(X_tr, y[tr_idx])
        oof_probs[val_idx] = clf.predict_proba(X_val)[:, 1]

    auc, adv, had_bad, n_bad = _safe_auc_and_adv(y, oof_probs)
    tpr = _tpr_at_fpr(y, oof_probs)
    return auc, adv, had_bad, n_bad, tpr


# --------------------------------------------------
# Main attack function
# --------------------------------------------------

def mia_attack(
    logits_train,
    logits_test,
    loss_train,
    loss_test,
    train_labels=None,
    test_labels=None,
    *,
    use_entropy_attacks:     bool = True,
    use_classifier_attacks:  bool = True,
    n_cv_splits:             int  = 5,
    random_state:            int  = 0,
):
    """
    Full TF-Privacy-equivalent MIA suite (Nasr et al., 2019; Salem et al., 2019).

    Attack inventory — mirrors TF-Privacy's AttackType enum:
      THRESHOLD_ATTACK            — loss threshold (Yeom et al., 2018)
      THRESHOLD_ENTROPY_ATTACK    — entropy threshold
      LOGISTIC_REGRESSION         — LR meta-classifier over model statistics
      RANDOM_FOREST               — RF meta-classifier over model statistics
      MULTI_LAYERED_PERCEPTRON    — MLP meta-classifier over model statistics
      K_NEAREST_NEIGHBORS         — kNN meta-classifier over model statistics

    Additionally includes:
      LIRA_SCALED_LOGIT_ATTACK    — single-model LiRA score (Carlini et al., 2022)

    Classifier attacks (LR / RF / MLP / kNN) use stratified k-fold CV to
    produce out-of-fold membership probability estimates, preventing data
    leakage and matching the evaluation protocol of TF-Privacy.

    The maximum AUC across all active attacks is accessible via
    get_result_with_max_auc(), providing a conservative upper bound on
    attacker advantage.

    Parameters
    ----------
    logits_train, logits_test  : ndarray (N, C)  — logits or probabilities
    loss_train,   loss_test    : ndarray (N,)    — per-sample cross-entropy loss
    train_labels, test_labels  : array-like int  — ground-truth class labels (optional)
    use_entropy_attacks        : enable THRESHOLD_ENTROPY_ATTACK (default True)
    use_classifier_attacks     : enable LR / RF / MLP / kNN attacks (default True)
    n_cv_splits                : CV folds for classifier attacks (default 5)
    random_state               : RNG seed
    """
    loss_train   = np.asarray(loss_train).reshape(-1)
    loss_test    = np.asarray(loss_test).reshape(-1)
    logits_train = np.asarray(logits_train)
    logits_test  = np.asarray(logits_test)

    y_true = np.concatenate([
        np.ones(len(loss_train),  dtype=int),
        np.zeros(len(loss_test),  dtype=int),
    ])

    results = {}

    # ── 1) THRESHOLD_ATTACK (Yeom et al.) ────────────────────────────────────
    scores_loss = -np.concatenate([loss_train, loss_test])
    auc, adv, had_bad, n_bad = _safe_auc_and_adv(y_true, scores_loss)
    tpr = _tpr_at_fpr(y_true, scores_loss)
    results["THRESHOLD_ATTACK"] = dict(auc=float(auc), adv=float(adv), tpr_at_1fpr=float(tpr),
                                       had_non_finite_scores=bool(had_bad),
                                       n_bad_scores=int(n_bad))

    # ── 2) THRESHOLD_ENTROPY_ATTACK ───────────────────────────────────────────
    if use_entropy_attacks:
        scores_ent = -np.concatenate([
            entropy_from_logits(logits_train),
            entropy_from_logits(logits_test),
        ])
        auc, adv, had_bad, n_bad = _safe_auc_and_adv(y_true, scores_ent)
        tpr = _tpr_at_fpr(y_true, scores_ent)
        results["THRESHOLD_ENTROPY_ATTACK"] = dict(auc=float(auc), adv=float(adv), tpr_at_1fpr=float(tpr),
                                                    had_non_finite_scores=bool(had_bad),
                                                    n_bad_scores=int(n_bad))

    # ── 3) LIRA_SCALED_LOGIT_ATTACK ──────────────────────────────────────────
    if train_labels is not None and test_labels is not None:
        tl = np.asarray(train_labels, dtype=int).reshape(-1)
        vl = np.asarray(test_labels,  dtype=int).reshape(-1)
        scores_lira = np.concatenate([
            lira_scaled_logit_score(logits_train, tl),
            lira_scaled_logit_score(logits_test,  vl),
        ])
        auc, adv, had_bad, n_bad = _safe_auc_and_adv(y_true, scores_lira)
        tpr = _tpr_at_fpr(y_true, scores_lira)
        results["LIRA_SCALED_LOGIT_ATTACK"] = dict(auc=float(auc), adv=float(adv), tpr_at_1fpr=float(tpr),
                                                    had_non_finite_scores=bool(had_bad),
                                                    n_bad_scores=int(n_bad))

    # ── 4-7) Classifier attacks ───────────────────────────────────────────────
    if use_classifier_attacks:
        X, y_clf = _build_feature_matrix(
            loss_train, loss_test,
            logits_train, logits_test,
            train_labels, test_labels,
        )

        classifiers = {
            # TF-Privacy: LOGISTIC_REGRESSION
            "LOGISTIC_REGRESSION": LogisticRegression(
                max_iter=1000, solver="lbfgs", C=1.0,
                random_state=random_state,
            ),
            # TF-Privacy: RANDOM_FOREST
            "RANDOM_FOREST": RandomForestClassifier(
                n_estimators=100, max_depth=None, min_samples_leaf=1,
                random_state=random_state, n_jobs=-1,
            ),
            # TF-Privacy: MULTI_LAYERED_PERCEPTRON
            # Architecture mirrors TF-Privacy's default: two hidden layers (64, 64),
            # relu activation, adam optimiser, early stopping via max_iter.
            "MULTI_LAYERED_PERCEPTRON": MLPClassifier(
                hidden_layer_sizes=(64, 64), activation="relu",
                solver="adam", max_iter=200,
                random_state=random_state,
            ),
            # TF-Privacy: K_NEAREST_NEIGHBORS
            # k=10 matches TF-Privacy's default n_neighbors.
            "K_NEAREST_NEIGHBORS": KNeighborsClassifier(
                n_neighbors=10, metric="minkowski", n_jobs=-1,
            ),
        }

        for name, clf in classifiers.items():
            auc, adv, had_bad, n_bad, tpr = _classifier_attack(
                X, y_clf, clf,
                n_splits=n_cv_splits,
                random_state=random_state,
            )
            results[name] = dict(auc=float(auc), adv=float(adv), tpr_at_1fpr=float(tpr),
                                 had_non_finite_scores=bool(had_bad),
                                 n_bad_scores=int(n_bad))

    return AttackResults(results)


def tf_attack(logits_train, logits_test, loss_train, loss_test,
              train_labels=None, test_labels=None, *, run_extended: bool = True):
    """Drop-in replacement for the original tf_attack wrapper."""
    return mia_attack(
        logits_train=logits_train,
        logits_test=logits_test,
        loss_train=loss_train,
        loss_test=loss_test,
        train_labels=train_labels,
        test_labels=test_labels,
        use_entropy_attacks=run_extended,
        use_classifier_attacks=run_extended,
    )

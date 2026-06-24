"""Evaluation metrics, group-level bootstrap, and plots for the final model.

This module extracts the evaluation logic previously inlined in
``notebooks/06_final_model_evaluation.ipynb`` so it can be imported and tested.
Metrics are computed from predicted probabilities (``y_prob``) and a decision
``threshold`` where applicable; AUROC/AUPRC are threshold-independent.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src.modeling import RANDOM_STATE


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, float]:
    """Compute discrimination, calibration, and threshold-based metrics.

    Returns AUROC, AUPRC, sensitivity, specificity, PPV, NPV, and Brier score
    (the metrics named in the project spec). Additional count-based fields
    (``f1``, ``flagged``, ``flagged_pct``, and the confusion-matrix cells) are
    included so a threshold sweep can be built directly from this function.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan

    return {
        "auroc": roc_auc_score(y_true, y_prob),
        "auprc": average_precision_score(y_true, y_prob),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "brier_score": brier_score_loss(y_true, y_prob),
        "threshold": float(threshold),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "flagged": int(y_pred.sum()),
        "flagged_pct": float(y_pred.mean()),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    groups: np.ndarray,
    n_boot: int = 1000,
    seed: int = RANDOM_STATE,
) -> tuple[float, float, float]:
    """Patient-level bootstrap 95% CI for ``metric_fn``.

    The dataset has many rows per patient (one ``ID`` has 266 rows), so the
    bootstrap resamples whole patient groups with replacement and pulls all
    rows for each sampled ``ID`` before scoring. Resampling at the row level
    would inflate the effective sample size and understate uncertainty.

    ``metric_fn`` must accept ``(y_true, y_prob)`` and return a float (e.g.
    ``roc_auc_score`` or a threshold-based sensitivity closure).

    Returns
    -------
    tuple[float, float, float]
        ``(point_estimate, lower_2.5pct, upper_97.5pct)`` where
        ``point_estimate`` is ``metric_fn`` evaluated on the full (observed)
        input and the bounds are percentiles of the bootstrap distribution.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    groups = np.asarray(groups)

    point_estimate = float(metric_fn(y_true, y_prob))

    unique_groups = np.unique(groups)
    group_to_rows = {g: np.flatnonzero(groups == g) for g in unique_groups}
    rng = np.random.default_rng(seed)

    estimates = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_groups, size=len(unique_groups), replace=True)
        idx = np.concatenate([group_to_rows[g] for g in sampled])
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            # Metric undefined when a bootstrap sample has a single class.
            continue
        try:
            estimates.append(float(metric_fn(yt, yp)))
        except ValueError:
            continue

    estimates = np.asarray(estimates, dtype=float)
    if estimates.size == 0:
        return (point_estimate, np.nan, np.nan)
    return (
        point_estimate,
        float(np.percentile(estimates, 2.5)),
        float(np.percentile(estimates, 97.5)),
    )


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ax: plt.Axes | None = None,
    n_bins: int = 10,
) -> plt.Axes:
    """Plot a reliability diagram annotated with the Brier score."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    prob_true, prob_pred = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy="quantile"
    )
    brier = brier_score_loss(y_true, y_prob)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label="Observed calibration")
    ax.set_xlabel("Mean predicted risk")
    ax.set_ylabel("Observed event rate")
    ax.set_title("Calibration curve")
    ax.legend(loc="upper left")
    ax.annotate(
        f"Brier score = {brier:.4f}",
        xy=(0.02, 0.92),
        xycoords="axes fraction",
        ha="left",
        va="top",
    )
    return ax


def plot_roc_pr(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ax: tuple[plt.Axes, plt.Axes] | None = None,
) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot combined ROC and precision-recall curves side by side.

    ``ax`` may be ``None`` (a new 1x2 figure is created) or a pair of axes
    ``(ax_roc, ax_pr)``. Returns ``(fig, (ax_roc, ax_pr))``.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if ax is None:
        fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(13, 6))
    else:
        ax_roc, ax_pr = ax
        fig = ax_roc.figure

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auroc = roc_auc_score(y_true, y_prob)
    ax_roc.plot(fpr, tpr, linewidth=2, label=f"AUROC = {auroc:.3f}")
    ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    ax_roc.set_xlabel("False positive rate")
    ax_roc.set_ylabel("True positive rate / sensitivity")
    ax_roc.set_title("ROC curve")
    ax_roc.legend(loc="lower right")

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    prevalence = float(y_true.mean())
    ax_pr.plot(recall, precision, linewidth=2, label=f"AUPRC = {auprc:.3f}")
    ax_pr.axhline(
        prevalence, linestyle="--", color="gray", label=f"Prevalence = {prevalence:.3f}"
    )
    ax_pr.set_xlabel("Recall / sensitivity")
    ax_pr.set_ylabel("Precision / PPV")
    ax_pr.set_title("Precision-recall curve")
    ax_pr.legend(loc="lower left")

    fig.tight_layout()
    return fig, (ax_roc, ax_pr)

"""Semantic-segmentation metrics matching the competition definitions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from railway3d_seg.labels import CLASS_NAMES, IGNORE_LABEL, NUM_CLASSES, validate_labels


def confusion_matrix(
    target: np.ndarray,
    prediction: np.ndarray,
    *,
    num_classes: int = NUM_CLASSES,
    ignore_label: int = IGNORE_LABEL,
) -> np.ndarray:
    truth = np.asarray(target, dtype=np.int64).reshape(-1)
    pred = np.asarray(prediction, dtype=np.int64).reshape(-1)
    if truth.shape != pred.shape:
        raise ValueError("target and prediction must contain the same number of points")
    valid = truth != ignore_label
    truth = truth[valid]
    pred = pred[valid]
    validate_labels(truth, allow_ignore=False)
    validate_labels(pred, allow_ignore=False)
    encoded = truth * num_classes + pred
    return np.bincount(encoded, minlength=num_classes**2).reshape(num_classes, num_classes)


@dataclass(frozen=True)
class SemanticMetrics:
    mean_iou: float
    overall_accuracy: float
    per_class_iou: dict[str, float]
    confusion: np.ndarray

    def as_dict(self) -> dict[str, object]:
        return {
            "mean_iou": self.mean_iou,
            "overall_accuracy": self.overall_accuracy,
            "per_class_iou": self.per_class_iou,
            "confusion_matrix": self.confusion.tolist(),
        }


def metrics_from_confusion(matrix: np.ndarray) -> SemanticMetrics:
    confusion = np.asarray(matrix, dtype=np.int64)
    if confusion.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError(f"confusion matrix must have shape {(NUM_CLASSES, NUM_CLASSES)}")
    true_positive = np.diag(confusion).astype(np.float64)
    union = confusion.sum(axis=1) + confusion.sum(axis=0) - true_positive
    iou = np.divide(
        true_positive,
        union,
        out=np.full_like(true_positive, np.nan),
        where=union > 0,
    )
    total = confusion.sum()
    accuracy = float(true_positive.sum() / total) if total else float("nan")
    return SemanticMetrics(
        mean_iou=float(np.nanmean(iou)),
        overall_accuracy=accuracy,
        per_class_iou={name: float(value) for name, value in zip(CLASS_NAMES, iou)},
        confusion=confusion,
    )


def evaluate_semantic(target: np.ndarray, prediction: np.ndarray) -> SemanticMetrics:
    return metrics_from_confusion(confusion_matrix(target, prediction))


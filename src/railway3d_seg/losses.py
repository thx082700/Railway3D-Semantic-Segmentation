"""Class-balanced cross entropy and multiclass Lovasz-Softmax loss."""

from __future__ import annotations


def _torch():
    try:
        import torch
        from torch.nn import functional
    except ImportError as error:  # pragma: no cover - optional training dependency
        raise ImportError("losses require PyTorch; install the train extra") from error
    return torch, functional


def _lovasz_grad(sorted_ground_truth):
    _torch_module, _ = _torch()
    count = sorted_ground_truth.numel()
    intersection = sorted_ground_truth.sum() - sorted_ground_truth.float().cumsum(0)
    union = sorted_ground_truth.sum() + (1.0 - sorted_ground_truth).float().cumsum(0)
    gradient = 1.0 - intersection / union.clamp_min(1e-12)
    if count > 1:
        gradient[1:count] -= gradient[:-1]
    return gradient


def lovasz_softmax_flat(probabilities, labels, *, ignore_label: int = -1):
    torch, _ = _torch()
    valid = labels != ignore_label
    probabilities = probabilities[valid]
    labels = labels[valid]
    if not probabilities.numel():
        return probabilities.sum() * 0.0
    losses = []
    for class_index in range(probabilities.shape[1]):
        foreground = (labels == class_index).float()
        if foreground.sum() == 0:
            continue
        errors = (foreground - probabilities[:, class_index]).abs()
        sorted_errors, permutation = torch.sort(errors, descending=True)
        sorted_foreground = foreground[permutation]
        losses.append(torch.dot(sorted_errors, _lovasz_grad(sorted_foreground)))
    return torch.stack(losses).mean() if losses else probabilities.sum() * 0.0


def semantic_loss(logits, labels, *, class_weights=None, lovasz_weight: float = 1.0):
    torch, functional = _torch()
    weights = None
    if class_weights is not None:
        weights = torch.as_tensor(class_weights, dtype=logits.dtype, device=logits.device)
    cross_entropy = functional.cross_entropy(logits, labels, weight=weights, ignore_index=-1)
    lovasz = lovasz_softmax_flat(torch.softmax(logits, dim=1), labels)
    total = cross_entropy + float(lovasz_weight) * lovasz
    return total, {"cross_entropy": cross_entropy.detach(), "lovasz": lovasz.detach()}

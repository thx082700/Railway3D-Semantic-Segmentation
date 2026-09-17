import numpy as np

from railway3d_seg.metrics import confusion_matrix, evaluate_semantic


def test_perfect_semantic_metrics():
    labels = np.arange(11, dtype=np.int64)
    metrics = evaluate_semantic(labels, labels)
    assert metrics.mean_iou == 1.0
    assert metrics.overall_accuracy == 1.0
    assert all(value == 1.0 for value in metrics.per_class_iou.values())


def test_confusion_matrix_ignores_minus_one():
    target = np.array([0, 1, -1, 2])
    prediction = np.array([0, 2, 9, 2])
    matrix = confusion_matrix(target, prediction)
    assert matrix.sum() == 3
    assert matrix[0, 0] == 1
    assert matrix[1, 2] == 1
    assert matrix[2, 2] == 1


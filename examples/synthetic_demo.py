"""CPU-only smoke demo for metrics and submission packaging."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from railway3d_seg.io import save_prediction
from railway3d_seg.metrics import evaluate_semantic
from railway3d_seg.submission import build_submission


def main() -> None:
    truth = np.arange(110, dtype=np.uint8) % 11
    prediction = truth.copy()
    prediction[::10] = (prediction[::10] + 1) % 11
    metrics = evaluate_semantic(truth, prediction)
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        save_prediction(root / "L5-1-M01-002.npy", prediction)
        report = build_submission(root, root / "submission.zip")
        print(f"mIoU={metrics.mean_iou:.3f}, OA={metrics.overall_accuracy:.3f}")
        print(f"validated {report.files} file / {report.points} points")


if __name__ == "__main__":
    main()


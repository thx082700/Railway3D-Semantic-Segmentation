"""Strict builder and validator for the flat Codabench submission archive."""

from __future__ import annotations

import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from railway3d_seg.io import competition_stem, load_prediction, read_ply


@dataclass(frozen=True)
class SubmissionReport:
    files: int
    points: int
    archive: Path | None


def validate_prediction_directory(
    prediction_dir: str | Path,
    *,
    test_clouds: Iterable[str | Path] | None = None,
) -> SubmissionReport:
    directory = Path(prediction_dir)
    files = sorted(directory.glob("*.npy"))
    nested = list(directory.glob("**/*.npy"))
    if not files:
        raise ValueError(f"no .npy predictions found in {directory}")
    if len(nested) != len(files):
        raise ValueError("predictions must be directly inside one flat directory")

    expected: dict[str, Path] | None = None
    if test_clouds is not None:
        expected = {competition_stem(path): Path(path) for path in test_clouds}
        actual = {path.stem for path in files}
        missing = sorted(set(expected) - actual)
        extra = sorted(actual - set(expected))
        if missing or extra:
            raise ValueError(f"filename mismatch; missing={missing}, extra={extra}")

    points = 0
    for path in files:
        prediction = np.load(path, allow_pickle=False)
        if prediction.dtype != np.uint8:
            raise ValueError(f"{path.name} must have dtype uint8, got {prediction.dtype}")
        prediction = load_prediction(path)
        points += int(prediction.size)
        if expected is not None:
            expected_points = read_ply(expected[path.stem]).size
            if prediction.size != expected_points:
                raise ValueError(
                    f"{path.name}: {prediction.size} labels for {expected_points} input points"
                )
    return SubmissionReport(len(files), points, None)


def build_submission(
    prediction_dir: str | Path,
    archive: str | Path,
    *,
    test_clouds: Iterable[str | Path] | None = None,
) -> SubmissionReport:
    report = validate_prediction_directory(prediction_dir, test_clouds=test_clouds)
    destination = Path(archive)
    destination.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(prediction_dir).glob("*.npy"))
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, arcname=path.name)
    with zipfile.ZipFile(destination) as bundle:
        names = bundle.namelist()
        if any("/" in name or "\\" in name for name in names):
            raise RuntimeError("internal error: submission archive is not flat")
    return SubmissionReport(report.files, report.points, destination)


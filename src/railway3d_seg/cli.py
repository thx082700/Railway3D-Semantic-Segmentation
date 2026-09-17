"""CPU-friendly command-line entry points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from railway3d_seg.io import describe_cloud, discover_ply_files, load_prediction, read_ply
from railway3d_seg.metrics import confusion_matrix, metrics_from_confusion
from railway3d_seg.submission import build_submission


def evaluate_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate WHU-Railway3D validation predictions")
    parser.add_argument("reference", type=Path, help="directory containing labeled PLY or NPY files")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    matrix = np.zeros((11, 11), dtype=np.int64)
    reference_files = sorted(args.reference.glob("*.npy"))
    if reference_files:
        for truth_path in reference_files:
            truth = np.load(truth_path, allow_pickle=False)
            prediction = load_prediction(args.predictions / truth_path.name)
            matrix += confusion_matrix(truth, prediction)
    else:
        for truth_path in discover_ply_files(args.reference):
            cloud = read_ply(truth_path, require_labels=True)
            prediction = load_prediction(args.predictions / f"{truth_path.stem}.npy")
            matrix += confusion_matrix(cloud.labels, prediction)  # type: ignore[arg-type]
    payload = metrics_from_confusion(matrix).as_dict()
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


def submission_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build a flat Codabench submission ZIP")
    parser.add_argument("predictions", type=Path)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--test-root", type=Path)
    args = parser.parse_args(argv)
    test_clouds = discover_ply_files(args.test_root) if args.test_root else None
    report = build_submission(args.predictions, args.archive, test_clouds=test_clouds)
    print(json.dumps({"archive": str(report.archive), "files": report.files, "points": report.points}))


def inspect_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Inspect PLY properties and label counts")
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--label-property")
    args = parser.parse_args(argv)
    for path in args.paths:
        print(json.dumps(describe_cloud(path, label_property=args.label_property), indent=2))


"""Audit and visualize a Codabench prediction archive without private test labels."""

from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from railway3d_seg.io import read_ply
from railway3d_seg.labels import CLASS_NAMES, NUM_CLASSES, validate_labels

CLASS_COLORS: tuple[str, ...] = (
    "#e63946",  # rails
    "#c89b3c",  # track bed
    "#7b2cbf",  # masts
    "#ff70a6",  # support devices
    "#00b4d8",  # overhead lines
    "#f77f00",  # fences
    "#6c757d",  # poles
    "#2a9d8f",  # vegetation
    "#457b9d",  # buildings
    "#8d6e63",  # ground
    "#adb5bd",  # others
)


@dataclass(frozen=True)
class ScenePrediction:
    name: str
    labels: np.ndarray
    source_dtype: str
    counts: np.ndarray


@dataclass(frozen=True)
class SubmissionSummary:
    archive: Path
    sha256: str
    archive_bytes: int
    scenes: tuple[ScenePrediction, ...]
    ignored_members: tuple[str, ...]

    @property
    def total_points(self) -> int:
        return sum(int(scene.labels.size) for scene in self.scenes)

    @property
    def counts(self) -> np.ndarray:
        return sum((scene.counts for scene in self.scenes), np.zeros(NUM_CLASSES, dtype=np.int64))


def read_submission_archive(path: str | Path) -> SubmissionSummary:
    """Load flat prediction vectors while retaining audit information about the original ZIP."""

    archive = Path(path)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    scenes: list[ScenePrediction] = []
    ignored: list[str] = []
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.namelist():
            if member.startswith("__MACOSX/") or not member.lower().endswith(".npy"):
                ignored.append(member)
                continue
            array = np.load(io.BytesIO(bundle.read(member)), allow_pickle=False)
            if array.ndim != 1:
                raise ValueError(f"{member}: expected a one-dimensional prediction vector")
            validate_labels(array, allow_ignore=False)
            scenes.append(
                ScenePrediction(
                    name=Path(member).stem,
                    labels=array.astype(np.uint8, copy=False),
                    source_dtype=str(array.dtype),
                    counts=np.bincount(array, minlength=NUM_CLASSES),
                )
            )
    if not scenes:
        raise ValueError(f"no prediction NPY files found in {archive}")
    scenes.sort(key=lambda scene: scene.name)
    return SubmissionSummary(
        archive=archive,
        sha256=digest,
        archive_bytes=archive.stat().st_size,
        scenes=tuple(scenes),
        ignored_members=tuple(ignored),
    )


def summary_payload(summary: SubmissionSummary) -> dict[str, object]:
    total = summary.total_points
    overall_counts = summary.counts
    return {
        "archive": summary.archive.name,
        "sha256": summary.sha256,
        "archive_bytes": summary.archive_bytes,
        "prediction_files": len(summary.scenes),
        "total_points": total,
        "ignored_zip_members": list(summary.ignored_members),
        "overall": {
            CLASS_NAMES[index]: {
                "label": index,
                "points": int(overall_counts[index]),
                "fraction": float(overall_counts[index] / total),
            }
            for index in range(NUM_CLASSES)
        },
        "scenes": [
            {
                "name": scene.name,
                "points": int(scene.labels.size),
                "source_dtype": scene.source_dtype,
                "class_counts": {
                    CLASS_NAMES[index]: int(scene.counts[index]) for index in range(NUM_CLASSES)
                },
            }
            for scene in summary.scenes
        ],
    }


def write_manifest(summary: SubmissionSummary, output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary_payload(summary), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _svg_header(width: int, height: int, title: str, description: str) -> list[str]:
    return [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        ),
        f'<title id="title">{html.escape(title)}</title>',
        f'<desc id="desc">{html.escape(description)}</desc>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        (
            '<style>text{font-family:Inter,Arial,sans-serif;fill:#172033}.title{font-size:28px;'
            'font-weight:700}.subtitle{font-size:15px;fill:#536176}.label{font-size:14px;}'
            '.small{font-size:12px;fill:#536176}.value{font-size:13px;font-weight:600}</style>'
        ),
    ]


def _write_svg(lines: list[str], output: str | Path) -> Path:
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join([*lines, "</svg>", ""]), encoding="utf-8")
    return destination


def write_overall_distribution_svg(summary: SubmissionSummary, output: str | Path) -> Path:
    width, height = 1120, 670
    counts = summary.counts
    total = summary.total_points
    lines = _svg_header(
        width,
        height,
        "Official submission label distribution",
        "Class proportions computed directly from recovered Codabench submission 353373.",
    )
    lines.extend(
        [
            '<text x="54" y="58" class="title">Official submission label distribution</text>',
            (
                f'<text x="54" y="88" class="subtitle">Submission 353373 · '
                f'{len(summary.scenes)} scenes · {total:,} point predictions</text>'
            ),
        ]
    )
    chart_x, chart_width = 265, 720
    bar_height, gap = 28, 20
    maximum = int(counts.max())
    for index, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS, strict=True)):
        y = 125 + index * (bar_height + gap)
        bar_width = chart_width * int(counts[index]) / maximum
        fraction = int(counts[index]) / total
        display_name = name.replace("_", " ").title()
        lines.extend(
            [
                f'<text x="54" y="{y + 20}" class="label">{html.escape(display_name)}</text>',
                (
                    f'<rect x="{chart_x}" y="{y}" width="{chart_width}" height="{bar_height}" '
                    'rx="5" fill="#e8edf3"/>'
                ),
                (
                    f'<rect x="{chart_x}" y="{y}" width="{bar_width:.2f}" '
                    f'height="{bar_height}" rx="5" fill="{color}"/>'
                ),
                (
                    f'<text x="{chart_x + bar_width + 10:.2f}" y="{y + 20}" class="value">'
                    f'{fraction:.1%}</text>'
                ),
            ]
        )
    lines.append(
        '<text x="54" y="648" class="small">Counts describe recovered predictions, not ground-truth frequency or accuracy.</text>'
    )
    return _write_svg(lines, output)


def write_scene_composition_svg(summary: SubmissionSummary, output: str | Path) -> Path:
    width, height = 1220, 700
    lines = _svg_header(
        width,
        height,
        "Per-scene composition of official predictions",
        "Normalized class composition for each recovered Urban Railway test prediction vector.",
    )
    lines.extend(
        [
            '<text x="54" y="58" class="title">Per-scene composition of official predictions</text>',
            '<text x="54" y="88" class="subtitle">Each bar is normalized to 100%; values come from the original accepted archive.</text>',
        ]
    )
    bar_x, bar_width, bar_height = 230, 920, 38
    for scene_index, scene in enumerate(summary.scenes):
        y = 130 + scene_index * 58
        lines.append(f'<text x="54" y="{y + 25}" class="label">{html.escape(scene.name)}</text>')
        cursor = float(bar_x)
        for class_index, color in enumerate(CLASS_COLORS):
            width_part = bar_width * int(scene.counts[class_index]) / scene.labels.size
            if width_part > 0:
                lines.append(
                    f'<rect x="{cursor:.2f}" y="{y}" width="{width_part:.2f}" '
                    f'height="{bar_height}" fill="{color}"/>'
                )
            cursor += width_part
        lines.append(
            f'<text x="{bar_x + bar_width + 14}" y="{y + 25}" class="small">'
            f'{scene.labels.size / 1_000_000:.1f}M</text>'
        )
    legend_y = 618
    for index, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS, strict=True)):
        column = index % 6
        row = index // 6
        x = 54 + column * 190
        y = legend_y + row * 28
        lines.extend(
            [
                f'<rect x="{x}" y="{y - 12}" width="14" height="14" rx="2" fill="{color}"/>',
                (
                    f'<text x="{x + 22}" y="{y}" class="small">'
                    f'{html.escape(name.replace("_", " ").title())}</text>'
                ),
            ]
        )
    return _write_svg(lines, output)


def _project_panel(
    xyz: np.ndarray,
    labels: np.ndarray,
    *,
    horizontal_axis: int,
    vertical_axis: int,
    x: float,
    y: float,
    width: float,
    height: float,
) -> list[str]:
    horizontal = xyz[:, horizontal_axis].astype(np.float64)
    vertical = xyz[:, vertical_axis].astype(np.float64)
    horizontal -= horizontal.min()
    vertical -= vertical.min()
    horizontal /= max(float(horizontal.max()), 1e-12)
    vertical /= max(float(vertical.max()), 1e-12)
    px = x + horizontal * width
    py = y + (1.0 - vertical) * height
    return [
        f'<circle cx="{point_x:.2f}" cy="{point_y:.2f}" r="0.75" '
        f'fill="{CLASS_COLORS[int(label)]}" fill-opacity="0.78"/>'
        for point_x, point_y, label in zip(px, py, labels, strict=True)
    ]


def write_spatial_prediction_svg(
    scene: ScenePrediction,
    ply_path: str | Path,
    output: str | Path,
    *,
    max_points: int = 30_000,
    seed: int = 42,
) -> Path:
    """Render true top/side projections when the matching official test PLY is available."""

    cloud = read_ply(ply_path)
    if cloud.size != scene.labels.size:
        raise ValueError(
            f"{scene.name}: {scene.labels.size} predictions for {cloud.size} PLY points"
        )
    rng = np.random.default_rng(seed)
    if cloud.size > max_points:
        selected = np.sort(rng.choice(cloud.size, size=max_points, replace=False))
    else:
        selected = np.arange(cloud.size)
    xyz = cloud.xyz[selected]
    labels = scene.labels[selected]
    width, height = 1440, 760
    lines = _svg_header(
        width,
        height,
        f"Spatial prediction: {scene.name}",
        "Top and side projections made by pairing the recovered prediction vector with the official PLY in original point order.",
    )
    lines.extend(
        [
            f'<text x="54" y="58" class="title">Spatial prediction · {html.escape(scene.name)}</text>',
            (
                f'<text x="54" y="88" class="subtitle">{cloud.size:,} predictions · '
                f'{selected.size:,} displayed · deterministic seed {seed}</text>'
            ),
            '<rect x="54" y="130" width="646" height="460" rx="8" fill="#111827"/>',
            '<rect x="740" y="130" width="646" height="460" rx="8" fill="#111827"/>',
            '<text x="66" y="120" class="label">Top view (X–Y)</text>',
            '<text x="752" y="120" class="label">Side view (X–Z)</text>',
        ]
    )
    lines.extend(
        _project_panel(
            xyz,
            labels,
            horizontal_axis=0,
            vertical_axis=1,
            x=62,
            y=138,
            width=630,
            height=444,
        )
    )
    lines.extend(
        _project_panel(
            xyz,
            labels,
            horizontal_axis=0,
            vertical_axis=2,
            x=748,
            y=138,
            width=630,
            height=444,
        )
    )
    legend_y = 646
    for index, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS, strict=True)):
        column = index % 6
        row = index // 6
        x = 54 + column * 225
        y = legend_y + row * 32
        lines.extend(
            [
                f'<rect x="{x}" y="{y - 13}" width="15" height="15" rx="2" fill="{color}"/>',
                (
                    f'<text x="{x + 23}" y="{y}" class="small">'
                    f'{html.escape(name.replace("_", " ").title())}</text>'
                ),
            ]
        )
    lines.append(
        '<text x="54" y="732" class="small">Test ground truth is private; colors show predictions only.</text>'
    )
    return _write_svg(lines, output)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Audit a Codabench archive and render prediction visualizations"
    )
    parser.add_argument("submission", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("results/submission_analysis"))
    parser.add_argument("--assets-dir", type=Path, default=Path("assets"))
    parser.add_argument(
        "--point-cloud-root",
        type=Path,
        help="official test PLY directory; enables real spatial top/side views",
    )
    parser.add_argument(
        "--scene",
        action="append",
        default=[],
        help="scene stem to spatially render; repeat for multiple scenes (default: first scene)",
    )
    parser.add_argument("--max-points", type=int, default=30_000)
    args = parser.parse_args(argv)

    summary = read_submission_archive(args.submission)
    manifest = write_manifest(summary, args.output_dir / "manifest.json")
    distribution = write_overall_distribution_svg(
        summary, args.assets_dir / "official_submission_distribution.svg"
    )
    composition = write_scene_composition_svg(
        summary, args.assets_dir / "official_submission_scene_composition.svg"
    )
    outputs = [manifest, distribution, composition]

    if args.point_cloud_root:
        requested = set(args.scene or [summary.scenes[0].name])
        available = {scene.name: scene for scene in summary.scenes}
        missing = sorted(requested - set(available))
        if missing:
            raise ValueError(f"unknown scene(s): {missing}")
        for scene_name in sorted(requested):
            ply_path = args.point_cloud_root / f"{scene_name}.ply"
            if not ply_path.exists():
                raise FileNotFoundError(f"matching official point cloud not found: {ply_path}")
            outputs.append(
                write_spatial_prediction_svg(
                    available[scene_name],
                    ply_path,
                    args.assets_dir / f"prediction_{scene_name}.svg",
                    max_points=args.max_points,
                )
            )

    print(
        json.dumps(
            {
                "archive": str(summary.archive),
                "sha256": summary.sha256,
                "scenes": len(summary.scenes),
                "points": summary.total_points,
                "outputs": [str(path) for path in outputs],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

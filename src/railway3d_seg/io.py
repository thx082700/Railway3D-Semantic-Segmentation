"""PLY input, prediction output, and manifest helpers."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from plyfile import PlyData

from railway3d_seg.labels import NUM_CLASSES, validate_labels

_LABEL_ALIASES = ("label", "labels", "semantic", "semantic_label", "class", "classification")
_ATTRIBUTE_ALIASES = {
    "intensity": ("intensity", "reflectance", "scalar_Intensity"),
    "scan_angle": ("scan_angle", "scan_angle_rank", "scanangle"),
    "return_number": ("return_number", "return_num", "number_of_returns", "returns"),
}


@dataclass(frozen=True)
class RailwayCloud:
    xyz: np.ndarray
    attributes: dict[str, np.ndarray]
    labels: np.ndarray | None
    property_names: tuple[str, ...]
    path: Path

    @property
    def size(self) -> int:
        return int(self.xyz.shape[0])


def _first_property(names: set[str], aliases: Iterable[str]) -> str | None:
    return next((candidate for candidate in aliases if candidate in names), None)


def read_ply(
    path: str | Path,
    *,
    require_labels: bool = False,
    label_property: str | None = None,
) -> RailwayCloud:
    """Read an official or alias-compatible WHU-Railway3D PLY tile.

    The loader deliberately reports every property name. Dataset releases can
    vary in attribute spelling, so ``railway3d-inspect`` should be run before
    training and ``label_property`` can be set explicitly when needed.
    """

    source = Path(path)
    ply = PlyData.read(source)
    if "vertex" not in ply:
        raise ValueError(f"{source} does not contain a vertex element")
    vertex = ply["vertex"].data
    names = tuple(vertex.dtype.names or ())
    name_set = set(names)
    missing_xyz = {name for name in ("x", "y", "z") if name not in name_set}
    if missing_xyz:
        raise ValueError(f"{source} is missing coordinates: {sorted(missing_xyz)}")
    xyz = np.column_stack([vertex[name] for name in ("x", "y", "z")]).astype(
        np.float32, copy=False
    )

    attributes: dict[str, np.ndarray] = {}
    for canonical, aliases in _ATTRIBUTE_ALIASES.items():
        match = _first_property(name_set, aliases)
        if match is not None:
            attributes[canonical] = np.asarray(vertex[match], dtype=np.float32)

    label_name = label_property or _first_property(name_set, _LABEL_ALIASES)
    labels = None
    if label_name is not None:
        if label_name not in name_set:
            raise ValueError(f"label property {label_name!r} is absent from {source}")
        labels = np.asarray(vertex[label_name], dtype=np.int64).reshape(-1)
        validate_labels(labels)
    if require_labels and labels is None:
        raise ValueError(
            f"{source} has no semantic-label property; available properties: {', '.join(names)}"
        )
    return RailwayCloud(xyz, attributes, labels, names, source)


def read_manifest(root: str | Path, manifest: str | Path) -> list[Path]:
    root_path = Path(root)
    entries: list[Path] = []
    with Path(manifest).open("r", encoding="utf-8") as stream:
        for raw in stream:
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            resolved = root_path / line
            if not resolved.exists():
                raise FileNotFoundError(f"manifest entry does not exist: {resolved}")
            entries.append(resolved)
    if not entries:
        raise ValueError(f"manifest has no point clouds: {manifest}")
    return entries


def discover_ply_files(root: str | Path) -> list[Path]:
    return sorted(Path(root).rglob("*.ply"))


def competition_stem(path: str | Path) -> str:
    return Path(path).stem


def save_prediction(path: str | Path, labels: np.ndarray) -> Path:
    """Save the exact Codabench contract: one flat uint8 vector per cloud."""

    prediction = np.asarray(labels).reshape(-1)
    validate_labels(prediction, allow_ignore=False)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.save(destination, prediction.astype(np.uint8, copy=False), allow_pickle=False)
    return destination


def load_prediction(path: str | Path) -> np.ndarray:
    prediction = np.load(Path(path), allow_pickle=False)
    if prediction.ndim != 1:
        raise ValueError(f"prediction must be one-dimensional: {path}")
    validate_labels(prediction, allow_ignore=False)
    return prediction.astype(np.uint8, copy=False)


def describe_cloud(path: str | Path, *, label_property: str | None = None) -> dict[str, object]:
    cloud = read_ply(path, label_property=label_property)
    description: dict[str, object] = {
        "path": str(cloud.path),
        "points": cloud.size,
        "properties": list(cloud.property_names),
        "detected_attributes": sorted(cloud.attributes),
        "has_labels": cloud.labels is not None,
        "bounds_min": cloud.xyz.min(axis=0).tolist() if cloud.size else [],
        "bounds_max": cloud.xyz.max(axis=0).tolist() if cloud.size else [],
    }
    if cloud.labels is not None:
        counts = np.bincount(cloud.labels[cloud.labels >= 0], minlength=NUM_CLASSES)
        description["class_counts"] = counts.tolist()
    return description

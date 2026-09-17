"""Block sampling, sparse voxelization, and dataset adapters."""

from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from railway3d_seg.io import RailwayCloud, read_ply


@dataclass(frozen=True)
class VoxelSample:
    coords: np.ndarray
    features: np.ndarray
    labels: np.ndarray | None
    inverse: np.ndarray
    point_indices: np.ndarray
    path: Path


def _normalized_attribute(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not values.size:
        return values
    low, high = np.percentile(values, [1.0, 99.0])
    scale = max(float(high - low), 1e-6)
    return np.clip((values - low) / scale, 0.0, 1.0).astype(np.float32, copy=False)


def build_features(
    xyz: np.ndarray,
    attributes: dict[str, np.ndarray],
    *,
    feature_properties: Sequence[str] = ("intensity",),
    spatial_scale: float = 50.0,
) -> np.ndarray:
    """Build centered XYZ plus robustly scaled sensor attributes."""

    points = np.asarray(xyz, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("xyz must have shape [N, 3]")
    centered = points - points.mean(axis=0, keepdims=True) if len(points) else points.copy()
    channels = [centered / max(float(spatial_scale), 1e-6)]
    for name in feature_properties:
        values = attributes.get(name)
        if values is None:
            values = np.zeros(len(points), dtype=np.float32)
        channels.append(_normalized_attribute(values).reshape(-1, 1))
    return np.concatenate(channels, axis=1).astype(np.float32, copy=False)


def voxelize_arrays(
    xyz: np.ndarray,
    attributes: dict[str, np.ndarray],
    labels: np.ndarray | None,
    point_indices: np.ndarray,
    *,
    path: str | Path,
    voxel_size: float,
    feature_properties: Sequence[str],
    spatial_scale: float,
) -> VoxelSample:
    """Voxelize selected point indices while retaining the point-to-voxel map."""

    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive")
    selected = np.asarray(point_indices, dtype=np.int64).reshape(-1)
    if selected.size == 0:
        raise ValueError("cannot voxelize an empty block")
    block_xyz = np.asarray(xyz, dtype=np.float32)[selected]
    origin = block_xyz.min(axis=0, keepdims=True)
    quantized = np.floor((block_xyz - origin) / float(voxel_size)).astype(np.int32)
    _, unique, inverse = np.unique(quantized, axis=0, return_index=True, return_inverse=True)
    block_attributes = {name: np.asarray(values)[selected] for name, values in attributes.items()}
    features = build_features(
        block_xyz,
        block_attributes,
        feature_properties=feature_properties,
        spatial_scale=spatial_scale,
    )
    voxel_labels = None if labels is None else np.asarray(labels, dtype=np.int64)[selected][unique]
    return VoxelSample(
        coords=quantized[unique],
        features=features[unique],
        labels=voxel_labels,
        inverse=inverse.astype(np.int64, copy=False),
        point_indices=selected,
        path=Path(path),
    )


def random_block_indices(
    xyz: np.ndarray,
    *,
    block_size: float,
    max_points: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a bounded XY block, preserving all classes present around its center."""

    points = np.asarray(xyz)
    if not len(points):
        raise ValueError("point cloud is empty")
    center = points[int(rng.integers(len(points))), :2]
    half = float(block_size) / 2.0
    mask = np.all(np.abs(points[:, :2] - center) <= half, axis=1)
    indices = np.flatnonzero(mask)
    if not len(indices):
        indices = np.array([int(rng.integers(len(points)))], dtype=np.int64)
    if max_points > 0 and len(indices) > max_points:
        indices = rng.choice(indices, size=max_points, replace=False)
    return np.sort(indices.astype(np.int64, copy=False))


def sliding_block_indices(
    xyz: np.ndarray,
    *,
    block_size: float,
    overlap: float,
) -> list[np.ndarray]:
    """Cover a scene with overlapping XY blocks for bounded-memory inference."""

    points = np.asarray(xyz)
    if not len(points):
        return []
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    lower = points[:, :2].min(axis=0)
    upper = points[:, :2].max(axis=0)
    stride = block_size * (1.0 - overlap)
    starts: list[np.ndarray] = []
    counts = np.maximum(np.ceil((upper - lower - block_size) / stride).astype(int) + 1, 1)
    for ix in range(int(counts[0])):
        for iy in range(int(counts[1])):
            start = lower + np.array([ix, iy], dtype=np.float64) * stride
            start = np.minimum(start, upper - block_size)
            start = np.maximum(start, lower)
            inside = np.all(
                (points[:, :2] >= start) & (points[:, :2] <= start + block_size), axis=1
            )
            indices = np.flatnonzero(inside).astype(np.int64, copy=False)
            if len(indices):
                starts.append(indices)
    if not starts:
        starts.append(np.arange(len(points), dtype=np.int64))
    return starts


def augment_xyz(
    xyz: np.ndarray, rng: np.random.Generator, *, rotate: bool = True, jitter: float = 0.01
) -> np.ndarray:
    points = np.asarray(xyz, dtype=np.float32).copy()
    center = points.mean(axis=0, keepdims=True)
    points -= center
    if rotate:
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        rotation = np.array(
            [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
            dtype=np.float32,
        )
        points[:, :2] = points[:, :2] @ rotation.T
    if jitter > 0:
        points += np.clip(rng.normal(0.0, jitter, points.shape), -3 * jitter, 3 * jitter)
    return points + center


class RailwayBlockDataset:
    """Random spatial blocks from labeled WHU-Railway3D tiles.

    A small LRU avoids repeatedly decoding the same very large PLY file inside
    each worker. Use ``num_workers: 0`` when RAM is limited.
    """

    def __init__(
        self,
        paths: Iterable[Path],
        *,
        voxel_size: float,
        block_size: float,
        max_points: int,
        feature_properties: Sequence[str],
        blocks_per_file: int = 8,
        augment: bool = False,
        seed: int = 2025,
        cache_size: int = 2,
        label_property: str | None = None,
    ):
        self.paths = list(paths)
        if not self.paths:
            raise ValueError("dataset needs at least one PLY file")
        self.voxel_size = float(voxel_size)
        self.block_size = float(block_size)
        self.max_points = int(max_points)
        self.feature_properties = tuple(feature_properties)
        self.blocks_per_file = int(blocks_per_file)
        self.augment = bool(augment)
        self.seed = int(seed)
        self.cache_size = int(cache_size)
        self.label_property = label_property
        self._cache: OrderedDict[Path, RailwayCloud] = OrderedDict()

    def __len__(self) -> int:
        return len(self.paths) * self.blocks_per_file

    def _cloud(self, path: Path) -> RailwayCloud:
        cloud = self._cache.get(path)
        if cloud is None:
            cloud = read_ply(path, require_labels=True, label_property=self.label_property)
            self._cache[path] = cloud
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        else:
            self._cache.move_to_end(path)
        return cloud

    def __getitem__(self, index: int) -> VoxelSample:
        path = self.paths[index % len(self.paths)]
        cloud = self._cloud(path)
        rng = np.random.default_rng(self.seed + index)
        indices = random_block_indices(
            cloud.xyz,
            block_size=self.block_size,
            max_points=self.max_points,
            rng=rng,
        )
        xyz = augment_xyz(cloud.xyz, rng) if self.augment else cloud.xyz
        return voxelize_arrays(
            xyz,
            cloud.attributes,
            cloud.labels,
            indices,
            path=path,
            voxel_size=self.voxel_size,
            feature_properties=self.feature_properties,
            spatial_scale=self.block_size,
        )


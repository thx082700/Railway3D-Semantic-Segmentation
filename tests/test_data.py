import numpy as np

from railway3d_seg.data import sliding_block_indices, voxelize_arrays


def test_voxel_inverse_restores_point_count(tmp_path):
    xyz = np.array([[0, 0, 0], [0.01, 0.01, 0], [1, 1, 0]], dtype=np.float32)
    labels = np.array([1, 1, 2], dtype=np.int64)
    sample = voxelize_arrays(
        xyz,
        {"intensity": np.array([10, 12, 20], dtype=np.float32)},
        labels,
        np.arange(3),
        path=tmp_path / "tile.ply",
        voxel_size=0.1,
        feature_properties=("intensity",),
        spatial_scale=10.0,
    )
    assert sample.coords.shape[0] == 2
    assert sample.features.shape == (2, 4)
    assert sample.labels[sample.inverse].shape == (3,)


def test_sliding_blocks_cover_every_point():
    axis = np.linspace(0, 100, 101)
    xyz = np.column_stack([axis, np.zeros_like(axis), np.zeros_like(axis)])
    blocks = sliding_block_indices(xyz, block_size=30.0, overlap=0.5)
    covered = np.zeros(len(xyz), dtype=bool)
    for indices in blocks:
        covered[indices] = True
    assert covered.all()


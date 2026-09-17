import io
import json
import zipfile

import numpy as np
from plyfile import PlyData, PlyElement

from railway3d_seg.visualization import (
    read_submission_archive,
    write_manifest,
    write_overall_distribution_svg,
    write_scene_composition_svg,
    write_spatial_prediction_svg,
)


def _write_archive(path):
    buffer = io.BytesIO()
    np.save(buffer, np.array([0, 1, 1, 4, 10], dtype=np.int8), allow_pickle=False)
    with zipfile.ZipFile(path, "w") as bundle:
        bundle.writestr("scene_a.npy", buffer.getvalue())
        bundle.writestr("__MACOSX/._scene_a.npy", b"metadata")


def test_archive_summary_and_nonspatial_svgs(tmp_path):
    archive = tmp_path / "submission_refined.zip"
    _write_archive(archive)
    summary = read_submission_archive(archive)

    assert summary.total_points == 5
    assert summary.scenes[0].source_dtype == "int8"
    assert summary.scenes[0].counts.tolist() == [1, 2, 0, 0, 1, 0, 0, 0, 0, 0, 1]
    assert summary.ignored_members == ("__MACOSX/._scene_a.npy",)

    manifest = write_manifest(summary, tmp_path / "manifest.json")
    distribution = write_overall_distribution_svg(summary, tmp_path / "distribution.svg")
    composition = write_scene_composition_svg(summary, tmp_path / "composition.svg")
    assert json.loads(manifest.read_text())["total_points"] == 5
    assert "Official submission label distribution" in distribution.read_text()
    assert "scene_a" in composition.read_text()


def test_spatial_svg_requires_matching_point_count(tmp_path):
    archive = tmp_path / "submission_refined.zip"
    _write_archive(archive)
    scene = read_submission_archive(archive).scenes[0]
    vertices = np.array(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.2), (2.0, 0.1, 0.3), (3.0, 0.2, 0.4), (4.0, 0.3, 0.5)],
        dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")],
    )
    ply_path = tmp_path / "scene_a.ply"
    PlyData([PlyElement.describe(vertices, "vertex")], text=False).write(ply_path)

    output = write_spatial_prediction_svg(scene, ply_path, tmp_path / "spatial.svg")
    rendered = output.read_text()
    assert "Top view (X–Y)" in rendered
    assert "Test ground truth is private" in rendered

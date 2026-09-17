import zipfile

import numpy as np
import pytest

from railway3d_seg.io import save_prediction
from railway3d_seg.submission import build_submission, validate_prediction_directory


def test_flat_uint8_submission(tmp_path):
    predictions = tmp_path / "predictions"
    save_prediction(predictions / "tile_a.npy", np.array([0, 1, 10]))
    save_prediction(predictions / "tile_b.npy", np.array([3, 4]))
    archive = tmp_path / "submission.zip"
    report = build_submission(predictions, archive)
    assert report.files == 2
    assert report.points == 5
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ["tile_a.npy", "tile_b.npy"]


def test_rejects_wrong_dtype(tmp_path):
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    np.save(predictions / "tile.npy", np.array([0, 1], dtype=np.int64))
    with pytest.raises(ValueError, match="uint8"):
        validate_prediction_directory(predictions)


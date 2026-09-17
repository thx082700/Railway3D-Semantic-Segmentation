# Recovered competition artifact

[`submission_refined.zip`](submission_refined.zip) is the exact archive downloaded from the
authenticated Codabench record for submission **353373**. It is preserved byte-for-byte so the
published repository can be audited against the official score record.

| Field | Value |
|---|---|
| Benchmark | Urban Railway PCSS |
| Submission ID | 353373 |
| Result | 79.29% mIoU / 93.14% OA |
| Prediction scenes | 8 |
| Point predictions | 147,847,797 |
| Archive size | 2,818,500 bytes |
| SHA-256 | `65026f9ebdd6faaac82da0aac2f6dec2a3c927d55613cf86d7c36828929d378b` |

This is a **prediction archive, not a model checkpoint**. It contains one label per official test
point, but it does not contain network weights or the XYZ coordinates needed to draw a spatial
point-cloud view by itself.

The recovered file records the historical submission exactly, including `int8` NPY vectors and
eight harmless `__MACOSX` metadata entries. Codabench accepted and scored that archive. New
submissions built by this repository use the cleaner documented contract: flat `uint8` vectors
with no metadata directories.

Run the audit and non-spatial visualizations with:

```bash
railway3d-visualize artifacts/submission_refined.zip
```

The WHU-Railway3D point clouds are not redistributed here. They remain governed by the dataset
provider's terms and must be requested from the official project.

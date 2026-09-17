# Competition result and provenance

## Verified historical submission

The public Codabench API identifies submission **353373** as a finished Urban Railway PCSS
submission owned by `thx0827`:

- file: `submission_refined.zip`
- submitted: 2025-08-14 13:05:35 UTC
- primary score: **79.2857% mIoU**
- overall accuracy: **93.1406%**
- listed on leaderboard: yes

The machine-readable record is preserved in
[`results/official_submission_353373.json`](../results/official_submission_353373.json). Anyone can
verify the current API response at
<https://www.codabench.org/api/submissions/353373/>.

## Reproducibility boundary

The original 2025 source tree, training log, and checkpoint were lost. The historical score is
therefore evidence of the submitted predictions, not a claim that the default configuration in
this reconstructed repository reproduces 79.29% without retraining and tuning. This release
reconstructs the full engineering path—data inspection, block sampling, sparse training,
overlap-voted inference, validation metrics, output validation, and ZIP creation—using the same
task contract.

The owner recovered the exact archive from the authenticated Codabench submission page on
2026-09-17. It is preserved at
[`artifacts/submission_refined.zip`](../artifacts/submission_refined.zip) with SHA-256
`65026f9ebdd6faaac82da0aac2f6dec2a3c927d55613cf86d7c36828929d378b`.

The archive contains 8 prediction vectors totaling 147,847,797 point labels. The historical file
used `int8` arrays and included eight `__MACOSX` metadata entries; Codabench nevertheless accepted
and scored it. The repository's current submission builder intentionally emits the stricter flat
`uint8` format documented by the benchmark.

The archive is sufficient to audit labels and create class-composition figures, but it is not a
checkpoint and does not contain XYZ coordinates. True spatial visualizations require the matching
official test PLY files, distributed separately under the provider's terms. Test-set ground truth
is not public.

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

The prediction archive itself is not committed because it could not be recovered from the public
endpoint and the test data remain governed by the dataset provider. If the owner later downloads
the original archive from the authenticated Codabench Resources page, it should be attached as a
GitHub Release asset with a checksum rather than silently presented as a newly generated output.


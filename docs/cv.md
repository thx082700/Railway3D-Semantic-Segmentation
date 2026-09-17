# CV-ready wording

Use the wording below only with a link to the public repository and the official competition
record.

**Railway Point-Cloud Semantic Segmentation — 9th National LiDAR Conference Challenge, Track 2**

- Developed a sparse 3D U-Net pipeline for 11-class semantic segmentation of large-scale mobile
  LiDAR railway scenes, including memory-bounded voxel/block processing, class-balanced
  CE + Lovasz training, overlap-voted inference, and automated Codabench packaging.
- Achieved **79.29% mIoU** and **93.14% overall accuracy** on the Urban Railway PCSS benchmark;
  per-class IoU reached 94.20% for Others, 92.64% for Overhead Lines, and 89.95% for Track Bed
  (Codabench submission 353373).
- Reconstructed and open-sourced the end-to-end training, evaluation, inference, and submission
  workflow after the original competition code was lost; preserved the verified historical score
  with explicit provenance and reproducibility limits.

Do not describe the repository as reproducing the 79.29% score from the included checkpoint:
there is no included historical checkpoint. If an award certificate specifically covers Track 2,
add the award as a separate honors line and use the exact prize name shown on that certificate.


# Dataset and submission contract

WHU-Railway3D covers about 30 km and 4.6 billion points across urban, rural, and plateau railway
scenes. Request it only through the [official dataset repository](https://github.com/WHU-USI3DV/WHU-Railway3D);
this project does not redistribute point clouds or annotations.

The recovered score in this repository belongs to the **Urban Railway** sub-benchmark. Its 11
class IDs are:

| ID | Class | ID | Class |
|---:|---|---:|---|
| 0 | rails | 6 | poles |
| 1 | track bed | 7 | vegetation |
| 2 | masts | 8 | buildings |
| 3 | support devices | 9 | ground |
| 4 | overhead lines | 10 | others |
| 5 | fences | | |

Before training, inspect a real file because PLY attribute spellings can vary by release:

```bash
railway3d-inspect data/WHU-Railway3D/Urban/Train/example.ply
```

The loader recognizes common aliases for intensity, scan angle, return number, and semantic
labels. Set `data.label_property` when the released property uses another name. Preserve the
organizer-provided train/validation/test partition; never create a random tile split that leaks
adjacent railway segments.

For Codabench, each test PLY needs one prediction vector:

```text
submission.zip
├── L5-1-M01-002.npy
├── ...
└── <same stem as every test PLY>.npy
```

Each array must be one-dimensional, `uint8`, have exactly one element per input point, and contain
only labels 0–10. The ZIP must be flat. `railway3d-submit` validates all of these conditions.


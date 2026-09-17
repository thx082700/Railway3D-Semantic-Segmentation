# Method

Large railway tiles are processed as overlapping XY blocks, voxelized at 8 cm by default, and
encoded as centered XYZ plus robustly scaled intensity. A four-level residual sparse 3D U-Net
predicts 11 semantic logits per occupied voxel.

Training optimizes:

```text
L = weighted_cross_entropy + lambda_lovasz * LovaszSoftmax
```

The class weights emphasize small infrastructure such as support devices, overhead lines, rails,
and masts. Lovasz-Softmax directly improves the IoU-oriented objective. Augmentation applies a
random yaw and bounded coordinate jitter.

At inference time, 50 m blocks overlap by 50%. Dense blocks are split along their widest axis to
bound memory. Voxel logits are mapped back to every original point, probabilities are averaged in
overlapping regions, and optional mirrored test-time augmentation adds a second vote. The final
point label is the argmax of the accumulated probabilities.

The default hyperparameters are documented starting points. Re-estimate class weights from the
official training split and tune voxel size, block size, loss weights, and inference overlap using
only the validation split.


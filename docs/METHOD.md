# PURE objective

PURE refines feed-forward 3D models using monocular depth, dynamic masks and
cross-frame matches. `SSLoss` combines three terms:

1. **Ordinal depth:** preserve the teacher's relative depth order.
2. **Local geometry:** align predicted points to teacher points inside local
   regions using scale and translation. Teacher points are `(xy * depth, depth)`.
3. **Static correspondence:** align world points matched by GIM-RoMa after
   confidence, cycle-consistency and dynamic-mask filtering.

```python
from pure import SSLoss, run_roma

roma = run_roma(gim_roma, images, valid_masks, dynamic_masks)
loss, details = SSLoss()(predictions, batch, roma)
```

`predictions` supplies `local_points` and `points`, each `[B,S,H,W,3]`.
The [cache format](DATA_FORMAT.md) defines teacher data and precomputed matches.
`run_roma` requires the upstream `batch_match` and `sample` interfaces.

The supplied training configs use AdamW (`lr=1e-5`, weight decay `0.05`, betas
`0.9/0.95`), cosine OneCycleLR, accumulation 2 and gradient clipping 1.0.
VGGT runs 12k updates; DA3 runs 5k; Pi3 supports 10k head initialization followed
by 5k post-training. Trainable layers are listed in the [training guide](TRAINING_INFERENCE_ZH.md).

The original teacher orchestration, supervised dataset mixture and complete
paper training pipeline were not recovered. These configs describe the available
cache-based implementation, not a verified reproduction of paper metrics.

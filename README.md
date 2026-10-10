# PURE

Annotation-free post-training for feed-forward 3D reconstruction.
This repository releases the training objective behind **PURE: Refining Feed-forward 3D Reconstruction with Unlabeled Post-training** (ACM Multimedia 2026).

- Project page: https://funnyguy525.github.io/PURE-Page/
- Paper: https://doi.org/10.1145/3767308.3836296

PURE fine-tunes a frozen-architecture model such as VGGT, Pi3, or Depth Anything 3 on unlabeled video. Supervision comes from a monocular geometry teacher (MoGe), dynamic masks (Grounded SAM 2), and dense matches (GIM-RoMa). Inference is unchanged from the base model.

## What is in this repository

The surviving training code is the objective itself:

| Piece | Role |
| --- | --- |
| `pure/loss.py` | Ordinal depth, local affine-invariant geometry, and static correspondence losses, plus RoMa match filtering |
| `pure/utils/alignment.py` | Robust scale-and-shift alignment used inside the geometry loss |
| `pure/utils/geometry.py` | Depth edges, focal recovery, mask-aware resize, and anchor sampling |

`SSLoss` is the original objective. The geometry helpers it imported lived in modules that were deleted with the rest of the training stack; `pure/utils` restores those call interfaces so the loss runs on its own. Dataloaders, backbone wrappers, and the distributed training loop are not part of this release.

## Install

```bash
pip install -r requirements.txt
```

## Objective

For an unlabeled clip, the base model predicts camera-frame points, world points, depth, and poses. `SSLoss` applies three terms.

1. **Ordinal depth.** Random pixel pairs must keep the teacher's depth order. The loss is invariant to the teacher's absolute scale.
2. **Local affine-invariant geometry.** Inside local balls, predicted points are aligned to pseudo points `P = d_mono * K_pre^{-1} p` with a per-ball scale and shift. This transfers surface detail without copying the teacher's global scale.
3. **Static correspondence.** GIM-RoMa matches that survive dynamic-mask, confidence, and cycle-consistency checks must land on the same 3D point. Matches are sampled uniformly rather than from the densest texture.

```python
from pure import SSLoss, run_roma

criterion = SSLoss()
roma = run_roma(gim_roma, images, static_masks, dynamic_masks)  # or precomputed
loss, details = criterion(predictions, batch, roma)
```

### Tensors

`predictions`

- `local_points`: `(B, S, H, W, 3)` camera-frame points
- `points`: `(B, S, H, W, 3)` world points used by the correspondence term

`batch`

- `depths`: `(B, S, H, W)` teacher depth
- `valid_masks`: `(B, S, H, W)` bool
- `dynamic_masks`: `(B, S, H, W)` bool, true on dynamic or sky pixels
- `distill_results["xy"]`: `(B, S, H, W, 2)` ray directions, so the pseudo points are `(xy * depth, depth)`

`run_roma` expects a module with `batch_match(img_a, img_b)` and `sample(matches, certainty, k)`, matching the GIM-RoMa interface used in training. Precomputed dictionaries with the keys read in `SSLoss.correspondence_loss` are also accepted.

## Training setup used in the paper

The numbers below are the schedule from the paper, not a launcher.

- Optimizer: AdamW, learning rate `1e-5`, weight decay `0.05`, betas `(0.9, 0.95)`
- Schedule: OneCycleLR, cosine, max learning rate `1e-5`
- 4× H800, gradient accumulation 2, gradient clip 1.0
- Views sampled from `[1, 24]`, aspect ratio in `[0.5, 2]`, about 160k–250k pixels
- VGGT: 12k iterations, intermediate global attention only, about 20 h
- Depth Anything 3: 5k iterations, same layer selection, about 8 h
- Pi3: replace the pixel-shuffle point head with a multi-level convolutional head, then 10k iterations of head initialization and 5k of joint post-training

A small amount of labeled synthetic data (OmniWorld, VKITTI, BEDLAM, PointOdyssey, Waymo) is mixed in to limit forgetting. The unlabeled real video is SpatialVID-HQ and Sekai-Real-Walking-HQ.

## Check

```bash
python examples/smoke_test.py
```

The script builds a synthetic two-view plane, runs `SSLoss`, and backpropagates. It does not download teacher models.

## Citation

```bibtex
@inproceedings{zhu2026pure,
  author    = {Zhu, Yaolong and Li, Peng and Lu, Jiahao and Zhao, Chengfeng and Li, Mengfei and Wang, Chao and Chen, Wenyue and Gu, Zekai and Ren, Chengwei and Wang, Ronggang and Liu, Yuan},
  title     = {{PURE}: Refining Feed-forward {3D} Reconstruction with Unlabeled Post-training},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  series    = {MM '26},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3767308.3836296},
  url       = {https://doi.org/10.1145/3767308.3836296}
}
```

# PURE teacher cache format, version 1

Each `.npz` stores a single clip, without a batch dimension. Loading uses
`allow_pickle=False`. `scripts/data/prepare_cache.py` supplies `images` from the
ordered RGB frame arguments and reads the remaining arrays from `--labels`.

| Array | Shape | Type and meaning |
| --- | --- | --- |
| `images` | `(S, 3, H, W)` | Float RGB in `[0, 1]`, before ImageNet normalization |
| `depths` | `(S, H, W)` | Float monocular teacher depth; positive on valid pixels |
| `teacher_xy` | `(S, H, W, 2)` | Float camera ray ratios `(X/Z, Y/Z)` from a frozen base model |
| `valid_masks` | `(S, H, W)` | Boolean usable teacher pixels |
| `dynamic_masks` | `(S, H, W)` | Boolean dynamic objects/sky; true means excluded from correspondence supervision |
| `dense_matches_forward` | `(S-1, H, W, 4)` | Float `(x_A, y_A, x_B, y_B)` sampled on the A grid |
| `dense_matches_backward` | `(S-1, H, W, 4)` | Float `(x_A, y_A, x_B, y_B)` sampled on the B grid |
| `dense_certainty_forward` | `(S-1, H, W)` | Float confidence in `[0, 1]` |
| `dense_certainty_backward` | `(S-1, H, W)` | Float confidence in `[0, 1]` |
| `matches_forward_mask` | `(S-1, H, W)` | Boolean forward sampled/filtered correspondences |
| `matches_backward_mask` | `(S-1, H, W)` | Boolean backward sampled/filtered correspondences |
| `format_version` | scalar | Integer `1`, added by the writer |
| `synthetic` | scalar | Boolean, added by the writer; true only for explicit test data |

`S >= 2`. `H` and `W` must be at least 28 and divisible by 14. Store finite
values in every float array, including invalid pixels. The validity mask
decides whether a teacher pixel contributes to supervised geometry.

Match coordinates use `[-1, 1]` with `align_corners=True`: the first/last pixel
centers have normalized coordinates -1/+1. For adjacent frames A/B, the
forward grid supplies B coordinates in channels 2:4; the backward grid
supplies A coordinates in channels 0:2. See the existing public
`run_roma`/`SSLoss.correspondence_loss` implementations for the exact contract.

Keep the cycle consistency, confidence, dynamic-mask, and sampling filtering
used by the upstream matcher. The cache validator checks shapes, dtypes,
selected-coordinate bounds, finite values, and invalid/dynamic source pixels.
It cannot prove that matches describe the right scene or that cached crops
are aligned. It does not manufacture missing matches or masks.

The public loss constructs teacher points as `(teacher_xy * depths, depths)`.
Its geometry branch uses `valid_masks`; dynamic pixels are excluded from the
correspondence branch, not automatically from all geometry supervision.

All source images and teacher outputs must share temporal order, resize,
crop, and intrinsics transformations. Do not resize RGB alone after generating
labels. Cache generation is deliberately offline so expensive teachers do not
run inside every training step.

For Python callers, `pure.training.cache.write_cache(path, arrays)` validates
and writes a clip. `to_batch` inserts the batch axis and maps `teacher_xy` to
the public objective's `gt['distill_results']['xy']` field.

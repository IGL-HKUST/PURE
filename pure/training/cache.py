"""Validated, pickle-free clip caches for the existing public SSLoss interface."""

from pathlib import Path

import numpy as np
import torch

MATCH_KEYS = (
    "dense_matches_forward",
    "dense_matches_backward",
    "dense_certainty_forward",
    "dense_certainty_backward",
    "matches_forward_mask",
    "matches_backward_mask",
)
ARRAY_KEYS = (
    "images",
    "depths",
    "teacher_xy",
    "valid_masks",
    "dynamic_masks",
) + MATCH_KEYS


def validate_cache(arrays):
    missing = set(ARRAY_KEYS) - set(arrays)
    if missing:
        raise ValueError(f"Missing cache arrays: {sorted(missing)}")
    images = arrays["images"]
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape (S, 3, H, W)")
    views, _, height, width = images.shape
    if views < 2 or min(height, width) < 28 or height % 14 or width % 14:
        raise ValueError(
            "Need >=2 views and H/W >=28 divisible by 14; resize labels and matches together"
        )
    expected = {
        "depths": (views, height, width),
        "teacher_xy": (views, height, width, 2),
        "valid_masks": (views, height, width),
        "dynamic_masks": (views, height, width),
    }
    for key in MATCH_KEYS:
        expected[key] = (
            (views - 1, height, width, 4)
            if "dense_matches" in key
            else (views - 1, height, width)
        )
    for key, shape in expected.items():
        if arrays[key].shape != shape:
            raise ValueError(f"{key}: expected {shape}, got {arrays[key].shape}")
    for key in ARRAY_KEYS:
        value = arrays[key]
        if "mask" in key:
            if value.dtype != np.bool_:
                raise ValueError(f"{key} must be boolean")
        elif (
            not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all()
        ):
            raise ValueError(f"{key} must contain finite floating-point values")
    if images.min() < 0 or images.max() > 1:
        raise ValueError("images must be RGB float values in [0, 1]")
    valid = arrays["valid_masks"]
    if not valid.any() or np.any(arrays["depths"][valid] <= 0):
        raise ValueError("Need valid pixels with positive teacher depths")
    for key in ("dense_certainty_forward", "dense_certainty_backward"):
        if arrays[key].min() < 0 or arrays[key].max() > 1:
            raise ValueError(f"{key} must be confidence values in [0, 1]")
    if not any(
        np.any(arrays[key]) for key in ("matches_forward_mask", "matches_backward_mask")
    ):
        raise ValueError("No correspondence samples: supply real filtered matches")
    for direction, point_slice in (
        ("forward", slice(None, -1)),
        ("backward", slice(1, None)),
    ):
        selected = arrays[f"matches_{direction}_mask"]
        if np.any(
            selected & (~valid[point_slice] | arrays["dynamic_masks"][point_slice])
        ):
            raise ValueError(
                f"{direction} matches include invalid or dynamic source pixels"
            )
        coords = arrays[f"dense_matches_{direction}"][selected]
        if len(coords) and np.any(np.abs(coords) > 1.0001):
            raise ValueError(
                f"{direction} selected coordinates must be normalized to [-1, 1]"
            )


def write_cache(path, arrays, *, synthetic=False):
    arrays = {key: np.asarray(arrays[key]) for key in ARRAY_KEYS}
    validate_cache(arrays)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # An open file prevents numpy from silently adding a second extension.
    with path.open("wb") as handle:
        np.savez_compressed(
            handle, **arrays, format_version=np.int64(1), synthetic=np.bool_(synthetic)
        )


def read_cache(path, *, allow_synthetic=False):
    with np.load(path, allow_pickle=False) as archive:
        if "format_version" not in archive or int(archive["format_version"]) != 1:
            raise ValueError(
                "Unsupported cache format; create it with scripts/data/prepare_cache.py"
            )
        if bool(archive["synthetic"]) and not allow_synthetic:
            raise ValueError(
                "Synthetic check data requires explicit allow_synthetic=True"
            )
        arrays = {key: archive[key].copy() for key in ARRAY_KEYS}
    validate_cache(arrays)
    return arrays


def to_batch(arrays, device="cpu"):
    tensor = {
        key: torch.from_numpy(np.ascontiguousarray(value)).unsqueeze(0).to(device)
        for key, value in arrays.items()
        if key in ARRAY_KEYS
    }
    gt = {
        "depths": tensor["depths"].float(),
        "valid_masks": tensor["valid_masks"],
        "dynamic_masks": tensor["dynamic_masks"],
        "distill_results": {"xy": tensor["teacher_xy"].float()},
    }
    roma = {
        key: tensor[key].float() if "mask" not in key else tensor[key]
        for key in MATCH_KEYS
    }
    return tensor["images"].float(), gt, roma


class ClipDataset:
    def __init__(self, root, *, allow_synthetic=False):
        self.paths = sorted(Path(root).glob("*.npz"))
        if not self.paths:
            raise ValueError(f"No .npz clip caches in {root}")
        self.allow_synthetic = allow_synthetic

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return read_cache(self.paths[index], allow_synthetic=self.allow_synthetic)

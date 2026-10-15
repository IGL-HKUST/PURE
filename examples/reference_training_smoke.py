"""CPU pipeline check with synthetic data and a tiny model, not paper results."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from pure.training.cache import ClipDataset, write_cache
from pure.training.engine import fit


def make_synthetic_clip(size=28):
    ys, xs = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    x = (xs + 0.5 - size / 2) / 100
    y = (ys + 0.5 - size / 2) / 100
    depth = (2 + 0.05 * x + 0.02 * y).astype(np.float32)
    rays = np.stack([x, y], -1).astype(np.float32)
    images = np.zeros((2, 3, size, size), dtype=np.float32)
    images[:, 0] = x
    images[:, 1] = y
    images = (images + 0.5).astype(np.float32)
    gx, gy = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    grid = np.stack([gx, gy], -1).astype(np.float32)
    arrays = {
        "images": images,
        "depths": np.stack([depth] * 2),
        "teacher_xy": np.stack([rays] * 2),
        "valid_masks": np.ones((2, size, size), bool),
        "dynamic_masks": np.zeros((2, size, size), bool),
    }
    for direction in ["forward", "backward"]:
        arrays[f"dense_matches_{direction}"] = np.concatenate([grid, grid], -1)[None]
        arrays[f"dense_certainty_{direction}"] = np.ones((1, size, size), np.float32)
        arrays[f"matches_{direction}_mask"] = np.ones((1, size, size), bool)
    return arrays


class TinyGeometry(torch.nn.Module):
    """Only a test double for exercising the real public objective and optimizer."""

    def __init__(self):
        super().__init__()
        self.depth_offset = torch.nn.Parameter(torch.tensor(0.1))
        self.ray_offset = torch.nn.Parameter(torch.tensor(0.01))

    def forward(self, images):
        xy = (images[:, :, :2] - 0.5).permute(0, 1, 3, 4, 2) + self.ray_offset
        depth = 2 + self.depth_offset + xy[..., 0] * 0.05 + xy[..., 1] * 0.02
        local = torch.cat((xy * depth[..., None], depth[..., None]), -1)
        return {"local_points": local, "points": local}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/reference-smoke")
    args = parser.parse_args()
    torch.set_num_threads(2)
    root = Path(args.output)
    write_cache(root / "cache/plane.npz", make_synthetic_clip(), synthetic=True)
    data = ClipDataset(root / "cache", allow_synthetic=True)
    config = {"steps": 2, "accumulation": 2, "lr": 1e-5, "seed": 0, "save_every": 1}
    model = TinyGeometry()
    before = model.ray_offset.detach().clone()
    path = fit(model, data, config, root / "train")
    assert not torch.equal(
        before, model.ray_offset.detach()
    ), "Optimizer did not update parameters"
    assert path.exists()
    print(
        "Synthetic cache -> public SSLoss -> backward -> optimizer -> checkpoint passed"
    )


if __name__ == "__main__":
    main()

"""Synthetic forward/backward check for the PURE objective.

Run from the repository root:

    python examples/smoke_test.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pure.loss import SSLoss


def _identity_grid(batch, length, height, width, device):
    ys = torch.linspace(-1, 1, height, device=device)
    xs = torch.linspace(-1, 1, width, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack((grid_x, grid_y), dim=-1)
    return grid.view(1, 1, height, width, 2).expand(batch, length, -1, -1, -1).contiguous()


def main():
    torch.manual_seed(0)
    device = "cpu"
    batch, views, height, width = 1, 2, 48, 48
    ys = torch.linspace(-1, 1, height)
    xs = torch.linspace(-1, 1, width)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    # A tilted plane, expressed as camera-frame points of a ~500px focal.
    focal_px = 500.0
    depth = 2.0 + 0.15 * grid_x + 0.05 * grid_y
    ray_x = (torch.arange(width) + 0.5 - width / 2) / focal_px
    ray_y = (torch.arange(height) + 0.5 - height / 2) / focal_px
    ray_y, ray_x = torch.meshgrid(ray_y, ray_x, indexing="ij")
    local = torch.stack((ray_x * depth, ray_y * depth, depth), dim=-1)
    local = local.view(1, 1, height, width, 3).expand(batch, views, -1, -1, -1).contiguous()
    local = local + 0.01 * torch.randn_like(local)
    local.requires_grad_(True)

    pred = {
        "local_points": local,
        "points": local.detach() + 0.02 * torch.randn_like(local),
    }
    pred["points"].requires_grad_(True)
    valid = torch.ones(batch, views, height, width, dtype=torch.bool)
    dynamic = torch.zeros_like(valid)
    gt = {
        "depths": depth.view(1, 1, height, width).expand(batch, views, -1, -1).contiguous(),
        "valid_masks": valid,
        "dynamic_masks": dynamic,
        "distill_results": {
            "xy": torch.stack((ray_x, ray_y), dim=-1).view(1, 1, height, width, 2).expand(batch, views, -1, -1, -1).contiguous(),
        },
    }
    grid = _identity_grid(batch, views - 1, height, width, device)
    zeros = torch.zeros_like(grid)
    matches = torch.cat((zeros, grid), dim=-1)
    certainty = valid[:, : views - 1].float()
    roma = {
        "dense_matches_forward": matches,
        "dense_matches_backward": torch.cat((grid, zeros), dim=-1),
        "dense_certainty_forward": certainty,
        "dense_certainty_backward": certainty,
        "matches_forward_mask": valid[:, : views - 1],
        "matches_backward_mask": valid[:, 1:],
    }

    loss_fn = SSLoss()
    total, details = loss_fn(pred, gt, roma)
    total.backward()
    print(f"loss={float(total):.6f}")
    for key, value in details.items():
        print(f"  {key}={float(value):.6f}")
    assert torch.isfinite(total), "loss is not finite"
    assert local.grad is not None and torch.isfinite(local.grad).all()
    print("smoke test passed")


if __name__ == "__main__":
    main()

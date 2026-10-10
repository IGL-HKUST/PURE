"""Geometry helpers required by the PURE post-training loss.

``depth_edge`` follows the usual relative-depth discontinuity test.
``recover_intrinsics``, ``mask_aware_nearest_resize`` and
``compute_anchor_sampling_weight`` implement the interfaces that
``pure.loss.SSLoss`` calls. The original utility modules were lost with
the rest of the training repository; these versions follow the camera
model and the local spherical neighborhood used in the paper.
"""
from __future__ import annotations

import math
from typing import Sequence, Tuple, Union

import torch
import torch.nn.functional as F


def depth_edge(
    depth: torch.Tensor,
    atol: float = None,
    rtol: float = None,
    kernel_size: int = 3,
    mask: torch.Tensor = None,
) -> torch.Tensor:
    """Pixels whose local depth range exceeds ``atol`` or ``rtol * depth``."""
    shape = depth.shape
    depth = depth.reshape(-1, 1, *shape[-2:])
    if mask is not None:
        mask = mask.reshape(-1, 1, *shape[-2:])
        valid = mask.bool()
        diff = (
            F.max_pool2d(torch.where(valid, depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2)
            + F.max_pool2d(torch.where(valid, -depth, -torch.inf), kernel_size, stride=1, padding=kernel_size // 2)
        )
    else:
        diff = (
            F.max_pool2d(depth, kernel_size, stride=1, padding=kernel_size // 2)
            + F.max_pool2d(-depth, kernel_size, stride=1, padding=kernel_size // 2)
        )
    edge = torch.zeros_like(depth, dtype=torch.bool)
    if atol is not None:
        edge |= diff > atol
    if rtol is not None:
        edge |= (diff / depth).nan_to_num_() > rtol
    return edge.reshape(shape)


def recover_intrinsics(points: torch.Tensor, mask: torch.Tensor = None, eps: float = 1e-6):
    """Recover a pinhole focal length from a camera-frame point map.

    ``points`` has shape ``(..., H, W, 3)`` and uses the pixel camera

        X = (u - cx) / f * Z,   Y = (v - cy) / f * Z,

    with the principal point at the image center. The second return value
    is the focal length in units of the image diagonal. That is the unit
    the local spherical loss expects:

        radius_3d = (0.5 / level) / focal_diag * Z.
    """
    *batch, height, width, _ = points.shape
    device, dtype = points.device, points.dtype
    diagonal = math.sqrt(height * height + width * width)

    ys = torch.arange(height, device=device, dtype=dtype) + 0.5
    xs = torch.arange(width, device=device, dtype=dtype) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    u = grid_x - width / 2.0
    v = grid_y - height / 2.0

    pts = points.reshape(-1, height, width, 3)
    if mask is None:
        valid = torch.ones(pts.shape[0], height, width, dtype=torch.bool, device=device)
    else:
        valid = mask.reshape(-1, height, width).bool()
    valid = valid & torch.isfinite(pts).all(dim=-1) & (pts[..., 2] > eps)

    x, y, z = pts.unbind(-1)
    weight = valid.to(dtype)
    numerator = (weight * (u * z * x + v * z * y)).sum(dim=(-2, -1))
    denominator = (weight * (x * x + y * y)).sum(dim=(-2, -1)).clamp_min(eps)
    focal_px = (numerator / denominator).abs()
    enough = weight.sum(dim=(-2, -1)) >= 16
    focal_px = torch.where(enough, focal_px, torch.ones_like(focal_px)).clamp_min(1.0)
    focal_diag = focal_px / diagonal
    return (
        focal_px.reshape(*batch),
        focal_diag.reshape(*batch),
        focal_px.reshape(*batch),
    )


def _nearest_valid_index(mask: torch.Tensor, target_h: int, target_w: int):
    """Index of the valid source pixel nearest each target-cell center.

    ``mask`` is ``(B, H, W)``. Returns ``src_index`` ``(B, target_h, target_w)``
    into a flattened ``H * W`` map, and a boolean validity mask.
    """
    batch, height, width = mask.shape
    device = mask.device
    win_h = max(1, math.ceil(height / target_h))
    win_w = max(1, math.ceil(width / target_w))

    ys = torch.linspace(0, height - 1, target_h, device=device)
    xs = torch.linspace(0, width - 1, target_w, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")

    oy = torch.arange(win_h, device=device) - (win_h // 2)
    ox = torch.arange(win_w, device=device) - (win_w // 2)
    off_y, off_x = torch.meshgrid(oy, ox, indexing="ij")
    cand_y = gy[..., None, None] + off_y
    cand_x = gx[..., None, None] + off_x
    inside = (cand_y >= 0) & (cand_y <= height - 1) & (cand_x >= 0) & (cand_x <= width - 1)
    cy = cand_y.clamp(0, height - 1).long()
    cx = cand_x.clamp(0, width - 1).long()
    pix = (cy * width + cx).reshape(-1)
    n_cand = win_h * win_w

    valid = mask.reshape(batch, height * width)[:, pix].reshape(batch, target_h, target_w, n_cand)
    valid = valid & inside.reshape(1, target_h, target_w, n_cand)
    dist = (cand_y - gy[..., None, None]).pow(2) + (cand_x - gx[..., None, None]).pow(2)
    dist = dist.reshape(1, target_h, target_w, n_cand).expand(batch, -1, -1, -1)
    dist = torch.where(valid, dist, torch.full_like(dist, 1e9))
    nearest = dist.argmin(dim=-1)
    chosen = (cy * width + cx).reshape(target_h, target_w, n_cand)
    src_index = chosen.reshape(1, target_h, target_w, n_cand).expand(batch, -1, -1, -1)
    src_index = src_index.gather(-1, nearest.unsqueeze(-1)).squeeze(-1)
    return src_index, valid.any(dim=-1)


def mask_aware_nearest_resize(
    inputs: Union[torch.Tensor, Sequence[torch.Tensor]],
    mask: torch.Tensor,
    size: Tuple[int, int],
):
    """Downsample maps by picking the nearest valid source pixel in each bin.

    ``inputs`` is a tensor or a sequence of tensors shaped ``(..., H, W, C)``.
    ``mask`` is ``(..., H, W)``. ``size`` is ``(target_height, target_width)``.
    Returns the resized inputs in the same nesting, plus the resized mask.
    """
    single = torch.is_tensor(inputs)
    tensors = [inputs] if single else list(inputs)
    target_h, target_w = int(size[0]), int(size[1])
    lead = mask.shape[:-2]
    height, width = mask.shape[-2:]
    batch = int(math.prod(lead)) if len(lead) else 1
    mask_b = mask.reshape(batch, height, width).bool()
    src_index, valid = _nearest_valid_index(mask_b, target_h, target_w)

    resized = []
    gather_index = src_index.reshape(batch, target_h * target_w)
    for tensor in tensors:
        channels = tensor.shape[-1]
        flat = tensor.reshape(batch, height * width, channels)
        sampled = torch.gather(
            flat, 1, gather_index.unsqueeze(-1).expand(-1, -1, channels)
        ).reshape(batch, target_h, target_w, channels)
        sampled = sampled * valid.unsqueeze(-1).to(sampled.dtype)
        resized.append(sampled.reshape(*lead, target_h, target_w, channels))
    valid = valid.reshape(*lead, target_h, target_w)
    if single:
        return resized[0], valid
    return tuple(resized), valid


@torch.no_grad()
def compute_anchor_sampling_weight(
    points: torch.Tensor,
    mask: torch.Tensor,
    radius_2d: int,
    radius_3d: torch.Tensor,
    num_test: int = 64,
) -> torch.Tensor:
    """Monte Carlo estimate of how well-supported a local anchor is.

    A pixel is a good patch center when random neighbors inside the image
    disk of radius ``radius_2d`` also fall inside the 3D ball of radius
    ``radius_3d``. The returned weights are zero on invalid pixels.
    """
    lead = points.shape[:-3]
    height, width = points.shape[-3:-1]
    device = points.device
    pts = points.reshape(-1, height, width, 3)
    valid = mask.reshape(-1, height, width).bool()
    radius = radius_3d.reshape(-1, height, width)
    batch = pts.shape[0]

    draw = torch.rand(num_test, device=device).sqrt() * float(radius_2d)
    theta = torch.rand(num_test, device=device) * (2 * math.pi)
    di = draw * torch.cos(theta)
    dj = draw * torch.sin(theta)

    ys = torch.arange(height, device=device, dtype=torch.float32)
    xs = torch.arange(width, device=device, dtype=torch.float32)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    pts_nchw = pts.permute(0, 3, 1, 2)
    mask_nchw = valid.float()[:, None]
    hits = []
    for k in range(num_test):
        sy = grid_y + di[k]
        sx = grid_x + dj[k]
        inside = (sy >= 0) & (sy <= height - 1) & (sx >= 0) & (sx <= width - 1)
        gx = (sx / max(width - 1, 1)) * 2 - 1
        gy = (sy / max(height - 1, 1)) * 2 - 1
        grid = torch.stack((gx, gy), dim=-1).unsqueeze(0).expand(batch, -1, -1, -1)
        sampled_pts = F.grid_sample(
            pts_nchw, grid, mode="nearest", align_corners=True, padding_mode="zeros"
        ).permute(0, 2, 3, 1)
        sampled_mask = F.grid_sample(
            mask_nchw, grid, mode="nearest", align_corners=True, padding_mode="zeros"
        )[:, 0]
        dist = (sampled_pts - pts).norm(dim=-1)
        hits.append(inside.unsqueeze(0) & (sampled_mask > 0.5) & (dist <= radius))
    weight = torch.stack(hits, dim=0).float().mean(dim=0) * valid.float()
    return weight.reshape(*lead, height, width)

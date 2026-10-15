"""Differentiable OpenCV camera geometry, with integer pixel-center coordinates."""

import torch


def homogeneous_pose(pose):
    if pose.shape[-2:] == (4, 4):
        return pose
    if pose.shape[-2:] != (3, 4):
        raise ValueError("Pose must end in (3,4) or (4,4)")
    bottom = torch.zeros_like(pose[..., :1, :])
    bottom[..., 0, 3] = 1
    return torch.cat((pose, bottom), dim=-2)


def invert_pose(pose):
    """Invert a rigid transform without detaching translation or rotation."""
    pose = homogeneous_pose(pose)
    rotation = pose[..., :3, :3].transpose(-1, -2)
    translation = -rotation @ pose[..., :3, 3:4]
    return homogeneous_pose(torch.cat((rotation, translation), dim=-1))


def depth_to_predictions(depth, intrinsics, camera_to_world, confidence=None):
    """depth (B,S,H,W), pixel K (B,S,3,3), camera-to-world (B,S,3/4,4)."""
    if depth.ndim != 4 or intrinsics.shape != depth.shape[:2] + (3, 3):
        raise ValueError("Incompatible depth/intrinsics shapes")
    depth, intrinsics = depth.float(), intrinsics.float()
    camera_to_world = homogeneous_pose(camera_to_world.float())
    height, width = depth.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=depth.device, dtype=depth.dtype),
        torch.arange(width, device=depth.device, dtype=depth.dtype),
        indexing="ij",
    )
    pixels = torch.stack((x, y, torch.ones_like(x)), dim=-1)
    # Supports skew as well as unequal focal lengths. No NumPy / detach here.
    rays = torch.einsum("bsij,hwj->bshwi", torch.linalg.inv(intrinsics), pixels)
    rays = rays / rays[..., 2:3]
    local = rays * depth[..., None]
    points = torch.einsum("bsij,bshwj->bshwi", camera_to_world[..., :3, :3], local)
    points = points + camera_to_world[..., :3, 3][..., None, None, :]
    return dict(
        depth=depth[..., None],
        local_points=local,
        points=points,
        xy=rays[..., :2],
        camera_poses=camera_to_world,
        intrinsics=intrinsics,
        conf=confidence,
    )


def validate_images(images):
    if images.ndim != 5 or images.shape[2] != 3:
        raise ValueError("Expected RGB images (B,S,3,H,W)")
    if min(images.shape[-2:]) < 28 or any(x % 14 for x in images.shape[-2:]):
        raise ValueError("H/W must be at least 28 and divisible by 14")

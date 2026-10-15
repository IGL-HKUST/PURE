"""PURE VGGT adapter: depth and camera branches of the official VGGT-1B.

Preserves all 1341 entries of the surviving checkpoint. Points are obtained
from depth and cameras, retaining gradients into both branches.
"""

import torch
from torch import nn
from pure._vendor.vggt.models.aggregator import Aggregator
from pure._vendor.vggt.heads.camera_head import CameraHead
from pure._vendor.vggt.heads.dpt_head import DPTHead
from pure._vendor.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from .geometry import depth_to_predictions, invert_pose, validate_images


class VGGT(nn.Module):
    def __init__(self, *, aggregator=None, camera_head=None, depth_head=None):
        super().__init__()
        self.aggregator = aggregator if aggregator is not None else Aggregator()
        self.camera_head = (
            camera_head if camera_head is not None else CameraHead(dim_in=2048)
        )
        self.depth_head = (
            depth_head
            if depth_head is not None
            else DPTHead(
                dim_in=2048, output_dim=2, activation="exp", conf_activation="expp1"
            )
        )

    def forward(self, images):
        validate_images(images)
        features, start = self.aggregator(images)
        with torch.autocast(device_type=images.device.type, enabled=False):
            features = [
                value.float() if value is not None else None for value in features
            ]
            pose = self.camera_head(features)[-1]
            depth, confidence = self.depth_head(
                features, images=images.float(), patch_start_idx=start
            )
            w2c, intrinsics = pose_encoding_to_extri_intri(pose, images.shape[-2:])
            return depth_to_predictions(
                depth.squeeze(-1), intrinsics, invert_pose(w2c), confidence
            )

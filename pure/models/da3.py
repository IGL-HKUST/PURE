"""PURE DA3-Giant adapter, reconstructed from the surviving da3.py.

The module names preserve all 847 entries of the surviving da3.bin.
Inputs are RGB [0,1]; normalization belongs to this adapter, not the cache.
"""

import torch
from torch import nn
from pure._vendor.da3.model.dinov2.dinov2 import DinoV2
from pure._vendor.da3.model.dualdpt import DualDPT
from pure._vendor.da3.model.cam_dec import CameraDec
from pure._vendor.da3.model.utils.transform import pose_encoding_to_extri_intri
from .geometry import depth_to_predictions, validate_images


class DA3(nn.Module):
    def __init__(self, *, backbone=None, head=None, cam_dec=None):
        super().__init__()
        self.backbone = (
            backbone
            if backbone is not None
            else DinoV2(
                name="vitg",
                out_layers=[19, 27, 33, 39],
                alt_start=13,
                qknorm_start=13,
                rope_start=13,
                cat_token=True,
            )
        )
        self.head = (
            head
            if head is not None
            else DualDPT(
                dim_in=3072,
                output_dim=2,
                features=256,
                out_channels=[256, 512, 1024, 1024],
            )
        )
        self.cam_dec = cam_dec if cam_dec is not None else CameraDec(dim_in=3072)
        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1),
            persistent=False,
        )

    def forward(self, images):
        validate_images(images)
        batch, views, _, height, width = images.shape
        features, _ = self.backbone(
            (images - self.image_mean) / self.image_std, cam_token=None
        )
        with torch.autocast(device_type=images.device.type, enabled=False):
            features = [(patch.float(), camera.float()) for patch, camera in features]
            output = self.head(features, height, width, patch_start_idx=0)
            depth = output["depth"].reshape(batch, views, height, width)
            pose = self.cam_dec(features[-1][1])
            # In the recovered DA3 forward, the decoded pose is camera-to-world.
            c2w, intrinsics = pose_encoding_to_extri_intri(pose, (height, width))
            return depth_to_predictions(
                depth, intrinsics, c2w, output.get("depth_conf")
            )

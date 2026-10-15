"""Recovered PURE-Pi3 v2, matching the surviving refined checkpoint.

The point decoder supplies the last convolutional feature level.
Weights are loaded explicitly. See THIRD_PARTY_NOTICES.md.
"""

import torch
import torch.nn as nn
from functools import partial
from .dinov2.layers import Mlp
from .geometry import homogenize_points
from .layers.pos_embed import RoPE2D, PositionGetter
from .layers.block import BlockRope
from .layers.attention import FlashAttentionRope
from .layers.transformer_head import (
    TransformerDecoder,
    LinearPts3d,
    ContextTransformerDecoder,
    Head,
)
from .layers.camera_head import CameraHead
from .dinov2.hub.backbones import dinov2_vitl14, dinov2_vitl14_reg
from torch.utils.checkpoint import checkpoint


class Pi3(nn.Module):
    def __init__(self):
        decoder_size = "large"
        pos_type = "rope100"
        use_global_points = False
        super().__init__()
        self.encoder = dinov2_vitl14_reg(pretrained=False)
        self.patch_size = 14
        del self.encoder.mask_token
        self.pos_type = pos_type if pos_type is not None else "none"
        self.rope = None
        if self.pos_type.startswith("rope"):
            if RoPE2D is None:
                raise ImportError(
                    "Cannot find cuRoPE2D, please install it following the README instructions"
                )
            freq = float(self.pos_type[len("rope") :])
            self.rope = RoPE2D(freq=freq)
            self.position_getter = PositionGetter()
        else:
            raise NotImplementedError
        if decoder_size == "small":
            dec_embed_dim = 384
            dec_num_heads = 6
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == "base":
            dec_embed_dim = 768
            dec_num_heads = 12
            mlp_ratio = 4
            dec_depth = 24
        elif decoder_size == "large":
            dec_embed_dim = 1024
            dec_num_heads = 16
            mlp_ratio = 4
            dec_depth = 36
        else:
            raise NotImplementedError
        self.decoder = nn.ModuleList(
            [
                BlockRope(
                    dim=dec_embed_dim,
                    num_heads=dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    drop_path=0.0,
                    norm_layer=partial(nn.LayerNorm, eps=1e-06),
                    act_layer=nn.GELU,
                    ffn_layer=Mlp,
                    init_values=0.01,
                    qk_norm=True,
                    attn_class=FlashAttentionRope,
                    rope=self.rope,
                )
                for _ in range(dec_depth)
            ]
        )
        self.dec_embed_dim = dec_embed_dim
        num_register_tokens = 5
        self.patch_start_idx = num_register_tokens
        self.register_token = nn.Parameter(
            torch.randn(1, 1, num_register_tokens, self.dec_embed_dim)
        )
        nn.init.normal_(self.register_token, std=1e-06)
        self.point_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=1024,
            rope=self.rope,
            use_checkpoint=True,
        )
        self.intermediate_layers = [4, 12, 18, 24, 30, 36]
        self.depth_refiner = Head(
            num_features=6,
            dim_in=1024,
            dim_out=[3],
            dim_proj=512,
            dim_upsample=[512, 256, 128, 64],
            dim_times_res_block_hidden=2,
            num_res_blocks=2,
            res_block_norm="group_norm",
            last_res_blocks=1,
            last_conv_channels=32,
            last_conv_size=1,
        )
        self.camera_decoder = TransformerDecoder(
            in_dim=2 * self.dec_embed_dim,
            dec_embed_dim=1024,
            dec_num_heads=16,
            out_dim=512,
            rope=self.rope,
            use_checkpoint=True,
        )
        self.camera_head = CameraHead(dim=512)
        self.use_global_points = use_global_points
        if use_global_points:
            self.global_points_decoder = ContextTransformerDecoder(
                in_dim=2 * self.dec_embed_dim,
                dec_embed_dim=1024,
                dec_num_heads=16,
                out_dim=1024,
                rope=self.rope,
            )
            self.global_point_head = LinearPts3d(
                patch_size=14, dec_embed_dim=1024, output_dim=3
            )
        image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("image_mean", image_mean)
        self.register_buffer("image_std", image_std)
        self.train_conf = False
        self.num_dec_blk_not_to_checkpoint = 4

    def decode(self, hidden, N, H, W):
        BN, hw, _ = hidden.shape
        B = BN // N
        final_output = []
        intermediate_features = []
        hidden = hidden.reshape(B * N, hw, -1)
        register_token = self.register_token.repeat(B, N, 1, 1).reshape(
            B * N, *self.register_token.shape[-2:]
        )
        hidden = torch.cat([register_token, hidden], dim=1)
        hw = hidden.shape[1]
        if self.pos_type.startswith("rope"):
            pos = self.position_getter(
                B * N, H // self.patch_size, W // self.patch_size, hidden.device
            )
        if self.patch_start_idx > 0:
            pos = pos + 1
            pos_special = (
                torch.zeros(B * N, self.patch_start_idx, 2)
                .to(hidden.device)
                .to(pos.dtype)
            )
            pos = torch.cat([pos_special, pos], dim=1)
        for i in range(len(self.decoder)):
            blk = self.decoder[i]
            if i % 2 == 0:
                pos = pos.reshape(B * N, hw, -1)
                hidden = hidden.reshape(B * N, hw, -1)
            else:
                pos = pos.reshape(B, N * hw, -1)
                hidden = hidden.reshape(B, N * hw, -1)
            if i >= self.num_dec_blk_not_to_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=pos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=pos)
            if i + 1 in [len(self.decoder) - 1, len(self.decoder)]:
                final_output.append(hidden.reshape(B * N, hw, -1))
            if i + 1 in self.intermediate_layers:
                intermediate_features.append(
                    hidden.reshape(B * N, hw, -1)[:, self.patch_start_idx :]
                )
        return (
            torch.cat([final_output[0], final_output[1]], dim=-1),
            pos.reshape(B * N, hw, -1),
            intermediate_features,
        )

    def forward(self, imgs):
        imgs = (imgs - self.image_mean) / self.image_std
        B, N, _, H, W = imgs.shape
        patch_h, patch_w = (H // 14, W // 14)
        imgs = imgs.reshape(B * N, _, H, W)
        hidden = self.encoder(imgs, is_training=True)
        if isinstance(hidden, dict):
            hidden = hidden["x_norm_patchtokens"]
        hidden, pos, intermediate_features = self.decode(hidden, N, H, W)
        point_hidden = self.point_decoder(hidden, xpos=pos)
        if self.train_conf:
            conf_hidden = self.conf_decoder(hidden, xpos=pos)
        camera_hidden = self.camera_decoder(hidden, xpos=pos)
        if self.use_global_points:
            context = (
                hidden.reshape(B, N, patch_h * patch_w + self.patch_start_idx, -1)[
                    :, 0:1
                ]
                .repeat(1, N, 1, 1)
                .reshape(B * N, patch_h * patch_w + self.patch_start_idx, -1)
            )
            global_point_hidden = self.global_points_decoder(
                hidden, context, xpos=pos, ypos=pos
            )
        with torch.autocast(device_type=imgs.device.type, enabled=False):
            intermediate_features = [
                intermediate_ft.float().reshape(B, N, *intermediate_ft.shape[1:])
                for intermediate_ft in intermediate_features
            ]
            last_feat_sliced = point_hidden[:, self.patch_start_idx :].float()
            intermediate_features[-1] = last_feat_sliced.reshape(
                B, N, -1, last_feat_sliced.shape[-1]
            )
            ret = self.depth_refiner(
                intermediate_features, imgs.reshape(B, N, _, H, W)
            )[0].permute(0, 1, 3, 4, 2)
            xy, z = ret.split([2, 1], dim=-1)
            z = torch.exp(z)
            local_points = torch.cat([xy * z, z], dim=-1)
            if self.train_conf:
                conf_hidden = conf_hidden.float()
                conf = self.conf_head(
                    [conf_hidden[:, self.patch_start_idx :]], (H, W)
                ).reshape(B, N, H, W, -1)
            else:
                conf = None
            camera_hidden = camera_hidden.float()
            camera_poses = self.camera_head(
                camera_hidden[:, self.patch_start_idx :], patch_h, patch_w
            ).reshape(B, N, 4, 4)
            if self.use_global_points:
                global_point_hidden = global_point_hidden.float()
                global_points = self.global_point_head(
                    [global_point_hidden[:, self.patch_start_idx :]], (H, W)
                ).reshape(B, N, H, W, -1)
            else:
                global_points = None
            points = torch.einsum(
                "bnij, bnhwj -> bnhwi", camera_poses, homogenize_points(local_points)
            )[..., :3]
        return dict(
            points=points,
            local_points=local_points,
            conf=conf,
            camera_poses=camera_poses,
            global_points=global_points,
            xy=xy,
        )

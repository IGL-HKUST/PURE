from .attention import FlashAttentionRope, FlashCrossAttentionRope
from .block import BlockRope, CrossBlockRope
from ..dinov2.layers import Mlp
import torch.nn as nn
from functools import partial
from torch.utils.checkpoint import checkpoint
import torch.nn.functional as F
from typing import *
import torch

class TransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
        need_project=True,
        use_checkpoint=False,
    ):
        super().__init__()

        self.projects = nn.Linear(in_dim, dec_embed_dim) if need_project else nn.Identity()
        self.use_checkpoint = use_checkpoint

        self.blocks = nn.ModuleList([
            BlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                drop_path=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                # attn_class=MemEffAttentionRope,
                attn_class=FlashAttentionRope,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, xpos=None):
        hidden = self.projects(hidden)
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint and self.training:
                hidden = checkpoint(blk, hidden, xpos=xpos, use_reentrant=False)
            else:
                hidden = blk(hidden, xpos=xpos)
        out = self.linear_out(hidden)
        return out

class LinearPts3d (nn.Module):
    """
    Linear head for dust3r
    Each token outputs: - 16x16 3D points (+ confidence)
    """

    def __init__(self, patch_size, dec_embed_dim, output_dim=3,):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Linear(dec_embed_dim, (output_dim)*self.patch_size**2)

    def forward(self, decout, img_shape):
        H, W = img_shape
        tokens = decout[-1]
        B, S, D = tokens.shape

        # extract 3D points
        feat = self.proj(tokens)  # B,S,D
        feat = feat.transpose(-1, -2).view(B, -1, H//self.patch_size, W//self.patch_size)
        feat = F.pixel_shuffle(feat, self.patch_size)  # B,3,H,W

        # permute + norm depth
        return feat.permute(0, 2, 3, 1)


class ContextTransformerDecoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dec_embed_dim=512,
        depth=5,
        dec_num_heads=8,
        mlp_ratio=4,
        rope=None,
    ):
        super().__init__()

        self.projects_x = nn.Linear(in_dim, dec_embed_dim)
        self.projects_y = nn.Linear(in_dim, dec_embed_dim)

        self.blocks = nn.ModuleList([
            CrossBlockRope(
                dim=dec_embed_dim,
                num_heads=dec_num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                act_layer=nn.GELU,
                ffn_layer=Mlp,
                init_values=None,
                qk_norm=False,
                # attn_class=MemEffAttentionRope,
                # cross_attn_class=MemEffCrossAttentionRope,
                attn_class=FlashAttentionRope,
                cross_attn_class=FlashCrossAttentionRope,
                rope=rope
            ) for _ in range(depth)])

        self.linear_out = nn.Linear(dec_embed_dim, out_dim)

    def forward(self, hidden, context, xpos=None, ypos=None):
        hidden = self.projects_x(hidden)
        context = self.projects_y(context)

        for i, blk in enumerate(self.blocks):
            hidden = blk(hidden, context, xpos=xpos, ypos=ypos)

        out = self.linear_out(hidden)

        return out





class ResidualConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int = None, hidden_channels: int = None, padding_mode: str = 'replicate', activation: Literal['relu', 'leaky_relu', 'silu', 'elu'] = 'relu', norm: Literal['group_norm', 'layer_norm'] = 'group_norm'):
        super(ResidualConvBlock, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        if hidden_channels is None:
            hidden_channels = in_channels

        if activation =='relu':
            activation_cls = lambda: nn.ReLU(inplace=True)
        elif activation == 'leaky_relu':
            activation_cls = lambda: nn.LeakyReLU(negative_slope=0.2, inplace=True)
        elif activation =='silu':
            activation_cls = lambda: nn.SiLU(inplace=True)
        elif activation == 'elu':
            activation_cls = lambda: nn.ELU(inplace=True)
        else:
            raise ValueError(f'Unsupported activation function: {activation}')

        self.layers = nn.Sequential(
            nn.GroupNorm(1, in_channels),
            activation_cls(),
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1, padding_mode=padding_mode),
            nn.GroupNorm(hidden_channels // 32 if norm == 'group_norm' else 1, hidden_channels),
            activation_cls(),
            nn.Conv2d(hidden_channels, out_channels, kernel_size=3, padding=1, padding_mode=padding_mode)
        )

        self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        skip = self.skip_connection(x)
        x = self.layers(x)
        x = x + skip
        return x


class Head(nn.Module):
    def __init__(
        self,
        num_features: int,
        dim_in: int,
        dim_out: List[int],
        dim_proj: int = 512,
        dim_upsample: List[int] = [256, 128, 128],
        dim_times_res_block_hidden: int = 1,
        num_res_blocks: int = 1,
        res_block_norm: Literal['group_norm', 'layer_norm'] = 'group_norm',
        last_res_blocks: int = 0,
        last_conv_channels: int = 32,
        last_conv_size: int = 1
    ):
        super().__init__()

        self.projects = nn.ModuleList([
            nn.Conv2d(in_channels=dim_in, out_channels=dim_proj, kernel_size=1, stride=1, padding=0,) for _ in range(num_features)
        ])

        self.upsample_blocks = nn.ModuleList([
            nn.Sequential(
                self._make_upsampler(in_ch + 2, out_ch),
                *(ResidualConvBlock(out_ch, out_ch, dim_times_res_block_hidden * out_ch, activation="relu", norm=res_block_norm) for _ in range(num_res_blocks))
            ) for in_ch, out_ch in zip([dim_proj] + dim_upsample[:-1], dim_upsample)
        ])

        self.output_block = nn.ModuleList([
            self._make_output_block(
                dim_upsample[-1] + 2, dim_out_, dim_times_res_block_hidden, last_res_blocks, last_conv_channels, last_conv_size, res_block_norm,
            ) for dim_out_ in dim_out
        ])


        # for output_seq in self.output_block:
        #     # 找到该 Sequential 中的最后一个 Conv2d
        #     conv_layers = [m for m in output_seq.modules() if isinstance(m, nn.Conv2d)]
        #     if len(conv_layers) > 0:
        #         last_conv = conv_layers[-1]
        #         nn.init.constant_(last_conv.weight, 0)
        #         if last_conv.bias is not None:
        #             nn.init.constant_(last_conv.bias, 0)

    # def _make_upsampler(self, in_channels: int, out_channels: int):
    #     upsampler = nn.Sequential(
    #         nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
    #         nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
    #     )
    #     upsampler[0].weight.data[:] = upsampler[0].weight.data[:, :, :1, :1]
    #     return upsampler

    def _make_upsampler(self, in_channels: int, out_channels: int):
        # 旧方法
        # upsampler = nn.Sequential(nn.ConvTranspose2d(...), nn.Conv2d(...))

        # ✨ **新方法：更平滑，伪影更少**
        upsampler = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate')
        )
        return upsampler

    def _make_output_block(self, dim_in: int, dim_out: int, dim_times_res_block_hidden: int, last_res_blocks: int, last_conv_channels: int, last_conv_size: int, res_block_norm: Literal['group_norm', 'layer_norm']):
        return nn.Sequential(
            nn.Conv2d(dim_in, last_conv_channels, kernel_size=3, stride=1, padding=1, padding_mode='replicate'),
            *(ResidualConvBlock(last_conv_channels, last_conv_channels, dim_times_res_block_hidden * last_conv_channels, activation='relu', norm=res_block_norm) for _ in range(last_res_blocks)),
            nn.ReLU(inplace=True),
            nn.Conv2d(last_conv_channels, dim_out, kernel_size=last_conv_size, stride=1, padding=last_conv_size // 2, padding_mode='replicate'),
        )

    def forward(
        self,
        hidden_states: List[torch.Tensor],
        image: torch.Tensor,
        frames_chunk_size: int = 8
    ) -> List[torch.Tensor]:
        """
        通过 Head 转发，支持按帧分块处理。

        Args:
            hidden_states (List[torch.Tensor]): 来自
                ViT 骨干的隐藏状态列表。
                每个张量的形状为 [B, S, N, C]。
            image (torch.Tensor): 输入图像（或图像序列）。
                形状为 [B, S, 3, H, W]。
            frames_chunk_size (int, optional): 每个块处理的帧数。
                如果为 None 或大于 S，则一次处理所有帧。默认为 8。

        Returns:
            List[torch.Tensor]: 输出张量列表。
                每个张量的形状为 [B, S, C_out, H, W]。
        """
        # 假设 image 形状为 [B, S, C, H, W]
        # 假设 hidden_states 是一个列表，
        #   列表中的每个张量形状为 [B, S, N, C]
        B, S, _, H, W = image.shape

        # 如果 frames_chunk_size 未指定或大于 S，则一次性处理所有帧
        if frames_chunk_size is None or frames_chunk_size >= S:
            # 1. 扁平化输入
            # [B, S, C, H, W] -> [B*S, C, H, W]
            image_flat = image.contiguous().view(B * S, -1, H, W)
            # [B, S, N, C] -> [B*S, N, C]
            hidden_states_flat = [
                h.contiguous().view(B * S, *h.shape[2:]) for h in hidden_states
            ]

            # 2. 调用实现
            output_list_flat = self._forward_impl(hidden_states_flat, image_flat)

            # 3. 恢复形状 [B*S, ...] -> [B, S, ...]
            output_list = [
                out.view(B, S, *out.shape[1:]) for out in output_list_flat
            ]
            return output_list

        # 否则，以块的形式处理帧以管理内存使用
        assert frames_chunk_size > 0

        # 准备一个列表来收集每个输出头的块
        all_outputs_collected = [[] for _ in self.output_block]

        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)
            S_chunk = frames_end_idx - frames_start_idx

            # 1. 切片并扁平化当前块
            # [B, S_chunk, C, H, W] -> [B*S_chunk, C, H, W]
            chunk_image = image[:, frames_start_idx:frames_end_idx].contiguous()
            chunk_image_flat = chunk_image.view(B * S_chunk, -1, H, W)

            # [B, S_chunk, N, C] -> [B*S_chunk, N, C]
            chunk_hidden_states_flat = []
            for h in hidden_states:
                h_chunk = h[:, frames_start_idx:frames_end_idx].contiguous()
                h_chunk_flat = h_chunk.view(B * S_chunk, *h_chunk.shape[2:])
                chunk_hidden_states_flat.append(h_chunk_flat)

            # 2. 处理块
            # chunk_output_list_flat 是一个列表，
            #   其中每个张量的形状为 [B*S_chunk, C_out, H, W]
            chunk_output_list_flat = self._forward_impl(
                chunk_hidden_states_flat, chunk_image_flat
            )

            for i, tensor_flat in enumerate(chunk_output_list_flat):
                # 恢复形状 [B, S_chunk, C_out, H, W]
                tensor_unflat = tensor_flat.view(B, S_chunk, *tensor_flat.shape[1:])
                all_outputs_collected[i].append(tensor_unflat)



        final_output_list = [
            torch.cat(tensors, dim=1) for tensors in all_outputs_collected
        ]

        return final_output_list


    def _forward_impl(
        self,
        hidden_states: List[torch.Tensor],
        image: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Head 模块的核心实现，处理扁平化的批次。

        Args:
            hidden_states (List[torch.Tensor]): 扁平化的隐藏状态列表。
                每个张量形状为 [B_flat, N, C]。
            image (torch.Tensor): 扁平化的图像张量。
                形状为 [B_flat, 3, H, W]。

        Returns:
            List[torch.Tensor]: 输出张量列表。
                每个张量形状为 [B_flat, C_out, H, W]。
        """
        # (这是你原来的 'forward' 方法，已重命名)
        # B_flat, _, img_h, img_w = image.shape
        img_h, img_w = image.shape[-2:]
        patch_h, patch_w = img_h // 14, img_w // 14

        # Process the hidden states
        # feat 形状为 [B_flat, N, C]
        # x 形状为 [B_flat, C_proj, H_patch, W_patch]
        x = torch.stack([
            proj(feat.permute(0, 2, 1).unflatten(2, (patch_h, patch_w)).contiguous())
                for proj, feat in zip(self.projects, hidden_states)
        ], dim=1).sum(dim=1)

        # Upsample stage
        # (patch_h, patch_w) -> (patch_h * 2, patch_w * 2) -> (patch_h * 4, patch_w * 4) -> (patch_h * 8, patch_w * 8)
        for i, block in enumerate(self.upsample_blocks):
            # UV coordinates is for awareness of image aspect ratio
            uv = normalized_view_plane_uv(width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device)
            # x.shape[0] 是 B_flat
            uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
            x = torch.cat([x, uv], dim=1)
            for layer in block:
                x = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)

        # (patch_h * 8, patch_w * 8) -> (img_h, img_w)
        x = F.interpolate(x, (img_h, img_w), mode="bilinear", align_corners=False)
        uv = normalized_view_plane_uv(width=x.shape[-1], height=x.shape[-2], aspect_ratio=img_w / img_h, dtype=x.dtype, device=x.device)
        uv = uv.permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], -1, -1, -1)
        x = torch.cat([x, uv], dim=1)


        if isinstance(self.output_block, nn.ModuleList):
            output = [torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False) for block in self.output_block]
        else:
            # (虽然你的 __init__ 总是创建 ModuleList,
            #  但保留这个 else 分支以匹配你的原始代码)
            output = torch.utils.checkpoint.checkpoint(self.output_block, x, use_reentrant=False)

        # output 是一个列表，其中每个张量的形状为 [B_flat, C_out, H, W]
        return output

def normalized_view_plane_uv(width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None) -> torch.Tensor:
    "UV with left-top corner as (-width / diagonal, -height / diagonal) and right-bottom corner as (width / diagonal, height / diagonal)"
    if aspect_ratio is None:
        aspect_ratio = width / height

    span_x = aspect_ratio / (1 + aspect_ratio ** 2) ** 0.5
    span_y = 1 / (1 + aspect_ratio ** 2) ** 0.5

    u = torch.linspace(-span_x * (width - 1) / width, span_x * (width - 1) / width, width, dtype=dtype, device=device)
    v = torch.linspace(-span_y * (height - 1) / height, span_y * (height - 1) / height, height, dtype=dtype, device=device)
    u, v = torch.meshgrid(u, v, indexing='xy')
    uv = torch.stack([u, v], dim=-1)
    return uv

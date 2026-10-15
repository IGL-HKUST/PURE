"""Post-training objectives for PURE.

``SSLoss`` is the annotation-free objective used to refine a feed-forward
3D reconstruction model. It combines an ordinal depth loss, a local
affine-invariant point loss, and a correspondence loss on static matches.
"""
import math
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils.alignment import align_points_scale_xyz_shift, align_points_scale_z_shift
from .utils.geometry import (
    compute_anchor_sampling_weight,
    depth_edge,
    mask_aware_nearest_resize,
    recover_intrinsics,
)

def weighted_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.mean(dim=dim, keepdim=keepdim)
    else:
        w = w.to(x.dtype)
        return (x * w).mean(dim=dim, keepdim=keepdim) / w.mean(dim=dim, keepdim=keepdim).add(eps)

def _smooth(err: torch.FloatTensor, beta: float = 0.0) -> torch.FloatTensor:
    if beta == 0:
        return err
    else:
        return torch.where(err < beta, 0.5 * err.square() / beta, err - 0.5 * beta)

def angle_diff_vec3(v1: torch.Tensor, v2: torch.Tensor, eps: float = 1e-12):
    return torch.atan2(torch.cross(v1, v2, dim=-1).norm(dim=-1) + eps, (v1 * v2).sum(dim=-1))

def harmonic_mean(x: torch.Tensor, w: torch.Tensor = None, dim: Union[int, torch.Size] = None, keepdim: bool = False, eps: float = 1e-7) -> torch.Tensor:
    if w is None:
        return x.add(eps).reciprocal().mean(dim=dim, keepdim=keepdim).reciprocal()
    else:
        w = w.to(x.dtype)
        return weighted_mean(x.add(eps).reciprocal(), w, dim=dim, keepdim=keepdim, eps=eps).add(eps).reciprocal()

@torch.no_grad()
def run_roma(gim_roma, rgbs, masks, dynamic_masks):
    """
    Runs inference with the RoMa model to compute dense matches and certainties.

    Args:
        gim_roma (nn.Module): The initialized RoMa model.
        rgbs (torch.Tensor): The input image tensor of shape (B, S, C, H, W).
        masks (torch.Tensor): Static masks for the images.
        dynamic_masks (torch.Tensor): Dynamic object masks for the images.

    Returns:
        dict: A dictionary containing the forward/backward dense matches,
              certainties, and sampled masks.
    """
    gim_roma.eval()
    batch_size, num_views, C, H, W = rgbs.shape
    if num_views == 1:
        return None
    
    gim_roma.h_resized, gim_roma.w_resized = H, W

    # Prepare batch for RoMa: (B * (S-1), C, H, W)
    imgs1 = rgbs[:, :-1, ...].reshape(-1, C, H, W)
    imgs2 = rgbs[:, 1:, ...].reshape(-1, C, H, W)

    # Run RoMa model
    dense_matches_forward, dense_matches_backward, dense_certainty_forward, dense_certainty_backward = \
        gim_roma.batch_match(imgs1, imgs2)


    def get_cycle_consistency_mask(
        dense_matches_forward,  # (B', H, W, 4) or (B', H, W, 2) for coords
        dense_matches_backward, # (B', H, W, 4) or (B', H, W, 2) for coords
        cycle_thresh_px=2.0
    ):
        """
        Computes a dense mask of cycle-consistent points.
        
        Args:
            dense_matches_forward: Matches from A -> B. Shape (B', H, W, 4)
            dense_matches_backward: Matches from B -> A. Shape (B', H, W, 4)
            cycle_thresh_px: The cycle error threshold in pixels.

        Returns:
            cycle_mask_forward (torch.Tensor): Dense boolean mask for forward matches (B', H, W)
            cycle_mask_backward (torch.Tensor): Dense boolean mask for backward matches (B', H, W)
        """
        B, H, W, _ = dense_matches_forward.shape
        device = dense_matches_forward.device
        coords_A_to_B = dense_matches_forward[..., 2:] # (B, H, W, 2)
        coords_B_to_A = dense_matches_backward[..., :2].permute(0, 3, 1, 2) # (B, 2, H, W)

        coords_cycled_back_to_A = F.grid_sample(
            coords_B_to_A,
            coords_A_to_B,
            align_corners=True,
            padding_mode="border",
            mode="bilinear"
        ).permute(0, 2, 3, 1) # (B, H, W, 2)

        y_A, x_A = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )
        coords_A = torch.stack((x_A, y_A), dim=-1).expand(B, -1, -1, -1) # (B, H, W, 2)

        cycle_error_fwd_norm = (coords_A - coords_cycled_back_to_A).norm(dim=-1) # (B, H, W)

        scale = max(H, W) / 2.0
        cycle_error_fwd_px = cycle_error_fwd_norm * scale
        
        cycle_mask_forward = (cycle_error_fwd_px < cycle_thresh_px)

        coords_B_to_A_grid = dense_matches_backward[..., :2] # (B, H, W, 2)
        coords_A_to_B_img = dense_matches_forward[..., 2:].permute(0, 3, 1, 2) # (B, 2, H, W)
        
        coords_cycled_back_to_B = F.grid_sample(
            coords_A_to_B_img,
            coords_B_to_A_grid,
            align_corners=True,
            padding_mode="border",
            mode="bilinear"
        ).permute(0, 2, 3, 1) # (B, H, W, 2)

        y_B, x_B = torch.meshgrid(
            torch.linspace(-1, 1, H, device=device),
            torch.linspace(-1, 1, W, device=device),
            indexing="ij"
        )
        coords_B = torch.stack((x_B, y_B), dim=-1).expand(B, -1, -1, -1) # (B, H, W, 2)
        
        cycle_error_bwd_norm = (coords_B - coords_cycled_back_to_B).norm(dim=-1)
        cycle_error_bwd_px = cycle_error_bwd_norm * scale
        
        cycle_mask_backward = (cycle_error_bwd_px < cycle_thresh_px)

        return cycle_mask_forward, cycle_mask_backward


    cycle_mask_forward, cycle_mask_backward = get_cycle_consistency_mask(
            dense_matches_forward, 
            dense_matches_backward,
            cycle_thresh_px=1
        ) # (B', H, W)

    certainty_mask_base = (masks & ~dynamic_masks).float() # (B, S, H, W)
    
    mask_fwd = certainty_mask_base[:, :-1].reshape(-1, H, W)
    mask_bwd = certainty_mask_base[:, 1:].reshape(-1, H, W)

    final_mask_fwd = mask_fwd * cycle_mask_forward.float()
    final_mask_bwd = mask_bwd * cycle_mask_backward.float()

    dense_certainty_forward = dense_certainty_forward.squeeze(-3) * final_mask_fwd
    dense_certainty_backward = dense_certainty_backward.squeeze(-3) * final_mask_bwd

    matches_forward_mask = gim_roma.sample(dense_matches_forward, dense_certainty_forward, 2000)
    matches_backward_mask = gim_roma.sample(dense_matches_backward, dense_certainty_backward, 2000)

    output = {
        "dense_matches_forward": dense_matches_forward.reshape(batch_size, -1, *dense_matches_forward.shape[1:]),
        "dense_matches_backward": dense_matches_backward.reshape(batch_size, -1, *dense_matches_backward.shape[1:]),
        "dense_certainty_forward": dense_certainty_forward.reshape(batch_size, -1, *dense_certainty_forward.shape[1:]),
        "dense_certainty_backward": dense_certainty_backward.reshape(batch_size, -1, *dense_certainty_backward.shape[1:]),
        "matches_forward_mask": matches_forward_mask.reshape(batch_size, -1, *matches_forward_mask.shape[1:]),
        "matches_backward_mask": matches_backward_mask.reshape(batch_size, -1, *matches_backward_mask.shape[1:]),
    }
    return output



class SSLoss(nn.Module):
    def __init__(
        self,
        base_chunk_size=32,
    ):
        super().__init__()
        self.base_chunk_size=base_chunk_size
        
    def forward(self, pred, gt, roma_results, mode='train'):
        final_loss = 0.0
        details = dict()

        gt['local_points_gt'] = torch.cat([gt['distill_results']['xy'] * gt['depths'][..., None], gt['depths'][..., None]], dim=-1)


        not_edge = ~depth_edge(gt['depths'], rtol=0.04)

        order_loss, order_loss_details = self.depth_order_loss(pred['local_points'][...,2], gt['depths'], mask=gt['valid_masks'])
        final_loss += order_loss
        details.update(order_loss_details)

        _, focal, _ = recover_intrinsics(pred['local_points'], gt['valid_masks'])


        local_mono_points_loss, local_point_loss_details = self.local_points_loss(points_pred=pred['local_points'], points_gt=gt['local_points_gt'], focal=focal.detach(), mask=gt['valid_masks'], not_edge_mask=not_edge)
        final_loss += local_mono_points_loss
        details.update(local_point_loss_details)

        correspondence_mask = ~gt['dynamic_masks'] & gt['valid_masks'] & not_edge
        corr_loss, corr_details = self.correspondence_loss(point_map=pred['points'], correspondence_results=roma_results, valid_mask=correspondence_mask)
        details.update(corr_details)
        final_loss += corr_loss 
        
        if mode == 'test':
            return final_loss, details, gt
        else:
            return final_loss, details


    
    def local_points_loss(self, points_pred, points_gt, focal, mask=None, not_edge_mask=None):

        if mask is None:
            mask = (points_pred[..., 2] > 0) & (points_gt[..., 2] > 1e-3) & (torch.isfinite(points_pred) & torch.isfinite(points_gt)).all(dim=-1)
        else:
            mask = mask.bool()

        B, S, H, W, _ = points_pred.shape

        local_loss = global_loss_32 = local_loss_2 = local_loss_4 = 0.0 * points_gt.mean()

        global_loss_32, gt_metric_scale = self.affine_invariant_global_loss(points_pred, points_gt, mask, align_resolution=32)

        if S < 2:
            local_loss += global_loss_32


        local_loss_2, local_misc = self.affine_invariant_local_loss(points_pred, points_gt, mask, focal, gt_metric_scale, level=2, align_resolution=16, num_patches=32, not_edge_mask=not_edge_mask)
        local_loss_4, local_misc = self.affine_invariant_local_loss(points_pred, points_gt, mask, focal, gt_metric_scale, level=4, align_resolution=12, num_patches=64, not_edge_mask=not_edge_mask)
        local_loss += (local_loss_2 + local_loss_4)

        return local_loss, dict(local_loss=local_loss, local_loss_2=local_loss_2, local_loss_4=local_loss_4, global_loss_32=global_loss_32)


    def correspondence_loss(self, 
                            point_map,     
                            correspondence_results,
                            valid_mask,
                            epsilon=1e-3
                            ):
        dense_matches_forward = correspondence_results['dense_matches_forward']
        dense_matches_backward = correspondence_results['dense_matches_backward']
        matches_forward_mask = correspondence_results['matches_forward_mask']
        matches_backward_mask = correspondence_results['matches_backward_mask']
        dense_certainty_forward = correspondence_results['dense_certainty_forward']
        dense_certainty_backward = correspondence_results['dense_certainty_backward']

        assert point_map.shape[1] == dense_matches_forward.shape[1] + 1
        B, L, H, W, _ = dense_matches_forward.shape

        if valid_mask is not None:
            matches_forward_mask = matches_forward_mask & valid_mask[:, :-1]
            matches_backward_mask = matches_backward_mask & valid_mask[:, 1:]

        dense_certainty_forward = dense_certainty_forward * matches_forward_mask.float()
        dense_certainty_backward = dense_certainty_backward * matches_backward_mask.float()

        point_map = point_map.permute(0, 1, 4, 2, 3)  # (B, L+1, 3, H, W)
        grid_AinB_forward = dense_matches_forward[..., 2:]
        grid_BinA_backward = dense_matches_backward[..., :2]

        sample_points_forward = F.grid_sample(
            point_map[:, 1:, ...].reshape(-1, 3, H, W),
            grid_AinB_forward.reshape(-1, H, W, 2),
            align_corners=True,
            padding_mode="border",
        ).reshape(B, L, 3, H, W).permute(0, 1, 3, 4, 2)

        sample_points_backward = F.grid_sample(
            point_map[:, :-1, ...].reshape(-1, 3, H, W),
            grid_BinA_backward.reshape(-1, H, W, 2),
            align_corners=True,
            padding_mode="border",
        ).reshape(B, L, 3, H, W).permute(0, 1, 3, 4, 2)

        point_map = point_map.permute(0, 1, 3, 4, 2)  # (B, L+1, H, W, 3)

        weight_forward = (
            2.0 / (point_map[:, :-1].norm(dim=-1) + sample_points_forward.norm(dim=-1) + 1e-6)
        ) * dense_certainty_forward
        weight_backward = (
            2.0 / (point_map[:, 1:].norm(dim=-1) + sample_points_backward.norm(dim=-1) + 1e-6)
        ) * dense_certainty_backward

        robust_error_3d_fwd = ((sample_points_forward - point_map[:, :-1]).pow(2).sum(-1) + epsilon**2).sqrt()
        robust_error_3d_bwd = ((sample_points_backward - point_map[:, 1:]).pow(2).sum(-1) + epsilon**2).sqrt()
        forward_error = robust_error_3d_fwd * weight_forward
        backward_error = robust_error_3d_bwd * weight_backward

        point_loss = (
            forward_error.sum() / (weight_forward.sum() + 1e-6)
            + backward_error.sum() / (weight_backward.sum() + 1e-6)
        )

        return point_loss, dict(corr_loss=point_loss)

    def _compute_single_tanh_loss(self, pred_depth_single, gt_depth_single, mask_single, pair_num, alpha):
        
        gt_depth = gt_depth_single[mask_single > 0]  # (N,)
        
        if gt_depth.shape[0] < 2:
            return pred_depth_single.reshape(-1)[:0].sum()

        depthmax = gt_depth.max()
        depthmin = gt_depth.min()
        interval = (depthmax - depthmin) / 10
        
        pred_depth = pred_depth_single[mask_single > 0]  # (N,)
        
        num_pixels = gt_depth.shape[0]
        index1 = torch.randperm(num_pixels, device=pred_depth.device)[:pair_num]
        index2 = torch.randperm(num_pixels, device=pred_depth.device)[:pair_num]
        
        pair_diff = torch.abs(gt_depth[index1] - gt_depth[index2])
        # Equal teacher depths carry no ordering and would divide by zero.
        threshold_msk = (pair_diff >= interval) & (pair_diff > 0)
        
        index1 = index1[threshold_msk]
        index2 = index2[threshold_msk]
        
        if index1.shape[0] == 0:
            return pred_depth_single.reshape(-1)[:0].sum()
            
        gt_diff = torch.abs(gt_depth[index1] - gt_depth[index2])
        gt_oder = torch.sign(gt_depth[index1] - gt_depth[index2])
        pred_diff = pred_depth[index1] - pred_depth[index2]

        loss = torch.mean(torch.abs(torch.tanh(alpha * pred_diff / gt_diff) - gt_oder))
        
        return loss


    def depth_order_loss(self, pred_depth, gt_depth, mask, pair_num=200000, alpha=100):

        B, S, H, W = pred_depth.shape

        pred_depth = pred_depth.reshape(B*S, H, W)
        gt_depth = gt_depth.reshape(B*S, H, W)
        mask = mask.reshape(B*S, H, W)

        BS = pred_depth.shape[0]


        losses = []
        for i in range(BS):
            loss_i = self._compute_single_tanh_loss(
                pred_depth[i], gt_depth[i], mask[i], pair_num, alpha
            )
            losses.append(loss_i)
        
        order_loss = torch.mean(torch.stack(losses)) # (B,)
        
        return order_loss, dict(depth_order_loss=order_loss)

        

    def affine_invariant_local_loss(
            self,
            pred_points: torch.Tensor,
            gt_points: torch.Tensor,
            gt_mask: torch.Tensor,
            focal: torch.Tensor,
            global_scale: torch.Tensor,
            level: Literal[1, 2, 4, 8, 16], # 扩展 level 类型
            align_resolution: int = 32,
            num_patches: int = 16,
            beta: float = 0.0,
            trunc: float = 1.0,
            sparsity_aware: bool = False,
            not_edge_mask=None,
        ):
            device, dtype = pred_points.device, pred_points.dtype
            *batch_shape, height, width, _ = pred_points.shape
            batch_size = math.prod(batch_shape)
            pred_points, gt_points, gt_mask, focal, global_scale = pred_points.reshape(-1, height, width, 3), gt_points.reshape(-1, height, width, 3), gt_mask.reshape(-1, height, width), focal.reshape(-1), global_scale.reshape(-1) if global_scale is not None else None

            radius_2d = math.ceil(0.5 / level * (height ** 2 + width ** 2) ** 0.5)
            radius_3d = 0.5 / level / focal[..., None, None] * gt_points[..., 2]

            if not_edge_mask is not None:
                not_edge_mask = not_edge_mask.reshape(-1, height, width)
                anchor_sampling_weights = compute_anchor_sampling_weight(gt_points, gt_mask&not_edge_mask, radius_2d, radius_3d, num_test=64)
            else:
                anchor_sampling_weights = compute_anchor_sampling_weight(gt_points, gt_mask, radius_2d, radius_3d, num_test=64)
            
            where_mask = torch.where(gt_mask)

            if where_mask[0].numel() == 0:
                return torch.tensor(0.0, dtype=dtype, device=device), {}

            valid_weights = anchor_sampling_weights[where_mask]
            if valid_weights.numel() == 0 or valid_weights.sum() == 0:

                random_selection = torch.randint(0, where_mask[0].numel(), (num_patches * batch_size,), device=device)

            else:
                random_selection = torch.multinomial(valid_weights, num_patches * batch_size, replacement=True)

            patch_batch_idx, patch_anchor_i, patch_anchor_j = [indices[random_selection] for indices in where_mask]

            patch_i, patch_j = torch.meshgrid(
                torch.arange(-radius_2d, radius_2d + 1, device=device),
                torch.arange(-radius_2d, radius_2d + 1, device=device),
                indexing='ij'
            )
            patch_i, patch_j = patch_i + patch_anchor_i[:, None, None], patch_j + patch_anchor_j[:, None, None]
            patch_mask = (patch_i >= 0) & (patch_i < height) & (patch_j >= 0) & (patch_j < width)
            patch_i, patch_j = patch_i.clamp(0, height - 1), patch_j.clamp(0, width - 1)

            gt_patch_anchor_points = gt_points[patch_batch_idx, patch_anchor_i, patch_anchor_j]
            gt_patch_radius_3d = 0.5 / level / focal[patch_batch_idx] * gt_patch_anchor_points[:, 2]
            gt_patch_points = gt_points[patch_batch_idx[:, None, None], patch_i, patch_j]
            pred_patch_points = pred_points[patch_batch_idx[:, None, None], patch_i, patch_j]

            gt_patch_dist = (gt_patch_points - gt_patch_anchor_points[:, None, None, :]).norm(dim=-1)
            patch_mask &= gt_mask[patch_batch_idx[:, None, None], patch_i, patch_j]
            gt_radius_mask = (gt_patch_dist <= gt_patch_radius_3d[:, None, None])
            patch_mask &= gt_radius_mask

            MINIMUM_POINTS_PER_PATCH = 32
            nonempty_indices = torch.where(patch_mask.sum(dim=(-2, -1)) >= MINIMUM_POINTS_PER_PATCH)[0]
            num_nonempty_patches = nonempty_indices.shape[0]

            if num_nonempty_patches == 0:
                return torch.tensor(0.0, dtype=dtype, device=device), {}


            patch_batch_idx = patch_batch_idx[nonempty_indices]
            patch_i = patch_i[nonempty_indices]
            patch_j = patch_j[nonempty_indices]
            patch_mask = patch_mask[nonempty_indices]
            gt_patch_points = gt_patch_points[nonempty_indices]
            gt_patch_radius_3d = gt_patch_radius_3d[nonempty_indices]
            gt_patch_anchor_points = gt_patch_anchor_points[nonempty_indices]
            pred_patch_points = pred_points[patch_batch_idx[:, None, None], patch_i, patch_j]


            # --- Align patch points ---
            (pred_patch_points_lr, gt_patch_points_lr), patch_lr_mask = mask_aware_nearest_resize(
                (pred_patch_points, gt_patch_points), mask=patch_mask, size=(align_resolution, align_resolution)
            )


            B_patches = num_nonempty_patches 
            CHUNK_SIZE = math.ceil(self.base_chunk_size * (32 / align_resolution)**2)

            local_scale_list = []
            local_shift_list = []


            for i in range(0, B_patches, CHUNK_SIZE):
                s_ = slice(i, i + CHUNK_SIZE) 

                pred_chunk_lr = pred_patch_points_lr[s_].flatten(-3, -2) # (chunk_size, N, 3) -> (chunk_size, N*3)
                gt_chunk_lr = gt_patch_points_lr[s_].flatten(-3, -2)     # (chunk_size, N, 3) -> (chunk_size, N*3)
                mask_chunk_lr = patch_lr_mask[s_].flatten(-2)            # (chunk_size, H_lr, W_lr) -> (chunk_size, N)
                radius_chunk = gt_patch_radius_3d[s_, None].add(1e-7)     # (chunk_size, 1)

                weight_chunk = mask_chunk_lr / radius_chunk

                chunk_scale, chunk_shift = align_points_scale_xyz_shift(
                    pred_chunk_lr,
                    gt_chunk_lr,
                    weight_chunk,
                    trunc=trunc
                )

                local_scale_list.append(chunk_scale)
                local_shift_list.append(chunk_shift)

                del pred_chunk_lr, gt_chunk_lr, mask_chunk_lr, radius_chunk, weight_chunk, chunk_scale, chunk_shift
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            local_scale = torch.cat(local_scale_list, dim=0) # (B_patches,)
            local_shift = torch.cat(local_shift_list, dim=0) # (B_patches, 3)

            if global_scale is not None:

                global_scale_patches = global_scale[patch_batch_idx]
                valid_global_scale_mask = (global_scale_patches > 1e-6)
                scale_differ = torch.zeros_like(local_scale)
                scale_differ[valid_global_scale_mask] = local_scale[valid_global_scale_mask] / global_scale_patches[valid_global_scale_mask]

                patch_valid = (scale_differ > 0.1) & (scale_differ < 10.0) & valid_global_scale_mask & (local_scale > 1e-6) 
            else:
                patch_valid = local_scale > 1e-6


            local_scale.masked_fill_(~patch_valid, 0.0)
            local_shift.masked_fill_(~patch_valid[:, None], 0.0)

            patch_mask &= patch_valid[:, None, None]

            # pred_patch_points: [B_patches, patch_h, patch_w, 3]
            # local_scale: [B_patches] -> [B_patches, 1, 1, 1]
            # local_shift: [B_patches, 3] -> [B_patches, 1, 1, 3]
            pred_patch_points_aligned = local_scale[:, None, None, None] * pred_patch_points + local_shift[:, None, None, :]

            gt_mean = harmonic_mean(gt_points[..., 2], gt_mask.reshape(-1, height, width), dim=(-2, -1))

            clamped_gt_depth = gt_patch_points[..., 2].clamp_min(0.1 * gt_mean[patch_batch_idx, None, None])
            patch_weight = patch_mask.float() / clamped_gt_depth

            weighted_loss = _smooth((pred_patch_points_aligned - gt_patch_points).abs() * patch_weight[..., None], beta=beta).mean(dim=-1)
            loss_sum = weighted_loss.sum(dim=(-2, -1))
            denom = patch_mask.sum(dim=(-2, -1)).clamp_min(1)
            loss_per_patch = loss_sum / denom

            if sparsity_aware:
                patch_lr_mask_nonempty = patch_lr_mask[nonempty_indices]
                sparsity = patch_mask.float().mean(dim=(-2, -1)) / patch_lr_mask_nonempty.float().mean(dim=(-2, -1)).clamp_min(1e-7)
                loss_per_patch = loss_per_patch / (sparsity + 1e-7)

            # Mean over the patches that landed in each batch element.
            # Empty elements stay zero instead of NaN.
            loss = torch.zeros(batch_size, dtype=loss_per_patch.dtype, device=loss_per_patch.device)
            counts = torch.zeros(batch_size, dtype=loss_per_patch.dtype, device=loss_per_patch.device)
            loss = loss.scatter_add(0, patch_batch_idx, loss_per_patch)
            counts = counts.scatter_add(0, patch_batch_idx, torch.ones_like(loss_per_patch))
            loss = loss / counts.clamp_min(1)
            loss = loss.reshape(batch_shape)


            misc = {}
            return loss.mean(), misc




    def affine_invariant_global_loss(
        self,
        pred_points: torch.Tensor, 
        gt_points: torch.Tensor, 
        mask: torch.Tensor, 
        align_resolution: int = 64, 
        beta: float = 0.0, 
        trunc: float = 1.0, 
        sparsity_aware: bool = False
        ):
        device = pred_points.device

        # Align
        (pred_points_lr, gt_points_lr), lr_mask = mask_aware_nearest_resize((pred_points, gt_points), mask=mask, size=(align_resolution, align_resolution))
        scale, shift = align_points_scale_z_shift(pred_points_lr.flatten(-3, -2), gt_points_lr.flatten(-3, -2), lr_mask.flatten(-2, -1) / gt_points_lr[..., 2].flatten(-2, -1).clamp_min(1e-2), trunc=trunc)
        valid = scale > 0
        scale, shift = torch.where(valid, scale, 0), torch.where(valid[..., None], shift, 0)

        pred_points = scale[..., None, None, None] * pred_points + shift[..., None, None, :]
            
        # Compute loss
        weight = (valid[..., None, None] & mask).float() / gt_points[..., 2].clamp_min(1e-5)
        weight = weight.clamp_max(10.0 * weighted_mean(weight, mask, dim=(-2, -1), keepdim=True))   # In case your data contains extremely small depth values
        err = _smooth((pred_points - gt_points).abs() * weight[..., None], beta=beta).mean(dim=-1)

        return err.mean(), scale.detach()

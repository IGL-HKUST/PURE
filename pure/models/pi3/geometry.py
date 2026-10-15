"""Homogeneous coordinates for the recovered Pi3 model."""

import torch


def homogenize_points(points):
    return torch.cat((points, torch.ones_like(points[..., :1])), dim=-1)

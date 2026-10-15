"""Shared explicit resize/crop-free input path for inference and label packing."""

import numpy as np
from PIL import Image


def load_images(paths, resize=None):
    images = []
    for path in paths:
        with Image.open(path) as im:
            im = im.convert("RGB")
            if resize:
                height, width = resize
                im = im.resize((width, height), Image.Resampling.BICUBIC)
            images.append(np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255)
    if not images or len({x.shape for x in images}) != 1:
        raise ValueError(
            "Images must have the same dimensions; use --resize H W explicitly"
        )
    result = np.stack(images)
    if min(result.shape[-2:]) < 28 or any(x % 14 for x in result.shape[-2:]):
        raise ValueError("H/W must be >=28 and divisible by 14; e.g. --resize 294 518")
    return result

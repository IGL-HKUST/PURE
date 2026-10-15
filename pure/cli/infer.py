"""Infer point maps, depth and camera poses for Pi3, DA3 or VGGT."""

import argparse
from pathlib import Path

import numpy as np
import torch
from pure.models import create_model
from pure.images import load_images
from pure.training.checkpoint import (
    model_state,
    read_checkpoint,
    normalize_model_config,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["pi3", "da3", "vggt"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--images", nargs="+", required=True, help="Paths in temporal order")
    p.add_argument("--resize", nargs=2, type=int, metavar=("H", "W"))
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    args = p.parse_args()
    images = load_images(args.images, args.resize)
    ckpt = read_checkpoint(args.checkpoint)
    config = normalize_model_config(ckpt.get("config", {}))
    if config.get("model", args.model) != args.model:
        p.error("--model disagrees with the training checkpoint")
    if (
        args.precision == "bf16"
        and args.device.startswith("cuda")
        and not torch.cuda.is_bf16_supported()
    ):
        p.error("CUDA device does not support BF16; use --precision fp32")
    model = create_model(args.model)
    model.load_state_dict(model_state(ckpt), strict=True)
    del ckpt
    model.to(args.device).eval()
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=torch.device(args.device).type,
            dtype=torch.bfloat16,
            enabled=args.precision == "bf16",
        ),
    ):
        result = model(torch.from_numpy(images).unsqueeze(0).to(args.device))
    arrays = {
        k: v[0].float().cpu().numpy()
        for k, v in result.items()
        if isinstance(v, torch.Tensor)
    }
    arrays.setdefault("depth", arrays["local_points"][..., 2:3])
    if any(not np.isfinite(x).all() for x in arrays.values()):
        raise FloatingPointError("Non-finite prediction; no output written")
    arrays.update(
        model=np.str_(args.model),
        image_paths=np.asarray(args.images),
        image_size=np.asarray(images.shape[-2:]),
    )
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as f:
        np.savez_compressed(f, **arrays)
    print(f"Predictions saved: {path}")


if __name__ == "__main__":
    main()

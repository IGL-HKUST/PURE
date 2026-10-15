"""Train PURE Pi3 / DA3 / VGGT with one GPU or torchrun DDP."""

import argparse
import json
import os
from pathlib import Path


import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from pure.models import create_model
from pure.training.cache import ClipDataset
from pure.training.checkpoint import normalize_model_config
from pure.training.loading import load_weights
from pure.training.engine import fit
from pure.training.policy import configure_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cuda")
    weights = p.add_mutually_exclusive_group(required=True)
    weights.add_argument("--checkpoint", help="Full refined checkpoint")
    weights.add_argument(
        "--base-checkpoint",
        help="Official base checkpoint; extra unused heads explicitly checked",
    )
    weights.add_argument("--resume", help="Resume training including optimizer and RNG")
    args = p.parse_args()
    config = normalize_model_config(json.loads(Path(args.config).read_text()))
    name = config.get("model", "pi3")
    if name == "pi3" and config["stage"] == "posttrain" and args.base_checkpoint:
        p.error(
            "Pi3 posttrain needs an initialized convolutional head; run head_init first"
        )
    # Validate the first cache before allocating a billion-parameter model.
    dataset = ClipDataset(args.data)
    dataset[0]
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        if args.device.startswith("cuda"):
            device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
            torch.cuda.set_device(device)
            dist.init_process_group("nccl")
        else:
            device = args.device
            dist.init_process_group("gloo")
    else:
        device = args.device
    try:
        torch.manual_seed(config["seed"])
        model = create_model(name)
        if not args.resume:
            report = load_weights(
                model,
                args.base_checkpoint or args.checkpoint,
                name,
                base=bool(args.base_checkpoint),
            )
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(json.dumps(report), flush=True)
        names = configure_model(model, config)
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                json.dumps(
                    {
                        "model": name,
                        "trainable_parameters": sum(
                            p.numel() for p in model.parameters() if p.requires_grad
                        ),
                        "trainable_tensor_count": len(names),
                    }
                ),
                flush=True,
            )
        model.to(device)
        if world > 1:
            model = DistributedDataParallel(
                model,
                device_ids=(
                    [torch.device(device).index] if device.startswith("cuda") else None
                ),
                find_unused_parameters=True,
            )
        fit(model, dataset, config, args.output, device=device, resume=args.resume)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

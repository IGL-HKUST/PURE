"""Cache-based training with accumulation, BF16, DDP, and exact sample resume."""

import json
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from pure import SSLoss
from .cache import to_batch
from .checkpoint import read_checkpoint, normalize_model_config


def configure_trainable(
    model, stage, *, decoder_start=11, decoder_end=36, train_head=False
):
    if stage not in {"head_init", "posttrain"}:
        raise ValueError("stage must be head_init or posttrain")
    model.requires_grad_(False)
    if stage == "head_init" or train_head:
        model.depth_refiner.requires_grad_(True)
    if stage == "posttrain":
        if not (0 <= decoder_start < decoder_end <= len(model.decoder)):
            raise ValueError("Invalid decoder interval")
        indices = [i for i in range(decoder_start, decoder_end) if i % 2 == 1]
        if not indices:
            raise ValueError("No global attention blocks selected")
        for index in indices:
            model.decoder[index].requires_grad_(True)
    names = [name for name, param in model.named_parameters() if param.requires_grad]
    if not names:
        raise ValueError("No trainable parameters")
    return names


def optimizer_step(
    model, loss_fn, batches, optimizer, *, device, clip_grad=1.0, precision="fp32"
):
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for index, arrays in enumerate(batches):
        images, gt, roma = to_batch(arrays, device)
        sync = (
            model.no_sync()
            if isinstance(model, DistributedDataParallel) and index + 1 < len(batches)
            else nullcontext()
        )
        with sync:
            with torch.autocast(
                device_type=torch.device(device).type,
                dtype=torch.bfloat16,
                enabled=precision == "bf16",
            ):
                prediction = model(images)
            # Geometry and robust solvers run in FP32, outside autocast.
            loss, _ = loss_fn(prediction, gt, roma)
            finite = torch.isfinite(loss).to(dtype=torch.int32)
            if dist.is_initialized():
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
            if not finite.item():
                raise FloatingPointError(
                    "Non-finite loss; optimizer step was not applied"
                )
            (loss / len(batches)).backward()
        losses.append(loss.detach().item())
    parameters = [p for p in model.parameters() if p.requires_grad]
    norm = torch.nn.utils.clip_grad_norm_(
        parameters, clip_grad, error_if_nonfinite=True
    )
    if not any(p.grad is not None for p in parameters):
        raise RuntimeError("No gradients reached trainable parameters")
    optimizer.step()
    return {"loss": sum(losses) / len(losses), "grad_norm": norm.item()}


class SampleStream:
    def __init__(self, length, seed):
        self.length = length
        self.generator = torch.Generator().manual_seed(seed)
        self.order = []
        self.offset = 0

    def next(self):
        if self.offset == len(self.order):
            self.order = torch.randperm(self.length, generator=self.generator).tolist()
            self.offset = 0
        index = self.order[self.offset]
        self.offset += 1
        return index

    def state_dict(self):
        return {
            "length": self.length,
            "order": self.order,
            "offset": self.offset,
            "generator": self.generator.get_state(),
        }

    def load_state_dict(self, state):
        if state["length"] != self.length or sorted(state["order"]) != list(
            range(self.length)
        ):
            raise ValueError("Resume dataset length or sample order differs")
        self.order, self.offset = state["order"], state["offset"]
        self.generator.set_state(state["generator"])


def _prepare_output(output, resume):
    # Only rank zero touches the directory; every rank receives the same failure.
    status = [None]
    if not dist.is_initialized() or dist.get_rank() == 0:
        try:
            output.mkdir(parents=True, exist_ok=True)
            continuing_here = resume is not None and Path(resume).resolve() == (
                output / "latest.pt"
            ).resolve()
            if any(output.iterdir()) and not continuing_here:
                raise ValueError(
                    f"Output directory is not empty: {output}. Use an empty directory "
                    "or resume its latest.pt checkpoint."
                )
        except (OSError, ValueError) as error:
            status[0] = str(error)
    if dist.is_initialized():
        dist.broadcast_object_list(status, src=0)
    if status[0] is not None:
        raise ValueError(status[0])


def fit(model, dataset, config, output, *, device="cpu", resume=None):
    config = normalize_model_config(config)
    if config["steps"] < 1 or config["accumulation"] < 1 or config["save_every"] < 1:
        raise ValueError("steps, accumulation, save_every must be positive")
    distributed = dist.is_initialized()
    rank = dist.get_rank() if distributed else 0
    world_size = dist.get_world_size() if distributed else 1
    raw_model = model.module if isinstance(model, DistributedDataParallel) else model
    precision = config.get("precision", "fp32")
    if precision not in {"fp32", "bf16"}:
        raise ValueError("precision must be fp32 or bf16")
    if (
        precision == "bf16"
        and torch.device(device).type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("This CUDA device does not support BF16; use precision=fp32")
    output = Path(output)
    _prepare_output(output, resume)
    torch.manual_seed(config["seed"] + rank)
    model.to(device).train()
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["lr"],
        weight_decay=0.05,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["lr"],
        total_steps=config["steps"],
        anneal_strategy="cos",
    )
    stream = SampleStream(len(dataset), config["seed"] + rank)
    loss_fn = SSLoss()
    start = 0
    manifest = [str(p.resolve()) for p in dataset.paths]
    if resume:
        previous = read_checkpoint(resume)
        if (
            normalize_model_config(previous["config"]) != config
            or previous["dataset"] != manifest
        ):
            raise ValueError(
                "Resume requires the same config and ordered dataset paths"
            )
        if previous.get("world_size", 1) != world_size:
            raise ValueError("Resume requires the same world size")
        start = previous["step"]
        if type(start) is not int or not 0 <= start <= config["steps"]:
            raise ValueError("Resume step must be between zero and total steps")
        raw_model.load_state_dict(previous["model"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        scheduler.load_state_dict(previous["scheduler"])
        rank_state = (
            previous["rank_states"][rank] if "rank_states" in previous else previous
        )
        stream.load_state_dict(rank_state["stream"])
        torch.set_rng_state(rank_state["rng"])
        if torch.device(device).type == "cuda":
            torch.cuda.set_rng_state_all(rank_state["cuda_rng"])
        del previous
        if start == config["steps"]:
            if rank == 0:
                print(f"Training already completed; restored {resume}", flush=True)
            return Path(resume)
    log_path = output / "metrics.jsonl"
    for step in range(start, config["steps"]):
        batches = [dataset[stream.next()] for _ in range(config["accumulation"])]
        metrics = optimizer_step(
            model,
            loss_fn,
            batches,
            optimizer,
            device=device,
            precision=precision,
            clip_grad=config.get("clip_grad", 1.0),
        )
        if distributed:
            mean_loss = torch.tensor(metrics["loss"], device=device)
            dist.all_reduce(mean_loss)
            metrics["loss"] = (mean_loss / world_size).item()
        metrics.update(step=step + 1, lr=optimizer.param_groups[0]["lr"])
        scheduler.step()
        if rank == 0:
            print(json.dumps(metrics), flush=True)
            with log_path.open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")
        if (step + 1) % config["save_every"] == 0 or step + 1 == config["steps"]:
            local_state = {
                "stream": stream.state_dict(),
                "rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state_all()
                    if torch.device(device).type == "cuda"
                    else []
                ),
            }
            rank_states = [None] * world_size
            if distributed:
                dist.all_gather_object(rank_states, local_state)
            else:
                rank_states = [local_state]
            if rank != 0:
                continue
            state = {
                "world_size": world_size,
                "rank_states": rank_states,
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                # PyTorch 2.2 includes a callable annealing function in state_dict;
                # the constructor restores that method, so serialize its data only.
                "scheduler": {
                    k: v for k, v in scheduler.state_dict().items() if not callable(v)
                },
                "stream": stream.state_dict(),
                "step": step + 1,
                "config": config,
                "dataset": manifest,
                "rng": torch.get_rng_state(),
                "cuda_rng": (
                    torch.cuda.get_rng_state_all()
                    if torch.device(device).type == "cuda"
                    else []
                ),
            }
            temporary = output / "latest.tmp"
            torch.save(state, temporary)
            temporary.replace(output / "latest.pt")
    if distributed:
        dist.barrier()
    return output / "latest.pt"

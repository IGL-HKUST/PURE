"""Explicit fine-tuning policies; VGGT selection is a reference default."""

from .engine import configure_trainable


def configure_model(model, config):
    name = config.get("model", "pi3")
    if name == "pi3":
        return configure_trainable(
            model,
            config["stage"],
            decoder_start=config.get("decoder_start", 11),
            decoder_end=config.get("decoder_end", 36),
            train_head=config.get("train_head", False),
        )
    if config["stage"] not in {"head_init", "posttrain"}:
        raise ValueError("stage must be head_init or posttrain")
    model.requires_grad_(False)
    head = model.head if name == "da3" else model.depth_head
    if config["stage"] == "head_init" or config.get("train_head", False):
        head.requires_grad_(True)
        if name == "da3":
            # DA3's auxiliary ray branch is retained for checkpoint compatibility,
            # but PURE uses the camera decoder's rays and does not supervise it.
            for key, param in head.named_parameters():
                if "_aux" in key:
                    param.requires_grad_(False)
    if config["stage"] == "posttrain":
        blocks = (
            model.backbone.pretrained.blocks
            if name == "da3"
            else model.aggregator.global_blocks
        )
        indices = config["train_blocks"]
        if (
            not indices
            or len(set(indices)) != len(indices)
            or any(not isinstance(i, int) or i < 0 or i >= len(blocks) for i in indices)
        ):
            raise ValueError("train_blocks must contain distinct valid block indices")
        if name == "da3" and any(i < 13 or i % 2 == 0 for i in indices):
            raise ValueError("DA3 global attention uses odd blocks at indices >=13")
        for i in indices:
            blocks[i].requires_grad_(True)
    if config.get("train_camera", False):
        (model.cam_dec if name == "da3" else model.camera_head).requires_grad_(True)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    if not names:
        raise ValueError("No trainable parameters")
    return names

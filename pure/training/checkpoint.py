"""Explicit loading with strict architecture coverage; never download weights."""

from pathlib import Path

import torch


def read_checkpoint(path):
    if Path(path).suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    return torch.load(path, map_location="cpu", weights_only=True)


def model_state(checkpoint):
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state, dict) or not all(
        isinstance(v, torch.Tensor) for v in state.values()
    ):
        raise ValueError(
            "Expected a tensor state_dict, or a checkpoint containing model/state_dict"
        )
    result = {}
    for key, value in state.items():
        for prefix in ("module.", "_orig_mod."):
            if key.startswith(prefix):
                key = key[len(prefix) :]
        if key in result:
            raise ValueError(f"Duplicate normalized checkpoint key: {key}")
        result[key] = value
    return result


def load_model(model, path, *, base=False):
    state = model_state(read_checkpoint(path))
    if not base:
        model.load_state_dict(state, strict=True)
        return {"missing": [], "unused": []}
    expected = model.state_dict()
    missing = sorted(key for key in expected if key not in state)
    unused = sorted(key for key in state if key not in expected)
    allowed_missing = [key for key in missing if key.startswith("depth_refiner.")]
    allowed_unused = [
        key
        for key in unused
        if key.startswith(("point_head.", "conf_head.", "conf_decoder."))
    ]
    if missing != allowed_missing or unused != allowed_unused:
        raise ValueError(
            f"Base Pi3 checkpoint does not cover the recovered architecture: missing={missing[:8]}, unused={unused[:8]}"
        )
    if not allowed_missing:
        raise ValueError(
            "Use checkpoint loading for a recovered PURE model; base loading expects a new depth_refiner"
        )
    # load_state_dict also rejects every shape mismatch, even in strict=False mode.
    model.load_state_dict(state, strict=False)
    return {"missing": missing, "unused": unused}


def normalize_model_config(config):
    """Accept old v2 checkpoint metadata, while rejecting retired Pi3 layouts."""
    config = dict(config)
    if config.get("model", "pi3") == "pi3":
        legacy_variant = config.pop("variant", "v2")
        if legacy_variant != "v2":
            raise ValueError(
                "This release supports only Pi3 v2; use a compatible checkpoint/config"
            )
    return config

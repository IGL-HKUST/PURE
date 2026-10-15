"""RGB semantic sky masks shared across backbones; no depth threshold."""

from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

REPO = "nvidia/segformer-b2-finetuned-ade-512-512"
REVISION = "de01bae28967510f9ddd496c60a969357195400c"
WEIGHT_SHA = "187ca07bea003a5717c63d04ea90b07f33cd033c0ebf44b4b89fce5070d6c8f3"


class SkySegmenter:
    def __init__(self, *, cache_dir=None, local_directory=None, device="cuda"):
        from transformers import (
            SegformerConfig,
            SegformerForSemanticSegmentation,
            SegformerImageProcessor,
        )
        from .hub import sha256

        names = ["config.json", "preprocessor_config.json", "pytorch_model.bin"]
        if local_directory:
            files = {name: Path(local_directory) / name for name in names}
        else:
            from huggingface_hub import hf_hub_download

            files = {
                name: Path(
                    hf_hub_download(REPO, name, revision=REVISION, cache_dir=cache_dir)
                )
                for name in names
            }
        if sha256(files["pytorch_model.bin"]) != WEIGHT_SHA:
            raise ValueError("Sky checkpoint SHA256 mismatch")
        config = SegformerConfig.from_json_file(str(files["config.json"]))
        self.sky_id = next(int(k) for k, v in config.id2label.items() if v == "sky")
        self.model = SegformerForSemanticSegmentation(config)
        self.model.load_state_dict(
            torch.load(
                files["pytorch_model.bin"], map_location="cpu", weights_only=True
            ),
            strict=True,
        )
        self.model.to(device).eval()
        self.device = device
        self.processor = SegformerImageProcessor.from_json_file(
            str(files["preprocessor_config.json"])
        )

    @torch.inference_mode()
    def __call__(self, images, batch_size=4):
        masks = []
        for start in range(0, len(images), batch_size):
            rgb = (
                (images[start : start + batch_size].numpy().transpose(0, 2, 3, 1) * 255)
                .round()
                .clip(0, 255)
                .astype(np.uint8)
            )
            inputs = self.processor(
                images=[Image.fromarray(x) for x in rgb], return_tensors="pt"
            ).pixel_values.to(self.device)
            # Segmenter stays FP32; the geometry model's autocast never leaks here.
            logits = self.model(pixel_values=inputs).logits
            labels = F.interpolate(
                logits, size=images.shape[-2:], mode="bilinear", align_corners=False
            ).argmax(1)
            masks.append((labels == self.sky_id).cpu().numpy())
        return np.concatenate(masks)

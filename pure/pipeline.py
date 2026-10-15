"""One reconstruction path shared by CLI and Gradio."""

import gc
import time
import numpy as np
import torch
from .export import export_reconstruction
from .sky import SkySegmenter, REPO as SKY_REPO, REVISION as SKY_REVISION


class ReconstructionPipeline:
    def __init__(
        self, model, *, device="cuda", sky_model_directory=None, cache_dir=None
    ):
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for the selected device")
        self.model = model.to(self.device).eval()
        self.sky_model_directory = sky_model_directory
        self.cache_dir = cache_dir

    def __call__(
        self,
        images,
        metadata,
        output,
        *,
        remove_sky=True,
        precision="auto",
        save_raw=False,
    ):
        if images.ndim != 4 or images.shape[1] != 3 or len(images) < 2:
            raise ValueError("Expected at least two RGB frames [V,3,H,W]")
        if not torch.isfinite(images).all() or images.min() < 0 or images.max() > 1:
            raise ValueError("Images must be finite RGB values in [0,1]")
        if precision not in ("auto", "fp32", "bf16"):
            raise ValueError("precision must be auto, fp32, or bf16")
        use_bf16 = precision == "bf16" or (
            precision == "auto"
            and self.device.type == "cuda"
            and torch.cuda.is_bf16_supported()
        )
        if (
            use_bf16
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise ValueError("This GPU does not support BF16; select fp32")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
        start = time.monotonic()
        with (
            torch.inference_mode(),
            torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=use_bf16),
        ):
            pred = self.model(images.unsqueeze(0).to(self.device))
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        inference_seconds = time.monotonic() - start
        arrays = {
            key: value[0].float().cpu().numpy()
            for key, value in pred.items()
            if isinstance(value, torch.Tensor)
        }
        del pred
        if remove_sky:
            segmenter = SkySegmenter(
                local_directory=self.sky_model_directory,
                cache_dir=self.cache_dir,
                device=str(self.device),
            )
            sky = segmenter(images.cpu())
            del segmenter
            gc.collect()
        else:
            sky = np.zeros((len(images), *images.shape[-2:]), dtype=bool)
        info = {
            **metadata,
            "model": self.model.config.backbone,
            "remove_sky": bool(remove_sky),
            "precision": "bf16" if use_bf16 else "fp32",
            "inference_seconds": inference_seconds,
            "sky_model": SKY_REPO if remove_sky else None,
            "sky_revision": SKY_REVISION if remove_sky else None,
            "peak_gpu_allocated_gib": (
                torch.cuda.max_memory_allocated(self.device) / 2**30
                if self.device.type == "cuda"
                else None
            ),
        }
        result = export_reconstruction(
            arrays, images.cpu().numpy(), sky, output, metadata=info, save_raw=save_raw
        )
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return result

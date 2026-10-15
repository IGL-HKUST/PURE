"""Shared RGB preprocessing for the Python API, CLI and Space."""

from dataclasses import dataclass
from pathlib import Path
import json
import re
import numpy as np
from PIL import Image, ImageOps
import torch


def natural_key(path):
    return [
        int(x) if x.isdigit() else x.lower()
        for x in re.split(r"(\d+)", Path(path).name)
    ]


@dataclass
class PUREImageProcessor:
    height: int = 294
    width: int = 518

    def __post_init__(self):
        if any(
            type(x) is not int or x < 28 or x % 14 for x in (self.height, self.width)
        ):
            raise ValueError("Image dimensions must be >=28 and divisible by 14")

    def _frame(self, image):
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = ImageOps.fit(
            image, (self.width, self.height), method=Image.Resampling.BICUBIC
        )
        return np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255

    def images(self, paths, *, max_frames=48, return_metadata=False):
        if max_frames < 2:
            raise ValueError("At least two frames are required")
        paths = list(paths)[:max_frames]
        if len(paths) < 2:
            raise ValueError("Provide at least two images in temporal order")
        frames, records = [], []
        for i, path in enumerate(paths):
            with Image.open(path) as image:
                records.append(
                    {
                        "input": Path(path).name,
                        "frame_index": i,
                        "original_size": list(image.size),
                        "timestamp_seconds": None,
                    }
                )
                frames.append(self._frame(image))
        tensor = torch.from_numpy(np.stack(frames))
        metadata = {
            "sampling": "provided image order",
            "frames": records,
            "frame_count": len(records),
            "processed_hw": [self.height, self.width],
            "preview_fps": 10.0,
        }
        return (tensor, metadata) if return_metadata else tensor

    def video(self, path, *, max_frames=48, start_frame=0, return_metadata=False):
        import av

        if not 0 <= start_frame <= 10000 or not 2 <= max_frames <= 512:
            raise ValueError("Invalid start frame or frame count")
        frames, records = [], []
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            rate = float(stream.average_rate) if stream.average_rate else 25.0
            for i, frame in enumerate(container.decode(video=0)):
                if i < start_frame:
                    continue
                image = frame.to_image()
                frames.append(self._frame(image))
                records.append(
                    {
                        "input": Path(path).name,
                        "frame_index": i,
                        "timestamp_seconds": (
                            float(frame.time) if frame.time is not None else None
                        ),
                        "original_size": list(image.size),
                    }
                )
                if len(frames) == max_frames:
                    break
        if len(frames) < 2:
            raise ValueError(
                "Fewer than two frames remain after the selected start frame"
            )
        timestamps = [r["timestamp_seconds"] for r in records]
        if all(t is not None for t in timestamps):
            steps = np.diff(timestamps)
            if (steps > 0).all():
                rate = float(1 / np.median(steps))
        metadata = {
            "sampling": "consecutive original frames, stride=1",
            "requested_frames": max_frames,
            "frame_count": len(records),
            "frames": records,
            "processed_hw": [self.height, self.width],
            "preview_fps": rate,
        }
        tensor = torch.from_numpy(np.stack(frames))
        return (tensor, metadata) if return_metadata else tensor

    @classmethod
    def from_pretrained(
        cls,
        path,
        *,
        subfolder="",
        revision=None,
        cache_dir=None,
        token=None,
        local_files_only=False,
    ):
        from .hub import _subfolder

        subfolder = _subfolder(subfolder)
        root = Path(path)
        if root.is_dir():
            config_path = root / subfolder / "preprocessor_config.json"
        else:
            from huggingface_hub import hf_hub_download

            config_path = Path(
                hf_hub_download(
                    str(path),
                    "preprocessor_config.json",
                    subfolder=subfolder or None,
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                )
            )
        config = json.loads(config_path.read_text())
        if config.get("resize_mode") != "preserve-aspect-center-crop" or config.get(
            "input_range"
        ) != [0, 1]:
            raise ValueError("Unsupported image processor configuration")
        return cls(*config["size"])

"""Hugging Face model directories for the three supported PURE backbones.

The installed PURE package supplies code; Hub repositories supply JSON and
Safetensors only. Checkpoint tensor names are preserved without a wrapper prefix.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import hashlib
import json
import shutil
import tempfile

import torch
from torch import nn
from .models import create_model
from .training.checkpoint import model_state, read_checkpoint, normalize_model_config

RESOURCE_ROOT = Path(__file__).parent / "resources"
BACKBONES = ("pi3", "da3", "vggt")


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class PUREConfig:
    backbone: str = "pi3"
    model_type: str = "pure"
    format_version: int = 1
    architecture_version: str = "recovered-v2"
    image_size: tuple = (294, 518)
    patch_size: int = 14

    def __post_init__(self):
        if self.backbone not in BACKBONES:
            raise ValueError(f"Unsupported backbone: {self.backbone}")
        if self.model_type != "pure" or self.format_version != 1:
            raise ValueError("Unsupported PURE model format")
        if self.architecture_version != "recovered-v2" or self.patch_size != 14:
            raise ValueError("Unsupported architecture version or patch size")
        if len(self.image_size) != 2 or any(
            type(x) is not int or x < 28 or x % 14 for x in self.image_size
        ):
            raise ValueError("image_size must be [height, width], divisible by 14")
        object.__setattr__(self, "image_size", tuple(self.image_size))

    def to_dict(self):
        data = asdict(self)
        data["image_size"] = list(self.image_size)
        return data


def _subfolder(value):
    if value and (
        Path(value).is_absolute() or ".." in Path(value).parts or "\\" in value
    ):
        raise ValueError("subfolder must be a relative Hub path")
    return value.strip("/")


def read_safetensors(directory):
    from safetensors.torch import load_file

    directory = Path(directory)
    index = directory / "model.safetensors.index.json"
    if not index.exists():
        return load_file(str(directory / "model.safetensors"), device="cpu")
    mapping = json.loads(index.read_text())["weight_map"]
    if not mapping:
        raise ValueError("Empty Safetensors shard index")
    state = {}
    for filename in sorted(set(mapping.values())):
        if Path(filename).name != filename or not filename.endswith(".safetensors"):
            raise ValueError("Invalid shard filename")
        shard = load_file(str(directory / filename), device="cpu")
        if set(state) & set(shard):
            raise ValueError("Duplicate tensor keys across shards")
        if any(mapping.get(key) != filename for key in shard):
            raise ValueError("Shard contents disagree with index")
        state.update(shard)
    if set(state) != set(mapping):
        raise ValueError("Incomplete Safetensors checkpoint")
    return state


class PURE(nn.Module):
    """Unified differentiable model with a JSON/Safetensors Hub interface."""

    def __init__(self, config=None):
        super().__init__()
        self.config = (
            config if isinstance(config, PUREConfig) else PUREConfig(**(config or {}))
        )
        self.backbone = create_model(self.config.backbone)

    def forward(self, images):
        result = self.backbone(images)
        if "depth" not in result:
            result["depth"] = result["local_points"][..., 2:3]
        return result

    @classmethod
    def from_checkpoint(cls, checkpoint, *, backbone, expected_sha256=None):
        """Convert a local legacy checkpoint; never perform a non-strict load."""
        if expected_sha256 and sha256(checkpoint) != expected_sha256:
            raise ValueError("Legacy checkpoint SHA256 mismatch")
        payload = read_checkpoint(checkpoint)
        metadata = normalize_model_config(payload.get("config", {}))
        if metadata.get("model", backbone) != backbone:
            raise ValueError("Checkpoint model metadata disagrees with backbone")
        model = cls(PUREConfig(backbone=backbone))
        model.backbone.load_state_dict(model_state(payload), strict=True, assign=True)
        return model.eval()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path,
        *,
        subfolder="",
        revision=None,
        cache_dir=None,
        token=None,
        local_files_only=False,
        device="cpu",
    ):
        subfolder = _subfolder(subfolder)
        root = Path(pretrained_model_name_or_path).expanduser()
        if not root.is_dir():
            from huggingface_hub import snapshot_download

            prefix = f"{subfolder}/" if subfolder else ""
            root = Path(
                snapshot_download(
                    str(pretrained_model_name_or_path),
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    local_files_only=local_files_only,
                    allow_patterns=[
                        prefix + "config.json",
                        prefix + "model*.safetensors",
                        prefix + "model.safetensors.index.json",
                    ],
                )
            )
        directory = root / subfolder
        config = PUREConfig(**json.loads((directory / "config.json").read_text()))
        model = cls(config)
        model.backbone.load_state_dict(
            read_safetensors(directory), strict=True, assign=True
        )
        return model.to(device).eval()

    @classmethod
    def from_legacy_hub(
        cls, backbone, *, cache_dir=None, local_files_only=False, device="cpu"
    ):
        """Explicit compatibility path for the original, pinned public .bin files."""
        from huggingface_hub import hf_hub_download

        if backbone not in BACKBONES:
            raise ValueError("Unknown backbone")
        manifest = json.loads((RESOURCE_ROOT / "legacy_weights.json").read_text())
        info = manifest["files"][backbone + ".bin"]
        path = hf_hub_download(
            manifest["repository"],
            backbone + ".bin",
            revision=manifest["revision"],
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
        if Path(path).stat().st_size != info["bytes"]:
            raise ValueError("Legacy checkpoint size mismatch")
        return cls.from_checkpoint(
            path, backbone=backbone, expected_sha256=info["lfs_sha256"]
        ).to(device)

    def save_pretrained(self, save_directory, *, max_shard_size="5GB", provenance=None):
        from huggingface_hub import save_torch_state_dict

        directory = Path(save_directory)
        directory.mkdir(parents=True, exist_ok=True)
        if any(directory.iterdir()):
            raise FileExistsError(
                "Choose an empty model directory to avoid overwriting another release"
            )
        # CPU contiguous tensors preserve original dtypes and parameter names.
        state = {
            k: v.detach().cpu().contiguous()
            for k, v in self.backbone.state_dict().items()
        }
        # DA3 shares auxiliary layers under several checkpoint names. Preserve
        # every key by cloning repeated storage before the Hub serializer sees it.
        seen_storage = set()
        for key, tensor in state.items():
            storage = (
                tensor.untyped_storage().data_ptr(),
                tensor.untyped_storage().nbytes(),
            )
            if storage in seen_storage:
                state[key] = tensor.clone()
            else:
                seen_storage.add(storage)
        save_torch_state_dict(
            state,
            str(directory),
            safe_serialization=True,
            max_shard_size=max_shard_size,
        )
        (directory / "config.json").write_text(
            json.dumps(self.config.to_dict(), indent=2) + "\n"
        )
        (directory / "preprocessor_config.json").write_text(
            json.dumps(
                {
                    "image_processor_type": "PUREImageProcessor",
                    "size": list(self.config.image_size),
                    "resample": "bicubic",
                    "resize_mode": "preserve-aspect-center-crop",
                    "input_range": [0, 1],
                    "normalization": "inside backbone",
                    "video_sampling": "consecutive frames; default first 48",
                },
                indent=2,
            )
            + "\n"
        )
        for name in ["LICENSE", "THIRD_PARTY_NOTICES.md", "LICENSES"]:
            source = RESOURCE_ROOT / name
            if source.is_dir():
                shutil.copytree(source, directory / name)
            else:
                shutil.copy2(source, directory / name)
        (directory / "README.md").write_text(model_card(self.config.backbone))
        if provenance:
            (directory / "provenance.json").write_text(
                json.dumps(provenance, indent=2) + "\n"
            )
        return directory

    def push_to_hub(self, repo_id, *, token=None, private=True, max_shard_size="5GB"):
        """Explicit upload; a new repository defaults to private."""
        from huggingface_hub import HfApi

        api = HfApi(token=token)
        with tempfile.TemporaryDirectory() as temp:
            directory = self.save_pretrained(
                Path(temp) / "model", max_shard_size=max_shard_size
            )
            api.create_repo(repo_id, private=private, exist_ok=True)
            return api.upload_folder(
                repo_id=repo_id,
                folder_path=directory,
                commit_message="Upload PURE model",
            )


def model_card(backbone):
    return f"""---
library_name: pure-posttraining
license: other
license_name: mixed-component-terms
license_link: https://huggingface.co/cyun9286/PURE/blob/main/studio/THIRD_PARTY_NOTICES.md
tags:
- pure
- {backbone}
- 3d-reconstruction
- depth-estimation
- safetensors
---
# PURE / {backbone.upper()}

Recovered PURE refined weights, serialized losslessly as Safetensors. Pi3 uses
only v2; DA3 is Giant; VGGT uses its depth/camera adapter. Install the accompanying
PURE source package (`pip install -e '.[inference]'`).

```python
import torch
from pure import PURE, PUREImageProcessor
model = PURE.from_pretrained("/path/to/this-model-directory").to("cuda")
images = PUREImageProcessor().images(["frame0.png", "frame1.png"])
with torch.inference_mode():
    predictions = model(images.unsqueeze(0).to("cuda"))
```

Replace the local directory with your Hub model ID after upload. For a combined
repository, pass `subfolder="{backbone}"`. Loading uses the installed `pure`
package, without `trust_remote_code` or Transformers AutoModel registration.

Inputs: RGB `[B, V, 3, H, W]` in `[0, 1]`, H/W multiples of 14. Outputs include
camera-to-world poses, local/world points, and relative depth. Translation and
depth are in native model units, not guaranteed metric units. Dynamic objects
can produce multi-frame ghosts. Sky filtering is an optional RGB postprocess.

Source: [FunnyGuy525/RefinedModel](https://huggingface.co/FunnyGuy525/RefinedModel),
revision `45b24b6ec0963003281189455a2e04d517634133`. See provenance.json when present.
Full-weight inference has run on H20 for public SpatialVID and SelfEvo examples.
Those are execution checks, not paper metric reproduction; no ground-truth
geometry or proof of the exact paper training run was recovered.

The original checkpoint repository did not specify a separate weight license.
The included notices preserve the component terms (including non-commercial
terms where applicable); conversion does not grant a new blanket weight license.
"""

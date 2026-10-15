"""Losslessly convert pinned legacy checkpoints into standard HF model folders."""

from pathlib import Path
import argparse
import gc
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from pure import PURE
from pure.training.checkpoint import read_checkpoint, model_state
from pure.hub import BACKBONES, RESOURCE_ROOT, read_safetensors, sha256


def convert(name, checkpoint, output, *, expected_sha=None, max_shard_size="5GB"):
    digest = sha256(checkpoint)
    if expected_sha and digest != expected_sha:
        raise ValueError(f"{name}: source hash mismatch")
    model = PURE.from_checkpoint(checkpoint, backbone=name)
    provenance = {
        "source_checkpoint": Path(checkpoint).name,
        "source_sha256": digest,
        "conversion": "lossless; original names and dtypes; no quantization",
        "package_version": "0.3.0",
    }
    if expected_sha:
        legacy = json.loads((RESOURCE_ROOT / "legacy_weights.json").read_text())
        provenance.update(
            source_repository=legacy["repository"], source_revision=legacy["revision"]
        )
    model.save_pretrained(output, max_shard_size=max_shard_size, provenance=provenance)
    restored = read_safetensors(output)
    original = model_state(read_checkpoint(checkpoint))
    if original.keys() != restored.keys():
        raise AssertionError("Converted tensor keys differ")
    for key, tensor in original.items():
        if tensor.dtype != restored[key].dtype or not torch.equal(
            tensor.cpu(), restored[key]
        ):
            raise AssertionError(f"Lossless round-trip failed: {key}")
    record = {
        "model": name,
        "status": "passed",
        "tensor_count": len(original),
        "tensor_elements": sum(x.numel() for x in original.values()),
        "exact_tensor_equality": True,
        "source_sha256": digest,
        "files": {
            p.name: {"bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(Path(output).glob("model*.safetensors"))
        },
    }
    (Path(output) / "conversion-verification.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    del model, restored, original
    gc.collect()
    return record


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--models", nargs="+", choices=BACKBONES, default=list(BACKBONES))
    p.add_argument("--max-shard-size", default="5GB")
    p.add_argument(
        "--custom-checkpoints",
        action="store_true",
        help="Skip the pinned public checkpoint hash requirement for your own trained weights",
    )
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((RESOURCE_ROOT / "legacy_weights.json").read_text())
    records = []
    for name in args.models:
        expected = (
            None
            if args.custom_checkpoints
            else manifest["files"][name + ".bin"]["lfs_sha256"]
        )
        record = convert(
            name,
            args.checkpoint_dir / (name + ".bin"),
            args.output / name,
            expected_sha=expected,
            max_shard_size=args.max_shard_size,
        )
        records.append(record)
        print(json.dumps(record), flush=True)
        (args.output / "conversion-summary.json").write_text(
            json.dumps(records, indent=2) + "\n"
        )
    (args.output / "README.md").write_text("""---
library_name: pure-posttraining
license: other
license_name: mixed-component-terms
license_link: https://huggingface.co/cyun9286/PURE/blob/main/studio/THIRD_PARTY_NOTICES.md
tags: [pure, 3d-reconstruction, depth-estimation, safetensors]
---
# PURE refined models

Each subfolder (`pi3`, `da3`, `vggt`) is a standalone HF model directory with
config, preprocessing config, Safetensors, model card, provenance and licenses.
Install the PURE source package, then load with:

```python
from pure import PURE
model = PURE.from_pretrained("cyun9286/PURE", subfolder="pi3").to("cuda")
```

A local directory works identically. See each model card for component terms,
checkpoint provenance and evaluation limits. Tensor conversion is lossless.
""")
    (args.output / ".gitattributes").write_text(
        "*.safetensors filter=lfs diff=lfs merge=lfs -text\n"
    )


if __name__ == "__main__":
    main()

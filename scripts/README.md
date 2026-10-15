# Commands

Run from the repository root. Each Python entry exposes `--help`; shell launchers
use `python3`, or the interpreter selected with `PYTHON=/path/to/python`.

| Task | Entry |
| --- | --- |
| Sample gallery | `bash scripts/demo/run_demo.sh` |
| Video/images → PLY, GLB and cameras | `python scripts/demo/reconstruct.py` |
| Local checkpoint → prediction NPZ | `python scripts/demo/infer_checkpoint.py` |
| Single GPU / single-node DDP | `bash scripts/train/run_train.sh` (`GPUS=4` for four GPUs) |
| Training with custom torchrun settings | `python scripts/train/train.py` |
| Prepare teacher caches | `python scripts/data/prepare_cache.py` |
| Check parameter shapes | `python scripts/eval/check_model_shapes.py` |
| GPU conversion parity and Studio checks | `python scripts/eval/validate_release.py` |
| Download pinned original weights | `python scripts/hub/download_weights.py` |
| Convert to Safetensors | `python scripts/hub/convert_weights.py` |
| Upload model folders | `python scripts/hub/upload.py` |

## Sample runner

```bash
bash scripts/demo/run_demo.sh --sample all --model all --output outputs/all-samples
bash scripts/demo/run_demo.sh --sample all --model all --dry-run
bash scripts/demo/run_demo.sh --sample penguin --download-only
```

Bundled inputs are verified by size and SHA256; missing files are downloaded.
Runs are sequential and refuse nonempty result folders. `--dry-run` prints the
plan without downloads, writes or model loading; `--download-only` checks inputs.

Install `.[inference]` for reconstruction/Hub tools, `.[training]` for training,
and `.[space]` for Studio. The GPU release check requires original and converted
weights, 48 PNG frames and a cached sky model; it checks execution, not paper metrics.

[Demo](../examples/README.md) · [Training](../docs/TRAINING_INFERENCE_ZH.md) · [Deployment](../docs/HUGGINGFACE_ZH.md)

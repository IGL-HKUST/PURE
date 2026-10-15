# Code map

| Path | Role |
| --- | --- |
| `app.py` | Gradio / Hugging Face Space |
| `pure/hub.py` | `PURE`, config, Hub loading and saving |
| `pure/processing.py` | Image preprocessing and consecutive video decoding |
| `pure/pipeline.py`, `pure/sky.py`, `pure/export.py` | Shared reconstruction, sky filtering and exports |
| `pure/models/`, `pure/_vendor/` | Model adapters and third-party network components |
| `pure/loss.py`, `pure/utils/` | PURE objective and geometry helpers |
| `pure/training/` | Caches, optimization, parameter selection and resume |
| `pure/cli/` | Shared train, checkpoint inference and cache commands |
| `scripts/` | [CLI entry points](../scripts/README.md), grouped by task |
| `configs/` | Training recipes |
| `examples/` | Demo videos, image sequence and source manifest |
| `weights/` | Weight manifests, tensor shapes and source provenance |
| `tests/` | Functional and regression tests |

The Studio and reconstruction CLI share `ReconstructionPipeline`. Public scripts
call library implementations; model/config/tensor names match the released weights.
Licenses and the legacy weight manifest in `pure/resources/` are packaged copies
needed by installed wheels and model export.

The workflow organization follows [GAE](https://github.com/TencentARC/GAE-GeometricAutoEncoder).

# Model weights

[cyun9286/PURE](https://huggingface.co/cyun9286/PURE) contains `pi3/`, `da3/` and
`vggt/`, each with config, processor config, Safetensors and provenance.

```python
from pure import PURE
model = PURE.from_pretrained("cyun9286/PURE", subfolder="pi3")
```

Local model directories work identically. See [conversion and deployment](HUGGINGFACE_ZH.md).

## Original checkpoints

The pinned source is [FunnyGuy525/RefinedModel](https://huggingface.co/FunnyGuy525/RefinedModel),
revision `45b24b6ec0963003281189455a2e04d517634133`.

| Model | File | Size | Tensor entries |
| --- | --- | --- | --- |
| Pi3 | `pi3.bin` | 3.70 GB | 1,238 |
| DA3 Giant | `da3.bin` | 4.82 GB | 847 |
| VGGT | `vggt.bin` | 4.63 GB | 1,341 |

```bash
python scripts/hub/download_weights.py --model all --output checkpoints
```

Downloads verify size and SHA256 against [manifest.json](../weights/manifest.json).
The shape manifests support topology checks. These files contain model parameters,
not optimizer state; use `--checkpoint` to start a new training run from them.

All three complete weights passed H20 sample inference and lossless conversion.
Their exact paper-run provenance and a separate upstream weight license were not
provided. Component terms are retained in [third-party notices](../THIRD_PARTY_NOTICES.md).
Source hashes are in [provenance.json](../weights/provenance.json) and
[vendor-provenance.json](../weights/vendor-provenance.json).

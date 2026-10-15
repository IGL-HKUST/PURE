# Third-party sources and license scope

The root MIT license covers the existing PURE release and newly authored
cache/training/CLI utilities. Recovered model files retain the terms of their
upstream sources. This notice does not relicense those components.

## Recovered Pi3 network

`pure/models/pi3/model.py`, `pure/models/pi3/layers/`, and
`pure/models/pi3/dinov2/` were recovered from the supplied `pi3.zip`.
The archive contained no root license or upstream revision identifier.
The network is derived from [Pi3](https://github.com/yyfz/Pi3), whose current
main-branch code license and training-branch notice differ. The training
branch's exact notice is retained in `LICENSES/Pi3-training.txt` (source blob
`327d795e289598855be6f74533dc6ecf19ea0188`). It starts with "Pi3 for
non-commercial purposes". Do not assume the current main branch's BSD label
retroactively covers every file in this unidentified archive revision.

Adaptations in this reference: retain only the archived v2 network topology;
remove implicit VGGT/deleted-server weight loads and automatic freezing;
expose explicit training stages; omit confidence/global-point training;
replace backend-selection contexts with PyTorch's automatic SDPA selection;
make disabled-autocast contexts device-aware; support meta construction for
parameter inspection. The archived normalization, feature levels, convolution
head, point projection, and camera conventions are preserved.

The archive originals are retained outside this release in the local audit
directory. No `.pyc` files, empty VGGT placeholder, unused SegFormer copy, or
unneeded dataset/debug utilities are distributed in the reference package.

## File-specific sources

| Files/component | Supplied/source notice | Included text |
| --- | --- | --- |
| `dinov2/`, `layers/attention.py`, `layers/block.py` | Meta Platforms, Inc. and affiliates; Apache-2.0 headers | `LICENSES/Apache-2.0.txt` |
| `layers/pos_embed.py` | Naver Corporation; CC BY-NC-SA 4.0 header, including noncommercial limitation | `LICENSES/CC-BY-NC-SA-4.0.txt`, `LICENSES/CroCo-NOTICE.txt` |
| `layers/camera_head.py` | Archive explicitly attributes adaptation from Niantic's marepo transformer | Full upstream notice in `LICENSES/marepo.txt` |
| Geometry/convolutional-head lineage | Archive geometry code attributes portions to MoGe; recovered head follows related multi-level residual-convolution design | Upstream combined MIT/Apache text retained in `LICENSES/MoGe.txt` for provenance; no claim that it supersedes any file-specific terms |

Sources checked during recovery:

- https://github.com/yyfz/Pi3/blob/training/LICENSE
- https://github.com/facebookresearch/dinov2/blob/main/LICENSE
- https://github.com/naver/croco/blob/master/LICENSE
- https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.txt
- https://github.com/nianticlabs/marepo/blob/main/LICENSE
- https://github.com/microsoft/MoGe/blob/main/LICENSE

The marepo attribution points to `transformer/transformer.py` at commit
`9a45e2bb07e5bb8cb997620088d352b439b13e0e`; the included upstream license
contains the terms and exceptions for that directory. Noncommercial terms
remain relevant to the recovered model. This package makes no blanket
commercial-use permission claim.

## Weights and datasets

The supplied ByteDance DA3 adaptation is preserved in the offline recovery
archive with its Apache-2.0 header. Its SHA-256 is
`60e8b85e7f8a38c731b13cdc987b71b7101159b3168aeea33e158c05ab381d1a`.
Its forward conventions guided `pure/models/da3.py`; the runtime uses vendored
DA3 modules and differentiable projection. See `weights/provenance.json`.

Model weights and training datasets are not bundled. Public SelfEvo demo
videos and images are included under `examples/`; their source revision and
hashes are recorded in `examples/reconstruction/sources.json`. Upstream terms
apply separately.
Training scripts accept local checkpoints; they do not obtain or redistribute
third-party models. The exact source revision of the recovered Pi3 code and
the final experimental variant remain unverified. The external Pi3 checkpoint
is structurally compatible with v2, while its training stage and evaluation
provenance remain unspecified. The external weights repository currently has
no model card or license metadata; the links do not grant additional rights.

## Vendored DA3 and VGGT runtime modules

- DA3: `pure/_vendor/da3/`, from ByteDance-Seed/Depth-Anything-3 revision
  `3d835ec1a5802d64a8b8b15f817a1ab54809bfe4`. Full upstream license is
  `LICENSES/DA3.txt`; file-specific Meta/ByteDance notices are retained.
- VGGT: `pure/_vendor/vggt/`, from facebookresearch/vggt revision
  `a288dd0f14786c93483e45524328726ab7b1b4ce`. Full upstream license is
  `LICENSES/VGGT.txt`, including its use restrictions. These components and
  their adaptations are subject to the upstream license, not the root MIT grant.
- Changes: import paths relocated under `pure._vendor`; CPU construction of
  the stochastic-depth schedule permits meta-device topology checks; DA3
  attention blocks use non-reentrant activation checkpointing during training;
  trailing whitespace removed. `weights/vendor-provenance.json` records file hashes.
- `pure/models/vggt.py` assembles the upstream aggregator, depth and camera
  heads with a differentiable geometry conversion. `pure/models/da3.py`
  reconstructs the surviving DA3 adapter and applies standard input normalization.
- No new permission for the external refined weight files is inferred from
  the code licenses. The weight repository did not supply license metadata.

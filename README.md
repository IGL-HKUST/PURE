# PURE: Refining Feed-forward 3D Reconstruction with Unlabeled Post-training

<p align="center">
  <a href="https://arxiv.org/abs/2610.14698"><img src="https://img.shields.io/badge/arXiv-2610.14698-b31b1b.svg" alt="arXiv Paper"></a> &nbsp;
  <a href="https://funnyguy525.github.io/PURE-Page/"><img src="https://img.shields.io/badge/Project-Page-Green" alt="Project Page"></a> &nbsp;
  <a href="https://huggingface.co/cyun9286/PURE"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Weights-blue" alt="Hugging Face Weights"></a>
</p>

<div align="center">

<p>
  Yaolong Zhu<sup>1</sup>,
  <a href="https://penghtyx.github.io/yuki-lipeng/">Peng Li</a><sup>2</sup>,
  <a href="https://github.com/jiah-cloud">Jiahao Lu</a><sup>2,▲</sup>,
  <a href="https://afterjourney00.github.io/">Chengfeng Zhao</a><sup>2</sup>,
  Mengfei Li<sup>2</sup>,
  Chao Wang<sup>3</sup>,
  Wenyue Chen<sup>1</sup>,<br>
  Zekai Gu<sup>2</sup>,
  Chengwei Ren<sup>2</sup>,
  <a href="https://www.ece.pku.edu.cn/info/1046/2147.htm">Ronggang Wang</a><sup>1,✉</sup>,
  <a href="https://liuyuan-pal.github.io/">Yuan Liu</a><sup>2,✉</sup>
</p>

<p>
  <sup>1</sup> Guangdong Provincial Key Laboratory of Ultra High Definition Immersive Media Technology, Shenzhen Graduate School, Peking University
  <br>
  <sup>2</sup> The Hong Kong University of Science and Technology
  <br>
  <sup>3</sup> Pengcheng Laboratory
  <br>
  <sup>▲</sup> Project lead
  &nbsp;&nbsp;·&nbsp;&nbsp;
  <sup>✉</sup> Co-corresponding authors
</p>

</div>

<p align="center">
  <b>Reconstruct 3D points, depth and camera poses from video or images with Pi3, DA3 Giant and VGGT.</b>
</p>

<p align="center">
  <a href="https://funnyguy525.github.io/PURE-Page/#long"><img src="assets/teaser.gif" alt="PURE full film: 2,000-frame reconstruction with growing overhead map, five short scenes and unlabeled post-training" width="100%"></a>
</p>

<p align="center">
  <b>PURE · From unlabeled video to dynamic 3D</b><br>
  Full 1:54 film: 2,000-frame reconstruction at 30 fps, overhead view, five short scenes, then the method.<br>
  A growing overhead map and synchronized RGB input are included. The animated preview covers the full film at 6 fps; the MP4 is 1080p / 30 fps.<br>
  <a href="https://funnyguy525.github.io/PURE-Page/#long">Watch on the project page</a> · <a href="assets/teaser.mp4">Download full MP4 (1:54)</a>
</p>

## Pipeline

<p align="center">
  <img src="assets/pipeline.jpg" alt="PURE post-training pipeline" width="100%">
</p>

## Demo

### Web app

Linux, Python 3.11 and a CUDA GPU. From the repository root:

```bash
pip install -e '.[space]'
PURE_MODEL_REPO=cyun9286/PURE python app.py
```

Open **http://localhost:7860** and upload a video or image sequence. The default uses **48 consecutive frames** and **removes sky**. Outputs: interactive 3D preview, PLY, camera poses, RGB/depth videos and ZIP.

Public weights download on first use; no login is required. [Deployment and offline setup](docs/HUGGINGFACE_ZH.md).

### Command line

Run the bundled demos from the repository root:

```bash
pip install -e '.[inference]'
bash scripts/demo/run_demo.sh                         # Motorcycle + PURE (Pi3)
bash scripts/demo/run_demo.sh --sample all --model all # Four samples × three models
```

Use `--sample motorcycle|penguin|sport|drift-straight` and `--model pi3|da3|vggt` to select a demo. Results go to `outputs/demo/<sample>-<model>/`; use `--output` for a new run or `--dry-run` to inspect commands. Videos use the first 48 consecutive frames; drift-straight uses all 10 images.

For your own video:

```bash
python scripts/demo/reconstruct.py \
  --pretrained cyun9286/PURE --subfolder pi3 \
  --video input.mp4 --frames 48 --output outputs/my-video
```

Choose `pi3`, `da3` or `vggt` with `--subfolder`. [Demo options](examples/README.md).

## Python API

```python
import torch
from pure import PURE, PUREImageProcessor

model = PURE.from_pretrained("cyun9286/PURE", subfolder="pi3").to("cuda")
images = PUREImageProcessor().video("input.mp4", max_frames=48)
with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
    predictions = model(images.unsqueeze(0).to("cuda"))
```

The API returns tensors. The CLI and Studio add sky filtering and exports. [Model formats and provenance](docs/WEIGHTS.md).

## Training

Prepare aligned teacher caches using the [training guide](docs/TRAINING_INFERENCE_ZH.md) and [data format](docs/DATA_FORMAT.md), then run:

```bash
pip install -e '.[training]'
python scripts/hub/download_weights.py --model da3 --output checkpoints
bash scripts/train/run_train.sh \
  --config configs/da3_posttrain.json --data data/da3_train \
  --checkpoint checkpoints/da3.bin --output outputs/train-da3
```

Prefix with `GPUS=4` for single-node DDP. Pi3 and VGGT use their matching configs and checkpoints.

## Citation

```bibtex
@inproceedings{zhu2026pure,
  author    = {Zhu, Yaolong and Li, Peng and Lu, Jiahao and Zhao, Chengfeng and Li, Mengfei and Wang, Chao and Chen, Wenyue and Gu, Zekai and Ren, Chengwei and Wang, Ronggang and Liu, Yuan},
  title     = {{PURE}: Refining Feed-forward {3D} Reconstruction with Unlabeled Post-training},
  booktitle = {Proceedings of the 34th ACM International Conference on Multimedia},
  series    = {MM '26},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3767308.3836296},
  url       = {https://doi.org/10.1145/3767308.3836296}
}
```

## Acknowledgements

PURE builds on [VGGT](https://github.com/facebookresearch/vggt), [Pi3](https://github.com/yyfz/Pi3), and [Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3). Pseudo supervision uses [MoGe](https://github.com/microsoft/MoGe), [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2), and [GIM](https://github.com/xuelunshen/gim). The scale-and-shift solver follows [Track4World](https://github.com/TencentARC/Track4World).

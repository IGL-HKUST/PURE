# Studio 与 Hugging Face

## 本地启动

Linux、Python 3.11、CUDA GPU；48 帧建议至少 24 GB 显存。

```bash
pip install -e '.[space]'
PURE_MODEL_REPO=cyun9286/PURE python app.py
```

公开权重首次使用时自动下载，无需登录。访问 `http://localhost:7860`。视频从指定帧开始连续读取，默认 48 帧；图片按文件名
自然排序。至少需要两帧，保持比例缩放后中心裁剪到 518×294。

默认输出去天空 PLY、相机 NPZ、RGB/深度视频和 ZIP。GLB 预览最多 15 万点，
剔除最远 1%，转到第一相机坐标并归一化；完整 PLY 保留所有有限、正深度的非天空点。
深度和位姿使用模型相对尺度，运动物体可能出现重影。勾选原始预测后 ZIP 会更大。

## Space 部署

将源码上传到 Gradio Space，选择 CUDA GPU。`requirements.txt` 安装依赖，
`app.py` 监听 `0.0.0.0:7860`。模型按需加载，重建请求串行执行。

| 变量 | 用途 |
| --- | --- |
| `PURE_MODEL_REPO` | 模型仓库 ID，内含 pi3/、da3/、vggt/ |
| `PURE_MODEL_REVISION` | 可选，固定权重提交 |
| `PURE_MODEL_ROOT` | 本地模型父目录，优先于仓库设置 |
| `PURE_SKY_MODEL_DIR` | 可选，本地 SegFormer 配置和权重 |
| `PURE_OUTPUT_DIR` | 输出目录，默认系统临时目录下 pure-studio |
| `PURE_KEEP_HOURS` | 结果保留小时数，默认 24；下一次请求前清理 |

私有仓库使用 Secret `HF_TOKEN`。不设置模型地址时，使用固定版本的原始公开
`.bin` 权重，并校验 SHA256。本版本需要常驻 CUDA GPU，尚未部署托管 Space。

## 权重转换与上传

```bash
python scripts/hub/download_weights.py --model all --output checkpoints
python scripts/hub/convert_weights.py --checkpoint-dir checkpoints --output hf_models
python scripts/hub/upload.py --folder hf_models --repo-id cyun9286/PURE
```

`hf_models/{pi3,da3,vggt}/` 各包含 `config.json`、`preprocessor_config.json`、
`model.safetensors`、模型卡、来源和许可证。转换逐张量核对 dtype 与数值；
`--max-shard-size 2GB` 可分片。已有模型输出目录不会覆盖。

自训练权重按 `pi3.bin`、`da3.bin`、`vggt.bin` 命名，使用 `--custom-checkpoints`；
`--models pi3` 可只转换一种模型。推理模型包不保存优化器状态。
上传脚本新建仓库默认为私有，`--public` 创建公开仓库；不改变已有仓库的可见性。

安装本项目后，可用 `PURE.from_pretrained("hf_models", subfolder="pi3")` 加载本地模型，
或用 Hub ID 加载，支持 revision、缓存和 token。`save_pretrained()` 保存到空目录。
## 在线 PURE Studio

已部署的演示：<https://huggingface.co/spaces/cyun9286/PURE-Studio>。

支持五个视频示例（Motorcycle、Penguin、Sport、Basketball close-up、Basketball wide view）和 Drift-straight 的 10 张图片。
点击示例会填入输入并切换输入类型，图片会显示缩略图；点击 Reconstruct 才会申请 GPU。
视频示例默认处理 16 个连续帧，图片示例处理全部 10 张图片，均可调整模型和帧数。

仓库根目录的 `app.py` 是通用 CUDA 入口，`space_app.py` 是 ZeroGPU 入口。
ZeroGPU 版本使用 Python 3.10.13、PyTorch 2.8.0、Gradio 5.49.1 和 `spaces`；
模型及天空分割权重先下载到缓存，再按帧数申请 GPU 时长，GPU 额度由 Hugging Face 管理。
部署为 Gradio Space 时将 `app_file` 设为 `space_app.py`，并使用根目录
`requirements.txt` 安装依赖。示例来自 SelfEvo 及项目主页的两个篮球原始视频，来源和校验值见
`examples/reconstruction/sources.json`。

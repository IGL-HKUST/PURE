# 训练与 checkpoint 推理

训练使用已对齐的教师缓存，支持 Pi3、DA3 Giant、VGGT、BF16、梯度累积和 DDP。
原始 MoGe / Grounded SAM 2 / GIM 预处理服务与数据混合流程未恢复，需要预先准备其输出。

## 安装与权重

```bash
pip install -e '.[training]'
python scripts/hub/download_weights.py --model all --output checkpoints
```

配置和权重必须对应同一模型。完整 refined checkpoint 严格加载；来源和大小见
[权重说明](WEIGHTS.md)。推理及网页演示使用方法见 [README](../README.md)。

## 准备数据

每个 `.npz` 保存一个 clip：RGB、单目教师深度、冻结底座射线、有效/动态掩码，以及
过滤后的相邻帧匹配。字段见 [DATA_FORMAT.md](DATA_FORMAT.md)。至少两帧，H/W 为
14 的倍数，所有数据必须共享帧顺序与像素网格；打包和训练不隐式缩放。

```bash
python scripts/data/prepare_cache.py \
  --images frames/0000.png frames/0001.png \
  --labels labels.npz --output data/train/clip000.npz
```

可加 `--teacher-prediction outputs/frozen_base.npz` 使用冻结底座推理结果中的 `xy`，
此时 labels 可省略 `teacher_xy`。它必须来自同一批预处理图像；不同底座分别准备缓存。
生产训练拒绝标为 synthetic 的测试缓存。

## 训练

```bash
bash scripts/train/run_train.sh \
  --config configs/da3_posttrain.json --data data/da3_train \
  --checkpoint checkpoints/da3.bin --output outputs/da3
```

Pi3 和 VGGT 换用对应的配置、缓存与 checkpoint。每个 microbatch 为一个 clip，
默认累积 2 次。配置 `precision` 为 `bf16` 或 `fp32`。

| 模型 | 默认训练层（从 0 编号） |
| --- | --- |
| Pi3 | decoder 11–35 中的奇数层；`train_head=true` 可同时训练卷积头 |
| DA3 | backbone blocks 19、27、33、39 |
| VGGT | global blocks 4、11、17、23 |

DA3/VGGT 使用 `train_blocks`、`train_head`、`train_camera` 调整范围。
VGGT 选层是恢复实现的参考配置，未确认论文最终选层。DA3 的辅助射线分支仅保留
用于 checkpoint 兼容，几何射线由相机分支生成。

### Pi3 底座初始化

现有 `pi3.bin` 已包含卷积头，直接 posttrain。仅从兼容的原始 Pi3 底座开始时先执行：

```bash
python scripts/train/train.py --config configs/pi3_head_init.json --data data/pi3_train \
  --base-checkpoint checkpoints/pi3_original.safetensors --output outputs/pi3-head
python scripts/train/train.py --config configs/pi3_posttrain.json --data data/pi3_train \
  --checkpoint outputs/pi3-head/latest.pt --output outputs/pi3-posttrain
```

不兼容 Pi3X。旧 v2 配置元数据可读取，v1 会被拒绝。DA3/VGGT 的 `--base-checkpoint`
只允许忽略明确列出的未使用官方分支，不允许缺失当前模型参数。

## 多卡与恢复

```bash
GPUS=4 bash scripts/train/run_train.sh \
  --config configs/vggt_posttrain.json --data data/vggt_train \
  --checkpoint checkpoints/vggt.bin --output outputs/vggt-ddp
GPUS=4 bash scripts/train/run_train.sh \
  --config configs/vggt_posttrain.json --data data/vggt_train \
  --resume outputs/vggt-ddp/latest.pt --output outputs/vggt-ddp
```

多机使用 `torchrun` 启动 `scripts/train/train.py` 并自行指定 rendezvous 参数。
恢复要求配置、缓存路径顺序和进程数相同，也应保持缓存内容不变。
`latest.pt` 保存模型、优化器、调度器、步数及各 rank 的 RNG/采样状态；
`metrics.jsonl` 保存训练 loss、梯度范数和学习率。有效批量为 `进程数 × accumulation`，
各卡独立采样，可能抽到同一 clip。当前循环没有独立验证集评估或 best checkpoint 选择。

新实验要求输出目录为空；断点续训显式传 `--resume`，可继续写入该目录的 `latest.pt`
或使用新的空目录。若 checkpoint 已完成总步数，加载模型后直接返回原 checkpoint，
不执行额外训练、不创建新的 checkpoint。

## Checkpoint 推理

```bash
python scripts/demo/infer_checkpoint.py --model da3 --checkpoint outputs/da3/latest.pt \
  --images frames/0000.png frames/0001.png --resize 294 518 \
  --output outputs/da3.npz --device cuda --precision bf16
```

`--resize H W` 直接缩放，不保持比例；不传时图片需同尺寸且 H/W 可被 14 整除。
RGB 使用 `[0,1]`，模型内部归一化。输出 NPZ 包含 `depth`、`local_points`、`points`、
`camera_poses`、`xy` 和输入元数据；DA3/VGGT 还输出 `intrinsics`、`conf`。
位姿是 camera-to-world，坐标为 OpenCV 右/下/前，深度和位移为模型相对单位。

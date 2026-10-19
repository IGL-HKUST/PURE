"""ZeroGPU entry point for the upstream PURE Reconstruction Studio."""

import os
from pathlib import Path

# Import before torch so Spaces can initialize its CUDA runtime.
import spaces
from huggingface_hub import hf_hub_download, snapshot_download

import app as studio
from pure.sky import REPO as SKY_REPO, REVISION as SKY_REVISION

MODEL_REPO = "cyun9286/PURE"
MODEL_REVISION = "1e674e573217af0680a6dc03c99c07c4686e5d9e"


def gpu_duration(source, video, files, backbone, count, start, remove_sky, save_raw):
    """Reserve less GPU time for small clips, including model-loading overhead."""
    return min(180, 30 + 2 * int(count))


_gpu_reconstruct = spaces.GPU(duration=gpu_duration)(studio.reconstruct)


def reconstruct(source, video, files, backbone, count, start, remove_sky, save_raw):
    """Cache public checkpoints before requesting scarce GPU time."""
    if backbone not in ("pi3", "da3", "vggt") or not 2 <= int(count) <= 48:
        raise studio.gr.Error("Choose a model and 2–48 frames.")
    if not 0 <= int(start) <= 10000:
        raise studio.gr.Error("Choose a first video frame between 0 and 10000.")
    if source == "Video" and not video:
        raise studio.gr.Error("Upload a video first.")
    if source == "Images" and (not files or len(files) < 2):
        raise studio.gr.Error("Upload at least two images.")
    if source not in ("Video", "Images"):
        raise studio.gr.Error("Choose Video or Images.")

    model_root = snapshot_download(
        MODEL_REPO,
        revision=MODEL_REVISION,
        allow_patterns=[f"{backbone}/config.json", f"{backbone}/model*.safetensors"],
    )
    os.environ["PURE_MODEL_ROOT"] = model_root
    if remove_sky:
        for filename in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
            sky_path = hf_hub_download(SKY_REPO, filename, revision=SKY_REVISION)
        os.environ["PURE_SKY_MODEL_DIR"] = str(Path(sky_path).parent)
    return _gpu_reconstruct(
        source, video, files, backbone, count, start, remove_sky, save_raw
    )


studio.reconstruct = reconstruct
demo = studio.build_app()

if __name__ == "__main__":
    studio.OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", "7860")),
        allowed_paths=[str(studio.OUTPUT_ROOT)],
        max_file_size="500mb",
    )

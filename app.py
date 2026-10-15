"""PURE Reconstruction Studio — Gradio / Hugging Face GPU Space."""

from pathlib import Path
import gc
import os
import shutil
import tempfile
import threading
import time

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import torch

from pure import PURE, PUREImageProcessor
from pure.processing import natural_key
from pure.pipeline import ReconstructionPipeline

OUTPUT_ROOT = Path(
    os.environ.get("PURE_OUTPUT_DIR", tempfile.gettempdir() + "/pure-studio")
).resolve()
MAX_FRAMES = 48
MODEL_NAMES = {"pi3": "PURE (Pi3)", "da3": "PURE (DA3)", "vggt": "PURE (VGGT)"}
_LOCK = threading.Lock()
_CACHED = None
_CACHE_KEY = None


def _pipeline(backbone):
    global _CACHED, _CACHE_KEY
    model_root = os.environ.get("PURE_MODEL_ROOT")
    model_repo = os.environ.get("PURE_MODEL_REPO")
    revision = os.environ.get("PURE_MODEL_REVISION")
    key = (backbone, model_root, model_repo, revision)
    if key != _CACHE_KEY:
        _CACHED = None
        _CACHE_KEY = None
        gc.collect()
        torch.cuda.empty_cache()
        if model_root:
            model = PURE.from_pretrained(model_root, subfolder=backbone)
        elif model_repo:
            model = PURE.from_pretrained(
                model_repo, subfolder=backbone, revision=revision
            )
        else:
            model = PURE.from_legacy_hub(backbone)
        if model.config.backbone != backbone:
            raise ValueError("Model configuration does not match the selected backbone")
        _CACHED = ReconstructionPipeline(
            model,
            device="cuda",
            sky_model_directory=os.environ.get("PURE_SKY_MODEL_DIR"),
        )
        _CACHE_KEY = key
    return _CACHED


def reconstruct(source, video, files, backbone, count, start, remove_sky, save_raw):
    count = int(count)
    start = int(start)
    if (
        backbone not in ("pi3", "da3", "vggt")
        or not 2 <= count <= MAX_FRAMES
        or not 0 <= start <= 10000
    ):
        raise gr.Error("Choose a model, 2–48 frames, and a non-negative start frame.")
    if source == "Video" and not video:
        raise gr.Error("Upload a video first.")
    if source == "Images" and (not files or len(files) < 2):
        raise gr.Error("Upload at least two images.")
    if source not in ("Video", "Images"):
        raise gr.Error("Choose Video or Images.")
    if not torch.cuda.is_available():
        raise gr.Error(
            "This app needs a CUDA GPU. Select GPU hardware for the Space; A10G 24 GB or larger is recommended for 48 frames."
        )
    with _LOCK:
        OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
        cutoff = time.time() - float(os.environ.get("PURE_KEEP_HOURS", "24")) * 3600
        for old in OUTPUT_ROOT.glob("run-*"):
            if old.is_dir() and not old.is_symlink() and old.stat().st_mtime < cutoff:
                shutil.rmtree(old)
        directory = Path(tempfile.mkdtemp(prefix="run-", dir=OUTPUT_ROOT))
        try:
            processor = PUREImageProcessor()
            if source == "Video":
                images, metadata = processor.video(
                    video, max_frames=count, start_frame=start, return_metadata=True
                )
            else:
                paths = sorted(files, key=natural_key)
                images, metadata = processor.images(
                    paths, max_frames=count, return_metadata=True
                )
                metadata["sampling"] = "natural filename order; first selected frames"
            result = _pipeline(backbone)(
                images, metadata, directory, remove_sky=remove_sky, save_raw=save_raw
            )
        except torch.cuda.OutOfMemoryError:
            global _CACHED, _CACHE_KEY
            _CACHED = None
            _CACHE_KEY = None
            gc.collect()
            torch.cuda.empty_cache()
            shutil.rmtree(directory, ignore_errors=True)
            raise gr.Error(
                "GPU memory is full. Reduce the frame count or use a larger GPU."
            ) from None
        except (ValueError, RuntimeError, OSError) as error:
            shutil.rmtree(directory, ignore_errors=True)
            raise gr.Error(str(error)) from error
    info = result["metadata"]
    status = f"**{MODEL_NAMES[backbone]} · {info['frame_count']} frames** · {info['exported_points']:,} exported points · sky removed {info['sky_fraction']:.2%} · inference {info['inference_seconds']:.2f}s"
    return (
        result["rgb"],
        result["depth"],
        result["preview"],
        result["ply"],
        result["cameras"],
        result["archive"],
        [(p, "Sky mask · pink pixels removed") for p in result["masks"]],
        status,
    )


def build_app():
    with gr.Blocks(
        title="PURE Reconstruction Studio",
        theme=gr.themes.Soft(),
        delete_cache=(3600, 86400),
    ) as demo:
        gr.Markdown(
            "# PURE Reconstruction Studio\nReconstruct depth, camera poses and a colored 3D point cloud with **PURE (Pi3) · PURE (DA3) · PURE (VGGT)**."
        )
        with gr.Row():
            with gr.Column(scale=1):
                source = gr.Radio(
                    ["Video", "Images"], value="Video", label="Input type"
                )
                video = gr.Video(label="Input video", sources=["upload"])
                files = gr.File(
                    label="Image sequence · natural filename order",
                    file_count="multiple",
                    file_types=[".png", ".jpg", ".jpeg", ".webp"],
                    type="filepath",
                    visible=False,
                )
                source.change(
                    lambda x: (
                        gr.update(visible=x == "Video"),
                        gr.update(visible=x == "Images"),
                    ),
                    inputs=source,
                    outputs=[video, files],
                    queue=False,
                )
                backbone = gr.Dropdown(
                    [(label, key) for key, label in MODEL_NAMES.items()],
                    value="pi3",
                    label="Model",
                )
                count = gr.Slider(
                    2,
                    MAX_FRAMES,
                    value=48,
                    step=1,
                    label="Frames · consecutive for video",
                )
                start = gr.Number(
                    value=0,
                    precision=0,
                    minimum=0,
                    maximum=10000,
                    label="First video frame",
                )
                remove_sky = gr.Checkbox(
                    value=True, label="Remove sky from point cloud"
                )
                save_raw = gr.Checkbox(
                    value=False, label="Include full raw predictions in ZIP (large)"
                )
                run = gr.Button("Reconstruct", variant="primary")
            with gr.Column(scale=2):
                cloud = gr.Model3D(
                    label="3D preview · drag to rotate",
                    height=480,
                    clear_color=[0.06, 0.08, 0.12, 1.0],
                )
                status = gr.Markdown()
        with gr.Row():
            rgb = gr.Video(label="Selected RGB frames", loop=True)
            depth = gr.Video(label="Relative depth · sky black", loop=True)
        masks = gr.Gallery(
            label="Sky mask check", columns=4, height="auto", show_label=True
        )
        with gr.Row():
            ply = gr.File(label="Full filtered point cloud · PLY")
            cameras = gr.File(label="Camera poses · NPZ")
            archive = gr.File(label="All outputs · ZIP")
        gr.Markdown(
            "Preview: sampled points with the farthest 1% removed. PLY: all valid non-sky points. Depth and translation use relative scale; moving objects may leave ghosts."
        )
        run.click(
            reconstruct,
            inputs=[source, video, files, backbone, count, start, remove_sky, save_raw],
            outputs=[rgb, depth, cloud, ply, cameras, archive, masks, status],
            concurrency_limit=1,
            concurrency_id="reconstruction",
            api_name="reconstruct",
        )
    return demo.queue(max_size=8, default_concurrency_limit=1)


demo = build_app()

if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    demo.launch(
        server_name=os.environ.get("HOST", "0.0.0.0"),
        server_port=int(os.environ.get("PORT", "7860")),
        allowed_paths=[str(OUTPUT_ROOT)],
        max_file_size="500mb",
    )

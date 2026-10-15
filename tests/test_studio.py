import json
from pathlib import Path
import zipfile
import numpy as np
import pytest
import torch
from PIL import Image
from pure.processing import PUREImageProcessor, natural_key
from pure.export import export_reconstruction, write_video, VERTEX_DTYPE


def test_consecutive_video_and_image_order(tmp_path):
    frames = np.stack([np.full((28, 42, 3), i * 20, dtype=np.uint8) for i in range(8)])
    path = tmp_path / "video.mp4"
    write_video(path, frames, 25)
    processor = PUREImageProcessor(28, 42)
    images, meta = processor.video(
        path, start_frame=2, max_frames=4, return_metadata=True
    )
    assert images.shape == (4, 3, 28, 42)
    assert [x["frame_index"] for x in meta["frames"]] == [2, 3, 4, 5]
    np.testing.assert_allclose(
        [x["timestamp_seconds"] for x in meta["frames"]],
        [0.08, 0.12, 0.16, 0.20],
        atol=1e-5,
    )
    assert images.mean((1, 2, 3)).tolist() == sorted(images.mean((1, 2, 3)).tolist())
    short, info = processor.video(
        path, start_frame=6, max_frames=48, return_metadata=True
    )
    assert len(short) == 2 and info["requested_frames"] == 48
    paths = []
    for index in [10, 2, 1]:
        p = tmp_path / f"frame{index}.png"
        Image.fromarray(frames[0]).save(p)
        paths.append(str(p))
    assert [Path(x).name for x in sorted(paths, key=natural_key)] == [
        "frame1.png",
        "frame2.png",
        "frame10.png",
    ]
    with pytest.raises(ValueError):
        processor.video(path, start_frame=7, max_frames=48)


def test_filtered_geometry_color_and_preview_exports(tmp_path):
    n, h, w = 3, 28, 42
    rng = np.random.default_rng(2)
    images = rng.random((n, 3, h, w), dtype=np.float32)
    depth = np.ones((n, h, w, 1), dtype=np.float32)
    points = rng.random((n, h, w, 3), dtype=np.float32)
    sky = np.zeros((n, h, w), bool)
    sky[:, :5] = True
    poses = np.broadcast_to(np.eye(4, dtype=np.float32), (n, 4, 4)).copy()
    pred = {"points": points, "depth": depth, "camera_poses": poses}
    meta = {"remove_sky": True, "preview_fps": 25, "frame_count": n}
    out = export_reconstruction(
        pred, images, sky, tmp_path, metadata=meta, save_raw=True
    )
    with open(out["ply"], "rb") as f:
        while f.readline() != b"end_header\n":
            pass
        vertices = np.fromfile(f, dtype=VERTEX_DTYPE)
    assert len(vertices) == int((~sky).sum())
    for i, k in enumerate(("x", "y", "z")):
        np.testing.assert_array_equal(vertices[k], points[~sky][:, i])
    rgb = (images.transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
    for i, k in enumerate(("red", "green", "blue")):
        np.testing.assert_array_equal(vertices[k], rgb[~sky][:, i])
    import trimesh

    scene = trimesh.load(out["preview"])
    assert sum(len(g.vertices) for g in scene.geometry.values()) == len(vertices)
    with zipfile.ZipFile(out["archive"]) as z:
        assert z.testzip() is None and "prediction.npz" in z.namelist()
    camera = np.load(out["cameras"])
    np.testing.assert_array_equal(camera["camera_poses"], poses)


def test_app_config_and_input_errors(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    import app

    assert app.demo.get_config_file()["mode"] == "blocks"
    with pytest.raises(Exception, match="Upload a video"):
        app.reconstruct("Video", None, None, "pi3", 48, 0, True, False)
    with pytest.raises(Exception, match="at least two"):
        app.reconstruct("Images", None, [], "pi3", 48, 0, True, False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(Exception, match="CUDA GPU"):
        app.reconstruct("Video", "some.mp4", None, "pi3", 48, 0, True, False)


def test_space_http_route(monkeypatch):
    monkeypatch.setenv("GRADIO_ANALYTICS_ENABLED", "False")
    import app
    from fastapi.testclient import TestClient
    from gradio.routes import App

    with TestClient(App.create_app(app.demo)) as client:
        response = client.get("/config")
        assert response.status_code == 200
        config = response.json()
        assert any(x.get("api_name") == "reconstruct" for x in config["dependencies"])

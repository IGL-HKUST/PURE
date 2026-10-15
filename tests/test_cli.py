import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import pytest
import torch
from examples.reference_training_smoke import make_synthetic_clip
from pure.training.cache import ClipDataset, write_cache
from test_adapters import small_da3, small_vggt
from test_pi3 import small_model


@pytest.mark.parametrize("name", ["pi3", "da3", "vggt"])
def test_training_to_inference_cli(name, tmp_path, monkeypatch):
    from pure.cli import train, infer

    torch.set_num_threads(2)
    torch.manual_seed(8)
    factory = {"pi3": small_model, "da3": small_da3, "vggt": small_vggt}[name]
    checkpoint = tmp_path / "initial.pt"
    torch.save(factory().state_dict(), checkpoint)
    config = json.loads(
        (Path(__file__).parents[1] / f"configs/{name}_posttrain.json").read_text()
    )
    config.update(steps=1, accumulation=1, save_every=1, precision="fp32", seed=8)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    clip = make_synthetic_clip()
    write_cache(tmp_path / "data/clip.npz", clip, synthetic=True)
    # Only the test injects synthetic-data permission. Production CLI rejects it.
    monkeypatch.setattr(
        train, "ClipDataset", lambda path: ClipDataset(path, allow_synthetic=True)
    )
    monkeypatch.setattr(train, "create_model", lambda *a, **k: factory())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train.py",
            "--config",
            str(config_path),
            "--data",
            str(tmp_path / "data"),
            "--output",
            str(tmp_path / "run"),
            "--checkpoint",
            str(checkpoint),
            "--device",
            "cpu",
        ],
    )
    train.main()
    frames = []
    for i, frame in enumerate(clip["images"]):
        path = tmp_path / f"{i}.png"
        Image.fromarray((frame.transpose(1, 2, 0) * 255).astype(np.uint8)).save(path)
        frames.append(str(path))
    output = tmp_path / "prediction.npz"
    monkeypatch.setattr(infer, "create_model", lambda *a, **k: factory())
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer.py",
            "--model",
            name,
            "--checkpoint",
            str(tmp_path / "run/latest.pt"),
            "--images",
            *frames,
            "--output",
            str(output),
            "--device",
            "cpu",
        ],
    )
    infer.main()
    with np.load(output, allow_pickle=False) as result:
        assert str(result["model"]) == name
        assert result["depth"].shape == (2, 28, 28, 1)
        assert result["points"].shape == (2, 28, 28, 3)
        assert result["camera_poses"].shape == (2, 4, 4)
        assert np.isfinite(result["points"]).all()

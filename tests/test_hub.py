import json
import pytest
import torch
from pure.hub import PURE, PUREConfig, read_safetensors
from test_pi3 import small_model
from test_adapters import small_da3, small_vggt


@pytest.mark.parametrize(
    "name,factory", [("pi3", small_model), ("da3", small_da3), ("vggt", small_vggt)]
)
@pytest.mark.parametrize("shard_size", ["5GB", 20000])
def test_lossless_hub_roundtrip(name, factory, shard_size, tmp_path, monkeypatch):
    import pure.hub as hub

    torch.set_num_threads(2)
    monkeypatch.setattr(hub, "create_model", lambda _: factory())
    torch.manual_seed(7)
    original = PURE(PUREConfig(backbone=name)).eval()
    images = torch.rand(1, 2, 3, 28, 28)
    with torch.inference_mode():
        before = original(images)
    path = tmp_path / name
    original.save_pretrained(path, max_shard_size=shard_size)
    restored = PURE.from_pretrained(tmp_path, subfolder=name)
    assert restored.config == original.config
    for key, value in original.backbone.state_dict().items():
        assert torch.equal(value, restored.backbone.state_dict()[key])
    with torch.inference_mode():
        after = restored(images)
    for key in before:
        if isinstance(before[key], torch.Tensor):
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)
    assert (path / "README.md").exists() and (path / "LICENSES/VGGT.txt").exists()
    if shard_size == 20000:
        assert (path / "model.safetensors.index.json").exists()
    with pytest.raises(FileExistsError):
        original.save_pretrained(path)


def test_config_and_shard_validation(tmp_path):
    with pytest.raises(ValueError):
        PUREConfig(backbone="unknown")
    with pytest.raises(ValueError):
        PUREConfig(image_size=(295, 518))
    with pytest.raises(ValueError):
        PURE.from_pretrained(tmp_path, subfolder="../escape")
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {"x": "../x.safetensors"}})
    )
    with pytest.raises(ValueError, match="filename"):
        read_safetensors(tmp_path)


def test_legacy_checkpoint_strict_and_metadata(tmp_path, monkeypatch):
    import pure.hub as hub

    monkeypatch.setattr(hub, "create_model", lambda _: small_model())
    path = tmp_path / "model.pt"
    model = small_model()
    torch.save(
        {"model": model.state_dict(), "config": {"model": "pi3", "variant": "v2"}}, path
    )
    restored = PURE.from_checkpoint(
        path, backbone="pi3", expected_sha256=hub.sha256(path)
    )
    assert restored.config.backbone == "pi3"
    with pytest.raises(ValueError, match="SHA256"):
        PURE.from_checkpoint(path, backbone="pi3", expected_sha256="bad")
    with pytest.raises(ValueError, match="metadata"):
        PURE.from_checkpoint(path, backbone="da3")
    state = model.state_dict()
    state.pop(next(iter(state)))
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        PURE.from_checkpoint(path, backbone="pi3")


def test_dtype_preserved_from_training_checkpoint(tmp_path, monkeypatch):
    import pure.hub as hub

    monkeypatch.setattr(hub, "create_model", lambda _: torch.nn.Linear(3, 2))
    state = torch.nn.Linear(3, 2).half().state_dict()
    path = tmp_path / "half.pt"
    torch.save(state, path)
    model = PURE.from_checkpoint(path, backbone="pi3")
    assert model.backbone.weight.dtype == torch.float16
    model.save_pretrained(tmp_path / "hub")
    reloaded = PURE.from_pretrained(tmp_path / "hub")
    assert reloaded.backbone.weight.dtype == torch.float16
    torch.testing.assert_close(
        reloaded.backbone.weight, state["weight"], rtol=0, atol=0
    )

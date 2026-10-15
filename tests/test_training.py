import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from examples.reference_training_smoke import TinyGeometry, make_synthetic_clip
from pure import SSLoss
from pure.training.cache import (
    ClipDataset,
    read_cache,
    to_batch,
    validate_cache,
    write_cache,
)
from pure.training.checkpoint import load_model, read_checkpoint
from pure.training.engine import SampleStream, fit, optimizer_step


@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)


def test_cache_roundtrip_and_synthetic_gate(tmp_path):
    arrays = make_synthetic_clip()
    path = tmp_path / "clip.npz"
    write_cache(path, arrays, synthetic=True)
    with pytest.raises(ValueError, match="Synthetic"):
        read_cache(path)
    actual = read_cache(path, allow_synthetic=True)
    for key in arrays:
        np.testing.assert_array_equal(actual[key], arrays[key])
    images, gt, roma = to_batch(actual)
    assert images.shape == (1, 2, 3, 28, 28)
    assert gt["distill_results"]["xy"].shape == (1, 2, 28, 28, 2)
    assert roma["dense_matches_forward"].shape == (1, 1, 28, 28, 4)


def test_cache_rejects_dynamic_matches_and_bad_shapes():
    arrays = make_synthetic_clip()
    arrays["dynamic_masks"][0, 0, 0] = True
    with pytest.raises(ValueError, match="dynamic"):
        validate_cache(arrays)
    arrays = make_synthetic_clip()
    arrays["teacher_xy"] = arrays["teacher_xy"][:, :-1]
    with pytest.raises(ValueError, match="teacher_xy"):
        validate_cache(arrays)


def test_actual_public_loss_updates_weights_and_rejects_nonfinite():
    model = TinyGeometry()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.ray_offset.detach().clone()
    torch.manual_seed(0)
    metrics = optimizer_step(
        model, SSLoss(), [make_synthetic_clip()], optimizer, device="cpu"
    )
    assert np.isfinite(metrics["loss"]) and np.isfinite(metrics["grad_norm"])
    assert not torch.equal(before, model.ray_offset.detach())
    # A bad forward must fail before any optimizer update.
    model.depth_offset.data.fill_(float("nan"))
    with pytest.raises((FloatingPointError, RuntimeError)):
        optimizer_step(
            model, SSLoss(), [make_synthetic_clip()], optimizer, device="cpu"
        )


def test_resume_restores_optimizer_rng_scheduler_and_stream(tmp_path, monkeypatch):
    write_cache(tmp_path / "data/a.npz", make_synthetic_clip(), synthetic=True)
    write_cache(tmp_path / "data/b.npz", make_synthetic_clip(), synthetic=True)
    dataset = ClipDataset(tmp_path / "data", allow_synthetic=True)
    config = {"steps": 3, "accumulation": 1, "lr": 1e-4, "seed": 7, "save_every": 1}
    full = TinyGeometry()
    final = fit(full, dataset, config, tmp_path / "full")
    import pure.training.engine as engine

    save = engine.torch.save

    def stop_after_first_step(state, path):
        save(state, path)
        if state["step"] == 1:
            Path(path).replace(Path(path).with_name("latest.pt"))
            raise InterruptedError("Simulated interruption after an optimizer step")

    with monkeypatch.context() as context:
        context.setattr(engine.torch, "save", stop_after_first_step)
        with pytest.raises(InterruptedError):
            fit(TinyGeometry(), dataset, config, tmp_path / "partial")
    resumed = TinyGeometry()
    resumed_path = fit(
        resumed,
        dataset,
        config,
        tmp_path / "partial",
        resume=tmp_path / "partial/latest.pt",
    )
    for name, value in full.state_dict().items():
        # Parallel FP32 reductions in the public loss can differ by a few ULPs.
        torch.testing.assert_close(
            resumed.state_dict()[name], value, rtol=1e-6, atol=1e-8
        )
    expected, actual = read_checkpoint(final), read_checkpoint(resumed_path)
    assert expected["step"] == actual["step"] == 3
    assert expected["scheduler"] == actual["scheduler"]
    assert torch.equal(expected["rng"], actual["rng"])
    assert expected["stream"]["order"] == actual["stream"]["order"]
    assert expected["stream"]["offset"] == actual["stream"]["offset"]
    assert torch.equal(expected["stream"]["generator"], actual["stream"]["generator"])


def test_strict_checkpoint_rejects_partial_weights(tmp_path):
    path = tmp_path / "incomplete.pt"
    torch.save({"depth_offset": torch.tensor(0.2)}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model(TinyGeometry(), path)


def test_sample_stream_repeats_after_restore():
    stream = SampleStream(4, 3)
    stream.next()
    state = copy.deepcopy(stream.state_dict())
    expected = [stream.next() for _ in range(10)]
    restored = SampleStream(4, 99)
    restored.load_state_dict(state)
    assert [restored.next() for _ in range(10)] == expected


@pytest.mark.parametrize("valid_pixels", [0, 1, 16])
@pytest.mark.parametrize("vary_prediction", [False, True])
def test_ordinal_loss_without_ordered_pairs_has_zero_gradient(valid_pixels, vary_prediction):
    pred = (torch.arange(16.) if vary_prediction else torch.ones(16)).requires_grad_()
    mask = torch.arange(16) < valid_pixels
    loss = SSLoss()._compute_single_tanh_loss(pred, torch.ones(16), mask, 16, 100)
    assert loss.item() == 0
    loss.backward()
    torch.testing.assert_close(pred.grad, torch.zeros_like(pred))


def test_constant_teacher_depth_can_train():
    arrays = make_synthetic_clip()
    arrays["depths"].fill(2)
    validate_cache(arrays)
    model = TinyGeometry()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    metrics = optimizer_step(model, SSLoss(), [arrays], optimizer, device="cpu")
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_nonconstant_ordinal_loss_preserves_original_formula():
    pred = torch.linspace(0.1, 0.9, 32, requires_grad=True)
    gt = torch.linspace(1, 4, 32)
    torch.manual_seed(5)
    first, second = torch.randperm(32), torch.randperm(32)
    keep = (gt[first] - gt[second]).abs() >= (gt.max() - gt.min()) / 10
    delta = gt[first[keep]] - gt[second[keep]]
    expected = (torch.tanh(100 * (pred[first[keep]] - pred[second[keep]]) / delta.abs()) - delta.sign()).abs().mean()
    torch.manual_seed(5)
    actual = SSLoss()._compute_single_tanh_loss(pred, gt, torch.ones(32), 32, 100)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_new_run_refuses_nonempty_output_without_changing_files(tmp_path):
    output = tmp_path / "run"
    output.mkdir()
    (output / "latest.pt").write_bytes(b"existing checkpoint")
    (output / "metrics.jsonl").write_text('{"step": 1}\n')
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    model = TinyGeometry()
    weights = copy.deepcopy(model.state_dict())
    config = {"steps": 1, "accumulation": 1, "lr": 1e-4, "seed": 7, "save_every": 1}
    with pytest.raises(ValueError, match="not empty"):
        fit(model, None, config, output)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    for key, value in weights.items():
        torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


def test_completed_resume_returns_existing_checkpoint_and_restores_model(tmp_path):
    write_cache(tmp_path / "data/a.npz", make_synthetic_clip(), synthetic=True)
    dataset = ClipDataset(tmp_path / "data", allow_synthetic=True)
    config = {"steps": 1, "accumulation": 1, "lr": 1e-4, "seed": 7, "save_every": 1}
    output = tmp_path / "full"
    output.mkdir()  # An existing empty directory is valid for a new run.
    trained = TinyGeometry()
    checkpoint = fit(trained, dataset, config, output)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    for destination in (output, tmp_path / "new"):
        restored = TinyGeometry()
        actual = fit(restored, dataset, config, destination, resume=checkpoint)
        assert actual.is_file() and actual == checkpoint
        for key, value in trained.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}
    assert not list((tmp_path / "new").iterdir())
    with pytest.raises(ValueError, match="not empty"):
        fit(TinyGeometry(), dataset, config, output)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "keep").write_text("existing run")
    with pytest.raises(ValueError, match="not empty"):
        fit(TinyGeometry(), dataset, config, unrelated, resume=checkpoint)
    state = read_checkpoint(checkpoint)
    for step in (-1, 2):
        state["step"] = step
        torch.save(state, tmp_path / "invalid.pt")
        with pytest.raises(ValueError, match="Resume step"):
            fit(TinyGeometry(), dataset, config, tmp_path / "invalid-output", resume=tmp_path / "invalid.pt")

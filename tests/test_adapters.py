import json
from pathlib import Path
import pytest
import torch
from torch import nn
from pure.models import create_model
from pure.models.geometry import depth_to_predictions, invert_pose
from pure.training.policy import configure_model
from pure.training.loading import load_weights
from pure.training.engine import optimizer_step
from pure import SSLoss
from examples.reference_training_smoke import make_synthetic_clip


def small_da3():
    from pure.models.da3 import DA3
    from pure._vendor.da3.model.dinov2.vision_transformer import DinoVisionTransformer
    from pure._vendor.da3.model.dualdpt import DualDPT
    from pure._vendor.da3.model.cam_dec import CameraDec

    class Backbone(nn.Module):
        def __init__(self):
            super().__init__()
            self.pretrained = DinoVisionTransformer(
                img_size=28,
                patch_size=14,
                embed_dim=16,
                depth=40,
                num_heads=2,
                block_chunks=0,
                alt_start=13,
                qknorm_start=13,
                rope_start=13,
                cat_token=True,
            )

        def forward(self, x, **kwargs):
            return self.pretrained.get_intermediate_layers(
                x, [19, 27, 33, 39], **kwargs
            )

    camera = CameraDec(dim_in=32)
    with torch.no_grad():
        camera.fc_fov[0].bias.fill_(1.0)
        camera.fc_qvec.bias.fill_(0.5)
    return DA3(
        backbone=Backbone(),
        head=DualDPT(dim_in=32, features=16, out_channels=[16, 16, 32, 32]),
        cam_dec=camera,
    )


def small_vggt():
    from pure.models.vggt import VGGT
    from pure._vendor.vggt.models.aggregator import Aggregator
    from pure._vendor.vggt.heads.camera_head import CameraHead
    from pure._vendor.vggt.heads.dpt_head import DPTHead

    camera = CameraHead(dim_in=32, num_heads=2, trunk_depth=1)
    with torch.no_grad():
        camera.pose_branch.fc2.bias[-2:].fill_(1.0)
        camera.pose_branch.fc2.bias[3:7].fill_(0.5)
    return VGGT(
        aggregator=Aggregator(
            img_size=28, embed_dim=16, depth=24, num_heads=2, patch_embed="conv"
        ),
        camera_head=camera,
        depth_head=DPTHead(
            dim_in=32,
            features=16,
            out_channels=[16, 16, 32, 32],
            output_dim=2,
            activation="exp",
            conf_activation="expp1",
        ),
    )


@pytest.mark.parametrize("name,count", [("da3", 847), ("vggt", 1341)])
def test_full_checkpoint_architecture_matches(name, count):
    with torch.device("meta"):
        model = create_model(name)
    expected = json.loads(
        (Path(__file__).parents[1] / f"weights/{name}-shapes.json").read_text()
    )
    assert len(expected) == count
    assert {k: list(v.shape) for k, v in model.state_dict().items()} == expected
    config = json.loads(
        (Path(__file__).parents[1] / f"configs/{name}_posttrain.json").read_text()
    )
    names = configure_model(model, config)
    prefix = (
        "backbone.pretrained.blocks." if name == "da3" else "aggregator.global_blocks."
    )
    assert all(n.startswith(prefix) for n in names)
    assert {int(n[len(prefix) :].split(".")[0]) for n in names} == set(
        config["train_blocks"]
    )


def test_geometry_projection_pose_inverse_and_gradients():
    depth = torch.full((1, 2, 28, 42), 2.0, requires_grad=True)
    k = (
        torch.tensor([[50.0, 3.0, 21.0], [0.0, 60.0, 14.0], [0.0, 0.0, 1.0]])
        .repeat(1, 2, 1, 1)
        .requires_grad_()
    )
    pose = torch.eye(4).repeat(1, 2, 1, 1)
    # nonidentity rotation + translation catch c2w/w2c confusion
    pose[0, 1, :3, :3] = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
    )
    pose[0, 1, :3, 3] = torch.tensor([1.0, 2.0, 3.0])
    pose.requires_grad_()
    result = depth_to_predictions(depth, k, pose)
    world = result["points"]
    inv = invert_pose(pose)
    local = (
        torch.einsum("bsij,bshwj->bshwi", inv[..., :3, :3], world)
        + inv[..., :3, 3][..., None, None, :]
    )
    torch.testing.assert_close(local, result["local_points"])
    pixel = torch.einsum("bsij,bshwj->bshwi", k, local)
    xy = pixel[..., :2] / pixel[..., 2:3]
    torch.testing.assert_close(xy[0, 0, 13, 17], torch.tensor([17.0, 13.0]))
    world.square().mean().backward()
    for tensor in (depth, k, pose):
        assert (
            tensor.grad is not None
            and torch.isfinite(tensor.grad).all()
            and tensor.grad.abs().sum() > 0
        )


@pytest.mark.parametrize("name,factory", [("da3", small_da3), ("vggt", small_vggt)])
def test_real_modules_forward_public_loss_backward_and_strict_reload(
    name, factory, tmp_path
):
    torch.set_num_threads(2)
    torch.manual_seed(8)
    model = factory().train()
    config = {
        "model": name,
        "stage": "posttrain",
        "train_blocks": [19, 27, 33, 39] if name == "da3" else [4, 11, 17, 23],
    }
    configure_model(model, config)
    images = torch.rand(1, 2, 3, 28, 28)
    prediction = model(images)
    assert prediction["depth"].shape == (1, 2, 28, 28, 1)
    assert prediction["camera_poses"].shape == (1, 2, 4, 4)
    assert torch.isfinite(prediction["points"]).all()
    target = next(p for p in model.parameters() if p.requires_grad)
    before = target.detach().clone()
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    metrics = optimizer_step(
        model, SSLoss(), [make_synthetic_clip()], optim, device="cpu"
    )
    assert metrics["grad_norm"] > 0
    assert not torch.equal(before, target)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    path = tmp_path / "weights.pt"
    torch.save(model.state_dict(), path)
    restored = factory()
    load_weights(restored, path, name)
    restored.eval()
    model.eval()
    with torch.no_grad():
        torch.testing.assert_close(restored(images)["points"], model(images)["points"])
    state = model.state_dict()
    state.pop(next(iter(state)))
    torch.save(state, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_weights(restored, path, name)


@pytest.mark.parametrize("factory", [small_da3, small_vggt])
def test_autocast_keeps_geometry_float32_and_gradients_finite(factory):
    torch.set_num_threads(2)
    torch.manual_seed(8)
    model = factory().train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(torch.rand(1, 2, 3, 28, 28))
    assert result["points"].dtype == torch.float32
    assert torch.isfinite(result["points"]).all()
    result["points"].square().mean().backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)

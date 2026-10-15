import pytest
import torch

from pure.models.pi3 import Pi3
from pure.models.pi3.layers.attention import FlashAttentionRope
from pure.models.pi3.layers.block import BlockRope
from pure.models.pi3.layers.camera_head import CameraHead
from pure.models.pi3.layers.transformer_head import Head, TransformerDecoder
from pure.training.engine import configure_trainable


def small_model():
    """Exercise the recovered forward with real modules and a reduced width."""
    with torch.device("meta"):
        model = Pi3()
    width = 8

    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, width, 14, 14)

        def forward(self, images, is_training=True):
            return {"x_norm_patchtokens": self.conv(images).flatten(2).transpose(1, 2)}

    model.encoder = Encoder()
    model.dec_embed_dim = width
    model.decoder = torch.nn.ModuleList(
        [
            BlockRope(
                dim=width, num_heads=2, attn_class=FlashAttentionRope, rope=model.rope
            )
            for _ in range(36)
        ]
    )
    model.register_token = torch.nn.Parameter(torch.randn(1, 1, 5, width) * 1e-6)
    model.point_decoder = TransformerDecoder(
        2 * width, width, width, depth=1, dec_num_heads=2, rope=model.rope
    )
    model.depth_refiner = Head(
        6, width, [3], dim_proj=32, dim_upsample=[32], last_conv_channels=32
    )
    model.camera_decoder = TransformerDecoder(
        2 * width, width, width, depth=1, dec_num_heads=2, rope=model.rope
    )
    model.camera_head = CameraHead(dim=width)
    model.image_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    model.image_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return model


def test_recovered_forward_and_backward_with_reduced_width():
    torch.set_num_threads(2)
    torch.manual_seed(1)
    model = small_model()
    prediction = model(torch.rand(1, 2, 3, 28, 28))
    assert prediction["local_points"].shape == (1, 2, 28, 28, 3)
    assert prediction["camera_poses"].shape == (1, 2, 4, 4)
    assert torch.isfinite(prediction["points"]).all()
    assert (prediction["local_points"][..., 2] > 0).all()
    prediction["points"].square().mean().backward()
    assert any(p.grad is not None for p in model.depth_refiner.parameters())
    assert any(p.grad is not None for p in model.point_decoder.parameters())


def test_parameter_count_and_explicit_freeze_policy():
    with torch.device("meta"):
        model = Pi3()
    assert sum(p.numel() for p in model.parameters()) == 923979951
    head = configure_trainable(model, "head_init")
    assert all(name.startswith("depth_refiner.") for name in head)
    selected = configure_trainable(model, "posttrain", train_head=False)
    assert all(
        name.startswith("decoder.") and int(name.split(".")[1]) % 2 for name in selected
    )
    joint = configure_trainable(model, "posttrain", train_head=True)
    assert any(name.startswith("depth_refiner.") for name in joint)
    assert not any(
        name.startswith(
            ("encoder.", "camera_decoder.", "camera_head.", "point_decoder.")
        )
        for name in joint
    )


def test_legacy_v2_config_compatibility_and_retired_layout_rejection():
    from pure.training.checkpoint import normalize_model_config

    assert normalize_model_config({"model": "pi3", "variant": "v2", "steps": 2}) == {
        "model": "pi3",
        "steps": 2,
    }
    with pytest.raises(ValueError, match="only Pi3 v2"):
        normalize_model_config({"variant": "v1"})

import json
from pathlib import Path

import torch

from pure.models.pi3 import Pi3


def test_surviving_pi3_checkpoint_shapes_match_v2_exactly():
    root = Path(__file__).resolve().parents[1]
    shapes = json.loads((root / "weights/pi3-shapes.json").read_text())
    with torch.device("meta"):
        model = Pi3()
    expected = {key: list(tensor.shape) for key, tensor in model.state_dict().items()}
    assert expected == shapes
    assert len(expected) == 1238

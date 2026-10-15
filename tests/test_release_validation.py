import pytest
import torch

from scripts.eval.validate_release import compare_outputs


@pytest.mark.parametrize("bad", [float("inf"), -float("inf"), float("nan")])
@pytest.mark.parametrize("side", ["original", "converted", "both"])
def test_parity_rejects_nonfinite(bad, side):
    before = torch.tensor([bad if side != "converted" else 1.0])
    after = torch.tensor([bad if side != "original" else 1.0])
    with pytest.raises(ValueError, match="non-finite"):
        compare_outputs({"depth": before}, {"depth": after})


def test_parity_requires_same_nonempty_outputs():
    value = torch.ones(2)
    assert compare_outputs({"depth": value}, {"depth": value.clone()}) == {"depth": 0.0}
    for before, after in [({}, {}), ({"depth": value}, {}),
                          ({"depth": value}, {"depth": value, "points": value})]:
        with pytest.raises(ValueError, match="keys"):
            compare_outputs(before, after)
    with pytest.raises(ValueError, match="nonempty tensor"):
        compare_outputs({"depth": torch.empty(0)}, {"depth": torch.empty(0)})
    for after in (torch.ones(1, 2), value + 1):
        with pytest.raises(AssertionError):
            compare_outputs({"depth": value}, {"depth": after})

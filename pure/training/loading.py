"""Strict refined-weight loading and explicit official-base subset loading."""

from .checkpoint import load_model, model_state, read_checkpoint


def load_weights(model, path, name, *, base=False):
    if name == "pi3":
        return load_model(model, path, base=base)
    state = model_state(read_checkpoint(path))
    if not base:
        model.load_state_dict(state, strict=True)
        return {"missing": [], "unused": []}
    # Official DA3 APIs often save the network under model.*.
    if name == "da3" and state and all(k.startswith("model.") for k in state):
        state = {k[6:]: v for k, v in state.items()}
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    extra = sorted(set(state) - set(expected))
    allowed = (
        ("cam_enc.", "gs_head.", "gs_adapter.")
        if name == "da3"
        else ("point_head.", "track_head.")
    )
    if missing or any(not k.startswith(allowed) for k in extra):
        raise ValueError(
            f"Base {name} incompatible: missing={missing[:8]}, unused={extra[:8]}"
        )
    model.load_state_dict(
        {k: v for k, v in state.items() if k in expected}, strict=True
    )
    return {"missing": [], "unused": extra}

"""PURE: unlabeled post-training losses for feed-forward 3D reconstruction."""

from .loss import SSLoss, run_roma

__all__ = ["SSLoss", "run_roma"]


def __getattr__(name):
    if name in {"PURE", "PUREConfig"}:
        from .hub import PURE, PUREConfig

        return {"PURE": PURE, "PUREConfig": PUREConfig}[name]
    if name == "PUREImageProcessor":
        from .processing import PUREImageProcessor

        return PUREImageProcessor
    raise AttributeError(name)


__all__ += ["PURE", "PUREConfig", "PUREImageProcessor"]

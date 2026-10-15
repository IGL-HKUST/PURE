"""Model registry. Imports are lazy to keep objective-only installations small."""


def create_model(name):
    if name == "pi3":
        from .pi3 import Pi3

        return Pi3()
    if name == "da3":
        from .da3 import DA3

        return DA3()
    if name == "vggt":
        from .vggt import VGGT

        return VGGT()
    raise ValueError(f"Unknown model {name!r}; choose pi3, da3, or vggt")

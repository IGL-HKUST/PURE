"""Check full model topology against surviving checkpoint metadata, without weights."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from pure.models import create_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["pi3", "da3", "vggt", "all"], default="all")
    args = parser.parse_args()
    for name in ["pi3", "da3", "vggt"] if args.model == "all" else [args.model]:
        with torch.device("meta"):
            model = create_model(name)
        shapes = {k: list(v.shape) for k, v in model.state_dict().items()}
        expected = json.loads(
            (
                Path(__file__).resolve().parents[2] / f"weights/{name}-shapes.json"
            ).read_text()
        )
        missing = sorted(set(expected) - set(shapes))
        extra = sorted(set(shapes) - set(expected))
        mismatch = [
            k for k in shapes.keys() & expected.keys() if shapes[k] != expected[k]
        ]
        print(
            json.dumps(
                dict(
                    model=name,
                    tensors=len(shapes),
                    missing=missing,
                    extra=extra,
                    shape_mismatches=mismatch,
                )
            )
        )
        if missing or extra or mismatch:
            raise SystemExit(1)


if __name__ == "__main__":
    main()

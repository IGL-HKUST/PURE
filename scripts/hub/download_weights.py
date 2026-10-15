"""Download pinned refined weights with resume, size and SHA256 verification."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=["pi3", "da3", "vggt", "all"], required=True)
    p.add_argument("--output", default="checkpoints")
    args = p.parse_args()
    manifest = json.loads(
        (Path(__file__).resolve().parents[2] / "weights/manifest.json").read_text()
    )
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    for name in ["pi3", "da3", "vggt"] if args.model == "all" else [args.model]:
        info = manifest["files"][name + ".bin"]
        path = root / (name + ".bin")
        if path.exists():
            if (
                path.stat().st_size == info["bytes"]
                and digest(path) == info["lfs_sha256"]
            ):
                print(f"Already verified: {path}")
                continue
            raise ValueError(
                f"{path} exists with different content; choose another output directory"
            )
        partial = path.with_suffix(".bin.part")
        subprocess.run(
            [
                "curl",
                "--fail",
                "--location",
                "--retry",
                "3",
                "--continue-at",
                "-",
                "--output",
                str(partial),
                info["download_url"],
            ],
            check=True,
        )
        if (
            partial.stat().st_size != info["bytes"]
            or digest(partial) != info["lfs_sha256"]
        ):
            raise ValueError(
                f"Size or SHA256 mismatch: {partial}; remove the partial file before retrying"
            )
        partial.replace(path)
        print(f"Verified: {path}")


if __name__ == "__main__":
    main()

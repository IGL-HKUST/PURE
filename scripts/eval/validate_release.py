"""GPU parity and complete Gradio callback smoke tests for converted weights."""

from pathlib import Path
import argparse, gc, json, os, sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from pure import PURE, PUREImageProcessor


def compare_outputs(before, after):
    """Require matching, nonempty, finite outputs before reporting parity."""
    if not before or before.keys() != after.keys():
        raise ValueError("Parity requires identical, nonempty output keys")
    errors = {}
    for key in before:
        left, right = before[key], after[key]
        for label, value in (("original", left), ("converted", right)):
            if not isinstance(value, torch.Tensor) or value.numel() == 0:
                raise ValueError(f"{key}: {label} output must be a nonempty tensor")
            if not torch.isfinite(value).all().item():
                raise ValueError(f"{key}: {label} output contains non-finite values")
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-5)
        errors[key] = float((left.double() - right.double()).abs().max())
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--model-root", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--sky-model-directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.images.glob("*.png"))
    assert len(paths) == 48
    images = PUREImageProcessor().images(paths[:2]).unsqueeze(0).cuda()
    records = []
    for name in ["pi3", "da3", "vggt"]:
        print("PARITY_START", name, flush=True)
        original = PURE.from_checkpoint(
            args.checkpoint_dir / (name + ".bin"), backbone=name
        ).cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            before = {
                k: v.float().cpu()
                for k, v in original(images).items()
                if isinstance(v, torch.Tensor)
            }
        del original
        gc.collect()
        torch.cuda.empty_cache()
        model = PURE.from_pretrained(args.model_root, subfolder=name, device="cuda")
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            after = {
                k: v.float().cpu()
                for k, v in model(images).items()
                if isinstance(v, torch.Tensor)
            }
        errors = compare_outputs(before, after)
        record = {"model": name, "parity": "passed", "max_abs_errors": errors}
        records.append(record)
        print(json.dumps(record, allow_nan=False), flush=True)
        del model, before, after
        gc.collect()
        torch.cuda.empty_cache()
    os.environ["PURE_MODEL_ROOT"] = str(args.model_root)
    os.environ["PURE_OUTPUT_DIR"] = str(args.output / "studio")
    os.environ["PURE_SKY_MODEL_DIR"] = str(args.sky_model_directory)
    import app

    for record in records:
        name = record["model"]
        print("APP_START", name, flush=True)
        result = app.reconstruct(
            "Images", None, [str(x) for x in paths], name, 48, 0, True, False
        )
        directory = Path(result[3]).parent
        meta = json.loads((directory / "metadata.json").read_text())
        assert meta["frame_count"] == 48
        assert meta["remove_sky"] and meta["sky_pixels_removed"] > 0
        for item in result[:6]:
            assert Path(item).is_file() and Path(item).stat().st_size > 0
        import trimesh

        scene = trimesh.load(result[2])
        preview_count = sum(len(x.vertices) for x in scene.geometry.values())
        assert 0 < preview_count <= 150000
        record["app"] = {
            "status": "passed",
            "frames": 48,
            "exports": {Path(x).name: Path(x).stat().st_size for x in result[:6]},
            "sky_pixels_removed": meta["sky_pixels_removed"],
            "exported_points": meta["exported_points"],
            "preview_points": preview_count,
            "output": str(directory),
        }
        (args.output / "gpu-validation.json").write_text(
            json.dumps(records, indent=2, allow_nan=False) + "\n"
        )
        print(json.dumps(record, allow_nan=False), flush=True)
    print("GPU_RELEASE_VALIDATION_PASSED", flush=True)


if __name__ == "__main__":
    main()

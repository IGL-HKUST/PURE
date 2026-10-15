"""Video or image sequence -> depth, camera poses, PLY and interactive GLB."""

from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pure import PURE, PUREImageProcessor
from pure.pipeline import ReconstructionPipeline


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--video")
    source.add_argument("--images", nargs="+", help="Images in temporal order")
    weights = p.add_mutually_exclusive_group(required=True)
    weights.add_argument("--pretrained", help="HF model ID or local directory")
    weights.add_argument(
        "--legacy-model",
        choices=["pi3", "da3", "vggt"],
        help="Download the pinned original public .bin checkpoint",
    )
    p.add_argument("--subfolder", default="")
    p.add_argument("--revision")
    p.add_argument("--frames", type=int, default=48)
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=["auto", "fp32", "bf16"], default="auto")
    p.add_argument("--keep-sky", action="store_true")
    p.add_argument("--save-raw", action="store_true")
    p.add_argument("--sky-model-directory")
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        p.error("Choose an empty output directory")
    if args.pretrained:
        model = PURE.from_pretrained(
            args.pretrained, subfolder=args.subfolder, revision=args.revision
        )
    else:
        model = PURE.from_legacy_hub(args.legacy_model)
    processor = PUREImageProcessor(*model.config.image_size)
    images, metadata = (
        processor.video(
            args.video,
            max_frames=args.frames,
            start_frame=args.start_frame,
            return_metadata=True,
        )
        if args.video
        else processor.images(args.images, max_frames=args.frames, return_metadata=True)
    )
    result = ReconstructionPipeline(
        model, device=args.device, sky_model_directory=args.sky_model_directory
    )(
        images,
        metadata,
        args.output,
        remove_sky=not args.keep_sky,
        precision=args.precision,
        save_raw=args.save_raw,
    )
    print(result["archive"])


if __name__ == "__main__":
    main()

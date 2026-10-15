"""Pack aligned RGB frames and upstream teacher outputs into a validated clip."""

import argparse


import numpy as np
from pure.images import load_images
from pure.training.cache import ARRAY_KEYS, write_cache


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images",
        nargs="+",
        required=True,
        help="Frame paths in temporal order; no implicit resizing",
    )
    parser.add_argument(
        "--labels",
        required=True,
        help="NPZ containing the aligned depth/ray/mask/match arrays",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--teacher-prediction",
        help="Optional infer.py NPZ from the frozen base; provides teacher_xy",
    )
    args = parser.parse_args()
    images = load_images(args.images)
    with np.load(args.labels, allow_pickle=False) as archive:
        arrays = {
            key: archive[key].copy()
            for key in ARRAY_KEYS
            if key != "images" and not (key == "teacher_xy" and args.teacher_prediction)
        }
    if args.teacher_prediction:
        with np.load(args.teacher_prediction, allow_pickle=False) as pred:
            arrays["teacher_xy"] = pred["xy"].copy()
    arrays["images"] = images
    write_cache(args.output, arrays)
    print(f"Validated clip saved: {args.output}")


if __name__ == "__main__":
    main()

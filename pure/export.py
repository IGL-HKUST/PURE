"""Export full filtered PLY, camera data, and bounded browser previews."""

from fractions import Fraction
from pathlib import Path
import json
import zipfile
import numpy as np

VERTEX_DTYPE = np.dtype(
    [(k, "<f4") for k in ("x", "y", "z")]
    + [(k, "u1") for k in ("red", "green", "blue")]
)


def write_ply(path, points, colors):
    with Path(path).open("wb") as f:
        f.write(
            (
                "ply\nformat binary_little_endian 1.0\n"
                + f"element vertex {len(points)}\n"
                + "".join(f"property float {k}\n" for k in ("x", "y", "z"))
                + "".join(f"property uchar {k}\n" for k in ("red", "green", "blue"))
                + "end_header\n"
            ).encode()
        )
        for start in range(0, len(points), 250000):
            xyz = points[start : start + 250000]
            rgb = colors[start : start + 250000]
            vertices = np.empty(len(xyz), dtype=VERTEX_DTYPE)
            for i, key in enumerate(("x", "y", "z")):
                vertices[key] = xyz[:, i]
            for i, key in enumerate(("red", "green", "blue")):
                vertices[key] = rgb[:, i]
            vertices.tofile(f)


def write_video(path, frames, fps):
    import av

    rate = Fraction(float(np.clip(fps, 1, 120))).limit_denominator(1001)
    with av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=rate)
        stream.width = frames.shape[2]
        stream.height = frames.shape[1]
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "fast"}
        for image in frames:
            for packet in stream.encode(
                av.VideoFrame.from_ndarray(image, format="rgb24")
            ):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def export_reconstruction(
    predictions,
    images,
    sky,
    directory,
    *,
    metadata,
    save_raw=False,
    preview_points=150000,
):
    import trimesh
    from matplotlib import colormaps

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    points = predictions["points"]
    depth = predictions["depth"][..., 0]
    poses = predictions["camera_poses"]
    if points.shape != sky.shape + (3,) or depth.shape != sky.shape:
        raise ValueError("Sky mask and geometry shapes disagree")
    if not all(np.isfinite(v).all() for v in predictions.values()):
        raise FloatingPointError("Non-finite model output")
    valid = np.isfinite(points).all(-1) & np.isfinite(depth) & (depth > 0) & ~sky
    if not valid.any():
        raise ValueError("No non-sky positive-depth points remain")
    rgb = (images.transpose(0, 2, 3, 1) * 255).clip(0, 255).astype(np.uint8)
    xyz = points[valid]
    colors = rgb[valid]
    ply = (
        directory / "points_no_sky.ply"
        if metadata["remove_sky"]
        else directory / "points.ply"
    )
    write_ply(ply, xyz, colors)
    low, high = np.percentile(depth[valid], [1, 99])
    scaled = (depth - low) / max(float(high - low), 1e-8)
    depth_rgb = (colormaps["viridis"](scaled.clip(0, 1))[..., :3] * 255).astype(
        np.uint8
    )
    depth_rgb[sky] = 0
    write_video(directory / "input.mp4", rgb, metadata["preview_fps"])
    write_video(directory / "depth.mp4", depth_rgb, metadata["preview_fps"])
    # Preview only: p99 cutoff and deterministic uniform sampling. PLY is unaffected.
    ids = np.flatnonzero(valid.ravel() & (depth.ravel() <= high))
    if len(ids) > preview_points:
        ids = ids[np.linspace(0, len(ids) - 1, preview_points).astype(int)]
    scale = float(np.median(depth[valid]))
    first = poses[0]
    display = (points.reshape(-1, 3)[ids] - first[:3, 3]) @ first[:3, :3] / scale
    display *= np.array([1, -1, -1])  # OpenCV -> glTF Y-up, front-facing camera-0 view.
    cloud = trimesh.points.PointCloud(display, colors=rgb.reshape(-1, 3)[ids])
    (directory / "preview.glb").write_bytes(
        trimesh.Scene(cloud).export(file_type="glb")
    )
    np.savez_compressed(
        directory / "cameras.npz",
        camera_poses=poses,
        **(
            {"intrinsics": predictions["intrinsics"]}
            if "intrinsics" in predictions
            else {}
        ),
    )
    np.savetxt(
        directory / "camera_poses.txt",
        poses.reshape(len(poses), 16),
        header="camera-to-world; row-major; native model units",
    )
    np.savez_compressed(directory / "sky_masks.npz", sky=sky)
    selected = np.linspace(0, len(rgb) - 1, min(4, len(rgb))).round().astype(int)
    from PIL import Image

    overlays = []
    for i in selected:
        overlay = rgb[i].copy()
        overlay[sky[i]] = (
            0.4 * overlay[sky[i]] + 0.6 * np.array([255, 0, 120])
        ).astype(np.uint8)
        path = directory / f"mask-{i:03d}.jpg"
        Image.fromarray(overlay).save(path, quality=90)
        overlays.append(str(path))
    info = {
        **metadata,
        "exported_points": int(valid.sum()),
        "sky_pixels_removed": int(sky.sum()),
        "sky_fraction": float(sky.mean()),
        "pointcloud_filter": "finite positive-depth points and optional semantic sky removal; no confidence filter",
        "preview_filter": "non-sky depth <= p99, at most 150k points, camera-0 coordinates / median depth; OpenCV to glTF axes",
        "depth_colors": "clip-wide p1-p99, sky black",
        "units": "native relative model scale; not guaranteed metric",
    }
    (directory / "metadata.json").write_text(json.dumps(info, indent=2) + "\n")
    if save_raw:
        np.savez_compressed(directory / "prediction.npz", images=images, **predictions)
    archive = directory / "reconstruction.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=3) as z:
        for file in sorted(directory.iterdir()):
            if file.is_file() and file != archive:
                z.write(file, file.name)
    return {
        "rgb": str(directory / "input.mp4"),
        "depth": str(directory / "depth.mp4"),
        "preview": str(directory / "preview.glb"),
        "ply": str(ply),
        "cameras": str(directory / "cameras.npz"),
        "archive": str(archive),
        "masks": overlays,
        "metadata": info,
    }

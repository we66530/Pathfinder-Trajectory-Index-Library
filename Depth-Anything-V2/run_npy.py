import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch

from depth_anything_v2.dpt import DepthAnythingV2


def make_safe_output_name(filename: str, all_filenames: list) -> str:
    """
    Create a safe output base name.

    Problem:
    left/frame_000001.png and right/frame_000001.png have the same basename.
    If we only use frame_000001, outputs will overwrite each other.

    Solution:
    If duplicate basenames exist, add parent folder name as prefix:
    left_frame_000001
    right_frame_000001
    """
    base = os.path.splitext(os.path.basename(filename))[0]

    all_bases = [os.path.splitext(os.path.basename(f))[0] for f in all_filenames]

    if all_bases.count(base) <= 1:
        return base

    parent = os.path.basename(os.path.dirname(filename))
    parent = parent.replace(" ", "_")

    if parent:
        return f"{parent}_{base}"

    return base


def load_filenames(img_path: str):
    """
    Load image filenames from:
    1. Single image file
    2. txt file containing image paths
    3. Folder path
    """
    if os.path.isfile(img_path):
        if img_path.lower().endswith(".txt"):
            with open(img_path, "r", encoding="utf-8") as f:
                filenames = f.read().splitlines()

            filenames = [x.strip().strip('"') for x in filenames if x.strip()]
        else:
            filenames = [img_path]
    else:
        # Recursive image search.
        all_files = glob.glob(os.path.join(img_path, "**/*"), recursive=True)

        valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
        filenames = [
            f for f in all_files
            if os.path.splitext(f)[1].lower() in valid_exts
        ]

    return filenames


def normalize_depth_for_vis(depth_float: np.ndarray) -> np.ndarray:
    """
    Normalize raw relative depth into uint8 0-255 visualization.
    """
    d_min = float(depth_float.min())
    d_max = float(depth_float.max())

    depth_vis = (depth_float - d_min) / (d_max - d_min + 1e-8) * 255.0
    depth_vis = depth_vis.astype(np.uint8)

    return depth_vis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Depth Anything V2 with .npy relative depth output")

    parser.add_argument(
        "--img-path",
        type=str,
        required=True,
        help="Path to an image, a folder, or a txt file containing image paths."
    )

    parser.add_argument(
        "--input-size",
        type=int,
        default=518,
        help="Input size for Depth Anything V2."
    )

    parser.add_argument(
        "--outdir",
        type=str,
        default="./vis_depth",
        help="Output folder."
    )

    parser.add_argument(
        "--encoder",
        type=str,
        default="vitl",
        choices=["vits", "vitb", "vitl", "vitg"],
        help="Depth Anything V2 encoder type."
    )

    parser.add_argument(
        "--pred-only",
        dest="pred_only",
        action="store_true",
        help="Only save the prediction visualization instead of original + depth side-by-side."
    )

    parser.add_argument(
        "--grayscale",
        dest="grayscale",
        action="store_true",
        help="Save visualization as grayscale instead of colorful palette."
    )

    parser.add_argument(
        "--save-norm-npy",
        dest="save_norm_npy",
        action="store_true",
        help="Also save normalized 0-255 depth as uint8 .npy. Raw relative depth .npy is always saved."
    )

    args = parser.parse_args()

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    print(f"[INFO] Device: {DEVICE}")
    print(f"[INFO] Encoder: {args.encoder}")

    model_configs = {
        "vits": {
            "encoder": "vits",
            "features": 64,
            "out_channels": [48, 96, 192, 384],
        },
        "vitb": {
            "encoder": "vitb",
            "features": 128,
            "out_channels": [96, 192, 384, 768],
        },
        "vitl": {
            "encoder": "vitl",
            "features": 256,
            "out_channels": [256, 512, 1024, 1024],
        },
        "vitg": {
            "encoder": "vitg",
            "features": 384,
            "out_channels": [1536, 1536, 1536, 1536],
        },
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])

    checkpoint_path = f"checkpoints/depth_anything_v2_{args.encoder}.pth"

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}\n"
            f"Please make sure the checkpoint exists in the checkpoints folder."
        )

    depth_anything.load_state_dict(
        torch.load(checkpoint_path, map_location="cpu")
    )

    depth_anything = depth_anything.to(DEVICE).eval()

    filenames = load_filenames(args.img_path)

    if len(filenames) == 0:
        raise RuntimeError(f"No image files found from: {args.img_path}")

    os.makedirs(args.outdir, exist_ok=True)

    cmap = matplotlib.colormaps.get_cmap("Spectral_r")

    print(f"[INFO] Total images: {len(filenames)}")
    print(f"[INFO] Output folder: {args.outdir}")

    for k, filename in enumerate(filenames):
        print(f"\n[INFO] Progress {k + 1}/{len(filenames)}: {filename}")

        raw_image = cv2.imread(filename)

        if raw_image is None:
            print(f"[WARN] Failed to read image, skipped: {filename}")
            continue

        out_base = make_safe_output_name(filename, filenames)

        # ------------------------------------------------------------
        # 1. Infer raw relative depth.
        # ------------------------------------------------------------
        with torch.no_grad():
            depth_float = depth_anything.infer_image(raw_image, args.input_size)

        depth_float = depth_float.astype(np.float32)

        # ------------------------------------------------------------
        # 2. Save raw relative depth as .npy.
        #    This is the important file for later scale fitting.
        # ------------------------------------------------------------
        relative_npy_path = os.path.join(
            args.outdir,
            out_base + "_relative_depth.npy"
        )

        np.save(relative_npy_path, depth_float)

        print(f"[INFO] Saved raw relative depth:")
        print(f"       {relative_npy_path}")
        print(f"[INFO] depth shape: {depth_float.shape}, dtype: {depth_float.dtype}")
        print(
            f"[INFO] depth range: "
            f"min={float(depth_float.min()):.6f}, "
            f"max={float(depth_float.max()):.6f}, "
            f"mean={float(depth_float.mean()):.6f}"
        )

        # ------------------------------------------------------------
        # 3. Normalize only for visualization.
        # ------------------------------------------------------------
        depth_vis = normalize_depth_for_vis(depth_float)

        if args.save_norm_npy:
            norm_npy_path = os.path.join(
                args.outdir,
                out_base + "_normalized_0to255.npy"
            )
            np.save(norm_npy_path, depth_vis)
            print(f"[INFO] Saved normalized uint8 depth:")
            print(f"       {norm_npy_path}")

        # ------------------------------------------------------------
        # 4. Convert visualization to grayscale or color.
        # ------------------------------------------------------------
        if args.grayscale:
            depth_vis_3ch = np.repeat(depth_vis[..., np.newaxis], 3, axis=-1)
        else:
            # cmap expects RGB-like indexing; convert to BGR for cv2.imwrite.
            depth_vis_3ch = (cmap(depth_vis)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)

        # ------------------------------------------------------------
        # 5. Save visualization PNG.
        # ------------------------------------------------------------
        out_png_path = os.path.join(args.outdir, out_base + ".png")

        if args.pred_only:
            cv2.imwrite(out_png_path, depth_vis_3ch)
        else:
            split_region = np.ones(
                (raw_image.shape[0], 50, 3),
                dtype=np.uint8
            ) * 255

            combined_result = cv2.hconcat([
                raw_image,
                split_region,
                depth_vis_3ch,
            ])

            cv2.imwrite(out_png_path, combined_result)

        print(f"[INFO] Saved visualization:")
        print(f"       {out_png_path}")

    print("\n[DONE]")

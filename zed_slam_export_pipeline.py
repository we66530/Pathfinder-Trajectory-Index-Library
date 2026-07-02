#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
zed_slam_export_pipeline.py

One-pass ZED SVO/SVO2 export pipeline:
  1) Build fused SLAM point cloud:        SLAM.ply
  2) Export trajectory nodes OBJ:         trajectory_nodes.obj
  3) Export per-frame CSV:                frame_camera_imu.csv
     - rectified camera intrinsics
     - raw camera intrinsics
     - image paths
     - IMU data closest to each image timestamp
     - positional tracking pose matrix for every frame if available
  4) Export corrected camera frustums OBJ: camera_poses_corrected.obj
  5) Export CloudCompare-friendly trajectory PLY files:
        trajectory_nodes_with_frame_scalar.ply
        trajectory_frame10_markers.ply
  6) Export left/right rectified images:
        left_rectified/frame_000000.png
        right_rectified/frame_000000.png

Default output folder:
  C:\Users\User\Desktop\SeekAndHide\ZED_SLAM\output

Typical use on Windows:
  python zed_slam_export_pipeline.py --svo "C:\Users\User\Desktop\your_file.svo2"

For a quick test:
  python zed_slam_export_pipeline.py --svo "C:\Users\User\Desktop\your_file.svo2" --max_frames 300
"""

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import cv2
import numpy as np
import pyzed.sl as sl


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULT_SVO_PATH = r"C:\Users\User\Desktop\zed2i_19700101_080111.svo2"
DEFAULT_OUT_DIR = r"C:\Users\User\Desktop\SeekAndHide\ZED_SLAM\output"

# Based on your previous observation: ZED camera visual forward should be local -Z.
# If frustums face backward in Blender/CloudCompare, change this to +1.0 or run
# with: --camera_forward_sign 1
DEFAULT_CAMERA_FORWARD_SIGN = -1.0
DEFAULT_FRAME_MARKER_EVERY_N_FRAMES = 10


# -----------------------------------------------------------------------------
# Small helpers
# -----------------------------------------------------------------------------

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sl_mat_to_bgr(mat: sl.Mat) -> Optional[np.ndarray]:
    """
    Convert ZED sl.Mat image to BGR for cv2.imwrite.
    ZED retrieve_image commonly returns BGRA for VIEW.LEFT / VIEW.RIGHT.
    """
    arr = mat.get_data()
    if arr is None:
        return None

    if arr.ndim == 3 and arr.shape[2] == 4:
        return cv2.cvtColor(arr, cv2.COLOR_BGRA2BGR)

    if arr.ndim == 3 and arr.shape[2] == 3:
        return arr

    return arr


def vector_to_list(v: Any, n: int = 3) -> List[Optional[float]]:
    """Convert ZED vector-like data into a Python list."""
    try:
        out = list(v)
        return [float(x) for x in out[:n]]
    except Exception:
        pass

    try:
        return [float(v[i]) for i in range(n)]
    except Exception:
        pass

    return [None] * n


def quaternion_to_list(q: Any) -> List[Optional[float]]:
    """Convert ZED quaternion-like data into [x, y, z, w]."""
    return vector_to_list(q, n=4)


def normalize_vec(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-12)


def pose_to_matrix(pose: sl.Pose) -> np.ndarray:
    """
    Convert ZED Pose to a 4x4 numpy matrix.
    This matrix is camera-to-world in the selected ZED world coordinate system.
    """
    T = pose.pose_data()
    return np.array(T.m, dtype=np.float64).reshape(4, 4)


def matrix_to_pose_values(M: Optional[np.ndarray], camera_forward_sign: float) -> Dict[str, Any]:
    """Flatten pose matrix and derived camera forward/up vectors for CSV."""
    out: Dict[str, Any] = {}

    pose_fields = [
        "pose_x", "pose_y", "pose_z",
        "forward_x", "forward_y", "forward_z",
        "up_x", "up_y", "up_z",
    ]
    matrix_fields = [f"m{r}{c}" for r in range(4) for c in range(4)]

    if M is None:
        for k in pose_fields + matrix_fields:
            out[k] = None
        return out

    pos = M[:3, 3]
    forward = normalize_vec(M[:3, :3] @ np.array([0.0, 0.0, camera_forward_sign], dtype=np.float64))
    up = normalize_vec(M[:3, :3] @ np.array([0.0, 1.0, 0.0], dtype=np.float64))

    out.update({
        "pose_x": pos[0],
        "pose_y": pos[1],
        "pose_z": pos[2],
        "forward_x": forward[0],
        "forward_y": forward[1],
        "forward_z": forward[2],
        "up_x": up[0],
        "up_y": up[1],
        "up_z": up[2],
    })

    for r in range(4):
        for c in range(4):
            out[f"m{r}{c}"] = M[r, c]

    return out


def enum_from_name(enum_obj: Any, name: str, fallback_name: Optional[str] = None) -> Any:
    """
    Resolve pyzed enum from string name, with fallback for SDK-version differences.
    Example: enum_from_name(sl.DEPTH_MODE, "NEURAL", "ULTRA")
    """
    try:
        return getattr(enum_obj, name)
    except Exception:
        if fallback_name is not None:
            return getattr(enum_obj, fallback_name)
        raise


# -----------------------------------------------------------------------------
# Calibration and IMU extraction
# -----------------------------------------------------------------------------

def get_camera_dict(cam: Any) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "fx": float(cam.fx),
        "fy": float(cam.fy),
        "cx": float(cam.cx),
        "cy": float(cam.cy),
        "h_fov": float(cam.h_fov),
        "v_fov": float(cam.v_fov),
        "d_fov": float(cam.d_fov),
        "image_size_width": int(cam.image_size.width),
        "image_size_height": int(cam.image_size.height),
    }

    try:
        d["distortion"] = [float(x) for x in cam.disto]
    except Exception:
        d["distortion"] = []

    return d


def get_calibration_info(zed: sl.Camera) -> Dict[str, Any]:
    cam_info = zed.get_camera_information()

    calib = cam_info.camera_configuration.calibration_parameters
    raw_calib = cam_info.camera_configuration.calibration_parameters_raw

    return {
        "note": "calibration_parameters are rectified parameters; calibration_parameters_raw are original raw camera parameters.",
        "serial_number": cam_info.serial_number,
        "camera_model": str(cam_info.camera_model),
        "camera_resolution_width": int(cam_info.camera_configuration.resolution.width),
        "camera_resolution_height": int(cam_info.camera_configuration.resolution.height),
        "fps": float(cam_info.camera_configuration.fps),
        "baseline_m": float(calib.get_camera_baseline()),
        "rectified_left": get_camera_dict(calib.left_cam),
        "rectified_right": get_camera_dict(calib.right_cam),
        "raw_left": get_camera_dict(raw_calib.left_cam),
        "raw_right": get_camera_dict(raw_calib.right_cam),
    }


def empty_imu_data() -> Dict[str, Any]:
    return {
        "imu_timestamp_ns": None,
        "imu_ax": None,
        "imu_ay": None,
        "imu_az": None,
        "imu_gx": None,
        "imu_gy": None,
        "imu_gz": None,
        "imu_qx": None,
        "imu_qy": None,
        "imu_qz": None,
        "imu_qw": None,
        "imu_temperature": None,
    }


def try_get_imu_data(sensors_data: sl.SensorsData) -> Dict[str, Any]:
    """
    Extract IMU data safely. Some SVO files may not contain full sensor streams.
    """
    out = empty_imu_data()

    try:
        imu = sensors_data.get_imu_data()
    except Exception:
        return out

    try:
        out["imu_timestamp_ns"] = int(imu.timestamp.get_nanoseconds())
    except Exception:
        pass

    try:
        acc = vector_to_list(imu.get_linear_acceleration(), n=3)
        out["imu_ax"], out["imu_ay"], out["imu_az"] = acc
    except Exception:
        pass

    try:
        gyro = vector_to_list(imu.get_angular_velocity(), n=3)
        out["imu_gx"], out["imu_gy"], out["imu_gz"] = gyro
    except Exception:
        pass

    try:
        quat = quaternion_to_list(imu.get_pose().get_orientation().get())
        out["imu_qx"], out["imu_qy"], out["imu_qz"], out["imu_qw"] = quat
    except Exception:
        pass

    try:
        out["imu_temperature"] = float(imu.temperature)
    except Exception:
        pass

    return out


def read_imu_for_current_image(zed: sl.Camera, sensors_data: sl.SensorsData) -> Dict[str, Any]:
    try:
        status = zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.IMAGE)
        if status != sl.ERROR_CODE.SUCCESS:
            return empty_imu_data()
        return try_get_imu_data(sensors_data)
    except Exception:
        return empty_imu_data()


# -----------------------------------------------------------------------------
# OBJ / PLY output
# -----------------------------------------------------------------------------

def save_fused_cloud_as_ply(fused_cloud: sl.FusedPointCloud, out_path: Path) -> Any:
    """
    Save fused point cloud as PLY. Tries several enum names for SDK compatibility.
    """
    out_path = Path(out_path)
    print("[INFO] Saving SLAM map:", out_path)

    for enum_name in ["PLY", "PLY_BIN", "PLY_ASCII"]:
        try:
            fmt = getattr(sl.MESH_FILE_FORMAT, enum_name)
            ok = fused_cloud.save(str(out_path), fmt)
            print(f"[INFO] Saved with sl.MESH_FILE_FORMAT.{enum_name}")
            return ok
        except Exception as e:
            print(f"[WARN] Save with sl.MESH_FILE_FORMAT.{enum_name} failed: {e}")

    ok = fused_cloud.save(str(out_path))
    print("[INFO] Saved with default save method")
    return ok


def write_trajectory_nodes_obj(
    path: Path,
    node_records: List[Dict[str, Any]],
    marker_size: float = 0.02,
) -> None:
    """
    Write trajectory nodes as OBJ.
    The main trajectory is a polyline. Each node also gets a small XYZ cross marker.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("# ZED trajectory nodes OBJ\n")
        f.write("# Contains trajectory line plus small cross markers at sampled nodes.\n")
        f.write("o trajectory_polyline\n")

        # First write node center vertices.
        for rec in node_records:
            p = rec["position"]
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")

        # Trajectory line as consecutive segments. This is more compatible than one very long l statement.
        for i in range(1, len(node_records)):
            f.write(f"l {i} {i + 1}\n")

        # Optional cross markers.
        if marker_size > 0 and len(node_records) > 0:
            f.write("o node_cross_markers\n")
            base_idx = len(node_records) + 1
            lines: List[List[int]] = []

            for rec in node_records:
                p = np.asarray(rec["position"], dtype=np.float64)
                pts = [
                    p + np.array([-marker_size, 0.0, 0.0]),
                    p + np.array([ marker_size, 0.0, 0.0]),
                    p + np.array([0.0, -marker_size, 0.0]),
                    p + np.array([0.0,  marker_size, 0.0]),
                    p + np.array([0.0, 0.0, -marker_size]),
                    p + np.array([0.0, 0.0,  marker_size]),
                ]

                idx0 = base_idx
                for pt in pts:
                    f.write(f"v {pt[0]} {pt[1]} {pt[2]}\n")
                lines.extend([[idx0, idx0 + 1], [idx0 + 2, idx0 + 3], [idx0 + 4, idx0 + 5]])
                base_idx += 6

            for a, b in lines:
                f.write(f"l {a} {b}\n")



def write_trajectory_nodes_with_frame_scalar_ply(
    path: Path,
    node_records: List[Dict[str, Any]],
) -> None:
    """
    Write all valid trajectory nodes as an ASCII PLY point cloud.

    CloudCompare can load the extra numeric PLY properties as scalar fields.
    Switch the active scalar field to "frame_number" to see/query the frame index.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write("comment ZED trajectory nodes with frame_number and node_index scalar fields\n")
        f.write(f"element vertex {len(node_records)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float frame_number\n")
        f.write("property float node_index\n")
        f.write("end_header\n")

        for node_index, rec in enumerate(node_records):
            p = np.asarray(rec["position"], dtype=np.float64)
            frame_number = float(rec["frame"])
            f.write(
                f"{p[0]} {p[1]} {p[2]} "
                f"255 0 0 "
                f"{frame_number} {float(node_index)}\n"
            )


def write_frame_markers_ply(
    path: Path,
    node_records: List[Dict[str, Any]],
    frame_marker_every_n_frames: int = DEFAULT_FRAME_MARKER_EVERY_N_FRAMES,
) -> int:
    """
    Write marker points for frames divisible by frame_marker_every_n_frames.

    Example with default 10:
      frame 0, 10, 20, 30, ...

    The points include frame_number and node_index scalar fields for CloudCompare.
    Returns the number of marker points written.
    """
    if frame_marker_every_n_frames <= 0:
        marker_records: List[Dict[str, Any]] = []
    else:
        marker_records = [
            rec for rec in node_records
            if int(rec["frame"]) % int(frame_marker_every_n_frames) == 0
        ]

    # Record node_index in the original full trajectory node list.
    rec_to_node_index = {id(rec): i for i, rec in enumerate(node_records)}

    with open(path, "w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(
            f"comment ZED trajectory marker nodes every {frame_marker_every_n_frames} frames\n"
        )
        f.write("comment Load in CloudCompare, set scalar field to frame_number, then use point labels.\n")
        f.write(f"element vertex {len(marker_records)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("property float frame_number\n")
        f.write("property float node_index\n")
        f.write("end_header\n")

        for rec in marker_records:
            p = np.asarray(rec["position"], dtype=np.float64)
            frame_number = float(rec["frame"])
            node_index = float(rec_to_node_index.get(id(rec), -1))
            f.write(
                f"{p[0]} {p[1]} {p[2]} "
                f"255 255 0 "
                f"{frame_number} {node_index}\n"
            )

    return len(marker_records)

def write_camera_poses_obj(
    path: Path,
    pose_records: List[Dict[str, Any]],
    camera_forward_sign: float = DEFAULT_CAMERA_FORWARD_SIGN,
    frustum_scale: float = 0.20,
    frustum_depth: float = 1.0,
) -> None:
    """
    Write trajectory line + corrected camera frustums as OBJ.
    camera_forward_sign = -1 uses local -Z as visual forward.
    """
    vertices: List[np.ndarray] = []
    lines: List[List[int]] = []
    centers_idx: List[int] = []

    z = camera_forward_sign * frustum_depth

    local = np.array([
        [0.0,  0.0, 0.0, 1.0],
        [-0.6, -0.35, z, 1.0],
        [ 0.6, -0.35, z, 1.0],
        [ 0.6,  0.35, z, 1.0],
        [-0.6,  0.35, z, 1.0],
    ], dtype=np.float64)
    local[:, :3] *= frustum_scale

    for rec in pose_records:
        M = rec["matrix"]
        base_idx = len(vertices) + 1
        world_pts = (M @ local.T).T[:, :3]

        for p in world_pts:
            vertices.append(p)

        c = base_idx
        p1, p2, p3, p4 = base_idx + 1, base_idx + 2, base_idx + 3, base_idx + 4
        centers_idx.append(c)

        lines.extend([
            [c, p1], [c, p2], [c, p3], [c, p4],
            [p1, p2], [p2, p3], [p3, p4], [p4, p1],
        ])

    for i in range(len(centers_idx) - 1):
        lines.append([centers_idx[i], centers_idx[i + 1]])

    with open(path, "w", encoding="utf-8") as f:
        f.write("# ZED camera trajectory and corrected camera frustums\n")
        f.write(f"# camera_forward_sign = {camera_forward_sign}\n")
        f.write("# If frustums face backward, run with --camera_forward_sign 1\n")
        f.write("o corrected_camera_frustums\n")

        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        for line in lines:
            f.write("l " + " ".join(str(x) for x in line) + "\n")


# -----------------------------------------------------------------------------
# CSV schema
# -----------------------------------------------------------------------------

def make_csv_fieldnames() -> List[str]:
    matrix_fields = [f"m{r}{c}" for r in range(4) for c in range(4)]

    return [
        "frame_index",
        "svo_position",
        "image_timestamp_ns",
        "left_image_path",
        "right_image_path",
        "tracking_state",
        "pose_valid",
        "pose_x", "pose_y", "pose_z",
        "forward_x", "forward_y", "forward_z",
        "up_x", "up_y", "up_z",
        *matrix_fields,
        "baseline_m",
        "left_fx", "left_fy", "left_cx", "left_cy", "left_distortion",
        "right_fx", "right_fy", "right_cx", "right_cy", "right_distortion",
        "raw_left_fx", "raw_left_fy", "raw_left_cx", "raw_left_cy", "raw_left_distortion",
        "raw_right_fx", "raw_right_fy", "raw_right_cx", "raw_right_cy", "raw_right_distortion",
        "imu_timestamp_ns",
        "imu_ax", "imu_ay", "imu_az",
        "imu_gx", "imu_gy", "imu_gz",
        "imu_qx", "imu_qy", "imu_qz", "imu_qw",
        "imu_temperature",
    ]


def make_static_csv_values(static_calib: Dict[str, Any]) -> Dict[str, Any]:
    left_rect = static_calib["rectified_left"]
    right_rect = static_calib["rectified_right"]
    raw_left = static_calib["raw_left"]
    raw_right = static_calib["raw_right"]

    return {
        "baseline_m": static_calib["baseline_m"],
        "left_fx": left_rect["fx"],
        "left_fy": left_rect["fy"],
        "left_cx": left_rect["cx"],
        "left_cy": left_rect["cy"],
        "left_distortion": json.dumps(left_rect["distortion"]),
        "right_fx": right_rect["fx"],
        "right_fy": right_rect["fy"],
        "right_cx": right_rect["cx"],
        "right_cy": right_rect["cy"],
        "right_distortion": json.dumps(right_rect["distortion"]),
        "raw_left_fx": raw_left["fx"],
        "raw_left_fy": raw_left["fy"],
        "raw_left_cx": raw_left["cx"],
        "raw_left_cy": raw_left["cy"],
        "raw_left_distortion": json.dumps(raw_left["distortion"]),
        "raw_right_fx": raw_right["fx"],
        "raw_right_fy": raw_right["fy"],
        "raw_right_cx": raw_right["cx"],
        "raw_right_cy": raw_right["cy"],
        "raw_right_distortion": json.dumps(raw_right["distortion"]),
    }


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------

def run_pipeline(
    svo_path: str,
    out_dir: str = DEFAULT_OUT_DIR,
    image_format: str = "png",
    max_frames: int = -1,
    depth_mode_name: str = "NEURAL",
    depth_maximum_distance: float = 20.0,
    mapping_resolution_meter: float = 0.03,
    mapping_range_meter: float = 10.0,
    map_request_every_n_frames: int = 30,
    node_every_n_frames: int = 1,
    pose_obj_every_n_frames: int = 5,
    frame_marker_every_n_frames: int = DEFAULT_FRAME_MARKER_EVERY_N_FRAMES,
    camera_forward_sign: float = DEFAULT_CAMERA_FORWARD_SIGN,
    frustum_scale: float = 0.20,
    frustum_depth: float = 1.0,
    node_marker_size: float = 0.02,
    disable_mapping: bool = False,
) -> Dict[str, str]:
    """
    Run the full export pipeline.

    This function is intentionally GUI-friendly: later run.py/Tkinter can import this
    function and call it from a worker thread.
    """
    svo_path = str(Path(svo_path))
    out_dir_path = Path(out_dir)

    left_dir = out_dir_path / "left_rectified"
    right_dir = out_dir_path / "right_rectified"
    ensure_dir(out_dir_path)
    ensure_dir(left_dir)
    ensure_dir(right_dir)

    out_map = out_dir_path / "SLAM.ply"
    out_nodes = out_dir_path / "trajectory_nodes.obj"
    out_nodes_frame_scalar = out_dir_path / "trajectory_nodes_with_frame_scalar.ply"
    out_frame_markers = out_dir_path / "trajectory_frame10_markers.ply"
    out_csv = out_dir_path / "frame_camera_imu.csv"
    out_cam_obj = out_dir_path / "camera_poses_corrected.obj"
    out_calib_json = out_dir_path / "camera_calibration_static.json"
    out_summary_json = out_dir_path / "export_summary.json"

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_path)
    init_params.svo_real_time_mode = False
    init_params.coordinate_units = sl.UNIT.METER
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
    init_params.depth_mode = enum_from_name(sl.DEPTH_MODE, depth_mode_name.upper(), fallback_name="ULTRA")
    init_params.depth_maximum_distance = float(depth_maximum_distance)

    print("[INFO] Opening SVO:", svo_path)
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"Failed to open SVO: {status}")

    tracking_enabled = False
    mapping_enabled = False

    try:
        print("[INFO] SVO opened.")

        tracking_params = sl.PositionalTrackingParameters()
        tracking_params.enable_imu_fusion = True
        tracking_params.set_floor_as_origin = False

        status = zed.enable_positional_tracking(tracking_params)
        if status != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError(f"Failed to enable positional tracking: {status}")
        tracking_enabled = True
        print("[INFO] Positional tracking enabled.")

        fused_cloud = sl.FusedPointCloud()
        if not disable_mapping:
            mapping_params = sl.SpatialMappingParameters()
            mapping_params.map_type = sl.SPATIAL_MAP_TYPE.FUSED_POINT_CLOUD
            mapping_params.resolution_meter = float(mapping_resolution_meter)
            mapping_params.range_meter = float(mapping_range_meter)
            mapping_params.use_chunk_only = False
            mapping_params.save_texture = False

            status = zed.enable_spatial_mapping(mapping_params)
            if status != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Failed to enable spatial mapping: {status}")
            mapping_enabled = True
            print("[INFO] Spatial mapping enabled.")

        static_calib = get_calibration_info(zed)
        with open(out_calib_json, "w", encoding="utf-8") as f:
            json.dump(static_calib, f, indent=2, ensure_ascii=False)

        static_csv_values = make_static_csv_values(static_calib)

        runtime_params = sl.RuntimeParameters()
        left_mat = sl.Mat()
        right_mat = sl.Mat()
        sensors_data = sl.SensorsData()
        pose = sl.Pose()

        node_records: List[Dict[str, Any]] = []
        camera_obj_pose_records: List[Dict[str, Any]] = []

        frame_idx = 0
        valid_pose_count = 0
        start_time = time.time()
        map_request_pending = False

        print("[INFO] Processing SVO and exporting frames...")

        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=make_csv_fieldnames())
            writer.writeheader()

            while True:
                if max_frames > 0 and frame_idx >= max_frames:
                    print(f"[INFO] max_frames reached: {max_frames}")
                    break

                grab_status = zed.grab(runtime_params)

                if grab_status == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                    print("[INFO] End of SVO reached.")
                    break

                if grab_status != sl.ERROR_CODE.SUCCESS:
                    print(f"[WARN] grab failed at frame {frame_idx}: {grab_status}")
                    frame_idx += 1
                    continue

                try:
                    svo_position = zed.get_svo_position()
                except Exception:
                    svo_position = frame_idx

                try:
                    image_timestamp_ns = int(
                        zed.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_nanoseconds()
                    )
                except Exception:
                    image_timestamp_ns = None

                # Rectified left/right images.
                left_name = f"frame_{frame_idx:06d}.{image_format}"
                right_name = f"frame_{frame_idx:06d}.{image_format}"
                left_path = left_dir / left_name
                right_path = right_dir / right_name

                zed.retrieve_image(left_mat, sl.VIEW.LEFT)
                zed.retrieve_image(right_mat, sl.VIEW.RIGHT)

                left_bgr = sl_mat_to_bgr(left_mat)
                right_bgr = sl_mat_to_bgr(right_mat)

                if left_bgr is not None:
                    cv2.imwrite(str(left_path), left_bgr)
                else:
                    print(f"[WARN] left image empty at frame {frame_idx}")

                if right_bgr is not None:
                    cv2.imwrite(str(right_path), right_bgr)
                else:
                    print(f"[WARN] right image empty at frame {frame_idx}")

                # Pose for the current frame.
                tracking_state = zed.get_position(pose, sl.REFERENCE_FRAME.WORLD)
                pose_valid = tracking_state == sl.POSITIONAL_TRACKING_STATE.OK
                M: Optional[np.ndarray] = None

                if pose_valid:
                    M = pose_to_matrix(pose)
                    valid_pose_count += 1
                    pos = M[:3, 3].copy()

                    if node_every_n_frames > 0 and frame_idx % node_every_n_frames == 0:
                        node_records.append({
                            "frame": frame_idx,
                            "position": pos,
                        })

                    if pose_obj_every_n_frames > 0 and frame_idx % pose_obj_every_n_frames == 0:
                        camera_obj_pose_records.append({
                            "frame": frame_idx,
                            "matrix": M.copy(),
                        })

                imu_info = read_imu_for_current_image(zed, sensors_data)

                row: Dict[str, Any] = {
                    "frame_index": frame_idx,
                    "svo_position": svo_position,
                    "image_timestamp_ns": image_timestamp_ns,
                    "left_image_path": str(left_path),
                    "right_image_path": str(right_path),
                    "tracking_state": str(tracking_state),
                    "pose_valid": int(bool(pose_valid)),
                }
                row.update(matrix_to_pose_values(M, camera_forward_sign=camera_forward_sign))
                row.update(static_csv_values)
                row.update(imu_info)
                writer.writerow(row)

                # Spatial map async update. Frame-based updates are more reliable for offline SVO
                # than wall-clock-only updates because SVO processing speed can vary a lot.
                if mapping_enabled:
                    try:
                        if map_request_pending:
                            if zed.get_spatial_map_request_status_async() == sl.ERROR_CODE.SUCCESS:
                                zed.retrieve_spatial_map_async(fused_cloud)
                                map_request_pending = False

                        if (
                            not map_request_pending
                            and map_request_every_n_frames > 0
                            and frame_idx % map_request_every_n_frames == 0
                        ):
                            zed.request_spatial_map_async()
                            map_request_pending = True
                    except Exception as e:
                        print(f"[WARN] spatial map async update failed at frame {frame_idx}: {e}")

                if frame_idx % 100 == 0:
                    elapsed = time.time() - start_time
                    print(
                        f"[INFO] frame={frame_idx}, "
                        f"tracking={tracking_state}, "
                        f"valid_poses={valid_pose_count}, "
                        f"nodes={len(node_records)}, "
                        f"elapsed={elapsed:.1f}s"
                    )

                frame_idx += 1

        # ------------------------------------------------------------------
        # Save trajectory outputs BEFORE extracting/saving SLAM.ply.
        # This is intentional: on some SVOs or SDK versions, spatial mapping
        # extraction/save can fail, but we still want the trajectory PLY/OBJ and
        # frame_camera_imu.csv to be available.
        # ------------------------------------------------------------------
        print("[INFO] Saving trajectory nodes OBJ:", out_nodes)
        write_trajectory_nodes_obj(out_nodes, node_records, marker_size=node_marker_size)

        print("[INFO] Saving trajectory nodes PLY with frame scalar:", out_nodes_frame_scalar)
        write_trajectory_nodes_with_frame_scalar_ply(out_nodes_frame_scalar, node_records)

        print(
            f"[INFO] Saving frame marker PLY every {frame_marker_every_n_frames} frames:",
            out_frame_markers,
        )
        frame_marker_count = write_frame_markers_ply(
            out_frame_markers,
            node_records,
            frame_marker_every_n_frames=frame_marker_every_n_frames,
        )

        print("[INFO] Saving corrected camera poses OBJ:", out_cam_obj)
        write_camera_poses_obj(
            out_cam_obj,
            camera_obj_pose_records,
            camera_forward_sign=camera_forward_sign,
            frustum_scale=frustum_scale,
            frustum_depth=frustum_depth,
        )

        if mapping_enabled:
            try:
                print("[INFO] Extracting whole spatial map...")
                zed.extract_whole_spatial_map(fused_cloud)
                save_fused_cloud_as_ply(fused_cloud, out_map)
            except Exception as e:
                print(f"[WARN] Failed to extract/save SLAM.ply, but trajectory outputs were already saved: {e}")
        else:
            print("[INFO] Mapping disabled; SLAM.ply was not generated.")

        summary = {
            "svo_path": svo_path,
            "out_dir": str(out_dir_path),
            "frames_processed": frame_idx,
            "valid_pose_count": valid_pose_count,
            "node_count": len(node_records),
            "frame_marker_every_n_frames": frame_marker_every_n_frames,
            "frame_marker_count": frame_marker_count,
            "camera_pose_obj_count": len(camera_obj_pose_records),
            "outputs": {
                "SLAM_ply": str(out_map),
                "trajectory_nodes_obj": str(out_nodes),
                "trajectory_nodes_with_frame_scalar_ply": str(out_nodes_frame_scalar),
                "trajectory_frame10_markers_ply": str(out_frame_markers),
                "frame_camera_imu_csv": str(out_csv),
                "camera_poses_corrected_obj": str(out_cam_obj),
                "camera_calibration_static_json": str(out_calib_json),
                "left_rectified_dir": str(left_dir),
                "right_rectified_dir": str(right_dir),
            },
        }
        with open(out_summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        print("\n[DONE]")
        print("Frames processed:", frame_idx)
        print("Valid poses:", valid_pose_count)
        print("SLAM map:", out_map)
        print("Trajectory nodes OBJ:", out_nodes)
        print("Trajectory nodes PLY with frame scalar:", out_nodes_frame_scalar)
        print(f"Frame marker PLY every {frame_marker_every_n_frames} frames:", out_frame_markers)
        print("Frame marker count:", frame_marker_count)
        print("Frame camera/IMU CSV:", out_csv)
        print("Corrected camera poses OBJ:", out_cam_obj)
        print("Left rectified images:", left_dir)
        print("Right rectified images:", right_dir)
        print("Calibration JSON:", out_calib_json)
        print("Summary JSON:", out_summary_json)

        return summary["outputs"]

    finally:
        if mapping_enabled:
            try:
                zed.disable_spatial_mapping()
            except Exception:
                pass

        if tracking_enabled:
            try:
                zed.disable_positional_tracking()
            except Exception:
                pass

        try:
            zed.close()
        except Exception:
            pass


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export ZED SVO/SVO2 to SLAM.ply, per-frame CSV, trajectory OBJ, and rectified stereo images."
    )

    parser.add_argument(
        "--svo",
        type=str,
        default=DEFAULT_SVO_PATH,
        help="Input .svo or .svo2 path."
    )

    parser.add_argument(
        "--out",
        type=str,
        default=DEFAULT_OUT_DIR,
        help="Output folder."
    )

    parser.add_argument(
        "--image_format",
        type=str,
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Output image format."
    )

    parser.add_argument(
        "--max_frames",
        type=int,
        default=-1,
        help="Export only first N frames. Use -1 for all frames."
    )

    parser.add_argument(
        "--depth_mode",
        type=str,
        default="NEURAL",
        help="ZED depth mode name. Common: NEURAL, ULTRA, QUALITY, PERFORMANCE."
    )

    parser.add_argument(
        "--depth_maximum_distance",
        type=float,
        default=20.0,
        help="Maximum depth distance in meters."
    )

    parser.add_argument(
        "--mapping_resolution_meter",
        type=float,
        default=0.03,
        help="Spatial mapping resolution in meters."
    )

    parser.add_argument(
        "--mapping_range_meter",
        type=float,
        default=10.0,
        help="Spatial mapping range in meters."
    )

    parser.add_argument(
        "--map_request_every",
        type=int,
        default=30,
        help="Request async spatial map update every N frames."
    )

    parser.add_argument(
        "--node_every",
        type=int,
        default=1,
        help="Save one trajectory node every N valid-pose frames. Default 1 = every valid frame."
    )

    parser.add_argument(
        "--pose_obj_every",
        type=int,
        default=5,
        help="Save one camera frustum every N valid-pose frames for camera_poses_corrected.obj."
    )

    parser.add_argument(
        "--frame_marker_every",
        type=int,
        default=DEFAULT_FRAME_MARKER_EVERY_N_FRAMES,
        help="Save one CloudCompare marker point every N frames in trajectory_frame10_markers.ply."
    )

    parser.add_argument(
        "--camera_forward_sign",
        type=float,
        default=DEFAULT_CAMERA_FORWARD_SIGN,
        choices=[-1.0, 1.0],
        help="Use -1 for local -Z visual forward, 1 for local +Z."
    )

    parser.add_argument(
        "--frustum_scale",
        type=float,
        default=0.20,
        help="Camera frustum scale in OBJ."
    )

    parser.add_argument(
        "--frustum_depth",
        type=float,
        default=1.0,
        help="Camera frustum local depth before scaling."
    )

    parser.add_argument(
        "--node_marker_size",
        type=float,
        default=0.02,
        help="Small cross marker size for trajectory_nodes.obj. Set 0 to disable markers."
    )

    parser.add_argument(
        "--disable_mapping",
        action="store_true",
        help="Disable spatial mapping. Useful for fast image/CSV test only."
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        svo_path=args.svo,
        out_dir=args.out,
        image_format=args.image_format,
        max_frames=args.max_frames,
        depth_mode_name=args.depth_mode,
        depth_maximum_distance=args.depth_maximum_distance,
        mapping_resolution_meter=args.mapping_resolution_meter,
        mapping_range_meter=args.mapping_range_meter,
        map_request_every_n_frames=args.map_request_every,
        node_every_n_frames=args.node_every,
        pose_obj_every_n_frames=args.pose_obj_every,
        frame_marker_every_n_frames=args.frame_marker_every,
        camera_forward_sign=args.camera_forward_sign,
        frustum_scale=args.frustum_scale,
        frustum_depth=args.frustum_depth,
        node_marker_size=args.node_marker_size,
        disable_mapping=args.disable_mapping,
    )


if __name__ == "__main__":
    main()

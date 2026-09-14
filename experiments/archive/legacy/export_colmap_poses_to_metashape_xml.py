"""Metashape-embedded helper used by export_colmap_poses_to_metashape.py."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import Metashape


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _quaternion_to_rotation(qw: float, qx: float, qy: float, qz: float):
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("invalid COLMAP quaternion")
    w, x, y, z = (value / norm for value in (qw, qx, qy, qz))
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _mat_transpose(matrix):
    return [[matrix[row][column] for row in range(3)] for column in range(3)]


def _mat_vec(matrix, vector):
    return [
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    ]


def _read_cameras(path: Path):
    cameras = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 5:
                raise ValueError(f"{path}:{line_number}: invalid camera row")
            camera_id = int(fields[0])
            camera = {
                "id": camera_id,
                "model": fields[1],
                "width": int(fields[2]),
                "height": int(fields[3]),
                "params": [float(value) for value in fields[4:]],
            }
            if camera["model"] != "FULL_OPENCV":
                raise ValueError(
                    f"the XML compatibility export requires FULL_OPENCV cameras; "
                    f"camera {camera_id} is {camera['model']}"
                )
            if len(camera["params"]) != 12:
                raise ValueError(
                    f"camera {camera_id}: FULL_OPENCV must contain 12 parameters"
                )
            cameras[camera_id] = camera
    if not cameras:
        raise ValueError(f"no cameras found in {path}")
    return cameras


def _read_poses(path: Path):
    poses = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.strip().split()
            if not fields or fields[0].startswith("#") or len(fields) != 10:
                continue
            image_id = int(fields[0])
            qw, qx, qy, qz = (float(value) for value in fields[1:5])
            t = [float(value) for value in fields[5:8]]
            camera_id = int(fields[8])
            name = fields[9]
            rotation_wc = _quaternion_to_rotation(qw, qx, qy, qz)
            rotation_cw = _mat_transpose(rotation_wc)
            center = [-value for value in _mat_vec(rotation_cw, t)]
            poses.append(
                {
                    "image_id": image_id,
                    "camera_id": camera_id,
                    "name": name,
                    "rotation_cw": rotation_cw,
                    "center": center,
                }
            )
    poses.sort(key=lambda pose: pose["image_id"])
    return poses


def _camera_matrix(rotation, center):
    return Metashape.Matrix(
        [
            [rotation[0][0], rotation[0][1], rotation[0][2], center[0]],
            [rotation[1][0], rotation[1][1], rotation[1][2], center[1]],
            [rotation[2][0], rotation[2][1], rotation[2][2], center[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def main() -> int:
    args = _parse_args()
    model = args.model.resolve()
    images = args.images.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cameras = _read_cameras(model / "cameras.txt")
    poses = _read_poses(model / "images.txt")
    for pose in poses:
        if pose["camera_id"] not in cameras:
            raise ValueError(
                f"pose image {pose['name']} references missing camera "
                f"{pose['camera_id']}"
            )
    image_paths = [images / pose["name"] for pose in poses]
    missing = [path for path in image_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(str(missing[0]))

    doc = Metashape.Document()
    chunk = doc.addChunk()
    chunk.addPhotos([str(path) for path in image_paths])
    if len(chunk.cameras) != len(poses):
        raise RuntimeError(
            "Metashape added an unexpected number of cameras: "
            f"{len(chunk.cameras)} != {len(poses)}"
        )

    sensors = list(chunk.sensors)
    if not sensors:
        raise RuntimeError("Metashape did not create a sensor")

    # Metashape sensors carry calibration, while COLMAP camera IDs distinguish
    # the left/right cameras.  Keep one Metashape sensor per COLMAP camera and
    # assign each imported photo to the corresponding sensor below.
    sensor_by_id = {}
    for index, camera_id in enumerate(sorted(cameras)):
        sensor = sensors[0] if index == 0 else chunk.addSensor()
        camera = cameras[camera_id]
        sensor_by_id[camera_id] = sensor
        width = camera["width"]
        height = camera["height"]
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = camera["params"]

        # Metashape 1.7's frame calibration has one focal length plus
        # affinity and skew.  This exactly represents the COLMAP linear K
        # matrix when f=fy, b1=fx-fy, b2=0.  Its legacy XML has no rational
        # k5/k6 fields; the exact values remain in the copied COLMAP package
        # and the exporter manifest.
        calibration = Metashape.Calibration()
        calibration.width = width
        calibration.height = height
        calibration.f = fy
        calibration.cx = cx - width * 0.5
        calibration.cy = cy - height * 0.5
        calibration.b1 = fx - fy
        calibration.b2 = 0.0
        calibration.k1 = k1
        calibration.k2 = k2
        calibration.k3 = k3
        calibration.k4 = k4
        calibration.p1 = p1
        calibration.p2 = p2
        # user_calib is the writable initial/precalibrated calibration.
        sensor.user_calib = calibration
        sensor.fixed_calibration = True
        sensor.label = f"COLMAP_FULL_OPENCV_camera_{camera_id}_legacy_compatible"

    print("Metashape camera labels:", [camera_item.label for camera_item in chunk.cameras[:3]])
    print("Metashape photo paths:", [camera_item.photo.path for camera_item in chunk.cameras[:3]])
    cameras_by_label = {}
    for camera_item in chunk.cameras:
        keys = {camera_item.label}
        if camera_item.photo is not None:
            photo_path = Path(camera_item.photo.path)
            keys.update({str(photo_path), photo_path.name, photo_path.stem})
        for key in keys:
            cameras_by_label[key] = camera_item
    for pose in poses:
        camera_item = cameras_by_label.get(pose["name"])
        if camera_item is None:
            camera_item = cameras_by_label.get(Path(pose["name"]).stem)
        if camera_item is None:
            raise RuntimeError(f"Metashape camera label not found: {pose['name']}")
        camera_item.sensor = sensor_by_id[pose["camera_id"]]
        camera_item.transform = _camera_matrix(pose["rotation_cw"], pose["center"])
        camera_item.enabled = True

    chunk.exportCameras(path=str(output), format=Metashape.CamerasFormatXML)
    if not output.is_file():
        raise RuntimeError(f"Metashape did not create {output}")

    # Read the generated XML back in a fresh chunk as a format-level check.
    check_doc = Metashape.Document()
    check_chunk = check_doc.addChunk()
    left_paths = [
        str(path)
        for path in image_paths
        if Path(path).parent.name.lower() == "left"
    ]
    right_paths = [
        str(path)
        for path in image_paths
        if Path(path).parent.name.lower() == "right"
    ]
    check_chunk.addPhotos([str(path) for path in image_paths])
    if left_paths and right_paths:
        check_right_sensor = check_chunk.addSensor()
        for item in check_chunk.cameras:
            if item.photo is not None and Path(item.photo.path).parent.name.lower() == "right":
                item.sensor = check_right_sensor
    print("Round-trip pre-import sensor count:", len(check_chunk.sensors))
    check_chunk.importCameras(path=str(output), format=Metashape.CamerasFormatXML)
    print("Round-trip sensor count:", len(check_chunk.sensors))
    print("Round-trip sensors:", [sensor.label for sensor in check_chunk.sensors])
    print(
        "Round-trip aligned labels:",
        [item.label for item in check_chunk.cameras if item.transform is not None][:10],
    )
    aligned = sum(1 for item in check_chunk.cameras if item.transform is not None)
    if aligned != len(poses):
        raise RuntimeError(
            "Metashape XML round-trip camera count mismatch: "
            f"{aligned} != {len(poses)}"
        )

    print("Metashape version:", Metashape.version)
    print("XML output:", output)
    print("COLMAP cameras:", {
        camera_id: camera["model"] for camera_id, camera in sorted(cameras.items())
    })
    print("Pose count:", len(poses))
    print("Legacy XML omits COLMAP k5/k6; exact values remain in the COLMAP package and manifest")
    Metashape.app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

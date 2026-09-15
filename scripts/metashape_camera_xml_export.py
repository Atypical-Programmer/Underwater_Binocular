"""Metashape 1.7-compatible runner for a COLMAP-to-XML camera export."""


import argparse
import json
import math
import sys
from pathlib import Path

import Metashape


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image-map", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-roundtrip", action="store_true")
    parser.add_argument("--verification-output", type=Path)
    return parser.parse_args()


def _normalize(value):
    return str(value).replace("\\", "/").casefold()


def _basename(value):
    return _normalize(value).rsplit("/", 1)[-1]


def _quaternion_to_rotation(qw, qx, qy, qz):
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("invalid COLMAP quaternion")
    w, x, y, z = (value / norm for value in (qw, qx, qy, qz))
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def _transpose(matrix):
    return [[matrix[row][column] for row in range(3)] for column in range(3)]


def _mat_vec(matrix, vector):
    return [
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    ]


def _read_cameras(path):
    cameras = {}
    with path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.strip().split()
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) < 5:
                raise ValueError(f"invalid camera row at {path}:{line_number}")
            camera_id = int(fields[0])
            cameras[camera_id] = {
                "model": fields[1],
                "width": int(fields[2]),
                "height": int(fields[3]),
                "params": [float(value) for value in fields[4:]],
            }
    if not cameras:
        raise ValueError(f"no cameras found in {path}")
    return cameras


def _read_poses(path):
    poses = []
    with path.open("r", encoding="utf-8-sig") as handle:
        lines = iter(handle)
        for line in lines:
            fields = line.strip().split(maxsplit=9)
            if not fields or fields[0].startswith("#"):
                continue
            if len(fields) != 10:
                raise ValueError(f"invalid pose row in {path}")
            image_id = int(fields[0])
            qw, qx, qy, qz = [float(value) for value in fields[1:5]]
            translation = [float(value) for value in fields[5:8]]
            camera_id = int(fields[8])
            source_name = fields[9].replace("\\", "/")
            rotation_wc = _quaternion_to_rotation(qw, qx, qy, qz)
            rotation_cw = _transpose(rotation_wc)
            center = [-value for value in _mat_vec(rotation_cw, translation)]
            poses.append(
                {
                    "image_id": image_id,
                    "camera_id": camera_id,
                    "source_name": source_name,
                    "label": source_name.rsplit("/", 1)[-1],
                    "rotation_cw": rotation_cw,
                    "center": center,
                }
            )
            try:
                next(lines)
            except StopIteration:
                raise ValueError(f"missing POINTS2D row in {path}") from None
    poses.sort(key=lambda pose: pose["image_id"])
    if not poses:
        raise ValueError(f"no poses found in {path}")
    return poses


def _read_image_map(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = {}
    for item in payload.get("images", []):
        image_path = item.get("path")
        if not image_path:
            continue
        for key in (item.get("source_name"), item.get("label"), image_path):
            if key:
                mapping[_normalize(key)] = str(image_path)
                mapping[_basename(key)] = str(image_path)
    return mapping


def _image_path(mapping, pose):
    for key in (pose["source_name"], pose["label"]):
        value = mapping.get(_normalize(key)) or mapping.get(_basename(key))
        if value:
            path = Path(value)
            if path.is_file():
                return path.resolve()
    raise FileNotFoundError("image mapping unavailable for {}".format(pose["source_name"]))


def _calibration(camera):
    model = camera["model"]
    params = camera["params"]
    if model == "FULL_OPENCV":
        if len(params) != 12:
            raise ValueError("FULL_OPENCV camera must have 12 parameters")
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4 = params[:10]
    elif model == "OPENCV":
        if len(params) != 8:
            raise ValueError("OPENCV camera must have 8 parameters")
        fx, fy, cx, cy, k1, k2, p1, p2 = params
        k3, k4 = 0.0, 0.0
    elif model == "PINHOLE":
        if len(params) != 4:
            raise ValueError("PINHOLE camera must have 4 parameters")
        fx, fy, cx, cy = params
        k1 = k2 = k3 = k4 = p1 = p2 = 0.0
    elif model == "SIMPLE_PINHOLE":
        if len(params) != 3:
            raise ValueError("SIMPLE_PINHOLE camera must have 3 parameters")
        f, cx, cy = params
        fx = fy = f
        k1 = k2 = k3 = k4 = p1 = p2 = 0.0
    elif model == "SIMPLE_RADIAL":
        if len(params) != 4:
            raise ValueError("SIMPLE_RADIAL camera must have 4 parameters")
        f, cx, cy, k1 = params
        fx = fy = f
        k2 = k3 = k4 = p1 = p2 = 0.0
    elif model == "RADIAL":
        if len(params) != 5:
            raise ValueError("RADIAL camera must have 5 parameters")
        f, cx, cy, k1, k2 = params
        fx = fy = f
        k3 = k4 = p1 = p2 = 0.0
    else:
        raise ValueError(f"unsupported COLMAP camera model: {model}")

    calibration = Metashape.Calibration()
    calibration.width = camera["width"]
    calibration.height = camera["height"]
    calibration.f = fy
    calibration.cx = cx - camera["width"] * 0.5
    calibration.cy = cy - camera["height"] * 0.5
    calibration.b1 = fx - fy
    calibration.b2 = 0.0
    calibration.k1 = k1
    calibration.k2 = k2
    calibration.k3 = k3
    calibration.k4 = k4
    calibration.p1 = p1
    calibration.p2 = p2
    return calibration


def _camera_matrix(rotation, center):
    return Metashape.Matrix(
        [
            [rotation[0][0], rotation[0][1], rotation[0][2], center[0]],
            [rotation[1][0], rotation[1][1], rotation[1][2], center[1]],
            [rotation[2][0], rotation[2][1], rotation[2][2], center[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def _camera_lookup(cameras):
    lookup = {}
    for camera in cameras:
        keys = {_normalize(camera.label)}
        if camera.photo is not None:
            keys.add(_normalize(camera.photo.path))
            keys.add(_basename(camera.photo.path))
            keys.add(Path(camera.photo.path).stem.casefold())
        for key in keys:
            lookup[key] = camera
    return lookup


def _add_right_sensor_for_roundtrip(chunk):
    """Keep the image group usable when the XML contains left/right sensors."""

    paths = [
        Path(camera.photo.path).parent.name.casefold()
        for camera in chunk.cameras
        if camera.photo is not None
    ]
    if "left" in paths and "right" in paths and len(chunk.sensors) == 1:
        chunk.addSensor()


def main():
    args = _parse_args()
    model = args.model.resolve()
    image_map_path = args.image_map.resolve()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    cameras = _read_cameras(model / "cameras.txt")
    poses = _read_poses(model / "images.txt")
    mapping = _read_image_map(image_map_path)
    image_paths = [_image_path(mapping, pose) for pose in poses]
    missing_camera_ids = sorted(set(pose["camera_id"] for pose in poses) - set(cameras))
    if missing_camera_ids:
        raise ValueError(f"poses reference missing camera ids: {missing_camera_ids}")

    doc = Metashape.Document()
    try:
        chunk = doc.addChunk()
        chunk.addPhotos([str(path) for path in image_paths])
        if len(chunk.cameras) != len(poses):
            raise RuntimeError(
                f"Metashape added {len(chunk.cameras)} cameras, expected {len(poses)}"
            )
        sensors = list(chunk.sensors)
        if not sensors:
            raise RuntimeError("Metashape did not create a sensor")

        sensor_by_id = {}
        for index, camera_id in enumerate(sorted(cameras)):
            sensor = sensors[0] if index == 0 else chunk.addSensor()
            sensor.user_calib = _calibration(cameras[camera_id])
            sensor.fixed_calibration = True
            sensor.label = "COLMAP_{}_camera_{}_legacy_compatible".format(
                cameras[camera_id]["model"], camera_id
            )
            sensor_by_id[camera_id] = sensor

        lookup = _camera_lookup(chunk.cameras)
        for pose in poses:
            keys = (
                _normalize(pose["source_name"]),
                _normalize(pose["label"]),
            )
            camera = next((lookup.get(key) for key in keys if lookup.get(key)), None)
            if camera is None:
                raise RuntimeError("Metashape camera label not found: {}".format(pose["source_name"]))
            camera.sensor = sensor_by_id[pose["camera_id"]]
            camera.transform = _camera_matrix(pose["rotation_cw"], pose["center"])
            camera.enabled = True

        chunk.exportCameras(path=str(output), format=Metashape.CamerasFormatXML)
        if not output.is_file():
            raise RuntimeError(f"Metashape did not create {output}")

        aligned = None
        if args.verify_roundtrip:
            check_doc = Metashape.Document()
            try:
                check_chunk = check_doc.addChunk()
                check_chunk.addPhotos([str(path) for path in image_paths])
                _add_right_sensor_for_roundtrip(check_chunk)
                check_chunk.importCameras(path=str(output), format=Metashape.CamerasFormatXML)
                aligned = sum(
                    1 for camera in check_chunk.cameras if camera.transform is not None
                )
                if aligned != len(poses):
                    raise RuntimeError(
                        f"Metashape XML round-trip aligned {aligned} cameras, expected {len(poses)}"
                    )
            finally:
                del check_doc

            verification_path = args.verification_output
            if verification_path is None:
                verification_path = output.with_suffix(".roundtrip.json")
            verification_path.parent.mkdir(parents=True, exist_ok=True)
            verification_path.write_text(
                json.dumps(
                    {"pose_count": len(poses), "aligned_count": aligned},
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

        print(f"Metashape version: {Metashape.version}")
        print(f"XML output: {output}")
        print(f"Pose count: {len(poses)}")
        if aligned is not None:
            print(f"Round-trip aligned count: {aligned}")
        sys.stdout.flush()
    finally:
        del doc
        Metashape.app.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

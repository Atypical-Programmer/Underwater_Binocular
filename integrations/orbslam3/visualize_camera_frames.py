"""Render ORB-SLAM3 camera poses and sampled camera coordinate frames.

The trajectory is the TUM-format CameraTrajectory.txt written by the stereo
runner.  Each pose is world-from-camera (T_wc): the camera center is the
translation and the three camera axes are the columns of R_wc.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_tum_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    positions: list[list[float]] = []
    rotations: list[np.ndarray] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split()
        if len(fields) < 8:
            continue
        timestamp, tx, ty, tz, qx, qy, qz, qw = map(float, fields[:8])
        q = np.array([qx, qy, qz, qw], dtype=np.float64)
        norm = float(np.linalg.norm(q))
        if norm < 1e-12:
            continue
        x, y, z, w = q / norm
        rotation = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )
        timestamps.append(timestamp)
        positions.append([tx, ty, tz])
        rotations.append(rotation)
    if not positions:
        raise ValueError(f"no valid TUM poses found in {path}")
    return np.asarray(timestamps), np.asarray(positions), np.asarray(rotations)


def read_tracking_states(path: Path) -> tuple[np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    valid: list[bool] = []
    if not path.exists():
        return np.empty(0), np.empty(0, dtype=bool)
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            timestamps.append(float(row["timestamp_sec"]))
            valid.append(row["pose_valid"] == "1")
    return np.asarray(timestamps), np.asarray(valid, dtype=bool)


def match_validity(pose_timestamps: np.ndarray, log_timestamps: np.ndarray, log_valid: np.ndarray) -> np.ndarray:
    if len(log_timestamps) == 0:
        return np.ones(len(pose_timestamps), dtype=bool)
    matched = np.ones(len(pose_timestamps), dtype=bool)
    for pose_index, timestamp in enumerate(pose_timestamps):
        insert = int(np.searchsorted(log_timestamps, timestamp))
        candidates = [index for index in (insert - 1, insert) if 0 <= index < len(log_timestamps)]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda index: abs(float(log_timestamps[index] - timestamp)))
        if abs(float(log_timestamps[nearest] - timestamp)) <= 1e-3:
            matched[pose_index] = bool(log_valid[nearest])
    return matched


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def color_ramp(value: float) -> tuple[int, int, int]:
    value = max(0.0, min(1.0, value))
    # Blue -> cyan -> yellow -> orange, easy to read on a dark background.
    stops = np.array([[45, 116, 190], [39, 190, 183], [244, 205, 73], [232, 111, 54]], dtype=float)
    scaled = value * (len(stops) - 1)
    left = min(int(scaled), len(stops) - 2)
    fraction = scaled - left
    rgb = stops[left] * (1 - fraction) + stops[left + 1] * fraction
    return tuple(int(channel) for channel in rgb)


def normalize(vector: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(vector))
    if length < 1e-12:
        return vector.copy()
    return vector / length


def make_projection(points: np.ndarray, box: tuple[int, int, int, int]):
    left, top, right, bottom = box
    center = points.mean(axis=0)
    _, _, vt = np.linalg.svd(points - center, full_matrices=False)
    horizontal = normalize(vt[0])
    depth = normalize(vt[2])
    vertical = normalize(0.82 * vt[1] + 0.48 * depth)

    projected_x = (points - center) @ horizontal
    projected_y = (points - center) @ vertical
    min_x, max_x = float(projected_x.min()), float(projected_x.max())
    min_y, max_y = float(projected_y.min()), float(projected_y.max())
    span = max(max_x - min_x, max_y - min_y, 1e-6)
    pad = 0.10 * span
    min_x -= pad
    max_x += pad
    min_y -= pad
    max_y += pad

    def project(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array)
        if array.ndim == 1:
            array = array[None, :]
        x = (array - center) @ horizontal
        y = (array - center) @ vertical
        px = left + (x - min_x) / (max_x - min_x) * (right - left)
        py = bottom - (y - min_y) / (max_y - min_y) * (bottom - top)
        return np.column_stack([px, py])

    return project, center, horizontal, vertical


def world_xy_projection(points: np.ndarray, box: tuple[int, int, int, int]):
    left, top, right, bottom = box
    x = points[:, 0]
    y = points[:, 1]
    min_x, max_x = float(x.min()), float(x.max())
    min_y, max_y = float(y.min()), float(y.max())
    span = max(max_x - min_x, max_y - min_y, 1e-6)
    pad = 0.10 * span
    min_x -= pad
    max_x += pad
    min_y -= pad
    max_y += pad

    def project(array: np.ndarray) -> np.ndarray:
        array = np.asarray(array)
        if array.ndim == 1:
            array = array[None, :]
        px = left + (array[:, 0] - min_x) / (max_x - min_x) * (right - left)
        py = bottom - (array[:, 1] - min_y) / (max_y - min_y) * (bottom - top)
        return np.column_stack([px, py])

    return project


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], color, width: int = 4):
    draw.line([start, end], fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    size = max(8, width * 3)
    left = (end[0] - ux * size - uy * size * 0.55, end[1] - uy * size + ux * size * 0.55)
    right = (end[0] - ux * size + uy * size * 0.55, end[1] - uy * size - ux * size * 0.55)
    draw.polygon([end, left, right], fill=color)


def panel(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str, title_font):
    left, top, right, bottom = box
    draw.rounded_rectangle(box, radius=18, fill=(20, 29, 43), outline=(66, 86, 110), width=2)
    draw.text((left + 22, top + 16), title, font=title_font, fill=(236, 242, 249))


def render(trajectory: Path, tracking_log: Path, output: Path) -> dict[str, float | int]:
    pose_timestamps, positions, rotations = read_tum_trajectory(trajectory)
    log_timestamps, log_valid = read_tracking_states(tracking_log)
    valid = match_validity(pose_timestamps, log_timestamps, log_valid)

    width, height = 2000, 1200
    image = Image.new("RGB", (width, height), (10, 16, 25))
    draw = ImageDraw.Draw(image)
    title_font = font(34, bold=True)
    section_font = font(24, bold=True)
    body_font = font(22)
    small_font = font(18)

    draw.text((54, 34), "ORB-SLAM3 stereo camera frames", font=title_font, fill=(245, 248, 252))
    draw.text((56, 82), "Sampled camera coordinate axes along the estimated T_wc trajectory", font=body_font, fill=(158, 177, 199))

    main_box = (52, 138, 1270, 1128)
    top_box = (1320, 138, 1948, 600)
    info_box = (1320, 632, 1948, 1128)
    panel(draw, main_box, "Oblique 3D projection", section_font)
    panel(draw, top_box, "World XY top view", section_font)
    panel(draw, info_box, "Coordinate frame / run summary", section_font)

    main_plot_box = (100, 218, 1220, 1070)
    top_plot_box = (1370, 218, 1900, 560)
    project, _, _, _ = make_projection(positions, main_plot_box)
    top_project = world_xy_projection(positions, top_plot_box)
    projected = project(positions)
    top_projected = top_project(positions)

    # Faint trajectory shadow gives depth to the oblique view.
    for index in range(1, len(projected)):
        start = tuple(projected[index - 1])
        end = tuple(projected[index])
        if not valid[index] or not valid[index - 1]:
            line_color = (136, 67, 75)
        else:
            line_color = color_ramp(index / max(1, len(projected) - 1))
        draw.line([start, end], fill=line_color, width=3)

    for index in range(1, len(top_projected)):
        start = tuple(top_projected[index - 1])
        end = tuple(top_projected[index])
        line_color = (132, 67, 77) if not valid[index] or not valid[index - 1] else color_ramp(index / max(1, len(top_projected) - 1))
        draw.line([start, end], fill=line_color, width=2)

    # Draw sampled camera frames.  The axis length is proportional to the full
    # trajectory extent, so frames remain visible without hiding the path.
    extent = float(np.linalg.norm(positions.max(axis=0) - positions.min(axis=0)))
    axis_length = max(extent * 0.025, 0.6)
    sample_count = min(36, len(positions))
    samples = np.linspace(0, len(positions) - 1, sample_count, dtype=int)
    for index in samples:
        center = positions[index]
        frame_points = np.vstack([center, center + rotations[index][:, 0] * axis_length, center + rotations[index][:, 1] * axis_length, center + rotations[index][:, 2] * axis_length])
        frame_projected = project(frame_points)
        origin = tuple(frame_projected[0])
        draw.ellipse((origin[0] - 5, origin[1] - 5, origin[0] + 5, origin[1] + 5), fill=(245, 245, 245))
        draw_arrow(draw, origin, tuple(frame_projected[1]), (235, 75, 82), 4)  # camera X
        draw_arrow(draw, origin, tuple(frame_projected[2]), (75, 205, 115), 4)  # camera Y
        draw_arrow(draw, origin, tuple(frame_projected[3]), (78, 145, 239), 4)  # camera Z

    start_2d = tuple(projected[0])
    end_2d = tuple(projected[-1])
    draw.ellipse((start_2d[0] - 9, start_2d[1] - 9, start_2d[0] + 9, start_2d[1] + 9), fill=(255, 255, 255), outline=(20, 20, 20), width=2)
    draw.ellipse((end_2d[0] - 9, end_2d[1] - 9, end_2d[0] + 9, end_2d[1] + 9), fill=(255, 151, 62), outline=(20, 20, 20), width=2)
    draw.text((start_2d[0] + 12, start_2d[1] - 24), "start", font=small_font, fill=(245, 248, 252))
    draw.text((end_2d[0] + 12, end_2d[1] - 24), "end", font=small_font, fill=(255, 183, 101))

    # Coordinate frame inset.
    frame_origin = (1735, 970)
    draw.ellipse((frame_origin[0] - 7, frame_origin[1] - 7, frame_origin[0] + 7, frame_origin[1] + 7), fill=(245, 245, 245))
    draw_arrow(draw, frame_origin, (1850, 970), (235, 75, 82), 6)
    draw_arrow(draw, frame_origin, (1735, 855), (75, 205, 115), 6)
    draw_arrow(draw, frame_origin, (1660, 1045), (78, 145, 239), 6)
    draw.text((1862, 951), "X camera", font=body_font, fill=(235, 75, 82))
    draw.text((1748, 826), "Y camera", font=body_font, fill=(75, 205, 115))
    draw.text((1565, 1044), "Z camera", font=body_font, fill=(78, 145, 239))

    # Summary text and axes notes.
    summary_lines = [
        f"TUM poses plotted: {len(positions):,}",
        f"Matched valid poses: {int(valid.sum()):,}",
        f"Invalid / lost poses: {int((~valid).sum()):,}",
        f"Position range: {np.ptp(positions, axis=0)[0]:.2f} x {np.ptp(positions, axis=0)[1]:.2f} x {np.ptp(positions, axis=0)[2]:.2f} m",
        "Red: camera X axis",
        "Green: camera Y axis",
        "Blue: camera Z axis",
        "Red path segments: invalid tracking",
    ]
    y = 680
    for line in summary_lines:
        draw.text((1352, y), line, font=body_font if y < 800 else small_font, fill=(206, 218, 231))
        y += 36

    # Start/end marker legend.
    draw.ellipse((1354, 1090, 1370, 1106), fill=(255, 255, 255))
    draw.text((1382, 1083), "start", font=small_font, fill=(206, 218, 231))
    draw.ellipse((1465, 1090, 1481, 1106), fill=(255, 151, 62))
    draw.text((1493, 1083), "end", font=small_font, fill=(206, 218, 231))

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)
    return {
        "poses": len(positions),
        "valid": int(valid.sum()),
        "invalid": int((~valid).sum()),
        "axis_length_m": axis_length,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=root / "outputs" / "20260802_150233_orbslam3_stereo" / "CameraTrajectory.txt",
    )
    parser.add_argument(
        "--tracking-log",
        type=Path,
        default=root / "outputs" / "20260802_150233_orbslam3_stereo" / "tracking_log.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "outputs" / "20260802_150233_orbslam3_stereo" / "camera_frames_3d.png",
    )
    args = parser.parse_args()
    summary = render(args.trajectory, args.tracking_log, args.output)
    print(f"Wrote {args.output.resolve()}")
    print(summary)


if __name__ == "__main__":
    main()

"""Render a metric top-view 2D plot of an ORB-SLAM3 TUM trajectory."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from visualize_camera_frames import color_ramp, font, read_tum_trajectory


def draw_arrow(draw: ImageDraw.ImageDraw, start: tuple[float, float], end: tuple[float, float], color, width: int = 4):
    draw.line([start, end], fill=color, width=width)
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = float(np.hypot(dx, dy))
    if length < 1e-6:
        return
    ux, uy = dx / length, dy / length
    size = max(8.0, width * 3.0)
    left = (end[0] - ux * size - uy * size * 0.55, end[1] - uy * size + ux * size * 0.55)
    right = (end[0] - ux * size + uy * size * 0.55, end[1] - uy * size - ux * size * 0.55)
    draw.polygon([end, left, right], fill=color)


def nice_step(span: float) -> float:
    raw = max(span / 6.0, 1e-9)
    exponent = np.floor(np.log10(raw))
    fraction = raw / (10.0 ** exponent)
    if fraction <= 1.0:
        nice = 1.0
    elif fraction <= 2.0:
        nice = 2.0
    elif fraction <= 5.0:
        nice = 5.0
    else:
        nice = 10.0
    return float(nice * (10.0 ** exponent))


def render(trajectory: Path, output: Path, plane: str) -> dict[str, float | int | str]:
    timestamps, positions, _ = read_tum_trajectory(trajectory)
    axis_map = {
        "xy": (0, 1, "X", "Y"),
        "xz": (0, 2, "X", "Z"),
        "yz": (1, 2, "Y", "Z"),
    }
    x_index, y_index, x_label, y_label = axis_map[plane]
    points = positions[:, [x_index, y_index]].astype(np.float64, copy=False)

    width, height = 1800, 1200
    image = Image.new("RGB", (width, height), (10, 16, 25))
    draw = ImageDraw.Draw(image)
    title_font = font(36, bold=True)
    section_font = font(25, bold=True)
    body_font = font(21)
    small_font = font(17)

    draw.text((55, 35), f"ORB-SLAM3 camera trajectory · {x_label}-{y_label} top view", font=title_font, fill=(245, 248, 252))
    draw.text(
        (57, 85),
        "Color changes from blue to orange with time; arrows show the travel direction",
        font=body_font,
        fill=(158, 177, 199),
    )

    plot_box = (120, 170, 1680, 1050)
    draw.rounded_rectangle(plot_box, radius=18, fill=(20, 29, 43), outline=(66, 86, 110), width=2)
    left, top, right, bottom = plot_box
    draw.text((left + 22, top + 16), f"World {x_label}-{y_label} plane (meters)", font=section_font, fill=(236, 242, 249))

    data_min = points.min(axis=0)
    data_max = points.max(axis=0)
    data_center = (data_min + data_max) * 0.5
    data_span = max(float(np.ptp(points, axis=0).max()), 1e-9)
    padding = data_span * 0.14
    view_span = data_span + 2.0 * padding
    cx = (left + right) * 0.5
    cy = (top + bottom) * 0.5 + 18
    scale = min((right - left - 80) / view_span, (bottom - top - 100) / view_span)

    def project(point: np.ndarray | list[float]) -> tuple[float, float]:
        px = cx + (float(point[0]) - float(data_center[0])) * scale
        py = cy - (float(point[1]) - float(data_center[1])) * scale
        return px, py

    # Equal-scale grid and tick labels.
    step = nice_step(data_span)
    x_start = np.floor((data_center[0] - view_span * 0.5) / step) * step
    x_stop = np.ceil((data_center[0] + view_span * 0.5) / step) * step
    y_start = np.floor((data_center[1] - view_span * 0.5) / step) * step
    y_stop = np.ceil((data_center[1] + view_span * 0.5) / step) * step
    grid_color = (44, 61, 82)
    label_color = (144, 164, 187)

    value = x_start
    while value <= x_stop + step * 0.5:
        px, _ = project([value, data_center[1]])
        if left + 35 <= px <= right - 35:
            draw.line([(px, top + 65), (px, bottom - 45)], fill=grid_color, width=1)
            draw.text((px - 24, bottom - 38), f"{value:.1f}", font=small_font, fill=label_color)
        value += step

    value = y_start
    while value <= y_stop + step * 0.5:
        _, py = project([data_center[0], value])
        if top + 65 <= py <= bottom - 45:
            draw.line([(left + 35, py), (right - 35, py)], fill=grid_color, width=1)
            draw.text((left + 6, py - 10), f"{value:.1f}", font=small_font, fill=label_color)
        value += step

    # Draw a subtle zero-axis only when it lies inside the displayed range.
    if data_center[0] - view_span * 0.5 <= 0.0 <= data_center[0] + view_span * 0.5:
        px, _ = project([0.0, data_center[1]])
        draw.line([(px, top + 65), (px, bottom - 45)], fill=(78, 101, 126), width=2)
    if data_center[1] - view_span * 0.5 <= 0.0 <= data_center[1] + view_span * 0.5:
        _, py = project([data_center[0], 0.0])
        draw.line([(left + 35, py), (right - 35, py)], fill=(78, 101, 126), width=2)

    projected = [project(point) for point in points]
    for index in range(1, len(projected)):
        draw.line([projected[index - 1], projected[index]], fill=color_ramp(index / max(1, len(projected) - 1)), width=4)

    arrow_stride = max(1, len(projected) // 24)
    for index in range(arrow_stride, len(projected), arrow_stride):
        draw_arrow(draw, projected[index - 1], projected[index], (235, 179, 75), width=3)

    start = projected[0]
    end = projected[-1]
    draw.ellipse((start[0] - 9, start[1] - 9, start[0] + 9, start[1] + 9), fill=(64, 218, 137), outline=(240, 255, 245), width=2)
    draw.ellipse((end[0] - 9, end[1] - 9, end[0] + 9, end[1] + 9), fill=(239, 91, 77), outline=(255, 244, 239), width=2)
    draw.text((start[0] + 14, start[1] - 25), "START", font=body_font, fill=(111, 235, 163))
    draw.text((end[0] + 14, end[1] - 25), "END", font=body_font, fill=(255, 133, 117))
    draw.text((right - 145, bottom - 38), f"{x_label} (m)", font=body_font, fill=(205, 219, 234))
    draw.text((left + 7, top + 68), f"{y_label} (m)", font=body_font, fill=(205, 219, 234))

    deltas = np.diff(positions, axis=0)
    path_length = float(np.linalg.norm(deltas, axis=1).sum())
    net_displacement = float(np.linalg.norm(positions[-1] - positions[0]))
    stats = (
        f"poses: {len(positions):,}    duration: {timestamps[-1] - timestamps[0]:.2f} s    "
        f"3D path: {path_length:.3f} m    start-end: {net_displacement:.3f} m"
    )
    draw.text((55, 1110), stats, font=body_font, fill=(158, 177, 199))
    draw.text((width - 425, 1110), "green START · red END", font=body_font, fill=(204, 216, 229))

    output.parent.mkdir(parents=True, exist_ok=True)
    image.save(output, format="PNG", optimize=True)
    return {
        "trajectory": str(trajectory),
        "plane": plane,
        "poses": int(len(positions)),
        "path_length_m": round(path_length, 3),
        "net_displacement_m": round(net_displacement, 3),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Render a top-view 2D ORB-SLAM3 trajectory")
    parser.add_argument(
        "--trajectory",
        type=Path,
        default=root / "output" / "20260802_150233_orbslam3_stereo" / "CameraTrajectory.txt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "output" / "20260802_150233_orbslam3_stereo" / "trajectory_top_view_xy.png",
    )
    parser.add_argument("--plane", choices=("xy", "xz", "yz"), default="xy")
    args = parser.parse_args()
    summary = render(args.trajectory, args.output, args.plane)
    print(f"Wrote {args.output.resolve()}")
    print(summary)


if __name__ == "__main__":
    main()

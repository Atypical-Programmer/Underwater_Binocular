"""Create a self-contained interactive 3D viewer for an ORB-SLAM3 trajectory.

The generated HTML uses only a canvas and inline JavaScript, so it can be opened
directly from disk without installing Plotly or starting a web server.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path

import numpy as np

from visualize_camera_frames import match_validity, read_tum_trajectory


def read_tracking_log(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    timestamps: list[float] = []
    frames: list[int] = []
    valid: list[bool] = []
    states: list[int] = []
    if not path.exists():
        return (np.empty(0), np.empty(0, dtype=np.int64), np.empty(0, dtype=bool), np.empty(0, dtype=np.int64))
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            timestamps.append(float(row["timestamp_sec"]))
            frames.append(int(row["frame"]))
            valid.append(row["pose_valid"] == "1")
            states.append(int(row["tracking_state"]))
    return (
        np.asarray(timestamps),
        np.asarray(frames, dtype=np.int64),
        np.asarray(valid, dtype=bool),
        np.asarray(states, dtype=np.int64),
    )


def nearest_log_rows(
    pose_timestamps: np.ndarray,
    log_timestamps: np.ndarray,
    log_frames: np.ndarray,
    log_states: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return matched frame IDs, tracking states, and match flags for TUM rows."""

    frame_ids = np.full(len(pose_timestamps), -1, dtype=np.int64)
    tracking_states = np.full(len(pose_timestamps), -1, dtype=np.int64)
    matched = np.zeros(len(pose_timestamps), dtype=bool)
    if len(log_timestamps) == 0:
        return frame_ids, tracking_states, matched

    for pose_index, timestamp in enumerate(pose_timestamps):
        insert = int(np.searchsorted(log_timestamps, timestamp))
        candidates = [index for index in (insert - 1, insert) if 0 <= index < len(log_timestamps)]
        if not candidates:
            continue
        nearest = min(candidates, key=lambda index: abs(float(log_timestamps[index] - timestamp)))
        if abs(float(log_timestamps[nearest] - timestamp)) <= 1e-3:
            frame_ids[pose_index] = int(log_frames[nearest])
            tracking_states[pose_index] = int(log_states[nearest])
            matched[pose_index] = True
    return frame_ids, tracking_states, matched


def rounded_vector(values: np.ndarray, digits: int = 7) -> list[float]:
    return [round(float(value), digits) for value in values]


def build_payload(trajectory: Path, tracking_log: Path, sample_count: int) -> dict:
    pose_timestamps, positions, rotations = read_tum_trajectory(trajectory)
    log_timestamps, log_frames, log_valid, log_states = read_tracking_log(tracking_log)
    valid = match_validity(pose_timestamps, log_timestamps, log_valid)
    frame_ids, tracking_states, matched = nearest_log_rows(
        pose_timestamps, log_timestamps, log_frames, log_states
    )

    bounds_min = positions.min(axis=0)
    bounds_max = positions.max(axis=0)
    center = (bounds_min + bounds_max) * 0.5
    scale = max(float(np.ptp(positions, axis=0).max()), 1e-9)
    normalized_positions = (positions - center) / scale

    extent = float(np.linalg.norm(bounds_max - bounds_min))
    axis_length_m = max(extent * 0.025, 0.6)
    axis_length = axis_length_m / scale

    sample_count = max(2, min(int(sample_count), len(positions)))
    sample_indices = np.linspace(0, len(positions) - 1, sample_count, dtype=int)
    # np.linspace can repeat indices for very short trajectories.
    sample_indices = np.unique(sample_indices)
    samples = []
    for pose_index in sample_indices:
        position = normalized_positions[pose_index]
        axes = [position + rotations[pose_index][:, axis] * axis_length for axis in range(3)]
        frame_id = int(frame_ids[pose_index]) if matched[pose_index] else None
        samples.append(
            {
                "pose_index": int(pose_index),
                "frame": frame_id,
                "timestamp": round(float(pose_timestamps[pose_index]), 6),
                "valid": bool(valid[pose_index]),
                "state": int(tracking_states[pose_index]),
                "position_m": rounded_vector(positions[pose_index], 5),
                "origin": rounded_vector(position),
                "axes": [rounded_vector(axis_endpoint) for axis_endpoint in axes],
            }
        )

    trajectory_data = [
        [round(float(position[0]), 7), round(float(position[1]), 7), round(float(position[2]), 7), int(valid[index])]
        for index, position in enumerate(normalized_positions)
    ]

    return {
        "trajectory": trajectory_data,
        "frames": samples,
        "stats": {
            "trajectory_file": trajectory.name,
            "tracking_file": tracking_log.name,
            "poses": int(len(positions)),
            "valid": int(valid.sum()),
            "invalid": int((~valid).sum()),
            "matched": int(matched.sum()),
            "axis_length_m": round(axis_length_m, 4),
            "range_m": rounded_vector(np.ptp(positions, axis=0), 3),
            "world_center_m": rounded_vector(center, 3),
        },
    }


HTML_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ORB-SLAM3 stereo camera frames</title>
<style>
:root {
  color-scheme: dark;
  --bg: #0a1019;
  --panel: #141d2b;
  --panel-2: #192538;
  --text: #edf3fa;
  --muted: #a9bbcf;
  --border: #31445d;
  --accent: #36c9b9;
}
* { box-sizing: border-box; }
html, body { width: 100%; height: 100%; margin: 0; overflow: hidden; background: var(--bg); color: var(--text); font-family: "Segoe UI", Arial, sans-serif; }
body { display: flex; flex-direction: column; }
.toolbar {
  min-height: 58px; display: flex; align-items: center; gap: 14px; flex-wrap: wrap;
  padding: 10px 16px; background: var(--panel); border-bottom: 1px solid var(--border);
  font-size: 14px;
}
.title { font-size: 18px; font-weight: 700; margin-right: 8px; white-space: nowrap; }
.subtle { color: var(--muted); }
button, .control {
  border: 1px solid #46617f; border-radius: 7px; background: var(--panel-2); color: var(--text);
  padding: 6px 10px; font: inherit;
}
button { cursor: pointer; }
button:hover { background: #223652; }
label { display: inline-flex; align-items: center; gap: 5px; white-space: nowrap; }
input[type="range"] { width: 120px; accent-color: var(--accent); vertical-align: middle; }
#viewer { position: relative; flex: 1 1 auto; min-height: 0; }
canvas { display: block; width: 100%; height: 100%; cursor: grab; }
canvas.dragging { cursor: grabbing; }
#help {
  position: absolute; left: 16px; bottom: 14px; max-width: 430px; padding: 8px 11px;
  border: 1px solid rgba(72, 96, 125, .75); border-radius: 7px; background: rgba(20, 29, 43, .88);
  color: var(--muted); font-size: 13px; pointer-events: none;
}
#tooltip {
  position: absolute; display: none; min-width: 210px; padding: 9px 11px; border-radius: 7px;
  border: 1px solid #5d7899; background: rgba(15, 23, 35, .95); box-shadow: 0 5px 20px rgba(0,0,0,.35);
  color: var(--text); font-size: 13px; line-height: 1.5; pointer-events: none; white-space: nowrap;
}
.footer { min-height: 30px; padding: 5px 16px; background: var(--panel); border-top: 1px solid var(--border); color: var(--muted); font-size: 12px; }
.legend { display: inline-flex; gap: 11px; margin-left: 18px; }
.dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 4px; }
.x { background: #eb4b57; } .y { background: #4dcc7a; } .z { background: #4d91ef; }
@media (max-width: 760px) {
  .title { width: 100%; }
  .toolbar { gap: 8px; }
  #help { font-size: 11px; max-width: 300px; }
}
</style>
</head>
<body>
<div class="toolbar">
  <div class="title">ORB-SLAM3 双目相机 Frame · 交互式 3D</div>
  <button id="reset">重置视角</button>
  <label><input id="showFrames" type="checkbox" checked>显示相机 frame</label>
  <label><input id="showFrustums" type="checkbox" checked>显示相机视锥</label>
  <label><input id="showInvalid" type="checkbox" checked>显示失效轨迹段</label>
  <label>frame 数量 <input id="frameCount" type="range" min="8" max="__MAX_FRAMES__" value="__MAX_FRAMES__"><span id="frameCountValue">__MAX_FRAMES__</span></label>
  <label>frame 大小 <input id="frameScale" type="range" min="0.35" max="2.50" step="0.05" value="1.00"><span id="frameScaleValue">1.00x</span></label>
  <span id="stats" class="subtle"></span>
</div>
<div id="viewer">
  <canvas id="canvas"></canvas>
  <div id="tooltip"></div>
  <div id="help">左键拖动：旋转 · 滚轮：缩放 · Shift+左键或右键拖动：平移 · 悬停采样 frame 查看信息</div>
</div>
<div class="footer">
  红 X / 绿 Y / 蓝 Z：相机坐标轴 · 青色：有效轨迹 · 红色：失效/丢跟踪段
  <span class="legend"><span><i class="dot x"></i>X</span><span><i class="dot y"></i>Y</span><span><i class="dot z"></i>Z</span></span>
</div>
<script>
"use strict";
const DATA = __PAYLOAD__;
const canvas = document.getElementById("canvas");
const viewer = document.getElementById("viewer");
const tooltip = document.getElementById("tooltip");
const showFrames = document.getElementById("showFrames");
const showFrustums = document.getElementById("showFrustums");
const showInvalid = document.getElementById("showInvalid");
const frameCount = document.getElementById("frameCount");
const frameCountValue = document.getElementById("frameCountValue");
const frameScale = document.getElementById("frameScale");
const frameScaleValue = document.getElementById("frameScaleValue");
const stats = document.getElementById("stats");
const ctx = canvas.getContext("2d");
let width = 0, height = 0, dpr = 1;
let view = { yaw: 0.78, pitch: -0.48, zoom: 2.1, panX: 0, panY: 0 };
let drag = null;
let hovered = null;

stats.textContent = `${DATA.stats.poses.toLocaleString()} poses · 有效 ${DATA.stats.valid.toLocaleString()} · 失效 ${DATA.stats.invalid.toLocaleString()} · 坐标范围 ${DATA.stats.range_m.map(v => v.toFixed(2)).join(" × ")} m`;

function resetView() {
  view = { yaw: 0.78, pitch: -0.48, zoom: 2.1, panX: 0, panY: 0 };
  draw();
}

function resize() {
  dpr = Math.min(window.devicePixelRatio || 1, 2);
  width = viewer.clientWidth;
  height = viewer.clientHeight;
  canvas.width = Math.max(1, Math.floor(width * dpr));
  canvas.height = Math.max(1, Math.floor(height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  draw();
}

function project(point) {
  const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw);
  const cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
  const x1 = cy * point[0] + sy * point[2];
  const z1 = -sy * point[0] + cy * point[2];
  const y2 = cp * point[1] - sp * z1;
  const z2 = sp * point[1] + cp * z1;
  const perspective = 3.0 + z2;
  const scale = Math.min(width, height) * 0.44 * view.zoom / perspective;
  return { x: width * 0.5 + view.panX + x1 * scale, y: height * 0.5 + view.panY - y2 * scale, depth: z2 };
}

function line3d(a, b, color, lineWidth, alpha = 1.0) {
  const p = project(a), q = project(b);
  ctx.globalAlpha = alpha;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
  ctx.globalAlpha = 1.0;
}

function drawWorldAxes() {
  const o = [-0.5, -0.5, -0.5];
  const length = 0.30;
  line3d(o, [o[0] + length, o[1], o[2]], "#657991", 1.2, .75);
  line3d(o, [o[0], o[1] + length, o[2]], "#657991", 1.2, .75);
  line3d(o, [o[0], o[1], o[2] + length], "#657991", 1.2, .75);
  const po = project(o);
  ctx.fillStyle = "#8ea2ba"; ctx.font = "12px Segoe UI, Arial";
  ctx.fillText("world", po.x + 5, po.y - 5);
}

function drawTrajectory() {
  function drawPath(isValid, color) {
    ctx.strokeStyle = color; ctx.lineWidth = 1.35; ctx.globalAlpha = .92;
    ctx.beginPath();
    let open = false;
    for (let i = 0; i < DATA.trajectory.length; i++) {
      const here = DATA.trajectory[i];
      const previous = i ? DATA.trajectory[i - 1] : null;
      const connected = isValid ? (here[3] && previous && previous[3]) : (!here[3] && previous && !previous[3]);
      if (connected) {
        const p = project(previous), q = project(here);
        if (!open) { ctx.moveTo(p.x, p.y); open = true; }
        ctx.lineTo(q.x, q.y);
      } else {
        open = false;
      }
    }
    ctx.stroke(); ctx.globalAlpha = 1.0;
  }
  drawPath(true, "#32c8b8");
  if (showInvalid.checked) drawPath(false, "#df616b");

  const start = project(DATA.trajectory[0]);
  const end = project(DATA.trajectory[DATA.trajectory.length - 1]);
  ctx.fillStyle = "#f5f7fa"; ctx.beginPath(); ctx.arc(start.x, start.y, 5, 0, Math.PI * 2); ctx.fill();
  ctx.fillStyle = "#ff9d3e"; ctx.beginPath(); ctx.arc(end.x, end.y, 5, 0, Math.PI * 2); ctx.fill();
  ctx.font = "12px Segoe UI, Arial"; ctx.fillText("start", start.x + 8, start.y - 8); ctx.fillText("end", end.x + 8, end.y - 8);
}

function visibleFrames() {
  const target = Math.min(Number(frameCount.value), DATA.frames.length);
  if (target >= DATA.frames.length) return DATA.frames;
  const result = [];
  for (let i = 0; i < target; i++) {
    const index = Math.round(i * (DATA.frames.length - 1) / Math.max(1, target - 1));
    result.push(DATA.frames[index]);
  }
  return result;
}

function add3(a, b) {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

function mul3(a, scalar) {
  return [a[0] * scalar, a[1] * scalar, a[2] * scalar];
}

function drawCameraFrustum(frame, hoveredFrame) {
  const origin = frame.origin;
  const xAxis = [frame.axes[0][0] - origin[0], frame.axes[0][1] - origin[1], frame.axes[0][2] - origin[2]];
  const yAxis = [frame.axes[1][0] - origin[0], frame.axes[1][1] - origin[1], frame.axes[1][2] - origin[2]];
  const zAxis = [frame.axes[2][0] - origin[0], frame.axes[2][1] - origin[1], frame.axes[2][2] - origin[2]];
  const size = Number(frameScale.value);
  // OpenCV/ORB-SLAM3 camera convention: +X right, +Y down, +Z forward.
  const forward = mul3(zAxis, 1.80 * size);
  const halfWidth = mul3(xAxis, 0.68 * size);
  const halfHeight = mul3(yAxis, 0.48 * size);
  const center = add3(origin, forward);
  const corners = [
    add3(add3(center, halfWidth), halfHeight),
    add3(add3(center, mul3(halfWidth, -1)), halfHeight),
    add3(add3(center, mul3(halfWidth, -1)), mul3(halfHeight, -1)),
    add3(add3(center, halfWidth), mul3(halfHeight, -1)),
  ];
  const edgeColor = frame.valid ? "#ffd166" : "#e27a81";
  const alpha = hoveredFrame ? 0.95 : (frame.valid ? 0.62 : 0.30);
  const lineWidth = hoveredFrame ? 2.8 : 1.35;
  for (const corner of corners) line3d(origin, corner, edgeColor, lineWidth, alpha);
  for (let index = 0; index < corners.length; index++) {
    line3d(corners[index], corners[(index + 1) % corners.length], edgeColor, lineWidth, alpha);
  }
  // A short optical-axis marker makes the viewing direction unambiguous.
  line3d(origin, center, "#ffffff", hoveredFrame ? 2.0 : 0.9, hoveredFrame ? 0.90 : 0.38);
}

function drawCameraFrames() {
  if (!showFrames.checked) return;
  for (const frame of visibleFrames()) {
    const hoveredFrame = hovered === frame;
    const alpha = frame.valid ? .88 : .42;
    const origin = frame.origin;
    if (showFrustums.checked) drawCameraFrustum(frame, hoveredFrame);
    line3d(origin, frame.axes[0], "#eb4b57", hoveredFrame ? 4.0 : 2.3, alpha);
    line3d(origin, frame.axes[1], "#4dcc7a", hoveredFrame ? 4.0 : 2.3, alpha);
    line3d(origin, frame.axes[2], "#4d91ef", hoveredFrame ? 4.0 : 2.3, alpha);
    const p = project(origin);
    ctx.fillStyle = hoveredFrame ? "#ffffff" : (frame.valid ? "#d9e5ee" : "#e27a81");
    ctx.beginPath(); ctx.arc(p.x, p.y, hoveredFrame ? 6 : 3.2, 0, Math.PI * 2); ctx.fill();
  }
}

function drawInset() {
  const ox = width - 96, oy = height - 78, length = 30;
  ctx.save(); ctx.globalAlpha = .95; ctx.lineWidth = 2;
  ctx.strokeStyle = "#eb4b57"; ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox + length, oy); ctx.stroke();
  ctx.strokeStyle = "#4dcc7a"; ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox, oy - length); ctx.stroke();
  ctx.strokeStyle = "#4d91ef"; ctx.beginPath(); ctx.moveTo(ox, oy); ctx.lineTo(ox - 18, oy + 17); ctx.stroke();
  ctx.fillStyle = "#eb4b57"; ctx.font = "12px Segoe UI, Arial"; ctx.fillText("X", ox + length + 5, oy + 4);
  ctx.fillStyle = "#4dcc7a"; ctx.fillText("Y", ox - 4, oy - length - 6);
  ctx.fillStyle = "#4d91ef"; ctx.fillText("Z", ox - 30, oy + 26);
  ctx.restore();
}

function draw() {
  ctx.clearRect(0, 0, width, height);
  const gradient = ctx.createLinearGradient(0, 0, 0, height);
  gradient.addColorStop(0, "#0c1421"); gradient.addColorStop(1, "#080d15");
  ctx.fillStyle = gradient; ctx.fillRect(0, 0, width, height);
  drawWorldAxes();
  drawTrajectory();
  drawCameraFrames();
  drawInset();
}

function updateTooltip(event) {
  if (drag || !showFrames.checked) { tooltip.style.display = "none"; hovered = null; return; }
  const rect = canvas.getBoundingClientRect();
  const x = event.clientX - rect.left, y = event.clientY - rect.top;
  let best = null, bestDistance = 15;
  for (const frame of visibleFrames()) {
    const p = project(frame.origin);
    const distance = Math.hypot(p.x - x, p.y - y);
    if (distance < bestDistance) { best = frame; bestDistance = distance; }
  }
  hovered = best;
  if (!best) { tooltip.style.display = "none"; draw(); return; }
  const frameText = best.frame === null ? `pose #${best.pose_index}` : `frame ${best.frame}`;
  const validText = best.valid ? "有效" : "失效/丢跟踪";
  tooltip.innerHTML = `<b>${frameText}</b> · ${validText}<br>timestamp: ${best.timestamp.toFixed(6)}<br>位置(m): [${best.position_m.map(v => v.toFixed(3)).join(", ")}]<br>tracking state: ${best.state}`;
  const left = Math.min(width - 230, Math.max(8, x + 14));
  const top = Math.min(height - 105, Math.max(8, y + 14));
  tooltip.style.left = `${left}px`; tooltip.style.top = `${top}px`; tooltip.style.display = "block";
  draw();
}

canvas.addEventListener("mousedown", event => {
  drag = { x: event.clientX, y: event.clientY, pan: event.shiftKey || event.button === 2 };
  canvas.classList.add("dragging");
  tooltip.style.display = "none";
  event.preventDefault();
});
window.addEventListener("mouseup", () => { drag = null; canvas.classList.remove("dragging"); });
window.addEventListener("mousemove", event => {
  if (drag) {
    const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
    drag.x = event.clientX; drag.y = event.clientY;
    if (drag.pan) { view.panX += dx; view.panY += dy; }
    else { view.yaw += dx * 0.008; view.pitch = Math.max(-1.45, Math.min(1.45, view.pitch + dy * 0.008)); }
    draw();
  } else {
    updateTooltip(event);
  }
});
canvas.addEventListener("wheel", event => {
  view.zoom = Math.max(.18, Math.min(8.0, view.zoom * Math.exp(-event.deltaY * .001)));
  draw(); event.preventDefault();
}, { passive: false });
canvas.addEventListener("contextmenu", event => event.preventDefault());
document.getElementById("reset").addEventListener("click", resetView);
showFrames.addEventListener("change", draw);
showFrustums.addEventListener("change", draw);
showInvalid.addEventListener("change", draw);
frameCount.addEventListener("input", () => { frameCountValue.textContent = frameCount.value; hovered = null; tooltip.style.display = "none"; draw(); });
frameScale.addEventListener("input", () => { frameScaleValue.textContent = `${Number(frameScale.value).toFixed(2)}x`; draw(); });
window.addEventListener("resize", resize);
resize();
</script>
</body>
</html>
'''


def render(trajectory: Path, tracking_log: Path, output: Path, sample_count: int) -> dict[str, int | float]:
    payload = build_payload(trajectory, tracking_log, sample_count)
    frame_count = len(payload["frames"])
    serialized = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # Numeric data cannot contain HTML, but escaping the closing tag makes this
    # safe if the payload format is extended later with user-provided strings.
    serialized = serialized.replace("</", "<\\/")
    html_text = (
        HTML_TEMPLATE.replace("__PAYLOAD__", serialized)
        .replace("__MAX_FRAMES__", str(frame_count))
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")
    return {
        "poses": int(payload["stats"]["poses"]),
        "valid": int(payload["stats"]["valid"]),
        "invalid": int(payload["stats"]["invalid"]),
        "sampled_frames": frame_count,
        "bytes": len(html_text.encode("utf-8")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a self-contained interactive 3D ORB-SLAM3 viewer")
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
        default=root / "outputs" / "20260802_150233_orbslam3_stereo" / "camera_frames_interactive.html",
    )
    parser.add_argument("--sample-count", type=int, default=80, help="number of camera frames shown by default")
    args = parser.parse_args()
    summary = render(args.trajectory, args.tracking_log, args.output, args.sample_count)
    print(f"Wrote {args.output.resolve()}")
    print(summary)


if __name__ == "__main__":
    main()

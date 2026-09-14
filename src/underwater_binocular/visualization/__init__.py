"""Visualization entrypoints consume package data models and do not own I/O handles."""

from ..depth.visualization import export_depth_video, render_depth_bgr

__all__ = ["export_depth_video", "render_depth_bgr"]

"""OpenCV rectification independent of the ZED SDK."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..calibration.models import StereoCalibration


@dataclass(frozen=True)
class RectifiedStereoModel:
    """Rectified camera model and maps for one calibration/size/alpha choice."""

    input_size: tuple[int, int]
    output_size: tuple[int, int]
    alpha: float
    r_left: np.ndarray
    r_right: np.ndarray
    p_left: np.ndarray
    p_right: np.ndarray
    q: np.ndarray
    map_left_x: np.ndarray
    map_left_y: np.ndarray
    map_right_x: np.ndarray
    map_right_y: np.ndarray
    focal_length_px: float
    baseline_m: float
    roi_left: tuple[int, int, int, int]
    roi_right: tuple[int, int, int, int]

    def rectify_pair(self, left_image: np.ndarray, right_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Remap one RAW_UNRECTIFIED image pair into RECTIFIED coordinates."""

        import cv2

        left = cv2.remap(left_image, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
        right = cv2.remap(right_image, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
        return left, right


def rectify_calibration(
    calibration: StereoCalibration,
    *,
    output_size: tuple[int, int] | None = None,
    alpha: float = 0.0,
    scale: float = 1.0,
) -> RectifiedStereoModel:
    """Build OpenCV ``stereoRectify`` geometry using metre translations."""

    import cv2

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("rectification alpha must be in [0, 1]")
    if not 0.0 < scale <= 1.0:
        raise ValueError("rectification scale must be in (0, 1]")
    input_size = calibration.resolution
    if output_size is None:
        output_size = (round(input_size[0] * scale), round(input_size[1] * scale))
    if output_size[0] <= 0 or output_size[1] <= 0:
        raise ValueError("rectified output size must be positive")
    k_left = calibration.left.matrix.copy()
    k_right = calibration.right.matrix.copy()
    rotation = np.asarray(calibration.rotation_left_to_right, dtype=np.float64)
    translation = np.asarray(calibration.translation_left_to_right_m, dtype=np.float64).reshape(3, 1)
    r_left, r_right, p_left, p_right, q, roi_left, roi_right = cv2.stereoRectify(
        k_left,
        np.asarray(calibration.left.distortion[:5], dtype=np.float64),
        k_right,
        np.asarray(calibration.right.distortion[:5], dtype=np.float64),
        input_size,
        rotation,
        translation,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=float(alpha),
        newImageSize=output_size,
    )
    map_left_x, map_left_y = cv2.initUndistortRectifyMap(k_left, calibration.left.distortion[:5], r_left, p_left, output_size, cv2.CV_32FC1)
    map_right_x, map_right_y = cv2.initUndistortRectifyMap(k_right, calibration.right.distortion[:5], r_right, p_right, output_size, cv2.CV_32FC1)
    focal = float(p_left[0, 0])
    baseline = abs(float(p_right[0, 3] / p_right[0, 0]))
    if not np.isfinite(focal) or focal <= 0.0 or not np.isfinite(baseline) or baseline <= 0.0:
        raise ValueError(f"invalid rectified focal/baseline: {focal}, {baseline}")
    return RectifiedStereoModel(
        input_size=input_size,
        output_size=output_size,
        alpha=float(alpha),
        r_left=r_left,
        r_right=r_right,
        p_left=p_left,
        p_right=p_right,
        q=q,
        map_left_x=map_left_x,
        map_left_y=map_left_y,
        map_right_x=map_right_x,
        map_right_y=map_right_y,
        focal_length_px=focal,
        baseline_m=baseline,
        roi_left=tuple(int(value) for value in roi_left),
        roi_right=tuple(int(value) for value in roi_right),
    )

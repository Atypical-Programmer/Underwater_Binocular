"""Real ALIKED extraction and cache handling for the SfM workflow.

The implementation follows the existing legacy pipeline's LightGlue ALIKED
adapter, but keeps optional torch/LightGlue imports inside the execution path.
No ORB fallback is provided here: a missing ALIKED dependency is an explicit
error.
"""

from __future__ import annotations

import importlib.metadata
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..calibration.loaders import sha256_file
from .features import FeatureSet, ImageRecord, restore_keypoints_to_original


class AlikedDependencyError(RuntimeError):
    """Raised when the requested ALIKED runtime is not installed."""


class AlikedConfig:
    """Configuration for the real LightGlue ALIKED extractor."""

    def __init__(
        self,
        *,
        model_name: str = "aliked-n16",
        resize: int | None = 1024,
        max_keypoints: int = 800,
        detection_threshold: float = 0.2,
        nms_radius: int = 2,
        device: str = "cuda",
    ) -> None:
        self.model_name = str(model_name)
        self.resize = None if resize in (None, 0) else int(resize)
        self.max_keypoints = int(max_keypoints)
        self.detection_threshold = float(detection_threshold)
        self.nms_radius = int(nms_radius)
        self.device = str(device)
        self.validate()

    def validate(self) -> None:
        if not self.model_name:
            raise ValueError("ALIKED model_name must be non-empty")
        if self.resize is not None and self.resize <= 0:
            raise ValueError("ALIKED resize must be positive or 0/None to disable resizing")
        if self.max_keypoints <= 0:
            raise ValueError("ALIKED max_keypoints must be positive")
        if not np.isfinite(self.detection_threshold):
            raise ValueError("ALIKED detection_threshold must be finite")
        if self.nms_radius < 0:
            raise ValueError("ALIKED nms_radius must be non-negative")
        if not self.device:
            raise ValueError("ALIKED device must be explicit")

    def to_mapping(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "resize": self.resize,
            "max_keypoints": self.max_keypoints,
            "detection_threshold": self.detection_threshold,
            "nms_radius": self.nms_radius,
            "device": self.device,
        }


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_torch_device(name: str) -> tuple[Any, Any]:
    """Import torch and resolve an explicit CPU/CUDA device without fallback."""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on optional install
        raise AlikedDependencyError(
            "ALIKED requires torch; install the project's [sfm] extra"
        ) from error
    requested = str(name).lower()
    if requested == "cuda":
        requested = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cuda"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for ALIKED but torch.cuda.is_available() is false")
    if requested not in {"cpu"} and not requested.startswith("cuda:"):
        raise ValueError("ALIKED device must be cpu, cuda, or cuda:<index>")
    device = torch.device(requested)
    if device.type == "cuda":
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"requested CUDA device {device.index} is not available; "
                f"device_count={torch.cuda.device_count()}"
            )
    return torch, device


def _read_rgb_tensor(path: Path, torch: Any, device: Any) -> tuple[Any, tuple[int, int]]:
    try:
        import cv2
    except ImportError as error:  # pragma: no cover - core dependency in normal installs
        raise AlikedDependencyError("ALIKED image extraction requires OpenCV") from error
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"OpenCV could not read image {path}")
    height, width = bgr.shape[:2]
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).contiguous().float().div_(255.0)
    return tensor.to(device), (int(height), int(width))


def _resize_tensor(
    image: Any, torch: Any, *, max_side: int | None
) -> tuple[Any, tuple[int, int]]:
    """Resize one tensor explicitly and return its processed H/W."""

    height, width = (int(value) for value in image.shape[-2:])
    if max_side is None or max(height, width) <= max_side:
        return image, (height, width)
    scale = max_side / max(height, width)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(width * scale)))
    if (new_height, new_width) == (height, width):
        return image, (height, width)
    import torch.nn.functional as functional

    resized = functional.interpolate(
        image[None], size=(new_height, new_width), mode="bilinear", align_corners=False
    )[0]
    return resized.contiguous(), (new_height, new_width)


def _first_batch(value: Any) -> np.ndarray:
    array = value.detach().cpu().numpy()
    if array.ndim >= 1 and array.shape[0] == 1:
        array = array[0]
    return np.asarray(array)


def _extract_one(model: Any, path: Path, config: AlikedConfig, torch: Any, device: Any) -> FeatureSet:
    image, original_size = _read_rgb_tensor(path, torch, device)
    processed, processed_size = _resize_tensor(image, torch, max_side=config.resize)
    with torch.inference_mode():
        # resize=None is intentional: the adapter controls the resize so the
        # inverse coordinate transform below is deterministic and testable.
        output = model.extract(processed, resize=None)
    keypoints_processed = _first_batch(output["keypoints"]).astype(np.float32, copy=False)
    descriptors = _first_batch(output["descriptors"]).astype(np.float32, copy=False)
    scores = _first_batch(output["keypoint_scores"]).astype(np.float32, copy=False).reshape(-1)
    keypoints = restore_keypoints_to_original(
        keypoints_processed,
        original_size_hw=original_size,
        extracted_size_hw=processed_size,
        coordinate_space="resized_image_pixels",
    )
    if descriptors.ndim != 2 or len(descriptors) != len(keypoints):
        raise RuntimeError(
            f"ALIKED returned incompatible keypoints/descriptors for {path}: "
            f"{keypoints.shape} and {descriptors.shape}"
        )
    if len(scores) != len(keypoints):
        raise RuntimeError(f"ALIKED returned incompatible scores for {path}")
    return FeatureSet(
        image_path=path,
        keypoints_xy=np.ascontiguousarray(keypoints, dtype=np.float32),
        descriptors=np.ascontiguousarray(descriptors, dtype=np.float32),
        model=f"ALIKED/{config.model_name}",
        scores=np.ascontiguousarray(scores, dtype=np.float32),
        image_size_hw=original_size,
        coordinate_space="original_image_pixels",
        resize=config.resize,
    )


def _cache_identity(record: ImageRecord, config: AlikedConfig) -> dict[str, Any]:
    return {
        "image_name": record.name,
        "side": record.side,
        "frame": int(record.frame),
        "timestamp_ns": int(record.timestamp_ns),
        "image_sha256": sha256_file(record.path),
        "model_name": config.model_name,
        "resize": config.resize,
        "max_keypoints": config.max_keypoints,
        "detection_threshold": config.detection_threshold,
        "nms_radius": config.nms_radius,
        "coordinate_space": "original_image_pixels",
    }


def _cache_path(feature_dir: Path, index: int) -> Path:
    return feature_dir / f"{index:06d}.npz"


def _load_cache(path: Path, record: ImageRecord) -> FeatureSet:
    with np.load(path, allow_pickle=False) as data:
        model = str(np.asarray(data["model"]).reshape(-1)[0])
        resize_value = int(np.asarray(data["resize"]).reshape(-1)[0])
        return FeatureSet(
            image_path=record.path,
            keypoints_xy=np.asarray(data["keypoints"], dtype=np.float32),
            descriptors=np.asarray(data["descriptors"], dtype=np.float32),
            model=model,
            scores=np.asarray(data["scores"], dtype=np.float32),
            image_size_hw=tuple(int(value) for value in np.asarray(data["image_size_hw"]).reshape(-1)),
            coordinate_space=str(np.asarray(data["coordinate_space"]).reshape(-1)[0]),
            resize=None if resize_value < 0 else resize_value,
            image_name=record.name,
            camera_side=record.side,
        )


def _save_cache(path: Path, feature: FeatureSet) -> None:
    scores = feature.scores
    if scores is None:
        scores = np.empty((len(feature.keypoints_xy),), dtype=np.float32)
    if feature.image_size_hw is None:
        raise ValueError("ALIKED cache requires image_size_hw")
    np.savez_compressed(
        path,
        keypoints=np.asarray(feature.keypoints_xy, dtype=np.float32),
        descriptors=np.asarray(feature.descriptors, dtype=np.float32),
        scores=np.asarray(scores, dtype=np.float32),
        image_size_hw=np.asarray(feature.image_size_hw, dtype=np.int32),
        model=np.asarray([feature.model]),
        resize=np.asarray([-1 if feature.resize is None else feature.resize], dtype=np.int32),
        coordinate_space=np.asarray([feature.coordinate_space]),
    )


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def extract_aliked_features(
    image_records: Sequence[ImageRecord],
    feature_dir: Path,
    config: AlikedConfig,
    *,
    resume: bool = True,
) -> tuple[list[FeatureSet], dict[str, Any]]:
    """Extract or load real ALIKED features with configuration-aware caching."""

    config.validate()
    if not image_records:
        raise ValueError("ALIKED feature extraction requires at least one image")
    feature_dir = feature_dir.expanduser().resolve()
    feature_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = feature_dir / "cache_manifest.json"
    old_manifest = _load_manifest(manifest_path)
    old_entries = old_manifest.get("entries", {})
    if not isinstance(old_entries, dict):
        old_entries = {}
    identities = {
        str(index): _cache_identity(record, config)
        for index, record in enumerate(image_records)
    }
    features: list[FeatureSet | None] = [None] * len(image_records)
    cache_hits = 0
    missing: list[int] = []
    for index, record in enumerate(image_records):
        cache_path = _cache_path(feature_dir, index)
        if resume and cache_path.is_file() and old_entries.get(str(index)) == identities[str(index)]:
            features[index] = _load_cache(cache_path, record)
            cache_hits += 1
        else:
            missing.append(index)

    started = time.perf_counter()
    torch = None
    device = None
    model = None
    if missing:
        torch, device = load_torch_device(config.device)
        try:
            from lightglue import ALIKED
        except ImportError as error:  # pragma: no cover - optional dependency
            raise AlikedDependencyError(
                "ALIKED requires the LightGlue package; install the project's [sfm] extra"
            ) from error
        model = ALIKED(
            model_name=config.model_name,
            max_num_keypoints=config.max_keypoints,
            detection_threshold=config.detection_threshold,
            nms_radius=config.nms_radius,
        ).eval().to(device)
        for index in missing:
            record = image_records[index]
            feature = _extract_one(model, record.path, config, torch, device)
            features[index] = FeatureSet(
                image_path=feature.image_path,
                keypoints_xy=feature.keypoints_xy,
                descriptors=feature.descriptors,
                model=feature.model,
                scores=feature.scores,
                image_size_hw=feature.image_size_hw,
                coordinate_space=feature.coordinate_space,
                resize=feature.resize,
                image_name=record.name,
                camera_side=record.side,
            )
            _save_cache(_cache_path(feature_dir, index), features[index])
        del model
        if device is not None and device.type == "cuda":
            torch.cuda.empty_cache()

    completed = [value for value in features if value is not None]
    if len(completed) != len(image_records):
        raise RuntimeError("ALIKED feature extraction did not produce every feature set")
    cache_manifest = {
        "algorithm": "ALIKED",
        "implementation": "lightglue.ALIKED",
        "config": config.to_mapping(),
        "entries": identities,
        "software": {
            "torch": _package_version("torch"),
            "lightglue": _package_version("lightglue"),
        },
    }
    manifest_path.write_text(
        json.dumps(cache_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    keypoint_counts = [len(feature.keypoints_xy) for feature in completed]
    summary = {
        "algorithm": "ALIKED",
        "implementation": "lightglue.ALIKED",
        "config": config.to_mapping(),
        "num_images": len(completed),
        "num_left_images": sum(feature.camera_side == "left" for feature in completed),
        "num_right_images": sum(feature.camera_side == "right" for feature in completed),
        "total_keypoints": int(sum(keypoint_counts)),
        "min_keypoints": int(min(keypoint_counts)),
        "median_keypoints": float(np.median(keypoint_counts)),
        "mean_keypoints": float(np.mean(keypoint_counts)),
        "max_keypoints_observed": int(max(keypoint_counts)),
        "cache_hits": cache_hits,
        "cache_misses": len(missing),
        "feature_coordinate_space": "original_image_pixels",
        "elapsed_seconds": time.perf_counter() - started,
        "software": cache_manifest["software"],
    }
    (feature_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return completed, summary

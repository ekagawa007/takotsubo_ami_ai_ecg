"""Serialize Grad-CAM artifacts without changing prediction values.

Only generic sample IDs enter artifact names. Images are re-encoded as RGB PNG
without source metadata. All payloads are returned to the CLI's protected save.
"""

import copy
import hashlib
import io
import math
from pathlib import Path

from .settings import IMAGE_SIZE


def validate_cam_options(model, target, normalization, opacity):
    """Lightweight validation used before loading inference dependencies."""
    from .gradcam import CAM_NORMALIZATIONS, CAM_TARGETS, selected_cam_models

    names = selected_cam_models(model)
    if not names:
        raise ValueError("Grad-CAM is available for ViT and ConvNeXtV2, not MLP-Mixer alone.")
    if target not in CAM_TARGETS or normalization not in CAM_NORMALIZATIONS:
        raise ValueError("Unsupported Grad-CAM target or normalization.")
    if target == "ensemble-predicted" and model != "ensemble":
        raise ValueError("The ensemble-predicted CAM target requires --model ensemble.")
    if not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("CAM opacity must be finite and between 0 and 1.")
    return names


def artifact_paths(output, image_count, models):
    """Deterministic destinations, known before any model is loaded."""
    output = Path(output)
    if type(image_count) is not int or image_count < 1:
        raise ValueError("Grad-CAM requires at least one image.")
    if not models or len(set(models)) != len(models) or any(name not in ("vit", "convnextv2") for name in models):
        raise ValueError("Unsupported Grad-CAM model selection.")
    paths = []
    for index in range(1, image_count + 1):
        prefix = f"{output.stem}_sample_{index:06d}"
        paths.append(output.parent / f"{prefix}_input.png")
        for name in models:
            paths.extend(output.parent / f"{prefix}_{name}_{suffix}" for suffix in
                         ("heatmap.png", "overlay.png", "cam.npy"))
    return paths


def _png(array):
    from PIL import Image

    stream = io.BytesIO()
    # Construct a fresh image rather than copying source .info/EXIF/ICC fields.
    Image.fromarray(array).save(stream, format="PNG")
    return stream.getvalue()


def render_map(rgb, values, opacity, is_constant=False):
    """A linear blue-to-yellow scale; uniform maps have a neutral display."""
    import numpy as np

    values = np.asarray(values)
    if values.shape != (IMAGE_SIZE, IMAGE_SIZE) or not np.isfinite(values).all():
        raise ValueError("Grad-CAM must be a finite 480x480 map.")
    if (values < 0).any() or (values > 1).any():
        raise ValueError("Grad-CAM values must lie between 0 and 1.")
    rgb = np.asarray(rgb)
    if rgb.shape != (IMAGE_SIZE, IMAGE_SIZE, 3) or rgb.dtype != np.uint8:
        raise ValueError("Grad-CAM display requires a 480x480 RGB image.")
    if not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("CAM opacity must be finite and between 0 and 1.")
    constant = bool(is_constant or float(np.ptp(values)) == 0.0)
    if constant:
        return np.full_like(rgb, 128), rgb.copy(), True
    low = np.array((30, 56, 130), dtype=np.float32)
    high = np.array((255, 224, 64), dtype=np.float32)
    colors = low + values[..., None].astype(np.float32) * (high - low)
    heatmap = np.rint(colors).clip(0, 255).astype(np.uint8)
    overlay = np.rint(rgb.astype(np.float32) * (1 - opacity) + heatmap.astype(np.float32) * opacity)
    return heatmap, overlay.clip(0, 255).astype(np.uint8), False


def serialize_gradcam(images, result, cam_result, output, opacity=0.6):
    """Attach manifests and return (path, bytes) pairs for one protected save.

    No files are written here. The original probability/logit fields remain
    unchanged. Per-image manifests also live in metadata for CSV sidecars.
    """
    import numpy as np
    from .image_input import open_supported_image, prepare_image_geometry
    from .inference import sha256_file
    from .security import require_patched_pillow

    require_patched_pillow()
    if not math.isfinite(opacity) or not 0 <= opacity <= 1:
        raise ValueError("CAM opacity must be finite and between 0 and 1.")
    rows = result["predictions"]
    maps = cam_result["images"]
    if len(images) != len(rows) or len(maps) != len(rows):
        raise ValueError("Grad-CAM and prediction image counts differ.")
    model = result["metadata"]["model"]
    from .gradcam import selected_cam_models
    names = selected_cam_models(model)
    expected_paths = artifact_paths(output, len(images), names)
    output = Path(output)
    contents, artifacts, manifests = [], [], []

    def append(path, data):
        contents.append((path, data))
        artifacts.append({"filename": path.name, "sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)})

    for index, (path, row, record) in enumerate(zip(images, rows, maps), 1):
        sample_id = f"sample_{index:06d}"
        if row["sample_id"] != sample_id or record["sample_id"] != sample_id or row["input_sha256"] != record["input_sha256"]:
            raise ValueError("Grad-CAM image correspondence differs from predictions.")
        if sha256_file(path) != row["input_sha256"]:
            raise ValueError("An input image changed before Grad-CAM rendering.")
        with open_supported_image(path, index) as source:
            image = prepare_image_geometry(source.convert("RGB"), result["metadata"]["settings"]["input_policy"])
            rgb = np.array(image, dtype=np.uint8, copy=True)
        if sha256_file(path) != row["input_sha256"]:
            raise ValueError("An input image changed during Grad-CAM rendering.")
        if set(record["models"]) != set(names):
            raise ValueError("Grad-CAM model correspondence differs from predictions.")
        prefix = f"{output.stem}_{sample_id}"
        input_path = output.parent / f"{prefix}_input.png"
        append(input_path, _png(rgb))
        manifest = {"input_image": input_path.name, "models": {}}
        for name in names:
            entry = record["models"][name]
            values = np.asarray(entry["map"], dtype=np.float32)
            heatmap, overlay, constant = render_map(rgb, values, opacity, entry["map_is_constant"])
            heat_path = output.parent / f"{prefix}_{name}_heatmap.png"
            overlay_path = output.parent / f"{prefix}_{name}_overlay.png"
            values_path = output.parent / f"{prefix}_{name}_cam.npy"
            append(heat_path, _png(heatmap))
            append(overlay_path, _png(overlay))
            stream = io.BytesIO()
            np.save(stream, values, allow_pickle=False)
            append(values_path, stream.getvalue())
            public_entry = {key: copy.deepcopy(entry[key]) for key in
                            ("target_id", "target_label", "target_source", "layers", "unaugmented_logits", "normalization")}
            public_entry["map_is_constant"] = constant
            public_entry["files"] = {"heatmap": heat_path.name, "overlay": overlay_path.name, "values": values_path.name}
            manifest["models"][name] = public_entry
        row["gradcam"] = manifest
        manifests.append({"sample_id": sample_id, "input_sha256": row["input_sha256"], **copy.deepcopy(manifest)})
    if [path for path, _ in contents] != expected_paths:
        raise ValueError("Grad-CAM output paths differ from the checked destinations.")
    result["metadata"]["gradcam"] = {
        **copy.deepcopy(cam_result["metadata"]),
        "rendering": {"colormap": "linear_blue_yellow", "low_rgb": [30, 56, 130], "high_rgb": [255, 224, 64],
                      "opacity": opacity, "constant_maps": "neutral_gray_heatmap_unchanged_input_overlay",
                      "intensity_is_probability": False},
        "artifacts": artifacts,
        "images": manifests,
    }
    return contents

"""Small, testable helpers for the Colab example; no kernel-side ML imports."""

import argparse
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

from .settings import ModelSettings


INPUT_KINDS = ("generated_synthetic_trace", "user_upload", "google_drive_file")


def choose_device(requested, cuda_available):
    """Prefer CUDA, but explicitly report every CPU path to the notebook."""
    if requested not in ("auto", "cuda", "cpu"):
        raise ValueError("Device must be auto, cuda or cpu.")
    if type(cuda_available) is not bool:
        raise ValueError("CUDA availability must be a boolean.")
    if requested == "cpu":
        return "cpu", "CPU selected. Paper TTA and Grad-CAM may take substantially longer than on a GPU."
    if cuda_available:
        return "cuda", None
    return "cpu", "GPU is unavailable; using CPU. Paper TTA is unchanged. Choose No TTA or fewer Custom views if needed."


def resolve_tta(mode, vit_views=24, vit_magnitude=1, convnextv2_views=1,
                convnextv2_magnitude=5, mlpmixer_views=32, mlpmixer_magnitude=3):
    """Return explicit per-model settings; a CPU selection never changes TTA."""
    if mode == "Paper defaults":
        values = ((24, 1), (1, 5), (32, 3))
    elif mode == "No TTA":
        values = ((1, 1), (1, 5), (1, 3))
    elif mode == "Custom":
        values = ((vit_views, vit_magnitude), (convnextv2_views, convnextv2_magnitude),
                  (mlpmixer_views, mlpmixer_magnitude))
    else:
        raise ValueError("TTA preset must be Paper defaults, No TTA or Custom.")
    result = {}
    for name, (views, magnitude) in zip(("vit", "convnextv2", "mlpmixer"), values):
        ModelSettings(views, magnitude).validate()
        result[name] = {"tta": views, "magnitude": magnitude}
    return result


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_synthetic(path):
    """Create the same synthetic image as the v0.3.1 example."""
    import math
    from .security import require_patched_pillow
    from .safe_output import write_files
    require_patched_pillow()
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (480, 480), "white")
    draw = ImageDraw.Draw(image)
    for position in range(0, 480, 12):
        draw.line((position, 0, position, 479), fill=(255, 225, 225))
        draw.line((0, position, 479, position), fill=(255, 225, 225))
    for row in range(4):
        points = [(x, 60 + row * 110 + int(8 * math.sin(x / 13) -
                  (30 if x % 90 == 45 else 0))) for x in range(480)]
        draw.line(points, fill=(35, 35, 35), width=2)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    payload = buffer.getvalue()
    path = Path(path)
    if path.exists() or path.is_symlink():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != payload:
            raise ValueError("The synthetic input path contains different data.")
    else:
        write_files([(path, payload)])


def inspect_input(path, kind, input_policy, preview_path=None):
    """Decode only in the inference environment and prepare a verified PNG."""
    from .image_input import open_supported_image, describe_image_geometry, prepare_image_geometry
    from .safe_output import write_files

    if kind not in INPUT_KINDS:
        raise ValueError("Unexpected input source type.")
    path = Path(path)
    if not path.is_file() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Choose an existing regular image file.")
    before = _sha(path)
    with open_supported_image(path, 1) as source:
        source.load()
        geometry = describe_image_geometry(source.size, input_policy)
        rgb = source.convert("RGB")
        prepared = prepare_image_geometry(rgb, input_policy)
        record = {
            "kind": kind, "size": list(source.size), "mode": source.mode,
            "format": source.format, "input_sha256": before,
            "rgb_pixel_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
            "geometry": geometry, "prepared_size": list(prepared.size),
            "prepared_rgb_pixel_sha256": hashlib.sha256(prepared.tobytes()).hexdigest(),
        }
        if preview_path is not None:
            buffer = BytesIO()
            prepared.save(buffer, format="PNG")
            preview = buffer.getvalue()
    if _sha(path) != before:
        raise ValueError("The input changed during preparation. Choose it again.")
    if preview_path is not None:
        write_files([(Path(preview_path), preview)], protected_paths=[path])
        record.update(preview_sha256=hashlib.sha256(preview).hexdigest(),
                      preview_size_bytes=len(preview))
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare a Colab input and safe display PNG.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--kind", choices=INPUT_KINDS, required=True)
    parser.add_argument("--input-policy", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--generate-synthetic", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.generate_synthetic:
            if args.kind != "generated_synthetic_trace":
                raise ValueError("Synthetic generation requires the synthetic source type.")
            generate_synthetic(args.image)
        record = inspect_input(args.image, args.kind, args.input_policy, args.preview)
    except Exception as error:
        # Expected validation messages contain no paths; OS errors may contain them.
        message = str(error) if isinstance(error, ValueError) else "Image preparation failed; check the image and inference environment."
        print(json.dumps({"error": message}), file=sys.stderr)
        return 2
    print(json.dumps(record))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Strict, local-only inference from tensor-only checkpoints.

This entrypoint follows the dataset evaluators' ImageNet normalization and
ten-bin RandAugment. A reduced image set, different worker count, hardware or
library versions can produce different TTA realizations/numerical results from
the historical full-dataset run. Paired-image validation is still required.
"""

from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import random

from . import __version__
from .image_input import ImageInputError, open_supported_image
from .security import require_patched_pillow, validate_vit_config
from .settings import IMAGE_SIZE, LABELS, MEAN, MODEL_NAMES, STD, InferenceSettings


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def combine_logits(model_logits, weights):
    """Average classwise raw logits, then apply one softmax (never mean probs)."""
    import numpy as np

    arrays = [np.asarray(values, dtype=np.float64) for values in model_logits]
    if not arrays or len(arrays) != len(weights):
        raise ValueError("Each model must have one ensemble weight.")
    if any(a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all() for a in arrays):
        raise ValueError("Expected finite (number of images, 2) raw logits.")
    if any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Model logits must have the same image order and shape.")
    coefficients = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(coefficients).all() or (coefficients < 0).any() or coefficients.max() <= 0:
        raise ValueError("Ensemble weights must be finite, nonnegative, and have positive sum.")
    # Scale first so finite user-supplied weights cannot overflow their sum.
    coefficients = coefficients / coefficients.max()
    coefficients = coefficients / coefficients.sum()
    logits = sum(a * w for a, w in zip(arrays, coefficients))
    return logits, softmax(logits)


def softmax(logits):
    import numpy as np

    values = np.asarray(logits, dtype=np.float64)
    exponentials = np.exp(values - values.max(axis=-1, keepdims=True))
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def classify_probabilities(probabilities, model="ensemble", threshold=None):
    """Preserve historical tie handling unless a threshold is explicitly set."""
    import numpy as np

    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
        raise ValueError("Expected finite two-class probabilities.")
    if (values < 0).any() or (values > 1).any() or not np.allclose(values.sum(axis=1), 1):
        raise ValueError("Expected normalized probabilities.")
    if threshold is None and model != "ensemble":
        return values.argmax(axis=1)
    cutoff = 0.5 if threshold is None else threshold
    if not np.isfinite(cutoff) or not 0 <= cutoff <= 1:
        raise ValueError("Threshold must be between 0 and 1.")
    return (values[:, 1] >= cutoff).astype(int)


def build_transform(model_name, augmented, magnitude, input_policy="strict480"):
    """Match original dataset transforms, including the ten magnitude bins."""
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    if model_name not in MODEL_NAMES:
        raise ValueError("Unknown model name.")
    if not 0 <= magnitude <= 9:
        raise ValueError("Magnitude must be 0..9 for the original ten-bin transform.")
    operations = []
    if model_name == "mlpmixer" or input_policy == "resize":
        operations.append(transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR))
    if augmented:
        # torchvision RandAugment has no horizontal/vertical flip operation.
        # The original subclass filtered child modules; that did not alter this
        # operation set. Explicit ten bins preserves its nonstandard strength.
        operations.append(transforms.RandAugment(
            num_ops=2, magnitude=magnitude, num_magnitude_bins=10,
            interpolation=InterpolationMode.BILINEAR, fill=0,
        ))
    operations.extend([transforms.ToTensor(), transforms.Normalize(mean=MEAN, std=STD)])
    return transforms.Compose(operations)


class ImageDataset:
    """A picklable dataset for optional DataLoader workers on Windows/Colab."""

    def __init__(self, paths, transform):
        self.paths = [Path(p) for p in paths]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        require_patched_pillow()

        try:
            with open_supported_image(self.paths[index], index + 1) as source:
                return self.transform(source.convert("RGB"))
        except ImageInputError:
            raise
        except (OSError, SyntaxError, ValueError):
            raise ValueError(f"Cannot prepare input image {index + 1}.") from None


def _weight_paths(weights_dir, overrides):
    root = Path(weights_dir).expanduser().resolve()
    defaults = {name: root / name / "model.safetensors" for name in MODEL_NAMES}
    overrides = overrides or {}
    for name, path in overrides.items():
        if name not in MODEL_NAMES:
            raise ValueError("Unknown model checkpoint override.")
        candidate = Path(path).expanduser().resolve()
        defaults[name] = candidate / "model.safetensors" if candidate.is_dir() else candidate
    return defaults


def _load_model(name, path, device):
    from safetensors.torch import load_file

    if path.suffix.lower() != ".safetensors":
        raise ValueError("Use an exported .safetensors checkpoint; pickle checkpoints are not accepted.")
    if not path.is_file():
        raise ValueError(f"Missing tensor-only checkpoint for {name}.")
    metadata = {"filename": "model.safetensors", "sha256": sha256_file(path)}
    if name == "vit":
        config_path = path.parent / "config.json"
        if not config_path.is_file():
            raise ValueError("ViT requires config.json beside model.safetensors.")
        with config_path.open("rb") as stream:
            config_bytes = stream.read(65537)
        if len(config_bytes) > 65536:
            raise ValueError("ViT config exceeds the supported file size.")
        config_data = validate_vit_config(json.loads(config_bytes.decode("utf-8")))
        from transformers import ViTConfig, ViTForImageClassification

        config = ViTConfig.from_dict(config_data)
        if config.num_labels != 2 or config.num_channels != 3 or config.image_size != IMAGE_SIZE:
            raise ValueError("Expected a 480-pixel, RGB, two-class ViT checkpoint.")
        model = ViTForImageClassification(config)
        metadata.update({"architecture": "ViTForImageClassification", "config_sha256": sha256_file(config_path)})
    else:
        import timm

        if name == "convnextv2":
            model = timm.create_model("convnextv2_base", pretrained=False, num_classes=2)
            architecture = "convnextv2_base"
        else:
            model = timm.create_model("mixer_b16_224", pretrained=False, img_size=IMAGE_SIZE, in_chans=3, num_classes=2)
            architecture = "mixer_b16_224"
        metadata["architecture"] = architecture
    state = load_file(str(path), device="cpu")
    if name == "vit":
        from .checkpoint_compat import prepare_vit_state_dict

        state, mapping_name = prepare_vit_state_dict(state, model.state_dict())
        if mapping_name is not None:
            metadata["checkpoint_key_mapping"] = mapping_name
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError:
        raise ValueError(f"Checkpoint tensor keys or shapes do not match the {name} architecture.") from None
    del state
    return model.to(device).eval(), metadata


def _validate_images(paths, input_policy):
    require_patched_pillow()

    if not paths:
        raise ValueError("At least one input image is required.")
    hashes = []
    for index, path in enumerate(paths, 1):
        if not path.is_file():
            raise ValueError(f"Input image {index} does not exist.")
        try:
            with open_supported_image(path, index) as source:
                if input_policy == "strict480" and source.size != (IMAGE_SIZE, IMAGE_SIZE):
                    raise ValueError(
                        f"Input image {index} is {source.width}x{source.height}; the default requires "
                        "prepared 480x480 images. Use --input-policy resize only for an explicit "
                        "resize experiment; this can differ from the paper's image preparation."
                    )
                source.verify()
        except (OSError, SyntaxError):
            raise ValueError(f"Input image {index} is not a readable image.") from None
        hashes.append(sha256_file(path))
    return hashes


def _predict_one_model(model, name, paths, settings, device, progress_callback=None):
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    spec = getattr(settings, name)
    accumulator = np.zeros((len(paths), 2), dtype=np.float64)
    use_amp = settings.precision == "bfloat16" and name != "vit"
    # The seed is reset once per augmentation pass, then images are processed
    # in input order. Do not reseed per image (that changes dataset semantics).
    for pass_index in range(spec.tta):
        pass_seed = settings.seed if pass_index == 0 else settings.seed + pass_index - 1
        torch.manual_seed(pass_seed)
        np.random.seed(pass_seed)
        random.seed(pass_seed)
        transform = build_transform(name, pass_index > 0, spec.magnitude, settings.input_policy)
        dataset = ImageDataset(paths, transform)
        loader = DataLoader(dataset, batch_size=settings.batch_size, shuffle=False,
                            num_workers=settings.num_workers, pin_memory=device == "cuda",
                            drop_last=False, persistent_workers=False)
        offset = 0
        with torch.inference_mode():
            for images in loader:
                images = images.to(device, non_blocking=device == "cuda")
                amp = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
                with amp:
                    result = model(images)
                    logits = result.logits if name == "vit" else result
                values = logits.float().cpu().numpy()
                if values.ndim != 2 or values.shape[1] != 2 or not np.isfinite(values).all():
                    raise ValueError(f"The {name} model returned invalid two-class logits.")
                accumulator[offset:offset + len(values)] += values.astype(np.float64)
                offset += len(values)
                if progress_callback is not None:
                    progress_callback({
                        "event": "progress", "message": "Running inference.",
                        "completed": pass_index * len(paths) + offset, "total": spec.tta * len(paths),
                        "model": name, "tta_index": pass_index + 1, "tta_total": spec.tta,
                        "images_completed": offset, "images_total": len(paths),
                    })
        if offset != len(paths):
            raise RuntimeError("The model did not return every input image.")
    return (accumulator / float(spec.tta)).astype(np.float32)


def predict(images, weights_dir="weights", model="ensemble", settings=None, checkpoint_overrides=None,
            progress_callback=None):
    """Return predictions and provenance without storing source absolute paths.

    `images` is an ordered sequence of local image paths. `checkpoint_overrides`
    maps model names to local safetensors files (or their containing folders).
    The ViT config must be beside its tensor file. This function never fetches
    models from a network endpoint and never unpickles checkpoints.
    An optional progress callback receives stage and completed-batch events;
    no progress is emitted by default. Completion after saving belongs to the CLI.
    """
    if progress_callback is not None:
        progress_callback({"event": "stage", "message": "Initializing inference libraries.",
                           "completed": 0, "total": 0, "model": None})
    import numpy as np
    import torch

    settings = settings or InferenceSettings()
    settings.validate()
    if model not in (*MODEL_NAMES, "ensemble"):
        raise ValueError("Model must be vit, convnextv2, mlpmixer, or ensemble.")
    selected = list(MODEL_NAMES) if model == "ensemble" else [model]
    device = "cuda" if settings.device == "auto" and torch.cuda.is_available() else settings.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable in this Python environment.")
    if settings.precision == "bfloat16" and (device != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("bfloat16 requires a CUDA device with bfloat16 support; use float32 otherwise.")
    # Match the evaluators' float32 path, avoiding implicit TF32 acceleration.
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
    paths = [Path(path).expanduser().resolve() for path in images]
    total = sum(getattr(settings, name).tta for name in selected) * len(paths)
    if progress_callback is not None:
        progress_callback({"event": "stage", "message": "Checking inputs and checkpoints.",
                           "completed": 0, "total": total, "model": None,
                           "model_count": len(selected)})
    input_hashes = _validate_images(paths, settings.input_policy)
    weight_paths = _weight_paths(weights_dir, checkpoint_overrides)
    # Validate the presence/format of every requested model before allocating.
    for name in selected:
        if not weight_paths[name].is_file() or weight_paths[name].suffix.lower() != ".safetensors":
            raise ValueError(f"Missing exported .safetensors checkpoint for {name}.")
    outputs, checkpoint_metadata = {}, {}
    completed = 0
    for model_index, name in enumerate(selected, 1):
        if progress_callback is not None:
            progress_callback({"event": "stage", "message": "Loading model.",
                               "completed": completed, "total": total, "model": name,
                               "model_index": model_index, "model_count": len(selected)})
        network, checkpoint_metadata[name] = _load_model(name, weight_paths[name], device)
        try:
            model_callback = None
            if progress_callback is not None:
                progress_callback({"event": "stage", "message": "Running inference.",
                                   "completed": completed, "total": total, "model": name,
                                   "model_index": model_index, "model_count": len(selected),
                                   "tta_index": 1, "tta_total": getattr(settings, name).tta,
                                   "images_completed": 0, "images_total": len(paths)})

                def model_callback(event):
                    progress_callback({**event, "completed": completed + event["completed"],
                                       "total": total, "model_index": model_index,
                                       "model_count": len(selected)})

            outputs[name] = _predict_one_model(network, name, paths, settings, device,
                                               progress_callback=model_callback)
        finally:
            del network
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()
        completed += getattr(settings, name).tta * len(paths)
    if progress_callback is not None:
        progress_callback({"event": "stage", "message": "Combining predictions.",
                           "completed": completed, "total": total, "model": None})
    if model == "ensemble":
        final_logits, probabilities = combine_logits([outputs[name] for name in MODEL_NAMES], settings.ensemble_weights)
    else:
        final_logits = outputs[model].astype(np.float64)
        probabilities = softmax(final_logits)
    individual_probabilities = {name: softmax(outputs[name]) for name in selected}
    decisions = (final_logits.argmax(axis=1) if model != "ensemble" and settings.threshold is None
                 else classify_probabilities(probabilities, model=model, threshold=settings.threshold))
    rows = []
    for index, digest in enumerate(input_hashes):
        pred_id = int(decisions[index])
        row = {
            "sample_id": f"sample_{index + 1:06d}",
            "input_sha256": digest,
            "models": {
                name: {"logits": outputs[name][index].tolist(),
                       "probabilities": individual_probabilities[name][index].tolist()}
                for name in selected
            },
            "logits": final_logits[index].tolist(),
            "probabilities": probabilities[index].tolist(),
            "pred_id": pred_id,
            "pred_label": LABELS[pred_id],
        }
        rows.append(row)
    versions = {"python": platform.python_version(), "package": __version__}
    for distribution in ("torch", "torchvision", "timm", "transformers", "safetensors", "numpy", "Pillow", "filelock"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return {
        "metadata": {
            "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": model, "class_order": list(LABELS), "number_of_images": len(paths),
            "settings": settings.to_dict(), "actual_device": device,
            "actual_precision": {name: (settings.precision if name != "vit" else "float32") for name in selected},
            "checkpoints": checkpoint_metadata, "versions": versions,
            "aggregation": "weighted_mean_raw_logits_then_softmax" if model == "ensemble" else "mean_tta_raw_logits_then_softmax",
            "decision_rule": "argmax_tie_tako" if model != "ensemble" and settings.threshold is None else "AMI_probability_greater_or_equal_threshold",
            "effective_threshold": (0.5 if settings.threshold is None else settings.threshold) if model == "ensemble" or settings.threshold is not None else None,
            "tta": {"identity_included": True, "num_ops": 2, "num_magnitude_bins": 10,
                    "interpolation": "bilinear", "fill": 0, "seed_scope": "dataset_pass"},
            "preprocessing": {"rgb": True, "mean": list(MEAN), "std": list(STD),
                              "image_size": IMAGE_SIZE, "input_policy": settings.input_policy,
                              "mixer_resize": "bilinear_480x480"},
            "validation_status": "historical_prediction_parity_not_yet_verified",
            "reproducibility_note": "TTA realizations depend on input order, dataset size, worker count and library versions; paired comparison with the historical run is required.",
        },
        "predictions": rows,
    }


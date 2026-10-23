"""Local Grad-CAM for the ViT and ConvNeXtV2 research models.

The layer choices and channel-gradient reductions follow the archived single
image evaluator. Maps are computed from one unaugmented float32 forward pass;
they are not attributions of the TTA or ensemble operation. The default input
normalization matches the validated CLI. ``legacy-vit`` instead uses the old
ViT visualizer's 0.5/0.5 normalization for explicitly labelled comparisons.

ViT uses PyTorch bicubic interpolation (align_corners=False) in place of the
archived OpenCV cubic resize. Numerical parity with historical rendered maps
must be checked separately; no OpenCV or plotting dependency is required.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import math
from pathlib import Path
import re

from .image_input import open_supported_image
from .inference import _load_model, _validate_images, _weight_paths, build_transform, sha256_file
from .settings import IMAGE_SIZE, LABELS, MEAN, STD

CAM_MODELS = ("vit", "convnextv2")
CAM_TARGETS = ("model-predicted", "ensemble-predicted", "tako", "AMI")
CAM_NORMALIZATIONS = ("inference", "legacy-vit")
CONVNEXT_LAYER = "stages.3.blocks.2.conv_dw"


def selected_cam_models(prediction_model):
    """Resolve the supported subset without importing ML dependencies."""
    if prediction_model == "ensemble":
        return list(CAM_MODELS)
    if prediction_model in CAM_MODELS:
        return [prediction_model]
    if prediction_model == "mlpmixer":
        raise ValueError("Grad-CAM is available for ViT and ConvNeXtV2; MLP-Mixer has no archived Grad-CAM implementation.")
    raise ValueError("Unknown prediction model for Grad-CAM.")


def _validate_options(target, normalization, prediction_model=None):
    if target not in CAM_TARGETS:
        raise ValueError("Unknown Grad-CAM target; use model-predicted, ensemble-predicted, tako or AMI.")
    if normalization not in CAM_NORMALIZATIONS:
        raise ValueError("Grad-CAM normalization must be inference or legacy-vit.")
    if target == "ensemble-predicted" and prediction_model != "ensemble":
        raise ValueError("The ensemble-predicted Grad-CAM target requires ensemble predictions.")


def build_cam_transform(name, normalization="inference", input_policy="strict480"):
    """Return a deterministic RGB transform; never apply random augmentation."""
    if name not in CAM_MODELS:
        raise ValueError("Grad-CAM is available only for ViT and ConvNeXtV2.")
    if normalization not in CAM_NORMALIZATIONS:
        raise ValueError("Grad-CAM normalization must be inference or legacy-vit.")
    if input_policy not in ("strict480", "resize"):
        raise ValueError("Grad-CAM input policy must be strict480 or resize.")
    if name != "vit" or normalization == "inference":
        return build_transform(name, False, 0, input_policy)
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    operations = []
    if input_policy == "resize":
        operations.append(transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BILINEAR))
    operations.extend([transforms.ToTensor(), transforms.Normalize(mean=(0.5,) * 3, std=(0.5,) * 3)])
    return transforms.Compose(operations)


def _layers(model, name):
    if name == "convnextv2":
        layer = dict(model.named_modules()).get(CONVNEXT_LAYER)
        if layer is None:
            raise ValueError("The required ConvNeXtV2 Grad-CAM layer is missing; no substitute layer was selected.")
        return [(CONVNEXT_LAYER, layer)]
    vit = getattr(model, "vit", None)
    if vit is None:
        raise ValueError("The required ViT Grad-CAM layers are missing.")
    if hasattr(vit, "layers"):
        blocks, prefix = vit.layers, "vit.layers"
    elif hasattr(vit, "encoder") and hasattr(vit.encoder, "layer"):
        blocks, prefix = vit.encoder.layer, "vit.encoder.layer"
    else:
        raise ValueError("The required ViT Grad-CAM layers are missing.")
    if len(blocks) < 3:
        raise ValueError("ViT Grad-CAM requires at least three transformer blocks.")
    selected = []
    for index in range(len(blocks) - 3, len(blocks)):
        layer = getattr(blocks[index], "layernorm_before", None)
        if layer is None:
            raise ValueError("A required ViT layernorm_before module is missing.")
        selected.append((f"{prefix}.{index}.layernorm_before", layer))
    return selected


def _normalize_map(values):
    # Keep the archived epsilon, including its behavior for low-amplitude maps.
    return (values - values.min()) / (values.max() - values.min() + 1e-8)


@contextmanager
def _float32_backends(torch, device):
    """Temporarily disable CUDA TF32; restore the caller's flags on every exit."""
    if device.type != "cuda":
        yield
        return
    original = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
                torch.backends.cudnn.benchmark)
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32, torch.backends.cudnn.benchmark = original


def compute_gradcam(model, name, input_tensor, target_class):
    """Compute one map and the forward logits without modifying model gradients.

    Only the selected layer activations receive autograd queries. Existing
    parameter gradients and per-module train/eval flags are preserved. A cloned
    input with gradients enabled also supports models with frozen parameters.
    """
    import numpy as np
    import torch
    from torch.nn import functional as F

    if name not in CAM_MODELS:
        raise ValueError("Grad-CAM is available only for ViT and ConvNeXtV2.")
    if type(target_class) is not int or target_class not in (0, 1):
        raise ValueError("Grad-CAM target class must be 0 (tako) or 1 (AMI).")
    if (not isinstance(input_tensor, torch.Tensor) or input_tensor.ndim != 4
            or input_tensor.shape[0] != 1 or input_tensor.shape[1] != 3
            or min(input_tensor.shape[2:]) < 1 or input_tensor.dtype != torch.float32
            or not bool(torch.isfinite(input_tensor).all())):
        raise ValueError("Grad-CAM requires one finite float32 RGB image tensor.")
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("Grad-CAM requires float32 model parameters.")
    selected = _layers(model, name)
    states = [(module, module.training) for module in model.modules()]
    handles, features = [], []

    def capture(module, inputs, output):
        if not isinstance(output, torch.Tensor):
            raise ValueError("A Grad-CAM target layer did not return a tensor.")
        features.append(output)

    try:
        model.eval()
        for _, layer in selected:
            handles.append(layer.register_forward_hook(capture))
        # inference_mode(False) is needed even when the caller wraps prediction
        # and visualization in one inference_mode scope. Clone inside it.
        with torch.inference_mode(False), torch.enable_grad(), _float32_backends(torch, input_tensor.device), \
                torch.autocast(device_type=input_tensor.device.type, enabled=False):
            pixels = input_tensor.detach().clone().requires_grad_(True)
            output = model(pixels)
            logits = output.logits if name == "vit" else output
            if (not isinstance(logits, torch.Tensor) or tuple(logits.shape) != (1, 2)
                    or logits.dtype != torch.float32 or not bool(torch.isfinite(logits).all())):
                raise ValueError("The Grad-CAM forward pass returned invalid two-class logits.")
            if len(features) != len(selected):
                raise ValueError("Each Grad-CAM target layer must run exactly once.")
            if any(f.dtype != torch.float32 or not bool(torch.isfinite(f).all()) for f in features):
                raise ValueError("Grad-CAM target-layer activations must be finite float32 tensors.")
            try:
                gradients = torch.autograd.grad(logits[0, target_class], tuple(features),
                                                retain_graph=False, create_graph=False)
            except RuntimeError:
                raise ValueError("Grad-CAM gradients could not be computed for the selected layers.") from None
            if any(not bool(torch.isfinite(g).all()) for g in gradients):
                raise ValueError("Grad-CAM target-layer gradients are not finite.")
            if name == "vit":
                maps = []
                for feature, gradient in zip(features, gradients):
                    if feature.ndim != 3 or feature.shape[0] != 1 or feature.shape != gradient.shape:
                        raise ValueError("ViT Grad-CAM requires batch-token-channel activations.")
                    tokens = feature.shape[1]
                    starts = [n for n in (1, 2) if tokens > n and math.isqrt(tokens - n) ** 2 == tokens - n]
                    if not starts:
                        raise ValueError("ViT Grad-CAM patch tokens do not form a square grid.")
                    start = starts[0]
                    patches = feature.detach()[0, start:, :]
                    weights = gradient[0, start:, :].mean(dim=0)
                    values = torch.relu((patches * weights).sum(dim=-1))
                    side = math.isqrt(values.numel())
                    maps.append(values.reshape(side, side))
                if any(m.shape != maps[0].shape for m in maps):
                    raise ValueError("ViT Grad-CAM layers have different patch grids.")
                low_resolution = _normalize_map(torch.stack(maps).mean(dim=0))
                heatmap = F.interpolate(low_resolution[None, None], size=pixels.shape[2:],
                                        mode="bicubic", align_corners=False)[0, 0]
                heatmap = _normalize_map(heatmap)
            else:
                feature, gradient = features[0].detach(), gradients[0]
                if feature.ndim != 4 or feature.shape[0] != 1 or feature.shape != gradient.shape:
                    raise ValueError("ConvNeXtV2 Grad-CAM requires batch-channel-height-width activations.")
                weights = gradient.mean(dim=(2, 3), keepdim=True)
                low_resolution = torch.relu((weights * feature).sum(dim=1, keepdim=True))
                # Constant coarse maps contain no spatial evidence. Avoid
                # amplifying interpolation round-off into artificial contrast.
                if bool(low_resolution.max() == low_resolution.min()):
                    heatmap = torch.zeros(pixels.shape[2:], device=pixels.device, dtype=torch.float32)
                else:
                    heatmap = F.interpolate(low_resolution, size=pixels.shape[2:],
                                            mode="bilinear", align_corners=False)[0, 0]
                    heatmap = _normalize_map(heatmap)
            if not bool(torch.isfinite(heatmap).all()):
                raise ValueError("Grad-CAM produced a non-finite heatmap.")
            result = heatmap.detach().cpu().numpy().astype(np.float32, copy=True)
            return {"map": result, "layers": [key for key, _ in selected],
                    "unaugmented_logits": logits.detach()[0].cpu().tolist(),
                    "map_is_constant": bool(result.max() == result.min())}
    finally:
        for handle in handles:
            handle.remove()
        features.clear()
        for module, training in states:
            module.training = training


def _prediction_contract(result, count, target, normalization):
    """Check the result structure before models are loaded or images decoded."""
    import numpy as np

    if not isinstance(result, dict) or not isinstance(result.get("metadata"), dict):
        raise ValueError("Grad-CAM requires a completed prediction result.")
    meta, rows = result["metadata"], result.get("predictions")
    selected = selected_cam_models(meta.get("model"))
    _validate_options(target, normalization, meta.get("model"))
    if (not isinstance(rows, list) or not rows or len(rows) != count
            or meta.get("number_of_images") != count or meta.get("class_order") != list(LABELS)):
        raise ValueError("Grad-CAM prediction image count or class order does not match.")
    settings = meta.get("settings")
    if not isinstance(settings, dict) or settings.get("input_policy") not in ("strict480", "resize"):
        raise ValueError("Grad-CAM requires the prediction input policy.")
    if meta.get("actual_device") not in ("cpu", "cuda"):
        raise ValueError("Grad-CAM requires a recorded CPU or CUDA prediction device.")
    checkpoints = meta.get("checkpoints")
    if not isinstance(checkpoints, dict):
        raise ValueError("Grad-CAM requires recorded checkpoint hashes.")
    for name in selected:
        identity = checkpoints.get(name)
        if not isinstance(identity, dict) or not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("sha256", ""))):
            raise ValueError("Grad-CAM requires recorded checkpoint hashes.")
        if name == "vit" and not re.fullmatch(r"[0-9a-f]{64}", str(identity.get("config_sha256", ""))):
            raise ValueError("Grad-CAM requires the recorded ViT configuration hash.")
    for index, row in enumerate(rows, 1):
        if (not isinstance(row, dict) or row.get("sample_id") != f"sample_{index:06d}"
                or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("input_sha256", "")))
                or not isinstance(row.get("models"), dict)):
            raise ValueError("Grad-CAM prediction input identities are invalid.")
        if type(row.get("pred_id")) is not int or row["pred_id"] not in (0, 1):
            raise ValueError("Grad-CAM prediction target class is invalid.")
        for name in selected:
            record = row["models"].get(name)
            if not isinstance(record, dict):
                raise ValueError("Grad-CAM is missing a selected model prediction.")
            try:
                logits = np.asarray(record.get("logits"), dtype=np.float64)
            except (TypeError, ValueError):
                raise ValueError("Grad-CAM requires finite two-class model logits.") from None
            if logits.shape != (2,) or not np.isfinite(logits).all():
                raise ValueError("Grad-CAM requires finite two-class model logits.")
    return selected, meta, rows


def generate_gradcam(images, prediction_result, weights_dir="weights", checkpoint_overrides=None,
                     *, target="model-predicted", normalization="inference", progress_callback=None):
    """Generate independent model maps for an existing, matching prediction run.

    The supplied result is never modified. Input and checkpoint hashes must
    match that run. Return arrays and non-identifying provenance; the caller
    owns rendering, safe output writes and completion reporting.
    """
    import numpy as np
    import torch

    paths = [Path(path).expanduser().resolve() for path in images]
    selected, prediction_meta, predictions = _prediction_contract(prediction_result, len(paths), target, normalization)
    device = prediction_meta["actual_device"]
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was used for the prediction but is unavailable for Grad-CAM.")
    input_policy = prediction_meta["settings"]["input_policy"]
    total = len(paths) * len(selected)
    if progress_callback is not None:
        progress_callback({"event": "stage", "message": "Checking Grad-CAM inputs and checkpoints.",
                           "completed": 0, "total": total, "model": None, "model_count": len(selected)})
    hashes = _validate_images(paths, input_policy)
    if hashes != [row["input_sha256"] for row in predictions]:
        raise ValueError("Grad-CAM image hashes do not match the prediction run.")
    weight_paths = _weight_paths(weights_dir, checkpoint_overrides)
    for name in selected:
        path = weight_paths[name]
        if not path.is_file() or path.suffix.lower() != ".safetensors":
            raise ValueError(f"Missing exported .safetensors checkpoint for {name} Grad-CAM.")
        if sha256_file(path) != prediction_meta["checkpoints"][name]["sha256"]:
            raise ValueError("Grad-CAM checkpoint hashes do not match the prediction run.")
        if name == "vit":
            config = path.parent / "config.json"
            if not config.is_file() or sha256_file(config) != prediction_meta["checkpoints"][name]["config_sha256"]:
                raise ValueError("Grad-CAM ViT configuration does not match the prediction run.")
    rows = [{"sample_id": p["sample_id"], "input_sha256": p["input_sha256"], "models": {}} for p in predictions]
    checkpoint_records = {}
    # Model construction initializes weights before loading the checkpoint.
    # Preserve RNG state even though this workflow uses no random augmentation.
    cuda_devices = [torch.cuda.current_device()] if device == "cuda" else []
    completed = 0
    with torch.inference_mode(False), torch.random.fork_rng(devices=cuda_devices):
        for model_index, name in enumerate(selected, 1):
            if progress_callback is not None:
                progress_callback({"event": "stage", "message": "Loading Grad-CAM model.",
                                   "completed": completed, "total": total, "model": name,
                                   "model_index": model_index, "model_count": len(selected)})
            network, identity = _load_model(name, weight_paths[name], device)
            try:
                expected = prediction_meta["checkpoints"][name]
                if identity.get("sha256") != expected["sha256"] or (name == "vit" and identity.get("config_sha256") != expected["config_sha256"]):
                    raise ValueError("A Grad-CAM checkpoint changed while loading.")
                checkpoint_records[name] = identity
                transform = build_cam_transform(name, normalization, input_policy)
                mean, std = ((0.5,) * 3, (0.5,) * 3) if name == "vit" and normalization == "legacy-vit" else (MEAN, STD)
                for index, (path, prediction) in enumerate(zip(paths, predictions)):
                    if sha256_file(path) != hashes[index]:
                        raise ValueError("A Grad-CAM input image changed after prediction.")
                    with open_supported_image(path, index + 1) as source:
                        tensor = transform(source.convert("RGB")).unsqueeze(0).to(device=device, dtype=torch.float32)
                    if sha256_file(path) != hashes[index]:
                        raise ValueError("A Grad-CAM input image changed during decoding.")
                    if target == "model-predicted":
                        class_id = int(np.argmax(prediction["models"][name]["logits"]))
                    elif target == "ensemble-predicted":
                        class_id = prediction["pred_id"]
                    else:
                        class_id = LABELS.index(target)
                    record = compute_gradcam(network, name, tensor, class_id)
                    record.update({"target_id": class_id, "target_label": LABELS[class_id],
                                   "target_source": target,
                                   "normalization": {"mean": list(mean), "std": list(std)},
                                   "matches_prediction_normalization": not (name == "vit" and normalization == "legacy-vit")})
                    rows[index]["models"][name] = record
                    del tensor
                    completed += 1
                    if progress_callback is not None:
                        progress_callback({"event": "progress", "message": "Computing Grad-CAM.",
                                           "completed": completed, "total": total, "model": name,
                                           "model_index": model_index, "model_count": len(selected),
                                           "images_completed": index + 1, "images_total": len(paths)})
            finally:
                del network
                gc.collect()
                if device == "cuda":
                    torch.cuda.empty_cache()
    return {"metadata": {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                         "method": "gradcam", "models": selected, "number_of_images": len(paths),
                         "target": target, "normalization": normalization, "input_policy": input_policy,
                         "actual_device": device, "actual_precision": "float32", "tta": 1,
                         "class_order": list(LABELS), "checkpoints": checkpoint_records,
                         "map_size": [IMAGE_SIZE, IMAGE_SIZE],
                         "prediction_created_utc": prediction_meta.get("created_utc"),
                         "attribution_scope": "individual_model_unaugmented_forward",
                         "validation_status": "historical_gradcam_parity_not_yet_verified",
                         "note": "These independent normalized model maps are not TTA or ensemble attributions. "
                                 "The legacy-vit option changes ViT normalization relative to CLI predictions; "
                                 "map intensity is not a probability or a clinical localization."},
            "images": rows}

#!/usr/bin/env python3
"""Export private research checkpoints as verified, tensor-only inference files.

Only run this command on checkpoints you own or otherwise trust. PyTorch inputs
are loaded with weights_only=True on PyTorch 2.10.0 or newer; custom Python
objects are never allowlisted. Restricted loading is not a substitute for
trusting the origin of a checkpoint.
The original files are read-only inputs. Optimizers, arguments, SWA states,
training histories, paths, and other checkpoint metadata are not exported.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


VIT_CONFIG_KEYS = frozenset({
    "architectures", "attention_probs_dropout_prob", "encoder_stride",
    "hidden_act", "hidden_dropout_prob", "hidden_size", "id2label",
    "image_size", "initializer_range", "intermediate_size", "label2id",
    "layer_norm_eps", "model_type", "num_attention_heads", "num_channels",
    "num_hidden_layers", "patch_size", "pooler_act", "pooler_output_size",
    "problem_type", "qkv_bias", "torch_dtype", "transformers_version",
})
PROCESSOR_KEYS = frozenset({
    "do_convert_rgb", "do_normalize", "do_rescale", "do_resize", "image_mean",
    "image_processor_type", "image_std", "resample", "rescale_factor", "size",
})
TENSOR_NAME = re.compile(r"[A-Za-z0-9_.]+\Z")


def require_patched_torch() -> None:
    """Fail before touching checkpoint files on vulnerable/unverified versions."""
    match = re.fullmatch(
        r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?(?:\+[A-Za-z0-9.]+)?",
        str(torch.__version__),
    )
    if match is None or tuple(map(int, match.groups())) < (2, 10, 0):
        raise ValueError(
            "Checkpoint export requires a stable PyTorch release >= 2.10.0. "
            "Use a separate updated export environment and only trusted checkpoints."
        )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """Hash exact bytes, including floating-point bit patterns and signed zero."""
    data = tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy()
    view = memoryview(data).cast("B")
    digest = hashlib.sha256()
    for start in range(0, len(view), 8 * 1024 * 1024):
        digest.update(view[start:start + 8 * 1024 * 1024])
    return digest.hexdigest()


def validate_json_values(value: Any) -> None:
    """Whitelisted configs must not carry paths or arbitrary nested objects."""
    if isinstance(value, dict):
        for key, item in value.items():
            validate_json_values(key)
            validate_json_values(item)
    elif isinstance(value, list):
        for item in value:
            validate_json_values(item)
    elif isinstance(value, str):
        if "/" in value or "\\" in value or any(ord(c) < 32 for c in value):
            raise ValueError("A configuration value contains a path or control character")
    elif value is not None and not isinstance(value, (int, float, bool)):
        raise TypeError("Unsupported configuration value")


def clean_json(source: Path, allowed: frozenset[str]) -> dict[str, Any]:
    data = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected a JSON configuration object")
    cleaned = {key: value for key, value in data.items() if key in allowed}
    validate_json_values(cleaned)
    return cleaned


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def validate_state(state: Mapping[str, torch.Tensor], head: str) -> dict[str, torch.Tensor]:
    if not isinstance(state, Mapping) or not state:
        raise ValueError("The selected model state is missing or empty")
    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not TENSOR_NAME.fullmatch(key):
            raise ValueError("Unexpected model tensor name")
        if key == "n_averaged" or key.startswith("module."):
            raise ValueError("Expected the ordinary model state, not an SWA or wrapped state")
        if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
            raise ValueError("The model state must contain only dense tensors")
        cleaned[key] = value.detach().cpu().contiguous()
    weight = cleaned.get(head + ".weight")
    bias = cleaned.get(head + ".bias")
    if weight is None or weight.ndim != 2 or weight.shape[0] != 2:
        raise ValueError("Expected a two-class classifier weight")
    if bias is None or tuple(bias.shape) != (2,):
        raise ValueError("Expected a two-class classifier bias")
    return cleaned


def save_verified(state: Mapping[str, torch.Tensor], target: Path) -> dict[str, Any]:
    target.mkdir(parents=True, exist_ok=False)
    output = target / "model.safetensors"
    save_file(dict(state), str(output), metadata={"format": "pt"})
    # Independently reopen every saved tensor; compare shape, dtype and raw bytes.
    aggregate = hashlib.sha256()
    with safe_open(str(output), framework="pt", device="cpu") as reopened:
        if set(reopened.keys()) != set(state):
            raise RuntimeError("Export changed the set of tensor names")
        if reopened.metadata() != {"format": "pt"}:
            raise RuntimeError("Unexpected exported metadata")
        for key in sorted(state):
            original = state[key]
            exported = reopened.get_tensor(key)
            if original.shape != exported.shape or original.dtype != exported.dtype:
                raise RuntimeError("Export changed a tensor shape or dtype")
            original_hash = tensor_sha256(original)
            if original_hash != tensor_sha256(exported):
                raise RuntimeError("Export changed tensor bytes")
            descriptor = json.dumps([key, str(original.dtype), list(original.shape), original_hash], separators=(",", ":"))
            aggregate.update(descriptor.encode("utf-8") + b"\n")
    return {
        "file": "model.safetensors",
        "sha256": file_sha256(output),
        "tensor_count": len(state),
        "tensor_elements": sum(value.numel() for value in state.values()),
        "tensor_content_sha256": aggregate.hexdigest(),
        "verification": "Every tensor name, shape, dtype and byte matched the selected source state.",
    }


def export_bundle(vit_checkpoint: Path, vit_processor: Path, convnextv2_checkpoint: Path,
                  mlpmixer_checkpoint: Path, output_dir: Path) -> dict[str, Any]:
    """Write a new bundle, publishing its manifest only after verification.

    Existing directories are never replaced. An interrupted or failed export
    may leave a partial directory without manifest.json; choose a new output
    directory when retrying. This avoids directory-renaming failures in Windows
    synchronized folders while keeping all original checkpoints untouched.
    """
    require_patched_torch()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError("Output directory already exists; choose a new directory")
    for required in (vit_checkpoint / "model.safetensors", vit_checkpoint / "config.json",
                     vit_processor, convnextv2_checkpoint, mlpmixer_checkpoint):
        if not required.is_file():
            raise FileNotFoundError("A required source weight or configuration file is missing")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(exist_ok=False)
    temporary = output_dir
    manifest: dict[str, Any] = {
        "format_version": 1,
        "class_order": ["tako", "AMI"],
        "purpose": "Inference-only weights; no training arguments, paths, optimizers or SWA state.",
        "models": {},
    }
    try:
        vit_config = clean_json(vit_checkpoint / "config.json", VIT_CONFIG_KEYS)
        if vit_config.get("model_type") != "vit" or vit_config.get("id2label") != {"0": "tako", "1": "AMI"}:
            raise ValueError("Unexpected ViT architecture or class order")
        processor = clean_json(vit_processor, PROCESSOR_KEYS)
        state = validate_state(load_file(str(vit_checkpoint / "model.safetensors"), device="cpu"), "classifier")
        omitted_training_only_tensors = []
        # The training notebook attaches FocalLoss as a child module. Its alpha
        # buffer is used by SAMTrainer.compute_loss after model(...).logits, and
        # is not part of ViTForImageClassification's inference architecture.
        # Exclude only this explicitly recognized two-class loss buffer; retain
        # all other keys so strict inference loading can reject unknown state.
        if "focal_loss.alpha" in state:
            alpha = state["focal_loss.alpha"]
            if tuple(alpha.shape) != (2,) or not alpha.is_floating_point():
                raise ValueError("Unexpected training-only focal_loss.alpha buffer")
            if not torch.isfinite(alpha).all() or not (alpha >= 0).all():
                raise ValueError("Invalid training-only focal_loss.alpha buffer")
            del state["focal_loss.alpha"]
            omitted_training_only_tensors.append("focal_loss.alpha")
        info = save_verified(state, temporary / "vit")
        info["architecture"] = "ViTForImageClassification"
        info["omitted_training_only_tensors"] = omitted_training_only_tensors
        info["source_sha256"] = file_sha256(vit_checkpoint / "model.safetensors")
        write_json(temporary / "vit" / "config.json", vit_config)
        write_json(temporary / "vit" / "preprocessor_config.json", processor)
        info["config_sha256"] = file_sha256(temporary / "vit" / "config.json")
        info["preprocessor_config_sha256"] = file_sha256(temporary / "vit" / "preprocessor_config.json")
        manifest["models"]["vit"] = info
        del state
        gc.collect()

        for name, source, architecture, head in (
            ("convnextv2", convnextv2_checkpoint, "convnextv2_base", "head.fc"),
            ("mlpmixer", mlpmixer_checkpoint, "mixer_b16_224", "head"),
        ):
            checkpoint = torch.load(source, map_location="cpu", weights_only=True)
            if not isinstance(checkpoint, dict) or "model" not in checkpoint:
                raise ValueError("Expected a checkpoint containing its ordinary model state")
            state = validate_state(checkpoint["model"], head)
            info = save_verified(state, temporary / name)
            info["architecture"] = architecture
            info["image_size"] = 480
            info["source_state"] = "model"
            info["source_sha256"] = file_sha256(source)
            manifest["models"][name] = info
            del state, checkpoint
            gc.collect()

        # A file allowlist keeps accidental logs or input paths out of the bundle.
        expected = {"vit/model.safetensors", "vit/config.json",
                    "vit/preprocessor_config.json", "convnextv2/model.safetensors",
                    "mlpmixer/model.safetensors"}
        actual = {p.relative_to(temporary).as_posix() for p in temporary.rglob("*") if p.is_file()}
        if actual != expected:
            raise RuntimeError("Unexpected exported files")
        write_json(temporary / "manifest.json", manifest)
    except BaseException:
        # Keep partial output available for inspection, but do not publish a
        # completion manifest or replace the original inputs.
        raise
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vit-checkpoint", type=Path, required=True, help="Directory containing ViT model.safetensors and config.json")
    parser.add_argument("--vit-processor", type=Path, required=True, help="ViT preprocessor_config.json used for the study")
    parser.add_argument("--convnextv2-checkpoint", type=Path, required=True, help="Selected ordinary ConvNeXtV2 checkpoint (ckpt_0011.pth)")
    parser.add_argument("--mlpmixer-checkpoint", type=Path, required=True, help="Selected ordinary MLP-Mixer checkpoint (ckpt_0011.pth)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory; it must not already exist")
    args = parser.parse_args()
    manifest = export_bundle(args.vit_checkpoint, args.vit_processor,
                             args.convnextv2_checkpoint, args.mlpmixer_checkpoint,
                             args.output_dir)
    for name, info in manifest["models"].items():
        print(f"{name}: exported and verified {info['tensor_count']} tensors; SHA-256 {info['sha256']}")


if __name__ == "__main__":
    main()

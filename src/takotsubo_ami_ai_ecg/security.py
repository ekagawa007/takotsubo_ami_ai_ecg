"""Small input guards that run before image decoding or model construction."""

import importlib.metadata
import math
import re


VIT_CONFIG_KEYS = frozenset({
    "architectures", "attention_probs_dropout_prob", "encoder_stride",
    "hidden_act", "hidden_dropout_prob", "hidden_size", "id2label",
    "image_size", "initializer_range", "intermediate_size", "label2id",
    "layer_norm_eps", "model_type", "num_attention_heads", "num_channels",
    "num_hidden_layers", "num_labels", "patch_size", "pooler_act",
    "pooler_output_size", "problem_type", "qkv_bias", "torch_dtype",
    "transformers_version",
})


def _stable_release(version):
    """Accept stable releases and local build suffixes, not release candidates."""
    match = re.fullmatch(
        r"(\d+)\.(\d+)\.(\d+)(?:\.post\d+)?(?:\+[A-Za-z0-9.]+)?", str(version)
    )
    return tuple(map(int, match.groups())) if match else None


def require_patched_pillow():
    """Reject vulnerable or stale Pillow modules before calling Image.open.

    Checking both versions also catches the common notebook case where pip has
    replaced Pillow on disk but the running kernel still has the old module.
    """
    import PIL

    installed = importlib.metadata.version("Pillow")
    loaded = PIL.__version__
    versions = (_stable_release(installed), _stable_release(loaded))
    if any(version is None or version < (13, 0, 0) for version in versions):
        raise ValueError(
            "Pillow >= 13.0.0 is required before reading images. Install "
            "requirements-inference.txt, then restart the Python/Colab session."
        )
    if installed != loaded:
        raise ValueError(
            "The installed and loaded Pillow versions differ. Restart the "
            "Python/Colab session before reading images."
        )
    try:
        from PIL import Image
    except ImportError:
        raise ValueError(
            "Pillow modules are inconsistent. Restart the Python/Colab session "
            "after installing requirements-inference.txt."
        ) from None
    if Image.__version__ != loaded:
        raise ValueError(
            "A cached Pillow image module has a different version. Restart the "
            "Python/Colab session before reading images."
        )


def validate_vit_config(data):
    """Allow only inert ViT architecture/label settings from a local JSON file.

    Transformers accepts additional execution-control attributes in a config.
    Reject unknown keys instead of passing them to ViTConfig, which prevents
    attention-kernel repository names and other private controls reaching model
    initialization. No attention backend override is inserted here.
    """
    if not isinstance(data, dict) or not set(data).issubset(VIT_CONFIG_KEYS):
        raise ValueError("ViT config contains unsupported configuration fields.")
    if data.get("model_type") != "vit":
        raise ValueError("The ViT checkpoint config must have model_type=vit.")
    if data.get("id2label") != {"0": "tako", "1": "AMI"}:
        raise ValueError("ViT config must explicitly map 0=tako and 1=AMI.")
    if "label2id" in data and data["label2id"] != {"tako": 0, "AMI": 1}:
        raise ValueError("Unexpected ViT label2id configuration.")

    integer_bounds = {
        "hidden_size": (1, 1024), "intermediate_size": (1, 4096),
        "num_hidden_layers": (1, 24), "num_attention_heads": (1, 16),
        "encoder_stride": (1, 480), "pooler_output_size": (1, 1024),
        "image_size": (480, 480), "num_channels": (3, 3),
        "num_labels": (2, 2), "patch_size": (16, 32),
    }
    for key, (lower, upper) in integer_bounds.items():
        if key in data:
            value = data[key]
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError("Invalid or unsupported ViT architecture dimensions.")
    if data.get("patch_size", 16) not in (16, 32):
        raise ValueError("ViT patch_size must be 16 or 32.")
    if data.get("hidden_size", 768) % data.get("num_attention_heads", 12):
        raise ValueError("ViT hidden_size must be divisible by num_attention_heads.")

    numeric_bounds = {
        "attention_probs_dropout_prob": (0, 1),
        "hidden_dropout_prob": (0, 1), "initializer_range": (0, 1),
        "layer_norm_eps": (0, 1),
    }
    for key, (lower, upper) in numeric_bounds.items():
        if key in data:
            value = data[key]
            if (type(value) not in (int, float) or not lower <= value <= upper
                    or not math.isfinite(value)
                    or (key == "layer_norm_eps" and value == 0)):
                raise ValueError("Invalid ViT numerical configuration.")
    if "qkv_bias" in data and type(data["qkv_bias"]) is not bool:
        raise ValueError("ViT qkv_bias must be a boolean.")
    choices = {
        "hidden_act": {"gelu", "gelu_new", "gelu_fast", "gelu_pytorch_tanh", "relu", "silu", "quick_gelu"},
        "pooler_act": {"tanh", "gelu", "relu", "silu"},
        "torch_dtype": {"float32", "float16", "bfloat16"},
        "problem_type": {"single_label_classification"},
    }
    for key, allowed in choices.items():
        if key in data and (not isinstance(data[key], str) or data[key] not in allowed):
            raise ValueError("Invalid or unsupported ViT configuration value.")
    if "architectures" in data and data["architectures"] != ["ViTForImageClassification"]:
        raise ValueError("Expected the ViTForImageClassification architecture.")
    if "transformers_version" in data and (
        not isinstance(data["transformers_version"], str)
        or len(data["transformers_version"]) > 64
        or not re.fullmatch(r"[A-Za-z0-9.+-]+", data["transformers_version"])
    ):
        raise ValueError("Invalid ViT version metadata.")
    return dict(data)

"""Strict, value-preserving ViT checkpoint key compatibility.

Transformers 5 renamed the legacy ViT encoder modules. These explicit names
match the ViTModel conversion documented in Transformers 5.10.4's
src/transformers/conversion_mapping.py. This helper changes dictionary keys
only; it never changes, drops, initializes, or combines checkpoint tensors.
"""

import re


_LEGACY_LAYER = re.compile(r"^vit\.encoder\.layer\.(\d+)\.(.+)$")
_LAYER_NAMES = (
    ("attention.attention.query.", "attention.q_proj."),
    ("attention.attention.key.", "attention.k_proj."),
    ("attention.attention.value.", "attention.v_proj."),
    ("attention.output.dense.", "attention.o_proj."),
    ("intermediate.dense.", "mlp.fc1."),
    ("output.dense.", "mlp.fc2."),
    ("layernorm_before.", "layernorm_before."),
    ("layernorm_after.", "layernorm_after."),
)


def _same_keys_and_shapes(state, expected):
    return set(state) == set(expected) and all(
        tuple(state[key].shape) == tuple(expected[key].shape) for key in expected
    )


def prepare_vit_state_dict(state, expected):
    """Return a fully matching state and the applied mapping name, if any.

    Unknown, mixed, incomplete, colliding or shape-incompatible layouts fail
    closed. The caller must still use load_state_dict(..., strict=True).
    """
    if _same_keys_and_shapes(state, expected):
        return state, None
    has_legacy = any(key.startswith("vit.encoder.layer.") for key in state)
    has_modular = any(key.startswith("vit.layers.") for key in state)
    expects_modular = any(key.startswith("vit.layers.") for key in expected)
    if not has_legacy or has_modular or not expects_modular:
        raise ValueError("Checkpoint tensor keys or shapes do not match the vit architecture.")
    converted = {}
    for key, value in state.items():
        match = _LEGACY_LAYER.fullmatch(key)
        target = key
        if match:
            layer, suffix = match.groups()
            for old_name, new_name in _LAYER_NAMES:
                if suffix.startswith(old_name) and suffix[len(old_name):] in ("weight", "bias"):
                    target = f"vit.layers.{layer}.{new_name}{suffix[len(old_name):]}"
                    break
            else:
                raise ValueError("Unsupported legacy ViT checkpoint field.")
        if target in converted:
            raise ValueError("ViT checkpoint key conversion would duplicate a tensor.")
        converted[target] = value
    if not _same_keys_and_shapes(converted, expected):
        raise ValueError("Checkpoint tensor keys or shapes do not match the vit architecture.")
    return converted, "hf_vit_legacy_to_modular_v1"

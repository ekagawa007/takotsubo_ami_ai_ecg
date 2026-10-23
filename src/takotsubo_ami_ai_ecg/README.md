# Shared inference implementation

This package implements the model loading, image preprocessing, TTA, raw-logit aggregation, and output generation used by the public CLI and Google Colab examples. Model weights and input/output locations are supplied explicitly; the model code does not mount Google Drive or depend on a personal filesystem layout.

The retained historical training/evaluation sources remain under `training/` and `evaluation/original/`. See [the inference guide](../../docs/inference.md) for paper-default settings, supported arguments, and the limits of prediction reproduction checks.

Version 0.2.1 includes single-frame input validation, protected result saving and a strict ViT checkpoint-key adapter for the tested Transformers 5.10.4 layout. The adapter changes parameter names only and preserves the selected tensors. Current dependencies and validation scope are documented separately from the retained historical training requirements.

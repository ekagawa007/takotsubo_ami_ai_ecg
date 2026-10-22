# Inference configuration

`paper_inference.json` records the paper-default model, TTA, class-order, and ensemble settings used by the new examples. CLI arguments allow deliberate changes to these defaults. Runtime outputs record the effective settings so an altered run is distinguishable from the default.

Model binaries are stored locally under `weights/` and are not tracked by Git. Download URLs and asset manifests must identify real published assets when a model release becomes available. See [the inference guide](../docs/inference.md).

# Inference configuration

`paper_inference.json` records the paper-default model, TTA, class-order, and ensemble settings used by the new examples. CLI arguments allow deliberate changes to these defaults. Runtime outputs record the effective settings so an altered run is distinguishable from the default.

Model binaries are stored locally under `weights/` and are not tracked by Git. The official inference bundle is published in the [v0.2.0 weight Release](https://github.com/ekagawa007/takotsubo_ami_ai_ecg/releases/tag/v0.2.0); the downloader verifies the archive and bundled files against the supplied checksums. See [the inference guide](../docs/inference.md).

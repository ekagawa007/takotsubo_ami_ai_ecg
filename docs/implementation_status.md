# Implementation and validation status

Updated 2026-10-22 (JST); documentation revision 0.1.7.

## Public source and inference implementation

- Original training, evaluation, human-study, and scheduling sources retain their filenames and notebook formats, with personal paths/examples generalized.
- The MIT License names Copyright (c) 2026 Eisuke Kagawa.
- Published `v0.1.1` includes the unchanged 2025-12-11 environment snapshot and reported training platforms. Existing release tags are retained.
- Internal publication-preparation records and their verifier are excluded from this revision's public tree.
- Shared inference code, four CLI wrappers, and a Colab companion provide configurable paper defaults. Historical training implementations remain separate.
- The selected ConvNeXtV2 inference state is the ordinary `model` in `ckpt_0011.pth`, not an SWA state. See [selected checkpoints](checkpoint_selection.md).
- Inference assets contain selected tensors and necessary loading configuration. Original training checkpoints, training-machine metadata, patient images, prediction tables, and backups remain private.

## Development checks completed on 2026-10-22

- All 730 exported inference tensors matched the selected original tensor names, shapes, dtypes and bytes. The training-only ViT `focal_loss.alpha` buffer was omitted explicitly and recorded in the asset manifest.
- All three selected models passed strict loading and ensemble inference on a generated 480 x 480 RGB input, using ViT 24/magnitude 1, ConvNeXtV2 without TTA, and MLP-Mixer 32/magnitude 3. This CPU execution check used `num_workers=0`.
- All 17 inference regression tests passed, including exact image-transform comparisons with the original evaluators and raw-logit aggregation checks.
- The Colab notebook passed notebook-format and Python-cell syntax checks. It has no saved patient data or prediction outputs.

These checks used native Windows, Python 3.10, CPU PyTorch 2.9.0 and torchvision 0.24.0. They do not represent a run in the reported Python 3.12.12 WSL/CUDA or Colab environments.

## Verification still required for reproduction

1. Compare new per-model and ensemble results with the original private ECG images and baseline raw-logit predictions. Match image preparation, sample order, TTA RNG settings, precision, and package versions.
2. Validate installation and execution in a clean WSL/Linux environment and a fresh Colab runtime. Distinguish newly tested runtime versions from the historical training record.
3. Link selected checkpoints and evaluation results to their originating runs, initialization chains, and validation split details.

A format check, static test, tensor comparison, or generated-image smoke test answers a narrower question than full prediction reproduction. Original patient-image parity, a complete training rerun, and fresh Colab execution are not claimed by this documentation revision.

## Later distribution

Prepare final model assets, checksum manifests, download locations, and weight-distribution terms before publishing them. Preserve private originals for provenance. Fix a paper-specific source/asset version and archive it on Zenodo when the paper metadata and corresponding assets are ready.

The human-study administrator remains a separate workflow requiring an interactive Linux/WSL terminal with graphical display; its runtime validation is pending.

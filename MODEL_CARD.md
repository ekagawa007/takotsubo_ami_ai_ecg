# Model card: takotsubo versus AMI ECG-image models

Updated 2026-10-28 (JST); documentation revision 0.1.21. Inference package `v0.3.1`; weight assets `v0.2.0`. The filelock 4.2.1 maintenance update passed candidate WSL/CUDA and Colab checks on 2026-10-28; final public-tag Colab acquisition is a separate post-publication check. See [implementation status](docs/implementation_status.md).

## Task and intended use

These research models classify a prepared ECG image as takotsubo syndrome or acute anterior STEMI. The implementation returns per-model predictions and an ensemble prediction. Class order is `0 = tako`, `1 = AMI`; the AMI label refers to the study's acute anterior STEMI comparison group.

The release supports research inference and reproducibility checks. Clinical deployment, prospective patient outcomes and performance across untested populations, image sources or ECG layouts have not been established by the checks documented in this repository. Generated sample images check execution and cannot establish diagnostic accuracy.

## Inputs and outputs

The default input is an already prepared **480 x 480 RGB image**, supplied as a single-frame PNG, JPEG, BMP, TIFF or WebP file. The loader converts to RGB and applies the original dataset evaluators' ImageNet mean/std normalization. The optional `--input-policy resize` resizes the complete image with bilinear interpolation; it does not establish equivalence to the study's upstream ECG preparation.

Outputs include two-class raw logits, probabilities, predicted classes, effective settings, checkpoint identities and hashes. Optional ViT/ConvNeXtV2 Grad-CAM maps use the unaugmented input; they do not explain the TTA-averaged ensemble or establish clinical localization accuracy. Saved input images and overlays contain the input pixels and must be handled as private outputs when real ECGs are used. See [Grad-CAM](docs/gradcam.md). The ensemble selects AMI when `prob_AMI >= 0.5`. Individual models use argmax, with an exact tie selecting tako, unless an explicit threshold is supplied. See [the inference guide](docs/inference.md).

## Selected checkpoints and default inference

| Model | Selected source state | Paper-default TTA |
|---|---|---|
| ViT | `checkpoint-6900/model.safetensors`, with model configuration | 24 views, RandAugment magnitude 1 |
| ConvNeXtV2 | `ckpt_0011.pth`, ordinary `model` state | 1 unchanged view; no TTA |
| MLP-Mixer | `ckpt_0011.pth`, ordinary `model` state | 32 views, RandAugment magnitude 3 |

TTA counts include the unchanged image. Each model averages its raw logits across its views. The ensemble gives the three models equal weight, averages their raw logits and then applies softmax. It does not average model probabilities. ConvNeXtV2 `swa_best.pth` and the checkpoint's SWA state are not used in this default inference path.

The defaults are float32, seed 1234, batch size 1 and four workers. Device selection is automatic unless explicitly set. TTA realizations depend on image order, dataset size, workers, numerical precision and software versions; identical seeds alone do not establish historical prediction equivalence.

## Training provenance and selection

The repository retains the supplied self-supervised pretraining, three-class fine-tuning and final two-class fine-tuning sources. Training and validation losses were reviewed, and the checkpoint with the lowest validation loss among epochs without apparent signs of overfitting was selected. This records the reported selection procedure; it is not proof that overfitting was absent.

The historical training record specifies Python 3.12.12 with WSL Ubuntu 22.04.5/RTX 5090 or Google Colab/A100. The original 165-package dependency record remains separate from the current inference environment.

All 730 selected inference tensors were verified against their original names, shapes, dtypes and values. The training-only ViT `focal_loss.alpha` buffer was explicitly omitted. The selected files do not by themselves establish a complete immutable chain of upstream model revisions, initialization runs, dataset splits and final prediction tables. See [checkpoint selection](docs/checkpoint_selection.md), [execution environments](docs/execution_environment.md) and [upstream notices](THIRD_PARTY_NOTICES.md).

## Distribution and terms

The Release assets are `takotsubo_ami_ai_ecg_weights_v0.2.0.zip` and `SHA256SUMS.txt`. The three safetensors model files total 958,657,088 bytes before packaging; the ZIP is 958,701,120 bytes. The distribution includes selected inference tensors and necessary configuration; it excludes patient images, prediction tables, optimizer/scheduler state and original training-machine paths. Model binaries are distributed separately from Git source.

Check the [v0.2.0 release page](https://github.com/ekagawa007/takotsubo_ami_ai_ecg/releases/tag/v0.2.0) for asset availability. The downloader command is:

```bash
python tools/download_weights.py --output-dir weights
```

The code is MIT licensed. Confirmed model-weight terms are separate: the ViT assets carry CC BY-NC 4.0 noncommercial terms and retained upstream notices; author-owned ConvNeXtV2 and MLP-Mixer contributions use MIT terms subject to the documented upstream notices. The complete three-model bundle includes the ViT condition. Read [WEIGHTS_LICENSE.md](WEIGHTS_LICENSE.md) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); do not infer unrestricted commercial rights from the code license.

## Validation scope

| Check | Result | Scope limit |
|---|---|---|
| v0.3.1 candidate WSL/RTX 5090, 2026-10-28 | 117 of 119 tests passed; two Windows-only tests skipped. Replacing filelock 3.32.3 with 4.2.1 preserved all predictions and both Grad-CAM maps exactly on one generated image with paper TTA | Isolated dependency overlay; float32, batch 1, zero workers; original environment retained |
| v0.3.1 candidate Colab/A100, 2026-10-28 | Python 3.13.16, filelock 4.2.1, CUDA float32; paper-TTA prediction and both Grad-CAM maps completed. Source/weight identities and nine output files verified | Verified candidate archive, one generated image, batch 1, four workers; no paired Grad-CAM-off run. Public-tag acquisition remains a separate check |
| v0.2.1 WSL/RTX 5090, Python 3.12.12, torch 2.13.0+cu130/torchvision 0.28.0+cu130 | Ten real ECG images matched frozen original-evaluator predictions at float32/no TTA; all models and the ensemble were within `atol=rtol=1e-5`, with zero class differences | The original evaluators were not rerun in the new environment; this is not a study performance estimate |
| v0.2.1 Colab/A100, Python 3.13.15, same GPU library pair | One generated RGB image completed inference with the paper TTA settings and float32; source, weights, input identity, saved records and ensemble calculation were checked | This row records the pre-release archive check; the later public-tag execution checks are complete and documented in implementation status |
| v0.3.0 public Colab/A100, 2026-10-24, Python 3.13.16 | One generated RGB image completed paper-TTA float32 inference and ViT/ConvNeXtV2 Grad-CAM; the release commit, source/weight hashes, prediction/run linkage and nine output files were verified | One-image execution check; no paired Grad-CAM-off comparison in this Colab run and no clinical-performance claim |
| v0.3.0 candidate WSL paper-TTA comparison | Predictions were exactly unchanged with Grad-CAM enabled or disabled on one generated image; two finite 480 x 480 maps and saved artifacts were checked | This tests the effect of optional Grad-CAM on prediction; it does not establish historical stochastic-TTA or reader-study heatmap equivalence |
| v0.2.1 tests | WSL: 83 passed, two Windows-only cases skipped. Native Windows CPU: 83 passed, two hardlink-permission cases skipped | 85 cases discovered in each environment; skips are platform-specific |
| CPU dependency comparisons | Ten-image no-TTA and one synthetic-image paper-TTA comparisons passed tolerance, with zero class changes | Native Windows CPU, float32, batch 1, zero workers |
| v0.2.0 distribution | Public archive and all 11 bundled hashes verified on 2026-10-22; selected weights unchanged through v0.3.1 | Distribution integrity does not establish clinical validity |

Current checks use Transformers 5.10.4, timm 1.0.22, numpy 2.2.6, Pillow 13.0.0 and safetensors 0.7.0. The ViT checkpoint-name adapter preserves tensor values and strict key/shape validation. Historical checkpoint identities remain the same.

Historical paper-TTA numerical equivalence, a complete training rerun, all-dataset evaluation reproduction and human-study runtime validation remain incomplete. No diagnostic performance metric is inferred from these execution checks. The latest detailed record and earlier version history are in [implementation status](docs/implementation_status.md).

## Data and citation

Patient ECG images and participant records are not included. Existing private inputs support local comparison, while publicly shareable synthetic inputs can exercise the software. For software v0.3.1, use the release citation in [CITATION.cff](CITATION.cff) and the [README](README.md#how-to-cite); no v0.3.1 DOI is assigned here. The archived [code v0.3.0 DOI](https://doi.org/10.5281/zenodo.23666137) identifies that earlier software version. When using the distributed weights, also cite the separate [weights v0.2.0 DOI](https://doi.org/10.5281/zenodo.23666262). These version-specific software/model DOIs are not the journal article DOI and must not be used to identify a different version. A final journal citation can be added when available. The code copyright is Copyright (c) 2026 Eisuke Kagawa.

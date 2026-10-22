# Model card: takotsubo versus AMI ECG-image models

Updated 2026-10-22 (JST). Package/release `v0.2.0`.

## Task and intended use

These research models classify a prepared ECG image as takotsubo syndrome or acute anterior STEMI. The implementation returns per-model predictions and an ensemble prediction. Class order is `0 = tako`, `1 = AMI`; the AMI label refers to the study's acute anterior STEMI comparison group.

The release supports research inference and reproducibility checks. Clinical deployment, prospective patient outcomes and performance across untested populations, image sources or ECG layouts have not been established by the checks documented in this repository. Generated sample images check execution and cannot establish diagnostic accuracy.

## Inputs and outputs

The default input is an already prepared **480 x 480 RGB image**. The loader converts to RGB and applies the original dataset evaluators' ImageNet mean/std normalization. The optional `--input-policy resize` resizes the complete image with bilinear interpolation; it does not establish equivalence to the study's upstream ECG preparation.

Outputs include two-class raw logits, probabilities, predicted classes, effective settings, checkpoint identities and hashes. The ensemble selects AMI when `prob_AMI >= 0.5`. Individual models use argmax, with an exact tie selecting tako, unless an explicit threshold is supplied. See [the inference guide](docs/inference.md).

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

| Revision | Check and result | Scope limit |
|---|---|---|
| Preceding inference commit `ff1b9bef9596a33cacf9dd67d42cd78d390a6211` | Original evaluators and CLI agreed on 10 real images in WSL/CUDA, float32, no TTA, within `atol=1e-5`, `rtol=1e-5`, with zero class differences | A paired implementation check on those images, not a study performance estimate |
| `0.2.0.dev1`, Pillow 13.0.0 | Updated CLI matched frozen original-evaluator CSVs on the same 10 real images; its new-CLI CSV was byte-identical to the preceding new-CLI CSV | Original evaluators were not rerun; the comparison used no TTA |
| `0.2.0.dev2` | 38 tests and seven simulated notebook cases passed; selected-model CPU inference on one synthetic image completed with paper TTA, live progress and prediction values exactly matching the prior CPU baseline | Synthetic-image CPU and simulated notebook checks |
| Colab execution | The preceding commit completed paper-TTA inference on one synthetic image on A100; the later English notebook with progress was reported to run successfully | The later saved runtime record was not reviewed for this update |
| `v0.2.0` distribution | 50 tests and five simulated notebook download cases passed; release ZIP installed into a fresh directory with all 11 file hashes verified; installed selected models completed synthetic-image paper-TTA CPU inference with predictions exactly matching the baseline | Live public URL download and actual Colab-download execution remain to be verified |

Historical paper-TTA numerical equivalence, a complete training rerun, all-dataset evaluation reproduction and human-study runtime validation remain incomplete. No diagnostic performance metric is inferred from these execution checks. The latest detailed record is [implementation status](docs/implementation_status.md).

## Data and citation

Patient ECG images and participant records are not included. Existing private inputs support local comparison, while publicly shareable synthetic inputs can exercise the software. Use the repository and the actual release version when describing software; final publication metadata and a version-specific DOI are not asserted by this model card. The code copyright is Copyright (c) 2026 Eisuke Kagawa.

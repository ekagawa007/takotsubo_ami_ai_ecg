# takotsubo_ami_ai_ecg

**Documentation version:** `0.1.9`

**Inference package version:** `0.2.0.dev1`

**Last updated:** 2026-10-22 (JST)

Research code for ECG-image models distinguishing takotsubo syndrome from acute anterior STEMI, including model development, ensemble evaluation, and a human diagnostic study.

## Current status

The original training, evaluation, and human-study sources are retained in their supplied formats, with personal paths and identifying-looking examples generalized. The published [v0.1.1 release](https://github.com/ekagawa007/takotsubo_ami_ai_ecg/releases/tag/v0.1.1) adds the historical requirements and training-environment documentation to the initial research-source release.

This revision adds shared inference code, per-model and ensemble CLI examples, and a Google Colab notebook. The paper settings are the defaults and can be changed with arguments. Selected model weights must be supplied locally; they are not included in the Git repository. At commit `ff1b9bef9596a33cacf9dd67d42cd78d390a6211`, paired WSL inference on 10 private ECG images agreed with the original evaluators within the configured numerical tolerance using float32 and no TTA. Colab inference on one synthetic image completed on an NVIDIA A100 with the paper TTA settings. Those earlier checks used Pillow 12.0.0. The updated package with Pillow 13.0.0 has now also passed the scoped WSL/CUDA no-TTA check described below; updated Colab execution remains pending. This repository provides research software; no clinical deployment validation is claimed.

This security revision requires Python 3.11 or newer and Pillow 13.0.0 for new inference installations, restricts ViT configuration fields, and requires PyTorch 2.10.0 or newer for checkpoint export. The Colab notebook handles dependency-change restarts, creates a separate output for each run, and displays subprocess errors. The original training environment snapshot is preserved.

The updated package `0.2.0.dev1` passed all 31 tests and inference with the three selected models on one synthetic 480 x 480 image using the paper TTA settings in an isolated native Windows CPU environment with Python 3.13.15, torch 2.11.0, torchvision 0.26.0 and Pillow 13.0.0. Result saving and the raw-logit ensemble calculation were checked. On WSL/CUDA with Pillow 13.0.0, the updated CLI also matched the saved original-evaluator results for the same 10 real ECG images using float32 and no TTA: all three models and the ensemble passed the configured tolerance, with zero class changes. This compared against frozen earlier CSVs; the original evaluators were not rerun. Updated Colab execution, WSL paper-TTA execution and historical paper-TTA numerical parity remain unverified. See [validation details and remaining checks](docs/implementation_status.md).

## Contents

| Location | Purpose |
|---|---|
| `requirements.txt` | Unmodified 165-package training environment record; snapshot date: 2025-12-11 |
| `requirements-inference.txt` | Dependencies for the new inference examples |
| `ek_schedule_exe.py` | Original helper for waiting until a specified local time |
| `training/pretrain/` | Original SSL code and supplied historical logs |
| `training/ft_3_class/` | Original three-class fine-tuning code and notebooks |
| `training/ft_2_class/` | Original two-class fine-tuning code and notebooks |
| `evaluation/original/` | Original single-image inference, dataset evaluation, and ensemble scripts |
| `human_test/` | Original study preparation, randomization, and administration scripts |
| `src/takotsubo_ami_ai_ecg/` | Shared public inference implementation |
| `examples/` | Three per-model CLIs, an ensemble CLI, and a Colab notebook |
| `configs/` | Paper inference settings |
| `tests/` | Inference validation checks and guidance |
| `weights/` | Local model/configuration files; model binaries are not committed |
| `outputs/` | Local runtime outputs; generated files are not committed |

## Inference quick start

From the repository root, using Python 3.11 or newer with a suitable PyTorch/torchvision installation:

```bash
python -m pip install -r requirements-inference.txt
python examples/predict_ensemble.py --image data/example_ecg.png --weights-dir weights --output outputs/prediction.json
```

Supply the selected local weights in the layout described in [weights/README.md](weights/README.md). The default input is a prepared RGB ECG image of **480 x 480 pixels**. Inputs with other dimensions are rejected unless `--input-policy resize` is requested explicitly; that option resizes the full image without cropping or padding.

| Model | TTA views, including the unchanged view | RandAugment magnitude |
|---|---:|---:|
| ViT | 24 | 1 |
| ConvNeXtV2 | 1 (no TTA) | Not applied |
| MLP-Mixer | 32 | 3 |

Each model averages raw logits across its TTA views. The ensemble then averages the three models' raw logits with equal weights and applies softmax. It does **not** average probabilities. Class order is `0 = tako`, `1 = AMI`; the ensemble selects AMI when `prob_AMI >= 0.5`. Individual-model outputs use the original argmax rule (an exact tie selects tako) unless a threshold is explicitly requested. TTA counts/magnitudes, ensemble weights, seed, device, and other run settings are configurable. See [the inference guide](docs/inference.md) and [examples](examples/README.md).

## Development provenance

ViT fine-tuning was developed interactively in Jupyter/Google Colab, and its original training implementation is provided in `.ipynb` format. These notebooks contain the original training code. ConvNeXtV2 and MLP-Mixer retain Python training implementations and the launcher notebooks that record run arguments. Original filenames, notebook formats, and learning procedures are preserved. The new inference implementation is maintained separately from those historical sources.

- [Development code and execution context](docs/development_code.md)
- [Selected model checkpoints](docs/checkpoint_selection.md)
- [Implementation and validation status](docs/implementation_status.md)

## Checkpoint selection

Training and validation losses were reviewed, and the checkpoint with the lowest validation loss among epochs without apparent signs of overfitting was selected.

The selected final assets are ViT `checkpoint-6900`, ConvNeXtV2 `ckpt_0011.pth` (ordinary `model` state), and MLP-Mixer `ckpt_0011.pth` (ordinary `model` state). ConvNeXtV2 `swa_best.pth` is not the selected inference checkpoint. The public inference files contain the selected tensor state and required loading settings. See [checkpoint selection details](docs/checkpoint_selection.md) for stored epoch/step information and remaining links to evaluation records.

## Execution environments

Training used Python **3.12.12**, with either **Windows 11 / WSL / Ubuntu 22.04.5 LTS / NVIDIA GeForce RTX 5090** or **Google Colab / NVIDIA A100**. This setup was reported as common across model training stages. Native Windows was the storage/editing host; WSL/Linux and Colab are the model-runtime targets.

The root [requirements.txt](requirements.txt) preserves all 165 entries and original bytes of the **2025-12-11** environment snapshot. It includes `torch==2.9.0+cu130`, `torchvision==0.24.0+cu130`, `timm==1.0.22`, and `transformers==4.57.3`. This historical record is separate from the smaller inference dependency list. See [environment details](docs/execution_environment.md) for reconstruction instructions and the distinction between recorded and newly validated environments.

| Workflow | Target environment | Validation status |
|---|---|---|
| New inference CLIs and shared package | Linux/WSL; same implementation in Colab | Prior revision: paired no-TTA agreement on 10 real ECG images in WSL; paper-TTA execution on one synthetic image in Colab. Security revision: 31 tests and synthetic-image paper-TTA inference passed on native Windows CPU; WSL/CUDA no-TTA results for 10 real images matched frozen original-evaluator CSVs. Updated Colab and WSL paper-TTA execution remain pending |
| Original training and evaluation | Linux/WSL or Colab | Source retained; complete reproduction pending |
| Human-study administrator | Interactive Linux/WSL terminal with graphical display | Source retained; environment validation pending |

## Distribution

Patient ECG data, participant records, credentials, private originals, and local backups are excluded. No public patient-image sample is required to use the examples: users provide their own permitted images. Existing private ECG images and prediction CSVs are used for reproduction checks without being published.

Code and documentation are available in the [GitHub repository](https://github.com/ekagawa007/takotsubo_ami_ai_ecg). Model weights are intended for separate Release assets after validation and confirmation of their distribution terms. A download URL is not claimed until an asset is published. The paper-specific release can later be archived on Zenodo for a version-specific DOI.

## License and citation

The code is licensed under the [MIT License](LICENSE). Copyright (c) 2026 Eisuke Kagawa. External libraries retain their respective licenses. Model weights are not included in this code release; their license will be documented separately. The complete paper author list and final paper metadata remain to be confirmed. See the [publication and licensing plan](docs/publication_plan_ja.md).

Documentation version `0.1.9` identifies this README revision; `0.2.0.dev1` identifies the current inference package. Published release [v0.1.1](https://github.com/ekagawa007/takotsubo_ami_ai_ecg/releases/tag/v0.1.1), commit `70ae84ce96bf939ef117b6d2924318aef8716b81`, identifies the requirements-containing source release. It does not include this later inference revision. Existing release tags are retained.

## Revision history

Documentation versions track README revisions. Update the version/date above and add the newest entry first.

| Date (JST) | Documentation version | Changes |
|---|---|---|
| 2026-10-22 | 0.1.9 | Recorded the updated package 0.2.0.dev1/Pillow 13.0.0 WSL/CUDA float32 no-TTA check on 10 real ECG images against frozen original-evaluator CSVs. All three models and the ensemble passed tolerance with zero class changes; image and selected-weight hashes matched. The original evaluators were not rerun. Updated Colab, WSL paper-TTA execution and historical paper-TTA numerical parity remain unverified. |
| 2026-10-22 | 0.1.8 | Required Pillow 13.0.0/Python 3.11+ for inference, restricted ViT configuration loading, and required PyTorch 2.10+ for checkpoint export. Improved Colab restart guidance, unique result filenames and error reporting. Recorded the completed WSL no-TTA comparison and Colab paper-TTA execution at the prior commit, separately from this revision's 31 passing tests and CPU paper-TTA execution check; WSL/Colab revalidation remains pending. Preserved the historical requirements snapshot. |
| 2026-10-22 | 0.1.7 | Removed internal source-mapping, publication-edit, and security-review records and their verifier from the public tree. Added shared inference, per-model/ensemble CLI and Colab examples with configurable paper defaults; documented the selected ordinary ConvNeXtV2 checkpoint, inference-only weight export, and remaining prediction/runtime validation. Recorded the published requirements-containing v0.1.1 release. |
| 2026-10-21 | 0.1.6 | Consolidated the dependency record into a single root requirements.txt with an undated filename. Preserved all 165 package pins and original file bytes; retained the snapshot date (2025-12-11) in documentation and updated links. Research code and installation order are unchanged. |
| 2026-10-21 | 0.1.5 | Added the unchanged 2025-12-11 requirements snapshot (165 pinned packages), root requirements entry point, and Python 3.12.12 / WSL Ubuntu 22.04.5 / RTX 5090 or Colab A100 training environment. Added WSL installation guidance, distinguished the currently reported driver from the historical environment, and recorded the published v0.1.0 release. Research sources and checkpoint-selection wording are unchanged; clean installation and model reproduction remain unvalidated. |
| 2026-10-21 | 0.1.4 | Removed personal Colab authorship, Drive-file and executor metadata from two notebooks after a renewed 71-file security review; updated public hashes and review records. Notebook cell sources, model settings and Python scripts are unchanged. |
| 2026-10-21 | 0.1.3 | Recorded the author-confirmed checkpoint-selection method: review training and validation losses and select the lowest-validation-loss epoch without apparent signs of overfitting. Research sources are unchanged; exact checkpoint identities remain to be documented. |
| 2026-10-21 | 0.1.2 | Added the MIT License with Copyright (c) 2026 Eisuke Kagawa; recorded the confirmed licensing decision and checkpoint-selection reporting guidance. Research sources and model settings are unchanged. |
| 2026-10-21 | 0.1.1 | Generalized personal/internal paths, workstation labels, and the identifying-looking example filename; included the supplied scheduling helper unchanged; recorded original/public hashes and edit reasons for 36 supplied assets; updated verification and execution guidance; added a staged publication and licensing plan. Model execution validation remains pending. |
| 2026-10-21 | 0.1.0 | Introduced documentation versioning, the last-updated date, and this revision history. Recorded the current preparation state: 35 original source files preserved with SHA-256 verification; ViT fine-tuning retained as original notebooks and ConvNeXtV2/MLP-Mixer training retained as scripts with launcher notebooks; Linux/WSL/Colab execution targets documented, with interactive Linux/WSL required for the human-study administrator; ignore rules expanded for local data, checkpoints, study records, and generated outputs. Model execution validation remains pending. |

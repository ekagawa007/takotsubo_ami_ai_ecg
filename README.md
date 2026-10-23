# takotsubo_ami_ai_ecg

**Documentation version:** `0.1.14`

**Inference package version:** `0.2.1`

**Last updated:** 2026-10-23 (JST)

Research code for ECG-image models distinguishing takotsubo syndrome from acute anterior STEMI, including model development, ensemble evaluation, and a human diagnostic study.

## Current status

Version `0.2.1` protects result-file saving, retains incomplete weight installations for inspection, and limits image decoding to supported single-frame raster formats. It also provides the tested inference dependency set and a strict ViT checkpoint-name adapter for Transformers 5.10.4. Selected tensor values, paper settings and class order are unchanged. The downloader continues to use the **v0.2.0 weight Release**.

On 2026-10-23, the updated inference implementation passed a WSL/RTX 5090 float32 comparison against frozen original-evaluator results on the same ten private ECG images with no TTA. All three models and the ensemble had zero class differences and were within `atol=rtol=1e-5`. The same dependency set completed Colab/A100 inference on one generated 480 x 480 image with the paper TTA settings, live progress and result saving. These are scoped implementation and execution checks, not clinical validation or historical paper-TTA numerical equivalence.

The Colab notebook uses a separate inference virtual environment, defaults to a generated example, accepts an existing permitted image as an option, and downloads the matching prediction and run record in one ZIP. The tested pre-release workflow supplied the reviewed source as a verified archive. The final notebook selects Git tag `v0.2.1`; that GitHub checkout path requires publication of the tag and a subsequent public-workflow check. See [implementation status](docs/implementation_status.md) and [release notes](docs/releases/v0.2.1.md).

The original training, evaluation and human-study sources retain the supplied learning procedures, with personal paths and identifying-looking examples generalized. The [model card](MODEL_CARD.md) describes the task, selected checkpoints and limitations. See [weight terms](WEIGHTS_LICENSE.md) and [upstream notices](THIRD_PARTY_NOTICES.md); the ViT noncommercial condition applies to the complete weight bundle.

## Contents

| Location | Purpose |
|---|---|
| `requirements.txt` | Unmodified 165-package training environment record; snapshot date: 2025-12-11 |
| `requirements-inference.txt` | Dependencies for the new inference examples |
| `tools/download_weights.py` | Download the separately published inference assets |
| `MODEL_CARD.md` | Model purpose, inputs, defaults, provenance and validation scope |
| `WEIGHTS_LICENSE.md` / `THIRD_PARTY_NOTICES.md` | Separate weight terms and upstream notices |
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

From the repository root, create a fresh Python 3.11+ environment. For the tested WSL/NVIDIA CUDA 13.0 configuration:

```bash
python -m venv .venv-inference
source .venv-inference/bin/activate
python -m pip install pip==26.2.2 setuptools==84.2.1
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-inference.txt
python -m pip check
python tools/download_weights.py --output-dir weights
python examples/predict_ensemble.py --image data/example_ecg.png --weights-dir weights --output outputs/prediction.json
```

For CPU installation, replace the PyTorch index with `https://download.pytorch.org/whl/cpu`. Install the selected PyTorch/torchvision build before the inference requirements; the requirements pin the tested base versions. See [setup details](docs/inference.md#setup).

The download command requires the `v0.2.0` Release asset. If it is unavailable, check the release page or use existing local weights to run inference. Review the [weight terms](WEIGHTS_LICENSE.md) before use.

Progress is enabled automatically: the CLI displays a bar in an interactive terminal and plain updates when redirected. Add `--progress off` to suppress it. The Colab notebook displays a live progress bar. Progress messages go to stderr and are separate from saved prediction files.

The downloader prepares the weight layout described in [weights/README.md](weights/README.md).

### ECG image files for the examples

Provide one **single-frame image per ECG** in one of these formats:

| File format | Filename extensions |
|---|---|
| PNG | `.png` |
| JPEG | `.jpg`, `.jpeg` |
| BMP | `.bmp` |
| TIFF (one page) | `.tif`, `.tiff` |
| WebP (still image) | `.webp` |

The file's contents must be a supported image; renaming an EPS, PDF, SVG, GIF, DICOM or other file to `.png` does not convert it and will not make it accepted. Animated PNG/WebP and multipage TIFF files are rejected. PNG is a suitable choice for saving prepared ECG images without adding lossy compression.

The default input is a prepared **480 x 480 pixel** ECG image. The loader converts supported images to RGB. Other dimensions are rejected unless `--input-policy resize` is requested explicitly; that option resizes the full image without cropping or padding, and does not reproduce the study's upstream ECG preparation. Lead layout, cropping and other preparation must already match the intended model input.

`data/example_ecg.png` in the CLI command is a placeholder: supply your own authorized image and change the path. Colab generates a synthetic image by default; select `INPUT_SOURCE = "file"` and edit `USER_IMAGE_PATH` to use an existing image. Patient ECG images are not included in the repository or weight archive. Synthetic images used for software checks do not establish clinical performance.

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
| New inference CLIs and shared package | WSL/RTX 5090, Python 3.12.12; Colab/A100, Python 3.13.15 | v0.2.1: ten-image WSL no-TTA agreement; Colab synthetic-image paper-TTA execution. Both use torch 2.13.0+cu130, torchvision 0.28.0+cu130, Transformers 5.10.4 and Pillow 13.0.0. See [implementation status](docs/implementation_status.md). |
| Original training and evaluation | Linux/WSL or Colab | Source retained; complete reproduction pending |
| Human-study administrator | Interactive Linux/WSL terminal with graphical display | Source retained; environment validation pending |

## Distribution

Patient ECG data, participant records, credentials, private originals, and local backups are excluded. No public patient-image sample is required to use the examples: users provide their own permitted images. Existing private ECG images and prediction CSVs are used for reproduction checks without being published.

Code and documentation are available in the [GitHub repository](https://github.com/ekagawa007/takotsubo_ami_ai_ecg). The `v0.2.0` assets are `takotsubo_ami_ai_ecg_weights_v0.2.0.zip` and `SHA256SUMS.txt`. The three selected safetensors files total 958,657,088 bytes; the ZIP is 958,701,120 bytes. Use the [release page](https://github.com/ekagawa007/takotsubo_ami_ai_ecg/releases/tag/v0.2.0) to check asset availability. The separate weight terms are confirmed: ViT uses CC BY-NC 4.0 with the documented upstream notices; author-owned ConvNeXtV2 and MLP-Mixer contributions use MIT terms. Live public-download verification is recorded separately. The paper-specific release can later be archived on Zenodo for a version-specific DOI.

## License and citation

The code is licensed under the [MIT License](LICENSE). Copyright (c) 2026 Eisuke Kagawa. External libraries retain their respective licenses. Model binaries are supplied separately from the Git source tree. Weight terms are documented in [WEIGHTS_LICENSE.md](WEIGHTS_LICENSE.md): the complete bundle includes the ViT noncommercial condition. The code's MIT License does not remove those separate terms. The complete paper author list and final paper metadata remain to be confirmed. See the [publication and licensing plan](docs/publication_plan_ja.md).

Documentation version `0.1.14` identifies this README revision; `0.2.1` identifies the inference package. The weights retain version `0.2.0`. Published release [v0.1.1](https://github.com/ekagawa007/takotsubo_ami_ai_ecg/releases/tag/v0.1.1), commit `70ae84ce96bf939ef117b6d2924318aef8716b81`, identifies the requirements-containing source release. It does not include this later inference revision. Existing release tags are retained.

## Revision history

Documentation versions track README revisions. Update the version/date above and add the newest entry first.

| Date (JST) | Documentation version | Changes |
|---|---|---|
| 2026-10-23 | 0.1.14 | Integrated the tested inference dependencies and strict ViT checkpoint-name adapter into package 0.2.1. Recorded WSL/RTX 5090 ten-image no-TTA agreement and Colab/A100 synthetic-image paper-TTA execution. Colab now uses a separate environment, selectable synthetic/file input and one ZIP for both result records. Source publication and the final tag-based Colab checkout remain separate checks. |
| 2026-10-23 | 0.1.13 | Prepared package 0.2.1: protected result saving, non-destructive handling of incomplete weight installations, supported single-frame image formats, and matching CLI/Colab input checks. Added the ECG image format table and documented the reviewed v0.2.0 public-download/Colab record. Patch and dependency verification scope is recorded separately. |
| 2026-10-22 | 0.1.12 | Added package 0.2.0 with a separate weights Release archive, checksums, download command and Colab Release-download default with a local option. Added the model card, confirmed separate weight terms, upstream notices and release notes. Recorded reported Colab progress success while retaining the documented validation limits. Passed 50 tests, five simulated notebook download cases and fresh archive-install/CPU inference checks with unchanged prediction values. Live public Release and Colab-download checks are recorded separately from local validation. |
| 2026-10-22 | 0.1.11 | Added live English model/TTA/batch progress to the CLI and Colab notebook, configurable as auto, bar, JSON events or off. Progress is separate from saved predictions, requires no new dependencies and retains the inference settings and random-number sequence. Package 0.2.0.dev2 passed 38 tests, seven simulated notebook cases and CPU paper-TTA inference with prediction values exactly matching the prior CPU result. Actual Colab/WSL GPU execution of this progress revision remains unverified; earlier validation records remain tied to their tested versions. |
| 2026-10-22 | 0.1.10 | Translated the public Colab notebook instructions, comments and user-facing messages into English. Execution behavior and paper-default settings are unchanged. |
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

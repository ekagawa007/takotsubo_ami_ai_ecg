# takotsubo_ami_ai_ecg

**Documentation version:** `0.1.21`

**Inference package version:** `0.3.1`

**Last updated:** 2026-10-28 (JST)

Research code for ECG-image models distinguishing takotsubo syndrome from acute anterior STEMI, including model development, ensemble evaluation, and a human diagnostic study.

## DISCLAIMER

This repository and its associated model weights are intended for research. No warranty is given regarding the clinical accuracy, reliability, or fitness for clinical use of the code, model weights, predictions, or Grad-CAM visualizations.

To the fullest extent permitted by applicable law and subject to the applicable licenses, the authors and copyright holders accept no liability for claims, losses, damages, or other consequences arising from use of these materials or their outputs, including use in diagnosis or treatment decisions.

The warranty exclusions and liability limitations in the [code license](LICENSE) and the respective [model-weight licenses](WEIGHTS_LICENSE.md) apply. This notice summarizes intended use and does not amend those licenses.

## Current status

[Version **v0.3.1**](https://github.com/ekagawa007/takotsubo_ami_ai_ecg/releases/tag/v0.3.1) explicitly pins `filelock==4.2.1` in inference requirements and records its version in CLI/Colab output. Model computation, selected weights, paper TTA and Grad-CAM calculations are unchanged. On 2026-10-28, WSL/RTX 5090 checks passed 117 of 119 tests (two Windows-only cases skipped) and preserved all prediction values and both Grad-CAM maps exactly across the filelock substitution. A fresh Colab/A100 environment completed paper-TTA inference and Grad-CAM on one generated image; source identities, weight hashes and all nine saved files were verified. The Colab check used the reviewed candidate archive; final public-tag source acquisition remains a separate post-publication check. See [implementation status](docs/implementation_status.md) and [release notes](docs/releases/v0.3.1.md). The Zenodo software archive remains **v0.3.0**; weights remain **v0.2.0**.

[Version `0.3.0`](https://github.com/ekagawa007/takotsubo_ami_ai_ecg/releases/tag/v0.3.0) is published at commit `341ac955353f535fb74a4b6d8005a8b3b1290ad4` and adds optional Grad-CAM output for ViT and ConvNeXtV2. The prediction engine, selected weights, class order, paper TTA settings and raw-logit ensemble are unchanged from `v0.2.1`. The downloader continues to use the **v0.2.0 weight Release**. Anonymous source-download verification matched all 111 reviewed public files.

Add `--gradcam` to an existing ViT, ConvNeXtV2 or ensemble prediction command to save the prepared input, heatmaps, overlays and float32 numerical maps beside the prediction. Each map uses one unaugmented float32 pass. By default it targets that individual model's TTA-logit argmax and uses the same ImageNet normalization as current inference. The two model targets can differ. These maps are not attributions of the three-model ensemble or its TTA average. See [Grad-CAM methods and options](docs/gradcam.md).

The archived ViT visualizer used mean/std 0.5 rather than current inference normalization. `--cam-normalization legacy-vit` exposes that difference for explicit comparisons; it does not establish reproduction of historical reader-study images. MLP-Mixer predictions remain part of the ensemble, but no MLP-Mixer or ensemble heatmap is supplied. The archived reader-study code uses ViT and ConvNeXtV2 visualizations.

On 2026-10-24, the published `v0.3.0` Colab workflow completed paper-TTA inference and Grad-CAM on one generated 480 x 480 image using A100, Python 3.13.16 and CUDA float32. The recorded source/weight identities, prediction calculation, both numerical maps and all nine downloaded files were verified. This run did not include a paired Grad-CAM-off comparison. Earlier paired CPU and WSL checks on the `0.3.0.dev0` candidate confirmed exactly unchanged predictions when Grad-CAM was enabled; the `v0.2.1` baseline passed the ten-image float32/no-TTA comparison. See [implementation status](docs/implementation_status.md) for the full dated evidence and remaining scope.

The [English Colab example](https://colab.research.google.com/github/ekagawa007/takotsubo_ami_ai_ecg/blob/main/examples/colab_inference.ipynb) uses a separate inference environment, defaults to a generated image, displays per-model Grad-CAM overlays and downloads matching predictions, the run record and visualization files together. Its source-loading cell selects `GIT_REF = "v0.3.1"`. On 2026-10-28, the matching candidate archive completed the Colab workflow; the remaining code cells are unchanged in this release. The final public-tag acquisition check is tracked separately from that successful candidate run. The completed v0.3.0 public-tag validation remains a dated record.

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

For CPU installation, replace the PyTorch index with `https://download.pytorch.org/whl/cpu`. Install the selected PyTorch/torchvision build before the inference requirements; the requirements pin the base versions and the reviewed filelock maintenance update. See [setup details](docs/inference.md#setup).

The historical `requirements.txt` is a training-environment record, not the installation list for current inference. For an existing inference environment, review [maintenance validation](docs/implementation_status.md) before applying `python -m pip install -r requirements-inference.txt`; copying source files alone does not update an already installed `filelock`. Use a separate environment when preserving an earlier run.

The download command requires the `v0.2.0` Release asset. If it is unavailable, check the release page or use existing local weights to run inference. Review the [weight terms](WEIGHTS_LICENSE.md) before use.

Progress is enabled automatically: the CLI displays a bar in an interactive terminal and plain updates when redirected. Add `--progress off` to suppress it. The Colab notebook displays a live progress bar. Progress messages go to stderr and are separate from saved prediction files.

The downloader prepares the weight layout described in [weights/README.md](weights/README.md).

### Optional Grad-CAM

```bash
python examples/predict_ensemble.py --image data/example_ecg.png --weights-dir weights --output outputs/prediction_with_cam.json --gradcam
```

`--cam-target model-predicted` is the default. Use `--cam-target tako`, `--cam-target AMI`, or `--cam-target ensemble-predicted` for an explicit alternative target. The last option chooses the ensemble's predicted class for each individual model map; it does not create an ensemble attribution. `--cam-opacity` sets overlay opacity (default `0.6`). Existing output protection applies to all PNG, NPY and prediction files. Read numerical maps with `numpy.load(path, allow_pickle=False)`.

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
| Optional Grad-CAM candidate checks | `0.3.0.dev0`: Windows CPU; WSL/RTX 5090, Python 3.12.12; Colab/A100, Python 3.13.15 | Synthetic paper-TTA GPU runs completed on WSL and Colab; enabling Grad-CAM preserved all predictions exactly in the paired WSL check and earlier CPU checks. 119 local tests: 116 passed, three skipped. |
| Published `v0.3.0` inference and Grad-CAM | Colab/A100, Python 3.13.16, CUDA float32 | Completed on 2026-10-24 with one generated 480 x 480 image and paper TTA. Source/weight identities, prediction calculation, two maps and the nine-file ZIP were verified. No paired Grad-CAM-off comparison was performed in this Colab run; historical-map equivalence remains unverified. |
| Original training and evaluation | Linux/WSL or Colab | Source retained; complete reproduction pending |
| Human-study administrator | Interactive Linux/WSL terminal with graphical display | Source retained; environment validation pending |

## Distribution

Patient ECG data, participant records, credentials, private originals, and local backups are excluded. No public patient-image sample is required to use the examples: users provide their own permitted images. Existing private ECG images and prediction CSVs are used for reproduction checks without being published.

Code and documentation are available in the [GitHub repository](https://github.com/ekagawa007/takotsubo_ami_ai_ecg). The `v0.2.0` assets are `takotsubo_ami_ai_ecg_weights_v0.2.0.zip` and `SHA256SUMS.txt`. The three selected safetensors files total 958,657,088 bytes; the ZIP is 958,701,120 bytes. Use the [release page](https://github.com/ekagawa007/takotsubo_ami_ai_ecg/releases/tag/v0.2.0) to check asset availability. The separate weight terms are confirmed: ViT uses CC BY-NC 4.0 with the documented upstream notices; author-owned ConvNeXtV2 and MLP-Mixer contributions use MIT terms. Live public-download verification is recorded separately. The [v0.3.0 source archive](https://doi.org/10.5281/zenodo.23666137) and [v0.2.0 weight archive](https://doi.org/10.5281/zenodo.23666262) are also published on Zenodo with separate version-specific DOIs.

## License and citation

The code is licensed under the [MIT License](LICENSE). Copyright (c) 2026 Eisuke Kagawa. External libraries retain their respective licenses. Model binaries are supplied separately from the Git source tree. Weight terms are documented in [WEIGHTS_LICENSE.md](WEIGHTS_LICENSE.md): the complete bundle includes the ViT noncommercial condition. The code's MIT License does not remove those separate terms. The complete paper author list and final paper metadata remain to be confirmed. See the [publication and licensing plan](docs/publication_plan_ja.md).

### How to cite

If you use the software, please cite the software record below. If you also use the distributed trained model weights, please cite the weight record as well. [CITATION.cff](CITATION.cff) contains the current software citation and separate references to the archived v0.3.0 software and v0.2.0 weights. These citations identify the archived versions used; a later software or weight version should be cited using its own record.

**Current software (v0.3.1; GitHub Release, no new DOI):**

Kagawa, E. (2026). *takotsubo_ami_ai_ecg: research software for ECG-image classification and Grad-CAM* (Version 0.3.1) [Computer software]. https://github.com/ekagawa007/takotsubo_ami_ai_ecg/releases/tag/v0.3.1

**Archived software (v0.3.0):**

Kagawa, E. (2026). *takotsubo_ami_ai_ecg: research software for ECG-image classification and Grad-CAM* (Version 0.3.0) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.23666137

```bibtex
@software{kagawa_takotsubo_ami_software_2026,
  author = {Kagawa, Eisuke},
  title = {{takotsubo_ami_ai_ecg: research software for ECG-image classification and Grad-CAM}},
  year = {2026},
  version = {0.3.0},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.23666137},
  url = {https://doi.org/10.5281/zenodo.23666137}
}
```

**Trained model weights (v0.2.0):**

Kagawa, E. (2026). *takotsubo_ami_ai_ecg: trained ViT, ConvNeXtV2 and MLP-Mixer inference weights* (Version 0.2.0) [Trained model weights]. Zenodo. https://doi.org/10.5281/zenodo.23666262

```bibtex
@misc{kagawa_takotsubo_ami_weights_2026,
  author = {Kagawa, Eisuke},
  title = {{takotsubo_ami_ai_ecg: trained ViT, ConvNeXtV2 and MLP-Mixer inference weights}},
  year = {2026},
  version = {0.2.0},
  publisher = {Zenodo},
  note = {Trained model weights},
  doi = {10.5281/zenodo.23666262},
  url = {https://doi.org/10.5281/zenodo.23666262}
}
```

The software DOI identifies the source at the original v0.3.0 commit `341ac955353f535fb74a4b6d8005a8b3b1290ad4`. The citation file and subsequent documentation updates are maintained on the default branch; the archived source remains that release snapshot. The software's first release date is 2026-10-23, and the weights' first release date is 2026-10-22. Both Zenodo records were published on 2026-10-27. Weight use remains subject to the model-specific terms in [WEIGHTS_LICENSE.md](WEIGHTS_LICENSE.md).

Documentation version `0.1.21` identifies this README revision; `0.3.1` identifies the current inference package, while `0.3.0` remains the Zenodo-archived software version. CITATION.cff identifies v0.3.1 by its release URL and separately references the archived software and weights. No Zenodo DOI is assigned to v0.3.1 here. The weights retain version `0.2.0`. Published release [v0.1.1](https://github.com/ekagawa007/takotsubo_ami_ai_ecg/releases/tag/v0.1.1), commit `70ae84ce96bf939ef117b6d2924318aef8716b81`, identifies the requirements-containing source release. It does not include this later inference revision. Existing release tags are retained.

## Revision history

Documentation versions track README revisions. Update the version/date above and add the newest entry first.

| Date (JST) | Documentation version | Changes |
|---|---|---|
| 2026-10-28 | 0.1.21 | Finalized v0.3.1 with completed WSL/RTX 5090 dependency-parity checks and Colab/A100 candidate inference/Grad-CAM verification. Updated public validation notes and the current-version citation while retaining the existing Zenodo version DOIs, licenses and weight assets. Final public-tag Colab acquisition remains a separate check. |
| 2026-10-27 | 0.1.20 | Prepared v0.3.1 with an explicit filelock 4.2.1 inference pin and version recording. Refreshed remaining publication-status documents and Zenodo links, preserving dated validation evidence, historical training requirements, original weights and existing release/DOI identities. Candidate validation is reported separately. |
| 2026-10-27 | 0.1.19 | Added CITATION.cff for the archived v0.3.0 software, a separate v0.2.0 weight reference, and Zenodo DOI links with plain-text and BibTeX citations. Recorded Zenodo publication and distinguished the archived release from later citation and documentation updates. |
| 2026-10-24 | 0.1.18 | Recorded the published v0.3.0 commit, anonymous verification of all 111 source files, and completed public-tag Colab/A100 paper-TTA inference and Grad-CAM on one generated image with Python 3.13.16. Updated current documentation and Colab Markdown; code cells, dependencies, package version, weights and release tags are unchanged. |
| 2026-10-24 | 0.1.17 | Added a prominent research-use disclaimer covering code, weights, predictions and Grad-CAM, with warranty and liability terms subject to applicable law and each license. Updated both Colab validation summaries to record completed candidate GPU checks and distinguish pending public-tag and historical-map checks. |
| 2026-10-23 | 0.1.16 | Prepared package 0.3.0 with unchanged tested Grad-CAM implementation. Recorded WSL/RTX 5090 paper-TTA prediction preservation, Colab/A100 candidate execution and artifact checks, and completed public-tag v0.2.1 Colab verification. Historical-map equivalence and final v0.3.0 public-tag execution remain separate checks. |
| 2026-10-23 | 0.1.15 | Prepared optional ViT/ConvNeXtV2 Grad-CAM CLI and English Colab examples; documented targets, no-TTA map generation, legacy ViT normalization difference, protected visualization saving and pending GPU/historical-map validation. |
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

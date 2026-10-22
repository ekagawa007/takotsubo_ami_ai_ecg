# Execution environments

Updated 2026-10-22 (JST); documentation revision 0.1.9.

## Recorded training environment

The following setup was reported as common across the ViT, ConvNeXtV2, and MLP-Mixer training workflows, including their pretraining and fine-tuning stages. Each run used one of the two compute environments below.

| Item | Local training | Google Colab training |
|---|---|---|
| Host / runtime | Windows 11 with WSL | Google Colab GPU runtime |
| Linux distribution | Ubuntu 22.04.5 LTS (Jammy Jellyfish) | Runtime image version not separately recorded |
| Python | 3.12.12 | 3.12.12, as reported for the shared training setup |
| GPU | NVIDIA GeForce RTX 5090 | NVIDIA A100 |
| NVIDIA driver | 617.42 in the terminal output supplied during this documentation update; the historical training driver is not established | Not separately recorded |

### Dependency record

The root [requirements.txt](../requirements.txt) is the environment record from around the end of training, dated 2025-12-11. Its 165 pinned entries are preserved byte-for-byte. SHA-256: `4cfcdbc4261f5de46fbfad6cbb0dd726d01112171a336b507729cc701de270c1`.

The public filename has no date suffix; the date of the supplied record is retained in this document and the README. This single file directly includes the complete supplied package list, including Jupyter and NVIDIA support packages. It is an environment snapshot, not a minimal dependency list or a validated portable lock file. A separate Colab package export was not supplied, so identical versions of all Colab-managed packages are not independently established.

| Package / build | Recorded version |
|---|---|
| torch | 2.9.0+cu130 |
| torchvision | 0.24.0+cu130 |
| PyTorch CUDA build | 13.0 (`cu130`) |
| timm | 1.0.22 |
| transformers | 4.57.3 |
| accelerate | 1.12.0 |
| numpy | 2.2.6 |
| scikit-learn | 1.8.0 |

The CUDA build suffix does not identify the local CUDA Toolkit installation or the historical NVIDIA driver. Earlier commented installation examples in the original notebooks are retained as historical source; the requirements snapshot dated 2025-12-11 is the supplied later environment record.

## Completed inference checks before the security update

The following environments ran code commit `ff1b9bef9596a33cacf9dd67d42cd78d390a6211` on 2026-10-22. They describe new validation runs, separately from the historical training record above.

| Item | WSL paired comparison | Colab execution check |
|---|---|---|
| Input and scope | 10 private real ECG images, old evaluators versus new CLI, no TTA | One synthetic image with paper TTA settings |
| Device / precision | CUDA / float32 for all models | NVIDIA A100-SXM4-40GB / CUDA / float32 for all models |
| Python | 3.12.12 | 3.13.15 |
| torch | 2.9.0+cu130 | 2.11.0+cu130 |
| torchvision | 0.24.0+cu130 | 0.26.0+cu130 |
| timm / transformers | 1.0.22 / 4.57.3 | 1.0.22 / 4.57.3 |
| Pillow / numpy / safetensors | 12.0.0 / 2.2.6 / 0.7.0 | 12.0.0 / 2.2.6 / 0.7.0 |
| Result | All models and ensemble within `atol=1e-5`, `rtol=1e-5`; zero class differences | Inference and result saving completed |

## Current security revision: verified CPU development environment

On 2026-10-22, package `0.2.0.dev1` passed all 31 tests in a new isolated native Windows environment: Python 3.13.15, torch 2.11.0+cpu, torchvision 0.26.0+cpu, Pillow 13.0.0, numpy 2.2.6, timm 1.0.22, transformers 4.57.3 and safetensors 0.7.0. The three actual selected models also completed float32 inference and result saving on one synthetic 480 x 480 RGB image with the paper TTA settings, batch size 1 and zero workers. The raw-logit ensemble calculation was checked independently.

This is an updated CPU development check. The separate WSL/CUDA no-TTA result is recorded below; updated Colab and WSL paper-TTA execution remain unverified.

## Current security revision: verified WSL/CUDA no-TTA environment

On 2026-10-22, package `0.2.0.dev1` completed no-TTA inference on the same 10 archived real ECG images using Python 3.12.12, torch 2.9.0+cu130, torchvision 0.24.0+cu130, Pillow 13.0.0, numpy 2.2.6, timm 1.0.22, transformers 4.57.3 and safetensors 0.7.0. Image and selected safetensors weight hashes matched the preceding paired run. Settings were CUDA/float32, TTA 1 for all models, batch size 1, four workers, seed 1234 and TF32 disabled.

The updated CLI matched the frozen original-evaluator CSVs within `atol=1e-5`, `rtol=1e-5`, with zero class changes for all three models and the ensemble. The original evaluators were not rerun. This scope establishes neither updated WSL paper-TTA execution nor historical paper-TTA numerical parity.

## New inference and checkpoint-export requirements

The current inference requirements use **Pillow 13.0.0** and therefore require **Python 3.11 or newer**. The loader checks the Pillow version before opening input images. ViT configuration is limited to supported fields to prevent configuration-controlled external code loading. Keep transformers 4.57.3 for the selected ViT checkpoint architecture; this change does not assume compatibility with Transformers 5.

Checkpoint export from trusted `.pth` files additionally requires a **stable PyTorch version of at least 2.10.0**. Use a compatible torch/torchvision pair in that export environment. The safetensors-only inference path is separate, so the already exported and verified assets do not require another export. Preserve the original training snapshot as a historical record; do not install it over the new inference environment to resolve dependencies.

The preceding-commit checks used Pillow 12.0.0. The updated Pillow 13.0.0 environment has passed the scoped CPU and WSL/CUDA no-TTA checks above; updated Colab and WSL paper-TTA execution remain pending. See [implementation status](implementation_status.md).

## Runtime targets

The repository can be stored or edited on native Windows 11. Model execution targets Linux, including WSL, and Google Colab. This choice does not require converting the original ViT training notebooks into Python scripts.

| Component | Linux/WSL | Google Colab |
|---|---|---|
| Public per-model and ensemble inference | Target: CLI | Target: notebook calling the same implementation |
| Original SSL and fine-tuning | Configure local paths and dependencies; notebooks can run through Jupyter | Configure Drive or runtime-local paths and dependencies |
| Original dataset/CSV evaluation | Configure data and output locations | Configure data and output locations |
| Original human-study administrator | Interactive terminal and a working graphical display required | Not a standard notebook workflow; a separate interactive UI would be needed |

The table describes runtime targets for the retained code and new inference examples. The reported historical training setup is recorded above. Final weights and baseline predictions are not included in the Git repository. The preceding inference revision passed the scoped WSL/Colab checks listed above. The updated package has also passed the CPU and WSL/CUDA no-TTA checks; full training-environment reconstruction, updated Colab and WSL paper-TTA execution remain separate tasks.

## Paths

Use Linux paths when running inside WSL. With a typical WSL drive mount, a file stored under the Windows C drive is reached under `/mnt/c/`; the runtime should receive the corresponding Linux path, not a `C:\...` string. The exact mount location depends on the user's WSL configuration.

Keep new shared inference code independent of a specific workstation or Drive layout. Accept input image, model/checkpoint, data-root and output paths as arguments or documented configuration. Put Colab-specific mounting and uploads in the notebook. The private original files retain their paths. Public copies use generic Linux or Google Drive paths; configure them for the actual checkout and storage before execution. Notebook launch paths use the reorganized `training/` and `evaluation/original/` directories.

The new inference CLI should work without a graphical desktop: save optional plots to files, and let the Colab notebook display them. The human-study administrator has a separate interactive display requirement.

## Generic path examples and scheduling helper

| Purpose | Linux/WSL example | Colab/Drive example |
|---|---|---|
| Repository checkout | `/path/to/takotsubo_ami_ai_ecg` | `/content/drive/MyDrive/takotsubo_ami_ai_ecg` |
| ECG data | `/path/to/data` | `/content/drive/MyDrive/data` |
| Checkpoints | `/path/to/checkpoints` | `/content/drive/MyDrive/checkpoints` |
| Private outputs | `/path/to/outputs` | `/content/drive/MyDrive/outputs` |

These are configuration examples. Data and checkpoints are not supplied. Preserve the model/run subdirectories, checkpoint filenames, class directories, and experimental arguments when replacing path roots for a real run.

`ek_schedule_exe.py` is supplied unchanged at repository root. When launching scripts from a checkout in Linux/WSL, make that root available to Python (replace the example root first):

```bash
cd /path/to/takotsubo_ami_ai_ecg
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import ek_schedule_exe; print("Scheduling helper import OK")'
```

For Jupyter/Colab, add the configured checkout root before executing the original training cell:

```python
from pathlib import Path
import sys

repo_root = Path("/content/drive/MyDrive/takotsubo_ami_ai_ecg")
sys.path.insert(0, str(repo_root))
import ek_schedule_exe
```

On Linux Jupyter, use the actual Linux checkout path instead. Configure notebook `%run`/script paths and data/checkpoint roots as well. Importing the helper does not wait; calling `wait_until_target_time` can wait until the next day. Its clock follows the runtime's local timezone. Scheduling flags and original learning settings are retained.

## Reconstructing the WSL/Linux environment

These steps reconstruct the historical 165-package training snapshot. That complete reconstruction has not yet been tested in a clean environment. Use it only with trusted research inputs and checkpoints; use the current inference requirements and the separate patched export environment for their respective workflows. Use a dedicated environment with Python **3.12.12** already available and a GPU driver compatible with the CUDA 13.0 build. Ubuntu's default Python is not a substitute for the recorded Python version.

Run from the repository root in WSL/Linux:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python --version
```

Confirm that the output is `Python 3.12.12` before continuing. Install the matching PyTorch and torchvision builds from the official CUDA 13.0 index first, then the complete supplied snapshot from the normal package index:

```bash
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements.txt
python -m pip check
```

The first command selects the CUDA 13.0 builds matching the `+cu130` entries. The index option is scoped to that command; requirements.txt does not add another global package index. A resolver error must be investigated rather than bypassed with `--no-deps` or by silently changing the historical snapshot.

Inspect the resulting runtime without loading a research checkpoint:

```bash
python -c "import torch, torchvision, timm, transformers; print('torch:', torch.__version__); print('torchvision:', torchvision.__version__); print('timm:', timm.__version__); print('transformers:', transformers.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

An installation/import check does not establish model reproduction. Configure data/checkpoint paths and validate the selected training/evaluation workflow separately. The selected checkpoints and original baseline predictions are required for prediction comparisons.

### Google Colab

The reported Colab training setup used Python 3.12.12 and an NVIDIA A100. Colab provides a managed runtime whose current Python, GPU, driver, and preinstalled package versions must be checked for each reproduction attempt. The local Ubuntu version and local driver above do not describe the Colab runtime.

Use the historical requirements as the version reference. The complete 165-package snapshot includes notebook-server and NVIDIA support packages; it has not been validated as an in-place replacement for Colab's managed environment. No one-command Colab installation is claimed here. The new `examples/colab_inference.ipynb` installs inference dependencies separately and uses the same shared inference code as the CLI. The preceding inference revision completed the scoped Colab run recorded above. After the new inference dependency installation changes a loaded package, restart the Colab session and rerun the setup cells before inference. The updated notebook and Pillow 13.0.0 still require a new Colab execution check; see [the inference guide](inference.md).

### Other workflows

- Human-study administration requires an interactive Linux/WSL terminal and a working graphical display. Terminal/display validation remains pending.
- CPU and GPU inference setup is documented in [the inference guide](inference.md). The smaller `requirements-inference.txt` is separate from the historical CUDA training snapshot.

### Installation references

- [Official PyTorch 2.9.0 installation commands](https://pytorch.org/get-started/previous-versions/#v290)
- [pip freeze: installed-package snapshots](https://pip.pypa.io/en/stable/cli/pip_freeze/)
- [pip check: installed dependency compatibility](https://pip.pypa.io/en/stable/cli/pip_check/)

## Current verification

The historical environment is documented from the supplied record. The new inference package and examples have their own dependency list and validation process. A successful package import or model load is not evidence of complete scientific reproduction. The preceding revision passed a paired no-TTA check on 10 real images in WSL and paper-TTA execution on one synthetic image in Colab. The current package passed 31 tests and selected-model paper-TTA execution on a synthetic image in the updated Windows CPU environment. The updated package also passed the WSL/CUDA no-TTA comparison on 10 real images against frozen original-evaluator CSVs. Historical paper-TTA numerical parity, updated Colab and WSL paper-TTA execution remain pending. See [implementation status](implementation_status.md).

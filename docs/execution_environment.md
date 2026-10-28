# Execution environments

Updated 2026-10-28 (JST); documentation revision 0.1.22.

## Tested inference environments

Use the current `requirements-inference.txt` in a new environment. Keep the original training environment and its root `requirements.txt` as historical records. Version v0.3.1 explicitly pins `filelock==4.2.1`. On 2026-10-28, its candidate passed a WSL/RTX 5090 check with Python 3.12.12 and a separate Colab/A100 run with Python 3.13.16. Both used CUDA float32 and the torch 2.13.0/torchvision 0.28.0 CUDA 13.0 pair. The WSL check used an isolated filelock overlay and preserved all prediction values and both Grad-CAM maps exactly across that dependency substitution on one generated image with paper TTA (batch 1, zero workers). Colab created a separate inference environment and completed the same paper-TTA settings with batch 1 and four workers, including validated prediction/run linkage and nine output files. That candidate run used a verified source archive; the completed public-tag follow-up is recorded below. See [implementation status](implementation_status.md).

On 2026-10-28, the public v0.3.1 tag at commit `aa1c8f4596b63bed39c53e390e67318d1836b7f7` also completed the Colab workflow on A100-SXM4-40GB with Python 3.13.16, filelock 4.2.1 and CUDA float32. It used one generated 480 x 480 RGB image, paper TTA, batch size 1, four workers and seed 1234. The public and candidate Colab runs recorded the same 57 installed package names/versions, matching input bytes and inference settings. All model/ensemble prediction values matched exactly, and source/weight/output checks passed. The [implementation status](implementation_status.md) records the small ViT map difference and the scope of this synthetic-image check.

The following table retains the v0.2.1 environment and comparison record:

| Item | WSL GPU comparison | Colab execution | CPU development checks |
|---|---|---|---|
| Python | 3.12.12 | 3.13.15 | 3.13.15 |
| Device | NVIDIA GeForce RTX 5090 | NVIDIA A100-SXM4-40GB | Native Windows CPU |
| torch | 2.13.0+cu130 | 2.13.0+cu130 | 2.13.0+cpu |
| torchvision | 0.28.0+cu130 | 0.28.0+cu130 | 0.28.0+cpu |
| transformers | 5.10.4 | 5.10.4 | 5.10.4 |
| timm / numpy / Pillow / safetensors | 1.0.22 / 2.2.6 / 13.0.0 / 0.7.0 | Same | Same |
| pip / setuptools | 26.2.2 / 84.2.1 | 26.2.2 / 84.2.1 | 26.2.2 / 84.2.1 |
| Scope | Ten real ECGs, float32, no TTA; frozen original-evaluator comparison | One synthetic image, float32, paper TTA; execution and record checks | Ten-image no-TTA and synthetic paper-TTA dependency comparisons |

All numerical comparisons passed their stated `atol=rtol=1e-5` criterion. All models and the ensemble had zero class changes on the ten-image checks. Colab execution does not establish historical paper-TTA numerical equivalence. See [implementation status](implementation_status.md) for exact results and test skips.

Both the v0.2.1 public-tag workflow and the v0.3.0 public-tag workflow subsequently completed their live Colab checks. On 2026-10-24, public v0.3.0 ran on an A100 with Python 3.13.16, CUDA float32, batch size 1 and four workers, using one generated 480 x 480 RGB image and paper TTA. ViT/ConvNeXtV2 Grad-CAM, prediction/run-record linkage, source and weight identities, and the nine-file results ZIP were verified. Its main inference library versions match those listed above; this does not imply that all installed packages or runtime details were identical. The v0.3.0 candidate also passed a paired Grad-CAM off/on WSL paper-TTA check with exactly unchanged predictions on one generated image. See [implementation status](implementation_status.md) for the scoped records.

The notebook uses a separate inference virtual environment and records its source commit and installed versions. The v0.3.1 candidate and public-tag workflow checks are complete as described above. They are separate from the dated v0.3.0 checks and do not establish historical stochastic-TTA equivalence.

### New inference setup

Use Python 3.11 or newer. The steps below install the tested CUDA 13.0 build in WSL/Linux; the NVIDIA driver must support that build.

```bash
python -m venv .venv-inference
source .venv-inference/bin/activate
python -m pip install pip==26.2.2 setuptools==84.2.1
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-inference.txt
python -m pip check
```

For CPU inference, replace the PyTorch index with `https://download.pytorch.org/whl/cpu`. Install the chosen pair before the requirements file. The exact base-version pins also accept the matching `+cpu` or `+cu130` build. Different hardware/build choices require separate checks.

Pillow is checked before opening images. ViT configuration is restricted to supported fields, and the strict checkpoint-name adapter supports the selected legacy tensor state under Transformers 5.10.4. The selected weights are unchanged. Exporting trusted original `.pth` checkpoints is a separate operation requiring stable PyTorch 2.10.0 or newer; the public safetensors files need no re-export.

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

## Runtime targets

The repository can be stored or edited on native Windows 11. Model execution targets Linux, including WSL, and Google Colab. This choice does not require converting the original ViT training notebooks into Python scripts.

| Component | Linux/WSL | Google Colab |
|---|---|---|
| Public per-model and ensemble inference | Target: CLI | Target: notebook calling the same implementation |
| Original SSL and fine-tuning | Configure local paths and dependencies; notebooks can run through Jupyter | Configure Drive or runtime-local paths and dependencies |
| Original dataset/CSV evaluation | Configure data and output locations | Configure data and output locations |
| Original human-study administrator | Interactive terminal and a working graphical display required | Not a standard notebook workflow; a separate interactive UI would be needed |

The table describes runtime targets for the retained code and new inference examples. The reported historical training setup is recorded above. Final weights and baseline predictions are not included in the Git repository. The published v0.2.1 and v0.3.0 workflows passed the scoped WSL and Colab checks above. Full training-environment reconstruction and historical paper-TTA numerical equivalence remain separate tasks.

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

These are configuration examples. Patient data and original training checkpoints are not supplied; selected inference weights are available in the separate v0.2.0 Release. Preserve the model/run subdirectories, checkpoint filenames, class directories, and experimental arguments when replacing path roots for a real run.

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

Use the historical requirements as the version reference. The complete 165-package snapshot includes notebook-server and NVIDIA support packages; it has not been validated as an in-place replacement for Colab's managed environment. No one-command Colab installation is claimed here. The new `examples/colab_inference.ipynb` creates a separate inference virtual environment and uses its interpreter for image handling, downloading and CLI inference. Its tested workflow completed the A100 run described above. The notebook kernel's ML packages are retained; normally no restart is needed. After a manual restart, rerun from the first code cell. See [the inference guide](inference.md).

### Other workflows

- Human-study administration requires an interactive Linux/WSL terminal and a working graphical display. Terminal/display validation remains pending.
- CPU and GPU inference setup is documented in [the inference guide](inference.md). The smaller `requirements-inference.txt` is separate from the historical CUDA training snapshot.

### Installation references

- [Official PyTorch 2.9.0 installation commands](https://pytorch.org/get-started/previous-versions/#v290)
- [pip freeze: installed-package snapshots](https://pip.pypa.io/en/stable/cli/pip_freeze/)
- [pip check: installed dependency compatibility](https://pip.pypa.io/en/stable/cli/pip_check/)

## Verification limits

The original training setup is reported from the supplied record, while current inference checks use the versions and inputs listed above. Complete training reconstruction, full-dataset results, historical paper-TTA numerical agreement and human-study runtime checks are not established by this update. See [implementation status](implementation_status.md).

# Execution environments

Updated 2026-10-22 (JST); documentation revision 0.1.7.

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

The table describes runtime targets for the retained code and new inference examples. The reported historical training setup is recorded above. Final weights and baseline predictions are not included, and clean-environment reproduction has not been performed during publication preparation.

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

These steps are based on the supplied dependency record and the official PyTorch distribution instructions. They have not yet been tested in a clean environment. Use a dedicated environment with Python **3.12.12** already available and a GPU driver compatible with the CUDA 13.0 build. Ubuntu's default Python is not a substitute for the recorded Python version.

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

Use the historical requirements as the version reference. The complete 165-package snapshot includes notebook-server and NVIDIA support packages; it has not been validated as an in-place replacement for Colab's managed environment. No one-command Colab installation is claimed here. The new `examples/colab_inference.ipynb` installs inference dependencies separately and uses the same shared inference code as the CLI. Fresh Colab execution validation remains pending; see [the inference guide](inference.md).

### Other workflows

- Human-study administration requires an interactive Linux/WSL terminal and a working graphical display. Terminal/display validation remains pending.
- CPU and GPU inference setup is documented in [the inference guide](inference.md). The smaller `requirements-inference.txt` is separate from the historical CUDA training snapshot.

### Installation references

- [Official PyTorch 2.9.0 installation commands](https://pytorch.org/get-started/previous-versions/#v290)
- [pip freeze: installed-package snapshots](https://pip.pypa.io/en/stable/cli/pip_freeze/)
- [pip check: installed dependency compatibility](https://pip.pypa.io/en/stable/cli/pip_check/)

## Current verification

The historical environment is documented from the supplied record. The new inference package and examples have their own dependency list and validation process. A successful package import or model load is not evidence of complete scientific reproduction. Fresh WSL/Linux and Colab validation, and comparison against original patient-image predictions, remain required. See [implementation status](implementation_status.md).

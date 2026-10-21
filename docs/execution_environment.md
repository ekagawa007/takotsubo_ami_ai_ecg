# Execution environments

## Runtime targets

The repository can be stored or edited on native Windows 11. Model execution targets Linux, including WSL, and Google Colab. This choice does not require converting the original ViT training notebooks into Python scripts.

| Component | Linux/WSL | Google Colab |
|---|---|---|
| Public per-model and ensemble inference | Target: CLI | Target: notebook calling the same implementation |
| Original SSL and fine-tuning | Configure local paths and dependencies; notebooks can run through Jupyter | Configure Drive or runtime-local paths and dependencies |
| Original dataset/CSV evaluation | Configure data and output locations | Configure data and output locations |
| Original human-study administrator | Interactive terminal and a working graphical display required | Not a standard notebook workflow; a separate interactive UI would be needed |

These are target environments, not a statement that all historical scripts have been executed successfully in them. Final weights, some dependencies and baseline predictions are still missing.

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

## Environment setup plan

- WSL/Linux: use a dedicated Python environment and record the tested Python, PyTorch, torchvision, timm and transformers versions. Add requirements only after confirming compatibility with the final weights.
- Colab: call the same inference implementation from the notebook, install the tested compatible dependencies, and validate in a fresh runtime. Record the tested runtime and device.
- CPU inference is the initial functional target; CUDA acceleration is optional and must be checked with the actual models and TTA configuration.
- Linux training may require GPU resources and model-specific dependencies. Confirm each historical workflow's original environment separately.
- Human test requires terminal keyboard handling and graphical windows. Validate those capabilities before collecting real responses.

## Current verification

Current public-source hashes and documented path edits are checked without executing training or inference. The supplied scheduling helper is inspected separately; its wait function is not invoked during preparation. A WSL availability query from the current assistant environment returned `Wsl/E_ACCESSDENIED`; no WSL runtime test was completed. This is an access limitation of this session and does not establish that the user's WSL installation is faulty. No Colab model execution has been performed.

## Source integrity check

From the repository root in Linux/WSL:

```bash
python3 tools/verify_source_mapping.py
```

The verifier only reads files and compares hashes. It is independent of model libraries and does not validate prediction performance.

# Development code and provenance

Updated 2026-10-21. The initial layout preserved 35 supplied files byte-for-byte. The public preparation copy now generalizes identifying/personal paths and examples, while preserving filenames and notebook formats. The recovered `ek_schedule_exe.py` adds one unchanged source file (36 supplied assets total). The private original source directory is untouched. Source and destination hashes plus edit reasons are recorded separately.

## Original formats

| Stage/model | Retained implementation | Execution context |
|---|---|---|
| ViT SSL | `.py` files | Configure generic data/checkpoint paths; include repository root on Python's import path for the scheduling helper |
| ViT FT3/FT2 | `.ipynb` training bodies | Notebook cells contain actual code and configured paths |
| ConvNeXtV2 SSL | `.py` trainer, wrapper, LR finder and evaluator | Keep local helper modules together |
| ConvNeXtV2 FT3/FT2 | `.py` plus launcher `.ipynb` | Notebook `%run` arguments are part of the retained configuration |
| MLP-Mixer SSL | `.py` training and evaluation | Original source and supplied logs retained |
| MLP-Mixer FT3/FT2 | `.py` plus launcher `.ipynb` | Some notebooks contain several launch configurations |
| Evaluation | Original `.py` files and ensemble launch notebook | Single-image inference and CSV aggregation are distinct operations |
| Human study | Four original `.py` scripts | Preparation, randomization, and administration |

All seven supplied notebooks have empty saved output lists. They preserve code and launch settings, not a complete record of every executed experiment. We have not inferred the final paper run solely from the latest filename or a notebook cell position.

## Training chain to establish

For each model, record the selected SSL checkpoint, the FT3 checkpoint initialized from it, and the final FT2 checkpoint used for external validation. Record corresponding notebook cells/commands, class order, preprocessing, TTA, checkpoint selection, and ensemble settings. These associations remain to be confirmed against the actual checkpoint files and original prediction results. The author has confirmed the selection method: Training and validation losses were reviewed, and the checkpoint with the lowest validation loss among epochs without apparent signs of overfitting was selected. Record the selected checkpoint and epoch/step, the detailed validation split and loss values, and whether the evaluated state was an ordinary or averaged state. See [checkpoint_selection.md](checkpoint_selection.md).

The input ECG dataset and final weights are not present in the supplied 35 files. Input data schema, split files, class definitions, and patient-level separation must be documented from the actual training pipeline. Do not invent those details from filenames.

## Dependencies and execution

The original `ek_schedule_exe.py` has been supplied and copied unchanged to the repository root. It uses only `datetime` and `time`; importing it does not wait. Calling `wait_until_target_time` waits until the specified local clock time, choosing the following day if that time has passed. Configure the runtime timezone and scheduling flags intentionally. Include the repository root on `PYTHONPATH` for CLI workflows or `sys.path` for notebooks as described in [execution_environment.md](execution_environment.md). Generic Linux/Google Drive paths, including notebook launch paths and helper paths, must be configured for the actual checkout, data, checkpoints, and output storage before execution.

Separate the original development environment (where verifiable) from the environment validated for the new public inference examples. Exact historic versions are not established by commands such as `pip install -U`.

The execution targets are Linux (including WSL) and Google Colab. Native Windows is a storage/editing host, not a target runtime. The new inference examples will use the same Linux-compatible Python implementation in WSL and Colab. This is a target specification; runtime validation is pending. The human-study administrator imports POSIX-specific modules including `fcntl`, `termios`, and `tty`, and uses interactive terminal input and graphical windows. It targets an interactive Linux/WSL session with a working display rather than a standard Colab notebook. See [execution_environment.md](execution_environment.md).

## Original evaluation versus new examples

`evaluation/original/` retains original prediction and evaluation implementations as comparison references. The CSV ensemble script combines previously generated predictions and does not itself load all three models from an ECG image. Confirm its selected raw-logit/probability branch and class alignment against the paper's actual inputs.

The new `src/takotsubo_ami_ai_ecg/` and `examples/` will implement convenient inference. Validate each model and the ensemble against original results under the same preprocessing and TTA conditions. Do not silently replace original methods with simpler defaults. Historical training files do not need to be rewritten to import the new package.

## Publication edits

Original source hashes remain the provenance baseline. Public copies replace personal/internal paths, workstation labels, and the identifying-looking example with generic locations and `example_ecg.png`. Edited files are labeled `verified_publication_edit`; their current hashes are in the mapping. The edit log records hashes, reasons, and static verification status without reproducing sensitive old values. The original files remain outside the public repository. Static checks confirm path/documentation changes and source structure; no training or prediction-parity claim follows from these checks. See the security review for remaining runtime-output risks.

## Distribution

Release assets must include everything required to load the selected final models, including configuration and image-processor metadata when needed. Weight files have not been inspected and are not included here. Public asset hashes, release URLs, model-weight license, paper citation, and Zenodo DOI are pending. The code license is MIT, with Copyright (c) 2026 Eisuke Kagawa.

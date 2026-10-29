# Development code and provenance

Updated 2026-10-29 (JST); documentation revision 0.1.27. The initial layout preserved 35 supplied files byte-for-byte. The public preparation copy now generalizes identifying/personal paths and examples, while preserving filenames and notebook formats. The recovered `ek_schedule_exe.py` adds one unchanged source file (36 supplied assets total). The private original source directory is untouched. Private originals are retained for provenance; internal preparation records are not part of the public source tree.

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

For each model, record the selected SSL checkpoint, the FT3 checkpoint initialized from it, and the final FT2 checkpoint used for external validation. Record corresponding notebook cells/commands, class order, preprocessing, TTA, checkpoint selection, and ensemble settings. The selected final FT2 checkpoint identities and exported tensors have been checked; the complete SSL -> FT3 -> FT2 initialization chain and final prediction-table correspondence still require the original experiment records. The author has confirmed the selection method: Training and validation losses were reviewed, and the checkpoint with the lowest validation loss among epochs without apparent signs of overfitting was selected. Record the selected checkpoint and epoch/step, the detailed validation split and loss values, and whether the evaluated state was an ordinary or averaged state. See [checkpoint_selection.md](checkpoint_selection.md).

The original supplied 35 files did not contain the input ECG dataset or final weights. The selected inference weights are now distributed separately as v0.2.0; patient ECG data remain private. Input data schema, split files, class definitions, and patient-level separation must be documented from the actual training pipeline. Do not invent those details from filenames.

## Dependencies and execution

The original `ek_schedule_exe.py` has been supplied and copied unchanged to the repository root. It uses only `datetime` and `time`; importing it does not wait. Calling `wait_until_target_time` waits until the specified local clock time, choosing the following day if that time has passed. Configure the runtime timezone and scheduling flags intentionally. Include the repository root on `PYTHONPATH` for CLI workflows or `sys.path` for notebooks as described in [execution_environment.md](execution_environment.md). Generic Linux/Google Drive paths, including notebook launch paths and helper paths, must be configured for the actual checkout, data, checkpoints, and output storage before execution.

The supplied [2025-12-11 dependency snapshot](../requirements.txt) and the reported Python 3.12.12 / WSL Ubuntu 22.04.5 / RTX 5090 or Colab A100 setup are documented in [execution_environment.md](execution_environment.md). This setup was reported as common across all model training stages. Keep this historical record separate from the validated public inference environments and subsequent dependency-maintenance checks. Earlier installation comments in the notebooks are retained; they do not supersede the later supplied snapshot.

The execution targets are Linux (including WSL) and Google Colab. Native Windows is a storage/editing host, not a target runtime. The new inference examples use the same Linux-compatible Python implementation in WSL and Colab. At commit `ff1b9bef9596a33cacf9dd67d42cd78d390a6211`, WSL validation established paired float32/no-TTA agreement on 10 real ECG images, and Colab completed paper-TTA inference on one synthetic image. The updated package `0.2.0.dev1` with Pillow 13.0.0 also passed the WSL/CUDA no-TTA comparison on those 10 images against frozen original-evaluator CSVs; the original evaluators were not rerun. Subsequent WSL paper-TTA checks verified unchanged predictions with Grad-CAM enabled or disabled, and the public v0.3.0 Colab workflow completed paper-TTA inference and Grad-CAM on one generated image on 2026-10-24. These execution checks do not establish numerical equivalence to historical stochastic paper-TTA predictions. See [implementation status](implementation_status.md) for the separate validation records. The human-study administrator imports POSIX-specific modules including `fcntl`, `termios`, and `tty`, and uses interactive terminal input and graphical windows. It targets an interactive Linux/WSL session with a working display rather than a standard Colab notebook. See [execution_environment.md](execution_environment.md).

The v0.3.1 dependency update was checked separately on 2026-10-28: WSL preserved prediction values and both Grad-CAM maps across the filelock substitution, and Colab completed paper-TTA inference and artifact saving from a verified candidate source archive. A subsequent Colab run on the same date loaded the public v0.3.1 tag at commit `aa1c8f4596b63bed39c53e390e67318d1836b7f7` and completed inference and Grad-CAM. All 20 recorded inference-source hashes, official selected weights/configuration and the nine-file output ZIP were verified. These checks use one generated image and do not establish complete training or historical-result reproduction. See [implementation status](implementation_status.md).

## Original evaluation versus new examples

`evaluation/original/` retains original prediction and evaluation implementations as comparison references. The CSV ensemble script combines previously generated predictions and does not itself load all three models from an ECG image. The author confirmed use of its raw-logit averaging branch, with class order `0 = tako`, `1 = AMI`. The paper-default new ensemble uses equal weights and applies softmax after averaging model logits.

The `src/takotsubo_ami_ai_ecg/` and `examples/` directories implement public inference and optional ViT/ConvNeXtV2 Grad-CAM. The completed ten-image comparison covers no-TTA inference; a comparison against historical paper-TTA predictions requires matching preprocessing, dataset order, randomness and runtime conditions. Do not silently replace original methods with simpler defaults. Historical training files do not need to be rewritten to import the new package.

## Publication edits

Public copies replace personal/internal paths, workstation labels, and the identifying-looking example with generic locations and `example_ecg.png`. Personal Colab execution metadata was removed while preserving notebook cell sources and model settings. Original research files and preparation records remain private. Historical training sources are maintained separately from the new inference package.

## Distribution

Inference assets include the selected tensor states and required loading configuration. The final weight folder has been inspected; original checkpoints can contain training-machine paths and optimizer/training state, so the original folder is not a public distribution artifact. The published v0.2.0 bundle contains only the selected states and required configuration, with verified tensor values and metadata. Its checksums and acquisition instructions are recorded in the [inference guide](inference.md). Code v0.4.0 is archived at [10.5281/zenodo.23763944](https://doi.org/10.5281/zenodo.23763944), preserving tag commit `6a16326b761c198c11776155102dc31d1308bfeb`. The earlier code v0.3.0 remains archived at [10.5281/zenodo.23666137](https://doi.org/10.5281/zenodo.23666137); the unchanged weights v0.2.0 are archived separately at [10.5281/zenodo.23666262](https://doi.org/10.5281/zenodo.23666262). See [CITATION.cff](../CITATION.cff) for the current software release citation and references to the archived software and weights and [WEIGHTS_LICENSE.md](../WEIGHTS_LICENSE.md) for the separate model-weight terms. These version-specific DOIs identify those archives, not later source revisions. A final journal citation can be added when available. The code license is MIT, with Copyright (c) 2026 Eisuke Kagawa.

# Implementation and validation status

Updated 2026-10-22 (JST); documentation revision 0.1.9.

## Public source and inference implementation

Original training, evaluation, human-study, and scheduling sources retain their filenames and notebook formats, with personal paths/examples generalized. The MIT License names Copyright (c) 2026 Eisuke Kagawa. Published `v0.1.1` contains the historical requirements record; that release predates the new inference implementation. Patient data, prediction tables, original training checkpoints, and private preparation records are excluded from the public tree.

The selected inference states are ViT `checkpoint-6900`, ConvNeXtV2 `ckpt_0011.pth` ordinary `model` state, and MLP-Mixer `ckpt_0011.pth` ordinary `model` state. All 730 exported inference tensors matched their original names, shapes, dtypes and values. The training-only ViT `focal_loss.alpha` buffer was omitted explicitly. See [selected checkpoints](checkpoint_selection.md).

## Completed validation of the preceding inference revision

The following results were obtained on 2026-10-22 at code commit `ff1b9bef9596a33cacf9dd67d42cd78d390a6211`, before the Pillow update and loader restrictions in this revision. Private input files and prediction rows remain local.

| Check | Conditions | Result |
|---|---|---|
| Paired original evaluator versus new CLI | Same 10 real ECG images, RGB 480 x 480, identical selected weights, CUDA, float32, no TTA, batch size 1, four workers | All three models and the raw-logit ensemble agreed within `abs(new-old) <= 1e-5 + 1e-5 * abs(old)`; no class differences in any of the four comparisons |
| Colab inference | One synthetic image, NVIDIA A100-SXM4-40GB, CUDA, float32, ViT TTA 24/magnitude 1, ConvNeXtV2 TTA 1, MLP-Mixer TTA 32/magnitude 3 | Inference, JSON saving, and equal raw-logit ensemble followed by softmax completed successfully |

The largest absolute logit difference across the paired comparisons was `2.356872555e-7`; the largest probability difference was `7.453082962e-8`. All 60 individual-model logits were identical after parsing the two CSV formats back to float32. This establishes numerical agreement for the tested images and no-TTA conditions; the comparison is not an estimate of clinical performance.

The WSL comparison used Python 3.12.12, torch 2.9.0+cu130, and torchvision 0.24.0+cu130. Colab used Python 3.13.15, torch 2.11.0+cu130, and torchvision 0.26.0+cu130. Both used timm 1.0.22, transformers 4.57.3, Pillow 12.0.0, numpy 2.2.6, and safetensors 0.7.0. See [execution environments](execution_environment.md).

Earlier development checks at that revision also passed 17 regression tests, exact preprocessing comparisons with the original evaluator transforms, and selected-model inference on a generated image using native Windows/Python 3.10/CPU PyTorch 2.9.0. Those results describe the earlier environment; they do not validate the new Pillow 13 environment.

## Completed CPU validation of the current security revision

On 2026-10-22, the updated inference package `0.2.0.dev1` passed all 31 tests (17 existing inference tests and 14 security tests) in a new isolated native Windows environment. It used Python 3.13.15, torch 2.11.0+cpu, torchvision 0.26.0+cpu, Pillow 13.0.0, numpy 2.2.6, timm 1.0.22, transformers 4.57.3 and safetensors 0.7.0.

The actual three selected safetensors models also completed inference on one deterministic synthetic 480 x 480 RGB image using float32, CPU, batch size 1 and zero workers. The settings were ViT TTA 24/magnitude 1, ConvNeXtV2 TTA 1, MLP-Mixer TTA 32/magnitude 3, seed 1234 and equal ensemble weights. The prediction JSON was saved, and the mean of the raw logits followed by softmax was verified independently. This checks model loading, execution and aggregation in the updated CPU environment; it does not validate ECG classification performance or updated GPU runtimes.

A separate preprocessing check compared the preceding Pillow 12.0.0/torch 2.9.0/torchvision 0.24.0 CPU environment with the updated Pillow 13.0.0/torch 2.11.0/torchvision 0.26.0 CPU environment. For one generated 480 x 480 RGB image and fixed per-view seeds, all 57 transformed-tensor SHA-256 hashes matched exactly (24 ViT, 1 ConvNeXtV2 and 32 MLP-Mixer views). This checks those direct transforms; it does not establish DataLoader-worker or full-dataset TTA equivalence.

## Completed WSL/CUDA no-TTA validation of the current security revision

On 2026-10-22, the updated CLI in package `0.2.0.dev1` was run on the same 10 archived real ECG images with Python 3.12.12, torch 2.9.0+cu130, torchvision 0.24.0+cu130, Pillow 13.0.0, numpy 2.2.6, timm 1.0.22, transformers 4.57.3 and safetensors 0.7.0. The image and selected safetensors weight hashes matched the earlier comparison. Settings were CUDA, float32, TTA 1 for every model, batch size 1, four workers, seed 1234 and TF32 disabled.

The comparison used the frozen original-evaluator CSVs from the preceding paired run; the original evaluators were not rerun in the updated environment. All three models and the ensemble passed `abs(new-old) <= 1e-5 + 1e-5 * abs(old)`, with zero class changes in each 10-image comparison. The maximum absolute logit difference was `2.3568725548983593e-7`, and the maximum probability difference was `7.453082961905722e-8`. This confirms the updated inference path against those saved no-TTA results for the tested images. The updated CLI prediction CSV was also byte-for-byte identical to the saved new-CLI CSV from the preceding paired run: all 160 individual-model/ensemble logit and probability values had zero difference, with unchanged classes. This exact match applies to these 10 images under no-TTA conditions; the small differences reported against the separately formatted original-evaluator CSVs remain as stated above.

## Security changes and remaining target-runtime confirmation

This revision requires Pillow 13.0.0 and Python 3.11 or newer for inference, rejects unsupported ViT configuration fields, and requires a stable PyTorch version of at least 2.10.0 for exporting trusted `.pth` checkpoints. Existing verified safetensors assets remain usable without re-exporting them. New inference does not use the checkpoint exporter's pickle-loading path.

The Colab notebook now makes dependency-change restarts explicit, creates separate results for repeated runs, reports captured errors, and refuses to read a prior result after a failed subprocess. Distributed notebooks contain no saved runtime outputs.

The earlier paired WSL and Colab checks remain provenance for the preceding commit. The updated package has passed the CPU checks and WSL/CUDA no-TTA comparison described above. Colab execution with Pillow 13.0.0 and WSL execution using the paper TTA settings remain unverified; the completed no-TTA comparison does not establish historical paper-TTA numerical parity.

## Remaining reproduction work

- Numerical parity with historical predictions generated using the paper TTA settings remains unconfirmed. Match image preparation, dataset order, augmentation randomness, precision and package versions when making that comparison.
- A complete training rerun, all-dataset evaluation reproduction and human-study runtime validation have not been completed during this publication update.
- Complete the links from selected checkpoints to training runs, initialization chains, validation splits and loss records using the actual private records.
- Document final weight-distribution terms, actual asset URLs, checksums and the paper-specific source version before distributing model assets. Existing release tags are retained.

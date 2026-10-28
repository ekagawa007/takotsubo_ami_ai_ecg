# Documenting the selected model checkpoints

Updated 2026-10-28 (JST); documentation revision 0.1.21.

## What identifies the paper model

Record the checkpoint actually used for the reported evaluation for each model. Training may continue beyond the selected epoch; the selected model does not have to be the last saved state. On 2026-10-21, the author confirmed the following selection method:

> Training and validation losses were reviewed, and the checkpoint with the lowest validation loss among epochs without apparent signs of overfitting was selected.

The training and validation loss histories were used to assess apparent overfitting, with validation loss guiding the choice of epoch. The final checkpoint files and stored epoch/step values are recorded below. Selected loss values, the detailed validation split, originating run/initialization chain, and correspondence to the final prediction tables still need to be linked to the actual experiment records.

A filename identifies an asset but does not explain how it was selected. The public record should contain:

| Field | What to record |
|---|---|
| Model and training stage | ViT / ConvNeXtV2 / MLP-Mixer; SSL, three-class fine-tuning, or final two-class fine-tuning |
| Selected checkpoint | Public-relative path or asset filename, and SHA-256 when the actual file is available |
| Selected epoch / global step | Record the stored epoch and/or step; state whether epoch numbering is zero- or one-based when necessary. A filename's checkpoint number can be a global step rather than an epoch. |
| Training duration | Distinguish the number of epochs actually completed from the epoch whose checkpoint was selected. Note resumed runs if applicable. |
| Selection criterion | Confirmed: review training and validation losses, then select the minimum-validation-loss epoch among those judged without apparent overfitting. Record the selected loss value when available. |
| Selection dataset | Training and validation loss histories were reviewed. Identify the actual validation split and document patient separation where applicable. Any additional use of test/external data has not been established by this statement. |
| Selection procedure | Review of loss histories followed by selection of an epoch. Do not infer automatic early stopping or identity with an automatically saved best state from this description. |
| Checkpoint type | Raw epoch state, SWA/EMA or other averaged state, and any selected best/SWA variant actually evaluated |
| Originating run | Training file and notebook cell or launch command, run/configuration identifier, and parent checkpoint(s) for SSL -> FT3 -> FT2 |

At minimum, display the selected final FT2 checkpoint, epoch/step, and selection criterion/dataset for each of the three models. Link to the retained launch settings and initialization chain to explain its origin. Intermediate SSL and FT3 checkpoints can be documented in a separate provenance table.

## Describe the actual decision

The author-confirmed method is the minimum-validation-loss selection described above, informed by review of both training and validation losses for apparent overfitting. This statement records the reported decision process. It does not claim automatic early stopping, a prospectively fixed numerical overfitting threshold, or proof that overfitting was absent.

Use validation/development data for model selection and keep final test/external evaluation separate when following a held-out evaluation design. If final test or external results influenced checkpoint selection, record that accurately and reconsider which data support an independent performance estimate; do not silently state that they were unused. [scikit-learn model-selection guidance](https://scikit-learn.org/stable/modules/cross_validation.html).

## Notes from the retained implementation

The current FT2 sources use different automatic saving rules: ViT configures best-model loading by evaluation loss; ConvNeXtV2 updates its ordinary best state on higher validation accuracy; MLP-Mixer updates it on equal or higher validation accuracy. These are implementation facts, not evidence that the corresponding automatically saved model was ultimately used in the paper.

The training implementations also have SWA-related behavior. Record whether the evaluated state was the ordinary model or an averaged state, including the actual key loaded from a checkpoint when it contains both. A filename such as `swa_best.pth` alone does not establish that an averaged state was selected by comparing all validation results.

The retained ConvNeXtV2 and MLP-Mixer dataset evaluators choose the lexicographically last `.pth` file when given a directory. For the final evaluation record, specify the exact checkpoint file rather than relying on directory auto-selection. For a ViT checkpoint directory, identify the weight and configuration files used together. Keep the checkpoint identity and evaluation command linked.

## Selected final inference states

| Model | Selected source | Stored epoch / step | State used for inference |
|---|---|---|---|
| ViT | `checkpoint-6900/model.safetensors` with model configuration | Trainer state: global step 6900, epoch 12 | Saved model tensor state |
| ConvNeXtV2 | `ckpt_0011.pth` | Stored epoch value 11 | Ordinary `model` state |
| MLP-Mixer | `ckpt_0011.pth` | Stored epoch value 11 | Ordinary `model` state |

Epoch values above are reported as stored; do not reinterpret a checkpoint suffix as a one-based completed-epoch count without checking its training loop. The ConvNeXtV2 ordinary checkpoint was confirmed for inference on 2026-10-22. Its `swa_model` and separate `swa_best.pth` are not used for the paper-default public inference path.

Public inference assets use the selected tensor states exported to safetensors, with only required loading configuration. The original training checkpoints remain private. See [the inference guide](inference.md) for the selected TTA and ensemble settings.

## Current status

The selection criterion, selected final files, stored epoch/step values, and intended ordinary weight states are documented. At commit `ff1b9bef9596a33cacf9dd67d42cd78d390a6211`, a paired comparison on 10 private real ECG images confirmed numerical agreement within the stated tolerance between the original evaluators and new CLI with identical weights, float32 and no TTA. See [validation scope](implementation_status.md). The updated package `0.2.0.dev1` with Pillow 13.0.0 also passed the WSL/CUDA float32 no-TTA check on those 10 images against the frozen original-evaluator CSVs; image and selected-weight hashes matched, and the original evaluators were not rerun. The v0.2.1 dependency update also passed the ten-image WSL no-TTA check and a Colab/A100 synthetic-image paper-TTA execution check. The subsequent Grad-CAM checks completed WSL paper-TTA execution on a generated image with exactly unchanged predictions when Grad-CAM was enabled or disabled. The public v0.3.0 Colab paper-TTA and Grad-CAM workflow was verified on 2026-10-24. On 2026-10-28, the v0.3.1 dependency update passed separate candidate WSL and Colab checks using these same weight identities; no checkpoint was reselected or retrained. Numerical parity with historical paper-TTA predictions remains unconfirmed; those synthetic checks do not establish it. The validation loss values, split details, originating run/initialization chain and complete final prediction-table correspondence still need to be linked to the experiment records. No test-independence claim is inferred from this statement or from an automatic saving branch.

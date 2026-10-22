# Running inference

Updated 2026-10-22 (JST); documentation revision 0.1.9.

The new CLIs and Colab notebook share one inference implementation. The defaults record the paper's selected models, TTA counts/magnitudes, class order, and raw-logit ensemble. They do not establish that a new runtime reproduces every saved prediction; compare against the original results using the same inputs and execution settings.

## Setup

New inference requires Python 3.11 or newer and Pillow 13.0.0. Python 3.12.12 matches the reported training Python version. Install a compatible PyTorch/torchvision pair for the runtime first. For a local CPU environment:

```bash
python -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-inference.txt
python -m pip check
```

The CPU pair above passed the current development checks with Python 3.13.15 and Pillow 13.0.0. For WSL/CUDA, the updated package passed the scoped 10-image no-TTA check with Python 3.12.12, torch 2.9.0+cu130, torchvision 0.24.0+cu130 and Pillow 13.0.0; use a compatible NVIDIA driver. That check compared against frozen original-evaluator CSVs and did not rerun the original evaluators. WSL paper-TTA execution remains unverified. Historical and newly checked environments are recorded separately in [execution environments](execution_environment.md). In Google Colab, inspect its existing PyTorch/torchvision pair before changing packages; the notebook preserves that pair and installs the inference dependencies separately. Neither an imported package nor a successful installation alone validates predictions. The full historical training snapshot in `requirements.txt` is preserved separately.

The CLI checks Pillow before opening images and accepts only supported ViT configuration fields. Use the new inference requirements even when preserving an older training environment for provenance. The checkpoint exporter separately requires stable PyTorch 2.10.0 or newer and trusted source checkpoints. These restrictions address published [Pillow image parsing](https://github.com/python-pillow/Pillow/security/advisories/GHSA-5xm2-6676-qcp9), [Transformers configuration loading](https://github.com/advisories/GHSA-29pf-2h5f-8g72), and [PyTorch checkpoint loading](https://github.com/pytorch/pytorch/security/advisories/GHSA-63cw-57p8-fm3p) issues.

Place locally supplied inference assets in the [documented weight layout](../weights/README.md). ViT requires its model and model configuration; its image-processor configuration is retained with the assets for provenance. ConvNeXtV2 and MLP-Mixer use the selected ordinary tensor states exported as safetensors. The examples do not download unpublished weights or fall back to newly initialized prediction heads.

## Basic commands

Run from the repository root:

```bash
python examples/predict_ensemble.py --image data/example_ecg.png --weights-dir weights --output outputs/prediction.json
python examples/predict_vit.py --image data/example_ecg.png --weights-dir weights --output outputs/vit.json
python examples/predict_convnextv2.py --image data/example_ecg.png --weights-dir weights --output outputs/convnextv2.json
python examples/predict_mlpmixer.py --image data/example_ecg.png --weights-dir weights --output outputs/mlpmixer.json
```

Use `--help` for the full argument list. CLI wrappers work directly from the checkout. An editable install with `python -m pip install -e . --no-deps` is optional after installing dependencies.

Input choices are a repeated `--image` argument, `--input-dir` for recursively sorted images, or `--input-csv` with a `path` column. The CSV preserves row order. Select one input mode for a run. Image and weight paths can be outside the repository; use permitted local data and keep predictions private.

## Paper settings and configurable arguments

| Setting | Default |
|---|---|
| ViT weights | Selected `checkpoint-6900` tensor state |
| ConvNeXtV2 weights | `ckpt_0011.pth`, ordinary `model` state; no SWA |
| MLP-Mixer weights | `ckpt_0011.pth`, ordinary `model` state |
| ViT TTA | `--vit-tta 24 --vit-magnitude 1` |
| ConvNeXtV2 TTA | `--convnextv2-tta 1` (unchanged image only) |
| MLP-Mixer TTA | `--mlpmixer-tta 32 --mlpmixer-magnitude 3` |
| Ensemble | `--ensemble-weights 1 1 1`, ordered ViT, ConvNeXtV2, MLP-Mixer |
| Class order | `0 = tako`, `1 = AMI` |
| Ensemble decision | AMI if `prob_AMI >= 0.5`; otherwise tako |
| Individual-model decision | Argmax; an exact tie selects tako. `--threshold` explicitly overrides this rule |
| TTA seed | `--seed 1234` |
| Input dimensions | `--input-policy strict480` |
| Device | `--device auto`; use `cpu` or `cuda` explicitly if needed |
| Numeric precision | `--precision float32`; `bfloat16` is an explicit alternative |
| Batch and workers | `--batch-size 1 --num-workers 4` |

TTA counts include the unchanged view. ViT therefore evaluates 1 unchanged + 23 transformed views; MLP-Mixer evaluates 1 unchanged + 31 transformed views. Magnitude is not applied when TTA is 1. If enabling ConvNeXtV2 TTA, set `--convnextv2-magnitude` deliberately; its evaluator-derived magnitude default is 5, not a paper no-TTA setting.

For each model, average its raw two-class logits over TTA views. For the default equal-weight ensemble:

```text
ensemble_logits = (vit_mean_logits + convnextv2_logits + mlpmixer_mean_logits) / 3
ensemble_probabilities = softmax(ensemble_logits)
```

The implementation does not average model probabilities or apply temperature scaling. Custom ensemble weights are normalized before raw-logit averaging. Changing arguments changes the evaluated method; effective settings are recorded with results. `--threshold 0.5`, for example, explicitly selects AMI at or above 0.5 for every requested prediction, including individual models.

## Input preparation and comparison with historical results

The default requires an already prepared **480 x 480** ECG image and converts it to RGB. It uses the original dataset evaluators' ImageNet mean/std normalization. The ViT dataset evaluator's explicit normalization is distinct from its saved processor configuration; the new default follows that dataset evaluator. The original single-image convenience scripts may pad or crop differently and are not interchangeable preprocessing references.

With `--input-policy resize`, the complete input image is resized to 480 x 480 with bilinear interpolation. No automatic square cropping or black padding is performed. This explicit option makes arbitrary image dimensions runnable; it does not establish equivalence to the paper's upstream ECG image preparation.

The TTA implementation uses the retained evaluator transformations. The unchanged view is evaluated first; the first augmented pass uses `seed`, and subsequent augmented passes increment that seed by one. TTA logits are accumulated in float64 and their mean is converted to float32. The ensemble combines those mean logits in float64, and softmax probabilities are calculated in float64. Reproducing random views also depends on sample order, workers, batch settings, library versions, and random-number consumption. Running one selected image can differ from processing that image later in the original full dataset even with the same seed and TTA count. `--num-workers 0` provides an explicit single-process option when the runtime cannot launch worker processes; changing it also changes the random-view sequence. Float32 and bfloat16 execution may also produce different logits.

For parity checks, compare each model's raw logits first, then the raw-logit ensemble. Use the original private images and their corresponding rows in the original prediction CSVs. Do not infer full scientific reproduction from a smoke test on generated images, successful weight loading, or matching class labels alone.

## Outputs

Choose `.json` or `.csv` through the output filename. CSV output includes a JSON metadata sidecar. Results contain model and ensemble logits/probabilities, predicted class, effective settings, and model identification/checksums. Inputs are represented by sequential sample identifiers and hashes rather than personal filenames or absolute paths. Predictions and image hashes can still be sensitive, so save results to ignored `outputs/` or another private location and review them before sharing.

## Google Colab and sample images

Open `examples/colab_inference.ipynb`, select a suitable runtime, configure the local/Drive weight directory, and supply your own permitted ECG image. If dependency installation changes a loaded package, restart the session and rerun setup from the first code cell. The notebook checks for a required restart before importing inference libraries. Each inference execution selects a new output filename, displays subprocess messages and stops on failure before reading a result. Save the prediction JSON and its matching run record. The notebook invokes the same implementation as the CLI. It does not require a patient image to be published in this repository.

A generated image can check input handling and model execution but cannot validate ECG classification performance. Real prediction parity uses non-public images and existing results locally. The preceding commit `ff1b9bef9596a33cacf9dd67d42cd78d390a6211` passed paired no-TTA comparison on 10 private real images in WSL and paper-TTA execution on one synthetic image in Colab. The updated package `0.2.0.dev1` with Pillow 13.0.0 also passed the 10-image WSL/CUDA float32 no-TTA comparison against the frozen original-evaluator CSVs, with unchanged image/weight hashes and zero class changes. The original evaluators were not rerun. Historical paper-TTA numerical parity, updated Colab and WSL paper-TTA execution remain pending. See [implementation status](implementation_status.md) for the exact scope.

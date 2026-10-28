# Running inference

Updated 2026-10-28 (JST); documentation revision 0.1.21. Inference package 0.3.1. See [implementation status](implementation_status.md) for checks by version.

The new CLIs and Colab notebook share one inference implementation. The defaults record the paper's selected models, TTA counts/magnitudes, class order, and raw-logit ensemble. They do not establish that a new runtime reproduces every saved prediction; compare against the original results using the same inputs and execution settings.

## Setup

Use a fresh Python 3.11+ environment. The tested WSL runtime uses Python 3.12.12 and an RTX 5090; The v0.3.1 candidate Colab check used Python 3.13.16 and an A100. Both GPU checks used torch 2.13.0+cu130, torchvision 0.28.0+cu130 and filelock 4.2.1. WSL used an isolated dependency overlay; Colab used a separately installed inference environment. See [implementation status](implementation_status.md) for the completed candidate checks and separate final public-tag acquisition check. From the repository root in WSL/Linux:

```bash
python -m venv .venv-inference
source .venv-inference/bin/activate
python -m pip install pip==26.2.2 setuptools==84.2.1
python -m pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
python -m pip install -r requirements-inference.txt
python -m pip check
```

For CPU use, replace only the PyTorch index with `https://download.pytorch.org/whl/cpu`. The CPU pair was checked on native Windows with Python 3.13.15. CUDA requires a compatible NVIDIA driver. Other hardware or library combinations need their own execution checks. Install the intended PyTorch/torchvision build before the inference requirements, which pin the tested base versions. Do not install the historical `requirements.txt` over this environment.

The inference dependencies are torch 2.13.0, torchvision 0.28.0, timm 1.0.22, transformers 5.10.4, numpy 2.2.6, Pillow 13.0.0, safetensors 0.7.0 and setuptools 84.2.1. The list is separate from the preserved training snapshot. Colab creates its own inference virtual environment and invokes that interpreter; it keeps the notebook kernel's ML packages intact.

The loader checks Pillow before opening images, accepts only supported ViT configuration fields, and loads selected safetensors assets. The ViT compatibility adapter maps the documented legacy checkpoint names to the Transformers 5 layout, preserving tensors and strict key/shape checking. Exporting trusted original `.pth` checkpoints remains a separate operation requiring stable PyTorch 2.10.0 or newer; existing verified inference assets do not need re-exporting.

Review the [weight terms](../WEIGHTS_LICENSE.md) and [model card](../MODEL_CARD.md), then download the published assets:

```bash
python tools/download_weights.py --output-dir weights
```

The downloader uses the v0.2.0 Release, verifies the archive and every bundled file, and reuses matching complete files. Public download and all 11 member checksums were verified on 2026-10-22; the updated Colab workflow also completed weight preparation and inference. ViT requires the selected model and configuration; the image-processor configuration is retained for provenance. ConvNeXtV2 and MLP-Mixer use the selected ordinary tensor states. No newly initialized prediction head is substituted. See [weight layout](../weights/README.md).

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

## Live progress

Live progress, introduced in package `0.2.0.dev2`, reports model loading, the current model/TTA pass and completed inference batches in English. Choose the display with `--progress`:

| Mode | Display |
|---|---|
| `auto` (default) | A live bar when stderr is an interactive terminal; plain progress updates when redirected |
| `bar` | A terminal progress bar |
| `json` | One JSON event per line, prefixed with `ECG_PROGRESS`, for notebook or programmatic use |
| `off` | No progress messages |

For example, to run without progress:

```bash
python examples/predict_ensemble.py --image data/example_ecg.png --weights-dir weights --output outputs/prediction.json --progress off
```

Progress is written to stderr, separately from saved JSON/CSV predictions. It introduces no additional package dependency and retains the inference parameters and random-number sequence.

The overall count is the number of images multiplied by the sum of the selected models' TTA views. With the paper ensemble settings, one image requires 57 image evaluations (24 + 1 + 32); 10 images require 570. Percentages reflect completed image evaluations. The display shows elapsed time; model loading, saving and different models' processing speeds mean this percentage does not estimate time remaining.

The Colab notebook reads CLI JSON progress events as they arrive and updates an HTML bar. Other subprocess messages remain visible. Failed or cancelled runs stop before result loading, and the subprocess is terminated when notebook streaming fails. The tested Colab/A100 workflow completed paper-TTA inference and saving with this progress path. See [implementation status](implementation_status.md) for the exact validation scope.

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

Open `examples/colab_inference.ipynb` in a fresh GPU runtime. Its default source reference is `GIT_REF = "v0.2.1"`, available after that tag is published. The notebook records the resolved commit and source identity. The pre-release Colab check used a verified archive of the same inference implementation; the final public tag checkout is a separate post-publication check.

The notebook creates an inference virtual environment and uses its Python for package checks, image handling, weight downloading and inference. It does not replace the kernel's ML packages, so a kernel restart is normally unnecessary. After a manual restart, rerun from the first code cell. A failed partial environment setup is not marked ready; use a fresh runtime after an installation failure.

The default `INPUT_SOURCE = "synthetic"` generates a 480 x 480 RGB trace for an execution check. For an existing image, select `INPUT_SOURCE = "file"` and edit `USER_IMAGE_PATH`. Set `USE_GOOGLE_DRIVE = True` only if that path or local weights require Drive. `IMAGE_PATH` is derived from the selected input mode; existing files are not overwritten by the generator. The input must pass the same supported-format and dimension checks as CLI input.

The default `WEIGHT_SOURCE = "release"` downloads the v0.2.0 assets outside the checkout. `WEIGHT_SOURCE = "local"` accepts already available inference weights. Each inference execution creates a new timestamped prediction JSON and matching `.colab_run.json`. The final cell downloads both in one ZIP. The run record links the prediction by SHA-256 and records the source, library versions, input identity and actual device. Keep these records privately and clear notebook outputs before sharing.

A generated image checks execution, not ECG classification performance. The ten-image WSL no-TTA comparison and the one-image Colab paper-TTA execution are separate checks; neither establishes historical full-dataset paper-TTA numerical equivalence.

## Input formats and saving in v0.2.1

Use a single-frame PNG, JPEG, BMP, TIFF or WebP image. The loader checks the contents using only these decoders, then converts to RGB. Animated and multipage images are rejected, including files renamed to a supported extension. The default remains 480 x 480 pixels; explicit `resize` changes dimensions without reproducing upstream ECG preparation. See the [README image format table](../README.md#ecg-image-files-for-the-examples).

Prediction outputs are new files by default. `--overwrite` explicitly replaces a prediction output, but cannot authorize writing over an input image, input CSV or model/configuration file, including a hardlink alias. Both a CSV and its `.metadata.json` file are checked. They are separate filesystem updates: if either fails, the command reports an incomplete save, not success. Inspect any partial output and use a new filename before retrying.

Use an output folder that you control. If weight installation fails after copying starts, incomplete destination files are retained rather than removed through paths that may have changed. Choose a new private output folder if a retry detects a conflicting partial file. Matching complete weight files can still be reused. This behavior does not change archive or member SHA-256 checks.

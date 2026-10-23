# Inference examples

Updated 2026-10-23 (JST); documentation revision 0.1.14. Package 0.2.1.

The four CLIs and the English Colab notebook share the same implementation:

- `predict_vit.py`
- `predict_convnextv2.py`
- `predict_mlpmixer.py`
- `predict_ensemble.py`
- `colab_inference.ipynb`

## CLI setup

Use a fresh Python 3.11+ environment. Install the tested torch 2.13.0/torchvision 0.28.0 pair for your hardware first, then the inference requirements. See [the setup instructions](../docs/inference.md#setup) for the CUDA 13.0 and CPU commands. From the repository root after setup:

```bash
python tools/download_weights.py --output-dir ../ecg_inference_weights
python examples/predict_ensemble.py --image data/example_ecg.png --weights-dir ../ecg_inference_weights --output outputs/prediction.json
python examples/predict_vit.py --help
```

The downloader obtains the selected inference assets from the published [v0.2.0 Release](https://github.com/ekagawa007/takotsubo_ami_ai_ecg/releases/tag/v0.2.0), verifies the archive and every bundled file, and reuses matching completed files. Conflicting files are not overwritten. Existing local weights may also be used; see [weight layout](../weights/README.md).

Paper defaults are ViT TTA 24/magnitude 1, ConvNeXtV2 no TTA, MLP-Mixer TTA 32/magnitude 3, and equal-weight raw-logit averaging followed by softmax. The default input is a prepared 480 x 480 image. Paths and inference settings are configurable. The full bundle includes the ViT CC BY-NC 4.0 condition; the code remains MIT licensed. Read [weight terms](../WEIGHTS_LICENSE.md) before use.

## Colab setup and inputs

Open `colab_inference.ipynb` in a fresh GPU runtime. The source defaults to `GIT_REF = "v0.2.1"`, which requires that tag to be published. The notebook records the resolved commit and source identity. The pre-release Colab/A100 run used a verified archive of the reviewed implementation; the final GitHub checkout cell remains a separate public-workflow check.

Setup creates a separate inference virtual environment at `/content/ecg_inference_venv`. `INFERENCE_PYTHON` is used for library checks, image processing, weight downloading and inference. The kernel's existing ML packages are retained, so normally no kernel restart is required. If the kernel is restarted manually, rerun from the first code cell. An incomplete installation is not reused as a ready environment; start a fresh runtime after an installation failure.

The default `INPUT_SOURCE = "synthetic"` generates a 480 x 480 RGB trace for a software execution check. To use an existing image:

1. Set `INPUT_SOURCE = "file"`.
2. Set `USER_IMAGE_PATH` to your permitted image, for example `/content/drive/MyDrive/ecg_inference/data/example.png`.
3. Set `USE_GOOGLE_DRIVE = True` if that path requires Drive, then execute the Drive cell before preparing inputs or local weights.

`IMAGE_PATH` is derived from the selected mode. The generator does not overwrite a supplied image. A generated trace has no diagnostic interpretation and does not validate classification performance.

The default `WEIGHT_SOURCE = "release"` downloads the v0.2.0 assets outside the checkout. To use assets already present in the runtime or Drive, choose `WEIGHT_SOURCE = "local"` and configure `WEIGHTS_DIR`. Drive mounting is optional and explicit. Weight-preparation failure stops the workflow before inference.

## Progress and saved results

The CLI defaults to `--progress auto`: a live terminal bar interactively and plain updates when redirected. Other modes are `bar`, `json` and `off`. Progress goes to stderr, independently of saved predictions. The Colab notebook requests JSON progress events and displays a live HTML bar while streaming other messages. The percentage counts completed image/view evaluations, not estimated time; model loading and saving can pause the display.

Every inference execution chooses a new timestamped output name and clears any previous in-memory result. A failed or cancelled subprocess is stopped before reading a result. Success is shown only after the matching files are saved:

- `prediction_<model>_<run>.json`: predictions, settings and model checksums.
- `prediction_<model>_<run>.colab_run.json`: source identity, library versions, input identity and the prediction file's SHA-256.

The final cell defaults to `DOWNLOAD_RESULTS = True` and downloads one ZIP containing both JSON files from the same run. Keep them privately; clear notebook outputs before sharing. The public notebook contains no stored execution outputs or patient images.

## Input formats and saving

Use a single-frame PNG, JPEG, BMP, TIFF or WebP image. The loader checks file contents with the supported decoders and converts to RGB. Animated, multipage and disguised unsupported images are rejected. The default requires 480 x 480 pixels. Explicit `resize` resizes the entire image with bilinear interpolation and may change the aspect ratio; it does not reproduce upstream ECG preparation. See the [README format table](../README.md#ecg-image-files-for-the-examples).

Prediction outputs must be new by default. `--overwrite` can explicitly replace a prediction file, but cannot authorize writing over an input image, input CSV or model/configuration file, including a hardlink alias. CSV and metadata are separate filesystem updates: a partial save reports failure. Inspect partial files and choose a new filename before retrying. Use an output directory you control.

The downloader retains incomplete files after an installation failure. A retry reuses matching complete files; conflicting partial files require inspection or a new destination. Archive and member hash checks remain mandatory.

The tested v0.2.1 implementation passed WSL/RTX 5090 float32/no-TTA comparison on ten real ECGs and Colab/A100 paper-TTA execution on one generated image with the new dependency set. See [implementation status](../docs/implementation_status.md) for exact scope, test skips and the remaining public-checkout verification.

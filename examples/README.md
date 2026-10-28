# Inference examples

Updated 2026-10-28 (JST); documentation revision 0.1.22. Inference package 0.3.1.

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
python examples/predict_ensemble.py --image data/example_ecg.png --weights-dir ../ecg_inference_weights --output outputs/prediction_cam.json --gradcam
python examples/predict_vit.py --help
```

The downloader obtains the selected inference assets from the published [v0.2.0 Release](https://github.com/ekagawa007/takotsubo_ami_ai_ecg/releases/tag/v0.2.0), verifies the archive and every bundled file, and reuses matching completed files. Conflicting files are not overwritten. Existing local weights may also be used; see [weight layout](../weights/README.md).

Paper defaults are ViT TTA 24/magnitude 1, ConvNeXtV2 no TTA, MLP-Mixer TTA 32/magnitude 3, and equal-weight raw-logit averaging followed by softmax. The default input is a prepared 480 x 480 image. Paths and inference settings are configurable. The full bundle includes the ViT CC BY-NC 4.0 condition; the code remains MIT licensed. Read [weight terms](../WEIGHTS_LICENSE.md) before use.

## Colab setup and inputs

Open the [English Colab notebook](https://colab.research.google.com/github/ekagawa007/takotsubo_ami_ai_ecg/blob/main/examples/colab_inference.ipynb) in a fresh GPU runtime. The source-loading cell selects `GIT_REF = "v0.3.1"`. On 2026-10-28, the public tag at commit `aa1c8f4596b63bed39c53e390e67318d1836b7f7` completed paper-TTA inference, Grad-CAM and nine-file ZIP saving on A100 with Python 3.13.16 and filelock 4.2.1, using one generated 480 x 480 image. Source, selected-weight and saved-output checks passed. See [implementation status](../docs/implementation_status.md) for the separate candidate/public-tag records and prediction/map comparison. The notebook records the resolved source identity and installed versions. The [v0.3.0 notebook](https://colab.research.google.com/github/ekagawa007/takotsubo_ami_ai_ecg/blob/v0.3.0/examples/colab_inference.ipynb) remains available for the archived software version.

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

The final cell defaults to `DOWNLOAD_RESULTS = True` and downloads one ZIP containing both JSON files and, when enabled, all Grad-CAM artifacts from the same run. The run record includes artifact hashes, and the download cell checks them. The Grad-CAM ZIP includes a prepared input-image copy, so keep it private when the input is private; clear notebook outputs before sharing. The public notebook contains no stored execution outputs or patient images.

## Input formats and saving

Use a single-frame PNG, JPEG, BMP, TIFF or WebP image. The loader checks file contents with the supported decoders and converts to RGB. Animated, multipage and disguised unsupported images are rejected. The default requires 480 x 480 pixels. Explicit `resize` resizes the entire image with bilinear interpolation and may change the aspect ratio; it does not reproduce upstream ECG preparation. See the [README format table](../README.md#ecg-image-files-for-the-examples).

Prediction outputs must be new by default. `--overwrite` can explicitly replace a prediction file, but cannot authorize writing over an input image, input CSV or model/configuration file, including a hardlink alias. CSV and metadata are separate filesystem updates: a partial save reports failure. Inspect partial files and choose a new filename before retrying. Use an output directory you control.

The downloader retains incomplete files after an installation failure. A retry reuses matching complete files; conflicting partial files require inspection or a new destination. Archive and member hash checks remain mandatory.

The v0.2.1 prediction baseline passed WSL/RTX 5090 float32/no-TTA comparison on ten real ECGs and final public-tag Colab/A100 paper-TTA execution on generated images. Candidate `0.3.0.dev0` passed 116 of 119 local tests (three hardlink tests skipped), selected-weight CPU checks with and without paper TTA, and paper-TTA GPU execution on WSL/RTX 5090 and Colab/A100. Paired CPU and WSL runs confirmed exactly unchanged predictions with Grad-CAM enabled or disabled; the Colab run verified execution, source identity and saved artifacts without an on/off comparison. Published package `0.3.0` differs only in documentation and version metadata from that candidate. Its public-tag Colab/A100 workflow subsequently passed on 2026-10-24 with Python 3.13.16, CUDA float32 and paper TTA on one generated image. Recorded source/weight identities, prediction calculations and all nine downloaded files were verified. This public Colab run also did not include an on/off comparison. Historical-map equivalence remains unverified. See [Grad-CAM details](../docs/gradcam.md) and [implementation status](../docs/implementation_status.md).

## Optional Grad-CAM

Add `--gradcam` to a ViT, ConvNeXtV2 or ensemble command. The notebook enables the same option with `ENABLE_GRADCAM = True`. Prediction settings and raw-logit averaging remain unchanged. Grad-CAM runs afterward, with one unaugmented float32 input per selected visualization model. Ensemble inference produces separate ViT and ConvNeXtV2 maps; it does not produce an averaged ensemble map. MLP-Mixer alone requires Grad-CAM to be disabled.

The defaults are `--cam-target model-predicted`, `--cam-normalization inference`, and `--cam-opacity 0.6`. Choose `--cam-target tako` or `--cam-target AMI` for a fixed class, or `--cam-target ensemble-predicted` with ensemble inference. The last option uses the final ensemble decision, including an explicit prediction threshold. Each panel and JSON entry records its actual target.

The normalization default matches the current CLI's ImageNet preprocessing. The archived ViT visualizer instead used mean/std 0.5. `--cam-normalization legacy-vit` exposes that old ViT normalization as a comparison option; it does not change prediction preprocessing or prove reproduction of the reader-study images.

Beside the prediction file, each image receives a prepared RGB input PNG and, for each visualized model, a color heatmap PNG, overlay PNG and float32 `.npy` array. Arrays can be loaded with `numpy.load(path, allow_pickle=False)`. All names derive from the prediction stem and anonymous sample number. The map values are relative within one image/model, not probabilities. See [the method and output specification](../docs/gradcam.md).

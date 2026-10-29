# Inference examples

Updated 2026-10-29 (JST); documentation revision 0.1.24. Inference package 0.4.0.

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

Open the [v0.4.0 Colab notebook](https://colab.research.google.com/github/ekagawa007/takotsubo_ami_ai_ecg/blob/v0.4.0/examples/colab_inference.ipynb); its source-loading cell selects `v0.4.0`. On 2026-10-29, the prepared candidate completed eight Colab runs covering all three image sources, uploaded-image geometry options, A100 paper TTA and CPU paper/no-TTA inference with Grad-CAM. The separate final run starting from the public tag remains pending. See [implementation status](../docs/implementation_status.md#v040-image-input-and-colab-controls).

Use the form controls in order:

1. Choose automatic device selection, GPU or CPU. Automatic selection prefers an available GPU and warns before CPU fallback. Setup creates the corresponding separate inference environment; CPU uses float32 and defaults to zero workers.
2. Prepare the official weights, or select existing local weights. Connect Drive when using Drive-hosted weights.
3. Choose Synthetic example, Upload from computer or Google Drive. Upload one permitted image, or connect Drive and enter its path. Originals are unchanged.
4. Choose the input policy and run input preparation. Inspect the prepared image preview and its dimensions before inference.
5. Choose Paper, No TTA or Custom, then execute **Select the model, TTA and Grad-CAM** and check its printed settings before inference. Paper retains ViT 24/magnitude 1, ConvNeXtV2 1 and MLP-Mixer 32/magnitude 3 on either device; No TTA must show one view for every model. CPU fallback never reduces TTA automatically.

Image policy defaults to `strict480`. Optional `square-resize`, `center-crop` and white `pad` policies support other dimensions; legacy `resize` retains full-image stretching. See [input geometry and examples](../docs/image_preparation.md) for exact cropping/padding rules and their limits. Custom TTA counts include the unchanged view and must be at least 1; magnitudes range from 0 to 9. Magnitude changes augmentation strength, while reducing view counts reduces repeated inference work.

The notebook uses the dedicated `INFERENCE_PYTHON` for image decoding/preparation, downloading and inference. The kernel's existing ML packages are retained. Rerun setup after a manual restart; an incomplete environment is not reused as ready. An input-selection failure or cancellation invalidates pending input/results, so an older image is not silently substituted. Drive image loading and Drive-hosted weight loading are independent choices.

The default `WEIGHT_SOURCE = "release"` downloads the v0.2.0 assets outside the checkout. `WEIGHT_SOURCE = "local"` uses existing files in `WEIGHTS_DIR`. Weight-preparation failure stops inference. A generated trace is an execution check, not a clinical prediction example.

## Progress and saved results

The CLI defaults to `--progress auto`: a live terminal bar interactively and plain updates when redirected. Other modes are `bar`, `json` and `off`. Progress goes to stderr, independently of saved predictions. The Colab notebook requests JSON progress events and displays a live HTML bar while streaming other messages. The percentage counts completed image/view evaluations, not estimated time; model loading and saving can pause the display.

Every inference execution chooses a new timestamped output name and clears any previous in-memory result. A failed or cancelled subprocess is stopped before reading a result. Success is shown only after the matching files are saved:

- `prediction_<model>_<run>.json`: predictions, settings and model checksums.
- `prediction_<model>_<run>.colab_run.json`: source identity, library versions, input identity and the prediction file's SHA-256.

The final cell defaults to `DOWNLOAD_RESULTS = True` and downloads one ZIP containing both JSON files and, when enabled, all Grad-CAM artifacts from the same run. The run record includes artifact hashes, and the download cell checks them. The Grad-CAM ZIP includes a prepared input-image copy, so keep it private when the input is private; clear notebook outputs before sharing. The public notebook contains no stored execution outputs or patient images.

## Input formats and saving

Use a single-frame PNG, JPEG, BMP, TIFF or WebP image. The loader checks file contents with supported decoders and converts to RGB. Animated, multipage and disguised unsupported images are rejected. Dimensions are strict480 by default; the [input-policy options](../docs/image_preparation.md#select-the-geometry) must be selected explicitly. These options do not establish equivalence to the paper's upstream ECG preparation.

Prediction outputs must be new by default. `--overwrite` can explicitly replace a prediction file, but cannot authorize writing over an input image, input CSV or model/configuration file, including a hardlink alias. CSV and metadata are separate filesystem updates: a partial save reports failure. Inspect partial files and choose a new filename before retrying. Use an output directory you control.

The downloader retains incomplete files after an installation failure. A retry reuses matching complete files; conflicting partial files require inspection or a new destination. Archive and member hash checks remain mandatory.

The v0.2.1 prediction baseline passed WSL/RTX 5090 float32/no-TTA comparison on ten real ECGs and final public-tag Colab/A100 paper-TTA execution on generated images. Candidate `0.3.0.dev0` passed 116 of 119 local tests (three hardlink tests skipped), selected-weight CPU checks with and without paper TTA, and paper-TTA GPU execution on WSL/RTX 5090 and Colab/A100. Paired CPU and WSL runs confirmed exactly unchanged predictions with Grad-CAM enabled or disabled; the Colab run verified execution, source identity and saved artifacts without an on/off comparison. Published package `0.3.0` differs only in documentation and version metadata from that candidate. Its public-tag Colab/A100 workflow subsequently passed on 2026-10-24 with Python 3.13.16, CUDA float32 and paper TTA on one generated image. Recorded source/weight identities, prediction calculations and all nine downloaded files were verified. This public Colab run also did not include an on/off comparison. Historical-map equivalence remains unverified. See [Grad-CAM details](../docs/gradcam.md) and [implementation status](../docs/implementation_status.md).

## Optional Grad-CAM

Add `--gradcam` to a ViT, ConvNeXtV2 or ensemble command. In Colab, `ENABLE_GRADCAM = True` is the default; the prepared input, heatmaps and overlays appear in the notebook after successful inference. Grad-CAM runs afterward, with one unaugmented float32 input per selected visualization model, using the same selected input geometry as prediction. Ensemble inference produces separate ViT and ConvNeXtV2 maps; it does not produce an averaged ensemble map. MLP-Mixer alone requires Grad-CAM to be disabled.

The defaults are `--cam-target model-predicted`, `--cam-normalization inference`, and `--cam-opacity 0.6`. Choose `--cam-target tako` or `--cam-target AMI` for a fixed class, or `--cam-target ensemble-predicted` with ensemble inference. The last option uses the final ensemble decision, including an explicit prediction threshold. Each panel and JSON entry records its actual target.

The normalization default matches the current CLI's ImageNet preprocessing. The archived ViT visualizer instead used mean/std 0.5. `--cam-normalization legacy-vit` exposes that old ViT normalization as a comparison option; it does not change prediction preprocessing or prove reproduction of the reader-study images.

Beside the prediction file, each image receives a prepared RGB input PNG and, for each visualized model, a color heatmap PNG, overlay PNG and float32 `.npy` array. Arrays can be loaded with `numpy.load(path, allow_pickle=False)`. All names derive from the prediction stem and anonymous sample number. The map values are relative within one image/model, not probabilities. See [the method and output specification](../docs/gradcam.md).

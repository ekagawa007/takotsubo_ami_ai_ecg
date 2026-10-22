# Inference examples

The four CLIs call the same shared implementation as the Colab notebook:

- `predict_vit.py`
- `predict_convnextv2.py`
- `predict_mlpmixer.py`
- `predict_ensemble.py`
- `colab_inference.ipynb`

Use Python 3.11 or newer with Pillow 13.0.0. Install a compatible PyTorch/torchvision pair first as described in [the inference guide](../docs/inference.md), then run from the repository root:

```bash
python -m pip install -r requirements-inference.txt
python tools/download_weights.py --output-dir ../ecg_inference_weights
python examples/predict_ensemble.py --image data/example_ecg.png --weights-dir ../ecg_inference_weights --output outputs/prediction.json
python examples/predict_vit.py --help
```

The downloader obtains the public inference weights from the [v0.2.0 Release](https://github.com/ekagawa007/takotsubo_ami_ai_ecg/releases/tag/v0.2.0), once that release and its asset are published. It verifies the archive and extracted files against SHA-256 checksums bundled with the code. Complete, verified files are reused; conflicting files are not overwritten. You can also supply existing local inference weights as described in [weights/README.md](../weights/README.md). Paper defaults are ViT TTA 24/magnitude 1, ConvNeXtV2 no TTA, MLP-Mixer TTA 32/magnitude 3, and equal-weight raw-logit ensemble averaging. The default expects prepared 480 x 480 input images. All paths and inference settings are configurable; Colab-specific Drive mounting remains in the notebook.

The confirmed weight terms are CC BY-NC 4.0 for ViT (noncommercial use, with upstream Apache notices) and MIT for ConvNeXtV2 and MLP-Mixer. The code remains MIT-licensed. See [WEIGHTS_LICENSE.md](../WEIGHTS_LICENSE.md) before downloading.

The weight asset contains no patient images. Supply an input image that you are authorized to use. See [the full inference guide](../docs/inference.md) for arguments, preprocessing, output fields, and remaining validation.

## Progress display

The CLI defaults to `--progress auto`, which displays a live bar in an interactive terminal. Use `--progress bar` to request the text bar explicitly, `--progress json` for structured updates on stderr, or `--progress off` to disable progress output. The Colab notebook requests JSON updates and renders a live HTML bar while displaying ordinary logs immediately. Progress reporting does not change inference settings or predictions.

## Colab sessions and saved results

The notebook defaults to `GIT_REF = "v0.2.0"`, which becomes available after publication of that release. It checks for the security, progress, downloader, and bundled checksum descriptor files, and records the resolved commit SHA.

The default `WEIGHT_SOURCE = "release"` downloads and verifies the public weights into `/content/ecg_inference_weights`, outside the Git checkout. Download status is streamed as it runs. No Google Drive mount is required. To use weights already in Colab or Drive, set `WEIGHT_SOURCE = "local"` and set `WEIGHTS_DIR` to that directory; mount Drive only if your image or local weights require it. The notebook does not proceed to inference when weight preparation fails.

When the installation cell changes package versions, restart the Colab session and run the notebook from its first code cell. A marker outside the checkout prevents continued use of the same Python process after changes, including when an installation partially fails. The notebook does not restart or terminate the session automatically. Its Pillow check also rejects a stale or unsupported loaded version before opening an input image.

Each inference execution chooses a new timestamped output name. It streams logs and errors as they arrive, stops on a nonzero exit code, and clears the previous in-memory result before starting. A native HTML progress bar shows overall completion, the current model, TTA view, image count, and elapsed time; no widget package is required. The percentage counts completed image/view evaluations, not estimated time. It can pause while models load or files are saved. Cancelling the cell or encountering a streaming/display error terminates the inference subprocess. Final success is shown only after both result files are saved. The two saved files belong to the same execution:

- `prediction_<model>_<run>.json`: predictions, settings and model checksums.
- `prediction_<model>_<run>.colab_run.json`: code commit, runtime information and the prediction file's SHA-256.

Set `DOWNLOAD_RESULTS = True` in the final code cell to download both. Keep these files privately; clear notebook outputs before sharing an executed notebook. The earlier `colab_run.json` filename remains a valid historical record, but new runs use a separate record per prediction file.

On 2026-10-22, the author reported a successful Colab run with live progress at commit `6f4b59c`. The new Release-download workflow has not yet been executed on Colab. Earlier validation records and their scope remain separate; see [implementation status](../docs/implementation_status.md) for current checks and remaining target-runtime validation.

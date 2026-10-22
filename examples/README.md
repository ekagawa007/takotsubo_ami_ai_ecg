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
python examples/predict_ensemble.py --image data/example_ecg.png --weights-dir weights --output outputs/prediction.json
python examples/predict_vit.py --help
```

Supply local inference weights as described in [weights/README.md](../weights/README.md). Paper defaults are ViT TTA 24/magnitude 1, ConvNeXtV2 no TTA, MLP-Mixer TTA 32/magnitude 3, and equal-weight raw-logit ensemble averaging. The default expects prepared 480 x 480 input images. All paths and inference settings are configurable; Colab-specific Drive mounting remains in the notebook.

The examples do not supply patient images or download unpublished model weights. See [the full inference guide](../docs/inference.md) for arguments, preprocessing, output fields, and remaining validation.

## Colab sessions and saved results

Use a code revision that includes the current security fixes. Record the resolved commit SHA; do not pin the notebook to an older inference revision that lacks these changes.

When the installation cell changes package versions, restart the Colab session and run the notebook from its first code cell. A marker outside the checkout prevents continued use of the same Python process after changes, including when an installation partially fails. The notebook does not restart or terminate the session automatically. Its Pillow check also rejects a stale or unsupported loaded version before opening an input image.

Each inference execution chooses a new timestamped output name. It captures and displays stdout and stderr, stops on a nonzero exit code, and clears the previous in-memory result before starting. The two saved files belong to the same execution:

- `prediction_<model>_<run>.json`: predictions, settings and model checksums.
- `prediction_<model>_<run>.colab_run.json`: code commit, runtime information and the prediction file's SHA-256.

Set `DOWNLOAD_RESULTS = True` in the final code cell to download both. Keep these files privately; clear notebook outputs before sharing an executed notebook. The earlier `colab_run.json` filename remains a valid historical record, but new runs use a separate record per prediction file.

The WSL and Colab results recorded before this update used Pillow 12.0.0. They establish the stated earlier validation scope; they are not a claim that this updated notebook has already been executed on Colab. See [implementation status](../docs/implementation_status.md) for current checks and remaining target-runtime validation.

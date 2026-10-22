# Inference examples

The four CLIs call the same shared implementation as the Colab notebook:

- `predict_vit.py`
- `predict_convnextv2.py`
- `predict_mlpmixer.py`
- `predict_ensemble.py`
- `colab_inference.ipynb`

Install a compatible PyTorch/torchvision pair first as described in [the inference guide](../docs/inference.md), then run from the repository root:

```bash
python -m pip install -r requirements-inference.txt
python examples/predict_ensemble.py --image data/example_ecg.png --weights-dir weights --output outputs/prediction.json
python examples/predict_vit.py --help
```

Supply local inference weights as described in [weights/README.md](../weights/README.md). Paper defaults are ViT TTA 24/magnitude 1, ConvNeXtV2 no TTA, MLP-Mixer TTA 32/magnitude 3, and equal-weight raw-logit ensemble averaging. The default expects prepared 480 x 480 input images. All paths and inference settings are configurable; Colab-specific Drive mounting remains in the notebook.

The examples do not supply patient images or download unpublished model weights. See [the full inference guide](../docs/inference.md) for arguments, preprocessing, output fields, and remaining validation.

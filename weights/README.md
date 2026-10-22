# Local inference weights

Model binaries are excluded from Git. Place the separately supplied inference assets in this local layout:

```text
weights/vit/model.safetensors
weights/vit/config.json
weights/vit/preprocessor_config.json
weights/convnextv2/model.safetensors
weights/mlpmixer/model.safetensors
```

The selected sources are ViT `checkpoint-6900/model.safetensors`, ConvNeXtV2 `ckpt_0011.pth` ordinary `model` state, and MLP-Mixer `ckpt_0011.pth` ordinary `model` state. ConvNeXtV2 `swa_best.pth` and the checkpoint's `swa_model` are not used for paper-default inference.

Prepare inference-only files from trusted local checkpoints: export the selected tensor state and only the configuration required to load it, verify that every tensor is unchanged, and inspect the exported metadata. Do not distribute the original training folder, optimizer/scheduler states, training arguments, prediction CSVs, or training-machine paths. Keep the original research checkpoints privately for provenance.

## Export from local research checkpoints

The helper `tools/export_inference_weights.py` requires a **stable PyTorch version of at least 2.10.0**, exports trusted local checkpoints with restricted loading and verifies exact tensor bytes. Earlier versions are rejected before checkpoint loading or output creation. Use only checkpoints you own or otherwise trust, even with `weights_only=True`. Existing verified safetensors assets can be used directly for inference without re-exporting them. Supply the selected files explicitly and choose a **new** output directory; an existing directory is not overwritten:

```bash
python tools/export_inference_weights.py \
  --vit-checkpoint /path/to/vit/checkpoint-6900 \
  --vit-processor /path/to/vit/preprocessor_config.json \
  --convnextv2-checkpoint /path/to/convnextv2/ckpt_0011.pth \
  --mlpmixer-checkpoint /path/to/mlpmixer/ckpt_0011.pth \
  --output-dir /path/to/inference_assets
```

Point `--weights-dir` to the resulting directory. The export includes `manifest.json` with asset identification and checksums. The recognized ViT `focal_loss.alpha` buffer belongs only to the training loss; the exporter records its omission and verifies all retained inference tensors against the original. Other unknown tensors are not silently discarded. If a historical checkpoint cannot be read by the restricted loader, investigate and sanitize that trusted local checkpoint first; the public helper does not enable arbitrary object unpickling.

The new inference package reads safetensors rather than arbitrary pickled checkpoints. ViT configuration fields are checked against an explicit allowlist before constructing a model. Private execution-control fields and unsupported fields are rejected. ViT's image-processor configuration is retained with its model for provenance; the paper evaluator's explicit preprocessing is documented in [the inference guide](../docs/inference.md).

No published weight-download URL or weight license is asserted yet. A public distribution must identify the actual assets, checksums, and applicable weight license. The code's MIT License does not by itself document the separate weight distribution terms.

# Inference weights

Updated 2026-10-23 (JST); documentation revision 0.1.14. Weight assets remain v0.2.0.

Version `v0.2.0` uses the separate Release asset `takotsubo_ami_ai_ecg_weights_v0.2.0.zip` and `SHA256SUMS.txt`. Model binaries remain excluded from Git. The three selected safetensors files total 958,657,088 bytes before packaging; the archive is 958,701,120 bytes.

Run from the repository root:

```bash
python tools/download_weights.py --output-dir weights
```

Check the [Release page](https://github.com/ekagawa007/takotsubo_ami_ai_ecg/releases/tag/v0.2.0) for asset availability. If the asset is unavailable, the command reports a download failure; existing verified local assets remain usable. Review [WEIGHTS_LICENSE.md](../WEIGHTS_LICENSE.md), [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) and the [model card](../MODEL_CARD.md) before use. The complete bundle includes the ViT noncommercial condition.

The downloader or a local installation supplies this layout:

```text
weights/vit/model.safetensors
weights/vit/config.json
weights/vit/preprocessor_config.json
weights/convnextv2/model.safetensors
weights/mlpmixer/model.safetensors
```

The selected sources are ViT `checkpoint-6900/model.safetensors`, ConvNeXtV2 `ckpt_0011.pth` ordinary `model` state, and MLP-Mixer `ckpt_0011.pth` ordinary `model` state. ConvNeXtV2 `swa_best.pth` and the checkpoint's `swa_model` are not used for paper-default inference.

The distributed assets are inference-only files. Existing verified assets need no new checkpoint conversion. To prepare an export from trusted local checkpoints: export the selected tensor state and only the configuration required to load it, verify that every tensor is unchanged, and inspect the exported metadata. Do not distribute the original training folder, optimizer/scheduler states, training arguments, prediction CSVs, or training-machine paths. Keep the original research checkpoints privately for provenance.

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

The release archive passed installation into a fresh directory with all 11 expected file hashes verified. The installed models completed paper-TTA CPU inference on one synthetic image with prediction values exactly matching the prior baseline. The separate weight terms are confirmed. Public URL download and all 11 member hashes were verified on 2026-10-22; the tested v0.2.1 Colab/A100 workflow also completed weight preparation and paper-TTA execution with these unchanged assets. The archive checksum is recorded in `SHA256SUMS.txt`; the selected file identities are recorded in the asset manifest. See [v0.2.0 release notes](../docs/releases/v0.2.0.md). The code's MIT License does not replace the separate weight terms.

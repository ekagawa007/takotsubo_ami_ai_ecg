# Grad-CAM examples

Updated 2026-10-24 (JST). Package 0.3.0; documentation revision 0.1.18.

The optional Grad-CAM path adds ViT and ConvNeXtV2 visualizations after the existing prediction step. It reuses the selected safetensors and checks their identity against the prediction record. Predictions still use the requested TTA, precision and raw-logit aggregation. Grad-CAM uses a separate **unaugmented float32** forward pass and does not alter the saved prediction values.

## Run

```bash
python examples/predict_ensemble.py \
  --image data/example_ecg.png \
  --weights-dir ../ecg_inference_weights \
  --output outputs/prediction_with_cam.json \
  --gradcam
```

The same flag is available in `predict_vit.py` and `predict_convnextv2.py`. There is no MLP-Mixer or averaged ensemble heatmap in this release. The archived reader-study workflow used separate ViT and ConvNeXtV2 maps; the new interface follows that scope.

The English Colab notebook exposes `ENABLE_GRADCAM`, `CAM_TARGET`, `CAM_NORMALIZATION` and `CAM_OPACITY`. Set `ENABLE_GRADCAM = False` for prediction only, including MLP-Mixer alone. Its source-loading cell selects the published `v0.3.0` tag. The notebook on `main` includes the updated validation summary; the code cells are unchanged.

## Target and preprocessing

| Option | Meaning |
|---|---|
| `--cam-target model-predicted` (default) | Each visualized model's raw-logit argmax from the prediction run. Targets may differ between models. |
| `--cam-target ensemble-predicted` | The final ensemble decision, including an explicit prediction threshold. Requires ensemble inference. |
| `--cam-target tako` / `--cam-target AMI` | The requested class, regardless of the prediction. |
| `--cam-normalization inference` (default) | Current CLI normalization: ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`. |
| `--cam-normalization legacy-vit` | Mean/std `(0.5, 0.5, 0.5)` for the ViT visualization only; ConvNeXtV2 keeps ImageNet normalization. Prediction settings remain unchanged. |
| `--cam-opacity 0.6` (default) | Color overlay blend. Zero shows the prepared input; one shows the heatmap. |

`model-predicted` uses each model's argmax, including when a custom threshold was used for the final prediction decision. The record identifies both decisions. Maps use the selected class's **raw logit**, not its softmax probability.

The default input policy requires an already prepared 480 x 480 image. The explicit `resize` policy bilinearly resizes the entire image to 480 x 480, without cropping or padding. The visualization uses this prepared image. It is not mapped back to a differently sized original image.

## Calculations

The implementation follows the gradient calculations and layer defaults in `evaluation/original/ensemble/predict_single_image_ensemble.py`:

| Model | Layers | Map construction |
|---|---|---|
| ViT | `layernorm_before` outputs from the final three transformer blocks; supports the validated Hugging Face layer layout | Exclude the class token; average gradients across patch tokens to weight channels; sum weighted features and apply ReLU per layer; average the three maps; min-max normalize; bicubic resize to 480 x 480; normalize again. |
| ConvNeXtV2 | `stages.3.blocks.2.conv_dw` | Average gradients over spatial positions to weight channels; sum weighted features and apply ReLU; bilinear resize with `align_corners=False`; min-max normalize. |

Min-max normalization uses epsilon `1e-8`. Layer selection is explicit; an incompatible model layout raises an error rather than selecting another layer silently. Gradients are enabled for this step even though ordinary prediction uses inference mode. Hooks are removed after use.

Each exported map records the target class, target source, layer names, normalization and unaugmented logits. The latter can differ from the TTA-averaged prediction logits. A TTA-selected target does **not** make the map a visualization of the full TTA average.

## Saved files

For a prediction named `prediction_with_cam.json` and sample `sample_000001`:

- `prediction_with_cam_sample_000001_input.png`: prepared RGB input.
- `prediction_with_cam_sample_000001_vit_heatmap.png`: ViT color heatmap.
- `prediction_with_cam_sample_000001_vit_overlay.png`: ViT overlay.
- `prediction_with_cam_sample_000001_vit_cam.npy`: normalized float32 ViT map.
- The corresponding three `convnextv2` files when that model is selected.

All files are saved beside the prediction output. The record stores anonymous sample IDs and hashes rather than source image names or absolute paths. The input PNG and overlays still contain image content; metadata omission is not image anonymization. Keep these outputs private when the original ECG is private.

The PNGs use a blue-to-yellow scale, with larger map values shown in yellow. A constant map is rendered in neutral gray, its overlay leaves the image unchanged, and the record reports `map_is_constant`. The `.npy` files preserve quantitative normalized map values and can be loaded with `numpy.load(path, allow_pickle=False)`. They are not pickle files.

The Colab download cell packages the prediction JSON, matching execution-record JSON and all visualization artifacts in one ZIP. Before display/download, it checks expected filenames and the hashes recorded for that run. With Grad-CAM disabled, only the two JSON files are included. The CLI retains its existing collision, input-protection and safe-output checks.

## Relationship to the archived visualizations

This addition is **not yet a verified reproduction of the historical reader-study images**. The code supports inspection and comparison with explicit provenance.

- The archived ViT visualizer used its image processor's mean/std 0.5, while the current validated inference CLI uses ImageNet normalization. The default follows current inference. `legacy-vit` changes only the map preprocessing and is a comparison mode.
- The archived source uses OpenCV bicubic resizing for the ViT map. This implementation uses PyTorch bicubic interpolation with `align_corners=False`; numerical map comparisons must account for that implementation difference.
- The old visualizer used JET colors and composite plotting figures, including a separate ViT attention panel. This example exports separate blue-to-yellow maps and overlays. It does not add an attention-map panel.
- Original code establishes available defaults, but does not establish which per-run layer overrides or settings produced every historical reader-study image.

The following checks were completed on 2026-10-23 with candidate package `0.3.0.dev0`. The published `0.3.0` source differs only in documentation and version metadata from that tested candidate:

- Local suite: 119 tests, with 116 passed and three hard-link tests skipped because the local test environment could not create the required links.
- Selected model checkpoints, one generated 480 x 480 image, CPU float32: both no-TTA and paper-TTA (24/1/32) runs completed. Enabling Grad-CAM left all saved prediction values exactly unchanged in both comparisons.
- WSL/RTX 5090, Python 3.12.12, CUDA float32: a paired paper-TTA run on one generated image confirmed exactly unchanged predictions with Grad-CAM enabled or disabled. It used batch size 1 and zero workers.
- Colab/A100, Python 3.13.15, CUDA float32: paper-TTA inference with Grad-CAM completed on one generated image, with batch size 1 and four workers. The candidate archive and all 111 recorded source hashes matched; the downloaded ZIP contained the matching two JSON records and seven visualization artifacts. This run did not include a paired Grad-CAM-off comparison.
- Both GPU runs used torch 2.13.0+cu130, torchvision 0.28.0+cu130, transformers 5.10.4, timm 1.0.22, numpy 2.2.6, Pillow 13.0.0 and safetensors 0.7.0. Selected-weight identities, artifact hashes and both finite float32 480 x 480 maps were checked. The earlier CPU no-TTA overlays were visually inspected.

On 2026-10-24, the published `v0.3.0` Colab workflow completed paper-TTA inference and Grad-CAM on one generated 480 x 480 image using A100, Python 3.13.16 and CUDA float32. Source and official-weight identities, prediction calculations, both finite float32 maps and all nine downloaded files were verified. Heatmap and overlay pixels matched recalculation from the saved numerical maps. This Colab run did not include a paired Grad-CAM-off comparison. See [implementation status](implementation_status.md) for the exact release and environment.

These are synthetic software checks. Comparison against historical reader-study maps remains pending and requires the corresponding original images and settings.

## Interpretation

Grad-CAM is a coarse, class-specific visualization of relative positive contributions at the selected layers. A map value is neither a probability nor a calibrated measure of diagnostic importance. Each map is normalized independently, so color intensity is not directly comparable between images or models. Neither a visually plausible heatmap nor a successful synthetic-image run establishes clinical validity.

Method reference: Selvaraju et al., [Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization](https://arxiv.org/abs/1610.02391).

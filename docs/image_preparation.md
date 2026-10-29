# Image preparation and Colab options

Updated 2026-10-29 (JST); documentation revision 0.1.24. These options are included in version `0.4.0`. The tested input/device combinations and the pending final public-tag check are described below.

## Choose an image

The Colab notebook offers three input sources:

| Source | Action |
|---|---|
| Synthetic example (default) | Generate the same 480 x 480 RGB trace used for software checks. It has no diagnostic interpretation. |
| Upload from computer | Select one permitted image in the browser dialog. The file is transferred to the Colab runtime and stored under a generated name in a separate working directory. |
| Google Drive | Connect Drive when prompted and specify the image path in the form field. The original file is read without modification. |

Drive connection for existing model weights remains separate from image selection. A synthetic or uploaded image can still be used with weights stored on Drive.

Only single-frame PNG, JPEG, BMP, TIFF and WebP images are accepted. Format detection uses the supported decoders, not the filename extension. Cancelled uploads, unreadable files, unsupported formats and failed input preparation stop the run; earlier images and results are not reused as the new selection. Select the input and preparation policy, run the preparation cell, and inspect its preview before running inference.

## Select the geometry

The CLI argument and Colab input-policy selector use the same preparation implementation. Originals are unchanged. Cropping or padding happens before resizing and before any TTA.

| Policy | Accepted dimensions | Preparation |
|---|---|---|
| `strict480` (default) | Exactly 480 x 480 | Keep the prepared geometry. Any other dimensions stop inference. |
| `square-resize` | Any supported square image | Resize the square to 480 x 480 using bilinear interpolation. Rectangles are rejected. |
| `center-crop` | Square or rectangle | Take the centered square whose side is the shorter original edge, then bilinearly resize to 480 x 480. |
| `pad` | Square or rectangle | Center the image in a white RGB `(255, 255, 255)` square whose side is the longer original edge, then bilinearly resize to 480 x 480. |
| `resize` (legacy) | Square or rectangle | Bilinearly resize the whole image directly to 480 x 480. A rectangle's aspect ratio changes. |

For a 1000 x 600 image, `center-crop` keeps the central 600 x 600 region, removing 200 pixels from each horizontal side. `pad` produces a 1000 x 1000 canvas by adding 200 white pixels above and below. Both are then resized to 480 x 480. For odd differences, the left/top offset is rounded down; the extra pixel is removed from or added to the right/bottom side. The padding canvas is limited to 40,000,000 pixels before allocation; larger requested canvases are rejected.

Cropping can remove ECG leads, labels or waveform segments. Padding retains the full image but reduces its occupied area within the square. Check the prepared preview, including lead layout, before inference. These operations make additional image sizes runnable; they do not establish clinical suitability or reproduce undocumented upstream ECG preparation.

The inference input, Grad-CAM calculation and exported prepared-input image use the same selected geometry. A map describes the prepared 480 x 480 image and is not projected back to the original image. Geometry information is recorded with the prediction, alongside the original input hash and effective policy.

## CLI examples

Run after installing the inference dependencies and preparing the official weights:

```bash
# A square image with dimensions other than 480 x 480.
python examples/predict_ensemble.py --image data/square_ecg.png --weights-dir weights --output outputs/square.json --input-policy square-resize

# A rectangular image: choose one of these explicit policies.
python examples/predict_ensemble.py --image data/rectangular_ecg.png --weights-dir weights --output outputs/cropped.json --input-policy center-crop
python examples/predict_ensemble.py --image data/rectangular_ecg.png --weights-dir weights --output outputs/padded.json --input-policy pad
```

Add `--gradcam` to save the prepared input and separate ViT/ConvNeXtV2 visualizations. Add `--device cpu --precision float32 --num-workers 0` for CPU execution. CLI `--device auto` selects available CUDA, otherwise CPU; an explicitly requested unavailable `--device cuda` remains an error.

## Colab device and TTA controls

Automatic device selection prefers an available GPU. If CUDA is unavailable, Colab displays a CPU warning and uses CPU; CPU can also be selected explicitly. Setup selects the corresponding PyTorch 2.13.0 / torchvision 0.28.0 CPU or CUDA 13.0 builds in a separate inference environment. It checks the selected environment before use. CPU execution uses float32 and defaults to zero data-loader workers. Model weights and prediction aggregation remain the same.

CPU fallback does not reduce TTA automatically. Choose a preset explicitly:

| TTA preset | Views | Magnitudes |
|---|---|---|
| Paper (default) | ViT 24; ConvNeXtV2 1; MLP-Mixer 32 | ViT 1; MLP-Mixer 3; ConvNeXtV2 magnitude is unused |
| No TTA | 1 unchanged view per selected model | Unused |
| Custom | Enter an integer at least 1 for each model | Enter an integer from 0 to 9 for each model |

After changing a form value, run the corresponding cell before starting inference. In particular, run **Select the model, TTA and Grad-CAM** after choosing a TTA preset; check its printed effective settings. `No TTA` must show one view for each model. Changing a form without executing its cell does not update the active Python settings.

The view count controls how many predictions are averaged. Magnitude controls the strength of the random augmentation; lowering magnitude alone does not reduce the number of forward passes. Setting a count to 1 disables augmentation for that model. CPU runs with paper TTA can be slow; No TTA is an explicit execution option and changes the evaluated method. Recorded outputs retain the selected settings and actual device.

The Colab notebook enables Grad-CAM by default and displays the prepared input, ViT/ConvNeXtV2 heatmaps and overlays in the notebook. Each map uses one unaugmented float32 pass, regardless of prediction TTA. There is no MLP-Mixer or combined ensemble heatmap. See [Grad-CAM](gradcam.md) for target options and interpretation.

## Validation scope

Local Windows CPU validation completed on 2026-10-29 (JST; 2026-10-28 UTC): 169 of 172 tests passed, with three hardlink tests skipped because hardlinks were unavailable and no failures or errors. On one generated 480 x 480 image with paper TTA, all three models and the ensemble retained exactly the saved baseline logits and probabilities; both Grad-CAM maps and the prepared input pixels also matched exactly.

For each new policy (`square-resize`, `center-crop`, `pad`), a selected-weight ViT/no-TTA comparison against a separately prepared Pillow image produced identical prepared pixels, logits, probabilities and numerical Grad-CAM map. These checks establish correspondence to the specified image operations for the tested generated fixtures, not clinical performance after resizing, cropping or padding. See the [validation record](implementation_status.md#v040-image-input-and-colab-controls) for the runtime and scope.

On 2026-10-29 (JST), eight runs of the prepared candidate completed on Colab. A100 runs covered the generated strict480 image, a 720 x 720 computer upload with `square-resize`, a 721 x 480 upload with `center-crop` and `pad`, and a 480 x 480 Google Drive image, all using paper TTA. CPU runs used the generated strict480 image twice with paper TTA and once with `No TTA`; automatic device selection recorded CPU with CUDA unavailable. All runs used float32, enabled Grad-CAM, and produced linked nine-file result ZIPs whose recorded source/weight identities and saved artifacts were checked. The final run starting from the public `v0.4.0` tag remains pending. These are software workflow checks, not clinical validation or exhaustive coverage of every image, format or custom setting. See [implementation status](implementation_status.md#v040-image-input-and-colab-controls).
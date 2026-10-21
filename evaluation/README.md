# Original prediction and evaluation code

`original/vit/`, `original/convnextv2/`, and `original/mlpmixer/` retain the original single-image and labeled-directory evaluation scripts. `original/ensemble/` contains both direct image inference and aggregation of three prediction CSVs, plus the original launch notebook. All retained scripts keep their original filenames and processing structure. Public copies generalize personal paths and examples, including the image name `example_ecg.png`; original/current hashes and edits are recorded in the source mapping.

These implementations will serve as references for prediction parity. Public examples will be created separately under `examples/` and will call `src/takotsubo_ami_ai_ecg/`. Hash and static-structure checks cover the documented publication edits; prediction validation remains pending.

Predictions and visualizations can contain source image paths and ECG images. Use the ignored root `outputs/` location or another private location, and review outputs before any sharing.

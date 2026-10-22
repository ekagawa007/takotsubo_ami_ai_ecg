# Original prediction and evaluation code

`original/vit/`, `original/convnextv2/`, and `original/mlpmixer/` retain the original single-image and labeled-directory evaluation scripts. `original/ensemble/` contains both direct image inference and aggregation of three prediction CSVs, plus the original launch notebook. The files retain their original filenames and processing structure; public copies generalize personal paths and identifying-looking examples.

These implementations serve as references for prediction parity. New examples under `examples/` call the shared `src/takotsubo_ami_ai_ecg/` package. The paper settings use per-model raw-logit TTA averaging followed by equal-weight raw-logit ensemble averaging. The CSV ensemble reference is `original/ensemble/predict_ensemble3_testset_from_csv_rawlogits.py` in its logit-averaging mode.

Single-image and dataset evaluators have different preprocessing details. The new defaults follow the dataset-evaluation geometry and normalization and require prepared 480 x 480 inputs; see [the inference guide](../docs/inference.md). Matching TTA count/magnitude alone does not establish identical random views or reproduced predictions.

Predictions and visualizations can contain source image paths and ECG images. Use the ignored root `outputs/` location or another private location, and review outputs before sharing.

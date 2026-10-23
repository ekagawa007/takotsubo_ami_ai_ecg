# Inference validation

Updated 2026-10-23 (JST); documentation revision 0.1.16.

Install the current inference dependencies, then run from the repository root:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

## Grad-CAM validation for the prepared 0.3.0 release

The prepared `0.3.0` source changes only documentation and version metadata from the tested `0.3.0.dev0` candidate. The checks below retain the version at which they were performed.

On 2026-10-23, the candidate `0.3.0.dev0` suite ran **119 tests: 116 passed, three skipped, zero failures/errors** in native Windows, Python 3.13.15, torch 2.13.0+cpu and the unchanged v0.2.1 dependency set. The three skipped cases require hardlinks unavailable in the selected temporary directory. The Grad-CAM protected-hardlink test also passed separately in a workspace directory where hardlinks were supported; exact environment-specific skips are retained in private records.

New tests cover an analytic ConvNeXtV2 map, archived ViT channel-gradient calculations, HF4/HF5 layer layouts, target modes, normalization, constant maps, hook/state/RNG preservation, input/checkpoint identity, visualization serialization, protected outputs and prediction preservation. They use small generated models/images and do not establish clinical localization validity.

Separate selected-weight CPU checks on one generated 480 x 480 image passed with TTA disabled and with the paper TTA settings (24/1/32): the three-model/ensemble prediction values were exactly unchanged when Grad-CAM was enabled, both maps were finite float32 480 x 480 arrays, and saved PNG/NPY/JSON records were checked. A subsequent paired WSL/RTX 5090 run also passed with exactly unchanged predictions under paper TTA, CUDA float32, batch 1 and zero workers, using Python 3.12.12, torch 2.13.0+cu130, torchvision 0.28.0+cu130 and transformers 5.10.4.

Colab/A100 candidate execution completed under paper TTA with Python 3.13.15, the same core inference pins, CUDA float32, batch 1 and four workers. The matching candidate archive/source hashes, prediction/run records, seven visualization artifacts and nine-member ZIP were verified. This Colab run did not include a paired Grad-CAM-off comparison. The final `v0.3.0` public-tag checkout/run and comparison with historical reader-study heatmaps remain pending. See [Grad-CAM](../docs/gradcam.md) and [implementation status](../docs/implementation_status.md).

## Inherited v0.2.1 checks

The v0.2.1 implementation's 85 discovered cases cover input validation, strict checkpoint loading and ViT key compatibility, class order, raw-logit aggregation, progress, downloading and file protection. They require no patient data or model download.

On 2026-10-23, WSL ran 85 cases: 83 passed and two Windows-only tests were skipped (alternate streams and directory-sharing behavior). Native Windows CPU also ran 85 cases: 83 passed and two hardlink tests were skipped because of account permissions. Actual skip names/reasons must be retained when reporting another environment.

The selected-model checks are separate from the suite: WSL/RTX 5090 float32/no-TTA inference on ten private images matched frozen original-evaluator results within `atol=rtol=1e-5`, with zero class changes for all three models and the ensemble. Colab/A100 completed one synthetic-image paper-TTA run with the same dependency set and checked result records. The tested implementation uses torch 2.13.0, torchvision 0.28.0, transformers 5.10.4 and Pillow 13.0.0; see [implementation status](../docs/implementation_status.md) for builds, exact differences and scope limits.

The pre-release v0.2.1 Colab workflow supplied the reviewed code as a verified archive. Two subsequent public-tag runs recorded commit `95bbe9998f7433e3f65eb000e2d70ac2b4e3ff32`, package 0.2.1 and the expected source/weight hashes; prediction/run linkage, raw-logit aggregation and both two-JSON ZIPs were verified. Historical paper-TTA numerical equivalence and clinical performance are not established by either test suite or generated-image execution.

## Earlier dated checks

The following entries retain their then-current evidence and pending items. Current results are summarized above.

On 2026-10-22, candidate package `0.2.0` passed the complete 50-test suite: 17 original inference tests, 14 security tests, seven progress tests and 12 downloader tests. A separate notebook harness passed five simulated download cases. The isolated native Windows runtime used Python 3.13.15, torch 2.11.0+cpu, torchvision 0.26.0+cpu and Pillow 13.0.0.

The prepared 958,701,120-byte Release ZIP was installed into a fresh directory with all 11 file SHA-256 checks passing. The three installed models completed CPU/float32 inference on one synthetic 480 x 480 RGB image using paper TTA, batch size 1, zero workers and seed 1234. Prediction values exactly matched the prior CPU baseline; live progress and output saving also passed. These checks used the prepared local archive. Public Release-download and actual Colab-download execution remain unverified pending publication.

On 2026-10-22, package `0.2.0.dev2` passed the complete 38-test suite: 17 original inference tests, 14 security tests and seven progress tests. The isolated native Windows environment used Python 3.13.15, torch 2.11.0+cpu, torchvision 0.26.0+cpu and Pillow 13.0.0.

The three selected models also completed CPU/float32 inference on one synthetic 480 x 480 RGB image with the paper TTA settings, batch size 1, zero workers and seed 1234. Progress was received while the subprocess ran, reached 57/57 image evaluations, and reported completion after output saving. Prediction values exactly matched the preceding `0.2.0.dev1` CPU baseline, with the same inference settings and checkpoint identities. A separate notebook harness passed seven simulated subprocess cases. Colab execution with the English progress notebook was subsequently reported successful; its saved runtime record was not reviewed for this update. WSL/GPU execution of the progress version remains unverified.

On 2026-10-22, the preceding package `0.2.0.dev1` passed all 31 tests (17 existing inference tests and 14 security tests) in an isolated native Windows environment using Python 3.13.15, CPU PyTorch 2.11.0, torchvision 0.26.0 and Pillow 13.0.0. The three selected models also completed float32 inference on one synthetic 480 x 480 RGB image using the paper TTA settings, batch size 1 and zero workers. Result saving and the mean-raw-logit/softmax ensemble calculation were verified. This development check does not replace WSL/CUDA or Colab execution.

Before the current security revision, all 17 inference tests passed on 2026-10-22 at commit `ff1b9bef9596a33cacf9dd67d42cd78d390a6211` on Windows with Python 3.10, CPU PyTorch 2.9.0 and torchvision 0.24.0. The checks include exact transform comparisons against the original evaluators on generated images. This is a development test environment, separate from the reported Python 3.12.12 training environment.

Scientific reproduction additionally requires the selected local model weights, original ECG images, and corresponding original raw-logit predictions. Compare each model first, then the ensemble, under the same image preprocessing, numerical precision, sample order, TTA seed and library versions. These private inputs and prediction tables are not included in the public repository. The preceding revision passed paired float32/no-TTA comparison on 10 real ECG images in WSL and paper-TTA execution on one synthetic image in Colab. The updated package `0.2.0.dev1` with Pillow 13.0.0 also passed the WSL/CUDA float32 no-TTA comparison on those 10 real images against the frozen original-evaluator CSVs. Image and selected-weight hashes matched; all three models and the ensemble passed `atol=1e-5`, `rtol=1e-5`, with zero class changes. The original evaluators were not rerun. The preceding English Colab notebook was subsequently reported to run successfully; its saved runtime record was not reviewed for this update. WSL paper-TTA execution remains pending, and historical paper-TTA numerical parity is also unconfirmed. The current test suite includes rejection checks for unsupported Pillow/PyTorch versions and unsafe ViT configuration fields. Run it with Python 3.11 or newer and the current inference requirements; the earlier Python 3.10 test environment cannot run Pillow 13.0.0.

## Input/output protection regression checks

Run `python -m unittest discover -s tests -v` in the inference environment. The additional fixtures exercise format impersonation, single-frame restrictions, output collisions, protected hardlinks, CSV sidecar conflicts, interrupted saves and directory replacement. They do not use patient data or download model weights. If the account cannot create symlinks or hardlinks, the corresponding tests are skipped and must be reported as untested on that platform.

On Windows, use a writable private temporary directory if the system temporary directory is restricted. `ECG_TEST_TMPDIR` configures the output-saving tests; standard `TEMP`/`TMP` configure the other fixtures. Test output should identify any skipped cases. WSL should exercise POSIX directory-relative saving and hardlink checks before claiming Linux validation.

# Implementation and validation status

Updated 2026-10-28 (JST); documentation revision 0.1.21. Inference package 0.3.1; weights 0.2.0.

## v0.3.1 maintenance release and validation

Version 0.3.1 explicitly pins `filelock==4.2.1` in inference requirements and records that version in CLI and Colab output metadata. Model tensors, preprocessing, prediction calculations, TTA and Grad-CAM calculations are unchanged. Historical training requirements remain unchanged.

The 2026-10-27 metadata review detected three active indexed advisories for the previously installed filelock 3.32.3: [GHSA-v4x8-3wr5-3wq8](https://github.com/tox-dev/filelock/security/advisories/GHSA-v4x8-3wr5-3wq8), [GHSA-9vhv-jxww-xfm5](https://github.com/tox-dev/filelock/security/advisories/GHSA-9vhv-jxww-xfm5), and [GHSA-567q-5qvw-pf92](https://github.com/tox-dev/filelock/security/advisories/GHSA-567q-5qvw-pf92). Upstream identifies 4.2.1 as patched. These locking scenarios require local filesystem access; a remote exploit through an ECG image was not demonstrated by this review.

| Check | Environment and conditions | Verified result |
|---|---|---|
| CPU dependency comparison, 2026-10-27 | Windows, Python 3.13.15, torch 2.13.0+cpu / torchvision 0.28.0+cpu; isolated filelock 4.2.1 overlay over the existing ML environment | 119 tests: 116 passed, three hardlink-permission cases skipped. One generated 480 x 480 image with paper TTA and float32 preserved all model/ensemble predictions and both Grad-CAM maps exactly across the filelock substitution. Original environment preserved. |
| WSL dependency comparison, 2026-10-28 | RTX 5090, Python 3.12.12, torch 2.13.0+cu130 / torchvision 0.28.0+cu130; filelock 3.32.3 versus isolated 4.2.1 overlay; one generated image, float32, paper TTA, batch 1, zero workers, seed 1234 | 119 tests: 117 passed, two Windows-only cases skipped, no failures/errors. All prediction values and both finite float32 480 x 480 maps were exactly unchanged. Source, weight/configuration and artifact hashes matched. Only filelock differed among 57 packages; original inventory remained unchanged. |
| Colab candidate execution, 2026-10-28 | A100-SXM4-40GB, Python 3.13.16, separate newly installed inference environment, CUDA float32, one generated 480 x 480 image, paper TTA, batch 1, four workers, seed 1234 | Candidate archive and 112 source hashes matched; filelock 4.2.1 and all required versions were recorded. Official selected-weight/config hashes, synthetic RGB pixels, prediction/run linkage and all nine ZIP members were verified. Softmax and raw-logit aggregation were independently recalculated; both maps and their rendered pixels matched. |
| Local notebook integration | Simulated setup, Colab APIs and GPU availability; tiny generated inputs and ZIP output cases | 25 checks passed. These mocked checks are separate from the live candidate Colab execution above. |

Both GPU checks used transformers 5.10.4, timm 1.0.22, numpy 2.2.6, Pillow 13.0.0 and safetensors 0.7.0. Paper TTA means ViT 24/magnitude 1, ConvNeXtV2 1 unchanged view and MLP-Mixer 32/magnitude 3; each model's mean raw logits are combined with equal weights before softmax. The WSL overlay check is not a fresh full-environment installation. The Colab run recorded certifi 2026.10.23, whereas the earlier WSL environment used 2026.10.19; the full installed environments are not claimed to be identical.

The Colab source archive SHA-256 was `9f87654d0839c9132499c6312b8341dc1d4c45c37f4a3a4ce48a7f5f1daddd9b`. The release finalization updates documentation, notebook Markdown and citation metadata. Executable source, dependencies and public notebook code cells match the tested candidate. The private candidate notebook used verified archive loading in place of the public Git checkout cell; its remaining code cells matched. The final public-tag Colab source-acquisition check remains a separate post-publication step. The candidate run did not include a paired Grad-CAM-off comparison.

The 2026-10-27 PyPI metadata review of the Windows 39-package and prior WSL 57-package sets with filelock substituted found zero active indexed advisories, yanked files or lookup errors. This version-level check was not repeated merely to finalize documentation and does not cover unknown vulnerabilities, individual PyTorch wheel builds, the OS or drivers.

These are software checks on generated images, not clinical validation or proof of historical stochastic-TTA/reader-study heatmap equivalence. The [software DOI](https://doi.org/10.5281/zenodo.23666137) still identifies archived v0.3.0, and the [weight DOI](https://doi.org/10.5281/zenodo.23666262) identifies v0.2.0. The v0.3.1 citation uses its GitHub Release URL until a new archive DOI is assigned. Existing tags and Zenodo records are preserved.

## Published v0.3.0 and final Colab verification

The [v0.3.0 release](https://github.com/ekagawa007/takotsubo_ami_ai_ecg/releases/tag/v0.3.0) identifies commit `341ac955353f535fb74a4b6d8005a8b3b1290ad4`. Anonymous download verification matched all 111 published source files to the reviewed release. The v0.3.0 source tag remains fixed. The October 24 documentation update described the completed checks without changing that release; the v0.3.1 maintenance release is tracked separately above.

On 2026-10-24, the public-tag Colab workflow completed prediction and Grad-CAM using the official v0.2.0 weights. The saved results were independently checked:

| Item | Verified scope |
|---|---|
| Input and runtime | One generated 480 x 480 RGB image; NVIDIA A100-SXM4-40GB, Python 3.13.16, CUDA float32; batch size 1, four workers, seed 1234, strict480 input policy |
| Prediction settings | ViT TTA 24/magnitude 1; ConvNeXtV2 TTA 1 (no TTA); MLP-Mixer TTA 32/magnitude 3; equal-weight raw-logit average followed by softmax |
| Source and weights | The run recorded v0.3.0 and the release commit above; all 20 recorded critical-source hashes, three selected-weight hashes and the ViT configuration hash matched |
| Prediction consistency | The prediction JSON matched its run-record hash. Recomputed per-model softmax, ensemble mean raw logits, ensemble softmax and the decision agreed with the saved values |
| Grad-CAM | ViT and ConvNeXtV2 each produced a finite, nonconstant float32 480 x 480 map; target classes, layers and normalization matched the recorded settings |
| Saved outputs | The ZIP contained the matching two JSON records, five RGB PNGs and two NPY arrays. All seven visualization-file hashes and sizes matched; heatmap and overlay pixels agreed with rendering recalculated from the saved numerical maps |

This public run used torch 2.13.0+cu130, torchvision 0.28.0+cu130, transformers 5.10.4, timm 1.0.22, numpy 2.2.6, Pillow 13.0.0 and safetensors 0.7.0. Relative to the earlier Grad-CAM Colab candidate, Python changed from 3.13.15 to 3.13.16; the recorded installed-package versions were the same. This comparison is not a new dependency vulnerability audit.

The public-tag Colab execution check is complete. It is a one-image software workflow check, without a paired Grad-CAM-off run in Colab. The paired prediction-preservation evidence comes from the CPU and WSL checks below. Historical stochastic TTA equivalence, historical reader-study heatmap equivalence and clinical performance are not established by this synthetic input.

The [published v0.3.0 Colab notebook](https://colab.research.google.com/github/ekagawa007/takotsubo_ami_ai_ecg/blob/v0.3.0/examples/colab_inference.ipynb) selects `GIT_REF = "v0.3.0"`. The maintenance notebook selects its own release version; the checks in this section describe v0.3.0.

## Grad-CAM implementation and validation

Version 0.3.0 adds optional ViT/ConvNeXtV2 Grad-CAM after prediction, with an unchanged prediction engine. It uses the archived layer choices and class-logit gradient reductions, defaults to current inference normalization, and offers an explicitly labelled legacy ViT normalization option. The maps use unaugmented float32 input and do not explain the TTA average or combined ensemble. See [Grad-CAM](gradcam.md) for preprocessing and numerical differences from the archived visualization.

The checks below used candidate package `0.3.0.dev0` on 2026-10-23. The released `0.3.0` source differs from that candidate only in documentation and version metadata; the implementation, dependency pins and notebook code cells are unchanged. These dated candidate results are retained separately from the completed public-tag verification above.

On 2026-10-23, the complete candidate suite ran 119 tests in native Windows/Python 3.13.15: 116 passed and three hardlink tests were skipped because the selected temporary directory did not support them. There were no failures or errors. The new Grad-CAM protected-hardlink test passed separately in a workspace directory with hardlink support. Exact skip names and test/source hashes are retained in private records.

Using the three official selected safetensors on one generated 480 x 480 image, CPU/float32 inference with no TTA and, separately, with paper TTA (ViT 24/magnitude 1, ConvNeXtV2 1, MLP-Mixer 32/magnitude 3) produced exactly the same logits, probabilities and classes with Grad-CAM enabled or disabled. Both visualized models produced finite float32 maps at 480 x 480; saved metadata and PNG/NPY files were checked. The CPU no-TTA overlays were visually inspected.

The candidate subsequently passed the following GPU checks on 2026-10-23:

| Check | Environment and conditions | Result |
|---|---|---|
| Paired Grad-CAM off/on | WSL/RTX 5090, Python 3.12.12, CUDA float32, one generated 480 x 480 image, paper TTA 24/magnitude 1, 1, and 32/magnitude 3; batch 1, zero workers | All three-model and ensemble prediction values were exactly unchanged; two finite float32 480 x 480 maps and their saved PNG/NPY artifacts verified |
| Colab candidate execution | A100-SXM4-40GB, Python 3.13.15, CUDA float32, one generated 480 x 480 image, same paper TTA; batch 1, four workers | Grad-CAM execution and saving completed; candidate archive, 111 source hashes, prediction/run linkage and all seven artifact hashes verified; downloaded ZIP matched the two JSONs and seven artifacts |

Both runs used torch 2.13.0+cu130, torchvision 0.28.0+cu130, transformers 5.10.4, timm 1.0.22, numpy 2.2.6, Pillow 13.0.0 and safetensors 0.7.0. Source and selected-weight identities matched the reviewed implementation and official assets. The Colab run did not include a paired Grad-CAM-off comparison. An additional WSL run with torch 2.9.0/torchvision 0.24.0/transformers 4.57.3 passed the same paired prediction check; that is compatibility evidence for the earlier environment, not the current recommended setup.

These are synthetic software checks. Historical reader-study heatmap equivalence and clinical localization validity remain unestablished. The prior v0.2.1 results below apply to its prediction workflow.

## Inherited v0.2.1 implementation

The v0.2.1 inference implementation protects JSON/CSV/metadata saving, retains uncertain partial weight installations, and permits only supported single-frame raster image decoding in the initial check and workers. The v0.2.1 requirements pinned its tested inference versions separately from the unchanged historical training requirements.

A strict compatibility adapter supports the selected ViT checkpoint with Transformers 5.10.4. It follows the documented legacy-to-modular names: 192 of 200 saved parameter keys are renamed, without changing their tensor objects or values. Missing, extra, ambiguous or shape-mismatched keys remain errors. The adapter also leaves already matching checkpoint keys unchanged. No model is retrained or re-exported for this change.

The English Colab notebook uses a separate inference virtual environment, configurable synthetic/file input, optional Drive mounting, live progress, unique output names and one ZIP containing the matching prediction/run records. Selected model weights, class order, normalization, paper TTA settings and raw-logit aggregation remain unchanged.

## Completed checks for the tested v0.2.1 implementation

| Check | Environment and input | Result |
|---|---|---|
| Full test suite | WSL, Python 3.12.12, torch 2.13.0+cu130, torchvision 0.28.0+cu130 | 85 discovered: 83 passed; two Windows-only alternate-stream/directory-sharing tests skipped |
| Full test suite | Native Windows CPU, Python 3.13.15, torch 2.13.0+cpu, torchvision 0.28.0+cpu | 85 discovered: 83 passed; two hardlink tests skipped because the account could not create hardlinks |
| Ten-image no-TTA comparison | Same archived private ECGs and selected weights; WSL/RTX 5090, CUDA, float32, batch 1, four workers, seed 1234, TF32 disabled | All three models and the ensemble agreed with frozen original-evaluator predictions within `atol=rtol=1e-5`; zero class differences in each 10-image comparison |
| Colab execution | A100-SXM4-40GB, Python 3.13.15, CUDA, float32; one generated 480 x 480 RGB image; paper TTA 24/magnitude 1, 1, and 32/magnitude 3 | Weight preparation, inference, live progress, prediction saving and matching run record completed; record/source/weight identities and raw-logit aggregation verified |
| CPU dependency comparison | Native Windows, Python 3.13.15, float32, batch 1, zero workers; ten private images without TTA and one generated image with paper TTA | Both comparisons passed `atol=rtol=1e-5`; zero class differences |

All current checks use timm 1.0.22, transformers 5.10.4, numpy 2.2.6, Pillow 13.0.0 and safetensors 0.7.0. The tested setup used pip 26.2.2 and setuptools 84.2.1. The original WSL and pre-release Colab dependency-review records contained the same 57 installed package names/versions. The later public-tag and Grad-CAM Colab records use certifi 2026.10.23 rather than the reviewed 2026.10.19; those full environments are not claimed to be identical. The version-level PyPI metadata check found no active indexed advisories, yanked files or lookup errors for the originally reviewed set on 2026-10-23. This limited metadata check is not an OS/driver audit or proof that all vulnerabilities are absent.

The WSL comparison's maximum absolute logit difference was `7.034942627e-7`; its maximum probability difference was `1.352990612e-7`. The independent review recomputed the comparison and the equal raw-logit mean followed by softmax. Input, source and selected-weight hashes matched the recorded artifacts. The comparison reused frozen original-evaluator outputs; it did not rerun those evaluators with the new libraries.

The Colab record identifies the generated input, and its RGB pixel hash matched independent regeneration. The reviewed prediction JSON matched its run-record hash. This is one-image execution evidence and carries no diagnostic-performance claim.

The earlier source-only saving/image-format patch also passed same-runtime CPU comparisons with exactly unchanged predictions. The later dependency change is assessed by the separate tolerance-based CPU and WSL comparisons above; exact cross-version equality is not claimed.

## v0.2.1 publication and remaining scope

The final public notebook passed format/syntax validation and 12 local checks covering source checkout handling, isolated setup/reuse/failure, input selection, progress/error handling and paired-record ZIP saving. These include mocked subprocess/network operations and local Pillow input handling; they are separate from the completed A100 inference run.

The tested pre-release Colab workflow loaded a verified source archive. The final public notebook changes source acquisition to the versioned GitHub checkout and records the resolved commit. The `v0.2.1` release was subsequently published at commit `95bbe9998f7433e3f65eb000e2d70ac2b4e3ff32`; the publication log confirms anonymous source-download verification and local tag fetching. Two final live public-tag Colab runs were subsequently verified: both recorded this commit, package 0.2.1, A100/CUDA float32 and paper TTA on one generated image. All 18 recorded source hashes and selected-weight/config hashes matched; each prediction hash matched its run record, and each ZIP contained the matching two JSONs. Independent recalculation confirmed per-model softmax, the mean raw logits and the ensemble decision. This completes the v0.2.1 public-tag execution check. The separate v0.3.0 public-tag check is recorded above.

The v0.2.0 public weight archive and all 11 bundled file hashes were verified on 2026-10-22. The v0.2.1 code continues to use those unchanged assets, so a new weight archive is not required.

Historical paper-TTA numerical equivalence, all-dataset evaluation reproduction, a complete training rerun and human-study runtime validation remain incomplete. The results above are implementation and execution checks on their stated inputs, not clinical validation. Original private ECGs and prediction tables are not published.

# Earlier implementation and validation records

Recorded 2026-10-22 (JST); documentation revision 0.1.12. The dated records below preserve their original evidence and then-pending items. Current results and remaining scope are reported above.

## Public source and inference implementation

Original training, evaluation, human-study, and scheduling sources retain their filenames and notebook formats, with personal paths/examples generalized. The MIT License names Copyright (c) 2026 Eisuke Kagawa. Published `v0.1.1` contains the historical requirements record; that release predates the new inference implementation. Patient data, prediction tables, original training checkpoints, and private preparation records are excluded from the public tree.

The selected inference states are ViT `checkpoint-6900`, ConvNeXtV2 `ckpt_0011.pth` ordinary `model` state, and MLP-Mixer `ckpt_0011.pth` ordinary `model` state. All 730 exported inference tensors matched their original names, shapes, dtypes and values. The training-only ViT `focal_loss.alpha` buffer was omitted explicitly. See [selected checkpoints](checkpoint_selection.md).

## Weight distribution 0.2.0

Version `0.2.0` includes a separate `takotsubo_ami_ai_ecg_weights_v0.2.0.zip` Release asset and `SHA256SUMS.txt`, a weight downloader, a Colab Release-download default with a local alternative, and a [model card](../MODEL_CARD.md). The three selected safetensors files total 958,657,088 bytes and match their expected asset-manifest hashes. Original training checkpoints and patient data remain excluded.

Package `0.2.0` identifies this distribution. Confirmed [weight terms](../WEIGHTS_LICENSE.md) and [upstream notices](../THIRD_PARTY_NOTICES.md) are separate from the code's MIT License: ViT uses CC BY-NC 4.0, and author-owned ConvNeXtV2 and MLP-Mixer contributions use MIT terms. The complete bundle includes the ViT noncommercial condition. The local archive and inference checks below have passed; live public Release-download and actual Colab-download execution remain unverified. Earlier inference checks retain their original version and scope.

On 2026-10-22, package `0.2.0` passed all 50 tests (the prior 38 tests plus 12 downloader tests), and a separate notebook harness passed five simulated download cases. The 958,701,120-byte release ZIP was installed with the new helper into a fresh directory, with all 11 expected file SHA-256 checks passing. This exercised the local release archive; live public URL download was not part of that check.

The three installed models then completed inference on one synthetic 480 x 480 RGB image using the paper TTA settings, CPU/float32, batch size 1, zero workers and seed 1234. Prediction values exactly matched the prior CPU baseline. Live progress arrived during execution, reached 57/57 image evaluations, and reported completion after output saving. The runtime was native Windows with Python 3.13.15, torch 2.11.0+cpu, torchvision 0.26.0+cpu, Pillow 13.0.0, numpy 2.2.6, timm 1.0.22, transformers 4.57.3 and safetensors 0.7.0.

## Completed progress revision 0.2.0.dev2

Package `0.2.0.dev2` adds English live model/TTA/batch progress to the CLI and Colab notebook. The CLI supports `--progress auto`, `bar`, `json` and `off`; progress goes to stderr and does not become part of prediction files. Colab consumes the progress events as the subprocess runs and updates its display. The progress count reflects completed image evaluations across TTA views, with elapsed time displayed separately. No new dependency or change to the inference settings or random-number sequence is introduced.

On 2026-10-22, package `0.2.0.dev2` passed all 38 tests: 17 original inference tests, 14 security tests and seven progress tests. The isolated native Windows environment used Python 3.13.15, torch 2.11.0+cpu, torchvision 0.26.0+cpu, Pillow 13.0.0, numpy 2.2.6, timm 1.0.22, transformers 4.57.3 and safetensors 0.7.0.

The three actual selected models also completed inference on one synthetic 480 x 480 RGB image using the paper TTA settings, CPU/float32, batch size 1, zero workers and seed 1234. Progress events arrived while the subprocess was running, the completed count reached 57/57, and the completion event followed creation of the output file. The saved prediction values exactly matched the preceding `0.2.0.dev1` CPU baseline; inference settings and checkpoint identities were unchanged. This comparison covers that image and execution configuration.

A separate notebook harness passed seven simulated subprocess cases. These checks exercise notebook handling without a live Colab runtime. The English Colab notebook with progress was subsequently reported to run successfully; its saved runtime record was not reviewed for this update. WSL/GPU execution of `0.2.0.dev2` remains unverified. The completed CPU and WSL results below remain records for package `0.2.0.dev1`.

## Completed validation of the preceding inference revision

The following results were obtained on 2026-10-22 at code commit `ff1b9bef9596a33cacf9dd67d42cd78d390a6211`, before the Pillow update and loader restrictions in this revision. Private input files and prediction rows remain local.

| Check | Conditions | Result |
|---|---|---|
| Paired original evaluator versus new CLI | Same 10 real ECG images, RGB 480 x 480, identical selected weights, CUDA, float32, no TTA, batch size 1, four workers | All three models and the raw-logit ensemble agreed within `abs(new-old) <= 1e-5 + 1e-5 * abs(old)`; no class differences in any of the four comparisons |
| Colab inference | One synthetic image, NVIDIA A100-SXM4-40GB, CUDA, float32, ViT TTA 24/magnitude 1, ConvNeXtV2 TTA 1, MLP-Mixer TTA 32/magnitude 3 | Inference, JSON saving, and equal raw-logit ensemble followed by softmax completed successfully |

The largest absolute logit difference across the paired comparisons was `2.356872555e-7`; the largest probability difference was `7.453082962e-8`. All 60 individual-model logits were identical after parsing the two CSV formats back to float32. This establishes numerical agreement for the tested images and no-TTA conditions; the comparison is not an estimate of clinical performance.

The WSL comparison used Python 3.12.12, torch 2.9.0+cu130, and torchvision 0.24.0+cu130. Colab used Python 3.13.15, torch 2.11.0+cu130, and torchvision 0.26.0+cu130. Both used timm 1.0.22, transformers 4.57.3, Pillow 12.0.0, numpy 2.2.6, and safetensors 0.7.0. See [execution environments](execution_environment.md).

Earlier development checks at that revision also passed 17 regression tests, exact preprocessing comparisons with the original evaluator transforms, and selected-model inference on a generated image using native Windows/Python 3.10/CPU PyTorch 2.9.0. Those results describe the earlier environment; they do not validate the new Pillow 13 environment.

## Completed CPU validation of security revision 0.2.0.dev1

On 2026-10-22, the updated inference package `0.2.0.dev1` passed all 31 tests (17 existing inference tests and 14 security tests) in a new isolated native Windows environment. It used Python 3.13.15, torch 2.11.0+cpu, torchvision 0.26.0+cpu, Pillow 13.0.0, numpy 2.2.6, timm 1.0.22, transformers 4.57.3 and safetensors 0.7.0.

The actual three selected safetensors models also completed inference on one deterministic synthetic 480 x 480 RGB image using float32, CPU, batch size 1 and zero workers. The settings were ViT TTA 24/magnitude 1, ConvNeXtV2 TTA 1, MLP-Mixer TTA 32/magnitude 3, seed 1234 and equal ensemble weights. The prediction JSON was saved, and the mean of the raw logits followed by softmax was verified independently. This checks model loading, execution and aggregation in the updated CPU environment; it does not validate ECG classification performance or updated GPU runtimes.

A separate preprocessing check compared the preceding Pillow 12.0.0/torch 2.9.0/torchvision 0.24.0 CPU environment with the updated Pillow 13.0.0/torch 2.11.0/torchvision 0.26.0 CPU environment. For one generated 480 x 480 RGB image and fixed per-view seeds, all 57 transformed-tensor SHA-256 hashes matched exactly (24 ViT, 1 ConvNeXtV2 and 32 MLP-Mixer views). This checks those direct transforms; it does not establish DataLoader-worker or full-dataset TTA equivalence.

## Completed WSL/CUDA no-TTA validation of security revision 0.2.0.dev1

On 2026-10-22, the updated CLI in package `0.2.0.dev1` was run on the same 10 archived real ECG images with Python 3.12.12, torch 2.9.0+cu130, torchvision 0.24.0+cu130, Pillow 13.0.0, numpy 2.2.6, timm 1.0.22, transformers 4.57.3 and safetensors 0.7.0. The image and selected safetensors weight hashes matched the earlier comparison. Settings were CUDA, float32, TTA 1 for every model, batch size 1, four workers, seed 1234 and TF32 disabled.

The comparison used the frozen original-evaluator CSVs from the preceding paired run; the original evaluators were not rerun in the updated environment. All three models and the ensemble passed `abs(new-old) <= 1e-5 + 1e-5 * abs(old)`, with zero class changes in each 10-image comparison. The maximum absolute logit difference was `2.3568725548983593e-7`, and the maximum probability difference was `7.453082961905722e-8`. This confirms the updated inference path against those saved no-TTA results for the tested images. The updated CLI prediction CSV was also byte-for-byte identical to the saved new-CLI CSV from the preceding paired run: all 160 individual-model/ensemble logit and probability values had zero difference, with unchanged classes. This exact match applies to these 10 images under no-TTA conditions; the small differences reported against the separately formatted original-evaluator CSVs remain as stated above.

## Security changes and remaining target-runtime confirmation

This revision requires Pillow 13.0.0 and Python 3.11 or newer for inference, rejects unsupported ViT configuration fields, and requires a stable PyTorch version of at least 2.10.0 for exporting trusted `.pth` checkpoints. Existing verified safetensors assets remain usable without re-exporting them. New inference does not use the checkpoint exporter's pickle-loading path.

The Colab notebook now makes dependency-change restarts explicit, creates separate results for repeated runs, reports captured errors, and refuses to read a prior result after a failed subprocess. Distributed notebooks contain no saved runtime outputs.

The earlier paired WSL and Colab checks remain provenance for the preceding commit. Package `0.2.0.dev1` passed the CPU checks and WSL/CUDA no-TTA comparison described above. The preceding English Colab notebook was subsequently reported to run successfully; its saved runtime record was not reviewed for this update. WSL execution using the paper TTA settings remains unverified, and the completed no-TTA comparison does not establish historical paper-TTA numerical parity.

## Remaining reproduction work

- Numerical parity with historical predictions generated using the paper TTA settings remains unconfirmed. Match image preparation, dataset order, augmentation randomness, precision and package versions when making that comparison.
- A complete training rerun, all-dataset evaluation reproduction and human-study runtime validation have not been completed during this publication update.
- Complete the links from selected checkpoints to training runs, initialization chains, validation splits and loss records using the actual private records.
- Record the final release commit, assets and checksums together, then verify the live public download and Colab workflow. Existing release tags are retained.

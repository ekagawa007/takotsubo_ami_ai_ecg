# Inference validation

Updated 2026-10-22 (JST); documentation revision 0.1.9.

The new inference tests check behaviors such as input validation, model-state loading, class order, and raw-logit aggregation. See [implementation status](../docs/implementation_status.md) for the distinction between code checks and full prediction reproduction.

After installing the inference dependencies, run from the repository root:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

On 2026-10-22, the current package `0.2.0.dev1` passed all 31 tests (17 existing inference tests and 14 security tests) in an isolated native Windows environment using Python 3.13.15, CPU PyTorch 2.11.0, torchvision 0.26.0 and Pillow 13.0.0. The three selected models also completed float32 inference on one synthetic 480 x 480 RGB image using the paper TTA settings, batch size 1 and zero workers. Result saving and the mean-raw-logit/softmax ensemble calculation were verified. This development check does not replace WSL/CUDA or Colab execution.

Before the current security revision, all 17 inference tests passed on 2026-10-22 at commit `ff1b9bef9596a33cacf9dd67d42cd78d390a6211` on Windows with Python 3.10, CPU PyTorch 2.9.0 and torchvision 0.24.0. The checks include exact transform comparisons against the original evaluators on generated images. This is a development test environment, separate from the reported Python 3.12.12 training environment.

Scientific reproduction additionally requires the selected local model weights, original ECG images, and corresponding original raw-logit predictions. Compare each model first, then the ensemble, under the same image preprocessing, numerical precision, sample order, TTA seed and library versions. These private inputs and prediction tables are not included in the public repository. The preceding revision passed paired float32/no-TTA comparison on 10 real ECG images in WSL and paper-TTA execution on one synthetic image in Colab. The updated package `0.2.0.dev1` with Pillow 13.0.0 also passed the WSL/CUDA float32 no-TTA comparison on those 10 real images against the frozen original-evaluator CSVs. Image and selected-weight hashes matched; all three models and the ensemble passed `atol=1e-5`, `rtol=1e-5`, with zero class changes. The original evaluators were not rerun. Updated Colab and WSL paper-TTA execution remain pending; historical paper-TTA numerical parity is also unconfirmed. The current test suite includes rejection checks for unsupported Pillow/PyTorch versions and unsafe ViT configuration fields. Run it with Python 3.11 or newer and the current inference requirements; the earlier Python 3.10 test environment cannot run Pillow 13.0.0.

# Inference validation

The new inference tests check behaviors such as input validation, model-state loading, class order, and raw-logit aggregation. See [implementation status](../docs/implementation_status.md) for the distinction between code checks and full prediction reproduction.

After installing the inference dependencies, run from the repository root:

```bash
python -m unittest discover -s tests -p "test_inference.py" -v
```

On 2026-10-22, all 17 tests passed on Windows with Python 3.10, CPU PyTorch 2.9.0 and torchvision 0.24.0. The checks include exact transform comparisons against the original evaluators on generated images. This is a development test environment, separate from the reported Python 3.12.12 training environment.

Scientific reproduction additionally requires the selected local model weights, original ECG images, and corresponding original raw-logit predictions. Compare each model first, then the ensemble, under the same image preprocessing, numerical precision, sample order, TTA seed and library versions. These private inputs and prediction tables are not included in the public repository. WSL/Linux and fresh Google Colab validation remain required.

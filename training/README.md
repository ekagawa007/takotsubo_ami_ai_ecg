# Original model-development code

The supplied source files retain their original filenames and formats. Public copies generalize paths and examples; original/current hashes and edit reasons are recorded in `docs/source_file_mapping.csv` and `docs/publication_edits.csv`. `pretrain/`, `ft_3_class/`, and `ft_2_class/` each contain `vit/`, `convnextv2/`, and `mlpmixer/`.

ViT fine-tuning notebooks contain the training implementation. ConvNeXtV2 and MLP-Mixer fine-tuning retain both the Python implementations and the notebooks that invoke them. Launch arguments in those notebooks are part of the development record.

Read [development_code.md](../docs/development_code.md) and [security_review.md](../docs/security_review.md) before running or publishing the historical files. Configure the generic data/checkpoint paths and repository import path before execution. The recovered scheduling helper is included at repository root. Final-run selection, third-party dependencies, and original environment versions still need confirmation.

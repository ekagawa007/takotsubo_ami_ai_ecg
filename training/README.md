# Original model-development code

The supplied source files retain their original filenames and formats. Public copies generalize personal paths and examples. `pretrain/`, `ft_3_class/`, and `ft_2_class/` each contain `vit/`, `convnextv2/`, and `mlpmixer/`.

ViT fine-tuning notebooks contain the training implementation. ConvNeXtV2 and MLP-Mixer fine-tuning retain both the Python implementations and the notebooks that invoke them. Launch arguments in those notebooks are part of the development record.

Read [development code](../docs/development_code.md) and [execution environments](../docs/execution_environment.md) before running the historical files. Configure the generic data/checkpoint paths and repository import path before execution. The scheduling helper is included at repository root. The reported training environment is recorded in the root `requirements.txt`; detailed run/initialization correspondence and complete reproduction remain to be verified.

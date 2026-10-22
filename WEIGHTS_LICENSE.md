# Model-weight distribution terms

Date: 2026-10-22. Applies to the inference assets accompanying release `v0.2.0`.

The repository's source code remains under the [MIT License](LICENSE-MIT.txt). The
model assets have the following separate terms. Copyright in the author's
contributions, to the extent applicable: Copyright (c) 2026 Eisuke Kagawa.

| Asset | Distribution terms |
|---|---|
| ViT inference weights and accompanying ViT configuration files | The author's contributions are licensed under [CC BY-NC 4.0](licenses/CC-BY-NC-4.0.txt). Retain the original MAE noncommercial notice and the Hugging Face distribution's [Apache-2.0](licenses/Apache-2.0.txt) notice described below. This release does not grant commercially unrestricted use of the ViT assets. |
| ConvNeXtV2 inference weights | The author's contributions are licensed under the [MIT License](LICENSE-MIT.txt). |
| MLP-Mixer inference weights | The author's contributions are licensed under the [MIT License](LICENSE-MIT.txt). |

These grants cover only rights the author holds. They do not replace upstream
licenses or grant rights to third-party material. The full three-model bundle
includes the ViT assets, so its use and redistribution must respect their
noncommercial terms. The separate MIT terms for ConvNeXtV2, MLP-Mixer and the
source code remain unchanged.

## ViT upstream notices

The retained training workflow initializes MAE pretraining from
[`facebook/vit-mae-base`](https://huggingface.co/facebook/vit-mae-base), then
loads locally trained checkpoints for three-class and two-class fine-tuning.
The Hugging Face model card declares Apache-2.0. The original
[`facebookresearch/mae`](https://github.com/facebookresearch/mae) release,
which supplies the original pretrained checkpoints, declares CC BY-NC 4.0.
The two published notices differ. This distribution retains both notices and
uses noncommercial terms for its ViT derivative; it does not represent the
Hugging Face label as overriding the original MAE notice.

The saved final checkpoint configuration does not identify an immutable
upstream model revision. The provenance above describes the retained training
workflow, rather than a complete reconstruction of every executed training
step. The reviewed upstream references and changes are recorded in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

When sharing the ViT assets or adaptations, retain attribution, the applicable
license texts and notices, and indicate your changes. A commercial permission
grant for third-party material cannot be supplied by this repository's author.

## Redistribution and scope

Redistribute each asset in accordance with its applicable license, including
required attribution and license notices. The supplied notice files are
`WEIGHTS_LICENSE.md`, `THIRD_PARTY_NOTICES.md`, `LICENSE-MIT.txt` and the relevant
files in `licenses/`. Asset checksums identify the unchanged release files.

These terms apply to the model assets, not to patient images, training datasets,
private evaluation records or third-party libraries. Those materials are not
included in the weight bundle. The applicable licenses contain their warranty
and liability provisions.

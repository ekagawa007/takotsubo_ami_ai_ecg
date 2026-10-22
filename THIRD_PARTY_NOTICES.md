# Model provenance and third-party notices

Reviewed 2026-10-22 for the `v0.2.0` inference asset release. Model distribution
terms are stated in [WEIGHTS_LICENSE.md](WEIGHTS_LICENSE.md).

## ViT initialization and MAE

The retained ViT pretraining implementation calls
`ViTMAEForPreTraining.from_pretrained("facebook/vit-mae-base")` when starting a
new run. The three-class fine-tuning notebook loads the local MAE checkpoint;
the two-class notebook loads the local three-class checkpoint. The selected
inference checkpoint is the resulting two-class ViT model.

These source-level facts support upstream attribution. The final saved
configuration and trainer state do not record an immutable upstream model
revision, so they do not independently prove the complete executed lineage.
The revisions below identify the notices reviewed for this release, not a
claimed historical download revision.

Original model and method: **Masked Autoencoders Are Scalable Vision Learners**,
Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár and Ross Girshick
(2021), [paper](https://arxiv.org/abs/2111.06377).

- Original release: <https://github.com/facebookresearch/mae>.
- Original license: [CC BY-NC 4.0](https://github.com/facebookresearch/mae/blob/main/LICENSE),
  preserved in [licenses/CC-BY-NC-4.0.txt](licenses/CC-BY-NC-4.0.txt).
- Original implementation's copyright notice: Copyright (c) Meta Platforms,
  Inc. and affiliates. All rights reserved.
- Transformers model distribution: <https://huggingface.co/facebook/vit-mae-base>.
- Reviewed model-card revision:
  [`25b184bea5538bf5c4c852c79d221195fdd2778d`](https://huggingface.co/facebook/vit-mae-base/blob/25b184bea5538bf5c4c852c79d221195fdd2778d/README.md).
  Its metadata declares Apache-2.0, and the card identifies its writer as the
  Hugging Face team. The Apache text is preserved in
  [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt).

The original MAE and Hugging Face notices differ. Both are retained here;
neither is presented as permission to remove the other's conditions. The ViT
assets are distributed under the noncommercial terms described in
`WEIGHTS_LICENSE.md`.

Changes represented by this research workflow include further ECG pretraining,
three-class fine-tuning and two-class fine-tuning with a classification head.
The inference export selects the final two-class prediction tensors, removes
training-only material, and minimizes configuration metadata. These are
modified downstream assets, not the original upstream model files. Upstream
authors do not endorse this research model or its predictions.

## ViT configuration and image processor

The fine-tuning notebooks also obtain configuration and image-processor
settings from
[`google/vit-base-patch16-384`](https://huggingface.co/google/vit-base-patch16-384),
whose model-card metadata declares Apache-2.0 (reviewed revision
`2960116e809e2fca84146dbb240289aee7db4827`). The corresponding model weights are
loaded from the local research checkpoints in the retained fine-tuning
workflow; this configuration reference is not a claim that Google model
weights initialized those stages. The exported configuration adapts the image
dimensions and class mapping for this study. The Apache license text is
included above.

## ConvNeXtV2 and MLP-Mixer

The retained self-supervised pretraining implementations construct their timm
models with `pretrained=False`. Subsequent fine-tuning loads the study's local
checkpoints. No separately downloaded pretrained ConvNeXtV2 or MLP-Mixer
weight asset is identified in that retained initialization workflow. This
statement describes the available code, not an independent reconstruction of
all historical runs.

The architectures and runtime implementations are provided through the
separately installed PyTorch, timm and Transformers packages. Those packages
retain their respective licenses; their source distributions are not bundled
with these inference weights. This notice does not replace package licenses.

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
単一画像の推測（MLP-Mixer）
- timm の mixer_b16_224 を img_size=480 で作成した 2-class (tako / AMI) を想定
- .pth（state_dict or {"model": ...}）をロード
- 画像1枚を指定して推論
- オプションで TTA（Flip除外 RandAugment）に対応
- オプションで可視化（入力画像 + 確率バー）を保存

★ 前処理は ViT 版と同じ方針：
   1) 元画像を読み込み
   2) 正方形でなければ pad_to_square
   3) 480x480 に resize
   4) transform は「Resizeを含めず」ToTensor/Normalize(+TTA)のみ

このスクリプトは、以下の2つと整合する前処理・ロード方針を意図：
- tako_ami_ft2_mlpmixer_20251102_3.py（学習）
- evaluate_dataset_dir_mlpmixer.py（評価）

使用法:
python predict_single_image_mlpmixer.py
    /path/to/data/example_ecg.png
    --model-path /path/to/checkpoints/ft_2_class/mlpmixer/tako_ami_ft2_mlpmixer_gd_20251110_3_dir/ckpt_0011.pth
    --device cuda
    --tta 16
    --tta-magnitude 3
    --save-dir /path/to/outputs

TTAはdefautlで16回、magnitudeは3に設定している。

version 1.0.0 20251206 first release
"""

import os
import sys
import argparse
import datetime
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

import timm
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import InterpolationMode, RandAugment


# --------------------------------------------------
# 定数（学習・評価と合わせる）
# --------------------------------------------------
LABEL_NAMES = ["tako", "AMI"]
LABEL2ID = {n: i for i, n in enumerate(LABEL_NAMES)}
ID2LABEL = {i: n for n, i in LABEL2ID.items()}

IMAGE_SIZE = 480  # ★ ViT版と同じ固定サイズ運用


# --------------------------------------------------
# RandAugment（Flip を除外） for TTA
# --------------------------------------------------
class RandAugmentNoFlip(RandAugment):
    def __init__(self, num_ops=2, magnitude=9, num_magnitude_bins=10,
                 interpolation=None, fill=0):
        super().__init__(num_ops, magnitude, num_magnitude_bins,
                         interpolation, fill)
        modules = list(self._modules.items())
        filtered = OrderedDict(
            (name, module)
            for name, module in modules
            if "Flip" not in module.__class__.__name__
        )
        self._modules = filtered


# --------------------------------------------------
# ViT版と同じ pad_to_square + resize(480)
# --------------------------------------------------
def pad_to_square(pil_img: Image.Image, fill=(0, 0, 0)) -> Image.Image:
    """長い辺の長さで正方形キャンバスを作り、中央に元画像を貼り付け"""
    w, h = pil_img.size
    side = max(w, h)
    new_img = Image.new(pil_img.mode, (side, side), color=fill)
    new_img.paste(pil_img, ((side - w) // 2, (side - h) // 2))
    return new_img


def preprocess_image(image_path: str):
    """
    画像を前処理（ViT版と同じ）
    - 正方形でない場合は padding
    - 480x480 にリサイズ
    """
    original_image = Image.open(image_path).convert("RGB")
    w, h = original_image.size
    print(f"元画像サイズ: {w} x {h}")

    if w != h:
        print("画像が正方形でないため、paddingして正方形にします")
        padded_image = pad_to_square(original_image)
        print(f"padding後サイズ: {padded_image.size[0]} x {padded_image.size[1]}")
    else:
        padded_image = original_image

    resized_image = padded_image.resize(
        (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
    )
    print(f"リサイズ後サイズ: {resized_image.size[0]} x {resized_image.size[1]}")

    return original_image, padded_image, resized_image


# --------------------------------------------------
# transform（★ Resizeを含めない）
# --------------------------------------------------
def build_identity_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

def build_tta_transform(magnitude: int):
    magnitude = int(max(0, min(10, magnitude)))
    return transforms.Compose([
        RandAugmentNoFlip(
            num_ops=2,
            magnitude=magnitude,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])


# --------------------------------------------------
# softmax（numpy）
# --------------------------------------------------
def softmax_np(logits, axis=1):
    z = logits - logits.max(axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=axis, keepdims=True)


# --------------------------------------------------
# モデルのロード（evaluate_dataset_dir_mlpmixer.py と同方針想定）
# --------------------------------------------------
def _strip_prefixes(k: str) -> str:
    if k.startswith("_orig_mod."):
        k = k[len("_orig_mod."):]
    if k.startswith("module."):
        k = k[len("module."):]
    return k


def resolve_checkpoint_path(model_path_or_file: str) -> str:
    """ディレクトリ指定なら .pth を探して最後の1つを使う（辞書順）"""
    path = model_path_or_file
    if os.path.isdir(path):
        cands = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.endswith(".pth")
        ]
        if not cands:
            raise FileNotFoundError(f"No .pth found under: {path}")
        path = sorted(cands)[-1]  # ★辞書順で最後
        print(f"[INFO] Using checkpoint file: {path}")
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def load_mixer(model_path_or_file: str, num_classes: int = 2,
               img_size: int = IMAGE_SIZE, device: str = "cpu"):
    """
    timm MLP-Mixer を作成し、.pth をロード。
    .pth が state_dict 直、または {"model": state_dict} の両対応。
    """
    model = timm.create_model(
        "mixer_b16_224",
        pretrained=False,
        img_size=img_size,   # ★学習と同じ 480 前提
        in_chans=3,
        num_classes=num_classes,
    )

    ckpt_path = resolve_checkpoint_path(model_path_or_file)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    new_state = {_strip_prefixes(k): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print("[WARN] Missing keys (likely ok):", list(missing)[:8], "...")
    if unexpected:
        print("[WARN] Unexpected keys:", list(unexpected)[:8], "...")

    model.to(device).eval()
    return model, ckpt_path


# --------------------------------------------------
# 単一画像推論
# --------------------------------------------------
def infer_once(model, img_pil: Image.Image, tfm, device: str):
    """
    img_pil は 480x480 に前処理済みの想定
    """
    x = tfm(img_pil).unsqueeze(0).to(device)  # (1,3,480,480)

    with torch.no_grad():
        out = model(x)
        logits = out.detach().cpu().numpy()

    probs = softmax_np(logits, axis=1)[0]
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])
    return probs, pred_idx, conf, logits[0]


def predict_single_image_base(image_path: str, model, device: str):
    original, padded, resized = preprocess_image(image_path)

    tfm = build_identity_transform()
    probs, pred_idx, conf, _ = infer_once(model, resized, tfm, device)

    return {
        "original_image": original,
        "padded_image": padded,
        "resized_image": resized,
        "predicted_class": pred_idx,
        "predicted_label": ID2LABEL[pred_idx],
        "confidence": conf,
        "probabilities": probs
    }


def predict_single_image_tta(image_path: str, model,
                             device: str, tta: int = 8, tta_magnitude: int = 5):
    """
    TTA で logits 平均。
    1回目は identity、2回目以降は RandAugmentNoFlip。
    ★ ViT版と同様に「pad→resize 済み 480」へ RA を適用。
    """
    tta = max(1, int(tta))
    tta_magnitude = int(max(0, min(10, tta_magnitude)))

    original, padded, resized = preprocess_image(image_path)

    tfm_id = build_identity_transform()
    tfm_tta = build_tta_transform(tta_magnitude)

    # 1回目: identity
    _, _, _, log0 = infer_once(model, resized, tfm_id, device)
    logits_sum = log0.copy()

    # 2回目以降: RandAugment
    for _ in range(tta - 1):
        _, _, _, logi = infer_once(model, resized, tfm_tta, device)
        logits_sum += logi

    logits_mean = logits_sum / float(tta)
    probs = softmax_np(logits_mean[None, :], axis=1)[0]
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])

    return {
        "original_image": original,
        "padded_image": padded,
        "resized_image": resized,
        "predicted_class": pred_idx,
        "predicted_label": ID2LABEL[pred_idx],
        "confidence": conf,
        "probabilities": probs,
        "tta": tta,
        "tta_magnitude": tta_magnitude
    }


# --------------------------------------------------
# 可視化（入力画像 + 確率バー）
# --------------------------------------------------
def visualize_prediction(img_pil: Image.Image,
                         base_result: dict,
                         tta_result: dict | None = None,
                         save_path: str | None = None):
    labels = LABEL_NAMES

    if tta_result is None:
        probs = base_result["probabilities"]
        title = (
            f"MLP-Mixer Prediction "
            f"(base: {base_result['predicted_label']} {base_result['confidence']*100:.1f}%)"
        )
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        ax_img, ax_bar = axes

        ax_img.imshow(img_pil)
        ax_img.set_title("Input (resized 480x480)")
        ax_img.axis("off")

        ax_bar.bar(labels, probs)
        ax_bar.set_ylim(0, 1)
        ax_bar.set_title(title)
        ax_bar.set_ylabel("Probability")

    else:
        probs_b = base_result["probabilities"]
        probs_t = tta_result["probabilities"]

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))
        ax_img, ax_bar_b, ax_bar_t = axes

        ax_img.imshow(img_pil)
        ax_img.set_title("Input (resized 480x480)")
        ax_img.axis("off")

        ax_bar_b.bar(labels, probs_b)
        ax_bar_b.set_ylim(0, 1)
        ax_bar_b.set_title(
            f"Base: {base_result['predicted_label']} {base_result['confidence']*100:.1f}%"
        )
        ax_bar_b.set_ylabel("Probability")

        ax_bar_t.bar(labels, probs_t)
        ax_bar_t.set_ylim(0, 1)
        ax_bar_t.set_title(
            f"TTA({tta_result['tta']}, mag={tta_result['tta_magnitude']}): "
            f"{tta_result['predicted_label']} {tta_result['confidence']*100:.1f}%"
        )

    plt.tight_layout()
    if save_path:
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"[INFO] Saved visualization: {save_path}")
    plt.show()


# --------------------------------------------------
# メイン
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Single-image inference for MLP-Mixer (tako/AMI) with ViT-style preprocessing"
    )
    parser.add_argument("image_path", type=str, help="入力画像のパス")
    parser.add_argument("--model-path", type=str, required=True,
                        help=".pth 直指定 or .pth を含むディレクトリ")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"],
                        help="使用デバイス")
    parser.add_argument("--tta", type=int, default=16,
                        help="TTA回数（1ならTTAなし）")
    parser.add_argument("--tta-magnitude", type=int, default=3,
                        help="RandAugment magnitude (0-10)")
    parser.add_argument("--save-dir", type=str, default=None,
                        help="指定すると可視化PNGを保存")

    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        print(f"[ERROR] 画像が見つかりません: {args.image_path}")
        sys.exit(1)
    if not os.path.exists(args.model_path):
        print(f"[ERROR] モデルパスが見つかりません: {args.model_path}")
        sys.exit(1)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDAが利用できません。CPUを使用します。")
        args.device = "cpu"

    args.tta = max(1, int(args.tta))
    args.tta_magnitude = int(max(0, min(10, args.tta_magnitude)))

    print(f"[INFO] device={args.device}, img_size={IMAGE_SIZE}")
    if args.tta <= 1:
        print("[INFO] TTA disabled.")
    else:
        print(f"[INFO] TTA enabled: tta={args.tta}, magnitude={args.tta_magnitude}")

    try:
        model, used_ckpt = load_mixer(
            args.model_path,
            num_classes=len(LABEL_NAMES),
            img_size=IMAGE_SIZE,
            device=args.device
        )
        print(f"[INFO] Loaded checkpoint: {used_ckpt}")

        # base
        base = predict_single_image_base(args.image_path, model, args.device)

        print("\n=== Base prediction ===")
        print(f"image: {args.image_path}")
        print(f"pred:  {base['predicted_label']} ({base['predicted_class']})")
        print(f"conf:  {base['confidence']:.4f}")
        print(f"tako:  {base['probabilities'][0]:.4f}")
        print(f"AMI:   {base['probabilities'][1]:.4f}")

        tta_res = None
        if args.tta > 1:
            tta_res = predict_single_image_tta(
                args.image_path, model, args.device,
                tta=args.tta, tta_magnitude=args.tta_magnitude
            )

            print("\n=== TTA prediction (logits-mean) ===")
            print(f"pred:  {tta_res['predicted_label']} ({tta_res['predicted_class']})")
            print(f"conf:  {tta_res['confidence']:.4f}")
            print(f"tako:  {tta_res['probabilities'][0]:.4f}")
            print(f"AMI:   {tta_res['probabilities'][1]:.4f}")

        # visualize (resized 480 を表示)
        if args.save_dir:
            base_name = os.path.splitext(os.path.basename(args.image_path))[0]
            out_png = os.path.join(args.save_dir, f"{base_name}_mlpmixer_pred.png")
        else:
            out_png = None

        visualize_prediction(
            base["resized_image"],
            base,
            tta_result=tta_res,
            save_path=out_png
        )

    except Exception as e:
        print(f"[ERROR] 推論中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
    print("Done.", datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9), "JST")))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConvNeXtV2 (timm) 単一画像 推論 + Grad-CAM 可視化
- 2クラス（tako / AMI）
- 非正方形はパディングして 480x480 にリサイズ
- 各クラス確率、Grad-CAM ヒートマップ＆重ね合わせを表示/保存

version 1.0.0 (2025-10-09)
"""

import os
import math
import argparse
import datetime
from collections import OrderedDict

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import timm
import matplotlib.pyplot as plt
from torchvision import transforms
from scipy.special import softmax

# ──────────────────────────────────────────
# 定数
# ──────────────────────────────────────────
IMAGE_SIZE = 480
LABEL_NAMES = ["tako", "AMI"]
ID2LABEL = {i: n for i, n in enumerate(LABEL_NAMES)}
LABEL2ID = {n: i for i, n in ID2LABEL.items()}

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


# ──────────────────────────────────────────
# 前処理ユーティリティ
# ──────────────────────────────────────────
def pad_to_square(pil_img, fill=(0, 0, 0)):
    w, h = pil_img.size
    side = max(w, h)
    new_img = Image.new(pil_img.mode, (side, side), color=fill)
    new_img.paste(pil_img, ((side - w)//2, (side - h)//2))
    return new_img

def preprocess_image(image_path):
    """正方形パディング→480にリサイズ→テンソル化＆正規化"""
    original = Image.open(image_path).convert("RGB")
    padded = pad_to_square(original) if original.size[0] != original.size[1] else original
    resized = padded.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    tensor = tfm(resized).unsqueeze(0)  # (1,C,H,W)
    return tensor, original, padded, resized


# ──────────────────────────────────────────
# モデルロード
# ──────────────────────────────────────────
def create_model(num_classes=2):
    # ConvNeXtV2-Base (timm)
    model = timm.create_model('convnextv2_base', pretrained=False, num_classes=num_classes)
    return model

def _sanitize_state_dict_for_load(sd: dict) -> dict:
    """_orig_mod. のようなプレフィックスを剥がす"""
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    return sd

def load_checkpoint(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    sd = _sanitize_state_dict_for_load(sd)
    incompat = model.load_state_dict(sd, strict=False)
    if getattr(incompat, "missing_keys", None):
        print("[WARN] missing keys:", incompat.missing_keys[:8], "...")
    if getattr(incompat, "unexpected_keys", None):
        print("[WARN] unexpected keys:", incompat.unexpected_keys[:8], "...")
    return model

# ──────────────────────────────────────────
# Grad-CAM（ConvNeXtV2）
#   - 最終ステージの最後の DWConv をフック（空間マップあり）
#   - w_c = GAP(∂y^c/∂A_c)、CAM = ReLU(Σ_c w_c * A_c)
# ──────────────────────────────────────────
class GradCamConvNeXt:
    def __init__(self, model: nn.Module):
        self.model = model.eval()
        self.activations = None
        self.gradients = None

        # まず最終ステージを取得
        last_stage = getattr(self.model, "stages", None)
        if last_stage is not None:
            last_stage = last_stage[-1]
        else:
            raise RuntimeError("ConvNeXtV2: model.stages が見つかりません")

        # 優先: 最終ブロックの dwconv
        target = None
        try:
            last_block = last_stage[-1]
            target = getattr(last_block, "dwconv", None)
        except Exception:
            target = None

        # 代替: ステージ内で「depthwise conv」を最後方から探索
        if target is None:
            for m in reversed(list(last_stage.modules())):
                if isinstance(m, nn.Conv2d) and m.groups == m.in_channels:
                    target = m
                    break
        if target is None:
            # 最悪のフォールバック（空間特徴が出る層）
            target = last_stage

        def fwd_hook(module, inp, out):
            self.activations = out
            if isinstance(out, torch.Tensor):
                out.retain_grad()
        def bwd_hook(module, gin, gout):
            self.gradients = gout[0]

        self.h1 = target.register_forward_hook(fwd_hook)
        self.h2 = target.register_full_backward_hook(bwd_hook)

    def __del__(self):
        for h in [getattr(self, "h1", None), getattr(self, "h2", None)]:
            if h is not None:
                try:
                    h.remove()
                except Exception:
                    pass

    @torch.no_grad()
    def _to_numpy01(self, x: torch.Tensor) -> np.ndarray:
        x = x.detach().cpu().numpy()
        x = (x - x.min()) / (x.max() - x.min() + 1e-8)
        return x

    def generate(self, inp: torch.Tensor, target_class: int, device: torch.device):
        """
        inp: (1,C,H,W) normalized tensor
        return: cam_small (h,w) in [0,1]
        """
        self.model.zero_grad(set_to_none=True)
        with torch.enable_grad():
            logits = self.model(inp.to(device))
            score = logits[:, target_class].sum()
            score.backward(retain_graph=False)

        A = self.activations            # (1,C,h,w)
        dYdA = self.gradients           # (1,C,h,w)

        if A is None or dYdA is None:
            raise RuntimeError("Hooks did not capture activations/gradients.")

        # グローバル平均プーリングで重み
        weights = dYdA.mean(dim=(2, 3), keepdim=True)      # (1,C,1,1)
        cam = torch.relu((weights * A).sum(dim=1, keepdim=False))  # (1,h,w) → (h,w)
        cam_small = self._to_numpy01(cam[0])
        return cam_small


# ──────────────────────────────────────────
# 可視化
# ──────────────────────────────────────────
def visualize(original_image, padded_image, resized_image,
              gradcam_map, pred_label, pred_prob,
              save_path=None, cmap='jet'):

    orig_w, orig_h = original_image.size
    is_square = (orig_w == orig_h)

    title_pred = f'Original (pred: {pred_label} {pred_prob*100:.1f}%)'
    title_cam  = f'Grad-CAM Map (class={pred_label}, p={pred_prob:.3f})'
    title_ovly = f'Grad-CAM Overlay (480×480, p={pred_prob:.3f})'

    if is_square:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        ax1.imshow(original_image); ax1.set_title(title_pred); ax1.axis('off')

        im2 = ax2.imshow(cv2.resize(gradcam_map, (IMAGE_SIZE, IMAGE_SIZE),
                                    interpolation=cv2.INTER_CUBIC),
                         cmap=cmap, vmin=0, vmax=1)
        ax2.set_title(title_cam); ax2.axis('off'); plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        ax3.imshow(resized_image)
        im3 = ax3.imshow(gradcam_map, cmap=cmap, alpha=0.55, vmin=0, vmax=1)
        ax3.set_title(title_ovly); ax3.axis('off'); plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    else:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(22, 5))
        ax1.imshow(original_image); ax1.set_title(title_pred); ax1.axis('off')
        ax2.imshow(padded_image);  ax2.set_title('Padded'); ax2.axis('off')

        im3 = ax3.imshow(cv2.resize(gradcam_map, (IMAGE_SIZE, IMAGE_SIZE),
                                    interpolation=cv2.INTER_CUBIC),
                         cmap=cmap, vmin=0, vmax=1)
        ax3.set_title(title_cam); ax3.axis('off'); plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        ax4.imshow(resized_image)
        im4 = ax4.imshow(gradcam_map, cmap=cmap, alpha=0.55, vmin=0, vmax=1)
        ax4.set_title(title_ovly); ax4.axis('off'); plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SAVE] 可視化を保存: {save_path}")
    plt.show()
    plt.close(fig)

# ──────────────────────────────────────────
# 推論
# ──────────────────────────────────────────
def predict_convnext_single(image_path, ckpt_path, device='cpu',
                            save_dir=None, cmap='jet', target_class=None):
    device = torch.device(device if (device == 'cuda' and torch.cuda.is_available()) else 'cpu')
    print(f"[Device] {device}")

    # 画像前処理
    x, original, padded, resized = preprocess_image(image_path)

    # モデル
    model = create_model(num_classes=len(LABEL_NAMES))
    model = load_checkpoint(model, ckpt_path)
    model = model.to(device)
    model.eval()

    # 予測
    with torch.inference_mode():
        logits = model(x.to(device))
        probs = softmax(logits.detach().cpu().numpy(), axis=1)[0]
        pred_id = int(np.argmax(probs))
        pred_label = ID2LABEL[pred_id]
        pred_prob = float(probs[pred_id])

    print("\n=== 予測結果 (ConvNeXtV2) ===")
    print(f"予測クラス: {pred_label} ({pred_id})")
    for i, name in enumerate(LABEL_NAMES):
        print(f"{name} 確率: {probs[i]:.4f}")

    # Grad-CAM（予測クラスに対して）
    cam_engine = GradCamConvNeXt(model)
    target_id = pred_id if target_class is None else int(target_class)
    if not (0 <= target_id < len(LABEL_NAMES)):
        raise ValueError(f"--target-class は 0..{len(LABEL_NAMES)-1} の範囲にしてください: {target_id}")
    cam_small = cam_engine.generate(x, target_class=target_id, device=device)
    del cam_engine
    cam_map = cv2.resize(cam_small, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    cam_map = (cam_map - cam_map.min()) / (cam_map.max() - cam_map.min() + 1e-8)

    # 表示/保存
    viz_path = None
    if save_dir:
        base = os.path.splitext(os.path.basename(image_path))[0]
        viz_path = os.path.join(save_dir, f"{base}_convnextv2_gradcam.png")

    cam_label = ID2LABEL[target_id]
    visualize(original, padded, resized, cam_map, cam_label, probs[target_id],
              save_path=viz_path, cmap=cmap)

    return {
        "predicted_class": pred_id,
        "predicted_label": pred_label,
        "probabilities": probs,
        "gradcam": cam_small
    }


# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────
def main():
    p = argparse.ArgumentParser("ConvNeXtV2 single-image inference + Grad-CAM")
    p.add_argument("image_path", type=str, help="入力画像へのパス")
    p.add_argument("--checkpoint", type=str, required=True, help="学習済みckpt（best.pth/ckpt_xxxx.pth/latest.pthなど）")
    p.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"])
    p.add_argument("--save-dir", type=str, default=None, help="可視化の保存先ディレクトリ")
    p.add_argument("--cmap", type=str, default="jet", help="カラーマップ（jet, viridis など）")
    p.add_argument("--target-class", type=int, default=None, help="Grad-CAMのクラスID（0:tako, 1:AMI）。未指定なら予測クラス。")
    args = p.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(args.image_path)
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(args.checkpoint)

    try:
        res = predict_convnext_single(
            image_path=args.image_path,
            ckpt_path=args.checkpoint,
            device=args.device,
            save_dir=args.save_dir,
            cmap=args.cmap,
            target_class=args.target_class
        )
        print("\n=== 詳細 ===")
        print(f"画像: {args.image_path}")
        print(f"予測: {res['predicted_label']}")
        print(f"確率: tako={res['probabilities'][0]:.4f}, AMI={res['probabilities'][1]:.4f}")
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback; traceback.print_exc()

    print('Done.', datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9), 'JST')))

if __name__ == "__main__":
    main()

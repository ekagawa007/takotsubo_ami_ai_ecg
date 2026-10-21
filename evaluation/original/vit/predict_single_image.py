#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
単一画像の推測とattention map表示
version 1.0.0 20250803 とりあえず動くようになった。
version 1.0.1 20250807 終了時刻を表示するようにした。
version 1.0.2 20250815 モデルとチェックポイントのパスを引数で指定できるようにした。
version 1.0.3 20250904 best, 2nd, 3rd modelのパスを追加し、bestをdefaultにした。もともとは2nd model。
version 1.0.4 20250904 画像が正方形でない場合はpaddingして正方形にするようにした。
version 1.0.5 20250904 attention mapの可視化を修正し、元画像とpadding画像の対応関係を正しく表示するようにした。
version 1.0.6 20250904 4画像表示の最後のパネルを480x480のpadding画像とattention mapの重ね合わせに修正した。
version 1.0.8 20250904 processorに入れる前に手動で480x480にリサイズするように修正し、可視化も480x480のリサイズ画像に対応させた。
version 1.0.9 20250904 attention mapとViT-Grad-CAMを表示するように修正、最後3層のみを使用するようにした。

使用法:
TTAなし:
python predict_single_image.py /path/to/image.png --checkpoint /path/to/checkpoint-xxxx --device cuda --save-dir /path/to/out

TTAあり:
python predict_single_image.py /path/to/image.png --checkpoint /path/to/checkpoint-xxxx --device cuda --save-dir /path/to/out --tta 8 --tta-magnitude 5
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns
from PIL import Image
from transformers import AutoModelForImageClassification, AutoImageProcessor
#from torchvision import transforms
from torchvision.transforms import RandAugment, InterpolationMode
from scipy.special import softmax
import argparse
from collections import OrderedDict
import cv2
import datetime
import math


# --------------------------------------------------
# デフォルト定数定義
# --------------------------------------------------

# デフォルトのモデルパス（評価で使用したモデルと同じ）
# best model
DEFAULT_CHECKPT_PARENT = "/path/to/checkpoints/ft_2_class/vit/tako_ami_ft2_vit_prod_gd_20250817_5_prt_ft_2class_ckpt"
DEFAULT_CHECKPT = f"{DEFAULT_CHECKPT_PARENT}/checkpoint-6900"

# 2nd model
#DEFAULT_CHECKPT_PARENT = "/path/to/checkpoints/ft_2_class/vit/tako_ami_ft2_vit_prod_gd_20250814_3_prt_ft_2class_ckpt"
#DEFAULT_CHECKPT = f"{DEFAULT_CHECKPT_PARENT}/checkpoint-7475"

# 3rd model
#DEFAULT_CHECKPT_PARENT = "/path/to/checkpoints/ft_2_class/vit/tako_ami_ft2_vit_prod_gd_20250815_2_prt_ft_2class_ckpt"
#DEFAULT_CHECKPT = f"{DEFAULT_CHECKPT_PARENT}/checkpoint-7475"

# ラベル定義
label_names = ["tako", "AMI"]
label2id = {n: i for i, n in enumerate(label_names)}
id2label = {i: n for n, i in label2id.items()}

# 画像サイズ
IMAGE_SIZE = 480

# 正規化パラメータ
#NORMALIZE_MEAN = [0.485, 0.456, 0.406]
#NORMALIZE_STD = [0.229, 0.224, 0.225]

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
# モデルとプロセッサのロード
# --------------------------------------------------

def load_model(checkpoint_path, parent_path=None):
    """
    モデルとプロセッサをロード
    
    Args:
        checkpoint_path (str): チェックポイントのパス
        parent_path (str): 親ディレクトリのパス（Noneの場合はcheckpoint_pathの親ディレクトリを使用）
    
    Returns:
        tuple: (model, processor)
    """
    print(f"モデルをロード中: {checkpoint_path}")
    
    # 親ディレクトリのパスが指定されていない場合は、checkpoint_pathの親ディレクトリを使用
    if parent_path is None:
        parent_path = os.path.dirname(checkpoint_path)
    
    model = AutoModelForImageClassification.from_pretrained(
        checkpoint_path,
        num_labels=len(label_names),
        id2label=id2label,
        label2id=label2id,
        output_attentions=True,  # attention mapを取得するため
    )
    
    processor = AutoImageProcessor.from_pretrained(parent_path)
    
    return model, processor

# --------------------------------------------------
# 画像の前処理
# --------------------------------------------------

def pad_to_square(pil_img, fill=(0, 0, 0)):
    """長い辺の長さで正方形キャンバスを作り、中央に元画像を貼り付け"""
    w, h = pil_img.size
    side = max(w, h)
    new_img = Image.new(pil_img.mode, (side, side), color=fill)
    new_img.paste(pil_img, ((side - w) // 2, (side - h) // 2))
    return new_img

def preprocess_image(image_path, processor):
    """画像を前処理（正方形でない場合はpaddingして正方形にし、480x480にリサイズ）"""
    # 画像を読み込み
    original_image = Image.open(image_path).convert('RGB')
    
    # 元の画像サイズを取得
    width, height = original_image.size
    print(f"元画像サイズ: {width} x {height}")
    
    # 正方形でない場合はpaddingして正方形にする
    if width != height:
        print("画像が正方形でないため、paddingして正方形にします")
        padded_image = pad_to_square(original_image)
        print(f"padding後サイズ: {padded_image.size[0]} x {padded_image.size[1]}")
    else:
        padded_image = original_image
    
    # 480x480にリサイズ
    resized_image = padded_image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    print(f"リサイズ後サイズ: {resized_image.size[0]} x {resized_image.size[1]}")
    
    # プロセッサで前処理（正規化とテンソル変換のみ）
    inputs = processor(images=resized_image, return_tensors="pt")
    
    return inputs, original_image, padded_image, resized_image

# --------------------------------------------------
# Attention mapの可視化
# --------------------------------------------------

def build_attention_map(attentions, image_size=IMAGE_SIZE):
    """
    最終層の CLS→パッチ の生アテンション（ヘッド平均）を画像サイズに拡大して返す
    attentions: tuple(len=L) of (B, H, N, N)
    """
    att = attentions[-1][0].detach().cpu().numpy()  # (H, N, N)
    att_cls = att[:, 0, 1:]                         # (H, N-1)
    cls_attention = att_cls.mean(axis=0)            # (N-1,)

    g = int(np.sqrt(cls_attention.shape[0]))
    assert g * g == cls_attention.shape[0], f"num_patchesが平方数でありません: {cls_attention.shape[0]}"

    att_small = cls_attention.reshape(g, g)         # (g, g)
    att_map = cv2.resize(att_small, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
    return att_map

def _patch_start_index(num_tokens: int) -> int:
    # CLSのみ→1、CLS+DIST→2（N-2が平方数ならDISTありとみなす）
    if int(math.isqrt(num_tokens - 1)) ** 2 == num_tokens - 1:
        return 1
    if int(math.isqrt(num_tokens - 2)) ** 2 == num_tokens - 2:
        return 2
    return 1

def vit_gradcam_hidden(model, pixel_values, target_class, last_k_layers: int = 4):
    """
    ViT-Grad-CAM（Transformerブロックの layernorm_before 出力にhook）
    - 直近k層（デフォルト4）の layernorm_before 出力 (B,N,C) を集め、
      それぞれの勾配でCAMを作って平均
    """
    model.eval()
    model.zero_grad(set_to_none=True)

    layers = model.vit.encoder.layer
    k = min(last_k_layers, len(layers))
    take_ids = list(range(len(layers)-k, len(layers)))  # 最後k層

    feats = []
    handles = []
    def _hook(module, inp, out):
        feats.append(out)   # (B,N,C) 計算グラフに接続済み

    # 各層の「注意直前」にhook
    for i in take_ids:
        handles.append(layers[i].layernorm_before.register_forward_hook(_hook))

    # forward（no_gradは使わない）
    out = model(pixel_values, return_dict=True)

    # hook解除
    for h in handles:
        h.remove()

    # 対象クラスのスコアで勾配
    score = out.logits[:, target_class].sum()
    grads = torch.autograd.grad(
        outputs=score,
        inputs=feats,        # 複数テンソルに一度に勾配
        retain_graph=False,
        create_graph=False
    )

    cams = []
    for f, g in zip(feats, grads):
        # f,g: (B,N,C)
        N = f.shape[1]
        start = _patch_start_index(N)
        g_p = g[0, start:, :]              # (num_patches, C)
        f_p = f.detach()[0, start:, :]     # (num_patches, C)

        weights = g_p.mean(dim=0)          # (C,)  チャネル平均のGradを重みとして
        cam_tokens = torch.relu((f_p * weights).sum(dim=-1))  # (num_patches,)

        num = cam_tokens.numel()
        gsize = int(math.isqrt(num))
        assert gsize * gsize == num, f"パッチ数が平方数でありません: {num}"
        cams.append(cam_tokens.reshape(gsize, gsize))

    cam = torch.stack(cams, dim=0).mean(0).detach().cpu().numpy()  # 層平均
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    print("grad max per layer:", [g.abs().max().item() for g in grads])

    return cam


def visualize_attn_and_gradcam(original_image, padded_image, resized_image,
                               attn_map, gradcam_map,
                               pred_label,
                               pred_prob,
                               pred_label_tta=None,
                               pred_prob_tta=None,
                               save_path=None, cmap='jet'):
    orig_w, orig_h = original_image.size
    is_square = (orig_w == orig_h)

    # ---- caption text ----
    if pred_prob_tta is not None:
        # ラベルが違っても分かるように両方表示
        label_tta = pred_label_tta if pred_label_tta is not None else pred_label
        title_pred = (
            f'Original (pred: {pred_label} '
            f'base={pred_prob*100:.1f}%, '
            f'TTA={label_tta} {pred_prob_tta*100:.1f}%)'
        )
        title_gradcam = (
            f'ViT-Grad-CAM Map (base={pred_label}, p={pred_prob:.3f} | '
            f'TTA={label_tta}, p={pred_prob_tta:.3f})'
        )
        title_overlay = (
            f'ViT-Grad-CAM Overlay (480x480, '
            f'base p={pred_prob:.3f} | TTA p={pred_prob_tta:.3f})'
        )
    else:
        title_pred = f'Original (pred: {pred_label} {pred_prob*100:.1f}%)'
        title_gradcam = f'ViT-Grad-CAM Map (class={pred_label}, p={pred_prob:.3f})'
        title_overlay = f'ViT-Grad-CAM Overlay (480x480, p={pred_prob:.3f})'

    if is_square:
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        ax1, ax2, ax3, ax4 = axes
        ax1.imshow(original_image); ax1.set_title(title_pred); ax1.axis('off')

        im2 = ax2.imshow(attn_map, cmap=cmap, alpha=0.9)
        ax2.set_title('Attention Map (CLS→patch, last, head-avg)'); ax2.axis('off')
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        im3 = ax3.imshow(gradcam_map, cmap=cmap, alpha=0.9, vmin=0, vmax=1)
        ax3.set_title(title_gradcam); ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        ax4.imshow(resized_image)
        im4 = ax4.imshow(gradcam_map, cmap=cmap, alpha=0.6, vmin=0, vmax=1)
        ax4.set_title(title_overlay); ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    else:
        fig, axes = plt.subplots(1, 5, figsize=(26, 5))
        ax1, ax2, ax3, ax4, ax5 = axes
        ax1.imshow(original_image); ax1.set_title(title_pred); ax1.axis('off')
        ax2.imshow(padded_image);  ax2.set_title('Padded Image'); ax2.axis('off')

        im3 = ax3.imshow(attn_map, cmap=cmap, alpha=0.9, vmin=0, vmax=1)
        ax3.set_title('Attention Map (CLS→patch, last, head-avg)'); ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        im4 = ax4.imshow(gradcam_map, cmap=cmap, alpha=0.9, vmin=0, vmax=1)
        ax4.set_title(title_gradcam); ax4.axis('off')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        ax5.imshow(resized_image)
        im5 = ax5.imshow(gradcam_map, cmap=cmap, alpha=0.6)
        ax5.set_title(title_overlay); ax5.axis('off')
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"可視化を保存しました: {save_path}")
    plt.show()

# --------------------------------------------------
# 予測実行
# --------------------------------------------------

def predict_single_image(image_path, model, processor, save_dir=None, cmap='jet', tta_result=None):
    """単一画像の予測を実行"""
    print(f"画像を読み込み中: {image_path}")
    
    # 画像の前処理
    inputs, original_image, padded_image, resized_image = preprocess_image(image_path, processor)
    
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # 予測実行
    model.eval()
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
        logits = outputs.logits
        attentions = outputs.attentions
    
    # 確率に変換
    probs = softmax(logits.detach().cpu().numpy(), axis=1)
    predicted_class = np.argmax(probs, axis=1)[0]
    confidence = np.max(probs, axis=1)[0]
    
    # 追加（pred_prob を明示的に作成）
    pred_prob = float(probs[0, predicted_class])

    # 結果表示
    print(f"\n=== 予測結果 ===")
    print(f"予測クラス: {id2label[predicted_class]} ({predicted_class})")
    print(f"確信度: {confidence:.4f}")
    print(f"tako確率: {probs[0, 0]:.4f}")
    print(f"AMI確率: {probs[0, 1]:.4f}")
    
    # Attention（生）と ViT-Grad-CAM を作る
    attn_map = build_attention_map(attentions, image_size=IMAGE_SIZE)

    # ViT-Grad-CAM（隠れ状態版）：予測クラスに対して
    gradcam_small = vit_gradcam_hidden(model, inputs["pixel_values"], target_class=predicted_class, last_k_layers=3)
    gradcam_map = cv2.resize(gradcam_small, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-8)

    # 表示・保存
    pred_label = id2label[predicted_class]
    pred_prob_base = pred_prob

    pred_label_tta = None
    pred_prob_tta = None
    if tta_result is not None:
        pred_label_tta = tta_result.get("predicted_label", None)
        conf = tta_result.get("confidence", None)
        pred_prob_tta = float(conf) if conf is not None else None

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

        path_parts = os.path.normpath(image_path).split(os.sep)
        prefix = ""
        try:
            if len(path_parts) >= 3:
                subdir1 = path_parts[-3]; subdir2 = path_parts[-2]
                prefix = f"{subdir1}_{subdir2}_"
            elif len(path_parts) == 2:
                subdir2 = path_parts[-2]
                prefix = f"{subdir2}_"
        except IndexError:
            prefix = ""

        base_name = os.path.splitext(os.path.basename(image_path))[0]
        viz_save_path = os.path.join(save_dir, f"{prefix}{base_name}_attn_gradcam.png")

        visualize_attn_and_gradcam(
            original_image, padded_image, resized_image,
            attn_map, gradcam_map,
            pred_label, pred_prob_base,
            pred_label_tta=pred_label_tta,
            pred_prob_tta=pred_prob_tta,
            save_path=viz_save_path, cmap=cmap
        )
    else:
        visualize_attn_and_gradcam(
            original_image, padded_image, resized_image,
            attn_map, gradcam_map,
            pred_label, pred_prob_base,
            pred_label_tta=pred_label_tta,
            pred_prob_tta=pred_prob_tta,
            save_path=None, cmap=cmap
        )

    return {
        'predicted_class': predicted_class,
        'predicted_label': id2label[predicted_class],
        'confidence': confidence,
        'probabilities': probs[0],
        'attention_maps': attentions
    }

def predict_single_image_tta(image_path, model, processor,
                             tta=8, tta_magnitude=5):
    """
    1枚画像に対して Test-Time Augmentation を行い、
    logits を平均した結果で予測する（attention/Grad-CAM は描画しない）。
    - 1回分は必ず "素の画像"（TTAなし）を含める。
    """
    print(f"\n[INFO] TTA 推論を開始します: tta={tta}, magnitude={tta_magnitude}")

    # 安全のためクリップ
    tta = max(1, int(tta))
    tta_magnitude = int(max(0, min(10, tta_magnitude)))

    # まず一度 preprocess_image を呼んで、正方形 padding + 480x480 画像を得る
    _, original_image, padded_image, resized_image = preprocess_image(
        image_path, processor
    )

    # 1回目は identity（拡張なし）
    images_for_tta = [resized_image]

    # 2回目以降は RandAugmentNoFlip を適用
    if tta > 1:
        aug = RandAugmentNoFlip(
            num_ops=2,
            magnitude=tta_magnitude,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        for _ in range(tta - 1):
            img_aug = aug(resized_image)
            images_for_tta.append(img_aug)

    logits_sum = None

    model.eval()
    with torch.no_grad():
        for i, img_aug in enumerate(images_for_tta):
            inputs = processor(images=img_aug, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

            outputs = model(**inputs)  # attentions は不要
            logits = outputs.logits.detach().cpu().numpy()  # (1, num_classes)

            if logits_sum is None:
                logits_sum = logits
            else:
                logits_sum += logits

    logits_mean = logits_sum / float(len(images_for_tta))
    probs = softmax(logits_mean, axis=1)
    predicted_class = int(np.argmax(probs, axis=1)[0])
    confidence = float(np.max(probs, axis=1)[0])

    print(f"\n=== TTAありの予測結果 ===")
    print(f"予測クラス (TTA): {id2label[predicted_class]} ({predicted_class})")
    print(f"確信度 (TTA):     {confidence:.4f}")
    print(f"tako確率 (TTA):   {probs[0, 0]:.4f}")
    print(f"AMI確率 (TTA):    {probs[0, 1]:.4f}")

    return {
        "predicted_class": predicted_class,
        "predicted_label": id2label[predicted_class],
        "confidence": confidence,
        "probabilities": probs[0],
    }

# --------------------------------------------------
# メイン関数
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='単一画像の推測とattention map表示')
    parser.add_argument('image_path', type=str, help='入力画像のパス')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPT, 
                       help=f'チェックポイントのパス (デフォルト: {DEFAULT_CHECKPT})')
    parser.add_argument('--parent-path', type=str, default=None,
                       help='親ディレクトリのパス（Noneの場合はcheckpoint_pathの親ディレクトリを使用）')
    parser.add_argument('--save-dir', type=str, help='Attention mapの保存ディレクトリ（指定すると保存される）')
    parser.add_argument('--cmap', type=str, default='jet', help='Attention mapのカラーマップ (hot, cool, viridis, plasma, jet等)')
    parser.add_argument('--device', type=str, default='cpu', help='使用デバイス (cpu/cuda)')
    parser.add_argument('--tta', type=int, default=16, help='Test-Time Augmentation の回数 (1 ならTTAなし)')
    parser.add_argument('--tta-magnitude', type=int, default=3, help='TTA 時 RandAugment の magnitude (0–10, 推奨 3–7)')    
    
    args = parser.parse_args()
    
    # 画像ファイルの存在確認
    if not os.path.exists(args.image_path):
        print(f"エラー: 画像ファイルが見つかりません: {args.image_path}")
        return
    
    # デバイス設定
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    print(f"使用デバイス: {device}")
    
    # TTA パラメータを安全な範囲にクリップ
    args.tta = max(1, int(args.tta))
    args.tta_magnitude = int(max(0, min(10, args.tta_magnitude)))
    if args.tta <= 1:
        print("[INFO] TTAは無効（1回、拡張なし）で推論します。")
    else:
        print(f"[INFO] TTAを有効化: 回数={args.tta}, magnitude={args.tta_magnitude}")

    try:
        # モデルとプロセッサをロード
        model, processor = load_model(args.checkpoint, args.parent_path)
        model = model.to(device)
        
        if args.tta <= 1:
            # 従来どおり：1回だけ推論＋可視化
            result = predict_single_image(
                args.image_path, model, processor, args.save_dir, args.cmap
            )
            
            print(f"\n=== 詳細結果 ===")
            print(f"画像パス: {args.image_path}")
            print(f"予測: {result['predicted_label']}")
            print(f"確信度: {result['confidence']:.4f}")
            print(f"tako確率: {result['probabilities'][0]:.4f}")
            print(f"AMI確率: {result['probabilities'][1]:.4f}")
        
        else:
            # 先に TTA ありで確率を計算（可視化なし）
            print("\n[INFO] 先に TTA ありで推論します（可視化なし）...")
            result_tta = predict_single_image_tta(
                args.image_path, model, processor,
                tta=args.tta, tta_magnitude=args.tta_magnitude
            )

            # 次に TTA なしで 1 回だけ推論（可視化あり）
            print("\n[INFO] 次に TTA なしで 1 回だけ推論します（可視化あり）...")
            result_base = predict_single_image(
                args.image_path, model, processor, args.save_dir, args.cmap,
                tta_result=result_tta
            )
            # まとめて表示
            print(f"\n=== 詳細結果（TTAなし）===")
            print(f"画像パス: {args.image_path}")
            print(f"予測: {result_base['predicted_label']}")
            print(f"確信度: {result_base['confidence']:.4f}")
            print(f"tako確率: {result_base['probabilities'][0]:.4f}")
            print(f"AMI確率: {result_base['probabilities'][1]:.4f}")

            print(f"\n=== 詳細結果（TTAあり, 平均ログit）===")
            print(f"予測: {result_tta['predicted_label']}")
            print(f"確信度: {result_tta['confidence']:.4f}")
            print(f"tako確率: {result_tta['probabilities'][0]:.4f}")
            print(f"AMI確率: {result_tta['probabilities'][1]:.4f}")

    except Exception as e:
        print(f"エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
    print('Done. ', datetime.datetime.now(datetime.timezone(datetime.timedelta(hours = 9), 'JST')))
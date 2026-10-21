#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
単一画像/ディレクトリ推論（Ensemble: ViT + ConvNeXtV2 + MLP-Mixer）

概要:
- 入力は「画像ファイル」または「画像ディレクトリ」。
- ディレクトリ指定時は配下の画像を列挙して順に推論（--recursive で再帰探索）。
- 入力画像は「非正方形→(padding or center crop)→480x480」に統一して推論。

保存（--save-dir 指定時）:
- 推論結果を save-dir 配下のサブディレクトリへ保存
  1) 可視化PNG  : save-dir/png/<image>_ensemble_probs.png
  2) 予測JSON   : save-dir/json/<image>_ensemble_preds.json
  3) 結果CSV    : save-dir/csv/ensemble_results.csv（--output-csv 未指定時）

- Grad-CAM / Attention の保存（表示はしない）
  - ViT       : save-dir/gradcam/vit/<image>_vit_attn_gradcam.png
  - ConvNeXtV2: save-dir/gradcam/convnextv2/<image>_convnext_gradcam_<layer>.png
  ※ --no-vit-maps / --no-convnext-gradcam で個別に無効化可能

追加オプション:
- --output-csv <path> :
   結果CSVの追記先を明示指定。
   指定した場合は --save-dir に関わらずこのパスへ追記（サブディレクトリ化はしない）。

- --ensemble-mode probs|logits :
   アンサンブル方式の切り替え
   - probs  : 各モデルの softmax確率を重み付き平均
   - logits : 各モデルの logits を重み付き平均 -> softmax（デフォルト）

- --mixer-temperature <float> :
   MLP-Mixer の logits に温度スケーリングを適用（T>0）。
   T>1 で極端な出力を緩和。
   T<1 でより尖らせる。

既存要件:
1) 重み付け平均のオプション --weights vit conv mixer
2) 3モデルそれぞれでTTAを引数指定
   - ViT       : default tta=24, magnitude=1
   - ConvNeXtV2: default tta=1  (実質TTAなし)
   - MLP-Mixer : default tta=32, magnitude=3
3) --save-dir で可視化PNG
   「入力480画像 + 3モデル確率 + アンサンブル確率」

前提:
- 3モデルとも 2クラス（tako / AMI）
- 3モデルとも 480x480 で学習
- 入力画像は ViT版と同じ方針で
  非正方形→(padding or cropping)→480x480

version 1.2.0 2025-12-21 save PNG/JSON/CSV into subdirectories under --save-dir (png/json/csv)
"""

import os
import argparse
import datetime
import json
import csv
from collections import OrderedDict
import glob
import math

import numpy as np
import torch
from PIL import Image

import timm

# ★表示不要/ヘッドレス環境でも安全に保存だけできるように Agg backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from torchvision import transforms
from torchvision.transforms import RandAugment, InterpolationMode
from transformers import AutoModelForImageClassification, AutoImageProcessor
import cv2
import torch.nn.functional as F
import sys

if sys.version_info < (3, 10):
    raise RuntimeError("This script requires Python 3.10+ (because of PEP604 type hints like str | None).")

# --------------------------------------------------
# 定数
# --------------------------------------------------
LABEL_NAMES = ["tako", "AMI"]
LABEL2ID = {n: i for i, n in enumerate(LABEL_NAMES)}
ID2LABEL = {i: n for n, i in LABEL2ID.items()}

IMAGE_SIZE = 480
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

# --------------------------------------------------
# 時刻（JST）
# --------------------------------------------------
def now_jst_iso():
    return datetime.datetime.now(
        datetime.timezone(datetime.timedelta(hours=9), "JST")
    ).isoformat()

# --------------------------------------------------
# RandAugment（Flip除外）
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
# 画像前処理（非正方形→正方形化→480）
# --------------------------------------------------
def pad_to_square(pil_img: Image.Image, fill=(0, 0, 0)) -> Image.Image:
    w, h = pil_img.size
    side = max(w, h)
    new_img = Image.new(pil_img.mode, (side, side), color=fill)
    new_img.paste(pil_img, ((side - w) // 2, (side - h) // 2))
    return new_img

def center_crop_to_square(pil_img: Image.Image) -> Image.Image:
    w, h = pil_img.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    right = left + side
    bottom = top + side
    return pil_img.crop((left, top, right, bottom))

def preprocess_image_480(image_path: str, square_mode: str = "pad"):
    original = Image.open(image_path).convert("RGB")
    w, h = original.size

    if w != h:
        if square_mode == "crop":
            squared = center_crop_to_square(original)
        else:
            squared = pad_to_square(original)
    else:
        squared = original

    resized = squared.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    return original, squared, resized

# --------------------------------------------------
# 画像収集（ファイルorディレクトリ）
# --------------------------------------------------
def collect_image_paths(
    input_path: str,
    recursive: bool = False,
    exts: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"),
):
    if os.path.isfile(input_path):
        return [input_path]
    if not os.path.isdir(input_path):
        raise FileNotFoundError(input_path)

    pattern_base = "**/*" if recursive else "*"
    paths = []
    for ext in exts:
        paths.extend(glob.glob(os.path.join(input_path, f"{pattern_base}{ext}"), recursive=recursive))
        paths.extend(glob.glob(os.path.join(input_path, f"{pattern_base}{ext.upper()}"), recursive=recursive))

    # 重複除去＋ソート
    paths = sorted(list(dict.fromkeys(paths)))
    return paths

# --------------------------------------------------
# 共通 softmax（numpy）
# --------------------------------------------------
def softmax_np(x, axis=1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)

# --------------------------------------------------
# Temperature scaling
# --------------------------------------------------
def apply_temperature_to_logits(logits: np.ndarray, temperature: float) -> np.ndarray:
    t = float(temperature)
    if not np.isfinite(t) or t <= 0:
        raise ValueError("--mixer-temperature must be a positive finite number.")
    return logits / t

# --------------------------------------------------
# save-dir 下のサブディレクトリ作成
# --------------------------------------------------
def ensure_output_subdirs(save_dir: str):
    """
    save_dir 配下に出力用サブディレクトリを作成して返す
    - save_dir/png
    - save_dir/json
    - save_dir/csv
    """
    png_dir = os.path.join(save_dir, "png")
    json_dir = os.path.join(save_dir, "json")
    csv_dir = os.path.join(save_dir, "csv")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(json_dir, exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)
    return png_dir, json_dir, csv_dir

# --------------------------------------------------
# ViT (Hugging Face)
# --------------------------------------------------
def load_vit(vit_checkpoint: str, vit_parent_path: str | None, device):
    if vit_parent_path is None:
        vit_parent_path = os.path.dirname(vit_checkpoint)

    try:
        model_and_info = AutoModelForImageClassification.from_pretrained(
            vit_checkpoint,
            num_labels=len(LABEL_NAMES),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            output_loading_info=True,
            ignore_mismatched_sizes=True,
        )
    except TypeError:
        model_and_info = AutoModelForImageClassification.from_pretrained(
            vit_checkpoint,
            num_labels=len(LABEL_NAMES),
            id2label=ID2LABEL,
            label2id=LABEL2ID,
            ignore_mismatched_sizes=True,
        )

    loading_info = None
    if isinstance(model_and_info, tuple) and len(model_and_info) == 2:
        model, loading_info = model_and_info
    else:
        model = model_and_info

    if loading_info is not None:
        missing = loading_info.get("missing_keys", [])
        unexpected = loading_info.get("unexpected_keys", [])
        mismatched = loading_info.get("mismatched_keys", [])
        if missing:
            print("[WARN] ViT missing keys:")
            for k in missing:
                print("  -", k)
        if unexpected:
            print("[WARN] ViT unexpected keys:")
            for k in unexpected:
                print("  -", k)
        if mismatched:
            print("[WARN] ViT mismatched keys:")
            for item in mismatched:
                print("  -", item)
        if (not missing) and (not unexpected) and (not mismatched):
            print("[INFO] ViT checkpoint loaded cleanly.")
    else:
        print("[INFO] ViT loaded (no loading_info available).")

    processor = AutoImageProcessor.from_pretrained(vit_parent_path)
    model.to(device).eval()

    if hasattr(model, "config"):
        model.config.output_attentions = False
        model.config.return_dict = True

    return model, processor, vit_parent_path

def vit_logits_once(model, processor, img_pil: Image.Image, device):
    inputs = processor(images=img_pil, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        out = model(**inputs)
        logits = out.logits.detach().cpu().numpy()
    return logits

def predict_vit_tta_logits(model, processor, resized_480: Image.Image,
                           device, tta: int = 16, magnitude: int = 1):
    tta = max(1, int(tta))
    magnitude = int(max(0, min(10, magnitude)))

    images = [resized_480]
    if tta > 1:
        aug = RandAugmentNoFlip(
            num_ops=2,
            magnitude=magnitude,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        for _ in range(tta - 1):
            images.append(aug(resized_480))

    logits_sum = None
    for img in images:
        lg = vit_logits_once(model, processor, img, device)
        logits_sum = lg if logits_sum is None else (logits_sum + lg)

    return logits_sum / float(len(images))

# --------------------------------------------------
# ConvNeXtV2 / MLP-Mixer 共通 logits 1回
# --------------------------------------------------
def conv_mixer_logits_once(model, img_pil: Image.Image, device):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    x = tfm(img_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x).detach().cpu().numpy()
    return logits

# --------------------------------------------------
# ConvNeXtV2 (timm)
# --------------------------------------------------
def _sanitize_state_dict_for_load(sd: dict) -> dict:
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    return sd

def load_convnextv2(ckpt_path: str, device):
    model = timm.create_model("convnextv2_base", pretrained=False, num_classes=len(LABEL_NAMES))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    sd = _sanitize_state_dict_for_load(sd)
    incompatible = model.load_state_dict(sd, strict=False)

    if incompatible.missing_keys:
        print("[WARN] ConvNeXtV2 missing keys:")
        for k in incompatible.missing_keys:
            print("  -", k)
    if incompatible.unexpected_keys:
        print("[WARN] ConvNeXtV2 unexpected keys:")
        for k in incompatible.unexpected_keys:
            print("  -", k)
    if not incompatible.missing_keys and not incompatible.unexpected_keys:
        print("[INFO] ConvNeXtV2 state_dict loaded cleanly.")

    model.to(device).eval()
    return model

def predict_convnext_tta_logits(model, resized_480: Image.Image,
                                device, tta: int = 1, magnitude: int = 3):
    tta = max(1, int(tta))
    magnitude = int(max(0, min(10, magnitude)))

    images = [resized_480]
    if tta > 1:
        aug = RandAugmentNoFlip(
            num_ops=2,
            magnitude=magnitude,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        for _ in range(tta - 1):
            images.append(aug(resized_480))

    logits_sum = None
    for img in images:
        lg = conv_mixer_logits_once(model, img, device)
        logits_sum = lg if logits_sum is None else (logits_sum + lg)

    return logits_sum / float(len(images))

# --------------------------------------------------
# MLP-Mixer (timm)
# --------------------------------------------------
def _strip_prefixes(k: str) -> str:
    if k.startswith("_orig_mod."):
        k = k[len("_orig_mod."):]
    if k.startswith("module."):
        k = k[len("module."):]
    return k

def resolve_mixer_checkpoint_path(model_path_or_file: str) -> str:
    path = model_path_or_file
    if os.path.isdir(path):
        cands = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pth")]
        if not cands:
            raise FileNotFoundError(f"No .pth found under: {path}")
        path = sorted(cands)[-1]
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path

def load_mlpmixer(model_path_or_file: str, device):
    model = timm.create_model(
        "mixer_b16_224",
        pretrained=False,
        img_size=IMAGE_SIZE,
        in_chans=3,
        num_classes=len(LABEL_NAMES),
    )
    ckpt_path = resolve_mixer_checkpoint_path(model_path_or_file)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)
    new_state = {_strip_prefixes(k): v for k, v in state.items()}
    incompatible = model.load_state_dict(new_state, strict=False)

    if incompatible.missing_keys:
        print("[WARN] MLP-Mixer missing keys:")
        for k in incompatible.missing_keys:
            print("  -", k)
    if incompatible.unexpected_keys:
        print("[WARN] MLP-Mixer unexpected keys:")
        for k in incompatible.unexpected_keys:
            print("  -", k)
    if not incompatible.missing_keys and not incompatible.unexpected_keys:
        print("[INFO] MLP-Mixer state_dict loaded cleanly.")

    model.to(device).eval()
    return model, ckpt_path

def predict_mlpmixer_tta_logits(model, resized_480: Image.Image,
                                device, tta: int = 16, magnitude: int = 3):
    tta = max(1, int(tta))
    magnitude = int(max(0, min(10, magnitude)))

    images = [resized_480]
    if tta > 1:
        aug = RandAugmentNoFlip(
            num_ops=2,
            magnitude=magnitude,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        for _ in range(tta - 1):
            images.append(aug(resized_480))

    logits_sum = None
    for img in images:
        lg = conv_mixer_logits_once(model, img, device)
        logits_sum = lg if logits_sum is None else (logits_sum + lg)

    return logits_sum / float(len(images))

# --------------------------------------------------
# Ensemble
# --------------------------------------------------
def normalize_weights(w):
    w = np.array(w, dtype=np.float64)
    if w.shape != (3,):
        raise ValueError("--weights は3つ指定してください: vit convnext mlpmixer")
    if np.any(w < 0):
        raise ValueError("--weights は非負で指定してください")
    s = float(w.sum())
    if s <= 0:
        raise ValueError("--weights の合計が0です")
    return w / s

def ensemble_probs(probs_vit, probs_conv, probs_mixer, weights):
    w = normalize_weights(weights)
    return (w[0] * probs_vit) + (w[1] * probs_conv) + (w[2] * probs_mixer)

def ensemble_logits(logits_vit, logits_conv, logits_mixer, weights):
    w = normalize_weights(weights)
    logits = (w[0] * logits_vit) + (w[1] * logits_conv) + (w[2] * logits_mixer)
    return softmax_np(logits, axis=1)[0]

def ensure_gradcam_subdirs(save_dir: str):
    """
    Grad-CAM/Attention を保存するサブフォルダを作って返す。
    - save_dir/gradcam/vit
    - save_dir/gradcam/convnextv2
    """
    grad_root = os.path.join(save_dir, "gradcam")
    vit_dir = os.path.join(grad_root, "vit")
    conv_dir = os.path.join(grad_root, "convnextv2")
    os.makedirs(vit_dir, exist_ok=True)
    os.makedirs(conv_dir, exist_ok=True)
    return vit_dir, conv_dir

# --------------------------------------------------
# 可視化：棒グラフ（保存のみ）
# --------------------------------------------------
def _pred_title(name, probs):
    pred_id = int(np.argmax(probs))
    conf = float(probs[pred_id])
    return f"{name}\n{ID2LABEL[pred_id]}  {conf*100:.1f}%"

def save_simple_viz(save_dir, base_name, resized_img,
                    probs_vit, probs_conv, probs_mixer, probs_ens):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{base_name}_ensemble_probs.png")

    labels = LABEL_NAMES
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    axes[0].imshow(resized_img)
    axes[0].set_title("Input (480x480)")
    axes[0].axis("off")

    axes[1].bar(labels, probs_vit);   axes[1].set_ylim(0, 1); axes[1].set_title(_pred_title("ViT", probs_vit))
    axes[2].bar(labels, probs_conv);  axes[2].set_ylim(0, 1); axes[2].set_title(_pred_title("ConvNeXtV2", probs_conv))
    axes[3].bar(labels, probs_mixer); axes[3].set_ylim(0, 1); axes[3].set_title(_pred_title("MLP-Mixer", probs_mixer))
    axes[4].bar(labels, probs_ens);   axes[4].set_ylim(0, 1); axes[4].set_title(_pred_title("Ensemble", probs_ens))

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved visualization: {out_path}")

# --------------------------------------------------
# JSON / CSV
# --------------------------------------------------
def build_result_dict(args, image_path, sizes_dict,
                      vit_parent_used, mixer_ckpt_used,
                      probs_vit, probs_conv, probs_mixer, probs_ens):
    w_norm = normalize_weights(args.weights)
    pred_id = int(np.argmax(probs_ens))
    conf = float(probs_ens[pred_id])

    return {
        "timestamp_jst": now_jst_iso(),
        "image_path": image_path,
        "labels": LABEL_NAMES,
        "image_size_policy": f"{args.square_mode}_to_square_if_needed_then_resize_480",
        "square_mode": args.square_mode,
        "sizes": sizes_dict,
        "device": str(args.device),
        "checkpoints": {
            "vit_checkpoint": args.vit_checkpoint,
            "vit_parent_path": vit_parent_used,
            "convnext_checkpoint": args.convnext_checkpoint,
            "mlpmixer_checkpoint": mixer_ckpt_used,
        },
        "tta": {
            "vit": {"tta": int(args.vit_tta), "magnitude": int(args.vit_tta_magnitude)},
            "convnext": {"tta": int(args.conv_tta), "magnitude": int(args.conv_tta_magnitude)},
            "mlpmixer": {"tta": int(args.mixer_tta), "magnitude": int(args.mixer_tta_magnitude)},
        },
        "ensemble": {
            "mode": str(args.ensemble_mode),
            "weights_raw": list(map(float, args.weights)),
            "weights_normalized": [float(x) for x in w_norm],
            "mixer_temperature": float(args.mixer_temperature),
        },
        "probs": {
            "vit": {"tako": float(probs_vit[0]), "AMI": float(probs_vit[1])},
            "convnextv2": {"tako": float(probs_conv[0]), "AMI": float(probs_conv[1])},
            "mlpmixer": {"tako": float(probs_mixer[0]), "AMI": float(probs_mixer[1])},
            "ensemble": {"tako": float(probs_ens[0]), "AMI": float(probs_ens[1])},
        },
        "prediction": {
            "predicted_id": pred_id,
            "predicted_label": ID2LABEL[pred_id],
            "confidence": conf,
        },
    }

def save_json_result(save_dir, base_name, result_dict):
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, f"{base_name}_ensemble_preds.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved JSON: {out_path}")

def write_one_line_csv(csv_path, result_dict):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    header = [
        "timestamp_jst","image_path","device",
        "vit_checkpoint","vit_parent_path","convnext_checkpoint","mlpmixer_checkpoint",
        "square_mode","orig_w","orig_h","square_side","resized_w","resized_h",
        "vit_tta","vit_mag","conv_tta","conv_mag","mixer_tta","mixer_mag",
        "ensemble_mode","mixer_temperature",
        "w_vit_raw","w_conv_raw","w_mixer_raw","w_vit","w_conv","w_mixer",
        "vit_prob_tako","vit_prob_AMI","conv_prob_tako","conv_prob_AMI",
        "mixer_prob_tako","mixer_prob_AMI","ens_prob_tako","ens_prob_AMI",
        "ens_predicted_id","ens_predicted_label","ens_confidence",
    ]

    ens_meta = result_dict.get("ensemble", {})
    w_norm = ens_meta.get("weights_normalized", [None, None, None])
    w_raw = ens_meta.get("weights_raw", [None, None, None])

    sizes = result_dict["sizes"]
    ckpt = result_dict["checkpoints"]
    tta = result_dict["tta"]
    probs = result_dict["probs"]
    pred = result_dict["prediction"]

    row = {
        "timestamp_jst": result_dict["timestamp_jst"],
        "image_path": result_dict["image_path"],
        "device": result_dict["device"],

        "vit_checkpoint": ckpt["vit_checkpoint"],
        "vit_parent_path": ckpt["vit_parent_path"],
        "convnext_checkpoint": ckpt["convnext_checkpoint"],
        "mlpmixer_checkpoint": ckpt["mlpmixer_checkpoint"],

        "square_mode": result_dict.get("square_mode", ""),
        "orig_w": sizes["original"][0],
        "orig_h": sizes["original"][1],
        "square_side": max(sizes["square"][0], sizes["square"][1]),
        "resized_w": sizes["resized"][0],
        "resized_h": sizes["resized"][1],

        "vit_tta": tta["vit"]["tta"],
        "vit_mag": tta["vit"]["magnitude"],
        "conv_tta": tta["convnext"]["tta"],
        "conv_mag": tta["convnext"]["magnitude"],
        "mixer_tta": tta["mlpmixer"]["tta"],
        "mixer_mag": tta["mlpmixer"]["magnitude"],

        "ensemble_mode": ens_meta.get("mode", ""),
        "mixer_temperature": ens_meta.get("mixer_temperature", ""),

        "w_vit_raw": w_raw[0],
        "w_conv_raw": w_raw[1],
        "w_mixer_raw": w_raw[2],
        "w_vit": w_norm[0],
        "w_conv": w_norm[1],
        "w_mixer": w_norm[2],

        "vit_prob_tako": probs["vit"]["tako"],
        "vit_prob_AMI": probs["vit"]["AMI"],
        "conv_prob_tako": probs["convnextv2"]["tako"],
        "conv_prob_AMI": probs["convnextv2"]["AMI"],
        "mixer_prob_tako": probs["mlpmixer"]["tako"],
        "mixer_prob_AMI": probs["mlpmixer"]["AMI"],
        "ens_prob_tako": probs["ensemble"]["tako"],
        "ens_prob_AMI": probs["ensemble"]["AMI"],

        "ens_predicted_id": pred["predicted_id"],
        "ens_predicted_label": pred["predicted_label"],
        "ens_confidence": pred["confidence"],
    }

    file_exists = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

# --------------------------------------------------
# ViT Attention / Grad-CAM（保存のみ）
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
    if g * g != cls_attention.shape[0]:
        raise ValueError(f"num_patches not square: {cls_attention.shape[0]}")

    att_small = cls_attention.reshape(g, g)
    att_map = cv2.resize(att_small, (image_size, image_size), interpolation=cv2.INTER_CUBIC)
    att_map = (att_map - att_map.min()) / (att_map.max() - att_map.min() + 1e-8)
    return att_map

def _patch_start_index(num_tokens: int) -> int:
    if int(math.isqrt(num_tokens - 1)) ** 2 == num_tokens - 1:
        return 1
    if int(math.isqrt(num_tokens - 2)) ** 2 == num_tokens - 2:
        return 2
    return 1

def vit_gradcam_hidden(model, pixel_values, target_class, last_k_layers: int = 4):
    """
    ViT-Grad-CAM（Transformerブロックの layernorm_before 出力にhook）
    """
    model.eval()
    model.zero_grad(set_to_none=True)

    layers = model.vit.encoder.layer
    k = min(last_k_layers, len(layers))
    take_ids = list(range(len(layers) - k, len(layers)))

    feats = []
    handles = []

    def _hook(module, inp, out):
        feats.append(out)  # (B,N,C)

    for i in take_ids:
        handles.append(layers[i].layernorm_before.register_forward_hook(_hook))

    # forward（勾配が必要なので no_grad は使わない）
    out = model(pixel_values, return_dict=True)

    for h in handles:
        h.remove()

    score = out.logits[:, target_class].sum()
    grads = torch.autograd.grad(
        outputs=score,
        inputs=feats,
        retain_graph=False,
        create_graph=False
    )

    cams = []
    for f, g in zip(feats, grads):
        N = f.shape[1]
        start = _patch_start_index(N)
        g_p = g[0, start:, :]              # (num_patches, C)
        f_p = f.detach()[0, start:, :]     # (num_patches, C)

        weights = g_p.mean(dim=0)          # (C,)
        cam_tokens = torch.relu((f_p * weights).sum(dim=-1))  # (num_patches,)

        num = cam_tokens.numel()
        gsize = int(math.isqrt(num))
        if gsize * gsize != num:
            raise ValueError(f"patch tokens not square: {num}")
        cams.append(cam_tokens.reshape(gsize, gsize))

    cam = torch.stack(cams, dim=0).mean(0).detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam

def save_vit_attn_and_gradcam_png(
    model, processor,
    original_image, squared_image, resized_image,
    device,
    probs_vit,
    image_path: str,
    save_dir: str,
    cmap: str = "jet",
):
    """
    1枚について ViTの
      - Attention map
      - ViT-Grad-CAM
      - overlay
    をまとめてPNG保存（表示なし）
    """
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    save_path = os.path.join(save_dir, f"{base_name}_vit_attn_gradcam.png")

    # attentions を取るため eager 指定（環境によって必要）
    if hasattr(model, "set_attn_implementation"):
        try:
            model.set_attn_implementation("eager")
        except Exception:
            pass
    elif hasattr(model, "config") and hasattr(model.config, "attn_implementation"):
        model.config.attn_implementation = "eager"

    model.eval()
    inputs = processor(images=resized_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # ★(改善1) attentions が None になりにくいよう、config も一時的に True にする
    old_cfg_attn = None
    if hasattr(model, "config"):
        old_cfg_attn = getattr(model.config, "output_attentions", None)
        try:
            model.config.output_attentions = True
        except Exception:
            old_cfg_attn = None

    try:
        with torch.no_grad():
            outputs = model(**inputs, output_attentions=True, return_dict=True)
            attentions = outputs.attentions
    finally:
        if old_cfg_attn is not None and hasattr(model, "config"):
            try:
                model.config.output_attentions = old_cfg_attn
            except Exception:
                pass

    if attentions is None:
        raise RuntimeError("ViT attentions is None (attn_implementation/config 設定を確認).")

    pred_id = int(np.argmax(probs_vit))
    pred_prob = float(probs_vit[pred_id])
    pred_label = ID2LABEL[pred_id]

    attn_map = build_attention_map(attentions, image_size=IMAGE_SIZE)

    gradcam_small = vit_gradcam_hidden(
        model,
        inputs["pixel_values"],
        target_class=pred_id,
        last_k_layers=3,
    )
    gradcam_map = cv2.resize(gradcam_small, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
    gradcam_map = (gradcam_map - gradcam_map.min()) / (gradcam_map.max() - gradcam_map.min() + 1e-8)

    orig_w, orig_h = original_image.size
    is_square = (orig_w == orig_h)

    title_pred = f"Original (pred: {pred_label} {pred_prob*100:.1f}%)"
    title_attn = "Attention Map (CLS→patch, last, head-avg)"
    title_grad = f"ViT-Grad-CAM (class={pred_label}, p={pred_prob:.3f})"
    title_ovl  = f"Overlay (480x480, p={pred_prob:.3f})"

    if is_square:
        fig, axes = plt.subplots(1, 4, figsize=(22, 5))
        ax1, ax2, ax3, ax4 = axes
        ax1.imshow(original_image); ax1.set_title(title_pred); ax1.axis("off")

        im2 = ax2.imshow(attn_map, cmap=cmap, alpha=0.9, vmin=0, vmax=1)
        ax2.set_title(title_attn); ax2.axis("off")
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

        im3 = ax3.imshow(gradcam_map, cmap=cmap, alpha=0.9, vmin=0, vmax=1)
        ax3.set_title(title_grad); ax3.axis("off")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        ax4.imshow(resized_image)
        im4 = ax4.imshow(gradcam_map, cmap=cmap, alpha=0.6, vmin=0, vmax=1)
        ax4.set_title(title_ovl); ax4.axis("off")
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    else:
        fig, axes = plt.subplots(1, 5, figsize=(26, 5))
        ax1, ax2, ax3, ax4, ax5 = axes
        ax1.imshow(original_image); ax1.set_title(title_pred); ax1.axis("off")
        ax2.imshow(squared_image);  ax2.set_title("Padded/Cropped Image"); ax2.axis("off")

        im3 = ax3.imshow(attn_map, cmap=cmap, alpha=0.9, vmin=0, vmax=1)
        ax3.set_title(title_attn); ax3.axis("off")
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

        im4 = ax4.imshow(gradcam_map, cmap=cmap, alpha=0.9, vmin=0, vmax=1)
        ax4.set_title(title_grad); ax4.axis("off")
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        ax5.imshow(resized_image)
        im5 = ax5.imshow(gradcam_map, cmap=cmap, alpha=0.6, vmin=0, vmax=1)
        ax5.set_title(title_ovl); ax5.axis("off")
        plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Saved ViT attn/gradcam: {save_path}")

# --------------------------------------------------
# ConvNeXtV2 Grad-CAM（保存のみ）
# --------------------------------------------------
def _resolve_convnext_target_layer(model, prefer="stages.3.blocks.2.conv_dw"):
    named = dict(model.named_modules())
    if prefer in named:
        return prefer

    cands = [k for k in named.keys() if ("conv_dw" in k or "dwconv" in k) and "stages.3.blocks" in k]
    if cands:
        return sorted(cands)[-1]

    cands = [k for k, m in named.items()
             if isinstance(m, torch.nn.Conv2d) and getattr(m, "groups", 1) == getattr(m, "in_channels", -1)]
    if cands:
        return sorted(cands)[-1]

    raise ValueError("No suitable dwconv layer found for Grad-CAM.")

def list_convnext_candidate_layers(model, stages_prefix=("stages.3", "stages.2")):
    names = []
    for name, module in model.named_modules():
        if any(name.startswith(p) for p in stages_prefix):
            if ("conv_dw" in name) or ("dwconv" in name):
                names.append(name)
    return names

class GradCAMConvNeXtV2:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = dict([*model.named_modules()])[target_layer]
        self.feature_maps = None
        self.gradients = None
        self.hook_handlers = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.feature_maps = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hook_handlers.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handlers.append(self.target_layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def generate(self, input_tensor, target_class=None):
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        loss = output[0, target_class]
        loss.backward()

        gradients = self.gradients
        feature_maps = self.feature_maps

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        grad_cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        grad_cam = torch.relu(grad_cam)

        grad_cam = F.interpolate(
            grad_cam, size=input_tensor.shape[2:], mode='bilinear', align_corners=False
        )
        grad_cam = grad_cam.squeeze().detach().cpu().numpy()
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min() + 1e-8)
        return grad_cam

def save_convnext_gradcam_png(
    model,
    original_image,
    squared_image,
    resized_image,
    device,
    probs_conv,
    image_path: str,
    save_dir: str,
    cmap: str = "jet",
    target_layers: list[str] | None = None,
):
    """
    ConvNeXtV2のGrad-CAMを指定層(複数可)で保存（表示なし）
    """
    os.makedirs(save_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    pred_id = int(np.argmax(probs_conv))
    pred_prob = float(probs_conv[pred_id])
    pred_label = ID2LABEL[pred_id]

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])
    x = tfm(resized_image).unsqueeze(0).to(device)

    if target_layers is None or len(target_layers) == 0:
        target_layers = [_resolve_convnext_target_layer(model)]

    for tl in target_layers:
        gradcam = GradCAMConvNeXtV2(model, target_layer=tl)
        try:
            cam = gradcam.generate(x, target_class=pred_id)
        finally:
            gradcam.remove_hooks()

        cam_resized = cv2.resize(cam, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)

        orig_w, orig_h = original_image.size
        is_square = (orig_w == orig_h)

        title_pred = f"Original (pred: {pred_label} {pred_prob*100:.1f}%)"
        title_cam = f"ConvNeXtV2 Grad-CAM (layer={tl}, class={pred_label}, p={pred_prob:.3f})"
        title_overlay = f"Overlay (layer={tl}, p={pred_prob:.3f})"

        if is_square:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            ax1, ax2, ax3 = axes
            ax1.imshow(original_image); ax1.set_title(title_pred); ax1.axis('off')

            im2 = ax2.imshow(cam_resized, cmap=cmap, alpha=0.9, vmin=0, vmax=1)
            ax2.set_title(title_cam); ax2.axis('off')
            plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

            ax3.imshow(resized_image)
            im3 = ax3.imshow(cam_resized, cmap=cmap, alpha=0.6, vmin=0, vmax=1)
            ax3.set_title(title_overlay); ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        else:
            fig, axes = plt.subplots(1, 4, figsize=(22, 5))
            ax1, ax2, ax3, ax4 = axes
            ax1.imshow(original_image); ax1.set_title(title_pred); ax1.axis('off')
            ax2.imshow(squared_image); ax2.set_title("Padded/Cropped Image"); ax2.axis('off')

            im3 = ax3.imshow(cam_resized, cmap=cmap, alpha=0.9, vmin=0, vmax=1)
            ax3.set_title(title_cam); ax3.axis('off')
            plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

            ax4.imshow(resized_image)
            im4 = ax4.imshow(cam_resized, cmap=cmap, alpha=0.6, vmin=0, vmax=1)
            ax4.set_title(title_overlay); ax4.axis('off')
            plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)

        plt.tight_layout()
        layer_tag = tl.replace(".", "_")
        save_path = os.path.join(save_dir, f"{base_name}_convnext_gradcam_{layer_tag}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[INFO] Saved ConvNeXt Grad-CAM: {save_path}")

# --------------------------------------------------
# main（ディレクトリ対応 + Grad-CAM保存）
# --------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Single-image or directory ensemble inference (ViT + ConvNeXtV2 + MLP-Mixer)")

    p.add_argument("input_path", type=str, help="画像ファイル or 画像ディレクトリ")
    p.add_argument("--recursive", action="store_true", help="ディレクトリ探索を再帰的にする")

    # model paths
    p.add_argument("--vit-checkpoint", type=str, required=True)
    p.add_argument("--vit-parent-path", type=str, default=None)
    p.add_argument("--convnext-checkpoint", type=str, required=True)
    p.add_argument("--mlpmixer-model-path", type=str, required=True)

    # device
    p.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])

    # weights
    p.add_argument("--weights", type=float, nargs=3, default=[1.0, 1.0, 1.0], metavar=("VIT", "CONV", "MIXER"))

    # TTA per model
    p.add_argument("--vit-tta", type=int, default=24)
    p.add_argument("--vit-tta-magnitude", type=int, default=1)
    p.add_argument("--conv-tta", type=int, default=1)
    p.add_argument("--conv-tta-magnitude", type=int, default=3)
    p.add_argument("--mixer-tta", type=int, default=32)
    p.add_argument("--mixer-tta-magnitude", type=int, default=3)

    # ensemble switches
    p.add_argument("--ensemble-mode", type=str, default="logits", choices=["probs", "logits"])
    p.add_argument("--mixer-temperature", type=float, default=1.0)

    # save
    p.add_argument("--square-mode", type=str, default="crop", choices=["pad", "crop"])
    p.add_argument("--save-dir", type=str, default=None, help="PNG/JSON/Grad-CAM/CSV を保存するディレクトリ")
    p.add_argument("--output-csv", type=str, default=None, help="CSV追記先（指定時は常にここ）")

    # feature map switches
    p.add_argument("--no-vit-maps", action="store_true", help="ViTのAttention/Grad-CAMを保存しない")
    p.add_argument("--no-convnext-gradcam", action="store_true", help="ConvNeXtV2のGrad-CAMを保存しない")
    p.add_argument("--cmap", type=str, default="jet", help="feature map用カラーマップ (jet, hot, viridis 等)")

    p.add_argument(
        "--conv-target-layers",
        type=str,
        default="auto",
        help=(
            "ConvNeXtV2 の Grad-CAM ターゲット層名。"
            "auto: 自動選択 / list: 候補を列挙して終了 / "
            "カンマ区切りで複数指定可"
        ),
    )

    args = p.parse_args()

    image_paths = collect_image_paths(args.input_path, recursive=args.recursive)
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No images found under: {args.input_path}")

    batch_mode = (len(image_paths) > 1) or os.path.isdir(args.input_path)

    # check model paths
    for path in [args.vit_checkpoint, args.convnext_checkpoint, args.mlpmixer_model_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(path)
    if args.vit_parent_path is not None and not os.path.exists(args.vit_parent_path):
        raise FileNotFoundError(args.vit_parent_path)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDAが利用できません。CPUを使用します。")
        args.device = "cpu"
    device = torch.device(args.device)

    # validate weights & temperature early
    w_norm = normalize_weights(args.weights)
    _ = apply_temperature_to_logits(np.zeros((1, len(LABEL_NAMES)), dtype=np.float32), args.mixer_temperature)

    print(f"[INFO] Device: {device}")
    print(f"[INFO] Found {len(image_paths)} image(s). batch_mode={batch_mode}")
    print(f"[INFO] Ensemble mode: {args.ensemble_mode}")
    print(f"[INFO] Mixer temperature: {args.mixer_temperature}")
    print(f"[INFO] Weights (normalized): vit={w_norm[0]:.3f}, conv={w_norm[1]:.3f}, mixer={w_norm[2]:.3f}")

    # load models once
    vit_model, vit_processor, vit_parent_used = load_vit(args.vit_checkpoint, args.vit_parent_path, device=device)
    conv_model = load_convnextv2(args.convnext_checkpoint, device=device)
    mixer_model, mixer_ckpt_used = load_mlpmixer(args.mlpmixer_model_path, device=device)

    # conv target layers option
    conv_target_layers_arg = args.conv_target_layers.strip()
    if conv_target_layers_arg == "list":
        cands = list_convnext_candidate_layers(conv_model)
        print("\n[INFO] ConvNeXtV2 Grad-CAM candidate layers:")
        for name in cands:
            print("  -", name)
        print("\n[INFO] Choose one (or more) and run with --conv-target-layers 'name1,name2,...'")
        return
    elif conv_target_layers_arg == "auto":
        conv_target_layers = None
    else:
        conv_target_layers = [s.strip() for s in conv_target_layers_arg.split(",") if s.strip()]

    # save-dir subdirs
    png_dir = json_dir = csv_dir = None
    if args.save_dir is not None:
        os.makedirs(args.save_dir, exist_ok=True)
        png_dir, json_dir, csv_dir = ensure_output_subdirs(args.save_dir)

    # CSV path:
    # - --output-csv があればそこへ（save-dir に関係なく）
    # - なければ save-dir/csv/ensemble_results.csv
    csv_path = None
    if args.output_csv is not None:
        csv_path = args.output_csv
    elif args.save_dir is not None:
        csv_path = os.path.join(csv_dir, "ensemble_results.csv")
        if batch_mode:
            print(f"[INFO] Batch CSV (default): {csv_path}")

    # Grad-CAM用サブフォルダはループ外で1回だけ作る
    vit_gc_dir = conv_gc_dir = None
    if args.save_dir is not None:
        vit_gc_dir, conv_gc_dir = ensure_gradcam_subdirs(args.save_dir)

    counts = {name: 0 for name in LABEL_NAMES}

    def _fmt(p_):
        return f"tako={p_[0]:.4f}, AMI={p_[1]:.4f}"

    for i, img_path in enumerate(image_paths, start=1):
        print("\n" + "="*80)
        print(f"[{i}/{len(image_paths)}] {img_path}")

        original, squared, resized = preprocess_image_480(img_path, square_mode=args.square_mode)
        ow, oh = original.size
        sw, sh = squared.size
        rw, rh = resized.size

        sizes_dict = {
            "original": [ow, oh],
            "square": [sw, sh],
            "resized": [rw, rh],
        }

        logits_vit = predict_vit_tta_logits(
            vit_model, vit_processor, resized, device,
            tta=args.vit_tta, magnitude=args.vit_tta_magnitude
        )
        logits_conv = predict_convnext_tta_logits(
            conv_model, resized, device,
            tta=args.conv_tta, magnitude=args.conv_tta_magnitude
        )
        logits_mixer = predict_mlpmixer_tta_logits(
            mixer_model, resized, device,
            tta=args.mixer_tta, magnitude=args.mixer_tta_magnitude
        )

        logits_mixer_scaled = apply_temperature_to_logits(logits_mixer, args.mixer_temperature)

        probs_vit = softmax_np(logits_vit, axis=1)[0]
        probs_conv = softmax_np(logits_conv, axis=1)[0]
        probs_mixer = softmax_np(logits_mixer_scaled, axis=1)[0]

        if args.ensemble_mode == "probs":
            probs_ens = ensemble_probs(probs_vit, probs_conv, probs_mixer, args.weights)
        else:
            probs_ens = ensemble_logits(logits_vit, logits_conv, logits_mixer_scaled, args.weights)

        pred_id = int(np.argmax(probs_ens))
        pred_label = ID2LABEL[pred_id]
        conf = float(probs_ens[pred_id])

        print("Per-model probabilities:")
        print(f"  ViT        : {_fmt(probs_vit)}")
        print(f"  ConvNeXtV2 : {_fmt(probs_conv)}")
        print(f"  MLP-Mixer  : {_fmt(probs_mixer)}")
        print(f"Ensemble (mode={args.ensemble_mode}): {_fmt(probs_ens)}")
        print(f"Pred: {pred_label} ({pred_id})  confidence={conf:.4f}")

        result_dict = build_result_dict(
            args=args,
            image_path=img_path,
            sizes_dict=sizes_dict,
            vit_parent_used=vit_parent_used,
            mixer_ckpt_used=mixer_ckpt_used,
            probs_vit=probs_vit,
            probs_conv=probs_conv,
            probs_mixer=probs_mixer,
            probs_ens=probs_ens,
        )

        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # save PNG/JSON（サブディレクトリへ）
        if args.save_dir is not None:
            save_simple_viz(png_dir, base_name, resized, probs_vit, probs_conv, probs_mixer, probs_ens)
            save_json_result(json_dir, base_name, result_dict)

        # Grad-CAM / Attn 保存（表示なし）
        if args.save_dir is None:
            if (not args.no_vit_maps) or (not args.no_convnext_gradcam):
                print("[WARN] --save-dir が無いので Grad-CAM/Attention は保存できません。")
        else:
            # ViT maps -> save-dir/gradcam/vit
            if not args.no_vit_maps:
                try:
                    save_vit_attn_and_gradcam_png(
                        vit_model, vit_processor,
                        original, squared, resized,
                        device,
                        probs_vit,
                        img_path,
                        save_dir=vit_gc_dir,
                        cmap=args.cmap,
                    )
                except Exception as e:
                    print(f"[WARN] ViT maps failed: {e}")

            # ConvNeXt Grad-CAM -> save-dir/gradcam/convnextv2
            if not args.no_convnext_gradcam:
                try:
                    save_convnext_gradcam_png(
                        conv_model,
                        original, squared, resized,
                        device,
                        probs_conv,
                        img_path,
                        save_dir=conv_gc_dir,
                        cmap=args.cmap,
                        target_layers=conv_target_layers,
                    )
                except Exception as e:
                    print(f"[WARN] ConvNeXt Grad-CAM failed: {e}")

        # CSV（指定があればそこへ、なければ save-dir/csv/ensemble_results.csv）
        if csv_path is not None:
            write_one_line_csv(csv_path, result_dict)

        counts[pred_label] += 1

    print("\n" + "#"*80)
    print(f"Done. {now_jst_iso()}")
    print(f"Total images: {len(image_paths)}")
    print("Prediction counts:")
    for k in LABEL_NAMES:
        print(f"  {k}: {counts[k]}")

if __name__ == "__main__":
    main()

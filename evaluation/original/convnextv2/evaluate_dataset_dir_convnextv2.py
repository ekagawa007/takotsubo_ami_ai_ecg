
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ConvNeXt V2 (timm) 2-class 評価スクリプト（tako / AMI）
- テストディレクトリを走査して推論
- ROC / PR / 混同行列 / Calibration（ECE, Brier, BSS）
- metrics.csv / metrics.json / predictions.csv を保存
- ViT版 evaluate_dataset_dir_ViT.py と引数・出力を可能な限り共通化
- Test-Time Augmentation (TTA) 対応

Version 1.0.0 (2025-11-22) first release

sample usage:
python evaluate_dataset_dir_convnextv2.py \\
  --model-path /path/to/ckpt_or_dir \\        # .pth 直指定 or .pth を含むディレクトリ
  --test-dir /path/to/test \\
  --output-dir /path/to/out \\
  --device cuda \\
  --batch-size 1 \\
  --calibration-bins 10 \\
  --tta 8 --tta-magnitude 3 --tta-mode mild\\
  --bf16                                 # （任意）CUDAでbfloat16を使う


"""

import os
import sys
import argparse
import datetime
import json
import csv
from collections import OrderedDict

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode, RandAugment
from PIL import Image

import timm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.calibration import calibration_curve

# --------------------------------------------------
# 定数（ViT版と合わせる）
# --------------------------------------------------
LABEL_NAMES = ["tako", "AMI"]
LABEL2ID = {n: i for i, n in enumerate(LABEL_NAMES)}
ID2LABEL = {i: n for n, i in LABEL2ID.items()}
IMAGE_SIZE = 480

# --------------------------------------------------
# データ拡張（ECGなので Flip を除外）
# --------------------------------------------------
class RandAugmentNoFlip(RandAugment):
    def __init__(self, num_ops=2, magnitude=9, num_magnitude_bins=10, interpolation=None, fill=None):
        super().__init__(num_ops, magnitude, num_magnitude_bins, interpolation, fill)
        modules = list(self._modules.items())
        filtered = OrderedDict(
            (name, module)
            for name, module in modules
            if "Flip" not in module.__class__.__name__
        )
        self._modules = filtered

def build_identity_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

def build_tta_transform(mode: str, magnitude: int):
    if mode == "randaug":
        return transforms.Compose([
            RandAugmentNoFlip(num_ops=2, magnitude=magnitude,
                              interpolation=InterpolationMode.BILINEAR, fill=0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    elif mode == "mild":
        return transforms.Compose([
            transforms.RandomAffine(
                degrees=0, translate=(0.01, 0.01), scale=(0.98, 1.02),
                interpolation=InterpolationMode.BILINEAR, fill=0
            ),
            transforms.ColorJitter(brightness=0.03, contrast=0.03, saturation=0.0, hue=0.0),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
        ])
    else:
        # デフォルトはidentity（TTAなしの1回目）
        return build_identity_transform()


# --------------------------------------------------
# Dataset
# --------------------------------------------------
class PatientDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        for patient_dir in sorted(os.listdir(root_dir)):
            p_path = os.path.join(root_dir, patient_dir)
            if not os.path.isdir(p_path):
                continue
            for label_dir in sorted(os.listdir(p_path)):
                if label_dir == "misc":
                    continue
                l_path = os.path.join(p_path, label_dir)
                if not os.path.isdir(l_path):
                    continue
                if   label_dir == "tako": y = LABEL2ID["tako"]
                elif label_dir == "AMI":  y = LABEL2ID["AMI"]
                else:
                    continue
                for f in sorted(os.listdir(l_path)):
                    if f.lower().endswith(".png"):
                        self.samples.append((os.path.join(l_path, f), y))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        img_path, y = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        x = self.transform(img) if self.transform is not None else img
        return {"pixel_values": x, "labels": torch.tensor(y, dtype=torch.long), "path": img_path}

def collate_fn(batch):
    xs = torch.stack([b["pixel_values"] for b in batch], dim=0)
    ys = torch.stack([b["labels"] for b in batch], dim=0)
    paths = [b["path"] for b in batch]
    return {"pixel_values": xs, "labels": ys, "path": paths}

# --------------------------------------------------
# 補助関数（ViT版と揃える）
# --------------------------------------------------
def softmax_np(logits, axis=1):
    z = logits - logits.max(axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=axis, keepdims=True)

def safe_roc_pr_auc(y_true_bin, y_score):
    pos = int(y_true_bin.sum())
    n = len(y_true_bin)
    if pos == 0 or pos == n:
        return None, None, None, None, np.nan, np.nan
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
    return fpr, tpr, precision, recall, auc(fpr, tpr), auc(recall, precision)

def safe_calibration_curve(y_true_bin, y_prob, n_bins=10, strategy="uniform"):
    pos = y_true_bin.sum()
    if pos == 0 or pos == len(y_true_bin):
        return None, None
    return calibration_curve(y_true_bin, y_prob, n_bins=n_bins, strategy=strategy)

def save_calib_csv(prefix, y_true_bin, y_prob, mean_pred, frac_pos, out_dir, n_bins):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, edges[1:-1], right=False)  # 0..n_bins-1
    counts = np.array([np.sum(bin_ids == i) for i in range(n_bins)])
    non_empty = np.where(counts > 0)[0]

    total = len(y_true_bin)
    weights = counts[non_empty] / total
    abs_gap = np.abs(np.array(frac_pos) - np.array(mean_pred))
    ece = float(np.sum(weights * abs_gap))

    csv_path = os.path.join(out_dir, f"calibration_{prefix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["bin", "pred_mean", "true_frac", "count", "abs_gap", "weight", "bin_left", "bin_right"])
        for i, b in enumerate(non_empty):
            w.writerow([
                int(b),
                float(mean_pred[i]),
                float(frac_pos[i]),
                int(counts[b]),
                float(abs_gap[i]),
                float(weights[i]),
                float(edges[b]),
                float(edges[b+1]),
            ])
        w.writerow([])
        w.writerow(["ECE", ece])
    print(f"[Calibration] Saved CSV: {csv_path} (ECE={ece:.6f})")
    return ece

def load_convnextv2(model_path_or_file, num_classes=2, device="cpu"):
    """
    timm ConvNeXt V2 モデルを作成し、.pth（state_dict or {'model': ...}）をロード。
    引数はファイルでもディレクトリでもOK。ディレクトリの場合は *.pth を探索。
    """
    model = timm.create_model("convnextv2_base", pretrained=False, num_classes=num_classes)

    # パス解決
    path = model_path_or_file
    if os.path.isdir(path):
        # 最も新しそうな *.pth を拾う
        cands = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".pth")]
        if not cands:
            raise FileNotFoundError(f"No .pth found under: {path}")
        path = sorted(cands)[-1]
        print(f"[INFO] Using checkpoint file: {path}")
    else:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")

    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    # 互換のため prefix を剥ぐ
    new_state = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod.") :]
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    if missing:
        print("[WARN] Missing keys (ok):", list(missing)[:8], "...")
    if unexpected:
        print("[WARN] Unexpected keys:", list(unexpected)[:8], "...")

    model.to(device).eval()
    return model

def _normalize_output_dir(output_dir: str, model_path: str, tta: int, tta_mag: int) -> str:
    """
    ユーザーが --output-dir に .pth ファイルを渡しても OK なように正規化する。
    - output_dir が .pth で終わる"ファイルパス"なら、その”横”に
        <.pthファイル名>.predictions_roc_pr[_tta_<n>_<m>] という新ディレクトリ名を作る
    - それ以外はそのまま返す
    """
    def make_suffix():
        return f".predictions_roc_pr_tta_{tta}_{tta_mag}" if tta > 1 else ".predictions_roc_pr"

    # .pthを渡されたら「横に」ディレクトリを作る（/path/ckpt.pth.predictions_...）
    if output_dir.lower().endswith(".pth") and os.path.isfile(output_dir):
        return output_dir + make_suffix()

    # “誤って” .pth/xxx のようにファイル配下を指していた場合も救済
    if output_dir.lower().endswith(".pth") and not os.path.isdir(output_dir):
        # ここに来るのは output_dir が存在しない .pth 文字列のケース
        # （基本は上の分岐で十分。万一でも安全側へ）
        return output_dir + make_suffix()

    # それ以外はそのまま
    return output_dir

# --------------------------------------------------
# Main evaluation (ViT版と引数共通化)
# --------------------------------------------------
def evaluate_model(model_path, test_dir, output_dir,
                   device="cpu", batch_size=1, calib_bins=10,
                   tta=1, tta_magnitude=5, num_workers=4, bf16=True, tta_mode="randaug"):
    # クリップ & バリデーション
    tta = max(1, int(tta))
    tta_magnitude = int(max(0, min(10, tta_magnitude)))

    print(f"[INFO] モデルパス: {model_path}")
    print(f"[INFO] テストデータ: {test_dir}")
    print(f"[INFO] デバイス: {device}")
    print(f"[INFO] Calibration bins: {calib_bins}")
    print(f"[INFO] TTA: {tta}")
    print(f"[INFO] TTA magnitude: {tta_magnitude}")

    # 出力パスの正規化（.pth を渡されてもOKにする）
    output_dir = _normalize_output_dir(output_dir, model_path, tta, tta_magnitude)
    print(f"[INFO] 出力ディレクトリ: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)

    # Dataset準備（transform は後で差し替え）
    test_ds = PatientDataset(test_dir, transform=None)
    print(f"[INFO] テストサンプル数: {len(test_ds)}")

    # モデル読み込み
    model = load_convnextv2(model_path, num_classes=len(LABEL_NAMES), device=device)

    use_cuda = (device == "cuda") and torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if (use_cuda and bf16) else torch.float32

    # ---------- まず identity ----------
    test_ds.transform = build_identity_transform()
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        pin_memory=True, drop_last=False, collate_fn=collate_fn
    )

    all_logits_list, all_labels_list, all_paths = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["pixel_values"].to(device, non_blocking=True)
            y = batch["labels"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
                logits = model(x)
            all_logits_list.append(logits.detach().cpu())
            all_labels_list.append(y.detach().cpu())
            all_paths.extend(batch["path"])

    logits_sum = torch.cat(all_logits_list, dim=0).float().numpy().astype(np.float64)
    labels = torch.cat(all_labels_list, dim=0).numpy()

    # ---------- 追加の TTA ----------
    if tta > 1:
        import random
        for t in range(tta - 1):
            seed = 1234 + t
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            test_ds.transform = build_tta_transform(tta_mode, tta_magnitude)
            test_loader = DataLoader(
                test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                pin_memory=True, drop_last=False, collate_fn=collate_fn
            )
            sub_logits_list = []
            with torch.no_grad():
                for batch in test_loader:
                    x = batch["pixel_values"].to(device, non_blocking=True)
                    with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
                        logits = model(x)
                    sub_logits_list.append(logits.detach().cpu())
            sub_logits = torch.cat(sub_logits_list, dim=0).float().numpy().astype(np.float64)
            assert sub_logits.shape == logits_sum.shape
            logits_sum += sub_logits

    logits = (logits_sum / float(tta)).astype(np.float32)
    preds = logits.argmax(axis=-1)
    probs = softmax_np(logits, axis=1)

    # --------------------------------------------------
    # 指標計算（ViT版と同じ構成）
    # --------------------------------------------------
    acc = accuracy_score(labels, preds)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0)
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0)
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0)

    cm = confusion_matrix(labels, preds, labels=list(range(len(LABEL_NAMES))))
    report = classification_report(
        labels, preds, target_names=LABEL_NAMES, output_dict=True, zero_division=0
    )

    # Specificity
    tn_for_0 = cm[1,1]; fp_for_0 = cm[1,0]
    tn_for_1 = cm[0,0]; fp_for_1 = cm[0,1]
    spec_0 = tn_for_0 / (tn_for_0 + fp_for_0) if (tn_for_0 + fp_for_0) > 0 else float("nan")
    spec_1 = tn_for_1 / (tn_for_1 + fp_for_1) if (tn_for_1 + fp_for_1) > 0 else float("nan")
    support0 = cm[0].sum()
    support1 = cm[1].sum()
    specificity_macro    = float(np.nanmean([spec_0, spec_1]))
    specificity_weighted = float(np.nansum([spec_0 * support0, spec_1 * support1]) / (support0 + support1))
    specificity_micro    = float(spec_1)  # AMI=positive と見做したTNR

    # ROC/PR（AMI=1 positive）
    y_true_ami  = labels
    y_score_ami = probs[:, 1]
    fpr, tpr, precision_ami, recall_ami, roc_auc, pr_auc_ami = safe_roc_pr_auc(y_true_ami, y_score_ami)

    # PR（tako=0 positive）
    y_true_tako  = (labels == 0).astype(int)
    y_score_tako = probs[:, 0]
    _, _, precision_tako, recall_tako, _, pr_auc_tako = safe_roc_pr_auc(y_true_tako, y_score_tako)

    # Calibration / ECE / Brier / BSS
    y_true_ami_bin = (labels == 1).astype(int)
    y_prob_ami = probs[:, 1]
    frac_pos_ami, mean_pred_ami = safe_calibration_curve(y_true_ami_bin, y_prob_ami, calib_bins)

    y_true_tako_bin = (labels == 0).astype(int)
    y_prob_tako = probs[:, 0]
    frac_pos_tako, mean_pred_tako = safe_calibration_curve(y_true_tako_bin, y_prob_tako, calib_bins)

    ece_ami = np.nan
    ece_tako = np.nan
    if frac_pos_ami is not None:
        ece_ami = save_calib_csv("AMI", y_true_ami_bin, y_prob_ami, mean_pred_ami, frac_pos_ami, output_dir, calib_bins)
    if frac_pos_tako is not None:
        ece_tako = save_calib_csv("tako", y_true_tako_bin, y_prob_tako, mean_pred_tako, frac_pos_tako, output_dir, calib_bins)

    brier_ami = float(np.mean((y_true_ami_bin  - y_prob_ami ) ** 2))
    brier_tako= float(np.mean((y_true_tako_bin - y_prob_tako) ** 2))

    base_p_ami  = y_true_ami_bin.mean()
    base_p_tako = y_true_tako_bin.mean()
    brier_base_ami  = float(np.mean((y_true_ami_bin  - base_p_ami ) ** 2))
    brier_base_tako = float(np.mean((y_true_tako_bin - base_p_tako) ** 2))
    bss_ami  = 1.0 - (brier_ami  / brier_base_ami)  if brier_base_ami  > 0 else np.nan
    bss_tako = 1.0 - (brier_tako / brier_base_tako) if brier_base_tako > 0 else np.nan

    # --------------------------------------------------
    # 出力
    # --------------------------------------------------
    print("\n" + "="*50)
    print("=== Test set evaluation (ConvNeXt V2) ===")
    print("="*50)
    print(f"Confusion matrix:\n{cm}")
    print(f"Accuracy (overall):      {acc:.4f}")
    print(f"Precision (macro):       {prec_macro:.4f}")
    print(f"Recall (macro):          {rec_macro:.4f}")
    print(f"Specificity (macro):     {specificity_macro:.4f}")
    print(f"F1 (macro):              {f1_macro:.4f}")
    print(f"Precision (micro):       {prec_micro:.4f}")
    print(f"Recall (micro):          {rec_micro:.4f}")
    print(f"Specificity (micro*):    {specificity_micro:.4f}")  # *AMI=positive
    print(f"F1 (micro):              {f1_micro:.4f}")
    print(f"Precision (weighted):    {prec_w:.4f}")
    print(f"Recall (weighted):       {rec_w:.4f}")
    print(f"Specificity (weighted):  {specificity_weighted:.4f}")
    print(f"F1 (weighted):           {f1_w:.4f}")
    print(f"Specificity per-class:   tako={spec_0:.4f}, AMI={spec_1:.4f}")

    print("\n=== Per-class metrics ===")
    for cls in LABEL_NAMES:
        print(f"{cls:10s}  Precision: {report[cls]['precision']:.4f}  "
              f"Recall: {report[cls]['recall']:.4f}  "
              f"F1: {report[cls]['f1-score']:.4f}")

    print("\n=== ROC/PR AUC ===")
    print("ROC AUC (AMI=pos): {:.4f}".format(roc_auc) if np.isfinite(roc_auc) else "ROC AUC (AMI=pos): n/a")
    print("PR  AUC (AMI=pos): {:.4f}".format(pr_auc_ami) if np.isfinite(pr_auc_ami) else "PR  AUC (AMI=pos): n/a")
    print("PR  AUC (tako=pos): {:.4f}".format(pr_auc_tako) if np.isfinite(pr_auc_tako) else "PR  AUC (tako=pos): n/a")

    print("\n=== Calibration metrics ===")
    print("ECE (AMI=positive):  {:.6f}".format(ece_ami) if np.isfinite(ece_ami) else "ECE (AMI=positive):  n/a")
    print("ECE (tako=positive): {:.6f}".format(ece_tako) if np.isfinite(ece_tako) else "ECE (tako=positive): n/a")
    print("Brier (AMI=positive):  {:.6f}".format(brier_ami))
    print("Brier (tako=positive): {:.6f}".format(brier_tako))
    print("Brier Skill Score (AMI=positive):  {:.6f}".format(bss_ami) if np.isfinite(bss_ami) else "BSS (AMI=positive):  n/a")
    print("Brier Skill Score (tako=positive): {:.6f}".format(bss_tako) if np.isfinite(bss_tako) else "BSS (tako=positive): n/a")

    # --- CSV: predictions ---
    pred_df = pd.DataFrame({
        "path": [p for p in all_paths],
        "true_id": [int(y) for y in labels],
        "true_label": [ID2LABEL[int(y)] for y in labels],
        "pred_id": [int(p) for p in preds],
        "pred_label": [ID2LABEL[int(p)] for p in preds],
        "logit_tako": logits[:, 0],
        "logit_AMI":  logits[:, 1],
        "prob_tako":  probs[:, 0],
        "prob_AMI":   probs[:, 1],
    })
    pred_csv_path = os.path.join(output_dir, "predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)

    # --- CSV: metrics ---
    metrics_df = pd.DataFrame({
        "metric": [
            "accuracy", "precision_macro", "recall_macro", "specificity_macro", "f1_macro",
            "roc_auc_AMIpos", "pr_auc_AMIpos", "pr_auc_takopos",
            "ECE_AMIpos", "ECE_takopos",
            "BSS_AMIpos", "BSS_takopos",
            "Brier_AMIpos", "Brier_takopos",
            "precision_micro", "recall_micro", "specificity_micro", "f1_micro",
            "precision_weighted", "recall_weighted", "specificity_weighted", "f1_weighted"
        ],
        "value": [
            acc, prec_macro, rec_macro, specificity_macro, f1_macro,
            roc_auc, pr_auc_ami, pr_auc_tako,
            float(ece_ami) if np.isfinite(ece_ami) else np.nan,
            float(ece_tako) if np.isfinite(ece_tako) else np.nan,
            bss_ami, bss_tako,
            brier_ami, brier_tako,
            prec_micro, rec_micro, specificity_micro, f1_micro,
            prec_w, rec_w, specificity_weighted, f1_w
        ]
    })
    metrics_csv_path = os.path.join(output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    # --- JSON: metrics（per-class含む） ---
    metrics_json = {
        "classes": LABEL_NAMES,
        "accuracy": acc,
        "precision_macro": prec_macro,
        "recall_macro": rec_macro,
        "specificity_macro": specificity_macro,
        "f1_macro": f1_macro,
        "precision_micro": prec_micro,
        "recall_micro": rec_micro,
        "specificity_micro": specificity_micro,
        "f1_micro": f1_micro,
        "precision_weighted": prec_w,
        "recall_weighted": rec_w,
        "specificity_weighted": specificity_weighted,
        "f1_weighted": f1_w,
        "per_class_specificity": {"tako": float(spec_0), "AMI": float(spec_1)},
        "roc_auc_AMIpos": roc_auc,
        "pr_auc_AMIpos": pr_auc_ami,
        "pr_auc_takopos": pr_auc_tako,
        "ECE_AMIpos": float(ece_ami) if np.isfinite(ece_ami) else None,
        "ECE_takopos": float(ece_tako) if np.isfinite(ece_tako) else None,
        "BSS_AMIpos": float(bss_ami) if np.isfinite(bss_ami) else None,
        "BSS_takopos": float(bss_tako) if np.isfinite(bss_tako) else None,
        "Brier_AMIpos": brier_ami,
        "Brier_takopos": brier_tako,
        "per_class": report,
        "confusion_matrix": cm.tolist(),
        "timestamp_jst": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9), "JST")).isoformat(),
    }
    metrics_json_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    # --- 可視化の保存 ---
    # CM
    plt.figure(figsize=(6, 5))
    sns.heatmap(np.array(cm), annot=True, fmt="d",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight'); plt.close()

    # ROC（AMI=pos）
    if np.isfinite(roc_auc):
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'ROC (AMI=pos) AUC = {roc_auc:.3f}', zorder=3, linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, zorder=2)
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (AMI=positive)')
        plt.legend(loc='lower right'); plt.grid(True, zorder=1)
        plt.xlim([0, 1]); plt.ylim([0, 1])
        roc_path = os.path.join(output_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight'); plt.close()

    # PR（AMI=pos）
    if np.isfinite(pr_auc_ami):
        plt.figure(figsize=(6, 6))
        plt.plot(recall_ami, precision_ami, label=f'PR (AMI=pos) AUC = {pr_auc_ami:.3f}', zorder=3, linewidth=2)
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title('Precision–Recall Curve (AMI=positive)')
        plt.legend(loc='lower left'); plt.grid(True, zorder=1)
        plt.xlim([0, 1]); plt.ylim([0, 1])
        pr_path = os.path.join(output_dir, "pr_curve_AMI.png")
        plt.savefig(pr_path, dpi=300, bbox_inches='tight'); plt.close()

    # PR（tako=pos）
    if np.isfinite(pr_auc_tako):
        plt.figure(figsize=(6, 6))
        plt.plot(recall_tako, precision_tako, label=f'PR (tako=pos) AUC = {pr_auc_tako:.3f}', zorder=3, linewidth=2)
        plt.xlabel('Recall'); plt.ylabel('Precision')
        plt.title('Precision–Recall Curve (tako=positive)')
        plt.legend(loc='lower left'); plt.grid(True, zorder=1)
        plt.xlim([0, 1]); plt.ylim([0, 1])
        pr_path = os.path.join(output_dir, "pr_curve_tako.png")
        plt.savefig(pr_path, dpi=300, bbox_inches='tight'); plt.close()

    # Calibration (AMI=pos)
    if frac_pos_ami is not None:
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "--", linewidth=1, label="Perfectly calibrated", alpha=0.7)
        plt.plot(mean_pred_ami, frac_pos_ami, marker="o", label="AMI (positive)")
        plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
        plt.title("Calibration plot (AMI=positive)")
        plt.grid(True, linestyle=":", linewidth=0.5); plt.legend(loc="upper left")
        plt.xlim([0, 1]); plt.ylim([0, 1]); plt.tight_layout()
        calib_ami_path = os.path.join(output_dir, "calibration_AMI.png")
        plt.savefig(calib_ami_path, dpi=150, bbox_inches="tight"); plt.close()

    # Calibration (tako=pos)
    if frac_pos_tako is not None:
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "--", linewidth=1, label="Perfectly calibrated", alpha=0.7)
        plt.plot(mean_pred_tako, frac_pos_tako, marker="o", label="tako (positive)")
        plt.xlabel("Predicted probability"); plt.ylabel("Observed frequency")
        plt.title("Calibration plot (tako=positive)")
        plt.grid(True, linestyle=":", linewidth=0.5); plt.legend(loc="upper left")
        plt.xlim([0, 1]); plt.ylim([0, 1]); plt.tight_layout()
        calib_tako_path = os.path.join(output_dir, "calibration_tako.png")
        plt.savefig(calib_tako_path, dpi=150, bbox_inches="tight"); plt.close()

    print("\n[INFO] 保存完了:")
    print("  -", pred_csv_path)
    print("  -", metrics_csv_path)
    print("  -", metrics_json_path)
    print("  -", cm_path)

    return {
        "accuracy": acc,
        "precision": prec_macro,
        "recall": rec_macro,
        "f1_score": f1_macro,
        "roc_auc": roc_auc,
        "pr_auc_ami": pr_auc_ami,
        "pr_auc_tako": pr_auc_tako,
        "ece_ami": ece_ami,
        "ece_tako": ece_tako,
        "brier_ami": brier_ami,
        "brier_tako": brier_tako,
        "bss_ami": bss_ami,
        "bss_tako": bss_tako,
        "confusion_matrix": cm,
        "specificity_macro": specificity_macro,
        "specificity_micro": specificity_micro,
        "specificity_weighted": specificity_weighted
    }

def main():
    parser = argparse.ArgumentParser(
        description="ConvNeXt V2 2-class evaluate (tako/AMI)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=r"""
使用例（ViT版と同等の引数）:
  python evaluate_dataset_dir_convnextv2.py \
    --model-path /path/to/ckpt_or_dir \
    --test-dir /path/to/test \
    --output-dir /path/to/out \
    --device cuda --batch-size 1 \
    --calibration-bins 10 \
    --tta 8 --tta-magnitude 5
"""
    )

    parser.add_argument("--model-path", type=str, required=True, help="ConvNeXt V2 の .pth もしくはディレクトリ")
    parser.add_argument("--test-dir", type=str, required=True, help="テストデータセットのディレクトリ")
    parser.add_argument("--output-dir", type=str, required=True, help="結果出力ディレクトリ")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="使用デバイス (default: cpu)")
    parser.add_argument("--batch-size", type=int, default=1, help="バッチサイズ (default: 1)")
    parser.add_argument("--calibration-bins", type=int, default=10, help="Calibration curveのビン数 (default: 10)")
    parser.add_argument("--tta", type=int, default=1, help="TTA の試行回数。1ならTTAなし（identityのみ）")
    parser.add_argument("--tta-magnitude", type=int, default=5, help="TTA時 RandAugment のmagnitude (0-10)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--bf16", action="store_true", help="(CUDA) bfloat16 で推論する")
    parser.add_argument("--tta-mode", type=str, default="randaug", choices=["randaug", "mild"], help="TTAの変換タイプ（randaug / mild）")


    args = parser.parse_args()

    if not os.path.exists(args.model_path):
        print(f"[ERROR] モデルパスが存在しません: {args.model_path}")
        sys.exit(1)
    if not os.path.exists(args.test_dir):
        print(f"[ERROR] テストデータディレクトリが存在しません: {args.test_dir}")
        sys.exit(1)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDAが利用できません。CPUを使用します。")
        args.device = "cpu"

    try:
        _ = evaluate_model(
            model_path=args.model_path,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            calib_bins=args.calibration_bins,
            tta=args.tta,
            tta_magnitude=args.tta_magnitude,
            num_workers=args.num_workers,
            bf16=bool(args.bf16),
            tta_mode=args.tta_mode,
        )
    except Exception as e:
        print(f"[ERROR] 評価中にエラーが発生しました: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    print(f"\n[INFO] 完了時刻: {datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9), 'JST'))}")

if __name__ == "__main__":
    main()

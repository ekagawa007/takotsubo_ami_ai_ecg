#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ViTモデルの性能評価スクリプト
- 2クラス分類（tako / AMI）
- テストデータセットでの性能指標計算
- ROC曲線、PR曲線、混同行列の可視化
- Calibration curve, ECE, Brier score, Brier Skill Score
- 結果をCSV/JSONファイルで保存
- TTA対応

version 1.1.0 (2025-01-27) Calibration, ECE, Brier score, BSS を追加
version 1.0.3 (2025-01-27) predictions.csvの出力フォーマットを修正
version 1.0.2 (2025-01-27) デフォルトバッチサイズを1に変更
version 1.0.1 (2025-10-27) predictionのcsvファイルのヘッダを修正
version 1.1.1 (2025-11-18) TTA 対応
version 1.1.2 (2026-05-18) --device cuda, no_cudaでおちるところを修正

TTAなし
python evaluate_dataset_dir_ViT.py \
  --model-path /path/to/checkpoint \
  --test-dir /path/to/test \
  --output-dir /path/to/out \
  --device cuda --batch-size 1

TTAあり
python evaluate_dataset_dir_ViT.py \
  --model-path /path/to/checkpoint \
  --test-dir /path/to/test \
  --output-dir /path/to/out \
  --device cuda --batch-size 1 \
  --tta 8 --tta-magnitude 5


基本的な使用例（バッチサイズ1、CPU使用）
python evaluate_dataset_dir_ViT.py \\
    --model-path /path/to/vit/checkpoint \\
    --test-dir /path/to/test/dataset \\
    --output-dir /path/to/output

GPU使用、バッチサイズ変更（メモリに余裕がある場合）
python evaluate_dataset_dir_ViT.py \\
    --model-path /path/to/vit/checkpoint \\
    --test-dir /path/to/test/dataset \\
    --output-dir /path/to/output \\
    --device cuda \\
    --batch-size 1 \\
    --calibration-bins 10
"""

import os
os.environ["WANDB_DISABLED"] = "true"
import sys
import argparse
import datetime
import json
import csv
from collections import OrderedDict

import torch
import PIL
from transformers import (AutoModelForImageClassification,
                          AutoImageProcessor,
                          TrainingArguments, Trainer)
from sklearn.metrics import (accuracy_score,
                             precision_recall_fscore_support,
                             confusion_matrix, classification_report, 
                             roc_curve, auc, precision_recall_curve)
from sklearn.calibration import calibration_curve
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms import RandAugment
from scipy.special import softmax
import pandas as pd

# --------------------------------------------------
# 定数
# --------------------------------------------------
LABEL_NAMES = ["tako", "AMI"]
LABEL2ID = {n: i for i, n in enumerate(LABEL_NAMES)}
ID2LABEL = {i: n for n, i in LABEL2ID.items()}
IMAGE_SIZE = 480

# --------------------------------------------------
# データ拡張設定（ECG用にflipを除外）
# --------------------------------------------------
class RandAugmentNoFlip(RandAugment):
    def __init__(self, num_ops=2, magnitude=9, num_magnitude_bins=10, interpolation=None, fill=None):
        super().__init__(num_ops, magnitude, num_magnitude_bins, interpolation, fill)
        # ① 子モジュール一覧 self._modules を OrderedDict に取り出し
        modules = list(self._modules.items())
        # ② "Flip" を含むクラス名のモジュールだけ除外
        filtered = OrderedDict(
            (name, module)
            for name, module in modules
            if "Flip" not in module.__class__.__name__
        )
        # ③ 書き戻し
        self._modules = filtered

# --------------------------------------------------
# データセットクラス
# --------------------------------------------------
class PatientDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        self.samples = []
        self.is_train = is_train
        
        # 患者ディレクトリを走査（ソートして順序を固定）
        for patient_dir in sorted(os.listdir(root_dir)):
            patient_path = os.path.join(root_dir, patient_dir)
            if not os.path.isdir(patient_path):
                continue

            # ラベルディレクトリを走査（ソートして順序を固定）
            for label_dir in sorted(os.listdir(patient_path)):

                # "misc" ディレクトリはスキップ
                if label_dir == "misc":
                    continue

                label_path = os.path.join(patient_path, label_dir)
                if not os.path.isdir(label_path):
                    continue

                # "tako"と"AMI"ディレクトリのみを対象
                if label_dir == "tako":
                    label = LABEL2ID["tako"]
                elif label_dir == "AMI":
                    label = LABEL2ID["AMI"]
                else:
                    continue

                # 画像ファイルを取得（PNG形式のみ, ソートして順序を固定）
                for img_file in sorted(os.listdir(label_path)):
                    if img_file.lower().endswith('.png'):
                        img_path = os.path.join(label_path, img_file)
                        self.samples.append((img_path, label))

        # データ拡張の設定
        if is_train:
            self.transform = transforms.Compose([
                RandAugmentNoFlip(
                    num_ops=2,
                    magnitude=9,
                    interpolation=PIL.Image.BILINEAR,
                    fill=0,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = PIL.Image.open(img_path).convert('RGB')
        img = self.transform(img)

        return {
            "pixel_values": img,
            "labels": torch.tensor(label)
        }

# --------------------------------------------------
# コレート関数
# --------------------------------------------------
def collate_fn(batch):
    pixel_values = torch.stack([b["pixel_values"] for b in batch])
    labels = torch.tensor([b["labels"] for b in batch])
    return {"pixel_values": pixel_values, "labels": labels}

# --------------------------------------------------
# 補助関数
# --------------------------------------------------
def safe_roc_pr_auc(y_true_bin, y_score):
    """陽性が0件 or 全件のときに安全に NaN を返す"""
    pos = int(y_true_bin.sum())
    n = len(y_true_bin)
    if pos == 0 or pos == n:
        return None, None, None, None, np.nan, np.nan
    fpr, tpr, _ = roc_curve(y_true_bin, y_score)
    precision, recall, _ = precision_recall_curve(y_true_bin, y_score)
    return fpr, tpr, precision, recall, auc(fpr, tpr), auc(recall, precision)

def safe_calibration_curve(y_true_bin, y_prob, n_bins=10, strategy="uniform"):
    """正例が 0 または 全部のときは None を返す"""
    pos = y_true_bin.sum()
    if pos == 0 or pos == len(y_true_bin):
        return None, None
    return calibration_curve(y_true_bin, y_prob, n_bins=n_bins, strategy=strategy)

def save_calib_csv(prefix, y_true_bin, y_prob, mean_pred, frac_pos, out_dir, n_bins):
    """
    calibration_curve が返す mean_pred / frac_pos（非空ビンのみ）に合わせて、
    等幅ビンで件数を数え、ECE = Σ w_k * |acc_k - conf_k| を計算して返す。
    CSVには pred_mean, true_frac, count, abs_gap, weight を保存。
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, edges[1:-1], right=False)  # 0..n_bins-1
    counts = np.array([np.sum(bin_ids == i) for i in range(n_bins)])
    non_empty = np.where(counts > 0)[0]  # calibration_curve と対応（昇順）

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

def build_identity_transform():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

def build_tta_transform(magnitude: int):
    # Flip は除外した RandAugment を軽めに適用
    return transforms.Compose([
        RandAugmentNoFlip(num_ops=2, magnitude=magnitude,
                          interpolation=PIL.Image.BILINEAR, fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])

# --------------------------------------------------
# 性能評価関数
# --------------------------------------------------
def evaluate_model(model_path, test_dir, output_dir, device='cpu', batch_size=1, calib_bins=10, tta=1, tta_magnitude=5):
    """
    ViTモデルの性能評価を実行
    
    Args:
        model_path (str): ViTモデルのパス
        test_dir (str): テストデータセットのディレクトリ
        output_dir (str): 結果出力ディレクトリ
        device (str): 使用デバイス
        batch_size (int): バッチサイズ
        calib_bins (int): Calibration curveのビン数
    """

    # --- TTA 入力値ガード ---
    tta = max(1, int(tta))
    tta_magnitude = int(max(0, min(10, tta_magnitude)))
    
    print(f"[INFO] モデルパス: {model_path}")
    print(f"[INFO] テストデータ: {test_dir}")
    print(f"[INFO] 出力ディレクトリ: {output_dir}")
    print(f"[INFO] デバイス: {device}")
    print(f"[INFO] Calibration bins: {calib_bins}")
    print(f"[INFO] TTA: {tta}")
    print(f"[INFO] TTA magnitude: {tta_magnitude}")
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # データセットの読み込み
    print("\n[INFO] テストデータセットを読み込み中...")
    test_ds = PatientDataset(test_dir, is_train=False)
    print(f"[INFO] テストサンプル数: {len(test_ds)}")
    
    # モデルとプロセッサのロード
    print("\n[INFO] モデルとプロセッサを読み込み中...")
    model = AutoModelForImageClassification.from_pretrained(
        model_path,
        num_labels=len(LABEL_NAMES),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )
    
    # プロセッサを読み込む（checkpointフォルダには通常含まれないため親ディレクトリも試す）
    try:
        processor = AutoImageProcessor.from_pretrained(model_path)
    except OSError:
        # 親ディレクトリから読み込む
        parent_dir = os.path.dirname(model_path.rstrip('/'))
        print(f"[INFO] checkpointディレクトリにプロセッサがないため、親ディレクトリから読み込みます: {parent_dir}")
        processor = AutoImageProcessor.from_pretrained(parent_dir)
    
    # Trainerの設定
    args = TrainingArguments(
        output_dir="tmp_eval",
        per_device_eval_batch_size=batch_size,
        dataloader_pin_memory=True,
        dataloader_num_workers=4,
        #no_cuda = (device != "cuda"),
        use_cpu=(device == "cpu"),
        fp16 = False,
        bf16 = False,
        report_to="none",          # ← 追加
        logging_strategy="no",     # ← 任意（ログ完全オフ）
    )
    
    trainer = Trainer(
        model=model,
        args=args,
        eval_dataset=test_ds,
        data_collator=collate_fn,
    )
    
    # 予測実行（TTA 対応）
    print("\n[INFO] 予測を実行中...")

    # ベースとなる Trainer は一度作って流用（下の trainer 定義はそのまま使う）
    # args / trainer の定義は既存のままでOK

    # まず identity 1回
    test_ds.transform = build_identity_transform()
    pred_identity = trainer.predict(test_ds)
    logits_sum = pred_identity.predictions.astype(np.float64)
    labels = pred_identity.label_ids  # ラベルは同じ

    # TTA が 2以上なら、追加で (tta-1) 回ランダムAugをかけて加算
    # 乱数シードを変えつつ、transform を RandAugmentNoFlip に差し替えて回す
    if tta > 1:
        import random
        for t in range(tta - 1):
            seed = 1234 + t
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)

            test_ds.transform = build_tta_transform(magnitude=tta_magnitude)
            pred_tta = trainer.predict(test_ds)
            # 念のため順序が同じか確認（通常は同じ）
            assert (pred_tta.label_ids == labels).all()
            logits_sum += pred_tta.predictions.astype(np.float64)

    # 平均logits
    logits = (logits_sum / float(tta)).astype(np.float32)
    preds = logits.argmax(axis=-1)

    
    # 性能指標の計算
    print("\n[INFO] 性能指標を計算中...")
    
    # Accuracy
    acc = accuracy_score(labels, preds)
    
    # Precision, Recall, F1 (macro, micro, weighted)
    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        labels, preds, average="micro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(labels, preds, labels=list(range(len(LABEL_NAMES))))
    
    # Classification report
    report = classification_report(
        labels, preds, target_names=LABEL_NAMES, output_dict=True, zero_division=0
    )
    
    # ===== Specificity（per-class / macro / micro / weighted） =====
    # cm = [[cm[0,0], cm[0,1]],
    #       [cm[1,0], cm[1,1]]]
    # クラス0(tako)のspecificity = TN_0/(TN_0+FP_0) = cm[1,1] / (cm[1,1] + cm[1,0])
    # クラス1(AMI) のspecificity = TN_1/(TN_1+FP_1) = cm[0,0] / (cm[0,0] + cm[0,1])
    tn_for_0 = cm[1,1]; fp_for_0 = cm[1,0]
    tn_for_1 = cm[0,0]; fp_for_1 = cm[0,1]
    spec_0 = tn_for_0 / (tn_for_0 + fp_for_0) if (tn_for_0 + fp_for_0) > 0 else float("nan")
    spec_1 = tn_for_1 / (tn_for_1 + fp_for_1) if (tn_for_1 + fp_for_1) > 0 else float("nan")

    support0 = cm[0].sum()
    support1 = cm[1].sum()
    specificity_macro    = float(np.nanmean([spec_0, spec_1]))
    specificity_weighted = float(np.nansum([spec_0 * support0, spec_1 * support1]) / (support0 + support1))
    specificity_micro    = float(spec_1)  # 便宜上：AMI=positive と見做したときの TNR

    # 確率の計算
    probs = softmax(logits, axis=1)
    
    # AMI=1 を positive
    y_true_ami = labels
    y_score_ami = probs[:, 1]
    fpr, tpr, precision_ami, recall_ami, roc_auc, pr_auc_ami = safe_roc_pr_auc(y_true_ami, y_score_ami)
    
    # tako=0 を positive（= "takoである" を1とした二値に変換）
    y_true_tako = (labels == 0).astype(int)
    y_score_tako = probs[:, 0]
    _, _, precision_tako, recall_tako, _, pr_auc_tako = safe_roc_pr_auc(y_true_tako, y_score_tako)
    
    # ---------- Calibration (AMI=pos, tako=pos) ------------
    # AMI（=1）を positive
    y_true_ami_bin = (labels == 1).astype(int)
    y_prob_ami = probs[:, 1]
    frac_pos_ami, mean_pred_ami = safe_calibration_curve(y_true_ami_bin, y_prob_ami, calib_bins)
    
    # tako（=0）を positive（one-vs-rest）
    y_true_tako_bin = (labels == 0).astype(int)
    y_prob_tako = probs[:, 0]
    frac_pos_tako, mean_pred_tako = safe_calibration_curve(y_true_tako_bin, y_prob_tako, calib_bins)
    
    # ECE の計算
    ece_ami = np.nan
    ece_tako = np.nan
    if frac_pos_ami is not None:
        ece_ami = save_calib_csv("AMI", y_true_ami_bin, y_prob_ami, mean_pred_ami, frac_pos_ami, output_dir, calib_bins)
    if frac_pos_tako is not None:
        ece_tako = save_calib_csv("tako", y_true_tako_bin, y_prob_tako, mean_pred_tako, frac_pos_tako, output_dir, calib_bins)
    
    # ------ Brier score（2クラス：正例の確率と0/1ラベルのMSE）------
    brier_ami = float(np.mean((y_true_ami_bin - y_prob_ami) ** 2))
    brier_tako = float(np.mean((y_true_tako_bin - y_prob_tako) ** 2))
    
    # Brier Skill Score
    base_p_ami = y_true_ami_bin.mean()
    base_p_tako = y_true_tako_bin.mean()
    brier_base_ami = float(np.mean((y_true_ami_bin - base_p_ami) ** 2))
    brier_base_tako = float(np.mean((y_true_tako_bin - base_p_tako) ** 2))
    bss_ami = 1.0 - (brier_ami / brier_base_ami) if brier_base_ami > 0 else np.nan
    bss_tako = 1.0 - (brier_tako / brier_base_tako) if brier_base_tako > 0 else np.nan
    
    # 結果の表示
    print("\n" + "="*50)
    print("=== Test set evaluation ===")
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
    
    # CSVファイルで結果を保存
    print("\n[INFO] 結果をCSVファイルに保存中...")
    
    # 画像パスとラベルを取得
    image_paths = [test_ds.samples[i][0] for i in range(len(test_ds))]
    true_ids = [int(labels[i]) for i in range(len(labels))]
    true_labels = [ID2LABEL[true_ids[i]] for i in range(len(true_ids))]
    
    # 予測結果の保存（添付画像のフォーマットに合わせる）
    results_df = pd.DataFrame({
        "path": image_paths,
        "true_id": true_ids,
        "true_label": true_labels,
        "pred_id": [int(preds[i]) for i in range(len(preds))],
        "pred_label": [ID2LABEL[int(preds[i])] for i in range(len(preds))],
        "logit_tako": logits[:, 0],
        "logit_AMI": logits[:, 1],
        "prob_tako": probs[:, 0],
        "prob_AMI": probs[:, 1]
    })
    
    pred_csv_path = os.path.join(output_dir, "predictions.csv")
    results_df.to_csv(pred_csv_path, index=False)
    
    # 性能指標の保存
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
    
    # JSON保存
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
    
    # 可視化の保存
    print("\n[INFO] 可視化を保存中...")
    
    # 混同行列
    plt.figure(figsize=(6, 5))
    sns.heatmap(np.array(cm), annot=True, fmt="d",
                xticklabels=LABEL_NAMES, yticklabels=LABEL_NAMES, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # ROC曲線（AMI=positive）
    if np.isfinite(roc_auc):
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, label=f'ROC (AMI=pos) AUC = {roc_auc:.3f}', zorder=3, linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, zorder=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (AMI=positive)')
        plt.legend(loc='lower right')
        plt.grid(True, zorder=1)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        roc_path = os.path.join(output_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # PR曲線（AMI=positive）
    if np.isfinite(pr_auc_ami):
        plt.figure(figsize=(6, 6))
        plt.plot(recall_ami, precision_ami, label=f'PR (AMI=pos) AUC = {pr_auc_ami:.3f}', zorder=3, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision–Recall Curve (AMI=positive)')
        plt.legend(loc='lower left')
        plt.grid(True, zorder=1)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        pr_path = os.path.join(output_dir, "pr_curve_AMI.png")
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # PR曲線（tako=positive）
    if np.isfinite(pr_auc_tako):
        plt.figure(figsize=(6, 6))
        plt.plot(recall_tako, precision_tako, label=f'PR (tako=pos) AUC = {pr_auc_tako:.3f}', zorder=3, linewidth=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision–Recall Curve (tako=positive)')
        plt.legend(loc='lower left')
        plt.grid(True, zorder=1)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        pr_path = os.path.join(output_dir, "pr_curve_tako.png")
        plt.savefig(pr_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Calibration plot (AMI=pos)
    if frac_pos_ami is not None:
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "--", linewidth=1, label="Perfectly calibrated", alpha=0.7)
        plt.plot(mean_pred_ami, frac_pos_ami, marker="o", label="AMI (positive)")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed frequency")
        plt.title("Calibration plot (AMI=positive)")
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.legend(loc="upper left")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        calib_ami_path = os.path.join(output_dir, "calibration_AMI.png")
        plt.savefig(calib_ami_path, dpi=150, bbox_inches="tight")
        plt.close()
    
    # Calibration plot (tako=pos)
    if frac_pos_tako is not None:
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "--", linewidth=1, label="Perfectly calibrated", alpha=0.7)
        plt.plot(mean_pred_tako, frac_pos_tako, marker="o", label="tako (positive)")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed frequency")
        plt.title("Calibration plot (tako=positive)")
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.legend(loc="upper left")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        calib_tako_path = os.path.join(output_dir, "calibration_tako.png")
        plt.savefig(calib_tako_path, dpi=150, bbox_inches="tight")
        plt.close()
    
    print(f"\n[INFO] 結果を保存しました:")
    print(f"  - {pred_csv_path}")
    print(f"  - {metrics_csv_path}")
    print(f"  - {metrics_json_path}")
    print(f"  - {cm_path}")
    
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
        "specificity_macro": specificity_macro,      # ← 追加
        "specificity_micro": specificity_micro,      # ← 追加
        "specificity_weighted": specificity_weighted # ← 追加
    }

# --------------------------------------------------
# メイン関数
# --------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="ViTモデルの性能評価スクリプト",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用例:
  # 基本的な使用（バッチサイズ1、CPU使用）
  python evaluate_dataset_dir_ViT.py \\
    --model-path /path/to/vit/checkpoint \\
    --test-dir /path/to/test/dataset \\
    --output-dir /path/to/output
  
  # GPU使用（推奨: バッチサイズ1）
  python evaluate_dataset_dir_ViT.py \\
    --model-path /path/to/vit/checkpoint \\
    --test-dir /path/to/test/dataset \\
    --output-dir /path/to/output \\
    --device cuda \\
    --batch-size 1 \\
    --calibration-bins 10
  
  # GPU使用、バッチサイズ増加、calibration bins変更
  python evaluate_dataset_dir_ViT.py \\
    --model-path /path/to/vit/checkpoint \\
    --test-dir /path/to/test/dataset \\
    --output-dir /path/to/output \\
    --device cuda \\
    --batch-size 4 \\
    --calibration-bins 15
        """
    )
    
    parser.add_argument("--model-path", type=str, required=True, help="ViTモデルのパス（checkpointディレクトリ）")
    parser.add_argument("--test-dir", type=str, required=True, help="テストデータセットのディレクトリ")
    parser.add_argument("--output-dir", type=str, required=True, help="結果出力ディレクトリ")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="使用デバイス (default: cpu)")
    parser.add_argument("--batch-size", type=int, default=1, help="バッチサイズ (default: 1)")
    parser.add_argument("--calibration-bins", type=int, default=10, help="Calibration curveのビン数 (default: 10)")
    parser.add_argument("--tta", type=int, default=1, help="TTA の試行回数。1ならTTAなし（identityのみ）")
    parser.add_argument("--tta-magnitude", type=int, default=5, help="TTA時 RandAugment のmagnitude (0-10, 推奨: 3-7)")

    
    args = parser.parse_args()
    
    # パスの存在確認
    if not os.path.exists(args.model_path):
        print(f"[ERROR] モデルパスが存在しません: {args.model_path}")
        sys.exit(1)
    
    if not os.path.exists(args.test_dir):
        print(f"[ERROR] テストデータディレクトリが存在しません: {args.test_dir}")
        sys.exit(1)
    
    # デバイスの確認
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDAが利用できません。CPUを使用します。")
        args.device = "cpu"
    
    # --- TTA 入力値ガード ---
    args.tta = max(1, int(args.tta))
    args.tta_magnitude = int(max(0, min(10, args.tta_magnitude)))
    # 任意: 範囲外だったら通知
    if args.tta < 2:
        print("[INFO] TTAは無効（identity のみ）です。")
    if not (0 <= args.tta_magnitude <= 10):
        print("[INFO] TTA magnitude を 0–10 にクリップしました。")

    try:
        # 性能評価実行
        results = evaluate_model(
            model_path=args.model_path,
            test_dir=args.test_dir,
            output_dir=args.output_dir,
            device=args.device,
            batch_size=args.batch_size,
            calib_bins=args.calibration_bins,
            tta=args.tta,
            tta_magnitude=args.tta_magnitude
        )
        
        print("\n" + "="*50)
        print("=== 評価完了 ===")
        print("="*50)
        print(f"Accuracy:      {results['accuracy']:.4f}")
        print(f"Precision:      {results['precision']:.4f}")
        print(f"Recall:         {results['recall']:.4f}")
        print(f"F1-Score:       {results['f1_score']:.4f}")
        print(f"ROC-AUC:        {results['roc_auc']:.4f}")
        print(f"PR-AUC (AMI):   {results['pr_auc_ami']:.4f}")
        print(f"PR-AUC (tako):  {results['pr_auc_tako']:.4f}")
        print(f"ECE (AMI):      {results['ece_ami']:.6f}")
        print(f"ECE (tako):     {results['ece_tako']:.6f}")
        print(f"Brier (AMI):    {results['brier_ami']:.6f}")
        print(f"Brier (tako):   {results['brier_tako']:.6f}")
        print(f"Specificity (macro): {results['specificity_macro']:.4f}")
        print(f"Specificity (micro): {results['specificity_micro']:.4f}")
        print(f"Specificity (weighted): {results['specificity_weighted']:.4f}")


        
    except Exception as e:
        print(f"[ERROR] 評価中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"\n[INFO] 完了時刻: {datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9), 'JST'))}")

if __name__ == "__main__":
    main()
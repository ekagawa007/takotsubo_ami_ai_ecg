#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
3つの predictions.csv をソフトアンサンブルし、
ViT版 evaluate_dataset_dir_ViT.py と同じ指標・出力ファイルを生成するスクリプト。

Version 1.0.1 2025-11-24 calibration関連の指標を出力するように修正。
Version 1.0.2 2025-11-25
- calibration関連の指標を出力
- 3モデルの重み付きアンサンブルに対応 (--w1, --w2, --w3)
Version 1.0.3 2025-12-08
- ensemble-mode の default を logits に統一
- read_predictions の suffix 周りのバグ修正（引数不一致/未定義変数）
- predictions.csv の一般的な形式（prob_tako/prob_AMI）を優先して解釈
Version 1.0.4 2026-10-06
- ID列の正規化オプションを追加 (--id-normalize, --id-anchor)
  環境ごとに絶対パスが異なる CSV（Google Drive / Ubuntu など）を突き合わせ可能に
- 真のラベル復元時にも同じ正規化を適用
- 共通IDが0件のときはエラーで終了するガードを追加

- 入力: 3 つの CSV (それぞれ同じサンプル集合)
- 出力: output_dir 以下に
    - predictions.csv
    - metrics.csv
    - metrics.json
    - confusion_matrix.png
    - roc_curve.png
    - pr_curve_AMI.png, pr_curve_tako.png
    - calibration_AMI.csv, calibration_tako.csv
    - calibration_AMI.png, calibration_tako.png

ラベルの想定:
- 0: tako
- 1: AMI

使用例
python predict_ensemble3_testset_from_csv_rawlogits.py \
  --csv1 /path/to/model1/predictions.csv \
  --csv2 /path/to/model2/predictions.csv \
  --csv3 /path/to/model3/predictions.csv \
  --output-dir /path/to/ensemble_out \
  --id-col path \
  --label-col true_id \
  --threshold 0.5 \
  --calibration-bins 10 \
  --w1 0.5 --w2 0.3 --w3 0.2 \
  --id-normalize anchor \
  --id-anchor ecg_dataset_480/

ID正規化 (--id-normalize):
- none (default): ID列をそのまま使用
- basename      : ファイル名のみで突き合わせ（ファイル名が一意な場合のみ）
- anchor        : --id-anchor で指定した文字列より後ろの相対パスで突き合わせ
                  例) /content/drive/MyDrive/data/ecg_dataset_480/test/a/AMI/1.png
                      /path/to/data/ecg_dataset_480/test/a/AMI/1.png
                      → いずれも test/a/AMI/1.png
  ※ 出力 predictions.csv の ID 列は正規化後の値になります。

アンサンブル方式:
- logits (default):
    CSVに logit 列があればそれを使用。無ければ prob から logit(p) を逆算し、
    重み付き平均して2クラスsoftmaxで復元。
- probs:
    prob をそのまま重み付き平均。

注意:
- 本スクリプトの specificity は「ViT版の計算と揃える」目的で
  独自実装になっている可能性があります。
  もし一般的な2x2定義に揃える必要があれば、別途調整してください。
"""

import argparse
import sys
from functools import reduce
import os
import csv
import json
import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    auc,
)
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------
# 定数 (ViT版と揃える)
# -----------------------------
LABEL_NAMES = ["tako", "AMI"]
LABEL2ID = {n: i for i, n in enumerate(LABEL_NAMES)}
ID2LABEL = {i: n for n, i in LABEL2ID.items()}


# -----------------------------
# 補助関数 (ViT版から移植)
# -----------------------------
def safe_roc_pr_auc(y_true_bin, y_score):
    """陽性が 0 件 or 全件のときに安全に NaN を返す"""
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
    # sklearnは (prob_true, prob_pred) を返す
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
        w.writerow([
            "bin", "pred_mean", "true_frac", "count",
            "abs_gap", "weight", "bin_left", "bin_right"
        ])
        for i, b in enumerate(non_empty):
            w.writerow([
                int(b),
                float(mean_pred[i]),
                float(frac_pos[i]),
                int(counts[b]),
                float(abs_gap[i]),
                float(weights[i]),
                float(edges[b]),
                float(edges[b + 1]),
            ])
        w.writerow([])
        w.writerow(["ECE", ece])
    print(f"[Calibration] Saved CSV: {csv_path} (ECE={ece:.6f})")
    return ece


# -----------------------------
# ID 正規化
# -----------------------------
def normalize_id(s: pd.Series, mode: str = "none", anchor: str = None) -> pd.Series:
    """環境ごとに異なる絶対パスを、突き合わせ用のキーに正規化する"""
    s = s.astype(str).str.replace("\\", "/", regex=False)
    if mode == "none":
        return s
    if mode == "basename":
        return s.str.split("/").str[-1]
    if mode == "anchor":
        if not anchor:
            raise ValueError("--id-normalize anchor には --id-anchor が必要です。")
        bad = ~s.str.contains(anchor, regex=False)
        if bad.any():
            raise ValueError(
                f"anchor '{anchor}' を含まないIDが {int(bad.sum())} 件あります。"
                f" 例: {s[bad].iloc[0]}"
            )
        return s.map(lambda x: x[x.rfind(anchor) + len(anchor):].lstrip("/"))
    raise ValueError(f"unknown id-normalize mode: {mode}")


# -----------------------------
# CSV読み込み
# -----------------------------
def read_predictions(path: str, id_col: str, suffix: str,
                     id_mode: str = "none", id_anchor: str = None) -> pd.DataFrame:
    """
    1つの predictions.csv を読み込み:
      - id_col: ID列 (例: path)。id_mode に従って正規化する。

      - 予測確率列は以下を順に探す:
          1) prob_AMI / prob_tako
          2) probability_AMI / probability_tako
          3) logit_AMI / logit_tako からシグモイド変換

      - ロジット列は以下を優先して取得:
          1) logit_AMI / logit_tako
          2) 上で確定した確率列から logit(p) を逆算

    戻り値:
      [id_col,
       prob_AMI_<suffix>, prob_tako_<suffix>,
       logit_AMI_<suffix>, logit_tako_<suffix>]
    """
    df = pd.read_csv(path)

    if id_col not in df.columns:
        raise ValueError(f"{path} に ID 列 {id_col} が存在しません。")

    df[id_col] = normalize_id(df[id_col], id_mode, id_anchor)

    # --- 1) 確率列の探索 ---
    prob_ami_col = None
    prob_tako_col = None

    if "prob_AMI" in df.columns:
        prob_ami_col = "prob_AMI"
    if "prob_tako" in df.columns:
        prob_tako_col = "prob_tako"

    if prob_ami_col is None and "probability_AMI" in df.columns:
        prob_ami_col = "probability_AMI"
    if prob_tako_col is None and "probability_tako" in df.columns:
        prob_tako_col = "probability_tako"

    # --- 2) ロジット列の探索（あれば保持） ---
    logit_ami_src = "logit_AMI" if "logit_AMI" in df.columns else None
    logit_tako_src = "logit_tako" if "logit_tako" in df.columns else None

    # --- 3) 確率が無い場合はロジットから生成 ---
    if prob_ami_col is None and logit_ami_src is not None:
        df["prob_AMI_from_logit"] = 1.0 / (1.0 + np.exp(-df[logit_ami_src].astype(float)))
        prob_ami_col = "prob_AMI_from_logit"
    if prob_tako_col is None and logit_tako_src is not None:
        df["prob_tako_from_logit"] = 1.0 / (1.0 + np.exp(-df[logit_tako_src].astype(float)))
        prob_tako_col = "prob_tako_from_logit"

    # --- 4) 片方だけある場合、和=1 と仮定して補完 ---
    if prob_ami_col is None and prob_tako_col is not None:
        df["prob_AMI_from_tako"] = 1.0 - df[prob_tako_col].astype(float)
        prob_ami_col = "prob_AMI_from_tako"
    if prob_tako_col is None and prob_ami_col is not None:
        df["prob_tako_from_AMI"] = 1.0 - df[prob_ami_col].astype(float)
        prob_tako_col = "prob_tako_from_AMI"

    if prob_ami_col is None or prob_tako_col is None:
        raise ValueError(
            f"{path} に予測確率列が見つかりませんでした。"
            f" prob_AMI/prob_tako, probability_AMI/probability_tako, "
            f"もしくは logit_AMI/logit_tako のいずれかが必要です。"
        )

    # --- 5) ロジット列の確定（無ければ確率から逆算） ---
    eps = 1e-7
    if logit_ami_src is None:
        p = np.clip(df[prob_ami_col].astype(float), eps, 1.0 - eps)
        df["logit_AMI_from_prob"] = np.log(p / (1.0 - p))
        logit_ami_src = "logit_AMI_from_prob"
    if logit_tako_src is None:
        p = np.clip(df[prob_tako_col].astype(float), eps, 1.0 - eps)
        df["logit_tako_from_prob"] = np.log(p / (1.0 - p))
        logit_tako_src = "logit_tako_from_prob"

    out = df[[id_col, prob_ami_col, prob_tako_col, logit_ami_src, logit_tako_src]].copy()
    out = out.rename(columns={
        prob_ami_col: f"prob_AMI_{suffix}",
        prob_tako_col: f"prob_tako_{suffix}",
        logit_ami_src: f"logit_AMI_{suffix}",
        logit_tako_src: f"logit_tako_{suffix}",
    })
    return out


def merge_on_id(dfs, id_col, how="inner"):
    return reduce(lambda l, r: pd.merge(l, r, on=[id_col], how=how), dfs)


# -----------------------------
# メイン処理
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description="3つの predictions.csv をソフトアンサンブルし、"
                    "evaluate_dataset_dir_ViT.py と同等の指標・出力ファイルを生成します。"
    )
    parser.add_argument("--csv1", required=True, help="1つ目の予測CSV (例: model1/predictions.csv)")
    parser.add_argument("--csv2", required=True, help="2つ目の予測CSV")
    parser.add_argument("--csv3", required=True, help="3つ目の予測CSV")
    parser.add_argument("--output-dir", required=True, help="結果出力ディレクトリ")
    parser.add_argument("--id-col", default="path", help="ID列名（例: path）")
    parser.add_argument(
        "--label-col", default="true_id",
        help="0/1 の真のラベル列名（例: true_id）"
    )
    parser.add_argument(
        "--id-normalize", default="none",
        choices=["none", "basename", "anchor"],
        help="ID列の正規化方法（環境間でパスが異なる場合に使用）"
    )
    parser.add_argument(
        "--id-anchor", default=None,
        help="anchor モードで使う区切り文字列（例: 'ecg_dataset_480/'）"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="2値化のしきい値 (default: 0.5)"
    )
    parser.add_argument(
        "--calibration-bins", type=int, default=10,
        help="Calibration curve のビン数 (default: 10)"
    )
    parser.add_argument(
        "--ensemble-mode",
        type=str,
        default="logits",
        choices=["logits", "probs"],
        help=(
            "アンサンブル方式。"
            "logits: 各モデルのロジット（無ければ logit(prob)）を重み付き平均 → softmax（default）。"
            "probs : 各モデルの確率をそのまま重み付き平均。"
        ),
    )
    parser.add_argument("--w1", type=float, default=1.0, help="csv1 の重み (default: 1.0)")
    parser.add_argument("--w2", type=float, default=1.0, help="csv2 の重み (default: 1.0)")
    parser.add_argument("--w3", type=float, default=1.0, help="csv3 の重み (default: 1.0)")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # ---------- CSV読み込み & マージ ----------
    try:
        d1 = read_predictions(args.csv1, args.id_col, "m1", args.id_normalize, args.id_anchor)
        d2 = read_predictions(args.csv2, args.id_col, "m2", args.id_normalize, args.id_anchor)
        d3 = read_predictions(args.csv3, args.id_col, "m3", args.id_normalize, args.id_anchor)
    except Exception as e:
        print(f"[ERROR] CSVの読み込みに失敗: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 重複IDチェック（正規化後） ---
    for name, d in [("csv1", d1), ("csv2", d2), ("csv3", d3)]:
        dup = d[args.id_col].duplicated().sum()
        if dup > 0:
            print(f"[WARN] {name} に重複IDが {dup} 行あります（正規化後）。", file=sys.stderr)

    try:
        merged = merge_on_id([d1, d2, d3], args.id_col, how="inner")
        merged = merged.sort_values(args.id_col).reset_index(drop=True)
    except Exception as e:
        print(f"[ERROR] CSVのマージに失敗: {e}", file=sys.stderr)
        sys.exit(1)

    # --- 3 CSVでIDが一致しているかのチェック ---
    if len(merged) != len(d1) or len(merged) != len(d2) or len(merged) != len(d3):
        print("[WARN] CSV間でIDが一致しない行があります。", file=sys.stderr)

        ids1 = set(d1[args.id_col].tolist())
        ids2 = set(d2[args.id_col].tolist())
        ids3 = set(d3[args.id_col].tolist())

        common = ids1 & ids2 & ids3
        only1 = ids1 - (ids2 | ids3)
        only2 = ids2 - (ids1 | ids3)
        only3 = ids3 - (ids1 | ids2)

        print(
            f"[WARN] rows: csv1={len(d1)}, csv2={len(d2)}, csv3={len(d3)}, merged(common)={len(merged)}",
            file=sys.stderr
        )
        print(
            f"[WARN] unique IDs: csv1={len(ids1)}, csv2={len(ids2)}, csv3={len(ids3)}, common={len(common)}",
            file=sys.stderr
        )
        print(f"[WARN] ids only in csv1: {len(only1)}", file=sys.stderr)
        print(f"[WARN] ids only in csv2: {len(only2)}", file=sys.stderr)
        print(f"[WARN] ids only in csv3: {len(only3)}", file=sys.stderr)

    if len(merged) == 0:
        print(
            "[ERROR] 3つのCSVで共通するIDが0件です。"
            " --id-normalize / --id-anchor を確認してください。",
            file=sys.stderr
        )
        print(f"[ERROR] csv1 の ID 例: {d1[args.id_col].iloc[0]}", file=sys.stderr)
        print(f"[ERROR] csv2 の ID 例: {d2[args.id_col].iloc[0]}", file=sys.stderr)
        print(f"[ERROR] csv3 の ID 例: {d3[args.id_col].iloc[0]}", file=sys.stderr)
        sys.exit(1)

    print(f"[INFO] merged samples: {len(merged)}")

    # ---------- ラベルを csv1 から復元（同じ正規化を適用） ----------
    try:
        base = pd.read_csv(args.csv1, usecols=[args.id_col, args.label_col])
        base[args.id_col] = normalize_id(base[args.id_col], args.id_normalize, args.id_anchor)
        base = base.drop_duplicates(subset=[args.id_col])
        merged = pd.merge(merged, base, on=args.id_col, how="left")
    except Exception as e:
        print(f"[ERROR] 真のラベル列の読み込みに失敗: {e}", file=sys.stderr)
        sys.exit(1)

    if merged[args.label_col].isna().any():
        n_na = int(merged[args.label_col].isna().sum())
        print(f"[ERROR] {n_na} サンプルで真のラベルが見つかりませんでした。", file=sys.stderr)
        sys.exit(1)

    # 3つの確率/ロジット列名を固定で参照（suffix に合わせる）
    ami_prob_cols = ["prob_AMI_m1", "prob_AMI_m2", "prob_AMI_m3"]
    tako_prob_cols = ["prob_tako_m1", "prob_tako_m2", "prob_tako_m3"]
    ami_logit_cols = ["logit_AMI_m1", "logit_AMI_m2", "logit_AMI_m3"]
    tako_logit_cols = ["logit_tako_m1", "logit_tako_m2", "logit_tako_m3"]

    # 確率列・ロジット列の存在チェック（read_predictions が補完する想定）
    for c in ami_prob_cols + tako_prob_cols + ami_logit_cols + tako_logit_cols:
        if c not in merged.columns:
            print(f"[ERROR] missing column: {c}", file=sys.stderr)
            sys.exit(1)

    # ---------- ソフトアンサンブル（重み付き平均） ----------
    weights = np.array([args.w1, args.w2, args.w3], dtype=float)

    if (not np.isfinite(weights).all()) or (weights.sum() <= 0):
        print("[WARN] 無効なweightsが指定されたため、[1,1,1] にフォールバックします。", file=sys.stderr)
        weights = np.array([1.0, 1.0, 1.0], dtype=float)

    w = weights / weights.sum()

    # 以降で ensemble ロジットを predictions.csv に保存するために保持
    logit_ami_ens = None
    logit_tako_ens = None

    if args.ensemble_mode == "probs":
        # -------- 確率平均 --------
        ami_stack = merged[ami_prob_cols].astype(float).values  # (N, 3)
        tako_stack = merged[tako_prob_cols].astype(float).values

        prob_ami = (ami_stack * w).sum(axis=1)
        prob_tako = (tako_stack * w).sum(axis=1)

        prob_ami = np.clip(prob_ami, 0.0, 1.0)
        prob_tako = np.clip(prob_tako, 0.0, 1.0)
        denom = prob_ami + prob_tako
        safe = denom > 0
        prob_ami[safe] = prob_ami[safe] / denom[safe]
        prob_tako[safe] = prob_tako[safe] / denom[safe]

    else:
        # -------- ロジット平均（raw logits がある前提。無い場合は read_predictions で補完） --------
        ami_logit_stack = merged[ami_logit_cols].astype(float).values  # (N, 3)
        tako_logit_stack = merged[tako_logit_cols].astype(float).values

        logit_ami = (ami_logit_stack * w).sum(axis=1)
        logit_tako = (tako_logit_stack * w).sum(axis=1)

        logit_ami_ens = logit_ami
        logit_tako_ens = logit_tako

        # 2クラス softmax
        m = np.maximum(logit_ami, logit_tako)
        exp_ami = np.exp(logit_ami - m)
        exp_tako = np.exp(logit_tako - m)
        denom = exp_ami + exp_tako
        prob_ami = exp_ami / denom
        prob_tako = exp_tako / denom

    merged["prob_AMI_ensemble"] = prob_ami
    merged["prob_tako_ensemble"] = prob_tako

    # 念のため正規化
    merged["prob_AMI_ensemble"] = merged["prob_AMI_ensemble"].clip(0.0, 1.0)
    merged["prob_tako_ensemble"] = merged["prob_tako_ensemble"].clip(0.0, 1.0)
    denom = (merged["prob_AMI_ensemble"] + merged["prob_tako_ensemble"]).replace(0, np.nan)
    idx = denom.notna()
    merged.loc[idx, "prob_AMI_ensemble"] = merged.loc[idx, "prob_AMI_ensemble"] / denom[idx]
    merged.loc[idx, "prob_tako_ensemble"] = merged.loc[idx, "prob_tako_ensemble"] / denom[idx]

    # ---------- ラベル/予測 ----------
    y_true = merged[args.label_col].values.astype(int)  # 0/1
    y_prob_ami = merged["prob_AMI_ensemble"].values.astype(float)
    y_prob_tako = merged["prob_tako_ensemble"].values.astype(float)
    y_pred = (y_prob_ami >= args.threshold).astype(int)

    # ---------- 性能指標 ----------
    print("\n[INFO] 指標を計算中...")

    acc = accuracy_score(y_true, y_pred)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    prec_micro, rec_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="micro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    report = classification_report(
        y_true, y_pred, labels=[0, 1], target_names=LABEL_NAMES,
        output_dict=True, zero_division=0
    )

    # Specificity（ViT版と整合させる意図のまま）
    tn_for_0 = cm[1, 1]
    fp_for_0 = cm[1, 0]
    tn_for_1 = cm[0, 0]
    fp_for_1 = cm[0, 1]
    spec_0 = tn_for_0 / (tn_for_0 + fp_for_0) if (tn_for_0 + fp_for_0) > 0 else float("nan")
    spec_1 = tn_for_1 / (tn_for_1 + fp_for_1) if (tn_for_1 + fp_for_1) > 0 else float("nan")
    support0 = cm[0].sum()
    support1 = cm[1].sum()
    specificity_macro = float(np.nanmean([spec_0, spec_1]))
    specificity_weighted = float(
        np.nansum([spec_0 * support0, spec_1 * support1]) / (support0 + support1)
    )
    specificity_micro = float(spec_1)  # 便宜上 AMI=positive の TNR

    # ROC/PR AUC
    y_true_ami = y_true
    y_score_ami = y_prob_ami
    fpr_ami, tpr_ami, precision_ami, recall_ami, roc_auc, pr_auc_ami = safe_roc_pr_auc(
        y_true_ami, y_score_ami
    )

    y_true_tako = (y_true == 0).astype(int)
    y_score_tako = y_prob_tako
    fpr_tako, tpr_tako, precision_tako, recall_tako, _, pr_auc_tako = safe_roc_pr_auc(
        y_true_tako, y_score_tako
    )

    # Calibration & Brier & BSS
    calib_bins = args.calibration_bins
    y_true_ami_bin = (y_true == 1).astype(int)
    y_true_tako_bin = (y_true == 0).astype(int)

    frac_pos_ami, mean_pred_ami = safe_calibration_curve(
        y_true_ami_bin, y_prob_ami, calib_bins
    )
    frac_pos_tako, mean_pred_tako = safe_calibration_curve(
        y_true_tako_bin, y_prob_tako, calib_bins
    )

    ece_ami = np.nan
    ece_tako = np.nan
    if frac_pos_ami is not None:
        ece_ami = save_calib_csv(
            "AMI",
            y_true_ami_bin,
            y_prob_ami,
            mean_pred_ami,
            frac_pos_ami,
            args.output_dir,
            calib_bins,
        )
    if frac_pos_tako is not None:
        ece_tako = save_calib_csv(
            "tako",
            y_true_tako_bin,
            y_prob_tako,
            mean_pred_tako,
            frac_pos_tako,
            args.output_dir,
            calib_bins,
        )

    brier_ami = float(np.mean((y_true_ami_bin - y_prob_ami) ** 2))
    brier_tako = float(np.mean((y_true_tako_bin - y_prob_tako) ** 2))
    base_p_ami = y_true_ami_bin.mean()
    base_p_tako = y_true_tako_bin.mean()
    brier_base_ami = float(np.mean((y_true_ami_bin - base_p_ami) ** 2))
    brier_base_tako = float(np.mean((y_true_tako_bin - base_p_tako) ** 2))
    bss_ami = 1.0 - (brier_ami / brier_base_ami) if brier_base_ami > 0 else np.nan
    bss_tako = 1.0 - (brier_tako / brier_base_tako) if brier_base_tako > 0 else np.nan

    # ---------- コンソール出力 ----------
    print("\n" + "=" * 50)
    print("=== Ensemble Test set evaluation ===")
    print("=" * 50)
    print(f"Ensemble mode: {args.ensemble_mode}")
    print(f"ID normalize : {args.id_normalize}" + (f" (anchor='{args.id_anchor}')" if args.id_normalize == "anchor" else ""))
    print(f"Samples      : {len(merged)}")
    print(f"Weights raw  : {[args.w1, args.w2, args.w3]}")
    print(f"Weights norm : {[float(x) for x in w]}")
    print(f"Confusion matrix:\n{cm}")
    print(f"Accuracy (overall):      {acc:.4f}")
    print(f"Precision (macro):       {prec_macro:.4f}")
    print(f"Recall (macro):          {rec_macro:.4f}")
    print(f"Specificity (macro):     {specificity_macro:.4f}")
    print(f"F1 (macro):              {f1_macro:.4f}")
    print(f"Precision (micro):       {prec_micro:.4f}")
    print(f"Recall (micro):          {rec_micro:.4f}")
    print(f"Specificity (micro*):    {specificity_micro:.4f}")
    print(f"F1 (micro):              {f1_micro:.4f}")
    print(f"Precision (weighted):    {prec_w:.4f}")
    print(f"Recall (weighted):       {rec_w:.4f}")
    print(f"Specificity (weighted):  {specificity_weighted:.4f}")
    print(f"F1 (weighted):           {f1_w:.4f}")
    print(f"Specificity per-class:   tako={spec_0:.4f}, AMI={spec_1:.4f}")

    print("\n=== Per-class metrics ===")
    for cls in LABEL_NAMES:
        print(
            f"{cls:10s}  Precision: {report[cls]['precision']:.4f}  "
            f"Recall: {report[cls]['recall']:.4f}  "
            f"F1: {report[cls]['f1-score']:.4f}"
        )

    print("\n=== ROC/PR AUC ===")
    print(
        "ROC AUC (AMI=pos): {:.4f}".format(roc_auc)
        if np.isfinite(roc_auc)
        else "ROC AUC (AMI=pos): n/a"
    )
    print(
        "PR  AUC (AMI=pos): {:.4f}".format(pr_auc_ami)
        if np.isfinite(pr_auc_ami)
        else "PR  AUC (AMI=pos): n/a"
    )
    print(
        "PR  AUC (tako=pos): {:.4f}".format(pr_auc_tako)
        if np.isfinite(pr_auc_tako)
        else "PR  AUC (tako=pos): n/a"
    )

    print("\n=== Calibration metrics ===")
    print(
        "ECE (AMI=positive):  {:.6f}".format(ece_ami)
        if np.isfinite(ece_ami)
        else "ECE (AMI=positive):  n/a"
    )
    print(
        "ECE (tako=positive): {:.6f}".format(ece_tako)
        if np.isfinite(ece_tako)
        else "ECE (tako=positive): n/a"
    )
    print("Brier (AMI=positive):  {:.6f}".format(brier_ami))
    print("Brier (tako=positive): {:.6f}".format(brier_tako))
    print(
        "Brier Skill Score (AMI=positive):  {:.6f}".format(bss_ami)
        if np.isfinite(bss_ami)
        else "BSS (AMI=positive):  n/a"
    )
    print(
        "Brier Skill Score (tako=positive): {:.6f}".format(bss_tako)
        if np.isfinite(bss_tako)
        else "BSS (tako=positive): n/a"
    )

    # ---------- predictions.csv の保存 ----------
    # ensemble logits を保存（logits モードでは平均ロジットを優先）
    if logit_ami_ens is None or logit_tako_ens is None:
        eps = 1e-7
        prob_tako_clip = np.clip(y_prob_tako, eps, 1 - eps)
        prob_ami_clip = np.clip(y_prob_ami, eps, 1 - eps)
        logit_tako = np.log(prob_tako_clip / (1.0 - prob_tako_clip))
        logit_ami = np.log(prob_ami_clip / (1.0 - prob_ami_clip))
    else:
        logit_ami = logit_ami_ens
        logit_tako = logit_tako_ens

    true_ids = y_true.astype(int)
    true_labels = [ID2LABEL[int(x)] for x in true_ids]
    pred_ids = y_pred.astype(int)
    pred_labels = [ID2LABEL[int(x)] for x in pred_ids]

    pred_df = pd.DataFrame({
        args.id_col: merged[args.id_col].values,
        "true_id": true_ids,
        "true_label": true_labels,
        "pred_id": pred_ids,
        "pred_label": pred_labels,
        "logit_tako": logit_tako,
        "logit_AMI": logit_ami,
        "prob_tako": y_prob_tako,
        "prob_AMI": y_prob_ami,
    })
    pred_csv_path = os.path.join(args.output_dir, "predictions.csv")
    pred_df.to_csv(pred_csv_path, index=False)

    # ---------- metrics.csv / metrics.json ----------
    metrics_df = pd.DataFrame({
        "metric": [
            "accuracy", "precision_macro", "recall_macro", "specificity_macro", "f1_macro",
            "roc_auc_AMIpos", "pr_auc_AMIpos", "pr_auc_takopos",
            "ECE_AMIpos", "ECE_takopos",
            "BSS_AMIpos", "BSS_takopos",
            "Brier_AMIpos", "Brier_takopos",
            "precision_micro", "recall_micro", "specificity_micro", "f1_micro",
            "precision_weighted", "recall_weighted", "specificity_weighted", "f1_weighted",
        ],
        "value": [
            acc, prec_macro, rec_macro, specificity_macro, f1_macro,
            roc_auc, pr_auc_ami, pr_auc_tako,
            float(ece_ami) if np.isfinite(ece_ami) else np.nan,
            float(ece_tako) if np.isfinite(ece_tako) else np.nan,
            bss_ami, bss_tako,
            brier_ami, brier_tako,
            prec_micro, rec_micro, specificity_micro, f1_micro,
            prec_w, rec_w, specificity_weighted, f1_w,
        ],
    })
    metrics_csv_path = os.path.join(args.output_dir, "metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)

    metrics_json = {
        "classes": LABEL_NAMES,
        "ensemble_mode": args.ensemble_mode,
        "id_normalize": args.id_normalize,
        "id_anchor": args.id_anchor,
        "n_samples": int(len(merged)),
        "input_csvs": [args.csv1, args.csv2, args.csv3],
        "weights_raw": [float(args.w1), float(args.w2), float(args.w3)],
        "weights_normalized": [float(x) for x in w],
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
        "threshold": args.threshold,
        "calibration_bins": calib_bins,
        "timestamp_jst": datetime.datetime.now(
            datetime.timezone(datetime.timedelta(hours=9), "JST")
        ).isoformat(),
    }
    metrics_json_path = os.path.join(args.output_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics_json, f, indent=2, ensure_ascii=False)

    # ---------- 可視化 ----------
    # Confusion matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        np.array(cm),
        annot=True,
        fmt="d",
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        cmap="Blues",
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Ensemble)")
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=300, bbox_inches="tight")
    plt.close()

    # ROC curve (AMI=positive)
    if np.isfinite(roc_auc):
        plt.figure(figsize=(6, 6))
        plt.plot(
            fpr_ami,
            tpr_ami,
            label=f"ROC (AMI=pos) AUC = {roc_auc:.3f}",
            zorder=3,
            linewidth=2,
        )
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5, zorder=2)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver Operating Characteristic (AMI=positive, Ensemble)")
        plt.legend(loc="lower right")
        plt.grid(True, zorder=1)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        roc_path = os.path.join(args.output_dir, "roc_curve.png")
        plt.savefig(roc_path, dpi=300, bbox_inches="tight")
        plt.close()

    # PR curve (AMI=positive)
    if np.isfinite(pr_auc_ami) and precision_ami is not None:
        plt.figure(figsize=(6, 6))
        plt.plot(
            recall_ami,
            precision_ami,
            label=f"PR (AMI=pos) AUC = {pr_auc_ami:.3f}",
            zorder=3,
            linewidth=2,
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve (AMI=positive, Ensemble)")
        plt.legend(loc="lower left")
        plt.grid(True, zorder=1)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        pr_ami_path = os.path.join(args.output_dir, "pr_curve_AMI.png")
        plt.savefig(pr_ami_path, dpi=300, bbox_inches="tight")
        plt.close()

    # PR curve (tako=positive)
    if np.isfinite(pr_auc_tako) and precision_tako is not None:
        plt.figure(figsize=(6, 6))
        plt.plot(
            recall_tako,
            precision_tako,
            label=f"PR (tako=pos) AUC = {pr_auc_tako:.3f}",
            zorder=3,
            linewidth=2,
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision–Recall Curve (tako=positive, Ensemble)")
        plt.legend(loc="lower left")
        plt.grid(True, zorder=1)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        pr_tako_path = os.path.join(args.output_dir, "pr_curve_tako.png")
        plt.savefig(pr_tako_path, dpi=300, bbox_inches="tight")
        plt.close()

    # Calibration plot (AMI=pos)
    if frac_pos_ami is not None:
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "--", linewidth=1, label="Perfectly calibrated", alpha=0.7)
        plt.plot(mean_pred_ami, frac_pos_ami, marker="o", label="AMI (positive)")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed frequency")
        plt.title("Calibration plot (AMI=positive, Ensemble)")
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.legend(loc="upper left")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        calib_ami_path = os.path.join(args.output_dir, "calibration_AMI.png")
        plt.savefig(calib_ami_path, dpi=150, bbox_inches="tight")
        plt.close()

    # Calibration plot (tako=pos)
    if frac_pos_tako is not None:
        plt.figure(figsize=(6, 6))
        plt.plot([0, 1], [0, 1], "--", linewidth=1, label="Perfectly calibrated", alpha=0.7)
        plt.plot(mean_pred_tako, frac_pos_tako, marker="o", label="tako (positive)")
        plt.xlabel("Predicted probability")
        plt.ylabel("Observed frequency")
        plt.title("Calibration plot (tako=positive, Ensemble)")
        plt.grid(True, linestyle=":", linewidth=0.5)
        plt.legend(loc="upper left")
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.tight_layout()
        calib_tako_path = os.path.join(args.output_dir, "calibration_tako.png")
        plt.savefig(calib_tako_path, dpi=150, bbox_inches="tight")
        plt.close()

    print("\n[INFO] 保存完了:")
    print("  -", pred_csv_path)
    print("  -", metrics_csv_path)
    print("  -", metrics_json_path)
    print("  -", cm_path)


if __name__ == "__main__":
    main()

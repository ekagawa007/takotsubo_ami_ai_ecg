'''
tako_ami_ft2_convnextv2_20251002.py
-------------------------------------------------
Fine-tuning script for ConvNeXt-Base on 2-class classification task.
Loads pre-trained weights from FCMAE and fine-tunes on custom dataset.

Usage:
    3 class学習済みから2 classを初期化
    python tako_ami_ft2_convnextv2_20251002.py \
    --init-from-3cls /path/to/3cls_run/best.pth \
    --three-label-order tako AMI non_tako_ami \
    --train-dir /path/to/train \
    --val-dir /path/to/val \
    --output runs/ft2_convnextv2 \
    --batch-size 64

    FCMAEから2 classを初期化
    python tako_ami_ft2_convnextv2_20251002.py \
    --pretrained-checkpoint /path/to/fcmae/checkpoint.pth \
    --train-dir ... --val-dir ... --output ...

    
'''

import argparse
import os
import math
import random
import time
import datetime
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import PIL
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode, RandAugment, RandomErasing
from PIL import Image
import timm
from timm.data.mixup import Mixup
from tqdm.auto import tqdm
from collections import OrderedDict
import re
from timm.loss import SoftTargetCrossEntropy
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix


#PRETRAINED_CONVNEXTV2_FCMAE_CKPT_PATH = "/path/to/checkpoints/pretrain/convnextv2/fcmae_convnextv2_base_480/ckpt_0099.pth"



# ──────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────

# ECGなので上下左右のflipを外す
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

class PatientDataset(Dataset):
    def __init__(self, root_dir, is_train=True):
        self.samples = []
        self.is_train = is_train
        
        # ラベル名の定義
        self.label_names = ["tako", "AMI"]
        self.label2id = {n: i for i, n in enumerate(self.label_names)}
        self.id2label = {i: n for n, i in self.label2id.items()}




        # 患者ディレクトリを走査
        for patient_dir in os.listdir(root_dir):
            patient_path = os.path.join(root_dir, patient_dir)
            if not os.path.isdir(patient_path):
                continue

            # ラベルディレクトリを走査
            for label_dir in os.listdir(patient_path):
                label_path = os.path.join(patient_path, label_dir)
                if not os.path.isdir(label_path):
                    continue

                # miscディレクトリは無視
                if label_dir == "misc":
                    continue

                # ラベルディレクトリ名からラベルを決定
                if label_dir == "tako":
                    label = self.label2id["tako"]
                elif label_dir == "AMI":
                    label = self.label2id["AMI"]
                else:
                    continue

                # 画像ファイルを取得（PNG形式のみ）
                for img_file in os.listdir(label_path):
                    if img_file.lower().endswith('.png'):
                        img_path = os.path.join(label_path, img_file)
                        self.samples.append((img_path, label))

        # データ拡張の設定
        if is_train:
            self.transform = transforms.Compose([
                #transforms.RandomHorizontalFlip(),
                #transforms.RandomVerticalFlip(),
                #transforms.RandomRotation(10),
                #transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                #transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                #transforms.RandomResizedCrop(480, scale=(0.8, 1.0)),
                RandAugmentNoFlip(
                    num_ops=2,
                    magnitude=9,
                    interpolation=InterpolationMode.BILINEAR,
                    fill=0,
                ),
                transforms.ToTensor(),               
                RandomErasing(
                    p=0.5,
                    scale=(0.02, 0.33),
                    ratio=(0.3, 3.3),
                    value='random',
                    inplace=False
                ),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                #transforms.Resize((480, 480)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return {
            "pixel_values": img,
            "labels": torch.tensor(label)
        }

# ──────────────────────────────────────────
# Model
# ──────────────────────────────────────────
def create_model(num_classes=2, img_size=480):
    """Create ConvNeXt-Base model for fine-tuning"""
    # Use ConvNeXt V2 which includes GRN
    model = timm.create_model('convnextv2_base', pretrained=False, num_classes=num_classes)
    return model

def remap_key(k: str) -> str:
    # 1) 余分なプレフィックスを落とす
    k = k.replace("_orig_mod.", "")
    k = k.replace("encoder.", "")

    # 2) token 単位で "name_3" → "name.3" に変換
    tokens = k.split(".")
    new_tokens = []
    for t in tokens:
        m = re.match(r"^([A-Za-z0-9_]+)_(\d+)$", t)
        if m:
            new_tokens.append(f"{m.group(1)}.{m.group(2)}")
        else:
            new_tokens.append(t)
    k = ".".join(new_tokens)
    return k

def load_fcmae_to_timm(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))

    # decoder/cls_token/pos_embed などは捨てる（バックボーンだけ使う）
    drop_substrings = ["decoder", "mask_token", "pos_embed", "cls_token", "head"]
    filtered = {}
    for k, v in sd.items():
        if any(s in k for s in drop_substrings):
            continue
        nk = remap_key(k)
        filtered[nk] = v

    msd = model.state_dict()
    loadable = {k: v for k, v in filtered.items()
                if k in msd and getattr(msd[k], "shape", None) == getattr(v, "shape", None)}

    missing = [k for k in msd.keys() if k not in loadable]
    unexpected = [k for k in filtered.keys() if k not in msd]

    print(f"Will load: {len(loadable)} keys")
    print(f"Missing: {len(missing)} | Unexpected (ignored): {len(unexpected)}")
    print("Missing (examples):", missing[:10])

    model.load_state_dict(loadable, strict=False)
    return model

# 追加: 3→2クラス用ローダ
def load_3cls_finetuned_to_2cls(model_2cls, ckpt_path_3cls, three_label_order=("tako","AMI","non_tako_ami")):
    """
    3クラスで学習済み ckpt を読み、backbone はそのまま、head は
    ["tako","AMI"] の行だけを抜き出して 2クラスヘッドに移植する。
    """
    ckpt = torch.load(ckpt_path_3cls, map_location="cpu")
    sd = ckpt.get("model", ckpt.get("state_dict", ckpt))
    msd = (model_2cls._orig_mod if hasattr(model_2cls, "_orig_mod") else model_2cls).state_dict()

    # 3クラスの順序から 2クラスで保持するインデックスを作る
    # three_label_order は 3クラス学習時の train_ds.label_names の順序
    keep_names = ["tako", "AMI"]
    for n in keep_names:
        if n not in three_label_order:
            raise ValueError(f'"{n}" is missing in --three-label-order: {three_label_order}')
    keep_idx = [three_label_order.index(n) for n in keep_names]  # 例: [0,1]

    # head のキー推定（timm convnextv2 は head.fc）
    head_w_key = None
    head_b_key = None
    for k in sd.keys():
        if k.endswith("head.fc.weight"):
            head_w_key = k
        if k.endswith("head.fc.bias"):
            head_b_key = k

    new_sd = {}

    # まずbackbone系をそのまま（shapeが合うものだけ）入れる
    for k, v in sd.items():
        if (k == head_w_key) or (k == head_b_key):
            continue
        if k in msd and getattr(msd[k], "shape", None) == getattr(v, "shape", None):
            new_sd[k] = v

    # ヘッド（線形）の取り扱い：3→2へ縦方向だけスライス
    if head_w_key is not None and head_b_key is not None:
        w3 = sd[head_w_key]  # [3, D]
        b3 = sd[head_b_key]  # [3]
        w2 = w3[keep_idx, :] # [2, D]
        b2 = b3[keep_idx]    # [2]

        # 宛先キー（2クラスモデル側）を探す
        dst_w_key = None
        dst_b_key = None
        for k in msd.keys():
            if k.endswith("head.fc.weight"):
                dst_w_key = k
            if k.endswith("head.fc.bias"):
                dst_b_key = k

        if dst_w_key is None or dst_b_key is None:
            raise RuntimeError("2-class model head.fc.* not found")

        # 形が合えば採用（out_features=2 想定）
        if msd[dst_w_key].shape == w2.shape and msd[dst_b_key].shape == b2.shape:
            new_sd[dst_w_key] = w2
            new_sd[dst_b_key] = b2
        else:
            print("Warning: head shape mismatch, fallback to backbone-only init.")
    else:
        print("Warning: head weight/bias not found in 3-class checkpoint; fallback to backbone-only init.")

    # strict=False でロード
    (model_2cls._orig_mod if hasattr(model_2cls, "_orig_mod") else model_2cls).load_state_dict(new_sd, strict=False)
    return model_2cls



# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("ConvNeXt Fine-tuner for 2-class classification")
    p.add_argument('--pretrained-checkpoint', default=None, help='Path to FCMAE pre-trained checkpoint')
    p.add_argument('--train-dir', required=True, help='Training data directory')
    p.add_argument('--val-dir', required=True, help='Validation data directory')
    p.add_argument('--output', required=True, help='Output directory for checkpoints and logs')
    p.add_argument('--img-size', type=int, default=480, help='Image size for training (default: 480)')
    p.add_argument('--batch-size', type=int, default=32, help='Batch size (default: 32)')
    p.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    p.add_argument('--epochs', type=int, default=50, help='Number of epochs (default: 50)')
    p.add_argument('--head-only-epochs', type=int, default=3, help='Number of epochs to train only classification head (default: 3)')
    p.add_argument('--num-workers', type=int, default=16, help='Number of workers (default: 16)')
    p.add_argument('--seed', type=int, default=None, help='Random seed (default: None)')
    p.add_argument('--resume', type=str, help='Resume from checkpoint')
    # parse_args() の引数に追加（お好みでデフォルト調整）
    p.add_argument('--use-swa', action='store_true', help='Enable Stochastic Weight Averaging')  # NEW
    p.add_argument('--swa-start-epoch', type=int, default=12, help='Epoch to start SWA (full finetune phase)')  # NEW
    p.add_argument('--swa-lr', type=float, default=5e-6, help='SWA learning rate')  # NEW
    p.add_argument('--swa-update-bn-every', type=int, default=0, help='Update BN every N epochs during SWA (0=end only)')  # NEW
    p.add_argument('--use-sam', action='store_true', help='Enable Sharpness-Aware Minimization')
    p.add_argument('--sam-rho', type=float, default=0.05, help='SAM neighborhood size (rho)')
    p.add_argument('--sam-adaptive', action='store_true', help='Use ASAM (scale perturbation by |w|)')
    p.add_argument('--find-lr', action='store_true', help='Run LR range test and exit')
    p.add_argument('--find-lr-end', type=float, default=1e-2, help='End LR (backbone) for LR range test')
    p.add_argument('--init-from-3cls', type=str, default=None, help='Path to 3-class fine-tuned checkpoint (best/latest). If set, use it to init 2-class model.')
    p.add_argument('--three-label-order', type=str, nargs='*', default=["tako","AMI","non_tako_ami"], help='Label order used in the 3-class training checkpoint.')
    p.add_argument('--mixup-prob', type=float, default=0.5, help='Probability of applying mixup (default: 0.5)')
    p.add_argument('--mixup-alpha', type=float, default=0.3, help='Mixup alpha parameter (default: 0.3)')
    p.add_argument('--cutmix-alpha', type=float, default=0.3, help='CutMix alpha parameter (default: 0.3)')
    p.add_argument('--switch-prob', type=float, default=0.5, help='Probability of switching between mixup and cutmix (default: 0.5)')
    p.add_argument('--label-smoothing', type=float, default=0.1, help='Label smoothing factor (default: 0.1)')
    
    return p.parse_args()

# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────
# ──────────────────────────────────────────
# LR Range Test (Learning Rate Finder)
# ──────────────────────────────────────────
@torch.no_grad()
def _count_train_batches(train_loader):
    try:
        return len(train_loader)
    except:
        return 1000  # fallback

def lr_range_test(
    model,
    train_loader,
    backbone_params,
    head_params,
    device,
    start_lr=1e-7,
    end_lr=1.0,
    num_iter=None,
    smooth_beta=0.98,
    max_loss_increase=4.0,
    disable_mixup=True,
    disable_random_erasing=True,
):
    """
    Returns: dict with keys:
      'lrs': list, 'losses': list, 'smoothed': list,
      'suggest_backbone_lr': float, 'suggest_head_lr': float
    """
    model.train()

    # 一時的に Mixup/Erasing を無効化（collate/transformに依存するため簡易対応）
    # → collate は hard label を使う簡易版に差し替える
    def _plain_collate(batch):
        x = torch.stack([b["pixel_values"] for b in batch])
        y = torch.tensor([b["labels"] for b in batch])
        return {"pixel_values": x, "labels": y}

    from torch.utils.data import DataLoader
    plain_loader = DataLoader(
        train_loader.dataset,
        batch_size=train_loader.batch_size,
        shuffle=True,
        num_workers=train_loader.num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=_plain_collate,
        persistent_workers=(train_loader.num_workers > 0),
        prefetch_factor=(4 if train_loader.num_workers > 0 else None),
    )

    # 変換側のRandomErasingを無効化（簡易：transformを舐めて置換）
    # 注意: Transformの構成に依存します。うまく見つからない場合はそのままでもOK。
    if disable_random_erasing and hasattr(plain_loader.dataset, "transform"):
        tfm = plain_loader.dataset.transform
        if isinstance(tfm, transforms.Compose):
            new_tfm = []
            for t in tfm.transforms:
                if isinstance(t, RandomErasing):
                    continue
                new_tfm.append(t)
            plain_loader.dataset.transform = transforms.Compose(new_tfm)

    # Optimizer（探索専用）：backboneとheadの比率を保った2PG
    tmp_opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": start_lr, "weight_decay": 0.0},
            {"params": head_params,     "lr": start_lr * 10.0, "weight_decay": 0.0},
        ]
    )
    criterion = nn.CrossEntropyLoss()

    # AMP（探索は安定重視で半精度でもOK）
    use_cuda = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32

    # 何iter回すか
    total_batches = _count_train_batches(plain_loader)
    if num_iter is None:
        num_iter = min(1000, total_batches)

    # LRを指数的に上げる係数
    lr_mult = (end_lr / start_lr) ** (1 / (num_iter - 1))

    lrs, losses, smooth_losses = [], [], []
    best_loss, avg_loss = float("inf"), 0.0
    it = 0

    for batch in plain_loader:
        it += 1
        if it > num_iter:
            break

        imgs = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        tmp_opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        loss.backward()
        # クリッピング少量
        clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
        tmp_opt.step()

        # 損失の指数移動平均でスムージング
        loss_val = loss.item()
        avg_loss = smooth_beta * avg_loss + (1 - smooth_beta) * loss_val
        smooth = avg_loss / (1 - smooth_beta ** it)

        # ログ
        current_backbone_lr = tmp_opt.param_groups[0]["lr"]
        lrs.append(current_backbone_lr)
        losses.append(loss_val)
        smooth_losses.append(smooth)
        best_loss = min(best_loss, smooth)

        # 発散検知：最良の何倍か超えたら打ち切り
        if smooth > best_loss * max_loss_increase:
            break

        # LRを次へ（両PGとも倍率維持）
        for i, pg in enumerate(tmp_opt.param_groups):
            base = start_lr if i == 0 else start_lr * 10.0
            # 次のiterのlr = 現在 * lr_mult
            pg["lr"] = pg["lr"] * lr_mult

    # 推奨LR：最急降下（勾配が最も負）の直前 or 最小スムーズ損失の1/5～1/10
    # ここでは簡易に「最小スムーズ損失indexの 2~3個 手前」を使う
    import numpy as np
    s = np.array(smooth_losses)
    i_min = int(s.argmin())
    i_pick = max(0, i_min - 2)  # 2つ手前
    suggest_backbone = float(lrs[i_pick])
    suggest_head = suggest_backbone * 10.0

    return {
        "lrs": lrs,
        "losses": losses,
        "smoothed": smooth_losses,
        "suggest_backbone_lr": suggest_backbone,
        "suggest_head_lr": suggest_head,
    }



def set_seed(seed):
    if seed is None:
        print("Seed not set (non-deterministic run).")
        return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ====== (E) calculate_accuracy を soft/hard 両対応にしておくと堅牢 ======
def calculate_accuracy(outputs, targets):
    """
    outputs: (B, C) のロジット
    targets: (B,) の整数 or (B, C) の one-hot/soft
    """
    _, predicted = torch.max(outputs, 1)               # (B,)
    if targets.dim() == 2:                             # soft/one-hot のとき
        targets = targets.argmax(dim=1)                # (B,)
    correct = (predicted == targets).sum().item()
    return correct / targets.size(0)

# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
def main():
    args = parse_args()
    if (args.resume is None) and (args.pretrained_checkpoint is None) and (args.init_from_3cls is None):
        raise ValueError("Provide either --init-from-3cls or --pretrained-checkpoint (or use --resume).")
    if (args.pretrained_checkpoint is not None) and (args.init_from_3cls is not None):
        print("[INFO] Both sources provided. Prioritizing --init-from-3cls and ignoring --pretrained-checkpoint.")
    set_seed(args.seed)

    IMG_SIZE = args.img_size
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Memory optimizations
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("CUDA optimizations enabled")


    # Load datasets
    train_ds = PatientDataset(args.train_dir, is_train=True)
    val_ds = PatientDataset(args.val_dir, is_train=False)
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Classes: {train_ds.label_names}")
    print(f"Class mapping: {train_ds.label2id}")
    
    # ラベル分布を表示する関数
    def print_label_distribution(dataset, dataset_name):
        labels = [label for _, label in dataset.samples]
        total = len(labels)

        print(f"\n{dataset_name}のラベル分布:")
        for label_name in dataset.label_names:
            count = labels.count(dataset.label2id[label_name])
            print(f"{label_name}: {count} ({count/total*100:.1f}%)")
        print(f"合計: {total}")

    # 各データセットのラベル分布を表示
    print_label_distribution(train_ds, "学習データ")
    print_label_distribution(val_ds, "検証データ")
    
    # クラス数を取得
    num_classes = len(train_ds.label_names)

    # ── WeightedRandomSampler（学習のみ）を自動生成 ──
    # train_ds.samples: List[(img_path, label)]
    train_labels = torch.tensor([lbl for _, lbl in train_ds.samples], dtype=torch.long)

    # クラス出現数（欠番クラスがあっても安全に）
    class_counts = torch.bincount(train_labels, minlength=num_classes).float().clamp_min(1.0)  # ゼロ割回避

    # 逆頻度（合計/各クラス）をベースにサンプル重みを作成
    class_weights = class_counts.sum() / class_counts                 # shape: (C,)
    sample_weights = class_weights[train_labels].double()             # shape: (N,)

    # エポックあたりのサンプル数（バッチで割り切れるように）
    if len(train_ds) >= args.batch_size:
        epoch_samples = (len(train_ds) // args.batch_size) * args.batch_size
        drop_last_train = True
    else:
        # 極小データセットでも必ず1バッチ以上回るように
        epoch_samples = args.batch_size
        drop_last_train = False

    # 再現性が必要なら generator にシードを渡す（任意）
    _sampler_gen = None
    if args.seed is not None:
        _sampler_gen = torch.Generator()
        _sampler_gen.manual_seed(int(args.seed))

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=epoch_samples,          # 同数にするなら len(train_ds)
        replacement=True,
        generator=_sampler_gen
    )
    
    # CutMixの設定
    mixup_fn = Mixup(
        mixup_alpha=args.mixup_alpha,
        cutmix_alpha=args.cutmix_alpha,
        cutmix_minmax=None,
        prob=args.mixup_prob,
        switch_prob=args.switch_prob,
        mode='batch',
        label_smoothing=args.label_smoothing,
        num_classes=num_classes
    )
    
    # ====== 学習用 collate：hard_labels も返す（精度計算用） ======
    def collate_fn_train(batch):
        x = torch.stack([b["pixel_values"] for b in batch])          # (B, C, H, W)
        hard_y = torch.tensor([b["labels"] for b in batch])          # (B,) int labels
        x_mix, y_soft = mixup_fn(x, hard_y)                          # y_soft: (B, C) soft targets
        return {"pixel_values": x_mix, "labels": y_soft, "hard_labels": hard_y}

    def collate_fn_eval(batch):
        x = torch.stack([b["pixel_values"] for b in batch])
        y = torch.tensor([b["labels"] for b in batch])               # ← 検証は整数ラベル
        return {"pixel_values": x, "labels": y}

    # DataLoader
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=False, #True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=drop_last_train,
        collate_fn=collate_fn_train,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=(4 if args.num_workers > 0 else None)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        collate_fn=collate_fn_eval,
        persistent_workers=(args.num_workers > 0), prefetch_factor=(4 if args.num_workers > 0 else None)
    )

    # Create model
    num_classes = len(train_ds.label_names)
    model = create_model(num_classes=num_classes, img_size=IMG_SIZE)
        
    # 3クラス ckpt から初期化 or FCMAE から初期化
    if args.init_from_3cls:
        print(f"[Init] Loading 2-class model from 3-class checkpoint: {args.init_from_3cls}")
        model = load_3cls_finetuned_to_2cls(model, args.init_from_3cls, tuple(args.three_label_order))
    else:
        print(f"[Init] Loading backbone from FCMAE checkpoint: {args.pretrained_checkpoint}")
        model = load_fcmae_to_timm(model, args.pretrained_checkpoint)
    model.to(device)
    base_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    # Compile model for faster training (PyTorch 2.0+)
    try:
        model = torch.compile(model, mode="default")
        print("Model compiled successfully for faster training")
    except Exception as e:
        print(f"Model compilation failed: {e}")
        print("Continuing without compilation")

    # Create output directory
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === LR Finder ===
    if args.find_lr:
        # SAM/SWAは探索では使わない（学習ループ前なので未使用のはず）
        # collate関数もplainに置き換えるため、train_loaderを使わずdatasetから作り直しは関数内で実施

        # 分類ヘッド/バックボーンのparam groups（compileラッパの中身に注意）
        base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        head_params = list(base_model.head.parameters()) if hasattr(base_model, 'head') else list(base_model.classifier.parameters())
        backbone_params = [p for p in base_model.parameters() if p not in set(head_params)]

        print("[LR Finder] start...")
        result = lr_range_test(
            model=base_model,
            train_loader=train_loader,
            backbone_params=backbone_params,
            head_params=head_params,
            device=device,
            start_lr=1e-7,
            end_lr=args.find_lr_end,   # 例: 1e-2 まで指数増加
            num_iter=min(1000, len(train_loader)),
        )
        # 出力：CSV保存
        lr_file = out_dir / "lr_finder.csv"
        with open(lr_file, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iter", "backbone_lr", "loss", "smoothed_loss"])
            for i, (lr, lo, sm) in enumerate(zip(result["lrs"], result["losses"], result["smoothed"])):
                w.writerow([i+1, f"{lr:.8e}", f"{lo:.6f}", f"{sm:.6f}"])

        print(f"[LR Finder] suggested backbone lr: {result['suggest_backbone_lr']:.3e}")
        print(f"[LR Finder] suggested head     lr: {result['suggest_head_lr']:.3e}")
        print(f"[LR Finder] curve saved to: {str(lr_file)}")
        return  # ここで終了（本訓練は行わない）

    # ====== (B) 損失関数を分ける ======
    train_criterion = SoftTargetCrossEntropy()   # 学習（Mixupあり、soft target）
    val_criterion   = nn.CrossEntropyLoss()      # 検証（Mixupなし、int label）

    # Separate optimizers for head-only and full training
    head_params = list(model.head.parameters()) if hasattr(model, 'head') else list(model.classifier.parameters())
    backbone_params = [p for p in model.parameters() if p not in set(head_params)]
    
    # Head-only optimizer (higher learning rate for classification head)
    head_optimizer = torch.optim.AdamW(head_params, lr=args.lr * 10, weight_decay=0.01)
    
    # Full model optimizer
    full_optimizer = torch.optim.AdamW(
        [{'params': backbone_params,  'lr': args.lr},
        {'params': head_params,'lr': args.lr * 10}],
        weight_decay=0.01
    )

    # Learning rate schedulers
    head_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(head_optimizer, T_max=args.head_only_epochs)
    full_scheduler = None
    warmup_full = 3
    
    # オプティマイザ・スケジューラの定義の直後（head/fullなどの下）
    swa_model = None
    swa_scheduler = None
    if args.use_swa:
        # ★分類ヘッド込みで平均するため base_model を渡す（compileラッパは渡さない）
        swa_model = AveragedModel(base_model, avg_fn=None)  # NEW: パラメタの単純平均
        # SWA 期間の LR スケジューラ
        swa_scheduler = SWALR(full_optimizer, swa_lr=args.swa_lr)  # NEW
        print(f"[SWA] enabled: start_epoch={args.swa_start_epoch}, swa_lr={args.swa_lr}")

    # Mixed Precision    
    use_cuda = torch.cuda.is_available()
    dtype = torch.bfloat16 if use_cuda else torch.float32

    scaler = torch.amp.GradScaler(enabled=(use_cuda and dtype == torch.float16))

    # Enable gradient accumulation for larger effective batch size
    accumulation_steps = 1  # Can be increased if needed

    # Initialize training history CSV
    history_file = out_dir / 'training_history.csv'
    if not history_file.exists():
        with open(history_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'epoch', 'train_loss', 'train_acc', 'val_loss', 'val_acc', 
                'precision_macro', 'recall_macro', 'f1_macro', 'lr'
            ])

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_acc = 0.0
    full_sched_state = None

    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        if 'head_optimizer' in ckpt:
            head_optimizer.load_state_dict(ckpt['head_optimizer'])
        if 'full_optimizer' in ckpt:
            full_optimizer.load_state_dict(ckpt['full_optimizer'])
        if 'head_scheduler' in ckpt:
            head_scheduler.load_state_dict(ckpt['head_scheduler'])
        # full_scheduler はここでは load しない
        if 'full_scheduler' in ckpt and ckpt['full_scheduler'] is not None:
            full_sched_state = ckpt['full_scheduler']   # ← ステートだけ保持
           
        start_epoch = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        print(f"Resuming from epoch {start_epoch}")

    # 再開時に、すでに full フェーズに入っているなら full_scheduler をここで必ず用意
    if start_epoch > args.head_only_epochs + 1 and full_scheduler is None:
        # 先に LR/initial_lr を希望値へ（例）
        backbone_lr = min(args.lr * 0.1, 3e-5)
        head_lr     = max((args.lr * 10) * 0.5, 1e-5)  # 再開時は好みで調整

        full_optimizer.param_groups[0]['lr'] = backbone_lr
        full_optimizer.param_groups[0]['initial_lr'] = backbone_lr
        full_optimizer.param_groups[1]['lr'] = head_lr
        full_optimizer.param_groups[1]['initial_lr'] = head_lr

        full_scheduler = SequentialLR(
            full_optimizer,
            schedulers=[
                LinearLR(full_optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_full),
                CosineAnnealingLR(full_optimizer, T_max=args.epochs - args.head_only_epochs - warmup_full),
            ],
            milestones=[warmup_full],
        )

        if full_sched_state is not None:
            try:
                full_scheduler.load_state_dict(full_sched_state)
            except Exception as e:
                print(f"Warning: could not load full_scheduler state: {e}")

    # Training loop
    for epoch in range(start_epoch, args.epochs + 1):
        swa_val_acc = None
        # ====== 切替ハンドリング：epoch == head_only_epochs の「直後」に実行 ======
        # エポックの最後ではなく「開始時」に行うのが安全です（ログ上も分かりやすい）
        if epoch == args.head_only_epochs + 1:
            # (A) optimizer を作り直し
            full_optimizer = torch.optim.AdamW(
                [
                    {'params': backbone_params, 'lr': args.lr},
                    {'params': head_params,     'lr': args.lr *10},  # ← ここが肝
                ],
                weight_decay=0.01
            )
            # (B) scheduler も作り直し
            warmup_full = 3
            full_scheduler = SequentialLR(
                full_optimizer,
                schedulers=[
                    LinearLR(full_optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_full),
                    CosineAnnealingLR(full_optimizer, T_max=args.epochs - args.head_only_epochs - warmup_full),
                ],
                milestones=[warmup_full],
            )
            
            # ★ (C) SWA を使うなら SWALR も “新しい optimizer” で作り直す
            if args.use_swa:
                swa_scheduler = SWALR(full_optimizer, swa_lr=args.swa_lr)


        # ▼ ここでSWA期間かを判定（“フル学習に入ってから”が前提）
        in_swa_phase = args.use_swa and (epoch >= args.swa_start_epoch)  # NEW

        # スケジューラの選択（SWA期間中は SWALR を使う）
        is_head_only = epoch <= args.head_only_epochs
        current_optimizer = head_optimizer if is_head_only else full_optimizer
        if is_head_only:
            current_scheduler = head_scheduler
        else:
            # full フェーズ
            if in_swa_phase and swa_scheduler is not None:
                current_scheduler = swa_scheduler  # NEW
            else:
                current_scheduler = full_scheduler
       
        # Freeze/unfreeze backbone
        for param in backbone_params:
            param.requires_grad = not is_head_only
        
        print(f"Epoch {epoch}/{args.epochs} - "
          f"{'Head-only' if is_head_only else ('Full+SWA' if in_swa_phase else 'Full')} training")
        

        def _params_to_perturb(m):
            # 今、更新対象（requires_grad=True）のパラメタだけ
            for p in m.parameters():
                if p.requires_grad and p.grad is not None:
                    yield p


        # Training phase
        model.train()
        train_loss = 0.0
        train_acc = 0.0
        train_steps = 0
        
        train_prog = tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs} [Train]', leave=False)
        # ====== (C) 学習ループ：損失と精度の計算を分ける ======
        # 学習ループ（train）
        for batch_idx, batch in enumerate(train_prog):
            imgs = batch['pixel_values'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            hard_labels = batch['hard_labels'].to(device, non_blocking=True)     # int labels (B,)

            with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_cuda):
                outputs = model(imgs)
                loss = train_criterion(outputs, labels)

            use_sam_now = args.use_sam and (epoch > args.head_only_epochs)
            current_optimizer.zero_grad(set_to_none=True)

            # ===== 1st forward/backward =====
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                # SAM のノルム計算は unscale 後の「実勾配」で行う
                scaler.unscale_(current_optimizer)
            else:
                loss.backward()


            if use_sam_now:
                # --- 対象パラメータを一度だけ列挙 ---
                params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]

                # --- grad norm を一度だけ計算（ASAMは |w|⊙g のノルム）---
                if args.sam_adaptive:
                    grad_norm = torch.norm(torch.stack([(p.detach().abs() * p.grad).norm(p=2) for p in params]), p=2).clamp_(min=1e-12)
                else:
                    grad_norm = torch.norm(torch.stack([p.grad.norm(p=2) for p in params]), p=2).clamp_(min=1e-12)

                # --- 摂動 ε を一度だけ付与 ---
                eps_buf = []
                with torch.no_grad():
                    for p in params:
                        scale = p.detach().abs() if args.sam_adaptive else 1.0
                        eps = (args.sam_rho / grad_norm) * scale * p.grad
                        p.add_(eps)
                        eps_buf.append(eps)

                current_optimizer.zero_grad(set_to_none=True)                        

                # --- 2nd forward/backward ---
                with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_cuda):
                    outputs_2 = model(imgs)
                    loss_2 = train_criterion(outputs_2, labels)

                if scaler.is_enabled():
                    scaler.scale(loss_2).backward()
                    scaler.unscale_(current_optimizer)  # clip前
                else:
                    loss_2.backward()

                # --- 摂動を元に戻す・step ---
                with torch.no_grad():
                    for p, eps in zip(params, eps_buf):
                        p.sub_(eps)

                clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                if scaler.is_enabled():
                    scaler.step(current_optimizer)
                    scaler.update()
                else:
                    current_optimizer.step()

                outputs = outputs_2  # ログ用
            else:
                # 通常更新
                if scaler.is_enabled():
                    scaler.unscale_(current_optimizer)
                clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=1.0)
                if scaler.is_enabled():
                    scaler.step(current_optimizer)
                    scaler.update()
                else:
                    current_optimizer.step()

            train_loss += loss.item()
            train_acc  += calculate_accuracy(outputs, hard_labels)
            train_steps += 1

        avg_train_loss = train_loss / train_steps
        avg_train_acc = train_acc / train_steps
        
        # === SWA の平均更新（エポック終了時に1回でOK） ===
        if in_swa_phase and (swa_model is not None):
            swa_model.update_parameters(base_model)  # NEW
            # お好みでBN更新を途中で入れたい場合（重いので通常は最後だけ）
            if args.swa_update_bn_every > 0 and (epoch % args.swa_update_bn_every == 0):
                print("[SWA] updating BN stats mid-training...")
                # BN更新時は学習時と同じ前処理/AMPでOK（推論モードではなくtrainモードで統計を計算）
                swa_model.train()
                update_bn(train_loader, swa_model, device=device)  # NEW
                swa_model.eval()

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_acc = 0.0
        val_steps = 0
        
        # ====== (D) 検証ループ：通常のCEで整数ラベル ======
        with torch.no_grad():
            val_prog = tqdm(val_loader, desc=f'Epoch {epoch}/{args.epochs} [Val]', leave=False)
            for batch in val_prog:
                imgs = batch['pixel_values'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)  # int label

                with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_cuda):
                    outputs = model(imgs)
                    loss = val_criterion(outputs, labels)

                val_loss += loss.item()
                val_acc += calculate_accuracy(outputs, labels)
                val_steps += 1

        avg_val_loss = val_loss / val_steps
        avg_val_acc = val_acc / val_steps
        
        # Update learning rate (ログ用の現在LRだけ取得)
        current_lr = current_optimizer.param_groups[0]['lr']

        # Print results
        print(f'Epoch {epoch}/{args.epochs}:')
        print(f'  Train - Loss: {avg_train_loss:.4f}, Acc: {avg_train_acc:.4f}')
        print(f'  Val   - Loss: {avg_val_loss:.4f}, Acc: {avg_val_acc:.4f}')
        print(f'  LR (current opt, all PG): {[pg["lr"] for pg in current_optimizer.param_groups]}')

        
        # Calculate detailed metrics for validation set
        model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['pixel_values'].to(device, non_blocking=True)
                labels = batch['labels'].to(device, non_blocking=True)
                
                with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_cuda):
                    outputs = model(imgs)
                
                predictions = torch.argmax(outputs, dim=1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        counts = torch.bincount(torch.tensor(all_predictions), minlength=num_classes)
        print("Pred histogram:", {train_ds.label_names[i]: int(counts[i]) for i in range(num_classes)})

        # Calculate detailed metrics
        
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average=None, labels=range(num_classes), zero_division=0
        )
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average="macro", zero_division=0
        )
        
        print(f'  Validation Metrics:')
        print(f'    Precision (macro): {precision_macro:.4f}')
        print(f'    Recall (macro): {recall_macro:.4f}')
        print(f'    F1 (macro): {f1_macro:.4f}')
        
        # Per-class metrics
        for i, class_name in enumerate(train_ds.label_names):
            print(f'    {class_name}: P={precision[i]:.4f}, R={recall[i]:.4f}, F1={f1[i]:.4f}')
        
        # Save to training history CSV with detailed metrics
        with open(history_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch, 
                f'{avg_train_loss:.4f}', 
                f'{avg_train_acc:.4f}', 
                f'{avg_val_loss:.4f}', 
                f'{avg_val_acc:.4f}', 
                f'{precision_macro:.4f}',
                f'{recall_macro:.4f}',
                f'{f1_macro:.4f}',
                f'{current_lr:.6f}'
            ])
        
        # 既存 ckpt 保存の直前で base_model を使用
        base_model = model._orig_mod if hasattr(model, '_orig_mod') else model  # 念のため再取得
        # Save checkpoint
        ckpt = {
            'model': base_model.state_dict(),
            'head_optimizer': head_optimizer.state_dict(),
            'full_optimizer': full_optimizer.state_dict(),
            'head_scheduler': head_scheduler.state_dict(),
            'full_scheduler': full_scheduler.state_dict() if full_scheduler is not None else None,
            'epoch': epoch,
            'best_val_acc': best_val_acc,
            'args': vars(args),
        }

        if swa_model is not None:
            ckpt['swa_model'] = swa_model.state_dict()  # ← 保存の前に

        torch.save(ckpt, out_dir / f'ckpt_{epoch:04d}.pth')
        torch.save(ckpt, out_dir / 'latest.pth')

        # best の更新：SWAありならSWAの成績も比較
        candidate_best = avg_val_acc
        candidate_tag  = 'vanilla'
        if swa_val_acc is not None and swa_val_acc > candidate_best:
            candidate_best = swa_val_acc
            candidate_tag  = 'swa'

        if candidate_best > best_val_acc:
            best_val_acc = candidate_best
            torch.save(ckpt, out_dir / 'best.pth')
            print(f'  New best validation accuracy ({candidate_tag}): {best_val_acc:.4f}')

        print(f'  LR: {current_lr:.6f}', " | all LRs:", [pg['lr'] for pg in current_optimizer.param_groups])

        # ====== エポック末尾（ログ出力の後など）での scheduler.step() は従来通り ======
        if current_scheduler is not None:
            current_scheduler.step()

    print(f"Training completed! Best validation accuracy: {best_val_acc:.4f}")
    

    
    # === SWA を使っていた場合は、最後にBNを確定更新して保存 ===
    if args.use_swa and (swa_model is not None):
        print("[SWA] final BN update & save swa_best.pth ...")
        swa_model.to(device)
        swa_model.train()
        update_bn(train_loader, swa_model, device=device)  # 最終確定
        swa_model.eval()
        torch.save({'model': swa_model.state_dict(),
                    'epoch': epoch,
                    'args': vars(args)}, out_dir / 'swa_best.pth')  # NEW

    # Final evaluation on validation set
    print("\n=== Final Validation Results ===")
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in val_loader:
            imgs = batch['pixel_values'].to(device, non_blocking=True)
            labels = batch['labels'].to(device, non_blocking=True)
            
            with torch.amp.autocast(device_type='cuda', dtype=dtype, enabled=use_cuda):
                outputs = model(imgs)
            
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate final metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average=None, labels=range(num_classes)
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        all_labels, all_predictions, average="macro"
    )
    
    print(f"Final Validation Metrics:")
    print(f"  Accuracy: {avg_val_acc:.4f}")
    print(f"  Precision (macro): {precision_macro:.4f}")
    print(f"  Recall (macro): {recall_macro:.4f}")
    print(f"  F1 (macro): {f1_macro:.4f}")
    
    # Per-class final metrics
    print(f"\nPer-class metrics:")
    for i, class_name in enumerate(train_ds.label_names):
        print(f"  {class_name}: P={precision[i]:.4f}, R={recall[i]:.4f}, F1={f1[i]:.4f}")

if __name__ == '__main__':
    main()
    print('Done. ', datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9), 'JST')))

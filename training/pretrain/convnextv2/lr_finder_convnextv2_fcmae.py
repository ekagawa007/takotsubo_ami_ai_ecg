#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ConvNeXtV2 + FCMAE 用 Learning Rate Finder (LR range test)

・あなたの train_convnextv2_fcmae.py と同等の前処理/サンプラ/モデル設定
・指数スケールで LR を start->end に増加
・平滑化損失(EMA)を記録し、急激な発散で自動停止
・CSV 出力（step, lr, loss, smoothed_loss）と推奨LRの標準出力

使い方(例):
python lr_finder_convnextv2_fcmae.py \
  --train-dirs /path/to/train1 /path/to/train2 \
  --img-size 480 \
  --batch-size 64 \
  --start-lr 1e-6 --end-lr 1e-2 \
  --max-steps 1000 \
  --output runs/fcmae_convnextv2_base_480/lr_finder
"""

import argparse, os, math, random, csv, itertools, datetime, time
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from torch.amp import GradScaler, autocast
from PIL import Image
from tqdm.auto import tqdm

from mae_fcmae_wrapper import FCMAE  # あなたのラッパーに依存

# ──────────────────────────────────────────
# Dataset (train_convnextv2_fcmae.py と同等)
# ──────────────────────────────────────────
class PretrainDataset(Dataset):
    IMG_EXT = ('.png', '.jpg', '.jpeg')
    def __init__(self, root, transform=None, cache_size=500):
        if not os.path.exists(root):
            raise FileNotFoundError(root)
        self.files = self._scan(root)
        self.transform = transform
        self.cache = {}
        self.cache_size = cache_size

    @staticmethod
    def _scan(root):
        paths = []
        for r, _, fns in os.walk(root):
            if 'misc' in r:
                continue
            for fn in fns:
                if fn.lower().endswith(PretrainDataset.IMG_EXT):
                    paths.append(os.path.join(r, fn))
        if not paths:
            raise RuntimeError(f'No images found under {root}')
        return paths

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        res = {"pixel_values": img}
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[idx] = res
        return res

# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("LR Finder for ConvNeXt-V2-FCMAE")
    p.add_argument('--train-dirs', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--img-size', type=int, default=480, choices=[384, 480])
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)

    p.add_argument('--start-lr', type=float, default=1e-6)
    p.add_argument('--end-lr', type=float, default=1e-2)
    p.add_argument('--max-steps', type=int, default=1000, help='最大イテレーション(短時間用に小さめ推奨)')
    p.add_argument('--smoothing-beta', type=float, default=0.98, help='EMA 平滑化係数')
    p.add_argument('--divergence-factor', type=float, default=5.0, help='best_loss の何倍で発散停止とみなすか')
    p.add_argument('--mask-ratio', type=float, default=0.75)
    return p.parse_args()

# ──────────────────────────────────────────
# Utils
# ──────────────────────────────────────────
def set_seed(seed):
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def build_loader(train_dirs, img_size, batch_size, num_workers):
    aug_tfms = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    # 並列構築は簡略化：大規模ならThreadPoolExecutorも可
    datasets = [PretrainDataset(p, aug_tfms) for p in train_dirs]
    train_ds = ConcatDataset(datasets)

    lengths = [len(d) for d in train_ds.datasets]
    cum = [0]
    for L in lengths:
        cum.append(cum[-1] + L)
    w = torch.empty(len(train_ds), dtype=torch.float)
    for i, L in enumerate(lengths):
        start, end = cum[i], cum[i+1]
        w[start:end] = 1.0 / L

    sampler = WeightedRandomSampler(w, num_samples=len(train_ds), replacement=True)

    loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                        num_workers=num_workers, pin_memory=True, drop_last=True,
                        persistent_workers=(num_workers > 0), prefetch_factor=4)
    return loader

# ──────────────────────────────────────────
# LR Range Test
# ──────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader = build_loader(args.train_dirs, args.img_size, args.batch_size, args.num_workers)

    # モデル (ConvNeXtV2 Base + FCMAE)
    model = FCMAE('convnextv2_base', img_size=args.img_size, patch_size=4, mask_ratio=args.mask_ratio)
    model.to(device)

    # torch.compile 追加
    try:
        model = torch.compile(model, mode="reduce-overheat")  # または mode="reduce-overhead"
        print("Model compiled successfully.")
    except Exception as e:
        print(f"torch.compile failed: {e}")

    # オプティマイザ（WDは本学習と合わせる）
    optim = torch.optim.AdamW(model.parameters(), lr=args.start_lr, weight_decay=0.05)

    # AMP 設定：CUDA があれば bf16 優先、なければ fp32
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    use_bf16 = (dtype == torch.bfloat16)
    scaler = GradScaler('cuda') if not use_bf16 else None

    # LR ステップ倍率（指数増加）
    num_steps = min(args.max_steps if hasattr(args, 'max-steps') else args.max_steps, args.max_steps)  # guard
    lr_mult = (args.end_lr / args.start_lr) ** (1 / max(1, num_steps - 1))
    current_lr = args.start_lr
    for pg in optim.param_groups:
        pg['lr'] = current_lr

    # 出力
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'lr_finder_history.csv'
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['step', 'lr', 'loss', 'smoothed_loss'])

    # ループ準備
    model.train()
    ema_beta = args.smoothing_beta
    avg_loss = 0.0
    best_loss = float('inf')
    best_lr = current_lr
    step = 0

    # DataLoader を必要ステップ分だけ回す（cycleで無限化）
    loop = itertools.islice(itertools.cycle(loader), num_steps)
    pbar = tqdm(loop, total=num_steps, desc='LR Range Test', leave=False)

    try:
        for batch in pbar:
            imgs = batch['pixel_values'].to(device, non_blocking=True)

            # forward
            with autocast('cuda', dtype=dtype):
                loss = model(imgs)

            # backward+step
            optim.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optim)
                scaler.update()
            else:
                loss.backward()
                optim.step()

            loss_item = loss.item()
            # EMA 平滑化損失
            avg_loss = ema_beta * avg_loss + (1 - ema_beta) * loss_item if step > 0 else loss_item
            smoothed = avg_loss / (1 - (ema_beta ** (step + 1)))

            # 記録
            with open(csv_path, 'a', newline='') as f:
                csv.writer(f).writerow([step, f'{current_lr:.8f}', f'{loss_item:.6f}', f'{smoothed:.6f}'])

            pbar.set_postfix(lr=f'{current_lr:.2e}', loss=f'{loss_item:.4f}', smoothed=f'{smoothed:.4f}')

            # ベスト更新
            if smoothed < best_loss:
                best_loss = smoothed
                best_lr = current_lr

            # 発散判定（急激に悪化したら打ち切り）
            if step > 10 and smoothed > args.divergence_factor * best_loss:
                print(f'Loss diverged (>{args.divergence_factor}× best). Early stop at step {step}, lr={current_lr:.3e}')
                break

            # 次ステップの LR を増加
            current_lr *= lr_mult
            for pg in optim.param_groups:
                pg['lr'] = current_lr

            step += 1

    except KeyboardInterrupt:
        print('Interrupted by user.')

    # 推奨LR（経験則いくつか表示）
    # 1) best_lr を 1/10 に (急勾配の少し手前を狙う)
    # 2) best_lr 自体
    rec1 = best_lr * 0.1
    rec2 = best_lr

    print('──────────────────────────────────────────')
    print(f'Best smoothed loss at LR ≈ {best_lr:.3e}')
    print(f'Recommendation (conservative):  {rec1:.3e}')
    print(f'Recommendation (aggressive):    {rec2:.3e}')
    print(f'CSV saved to: {csv_path}')
    print('Done.', datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9), 'JST')))
    print('──────────────────────────────────────────')

if __name__ == '__main__':
    main()

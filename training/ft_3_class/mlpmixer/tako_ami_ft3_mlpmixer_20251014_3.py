#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
tako_ami_ft3_mlpmixer_20251014_3.py
-------------------------------------------------
Fine-tuning script for **MLP-Mixer-B/16** (timm) on 3-class classification at 480×480.
+ LR Finder support via --find-lr

Usage (LR Finder only):
  python tako_ami_ft3_mlpmixer_20251014_4.py \
    --pretrained-mae /path/to/ckpt_0099.pth \
    --train-dir /path/to/train --val-dir /path/to/valid \
    --output /path/to/run \
    --batch-size 32 \
    --find-lr --find-lr-end 1e-2

Version: 1.0.0 (2025-10-14, JST) first version
Version: 1.1.0 (2025-10-14, JST) add LR Finder
"""

import argparse, os, math, random, csv, datetime, re
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.transforms import InterpolationMode, RandAugment, RandomErasing
from PIL import Image

import timm
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from sklearn.metrics import precision_recall_fscore_support

from tqdm.auto import tqdm


# ──────────────────────────────────────────
# Utils (compile-aware)
# ──────────────────────────────────────────
def get_base_model(m):
    return m._orig_mod if hasattr(m, "_orig_mod") else m

def normalize_state_dict_keys(state: dict) -> dict:
    """strip _orig_mod. / module. prefixes (compile/DDP safe)"""
    out = {}
    for k, v in state.items():
        nk = k
        if nk.startswith("_orig_mod."):
            nk = nk[len("_orig_mod."):]
        if nk.startswith("module."):
            nk = nk[len("module."):]
        out[nk] = v
    return out


# ──────────────────────────────────────────
# Dataset (患者階層・クラスマップはConvNeXt版に準拠)
# ──────────────────────────────────────────
class RandAugmentNoFlip(RandAugment):
    """timm風の RandAugment から Flip 系を除外"""
    def __init__(self, num_ops=2, magnitude=9, num_magnitude_bins=10, interpolation=None, fill=0):
        super().__init__(num_ops, magnitude, num_magnitude_bins, interpolation, fill)
        modules = list(self._modules.items())
        filtered = OrderedDict((n, m) for n, m in modules if "Flip" not in m.__class__.__name__)
        self._modules = filtered

class PatientDataset(Dataset):
    def __init__(self, root_dir, is_train=True, img_size=480):
        self.samples = []
        self.is_train = is_train
        self.img_size = img_size

        self.label_names = ["tako", "AMI", "non_tako_ami"]
        self.label2id = {n: i for i, n in enumerate(self.label_names)}
        self.id2label = {i: n for n, i in self.label2id.items()}
        non_tako_ami_dirs = ["nontako", "nonAMI", "normal"]

        for patient in os.listdir(root_dir):
            pdir = os.path.join(root_dir, patient)
            if not os.path.isdir(pdir):
                continue
            for lab in os.listdir(pdir):
                ldir = os.path.join(pdir, lab)
                if not os.path.isdir(ldir) or lab == "misc":
                    continue
                if lab == "tako":
                    y = self.label2id["tako"]
                elif lab == "AMI":
                    y = self.label2id["AMI"]
                elif lab in non_tako_ami_dirs:
                    y = self.label2id["non_tako_ami"]
                else:
                    continue
                for fn in os.listdir(ldir):
                    if fn.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                        self.samples.append((os.path.join(ldir, fn), y))

        mean = [0.485, 0.456, 0.406]
        std  = [0.229, 0.224, 0.225]

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
                RandAugmentNoFlip(num_ops=1, magnitude=7, interpolation=InterpolationMode.BILINEAR, fill=0),
                transforms.RandomRotation(degrees=2),
                transforms.ToTensor(),
                #RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value='random', inplace=False),
                transforms.Normalize(mean=mean, std=std),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size), interpolation=InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert('RGB')
        x = self.transform(img)
        return {"pixel_values": x, "labels": torch.tensor(y)}


# ──────────────────────────────────────────
# Model & MAE-ckpt loader (MixerMAE → timm MLP-Mixer)
# ──────────────────────────────────────────
def create_mixer(num_classes=3, img_size=480):
    # timm の mixer_b16_224 は img_size を任意に指定可能
    model = timm.create_model(
        'mixer_b16_224',
        pretrained=False,
        img_size=img_size,
        in_chans=3,
        num_classes=num_classes
    )
    return model

def _remap_key_for_mixer(k: str) -> str:
    """
    MixerMAE 事前学習の encoder.* を timm Mixer に合わせる:
      - drop prefixes: _orig_mod., module., encoder.
      - "name_3" → "name.3" のような index のアンダースコア→ドット変換
    """
    k = k.replace("_orig_mod.", "")
    k = k.replace("module.", "")
    k = k.replace("encoder.", "")
    toks = k.split(".")
    out = []
    for t in toks:
        m = re.match(r"^([A-Za-z0-9_]+)_(\d+)$", t)
        out.append(f"{m.group(1)}.{m.group(2)}" if m else t)
    return ".".join(out)

def load_mae_encoder_to_mixer(model: nn.Module, ckpt_path: str):
    """
    あなたの MixerMAE の ckpt から timm Mixer へのロード。
    decoder/mask_token/pos_embed/cls_token/head は破棄。
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    raw = ckpt.get("model", ckpt.get("state_dict", ckpt))
    drop = ("decoder", "mask_token", "pos_embed", "cls_token", "head")

    mapped = {}
    for k, v in raw.items():
        if any(d in k for d in drop):
            continue
        nk = _remap_key_for_mixer(k)
        mapped[nk] = v

    msd = model.state_dict()
    loadable = {k: v for k, v in mapped.items() if k in msd and msd[k].shape == v.shape}

    miss = [k for k in msd.keys() if k not in loadable]
    unex = [k for k in mapped.keys() if k not in msd]
    print(f"[MAE→Mixer] loadable={len(loadable)}  missing={len(miss)}  unexpected={len(unex)}")
    if miss:
        print("  missing(ex):", miss[:8])
    if unex:
        print("  unexpected(ex):", unex[:8])

    model.load_state_dict(loadable, strict=False)
    return model


# ──────────────────────────────────────────
# LR Finder
# ──────────────────────────────────────────
@torch.no_grad()
def _count_batches(loader):
    try:
        return len(loader)
    except:
        return 1000

def make_loader_kwargs(batch_size, num_workers, drop_last):
    kw = dict(batch_size=batch_size, pin_memory=True, drop_last=drop_last)
    if num_workers > 0:
        kw.update(num_workers=num_workers, persistent_workers=True, prefetch_factor=4)
    else:
        kw.update(num_workers=0)  # persistent_workers/prefetch_factor は付けない
    return kw

def build_plain_loader(train_ds, batch_size, num_workers, sampler_weights):
    """Mixup/CutMix, RandomErasingを無効化した素のDataLoaderを作る（LR Finder用）"""
    # sampler は学習時と同様の WeightedRandomSampler を使う
    epoch_samples = (len(train_ds) // batch_size) * batch_size if len(train_ds) >= batch_size else batch_size
    sampler = WeightedRandomSampler(sampler_weights, num_samples=epoch_samples, replacement=True)

    # transform から RandomErasing を取り除く（見つかった場合）
    if hasattr(train_ds, "transform") and isinstance(train_ds.transform, transforms.Compose):
        new_tfms = []
        for t in train_ds.transform.transforms:
            if isinstance(t, RandomErasing):
                continue
            new_tfms.append(t)
        train_ds_plain = PatientDataset.__new__(PatientDataset)  # シャローコピー的に最小限
        train_ds_plain.__dict__.update(train_ds.__dict__)
        train_ds_plain.transform = transforms.Compose(new_tfms)
    else:
        train_ds_plain = train_ds

    def _collate(batch):
        x = torch.stack([b["pixel_values"] for b in batch])
        y = torch.tensor([b["labels"] for b in batch])
        return {"pixel_values": x, "labels": y}

    # --- LR Finder 用 plain loader ---
    kw = make_loader_kwargs(batch_size, num_workers, drop_last=True)
    loader = DataLoader(
        train_ds_plain,
        sampler=sampler, shuffle=False, collate_fn=_collate,
        **kw
    )

    return loader

def lr_range_test(
    model: nn.Module,
    train_loader_plain: DataLoader,
    backbone_params,
    head_params,
    device,
    start_lr=1e-7,
    end_lr=1e-2,
    num_iter=1000,
    smooth_beta=0.98,
    max_increase=4.0,
):
    """Exponential LR finder (Smith 2017) with 2 param groups: backbone & head(x10)."""
    model.train()
    ce = nn.CrossEntropyLoss()

    # 一時オプティマイザ
    opt = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": start_lr, "weight_decay": 0.0},
            {"params": head_params,     "lr": start_lr * 10.0, "weight_decay": 0.0},
        ]
    )

    use_cuda = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32

    # 指数増加倍率
    total = max(2, num_iter)
    mult = (end_lr / start_lr) ** (1 / (total - 1))

    lrs, losses, smooth = [], [], []
    avg, best = 0.0, float("inf")
    it = 0

    loader_iter = iter(train_loader_plain)
    while it < num_iter:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader_plain)
            batch = next(loader_iter)

        it += 1
        x = batch["pixel_values"].to(device, non_blocking=True)
        y = batch["labels"].to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_cuda):
            logits = model(x)
            loss = ce(logits, y)
        loss.backward()
        clip_grad_norm_([p for p in model.parameters() if p.grad is not None], 1.0)
        opt.step()

        # ログ
        # backbone PG(0) のLRを代表値として記録
        curr_lr = opt.param_groups[0]["lr"]
        lrs.append(curr_lr)
        lv = float(loss.item())
        losses.append(lv)
        avg = smooth_beta * avg + (1 - smooth_beta) * lv
        sm = avg / (1 - smooth_beta ** it)
        smooth.append(sm)
        best = min(best, sm)

        # 発散検知
        if not math.isfinite(sm) or sm > best * max_increase:
            break

        # 次のLRに指数更新（両PGとも倍率維持）
        for pg in opt.param_groups:
            pg["lr"] *= mult

    # 推奨LR（保守的）：スムーズ損失最小点の2つ手前
    i_min = max(0, smooth.index(min(smooth)))
    pick = max(0, i_min - 2)
    suggest_backbone = lrs[pick]
    suggest_head     = suggest_backbone * 10.0

    return {
        "lrs": lrs,
        "loss": losses,
        "smooth": smooth,
        "suggest_backbone_lr": suggest_backbone,
        "suggest_head_lr": suggest_head,
    }


# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("MLP-Mixer (mixer_b16) 3-class fine-tune @480")
    p.add_argument('--train-dir', required=True)
    p.add_argument('--val-dir',   required=True)
    p.add_argument('--output',    required=True)
    p.add_argument('--pretrained-mae', default=None, help='MixerMAE ckpt path (from pretraining)')
    p.add_argument('--img-size', type=int, default=480)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--head-only-epochs', type=int, default=3)
    p.add_argument('--lr', type=float, default=1.01e-5)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--resume', type=str, default=None)

    # options
    p.add_argument('--use-swa', action='store_true')
    p.add_argument('--swa-start-epoch', type=int, default=12)
    p.add_argument('--swa-lr', type=float, default=5e-6)
    p.add_argument('--use-sam', action='store_true')
    p.add_argument('--sam-rho', type=float, default=0.05)
    p.add_argument('--sam-adaptive', action='store_true')
    p.add_argument('--mixup-prob', type=float, default=0.5)
    p.add_argument('--switch-prob', type=float, default=0.5)
    p.add_argument('--mixup-alpha', type=float, default=0.3)
    p.add_argument('--cutmix-alpha', type=float, default=0.3)
    p.add_argument('--mixup-label-smoothing', type=float, default=0.1)

    # LR Finder
    p.add_argument('--find-lr', action='store_true', help='Run LR range test and exit')
    p.add_argument('--find-lr-start', type=float, default=1e-7, help='LR finder: start LR (backbone)')
    p.add_argument('--find-lr-end', type=float, default=1e-2, help='LR finder: end LR (backbone)')
    p.add_argument('--find-lr-iters', type=int, default=800, help='LR finder: number of iterations')
    p.add_argument('--find-lr-smooth', type=float, default=0.98, help='LR finder: EMA smoothing beta')
    p.add_argument('--find-lr-diverge', type=float, default=4.0, help='Early stop when smoothed loss > best * this')
    return p.parse_args()

def set_seed(seed):
    if seed is None:
        return
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    args = parse_args()

    if args.find_lr:
        if not (args.find_lr_start > 0 and args.find_lr_end > 0 and args.find_lr_end > args.find_lr_start):
            raise ValueError("--find-lr-start/end が不正です（正値かつ end > start にしてください）")
        if args.find_lr_iters < 10:
            raise ValueError("--find-lr-iters は少なくとも 10 以上を推奨します")

    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Datasets / Loaders
    train_ds = PatientDataset(args.train_dir, is_train=True,  img_size=args.img_size)
    val_ds   = PatientDataset(args.val_dir,   is_train=False, img_size=args.img_size)
    num_classes = len(train_ds.label_names)

    # Sampler weights for class balance
    labels = torch.tensor([y for _, y in train_ds.samples])
    cls_counts = torch.bincount(labels, minlength=num_classes).float().clamp_min(1.0)
    cls_w = cls_counts.sum() / cls_counts
    sample_w = cls_w[labels].double()

    # Standard train/val loaders（本学習用）
    def make_mixup():
        return Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=args.mixup_prob,
            switch_prob=args.switch_prob,
            mode='batch',
            label_smoothing=args.mixup_label_smoothing,
            num_classes=num_classes
        )

    mixup_fn = make_mixup()

    def collate_train(batch):
        x = torch.stack([b["pixel_values"] for b in batch])
        y = torch.tensor([b["labels"] for b in batch])
        if mixup_fn.prob > 0.0:
            x, y_soft = mixup_fn(x, y)
            return {"pixel_values": x, "labels": y_soft, "hard_labels": y}
        else:
            return {"pixel_values": x, "labels": y, "hard_labels": y}

    def collate_eval(batch):
        x = torch.stack([b["pixel_values"] for b in batch])
        y = torch.tensor([b["labels"] for b in batch])
        return {"pixel_values": x, "labels": y}

    epoch_samples = (len(train_ds) // args.batch_size) * args.batch_size if len(train_ds) >= args.batch_size else args.batch_size
    train_sampler = WeightedRandomSampler(sample_w, num_samples=epoch_samples, replacement=True)
    # --- 本学習 ---
    kw_train = make_loader_kwargs(args.batch_size, args.num_workers, drop_last=True)
    #kw_train_shuf = {**kw_train, "shuffle": True}  # 重複引数は足さない

    train_loader = DataLoader(
        train_ds, sampler=train_sampler, shuffle=False, collate_fn=collate_train, **kw_train
    )
    '''
    # ★ tiny 過学習テスト用：サンプラを外して shuffle=True
    train_loader = DataLoader(
        train_ds, collate_fn=collate_train, **kw_train_shuf
    )
    '''
    
    kw_val = make_loader_kwargs(args.batch_size, args.num_workers, drop_last=False)
    val_loader = DataLoader(
        val_ds, shuffle=False, collate_fn=collate_eval, **kw_val
    )


    # Model
    model = create_mixer(num_classes=num_classes, img_size=args.img_size)
    if args.pretrained_mae:
        model = load_mae_encoder_to_mixer(model, args.pretrained_mae)
    model.to(device)

    # Compile (optional)
    try:
        model = torch.compile(model, mode='default')
        print("Model compiled.")
    except Exception as e:
        print(f"Compile skipped: {e}")
    
    base = get_base_model(model)

    # ===== LR Finder モード =====
    if args.find_lr:
        # backbone / head のPGを作成（compile対応で base から）
        if hasattr(base, 'head'):
            head_params = list(base.head.parameters())
            backbone_params = [p for n, p in base.named_parameters() if not n.startswith('head.')]
        elif hasattr(base, 'classifier'):
            head_params = list(base.classifier.parameters())
            backbone_params = [p for n, p in base.named_parameters() if not n.startswith('classifier.')]
        else:
            raise RuntimeError("Neither 'head' nor 'classifier' found in model.")

        # Mixup/Erasing無効のプレーンLoader
        plain_loader = build_plain_loader(
            train_ds=train_ds,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            sampler_weights=sample_w
        )

        out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
        print("[LR Finder] start...")
        res = lr_range_test(
            model=base,
            train_loader_plain=plain_loader,
            backbone_params=backbone_params,
            head_params=head_params,
            device=device,
            start_lr=args.find_lr_start,
            end_lr=args.find_lr_end,
            num_iter=min(args.find_lr_iters, _count_batches(plain_loader)),
            smooth_beta=args.find_lr_smooth,
            max_increase=args.find_lr_diverge,
        )
        # CSV 保存
        csv_path = out_dir / "lr_finder.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["iter", "backbone_lr", "loss", "smoothed_loss"])
            for i, (lr, lo, sm) in enumerate(zip(res["lrs"], res["loss"], res["smooth"])):
                w.writerow([i+1, f"{lr:.8e}", f"{lo:.6f}", f"{sm:.6f}"])

        print(f"[LR Finder] suggested backbone lr: {res['suggest_backbone_lr']:.3e}")
        print(f"[LR Finder] suggested head     lr: {res['suggest_head_lr']:.3e}")
        print(f"[LR Finder] curve saved to: {str(csv_path)}")
        return  # ここで終了（学習は行わない）

    # ===== ここから通常学習 =====
    train_crit = SoftTargetCrossEntropy()
    val_crit   = nn.CrossEntropyLoss()

    # Params split
    if hasattr(base, 'head'):
        head_params = list(base.head.parameters())
        backbone_params = [p for n, p in base.named_parameters() if not n.startswith('head.')]
    elif hasattr(base, 'classifier'):
        head_params = list(base.classifier.parameters())
        backbone_params = [p for n, p in base.named_parameters() if not n.startswith('classifier.')]
    else:
        raise RuntimeError("Neither 'head' nor 'classifier' found in model.")

    # Optimizers
    head_opt = torch.optim.AdamW(head_params, lr=args.lr * 10, weight_decay=0.01)
    full_opt = torch.optim.AdamW(
        [{"params": backbone_params, "lr": args.lr},
         {"params": head_params,     "lr": args.lr * 10}],
        weight_decay=0.01
    )

    # Schedulers
    head_sch = CosineAnnealingLR(head_opt, T_max=args.head_only_epochs, eta_min=args.lr*5)
    warmup_full = 3
    full_sch = None

    # SWA
    swa_model, swa_sch = None, None
    if args.use_swa:
        swa_model = AveragedModel(base)
        print(f"[SWA] start @ epoch >= {args.swa_start_epoch}")

    # AMP (bfloat16推奨)
    use_cuda = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if use_cuda else torch.float32
    scaler = torch.amp.GradScaler(enabled=False)  # bf16なら不要

    # Resume
    start_epoch, best_val_acc = 1, 0.0
    if args.resume and os.path.exists(args.resume):
        print(f"Resume from: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        # model
        state = ckpt.get('model', ckpt)
        state = normalize_state_dict_keys(state)
        inc = base.load_state_dict(state, strict=False)
        if getattr(inc, "missing_keys", None):
            print("  missing:", list(inc.missing_keys)[:8], "...")
        if getattr(inc, "unexpected_keys", None):
            print("  unexpected:", list(inc.unexpected_keys)[:8], "...")
        # opts/schs
        if 'head_opt' in ckpt: head_opt.load_state_dict(ckpt['head_opt'])
        if 'full_opt' in ckpt: full_opt.load_state_dict(ckpt['full_opt'])
        if 'head_sch' in ckpt: head_sch.load_state_dict(ckpt['head_sch'])
        if args.use_swa and ('swa_model' in ckpt) and swa_model is not None:
            swa_state = normalize_state_dict_keys(ckpt['swa_model'])
            inc = swa_model.load_state_dict(swa_state, strict=False)
            if getattr(inc, "missing_keys", None) or getattr(inc, "unexpected_keys", None):
                print("[resume][SWA] missing:", getattr(inc, "missing_keys", [])[:6],
                      "| unexpected:", getattr(inc, "unexpected_keys", [])[:6])
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)

    bb_params = sum(p.numel() for n, p in get_base_model(model).named_parameters() if 'head' not in n and p.requires_grad)
    print(f"Backbone trainable params: {bb_params:,}")

    # Prepare full scheduler if entering full phase on resume
    if start_epoch > args.head_only_epochs + 1 and full_sch is None:
        full_sch = SequentialLR(
            full_opt,
            schedulers=[
                LinearLR(full_opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_full),
                CosineAnnealingLR(full_opt, T_max=args.epochs - args.head_only_epochs - warmup_full),
            ],
            milestones=[warmup_full],
        )
        if args.use_swa:
            swa_sch = SWALR(full_opt, swa_lr=args.swa_lr)

    # Output dir & history
    out_dir = Path(args.output); out_dir.mkdir(parents=True, exist_ok=True)
    hist = out_dir / 'training_history.csv'
    if not hist.exists():
        with open(hist, 'w', newline='') as f:
            csv.writer(f).writerow(['epoch','train_loss','train_acc','val_loss','val_acc','precision_macro','recall_macro','f1_macro','lr'])

    # Train
    for epoch in range(start_epoch, args.epochs + 1):
        base = get_base_model(model)

        is_head_only = (epoch <= args.head_only_epochs)
        # mixup on/off
        mixup_fn.prob = 0.0 if is_head_only else args.mixup_prob
        # NEW: head-only 期は smoothing も 0 に、以降は指定値に戻す
        mixup_fn.label_smoothing = 0.0 if is_head_only else args.mixup_label_smoothing
        
        # switch from head-only to full at the start of the first full epoch
        if epoch == args.head_only_epochs + 1:
            full_sch = SequentialLR(
                full_opt,
                schedulers=[
                    LinearLR(full_opt, start_factor=0.1, end_factor=1.0, total_iters=warmup_full),
                    CosineAnnealingLR(full_opt, T_max=args.epochs - args.head_only_epochs - warmup_full),
                ],
                milestones=[warmup_full],
            )
            if args.use_swa:
                swa_sch = SWALR(full_opt, swa_lr=args.swa_lr)

        # freeze backbone during head-only
        for p in backbone_params:
            p.requires_grad = not is_head_only

        # pick optimizer/scheduler
        opt = head_opt if is_head_only else full_opt
        sch = head_sch if is_head_only else (swa_sch if (args.use_swa and epoch >= args.swa_start_epoch) else full_sch)

        # Train epoch
        model.train()
        tr_loss = tr_acc = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        for batch in pbar:
            x = batch['pixel_values'].to(device, non_blocking=True)
            y_soft = batch['labels'].to(device, non_blocking=True)
            y_hard = batch['hard_labels'].to(device, non_blocking=True)

            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_cuda):
                logits = model(x)
                if torch.is_floating_point(y_soft) and y_soft.ndim == 2:
                    # mixup/cutmix（ソフトラベル）のとき
                    loss = train_crit(logits, y_soft)
                else:
                    # ソフトラベルでない（整数ラベル）のとき
                    loss = nn.functional.cross_entropy(logits, y_hard)

            opt.zero_grad(set_to_none=True)

            # SAM?
            use_sam_now = (args.use_sam and not is_head_only)
            if use_sam_now:
                # 1st
                loss.backward()
                # eps
                params = [p for p in model.parameters() if p.requires_grad and p.grad is not None]
                gn = torch.norm(torch.stack([(p.detach().abs() * p.grad).norm() if args.sam_adaptive else p.grad.norm()
                                             for p in params]), 2).clamp_min(1e-12)
                eps_buf = []
                with torch.no_grad():
                    for p in params:
                        scale = p.detach().abs() if args.sam_adaptive else 1.0
                        eps = (args.sam_rho / gn) * scale * p.grad
                        p.add_(eps); eps_buf.append(eps)
                opt.zero_grad(set_to_none=True)
                # 2nd
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_cuda):
                    logits2 = model(x)
                    if torch.is_floating_point(y_soft) and y_soft.ndim == 2:
                        loss2 = train_crit(logits2, y_soft)
                    else:
                        loss2 = nn.functional.cross_entropy(logits2, y_hard)
                loss2.backward()
                # restore
                with torch.no_grad():
                    for p, eps in zip(params, eps_buf):
                        p.sub_(eps)
                clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                logits_final = logits2
                current_loss = loss2
            else:
                loss.backward()
                clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                logits_final = logits
                current_loss = loss

            tr_loss += current_loss.item()
            tr_acc  += (logits_final.argmax(1) == y_hard).float().mean().item()

        tr_loss /= max(1, len(train_loader))
        tr_acc  /= max(1, len(train_loader))

        # SWA param averaging
        if args.use_swa and epoch >= args.swa_start_epoch and 'swa_model' in locals() and (swa_model is not None):
            swa_model.update_parameters(base)

        # Val
        model.eval()
        va_loss = va_acc = 0.0
        all_pred, all_true = [], []
        with torch.no_grad():
            pbar = tqdm(val_loader, desc=f"Epoch {epoch}/{args.epochs} [Val]", leave=False)
            for batch in pbar:
                x = batch['pixel_values'].to(device, non_blocking=True)
                y = batch['labels'].to(device, non_blocking=True)
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype, enabled=use_cuda):
                    logits = model(x)
                    loss = val_crit(logits, y)
                va_loss += loss.item()
                preds = logits.argmax(dim=1)
                va_acc  += (preds == y).float().mean().item()
                all_pred.extend(preds.cpu().tolist())
                all_true.extend(y.cpu().tolist())
        va_loss /= max(1, len(val_loader))
        va_acc  /= max(1, len(val_loader))

        # Metrics
        prec, rec, f1, _ = precision_recall_fscore_support(all_true, all_pred, average="macro", zero_division=0)

        # Step scheduler
        if sch is not None:
            sch.step()

        # Log
        cur_lrs = [pg['lr'] for pg in opt.param_groups]
        print(f"Epoch {epoch}/{args.epochs} | Train: loss={tr_loss:.4f} acc={tr_acc:.4f} | Val: loss={va_loss:.4f} acc={va_acc:.4f} | LR={cur_lrs}")

        with open(hist, 'a', newline='') as f:
            csv.writer(f).writerow([epoch, f"{tr_loss:.4f}", f"{tr_acc:.4f}", f"{va_loss:.4f}", f"{va_acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}", f"{cur_lrs}"])

        # Save ckpt (always save with base model weights)
        base = get_base_model(model)
        ckpt = {
            "model": base.state_dict(),
            "head_opt": head_opt.state_dict(),
            "full_opt": full_opt.state_dict(),
            "head_sch": head_sch.state_dict(),
            "full_sch": full_sch.state_dict() if full_sch is not None else None,
            "epoch": epoch,
            "best_val_acc": va_acc,
            "args": vars(args),
        }
        if 'swa_model' in locals() and (swa_model is not None):
            ckpt["swa_model"] = swa_model.state_dict()
        torch.save(ckpt, out_dir / f"ckpt_{epoch:04d}.pth")
        torch.save(ckpt, out_dir / "latest.pth")

        # Best
        if va_acc >= best_val_acc:
            best_val_acc = va_acc
            torch.save(ckpt, out_dir / "best.pth")
            print(f"  * New best acc: {best_val_acc:.4f}")

    # Final SWA BN update & save (optional)
    if 'swa_model' in locals() and (swa_model is not None):
        print("[SWA] final BN update...")
        swa_model.to(device)
        swa_model.train()
        update_bn(train_loader, swa_model, device=device)
        swa_model.eval()
        torch.save({"model": swa_model.state_dict(), "args": vars(args)}, out_dir / "swa_best.pth")

    print("Done.", datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9), 'JST')))


if __name__ == "__main__":
    main()

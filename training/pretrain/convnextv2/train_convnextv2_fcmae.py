'''train_convnextv2_fcmae.py
-------------------------------------------------
Example script that leverages *mae_wrapper.py* to pre-train a ConvNeXt-V2-Base
backbone with FCMAE on 384×384 or 480×480 images.

ConvNeXt V2 includes GRN (Global Response Normalization) which improves
performance compared to ConvNeXt V1.

The data-loading logic (PretrainDataset, WeightedRandomSampler, etc.) follows
exactly the style in the user-supplied notebook snippet, but distilled to the
minimum necessary so that you can just **edit the DIR constants** and run:

    python train_convnextv2_fcmae.py \
        --train-dirs /path/to/train1 /path/to/train2 \
        --img-size 480 \
        --output     runs/fcmae_convnextv2_base_480

GPU ≥24 GB is recommended for the default batch-size 64.
Scheduled run options:
    --target-hour 23        Target hour (0-23)
    --target-minute 0       Target minute (0-59)

version 1.0.2 2025-08-11 sleep関連の引数と処理を削除。
'''

import argparse, os, math, random, datetime
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from torch.amp import GradScaler, autocast
from PIL import Image
from timm.data import create_transform
from mae_fcmae_wrapper import FCMAE   # ← canvas file from previous turn
from tqdm.auto import tqdm
from ek_schedule_exe import wait_until_target_time
from collections import OrderedDict
import sys

IS_TTY = sys.stdout.isatty()

torch.set_float32_matmul_precision('high')  # TF32 を積極活用

# ──────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────
class PretrainDataset(Dataset):
    """ImageFolder-like loader w/ optional augmentation & simple caching"""
    IMG_EXT = ('.png', '.jpg', '.jpeg')
    def __init__(self, root, transform=None, cache_size=2000):
        if not os.path.exists(root):
            raise FileNotFoundError(root)
        self.files = self._scan(root)
        self.transform = transform
        self.cache_size = cache_size
        self._img_cache = OrderedDict()  # idx -> PIL.Image

    @staticmethod
    def _scan(root):
        paths = []
        for r, _, fns in os.walk(root):
            if 'misc' in r:  # skip misc
                continue
            for fn in fns:
                if fn.lower().endswith(PretrainDataset.IMG_EXT):
                    paths.append(os.path.join(r, fn))
        if not paths:
            raise RuntimeError(f'No images found under {root}')
        return paths

    def __len__(self):
        return len(self.files)

    def _get_decoded(self, idx):
        # LRU: ヒットなら末尾へ、ミスならロード
        if idx in self._img_cache:
            img = self._img_cache.pop(idx)
            self._img_cache[idx] = img
            return img
        img = Image.open(self.files[idx]).convert('RGB')
        # 追加：Pillow-SIMD を入れておくとデコード自体も速い
        self._img_cache[idx] = img
        if len(self._img_cache) > self.cache_size:
            self._img_cache.popitem(last=False)  # 最古を削除
        return img

    def __getitem__(self, idx):
        img = self._get_decoded(idx)          # ここでデコードキャッシュ
        if self.transform:
            out = self.transform(img)         # オーグは毎回実行（多様性維持）
        else:
            out = transforms.ToTensor()(img)
        return {"pixel_values": out}
    

# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser("ConvNeXt-V2-FCMAE pretrainer")
    p.add_argument('--train-dirs', nargs='+', required=True, help='one or more training image roots')
    p.add_argument('--output', required=True)
    p.add_argument('--img-size', type=int, default=480, choices=[384, 480], help='Image size for training (default: 480)')
    p.add_argument('--total-epochs', type=int, default=100, help='Total epochs for training (default: 100)')
    p.add_argument('--run-epochs', type=int, help='Number of epochs to run in this session (default: total-epochs)')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--num-workers', type=int, default=16)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--resume', type=str, help='Resume from checkpoint')
    p.add_argument('--target-hour', type=int, help='Target hour for scheduled run (0-23)')
    p.add_argument('--target-minute', type=int, help='Target minute for scheduled run (0-59)')
    return p.parse_args()

# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────

def set_seed(seed):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    IMG_SIZE = args.img_size
    PATCH_SIZE = 4  # ConvNeXt stem stride

    # augment (random crop 0.8-1.0) + timm mean/std norm
    aug_tfms = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    
    # build datasets in parallel
    def build(ds_paths, tfm):
        with ThreadPoolExecutor() as exe:
            return ConcatDataset([exe.submit(PretrainDataset,p,tfm).result() for p in ds_paths])
    train_ds = build(args.train_dirs, aug_tfms)

    # train_ds は ConcatDataset
    lengths = [len(d) for d in train_ds.datasets]  # 各サブDSのサイズ
    cum = [0]
    for L in lengths:
        cum.append(cum[-1] + L)

    w = torch.empty(len(train_ds), dtype=torch.float)
    for i, L in enumerate(lengths):
        start, end = cum[i], cum[i+1]
        w[start:end] = 1.0 / L  # 所属DSの逆数を割当

    sampler = WeightedRandomSampler(w, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        pin_memory_device='cuda',   
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=8 #4
    )

    # model (using mae_wrapper with ConvNeXt V2)
    model = FCMAE('convnextv2_base', img_size=IMG_SIZE, patch_size=PATCH_SIZE, mask_ratio=0.75)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # ★ 追加: channels_last に変換
    if torch.cuda.is_available():
        model.to(memory_format=torch.channels_last)

    print(f"Using ConvNeXt V2 with GRN (Global Response Normalization)")
    print(f"Model: {model}")
    
    # Compile model for faster training (PyTorch 2.0+)
    #共有メモリを超過して、結局cuBLASに落ち着いているようで、のmode='default'に戻した。
    '''
    try:
        model = torch.compile(model, mode="max-autotune", fullgraph=True, dynamic=False)
        print("Model compiled (max-autotune, fullgraph)")
    except Exception as e:
        print(f"Model compilation failed: {e}")

    '''
    try:
        model = torch.compile(model, mode="default")
        print("Model compiled successfully for faster training")
    except Exception as e:
        print(f"Model compilation failed: {e}")
        print("Continuing without compilation")
    
    # optimizer & sched

    # optimizer
    # 変更: fused AdamW を試す（未対応環境では自動で通常版にフォールバック）
    try:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05, fused=True)
        print("Using fused AdamW")
    except TypeError:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        print("Using standard AdamW")

    # Use BF16 for RTX 5090 (better than FP16)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    use_bf_16 = (dtype == torch.bfloat16)
    scaler = GradScaler('cuda') if not use_bf_16 else None
    
    # Memory optimizations for ConvNeXt V2
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False
        print("CUDA optimizations enabled for ConvNeXt V2 (bf16 compatible)")
    
    # Gradient accumulation for larger effective batch size
    accumulation_steps = 1  # Can be increased if needed

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Initialize training history CSV
    history_file = out_dir / 'training_history.csv'
    if not history_file.exists():
        with open(history_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'train_loss'])

    # Set run epochs (default to total epochs if not specified)
    if args.run_epochs is None:
        args.run_epochs = args.total_epochs
    
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.total_epochs
    
    # Calculate current step for learning rate scheduling
    current_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        current_step = ckpt.get('step', 0)  # Load saved step
    
    def lr_func(step):
        pct = step / total_steps
        return 0.5 * (1 + math.cos(math.pi * pct))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    # Scheduled start (wait until target time if specified)
    if args.target_hour is not None or args.target_minute is not None:
        hour = args.target_hour if args.target_hour is not None else 0
        minute = args.target_minute if args.target_minute is not None else 0
        wait_until_target_time(hour, minute)

    # training loop
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        print(f"Loading checkpoint from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optim.load_state_dict(ckpt['optimizer'])
        sched.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Resuming from epoch {start_epoch}")

    try:
        end_epoch = min(start_epoch + args.run_epochs - 1, args.total_epochs)
        for epoch in range(start_epoch, end_epoch + 1):
            model.train()
            running = 0.0
            prog = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False, disable=not IS_TTY)
            
            for step, batch in enumerate(prog):
                imgs = batch['pixel_values'].to(device, non_blocking=True)
                
                if imgs.is_contiguous(memory_format=torch.contiguous_format):
                    imgs = imgs.to(memory_format=torch.channels_last)
                
                with autocast('cuda', dtype=dtype):
                    loss = model(imgs)

                optim.zero_grad(set_to_none=True)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optim); scaler.update()
                else:
                    loss.backward()
                    optim.step()
                
                sched.step()

                running += loss.item()

                if (step & 31) == 0:
                    if IS_TTY:
                        prog.set_postfix(loss=loss.item())
                    else:
                        print(f"[{epoch}:{step}/{len(train_loader)}] loss={loss.item():.4f}", flush=True)

            avg = running / len(train_loader)
            print(f'Epoch {epoch}/{args.total_epochs}  train_loss: {avg:.4f} (ConvNeXt V2 with GRN)')
            
            # Save to training history CSV
            with open(history_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, f'{avg:.4f}'])
            
            # Save checkpoint
            ckpt = {
                'model': model.state_dict(),
                'optimizer': optim.state_dict(),
                'scheduler': sched.state_dict(),
                'epoch': epoch,
                'train_loss': avg,
                'args': vars(args),
                'step': current_step + epoch * steps_per_epoch
            }
            torch.save(ckpt, out_dir / f'ckpt_{epoch:04d}.pth')
            torch.save(ckpt, out_dir / 'latest.pth')

            if epoch >= args.total_epochs:
                print(f"Training completed! Reached {args.total_epochs} epochs with ConvNeXt V2.")
                break

    except KeyboardInterrupt:
        # ★ 割り込み時の緊急保存（進捗を latest に反映）
        print('Interrupted. Saving emergency checkpoint...')
        emergency = {
            'model': model.state_dict(),
            'optimizer': optim.state_dict(),
            'scheduler': sched.state_dict(),
            'epoch': max(1, locals().get('epoch', start_epoch) - 1),  # 直前完了epochを推定
            'args': vars(args)
        }
        torch.save(emergency, out_dir / 'latest.pth')
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        torch.save(emergency, out_dir / f'ckpt_interrupt_{ts}.pth')
        raise  # systemd に“正常な割り込み終了”として返す（SuccessExitStatusでOK）        
        

if __name__ == '__main__':
    main()
    print('Done. ', datetime.datetime.now(datetime.timezone(datetime.timedelta(hours = 9), 'JST')))

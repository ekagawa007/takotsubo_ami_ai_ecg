"""
train_mlpmixer_mae.py
-------------------------------------------------
MAE-style pretraining script for **MLP-Mixer-B/16** at **480×480, 3-channel** using PyTorch + timm.

Design notes specific to MLP-Mixer:
- MLP-Mixer's token-mixing MLP depends on the **fixed number of patches S**. Unlike ViT,
  we **cannot** pass only visible tokens to the encoder without changing S. Therefore,
  this implementation keeps the full S tokens in the encoder and **replaces masked patches
  with a learned mask token** (or zeroed pixels prior to patch embed). The loss is computed
  **only on masked patches** (MAE objective).
- This keeps the code simple and faithful to MAE's reconstruction target, while remaining
  compatible with Mixer.

Resume / checkpoint I/O follows the style of your *convnextv2-fcmae* reference:
- --resume <ckpt.pth> to continue training (model/optimizer/scheduler/epoch/step loaded)
- Epoch-wise checkpoints saved to OUT/ckpt_XXXX.pth and OUT/latest.pth

Usage example
-------------
python train_mlpmixer_mae.py \
    --train-dirs /path/to/img_root1 /path/to/img_root2 \
    --img-size 480 \
    --output runs/mae_mixer_b16_480 \
    --total-epochs 200 --batch-size 128
    --compile-mode default

GPU ≥24 GB recommended for the default settings. RTX 5090 (32 GB) is ideal.

Version: 1.0.0 (2025-10-02, JST)
Version: 1.0.1 (2025-10-03, JST) --compile-mode 引数を追加
Version: 1.0.2 (2025-10-04, JST) 重みの名前のをcompile前のものに修正
Version: 1.0.3 (2025-10-07, JST) 画像のaugmentationを変更
Version: 1.0.4 (2025-10-25, JST) チェックポイントの保存を原子的に修正
"""

import argparse, os, math, random, csv, datetime, sys, tempfile, time
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from torch.amp import GradScaler, autocast
from PIL import Image
import timm
import signal, functools

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(x, **k):
        return x

# import ek_schedule_exe  ← 削除
try:
    import ek_schedule_exe
except Exception:
    ek_schedule_exe = None

IS_TTY = sys.stdout.isatty() or os.environ.get('FORCE_PROGRESS') == '1'
USE_TQDM = os.environ.get("USE_TQDM", "0") == '1'

torch.set_float32_matmul_precision('high')  # enable TF32 on supported GPUs

# ──────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────
class PretrainDataset(Dataset):
    """ImageFolder-like loader with optional caching. Returns dict w/ 'pixel_values'."""
    IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
    def __init__(self, root, transform=None, cache_size=500):
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
            if 'misc' in r.lower():
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
        if idx in self._img_cache:  # LRU update
            img = self._img_cache.pop(idx)
            self._img_cache[idx] = img
            return img
        img = Image.open(self.files[idx]).convert('RGB')
        self._img_cache[idx] = img
        if len(self._img_cache) > self.cache_size:
            self._img_cache.popitem(last=False)
        return img

    def __getitem__(self, idx):
        img = self._get_decoded(idx)
        if self.transform:
            out = self.transform(img)
        else:
            out = transforms.ToTensor()(img)
        return {"pixel_values": out}

# ──────────────────────────────────────────
# MAE for MLP-Mixer
# ──────────────────────────────────────────
class MixerMAE(nn.Module):
    """MAE-style wrapper for timm MLP-Mixer (B/16), img_size multiple of patch_size.

    - Keeps full sequence length S in the encoder; masked patches get a learned token.
    - Reconstruction is predicted for **all** patches; loss is computed on masked ones only.
    """
    def __init__(
        self,
        mixer_variant: str = 'mixer_b16_224',
        img_size: int = 480,
        in_chans: int = 3,
        patch_size: int = 16,
        embed_dim: int = 768,  # mixer-b/16 hidden dim
        decoder_dim: int = 256,
        decoder_depth: int = 3,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        assert img_size % patch_size == 0, 'img_size must be divisible by patch_size'
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        
        # 1) Backbone (timm MLP-Mixer) as encoder core
        #    We create without classifier head and at target img size.
        self.encoder = timm.create_model(
            mixer_variant,
            pretrained=False,
            img_size=img_size,
            in_chans=in_chans,
            num_classes=0,  # remove classification head
        )
        # The encoder should have .stem (PatchEmbed), .blocks (nn.Sequential or list), .norm
        assert hasattr(self.encoder, 'stem'), 'Expected MlpMixer to expose .stem'
        assert hasattr(self.encoder, 'blocks'), 'Expected MlpMixer to expose .blocks'
        assert hasattr(self.encoder, 'norm'), 'Expected MlpMixer to expose .norm'

        # latent dim from encoder
        # timm mixer_b16 defaults to embed_dim=768; infer from first block if available
        if hasattr(self.encoder, 'channels'):
            enc_dim = self.encoder.channels
        else:
            # safer: infer by running a dummy forward through stem
            with torch.no_grad():
                dummy = torch.zeros(1, in_chans, img_size, img_size)
                x = self.encoder.stem(dummy)  # [1, S, C]
                enc_dim = x.shape[-1]
        self.enc_dim = enc_dim

        # 2) Learned mask token in encoder space
        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.enc_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        # 3) Lightweight decoder (per-token MLP → pixels of a patch)
        #    Predicts patch_size*patch_size*in_chans values for each token.
        dec_layers = []
        dim_in = self.enc_dim
        for _ in range(decoder_depth - 1):
            dec_layers += [nn.Linear(dim_in, decoder_dim), nn.GELU()]
            dim_in = decoder_dim
        dec_layers += [nn.Linear(dim_in, patch_size * patch_size * in_chans)]
        self.decoder = nn.Sequential(*dec_layers)

    # ---- helpers ----
    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        """imgs: [B,3,H,W] → patches [B, S, P] where P = p*p*3"""
        p = self.patch_size
        B, C, H, W = imgs.shape
        h = H // p
        w = W // p
        x = imgs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 3, 5, 1).reshape(B, h*w, p*p*C)
        return x

    def unpatchify(self, patches: torch.Tensor) -> torch.Tensor:
        """patches [B,S,P] → imgs [B,3,H,W] (mainly for visualization/debug)"""
        p = self.patch_size
        B, S, P = patches.shape
        C = 3
        h = w = int(math.sqrt(S))
        assert h * w == S
        x = patches.reshape(B, h, w, p, p, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, h*p, w*p)
        return x

    @staticmethod
    def _random_mask(B: int, S: int, mask_ratio: float, device: torch.device):
        """Generate random boolean mask: True for positions to be masked."""
        num_mask = int(S * mask_ratio)
        noise = torch.rand(B, S, device=device)
        # argsort noise to pick the smallest num_mask positions as masked
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(B, S, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :num_mask], True)
        return mask

    def forward(self, imgs: torch.Tensor):
        """Compute MAE loss on masked patches.
        imgs: [B,3,H,W] in [0,1] range typically after ToTensor(); assumed normalized later.
        """
        B, _, H, W = imgs.shape
        S = (H // self.patch_size) * (W // self.patch_size)
        mask = self._random_mask(B, S, self.mask_ratio, imgs.device)

        # encoder tokens
        x = self.encoder.stem(imgs)   # [B,S,C]
        # replace masked tokens with mask_token (broadcast then masked_scatter)
        mask_tok = self.mask_token.expand(B, S, -1)
        x = torch.where(mask.unsqueeze(-1), mask_tok, x)

        # mixer blocks + norm
        for blk in (self.encoder.blocks if isinstance(self.encoder.blocks, (nn.Sequential, nn.ModuleList)) else [self.encoder.blocks]):
            x = blk(x)
        if getattr(self.encoder, 'norm', None) is not None:
            x = self.encoder.norm(x)

        # decode per token → per-patch pixels
        pred = self.decoder(x)  # [B,S,P]

        # target: (optionally) normalized pixels per patch
        with torch.no_grad():
            target = self.patchify(imgs)

        # loss on masked patches only
        mask_f = mask.float().unsqueeze(-1)  # [B,S,1]
        loss = ((pred - target) ** 2) * mask_f
        P = pred.shape[2]
        loss = loss.sum() / (mask_f.sum() * P + 1e-8)
        return loss

def _sd_to_save(model: nn.Module):
    """torch.compile されていても '生モデル' の state_dict を返す"""
    base = model._orig_mod if hasattr(model, "_orig_mod") else model
    return base.state_dict()

def _sanitize_state_dict_for_load(sd: dict) -> dict:
    """_orig_mod. プレフィックスが付いている state_dict を生モデル用に剥がす"""
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k[len("_orig_mod."):]: v for k, v in sd.items()}
    return sd

#原子的save, いったんtmpファイルに保存してから、pathに置き換える。
def atomic_torch_save(obj, path):
    d = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(dir=d, delete=False) as f:
        tmp = f.name
    try:
        torch.save(obj, tmp)
        os.replace(tmp, path)  # atomic
    finally:
        try: os.remove(tmp)
        except FileNotFoundError: pass

def append_history_atomic(path, row):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', dir=d, delete=False, newline='') as f:
        tmp = f.name
        if os.path.exists(path):
            with open(path, 'r', newline='') as rf:
                f.write(rf.read())
        csv.writer(f).writerow(row)
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

#torch.load(args.resume, ...) が失敗したら、同ディレクトリの ckpt_*.pth を新しい順に試し、最初に読めた正常 ckpt に自動で差し戻します
def _safe_load_resume(resume_path: str | Path, device):
    """latest.pth が壊れていても、最後に正常な ckpt_*.pth に自動でフォールバック"""
    import glob
    resume_path = Path(resume_path)
    out_dir = resume_path.parent

    # まずは素直に latest.pth を試す
    try:
        return torch.load(resume_path, map_location=device), str(resume_path)
    except Exception as e:
        print(f"[RESUME] '{resume_path.name}' の読み込みに失敗: {e}")
        print("[RESUME] ckpt_*.pth からの復旧を試みます…")

    # 新しい順に ckpt_*.pth を試す
    cands = sorted(glob.glob(str(out_dir / "ckpt_*.pth")), key=os.path.getmtime, reverse=True)
    for p in cands:
        try:
            ckpt = torch.load(p, map_location=device)
            print(f"[RESUME] フォールバック成功: {Path(p).name}")
            # ついでに latest.pth を原子的に修復
            try:
                atomic_torch_save(ckpt, out_dir / "latest.pth")
                print("[RESUME] latest.pth を正常 ckpt に置き換えました")
            except Exception as e:
                print(f"[RESUME] latest.pth の修復に失敗: {e}")
            return ckpt, p
        except Exception as e:
            print(f"[RESUME] 破損: {Path(p).name}: {e}")
    raise RuntimeError("有効な ckpt_*.pth が見つかりませんでした")

# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser('MLP-Mixer MAE pretrainer')
    p.add_argument('--train-dirs', nargs='+', required=True, help='one or more training image roots')
    p.add_argument('--output', required=True)
    p.add_argument('--img-size', type=int, default=480, choices=[384, 480])
    p.add_argument('--total-epochs', type=int, default=100)
    p.add_argument('--run-epochs', type=int, help='epochs to run in this session (default: total)')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr', type=float, default=3.27e-4)
    p.add_argument('--weight-decay', type=float, default=0.05)
    p.add_argument('--mask-ratio', type=float, default=0.75)
    p.add_argument('--decoder-dim', type=int, default=256)
    p.add_argument('--decoder-depth', type=int, default=3)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=None)
    p.add_argument('--resume', type=str, help='Resume from checkpoint')
    p.add_argument('--lr-find', action='store_true', help='Run LR range test and exit')
    p.add_argument('--lr-start', type=float, default=1e-6, help='LR finder: start LR (default: 1e-6)')
    p.add_argument('--lr-end', type=float, default=1.0, help='LR finder: end LR (default: 1.0)')
    p.add_argument('--lr-find-steps', type=int, default=800, help='LR finder: number of steps (batches)')
    p.add_argument('--lr-find-smoothing', type=float, default=0.98, help='LR finder: EMA smoothing factor')
    p.add_argument('--lr-find-diverge-mult', type=float, default=10.0, help='LR finder: early stop if loss > diverge_mult * best')
    p.add_argument('--compile-mode', type=str, default='default', choices=['default', 'max-autotune'], help='torch.compile mode (default: default)')
    return p.parse_args()

# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────

def set_seed(seed):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def lr_range_test(model, train_loader, device, amp_dtype, out_dir,
                  lr_start=1e-6, lr_end=1.0, num_steps=800, smoothing=0.98, diverge_mult=10.0,
                  weight_decay=0.05):
    """
    Exponential LR range test (Smith 2017).
    - 学習率を lr_start → lr_end に指数的に上げながら num_steps イテ分回す
    - 各ステップの loss（EMA平滑化込み）を記録
    - 早期終了: loss が最良値の diverge_mult 倍を超えたら停止
    """
    model.train()
    try:
        optim = torch.optim.AdamW(model.parameters(), lr=lr_start, weight_decay=weight_decay, fused=True)
    except TypeError:
        optim = torch.optim.AdamW(model.parameters(), lr=lr_start, weight_decay=weight_decay)

    scaler = None if amp_dtype == torch.bfloat16 else GradScaler('cuda')
    total = max(1, num_steps)
    mult = (lr_end / lr_start) ** (1 / total)

    MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    # ログ用
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / 'lr_find.csv'
    with open(csv_path, 'w', newline='') as f:
        csv.writer(f).writerow(['step', 'lr', 'loss', 'loss_ema'])

    best_loss = float('inf')
    loss_ema = None
    step = 0

    loader_iter = iter(train_loader)
    while step < total:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            batch = next(loader_iter)

        imgs = batch['pixel_values'].to(device, non_blocking=True)
        if imgs.is_contiguous(memory_format=torch.contiguous_format):
            imgs = imgs.to(memory_format=torch.channels_last)

        imgs.sub_(MEAN).div_(STD)

        # 学習率を設定（指数増加）
        lr = lr_start * (mult ** step)
        for pg in optim.param_groups:
            pg['lr'] = lr

        optim.zero_grad(set_to_none=True)
        with autocast('cuda', dtype=amp_dtype):
            loss = model(imgs)

        if scaler is None:
            loss.backward()
            optim.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

        loss_val = float(loss.item())
        if loss_ema is None:
            loss_ema = loss_val
        else:
            loss_ema = smoothing * loss_ema + (1 - smoothing) * loss_val

        # 記録
        with open(csv_path, 'a', newline='') as f:
            csv.writer(f).writerow([step, f'{lr:.6g}', f'{loss_val:.6g}', f'{loss_ema:.6g}'])

        # 早期終了：発散検知
        best_loss = min(best_loss, loss_ema)
        if loss_ema > diverge_mult * best_loss or not math.isfinite(loss_ema):
            print(f'[LR-FIND] early stop at step {step}: loss_ema={loss_ema:.4g}, best={best_loss:.4g}')
            break

        step += 1

    # 推奨LRのヒューリスティクス：
    # 「急降下が始まる辺り」を使う。実装簡易化のため、最良EMA損失の0.8〜0.9倍になる少し前のlrを選ぶ。
    # ここでは単純に "最良EMA損失を与えた時点の lr の 1/3" を提案（保守的で安定）。
    # 厳密にやるなら csv を読んで d(loss)/d(log(lr)) が最小の点を探す。
    import pandas as pd
    try:
        df = pd.read_csv(csv_path)
        idx_min = df['loss_ema'].idxmin()
        lr_at_min = float(df.loc[idx_min, 'lr'])
        suggested = lr_at_min / 3.0
    except Exception:
        suggested = lr_start * (mult ** max(1, step//2)) / 3.0  # フォールバック

    txt = out_dir / 'best_lr.txt'
    with open(txt, 'w') as f:
        f.write(f'suggested_lr,{suggested:.6g}\n')
    print(f'[LR-FIND] suggested base LR ≈ {suggested:.6g} (per-batch={train_loader.batch_size})')
    print(f'[LR-FIND] results saved to {csv_path} and {txt}')
    return suggested

# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    time.sleep(2)

    assert torch.cuda.is_available(), "This script requires CUDA (GPU)."
    device = torch.device('cuda')

    GPU_MEAN = torch.tensor([0.485, 0.456, 0.406], device='cuda').view(1,3,1,1)
    GPU_STD  = torch.tensor([0.229, 0.224, 0.225], device='cuda').view(1,3,1,1)

    IMG_SIZE = args.img_size
    PATCH_SIZE = 16  # Mixer-B/16

    # augmentations: light crop + Imagenet mean/std
    aug_tfms = transforms.Compose([
        #transforms.RandomResizedCrop(IMG_SIZE, scale=(0.9, 1.0), ratio=(1.0, 1.0)),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0), ratio=(0.95, 1.05)),
        #transforms.RandomAffine(degrees=2.0, translate=(0.02, 0.02)),
        transforms.RandomAffine(degrees=5.0, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=2.0),
        transforms.ColorJitter(brightness=0.05, contrast=0.05),
        transforms.ToTensor(),
    ])

    def normalize_inplace_(x):
        x.sub_(GPU_MEAN).div_(GPU_STD)
        return x

    # 3) CUDA prefetcher
    class CudaPrefetcher:
        def __init__(self, loader, device='cuda'):
            self.loader = loader
            self.device = device
            self.stream = torch.cuda.Stream()
            self.iter = iter(loader)
            self.next_batch = None
            self.preload()

        def preload(self):
            try:
                batch = next(self.iter)
            except StopIteration:
                self.next_batch = None
                return
            with torch.cuda.stream(self.stream):
                imgs = batch['pixel_values'].to(self.device, non_blocking=True)
                # channels_last はテンソルのみ（モデルに .to(memory_format=...) は不要）
                if imgs.is_contiguous(memory_format=torch.contiguous_format):
                    imgs = imgs.to(memory_format=torch.channels_last)
                normalize_inplace_(imgs)  # ★ GPUで正規化
                self.next_batch = imgs

        def __iter__(self):
            return self

        def __next__(self):
            torch.cuda.current_stream().wait_stream(self.stream)
            if self.next_batch is None:
                raise StopIteration
            batch = self.next_batch
            self.preload()
            return batch


    # Build datasets in parallel like the reference
    def build_concat(ds_paths, tfm):
        with ThreadPoolExecutor() as exe:
            return ConcatDataset([exe.submit(PretrainDataset, p, tfm).result() for p in ds_paths])
    train_ds = build_concat(args.train_dirs, aug_tfms)

    # WeightedRandomSampler across multiple roots
    lengths = [len(d) for d in train_ds.datasets]
    cum = [0]
    for L in lengths:
        cum.append(cum[-1] + L)
    w = torch.empty(len(train_ds), dtype=torch.float)
    for i, L in enumerate(lengths):
        start, end = cum[i], cum[i+1]
        w[start:end] = 1.0 / L
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
        prefetch_factor=6,
    )

    # Model
    model = MixerMAE(
        mixer_variant='mixer_b16_224',
        img_size=IMG_SIZE,
        in_chans=3,
        patch_size=PATCH_SIZE,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        mask_ratio=args.mask_ratio,
    )
    model.to(device)

    # channels_last for speed on Ampere/Blackwell
    # if torch.cuda.is_available():
    #    model.to(memory_format=torch.channels_last)


    # Mixed precision: prefer bfloat16 on RTX 5090
    use_bf16 = torch.cuda.is_available()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = None if use_bf16 else GradScaler('cuda')

    # CUDA perf knobs
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.deterministic = False

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- LR Range Test モード ---
    if args.lr_find:
        suggested = lr_range_test(
            model=model,
            train_loader=train_loader,
            device=device,
            amp_dtype=amp_dtype,
            out_dir=out_dir,
            lr_start=args.lr_start,
            lr_end=args.lr_end,
            num_steps=args.lr_find_steps,
            smoothing=args.lr_find_smoothing,
            diverge_mult=args.lr_find_diverge_mult,
            weight_decay=args.weight_decay,
        )
        # ここで終了（学習本番は行わない）
        return

    # training history CSV
    history_file = out_dir / 'training_history.csv'
    if not history_file.exists():
        append_history_atomic(str(history_file), ['epoch', 'train_loss'])

    # epochs
    if args.run_epochs is None:
        args.run_epochs = args.total_epochs

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.total_epochs

    # Resume
    start_epoch = 1
    current_step = 0
    ckpt = None
    if args.resume and os.path.exists(args.resume):
        print(f'Loading checkpoint from {args.resume}')
        ckpt, used_path = _safe_load_resume(args.resume, device)
        sd = _sanitize_state_dict_for_load(ckpt['model'])
        incompatible = model.load_state_dict(sd, strict=False)
        if getattr(incompatible, "missing_keys", None) or getattr(incompatible, "unexpected_keys", None):
            print(f"[WARN] resume missing={list(incompatible.missing_keys)} unexpected={list(incompatible.unexpected_keys)}")

        # === ここから互換性チェックを追加（警告のみの安全版） ===
        ckpt_args = ckpt.get('args', {})
        mismatch = []
        for k in ['img_size', 'mask_ratio']:
            if k in ckpt_args and k in vars(args) and ckpt_args[k] != vars(args)[k]:
                mismatch.append((k, ckpt_args[k], vars(args)[k]))
        if mismatch:
            print('[RESUME] config mismatch detected:', mismatch)
            # 厳格運用したい場合は次の2行のコメントアウトを外す：
            # raise RuntimeError(f'Config mismatch: {mismatch}')
            # → _safe_load_resume を自作で再呼びして別 ckpt に切替える処理を書く
        # === 互換性チェックここまで ===

        print('loaded weight norm (stem.proj):',
            sum(p.norm().item() for n, p in model.named_parameters()
                if 'encoder.stem.proj.weight' in n))
        start_epoch = ckpt.get('epoch', 0) + 1
        current_step = ckpt.get('step', 0)
        
    # Compile model based on argument
    try:
        if args.compile_mode == 'max-autotune':
            model = torch.compile(model, mode='max-autotune', fullgraph=True, dynamic=False)
            print('Model compiled (max-autotune)')
        else:  # default
            model = torch.compile(model, mode='default')
            print('Model compiled successfully (torch.compile default)')
    except Exception as e:
        print(f'Compile failed: {e}\nContinuing without compilation.')

    # --- Optimizer / Scheduler を作成してから、resume 時は state を復元 ---
    try:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, fused=True)
        print('Using fused AdamW')
    except TypeError:
        optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        print('Using standard AdamW')

    # cosine schedule like the reference
    def lr_func(step):
        pct = step / max(1, total_steps)
        return 0.5 * (1.0 + math.cos(math.pi * pct))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func)

    # --- Optimizer / Scheduler の復元（作成後に一度だけ！）---
    if ckpt is not None and 'optimizer' in ckpt and 'scheduler' in ckpt:
        optim.load_state_dict(ckpt['optimizer'])
        sched.load_state_dict(ckpt['scheduler'])
        print(f'Resumed at epoch {start_epoch}')

    # --- Optimizer / Scheduler 復元の直後あたり ---
    status = {'epoch': start_epoch - 1, 'avg': None, 'step': current_step}

    def save_latest(extra: dict | None = None):
        state = {
            'model': _sd_to_save(model),
            'optimizer': optim.state_dict(),
            'scheduler': sched.state_dict(),
            'epoch': status['epoch'],
            'train_loss': status['avg'],
            'args': vars(args),
            'step': status['step'],
        }
        if extra:
            state.update(extra)
        atomic_torch_save(state, out_dir / 'latest.pth')

    def _graceful_exit(*_):
        try:
            print('[SIGTERM] saving latest checkpoint and exiting...', flush=True)
            save_latest({'sigterm': True})
        finally:
            raise SystemExit(0)

    signal.signal(signal.SIGTERM, _graceful_exit)

    # if ek_schedule_exe:
    #     ek_schedule_exe.wait_until_target_time(23, 0)


    # Train
    try:
        end_epoch = min(start_epoch + args.run_epochs - 1, args.total_epochs)
        for epoch in range(start_epoch, end_epoch + 1):
            status['epoch'] = epoch
            model.train()
            running = 0.0
            
            prefetcher = CudaPrefetcher(train_loader, device)
            for step, imgs in enumerate(tqdm(prefetcher, desc=f'Epoch {epoch}', leave=False, disable=not USE_TQDM)):
                with autocast('cuda', dtype=amp_dtype):
                    loss = model(imgs)

                optim.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optim)
                    scaler.update()
                else:
                    loss.backward()
                    optim.step()

                sched.step()
                current_step += 1
                running += float(loss.item())

                if (step & 1023) == 0:
                    print(f'[{epoch}:{step}] loss={loss.item():.4f}', flush=True)
                
                status['step'] = current_step

            avg = running / len(train_loader)
            status['avg'] = avg
            print(f'Epoch {epoch}/{args.total_epochs}  train_loss: {avg:.4f}  (Mixer-MAE)')
            append_history_atomic(str(history_file), [epoch, f'{avg:.4f}'])

            # Save ckpt
            save_obj = {
                'model': _sd_to_save(model),
                'optimizer': optim.state_dict(),
                'scheduler': sched.state_dict(),
                'epoch': epoch,
                'train_loss': avg,
                'args': vars(args),
                'step': current_step,
            }
            atomic_torch_save(save_obj, out_dir / f'ckpt_{epoch:04d}.pth')
            atomic_torch_save(save_obj, out_dir / 'latest.pth')

            if epoch >= args.total_epochs:
                print('Training completed!')
                break

    except Exception:
        print('Exception occurred. Saving emergency checkpoint...')
        emergency = {
            'model': _sd_to_save(model),
            'optimizer': optim.state_dict(),
            'scheduler': sched.state_dict(),
            'epoch': max(1, status.get('epoch', 1) - 1),
            'args': vars(args),
            'step': status.get('step', 0),
        }
        atomic_torch_save(emergency, out_dir / 'latest.pth')
        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        atomic_torch_save(emergency, out_dir / f'ckpt_interrupt_{ts}.pth')
        raise

    # --- optional: 明示的クリーンアップ（安全策） ---
    try:
        # torch.compile している場合は _orig_mod も消しておく
        base = model._orig_mod if hasattr(model, "_orig_mod") else model
        del base, model, optim, sched, train_loader, train_ds
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    except Exception:
        pass

if __name__ == '__main__':
    main()
    print('Done. ', datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=9), 'JST')))

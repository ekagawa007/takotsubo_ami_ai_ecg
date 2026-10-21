# evaluate_mlpmixer_mae.py
# -------------------------------------------------
# Evaluate MAE-style Mixer checkpoints on a validation set and append results to evaluation.csv
#
# - Scans ckpt directory for files: ckpt_XXXX.pth or ckpt-XXXX.pth
# - Skips ones already present in evaluation.csv
# - Loads model (params inferred from ckpt['args'] unless overridden)
# - Computes average MAE reconstruction loss over val set
# - Appends: ckpt, epoch, train_loss, eval_loss, num_images, mask_ratio, img_size, batch_size, timestamp
#
# Usage example:
# python evaluate_mlpmixer_mae.py \
#   --val-dirs /path/to/val1 /path/to/val2 \
#   --ckpt-dir runs/mae_mixer_b16_480 \
#   --batch-size 64 --num-workers 8
#
# version 1.0.0 2025-10-04

import argparse, os, re, csv, sys, math, datetime
from pathlib import Path
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
import timm

try:
    from tqdm.auto import tqdm
except Exception:
    def tqdm(x, **k): return x

IS_TTY = sys.stdout.isatty()
#torch.set_float32_matmul_precision('high')

# ──────────────────────────────────────────
# Dataset (学習スクリプトと互換)
# ──────────────────────────────────────────
class PretrainDataset(Dataset):
    IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp')
    def __init__(self, root, transform=None, cache_size=64):
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
        if idx in self._img_cache:  # LRU
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
# Model (学習スクリプトと同等)
# ──────────────────────────────────────────
class MixerMAE(nn.Module):
    def __init__(
        self,
        mixer_variant: str = 'mixer_b16_224',
        img_size: int = 480,
        in_chans: int = 3,
        patch_size: int = 16,
        decoder_dim: int = 256,
        decoder_depth: int = 3,
        mask_ratio: float = 0.75,
    ):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio

        self.encoder = timm.create_model(
            mixer_variant,
            pretrained=False,
            img_size=img_size,
            in_chans=in_chans,
            num_classes=0,
        )
        assert hasattr(self.encoder, 'stem')
        assert hasattr(self.encoder, 'blocks')
        assert hasattr(self.encoder, 'norm')

        with torch.no_grad():
            dummy = torch.zeros(1, in_chans, img_size, img_size)
            x = self.encoder.stem(dummy)  # [1, S, C]
            self.enc_dim = x.shape[-1]

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.enc_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        dec_layers = []
        dim_in = self.enc_dim
        for _ in range(decoder_depth - 1):
            dec_layers += [nn.Linear(dim_in, decoder_dim), nn.GELU()]
            dim_in = decoder_dim
        dec_layers += [nn.Linear(dim_in, patch_size * patch_size * in_chans)]
        self.decoder = nn.Sequential(*dec_layers)

    def patchify(self, imgs: torch.Tensor) -> torch.Tensor:
        p = self.patch_size
        B, C, H, W = imgs.shape
        h, w = H // p, W // p
        x = imgs.reshape(B, C, h, p, w, p).permute(0, 2, 4, 3, 5, 1).reshape(B, h*w, p*p*C)
        return x

    @staticmethod
    def _random_mask(B: int, S: int, mask_ratio: float, device: torch.device):
        num_mask = int(S * mask_ratio)
        noise = torch.rand(B, S, device=device)
        ids_shuffle = torch.argsort(noise, dim=1)
        mask = torch.zeros(B, S, dtype=torch.bool, device=device)
        mask.scatter_(1, ids_shuffle[:, :num_mask], True)
        return mask

    def forward(self, imgs: torch.Tensor):
        B, _, H, W = imgs.shape
        S = (H // self.patch_size) * (W // self.patch_size)
        mask = self._random_mask(B, S, self.mask_ratio, imgs.device)

        x = self.encoder.stem(imgs)   # [B,S,C]
        mask_tok = self.mask_token.expand(B, S, -1)
        x = torch.where(mask.unsqueeze(-1), mask_tok, x)

        blocks = self.encoder.blocks if isinstance(self.encoder.blocks, (nn.Sequential, nn.ModuleList)) else [self.encoder.blocks]
        for blk in blocks:
            x = blk(x)
        if getattr(self.encoder, 'norm', None) is not None:
            x = self.encoder.norm(x)

        pred = self.decoder(x)  # [B,S,P]
        with torch.no_grad():
            target = self.patchify(imgs)
        mask_f = mask.float().unsqueeze(-1)
        loss = ((pred - target) ** 2) * mask_f
        P = pred.shape[2]
        loss = loss.sum() / (mask_f.sum() * P + 1e-8)
        return loss

# ──────────────────────────────────────────
# Utils
# ──────────────────────────────────────────
def build_val_dataset(paths, img_size):
    tfm = transforms.Compose([
        transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),  # float32 [0,1]
    ])
    with ThreadPoolExecutor() as exe:
        datasets = [exe.submit(PretrainDataset, p, tfm, 64).result() for p in paths]
    return ConcatDataset(datasets)

def list_ckpts(ckpt_dir: Path):
    pats = []
    pats += list(ckpt_dir.glob('ckpt_*.pth'))
    pats += list(ckpt_dir.glob('ckpt-*.pth'))
    # 降順ではなく昇順に並べる（epoch番号でソート）
    def ep(path: Path):
        m = re.search(r'ckpt[_-](\d+)\.pth$', path.name)
        return int(m.group(1)) if m else -1
    return sorted(pats, key=ep)

def load_evaluated(csv_path: Path):
    done = set()
    if csv_path.exists():
        with open(csv_path, 'r', newline='') as f:
            r = csv.DictReader(f)
            for row in r:
                done.add(row['ckpt'])
    return done

# --- 追加: フォーマッタ（安全に文字列化）
def _fmt6(x):
    try:
        return f'{float(x):.6f}'
    except Exception:
        return ''

def append_row(csv_path: Path, row: dict):
    write_header = not csv_path.exists()
    fieldnames = ['ckpt','epoch','train_loss','eval_loss','num_images','mask_ratio','img_size','batch_size','timestamp']
    with open(csv_path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        if write_header:
            w.writeheader()
        w.writerow(row)

# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser('Evaluate Mixer-MAE checkpoints')
    ap.add_argument('--val-dirs', nargs='+', required=True, help='validation image roots (one or more)')
    ap.add_argument('--ckpt-dir', required=True, help='directory containing ckpt_XXXX.pth')
    ap.add_argument('--img-size', type=int, default=None, help='override image size; default=use ckpt args')
    ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--num-workers', type=int, default=8)
    ap.add_argument('--mask-ratio', type=float, default=None, help='override; default=use ckpt args')
    ap.add_argument('--csv', type=str, default=None, help='path to evaluation.csv (default: <ckpt-dir>/evaluation.csv)')
    ap.add_argument('--all', action='store_true', help='evaluate all ckpts (ignore evaluation.csv)')
    ap.add_argument('--seed', type=int, default=1234, help='seed for deterministic masking during eval')
    args = ap.parse_args()

    assert torch.cuda.is_available(), 'CUDA GPU required'
    device = torch.device('cuda')

    ckpt_dir = Path(args.ckpt_dir)
    csv_path = Path(args.csv) if args.csv else (ckpt_dir / 'evaluation.csv')

    done = set() if args.all else load_evaluated(csv_path)
    
    ckpts = list_ckpts(ckpt_dir)
    if not ckpts:
        print(f'No ckpt_*.pth found under {ckpt_dir}')
        return

    # Validation dataset (デフォは ckpt の img_size を使いたいが、複数 ckpt を処理するため
    # ここでは最初の ckpt の img_size か CLI の --img-size を使用)
    first_args_imgsize = None
    for c in ckpts:
        try:
            tmp = torch.load(c, map_location='cpu')
            first_args_imgsize = tmp.get('args', {}).get('img_size', None)
            del tmp
            if first_args_imgsize is not None:
                break
        except Exception:
            continue
    eval_img_size = args.img_size if args.img_size is not None else (first_args_imgsize or 480)

    val_ds = build_val_dataset(args.val_dirs, eval_img_size)
    loader_kwargs = dict(
        dataset=val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        #pin_memory_device='cuda',
        drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )
    if args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = 2
    val_loader = DataLoader(**loader_kwargs)

    GPU_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1,3,1,1)
    GPU_STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1,3,1,1)

    def gpu_normalize_(x):
        # x: float32 [0,1]
        x.sub_(GPU_MEAN).div_(GPU_STD)
        return x

    evaluated = 0
    for ckpt_path in ckpts:
        # 既に CSV にある ckpt はスキップ（--all のときは done が空集合なので全実行）
        if ckpt_path.name in done:
            continue

        # ckpt ロード
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
        except Exception as e:
            print(f'[SKIP] failed to load {ckpt_path.name}: {e}')
            continue

        ckpt_args = ckpt.get('args', {})
        img_size = args.img_size if args.img_size is not None else int(ckpt_args.get('img_size', eval_img_size))
        mask_ratio = args.mask_ratio if args.mask_ratio is not None else float(ckpt_args.get('mask_ratio', 0.75))
        decoder_dim = int(ckpt_args.get('decoder_dim', 256))
        decoder_depth = int(ckpt_args.get('decoder_depth', 3))
        mixer_variant = ckpt_args.get('mixer_variant', 'mixer_b16_224')

        # モデル作成＆ロード
        model = MixerMAE(
            mixer_variant=mixer_variant,
            img_size=img_size,
            in_chans=3,
            patch_size=16,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            mask_ratio=mask_ratio,
        ).to(device)
        incompatible = model.load_state_dict(ckpt['model'], strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            print(f"[WARN] {ckpt_path.name} missing={list(incompatible.missing_keys)} "
                f"unexpected={list(incompatible.unexpected_keys)}")

        model.eval()

        # 評価ループ
        use_bf16 = torch.cuda.is_available()
        amp_dtype = torch.bfloat16 if use_bf16 else torch.float16

        # 可能なら、異なる ckpt 間で同一性を保つために固定 seed
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

        total_loss = 0.0
        num_batches = 0
        num_images = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Evaluating {ckpt_path.name}', leave=False, disable=not IS_TTY):
                imgs = batch['pixel_values'].to(device, non_blocking=True)
                # 入力解像度が ckpt と違う場合でも CenterCrop/Resize で合わせているはずだが、
                # 念のため channels_last にして正規化を GPU 側で実行
                if imgs.is_contiguous(memory_format=torch.contiguous_format):
                    imgs = imgs.to(memory_format=torch.channels_last)
                gpu_normalize_(imgs)

                with torch.amp.autocast('cuda', dtype=amp_dtype):
                    loss = model(imgs)

                bsz = imgs.size(0)
                total_loss += float(loss.item()) * bsz  # 画像数重みづけ平均
                num_images += bsz
                num_batches += 1

        avg_loss = total_loss / max(1, num_images)
        epoch = int(ckpt.get('epoch', -1))
        ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        row = {
            'ckpt': ckpt_path.name,
            'epoch': epoch,
            'train_loss': _fmt6(ckpt.get('train_loss')),  # 先に train_loss
            'eval_loss': f'{avg_loss:.6f}',               # loss → eval_loss にリネーム
            'num_images': num_images,
            'mask_ratio': mask_ratio,
            'img_size': img_size,
            'batch_size': args.batch_size,
            'timestamp': ts,
        }
        append_row(csv_path, row)

        evaluated += 1
        print(f'[OK] {ckpt_path.name}: eval_loss={avg_loss:.6f} (N={num_images}) → appended to {csv_path.name}')
        
    if evaluated == 0:
        print('No new checkpoints to evaluate.')


if __name__ == '__main__':
    main()

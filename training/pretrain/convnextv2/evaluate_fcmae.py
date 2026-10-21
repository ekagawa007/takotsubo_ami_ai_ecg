'''
evaluate_fcmae.py
-------------------------------------------------
Evaluation script for FCMAE models.
Loads a checkpoint and evaluates on validation data.

Usage:
    python evaluate_fcmae.py --checkpoint-dir /path/to/checkpoints --val-dirs /path/to/val --img-size 480

version 1.0.1 2025-08-11 訓練の途中で強制終了すると、latest.pthが更新されず、すでに評価してあるものも
                         再度評価するようで、latest.pthを読まないように修正した。
'''

import argparse
import os
import csv
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from torch.amp import autocast
from PIL import Image
from mae_fcmae_wrapper import FCMAE
from tqdm.auto import tqdm

# ──────────────────────────────────────────
# Dataset (same as training)
# ──────────────────────────────────────────
class PretrainDataset(Dataset):
    """ImageFolder‑like loader w/ optional augmentation & simple caching"""
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

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
        img = Image.open(self.files[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        res = {"pixel_values": img}
        # naive cache
        if len(self.cache) >= self.cache_size:
            self.cache.pop(next(iter(self.cache)))
        self.cache[idx] = res
        return res

# ──────────────────────────────────────────
# CLI
# ──────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser("FCMAE Evaluator")
    p.add_argument('--checkpoint-dir', required=True, help='Directory containing checkpoints to evaluate')
    p.add_argument('--val-dirs', nargs='+', required=True, help='Validation data directories')
    p.add_argument('--img-size', type=int, default=480, choices=[384, 480], help='Image size for evaluation (default: 480, will be overridden by checkpoint if available)')
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--num-workers', type=int, default=8)
    p.add_argument('--seed', type=int, default=None)
    return p.parse_args()

# ──────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────
def set_seed(seed):
    import random
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ──────────────────────────────────────────
# Main
# ──────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    # Find all checkpoint files in the directory
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {args.checkpoint_dir}")
    
    # Find all .pth files except latest.pth and sort them by name
    checkpoint_files = sorted(
        [p for p in checkpoint_dir.glob('*.pth') if p.name != 'latest.pth'],
        key=lambda x: x.name
    )

    print(f"Found {len(checkpoint_files)} checkpoint files in {args.checkpoint_dir}")
    print("Checkpoints will be evaluated in this order:")
    for i, ckpt_file in enumerate(checkpoint_files, 1):
        print(f"  {i:2d}. {ckpt_file.name}")
    
    # Load existing evaluation results
    eval_file = checkpoint_dir / 'evaluation_results.csv'
    evaluated_checkpoints = set()
    if eval_file.exists():
        with open(eval_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                evaluated_checkpoints.add(row['checkpoint'])
        print(f"Found {len(evaluated_checkpoints)} already evaluated checkpoints")
    
    # Filter unevaluated checkpoints
    unevaluated_checkpoints = []
    for checkpoint_file in checkpoint_files:
        checkpoint_name = checkpoint_file.stem
        if checkpoint_name not in evaluated_checkpoints:
            unevaluated_checkpoints.append(checkpoint_file)
    
    if not unevaluated_checkpoints:
        print("All checkpoints have been evaluated!")
        return
    
    print(f"Found {len(unevaluated_checkpoints)} unevaluated checkpoints")
    
    # Load first checkpoint to get model configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    first_ckpt = torch.load(unevaluated_checkpoints[0], map_location=device)
    
    # Extract model configuration from checkpoint
    saved_args = first_ckpt.get('args', {})
    # Use command line argument as default, but override with checkpoint if available
    img_size = saved_args.get('img_size', args.img_size)
    patch_size = saved_args.get('patch_size', 4)
    mask_ratio = saved_args.get('mask_ratio', 0.75)
    
    print(f"Model config: img_size={img_size}, patch_size={patch_size}, mask_ratio={mask_ratio}")
    print(f"Using image size: {img_size} (from checkpoint: {saved_args.get('img_size', 'not found')}, command line: {args.img_size})")
    
    # Create model
    model = FCMAE('convnextv2_base', img_size=img_size, patch_size=patch_size, mask_ratio=mask_ratio)
    model.to(device)
    
    # Data transforms (validation only)
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])
    
    # Build validation dataset
    def build(ds_paths, tfm):
        with ThreadPoolExecutor() as exe:
            return ConcatDataset([exe.submit(PretrainDataset,p,tfm).result() for p in ds_paths])
    val_ds = build(args.val_dirs, val_tfms)
    
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True, drop_last=False)
    
    # Use BF16 for RTX 5090
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # Check if evaluation results file exists
    file_exists = eval_file.exists()
    
    # Evaluate each unevaluated checkpoint
    for checkpoint_file in unevaluated_checkpoints:
        print(f"\nEvaluating {checkpoint_file.name}...")
        
        # Load checkpoint
        ckpt = torch.load(checkpoint_file, map_location=device)
        
        # Handle torch.compile() state dict (remove _orig_mod. prefix if present)
        state_dict = ckpt['model']
        if any(key.startswith('_orig_mod.') for key in state_dict.keys()):
            print("Detected torch.compile() state dict, removing _orig_mod. prefix...")
            new_state_dict = {}
            for key, value in state_dict.items():
                if key.startswith('_orig_mod.'):
                    new_key = key[10:]  # Remove '_orig_mod.' prefix
                    new_state_dict[new_key] = value
                else:
                    new_state_dict[key] = value
            state_dict = new_state_dict
        
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"Epoch: {ckpt['epoch']}")
        print(f"Train loss: {ckpt.get('train_loss', 'N/A')}")
        
        # Evaluation
        print(f"Evaluating on {len(val_ds)} validation samples...")
        val_running = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for val_batch in tqdm(val_loader, desc=f"Evaluating {checkpoint_file.name}"):
                val_imgs = val_batch['pixel_values'].to(device, non_blocking=True)
                with autocast('cuda', dtype=dtype):
                    val_loss = model(val_imgs)
                val_running += val_loss.item()
                num_batches += 1
        
        val_avg = val_running / num_batches
        
        # Save evaluation results to CSV
        checkpoint_name = checkpoint_file.stem
        
        with open(eval_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Write header if file doesn't exist
            if not file_exists:
                writer.writerow(['checkpoint', 'epoch', 'train_loss', 'val_loss'])
                file_exists = True
            
            # Write evaluation result
            writer.writerow([
                checkpoint_name,
                ckpt['epoch'],
                ckpt.get('train_loss', 'N/A'),
                f'{val_avg:.4f}'
            ])
        
        print(f"Validation loss: {val_avg:.4f}")
        print(f"Results saved to: {eval_file}")
    
    print(f"\nEvaluation completed! Results saved to: {eval_file}")
    print('Done. ', datetime.datetime.now(datetime.timezone(datetime.timedelta(hours = 9), 'JST')))

if __name__ == '__main__':
    main() 
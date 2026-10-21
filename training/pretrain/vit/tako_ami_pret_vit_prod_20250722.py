import os, torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from transformers import (
    ViTMAEForPreTraining,
    AutoImageProcessor,
    TrainingArguments,
    Trainer,
    default_data_collator
)
import matplotlib.pyplot as plt
from ek_schedule_exe import wait_until_target_time
import datetime
import random
import tensorboard
from concurrent.futures import ThreadPoolExecutor
import time
import math

torch.manual_seed(42)
random.seed(42)

# スケジュール実行用の設定
SCHEDULED_RUN = True #False #True
TARGET_HOUR = 1 #22 #23 #0 #23
TARGET_MINUTE = 0 #30

# レジューム用の設定
RESUME_FROM_CHECKPOINT = True #False #True  # レジュームするかどうか
# CHECKPOINT_PATH = f"{MODEL_SAVE_DIR}/checkpoint-4868"  # チェックポイントのパス（XXXXは実際のステップ数）

NOTEBOOK_STEM = "tako_ami_pret_vit_prod_20250722"

IMG_PRETRAIN_TRAIN_DIR = "/path/to/data/pretrain_ecg_480/train"
#IMG_PRETRAIN_TRAIN_DIR = "/path/to/data/pretrain_ecg_480/train"
#IMG_PRETRAIN_ORIG_TRAIN_DIR = "/path/to/data/ecg_dataset_480_previous/pretrain_train"
IMG_PRETRAIN_ORIG_TRAIN_DIR = "/path/to/data/ecg_dataset_480/train"
IMG_PRETRAIN_VALID_DIR = "/path/to/data/pretrain_ecg_480/valid"
#IMG_PRETRAIN_VALID_DIR = "/path/to/data/pretrain_ecg_480/valid"
#IMG_PRETRAIN_ORIG_VALID_DIR = "/path/to/data/ecg_dataset_480_previous/pretrain_valid"
IMG_PRETRAIN_ORIG_VALID_DIR = "/path/to/data/ecg_dataset_480/pt_valid"

MODEL_NAME       = "facebook/vit-mae-base"
MODEL_SAVE_DIR   = f"/path/to/checkpoints/pretrain/vit/mae_pretrained_{NOTEBOOK_STEM}"
IMAGE_SIZE       = 480
BATCH_SIZE       = 128 #256 #128 #8 #16 #32 #64 #128 #256 #128 #64 #128 #256 #128 #64
# 夜間学習用の設定
# 学習率設定
INITIAL_LR       = 1e-4  # 初期学習率
TOTAL_EPOCHS     = 100  # 総学習エポック数
EPOCHS_PER_SESSION = 5 #29 #20  # 1回の学習セッションあたりのエポック数
NUM_EPOCHS       = 100 #49 #36 #セッションの終了epoch

def cosine_annealing_lr(epoch, max_epochs, initial_lr, final_lr=0):
    """
    コサインアニーリング学習率の計算
    
    Args:
        epoch: 現在のエポック
        max_epochs: 最大エポック数
        initial_lr: 初期学習率
        final_lr: 最終学習率（デフォルト: 0）
    
    Returns:
        現在のエポックの学習率
    """
    return final_lr + (initial_lr - final_lr) * 0.5 * (1 + math.cos(math.pi * epoch / max_epochs))

MIN_LR = cosine_annealing_lr(NUM_EPOCHS, TOTAL_EPOCHS, INITIAL_LR, 0)
LR = cosine_annealing_lr(NUM_EPOCHS - EPOCHS_PER_SESSION - 1, TOTAL_EPOCHS, INITIAL_LR, 0)
#MIN_LR           = 7.27e-5 #8.75e-5 # 最低学習率 (手動で変更)
#LR               = 9.38e-5 #9.98e-5 #INITIAL_LR  # 現在の学習率（手動で変更）

def calculate_lr_for_epoch(current_epoch, total_epochs=100, initial_lr=INITIAL_LR, min_lr=MIN_LR):
    """指定エポックでの学習率を計算（コサイン減衰 + 最低学習率保証）"""
    progress = current_epoch / total_epochs
    cosine_lr = initial_lr * 0.5 * (1 + math.cos(math.pi * progress))
    return max(cosine_lr, min_lr)  # 最低学習率を保証

def print_lr_schedule(start_epoch, end_epoch, total_epochs=100):
    """指定エポック範囲での学習率スケジュールを表示"""
    print(f"学習率スケジュール（エポック{start_epoch}-{end_epoch}）:")
    print(f"  初期学習率: {INITIAL_LR:.2e}")
    print(f"  最低学習率: {MIN_LR:.2e}")
    for epoch in range(start_epoch, min(end_epoch + 1, total_epochs + 1)):
        lr = calculate_lr_for_epoch(epoch, total_epochs, INITIAL_LR, MIN_LR)
        print(f"  エポック{epoch}: {lr:.2e}")

def find_latest_checkpoint(model_dir):
    """最新のチェックポイントを自動検出"""
    if not os.path.exists(model_dir):
        return None
    
    checkpoints = []
    for item in os.listdir(model_dir):
        if item.startswith("checkpoint-"):
            checkpoints.append(item)
    
    if not checkpoints:
        return None
    
    # ステップ数でソートして最新を取得
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    latest_checkpoint = os.path.join(model_dir, checkpoints[-1])
    print(f"最新のチェックポイントを検出: {latest_checkpoint}")
    return latest_checkpoint

def calculate_remaining_epochs(checkpoint_path, total_epochs, epochs_per_session):
    """残りエポック数を計算"""
    if not checkpoint_path:
        return epochs_per_session
    
    # チェックポイントから現在のエポック数を推定
    checkpoint_name = os.path.basename(checkpoint_path)
    if "checkpoint-" in checkpoint_name:
        step_number = int(checkpoint_name.split("-")[1])
        current_epoch = step_number / (len(train_ds) // BATCH_SIZE)
        
        # 残りエポック数を計算
        remaining_epochs = total_epochs - current_epoch
        if remaining_epochs <= 0:
            print(f"学習完了: {current_epoch:.1f}エポック達成")
            return 0
        elif remaining_epochs < epochs_per_session:
            print(f"最終セッション: 残り{remaining_epochs:.1f}エポック")
            return int(remaining_epochs)
        else:
            print(f"セッション継続: 残り{remaining_epochs:.1f}エポック")
            return epochs_per_session
    
    return epochs_per_session


processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)

# 1) augmentation 用の transform を定義（ToTensor/Normalize は不要）
augment = transforms.Compose([
    transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
    # RandomHorizontalFlip/VerticalFlip は心電図では NG
    # 必要なら小さめのガウシアンノイズや微小シフトをここに入れる
])


class PretrainDataset(Dataset):
    def __init__(self, root, processor, augment=None, cache_size=1000):
        if not os.path.exists(root):
            raise ValueError(f"指定されたディレクトリが存在しません: {root}")
            
        self.files = []
        self.processor = processor
        self.augment = augment
        self.cache = {}
        self.cache_size = cache_size
        
        # キャッシュファイルのパス
        cache_file = f"{root}_filelist_cache.txt"
        
        # キャッシュファイルが存在する場合は読み込み
        if os.path.exists(cache_file):
            print(f"キャッシュファイルから読み込み中: {cache_file}")
            with open(cache_file, 'r') as f:
                self.files = [line.strip() for line in f.readlines()]
            print(f"キャッシュから読み込んだ画像ファイル数: {len(self.files)}")
        else:
            # os.walkを使用して高速化
            print(f"ディレクトリ探索中: {root}")
            for root_dir, dirs, files in os.walk(root):
                # miscディレクトリをスキップ
                if "misc" in dirs:
                    dirs.remove("misc")
                
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        file_path = os.path.join(root_dir, file)
                        if os.path.isfile(file_path):
                            self.files.append(file_path)
            
            # キャッシュファイルに保存
            print(f"ファイルリストをキャッシュに保存: {cache_file}")
            with open(cache_file, 'w') as f:
                for file_path in self.files:
                    f.write(f"{file_path}\n")
        
        if len(self.files) == 0:
            raise ValueError(f"指定されたディレクトリに画像ファイルが見つかりません: {root}")
            
        print(f"読み込んだ画像ファイル数: {len(self.files)}")
        print(f"最初の数個のファイルパス:")
        for i in range(min(5, len(self.files))):
            print(self.files[i])

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
            
        try:
            img = Image.open(self.files[idx]).convert("RGB")
            # ↓ ここでまず augmentation をかける
            if self.augment is not None:
                img = self.augment(img)
            # ↓ そのあと processor がリサイズ・正規化・tensor化
            pixel_values = processor(images=img, return_tensors="pt").pixel_values[0]
            result = {"pixel_values": pixel_values}
            
            if len(self.cache) >= self.cache_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
            self.cache[idx] = result
            return result
        except Exception as e:
            print(f"エラーが発生したファイル: {self.files[idx]}")
            raise e

# ────────────────────────────────────
# 1) トレーニング用データセットの準備（並列処理で高速化）
# ────────────────────────────────────
print("データセット初期化開始...")
start_time = time.time()

def create_dataset(dataset_path, augment):
    return PretrainDataset(dataset_path, processor, augment=augment)

# 並列でデータセットを作成
with ThreadPoolExecutor(max_workers=2) as executor:
    physio_future = executor.submit(create_dataset, IMG_PRETRAIN_TRAIN_DIR, augment)
    orig_future = executor.submit(create_dataset, IMG_PRETRAIN_ORIG_TRAIN_DIR, augment)
    
    physio_train_ds = physio_future.result()
    orig_train_ds = orig_future.result()

train_ds = ConcatDataset([physio_train_ds, orig_train_ds])

n_phys = len(physio_train_ds)
n_our = len(orig_train_ds)
print(f"PhysioNetデータセット: {n_phys}枚")
print(f"オリジナルデータセット: {n_our}枚")
print(f"合計: {n_phys + n_our}枚")
print(f"データセット初期化完了: {time.time() - start_time:.2f}秒")

# 重みの構築を最適化
weights = torch.cat([
    torch.ones(n_phys) / n_phys,
    torch.ones(n_our) / n_our
])
train_sampler = WeightedRandomSampler(weights, num_samples=len(train_ds), replacement=True)

# ────────────────────────────────────
# 2) 検証用データセットの準備（並列処理で高速化）
# ────────────────────────────────────
print("検証データセット初期化開始...")
valid_start_time = time.time()

# 並列で検証データセットを作成
with ThreadPoolExecutor(max_workers=2) as executor:
    physio_valid_future = executor.submit(create_dataset, IMG_PRETRAIN_VALID_DIR, None)
    orig_valid_future = executor.submit(create_dataset, IMG_PRETRAIN_ORIG_VALID_DIR, None)
    
    physio_valid_ds = physio_valid_future.result()
    orig_valid_ds = orig_valid_future.result()

valid_ds = ConcatDataset([physio_valid_ds, orig_valid_ds])
print(f"検証データセット初期化完了: {time.time() - valid_start_time:.2f}秒")

# （検証は順番に見るだけで OK なら sampler は不要です）
# valid_sampler = WeightedRandomSampler([...], num_samples=len(valid_ds), replacement=True)



# ────────────────────────────────────
# 3) Trainer の定義
# ────────────────────────────────────
class DomainMixTrainer(Trainer):
    def get_train_dataloader(self):
        return DataLoader(
            train_ds,
            sampler=train_sampler,
            batch_size=self.args.train_batch_size,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )
    def get_eval_dataloader(self, eval_dataset=None):
        return DataLoader(
            valid_ds,
            batch_size=self.args.eval_batch_size,
            shuffle=False,            # 好みに応じて sampler も可
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

# モデルの初期化
if RESUME_FROM_CHECKPOINT:
    checkpoint_path = find_latest_checkpoint(MODEL_SAVE_DIR)
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"チェックポイントから学習を再開します: {checkpoint_path}")
        model = ViTMAEForPreTraining.from_pretrained(checkpoint_path)
    else:
        print("チェックポイントが見つかりません。新規に学習を開始します")
        model = ViTMAEForPreTraining.from_pretrained(MODEL_NAME)
else:
    print("新規に学習を開始します")
    model = ViTMAEForPreTraining.from_pretrained(MODEL_NAME)

args = TrainingArguments(
    output_dir          = MODEL_SAVE_DIR,
    eval_strategy       = "epoch",
    save_strategy       = "epoch",
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size = 16,
    num_train_epochs    = NUM_EPOCHS,
    learning_rate       = LR,
    lr_scheduler_type   = "cosine",
    warmup_ratio        = 0.05,
    logging_dir         = "./tb_logs",
    logging_strategy    = "epoch",
    report_to           = "tensorboard",
    prediction_loss_only = True,
    load_best_model_at_end = False,
    remove_unused_columns = False,
    dataloader_num_workers = 16, #8,
    dataloader_pin_memory = True,
    dataloader_prefetch_factor = 2,
    dataloader_persistent_workers = True,
    bf16 = torch.cuda.is_bf16_supported(),
)

# 再構成損失 (MSE) だけをログ
def mae_loss_only(eval_pred): return {}

'''
trainer = Trainer(
    model           = model,
    args            = args,
    train_dataset   = train_ds,
    eval_dataset    = val_ds,
    compute_metrics = None, #lambda p: {"val_mse": p.loss}
)
'''

# 5 Trainer のセットアップはほぼそのまま
trainer = DomainMixTrainer(
    model           = model,
    args            = args,
    train_dataset   = None,
    eval_dataset    = valid_ds,
    data_collator   = default_data_collator,
    compute_metrics = None,
)


if SCHEDULED_RUN: wait_until_target_time(TARGET_HOUR, TARGET_MINUTE)

# 学習開始時間を記録
train_start_time = time.time()
print(f"学習開始: {datetime.datetime.now(datetime.timezone(datetime.timedelta(hours = 9), 'JST'))}")

# レジュームの設定と動的エポック数調整
if RESUME_FROM_CHECKPOINT:
    checkpoint_path = find_latest_checkpoint(MODEL_SAVE_DIR)
    if checkpoint_path and os.path.exists(checkpoint_path):
        # チェックポイントの情報を表示
        checkpoint_name = os.path.basename(checkpoint_path)
        if "checkpoint-" in checkpoint_name:
            step_number = int(checkpoint_name.split("-")[1])
            current_epoch = step_number / (len(train_ds) // BATCH_SIZE)
            print(f"チェックポイント情報:")
            print(f"  ファイル: {checkpoint_name}")
            print(f"  ステップ数: {step_number}")
            print(f"  現在のエポック: {current_epoch:.2f}")
            
            # 残りエポック数を計算してNUM_EPOCHSを動的調整
            remaining_epochs = calculate_remaining_epochs(checkpoint_path, TOTAL_EPOCHS, EPOCHS_PER_SESSION)
            if remaining_epochs == 0:
                print("学習が完了しています。プログラムを終了します。")
                exit(0)
            elif remaining_epochs < 0:
                print(f"エラー: 予期しないエポック数 {remaining_epochs}")
                exit(1)
            
            # 今回の学習エポック数を設定
            current_session_epochs = remaining_epochs
            print(f"  今回の学習エポック数: {current_session_epochs}")
            
            # 学習率情報を表示
            current_lr = calculate_lr_for_epoch(current_epoch, TOTAL_EPOCHS, INITIAL_LR, MIN_LR)
            print(f"  現在の学習率: {current_lr:.2e}")
            print(f"  設定された学習率: {LR:.2e}")
            
            # 今回のセッションでの学習率範囲を表示
            end_epoch = min(current_epoch + current_session_epochs, TOTAL_EPOCHS)
            print_lr_schedule(int(current_epoch), int(end_epoch), TOTAL_EPOCHS)
            
            # TrainingArgumentsを再作成
            args = TrainingArguments(
                output_dir          = MODEL_SAVE_DIR,
                eval_strategy       = "epoch",
                save_strategy       = "epoch",
                per_device_train_batch_size = BATCH_SIZE,
                per_device_eval_batch_size = 16,
                num_train_epochs    = NUM_EPOCHS, #current_session_epochs,  # 動的に調整された値
                learning_rate       = LR,
                lr_scheduler_type   = "cosine",
                warmup_ratio        = 0.05,
                logging_dir         = "./tb_logs",
                logging_strategy    = "epoch",
                report_to           = "tensorboard",
                prediction_loss_only = True,
                load_best_model_at_end = False,
                remove_unused_columns = False,
                dataloader_num_workers = 16,
                dataloader_pin_memory = True,
                dataloader_prefetch_factor = 2,
                dataloader_persistent_workers = True,
                bf16 = torch.cuda.is_bf16_supported(),
            )
            
            # Trainerを再作成
            trainer = DomainMixTrainer(
                model           = model,
                args            = args,
                train_dataset   = None,
                eval_dataset    = valid_ds,
                data_collator   = default_data_collator,
                compute_metrics = None,
            )
        
        trainer.train(resume_from_checkpoint=checkpoint_path)
    else:
        # 初回実行時は通常のtrainerを使用
        print(f"初回実行: {EPOCHS_PER_SESSION}エポックで学習開始")
        print(f"  設定された学習率: {LR:.2e}")
        print_lr_schedule(0, EPOCHS_PER_SESSION, TOTAL_EPOCHS)
        trainer.train()
else:
    trainer.train()

print(f"学習完了: {time.time() - train_start_time:.2f}秒")
# log_history は dict のリストで、
# それぞれに 'loss'（トレーニング損失）, 'epoch', 'val_mse'（検証MSE）が入っている
logs = trainer.state.log_history


# ★ この重みをフェーズ②で fine-tune ★
model.save_pretrained(MODEL_SAVE_DIR)
processor.save_pretrained(MODEL_SAVE_DIR)



# 各ステップ・各エポックのトレーニング損失
train_steps = [l["step"] for l in logs if "loss" in l]
train_loss  = [l["loss"] for l in logs if "loss" in l]

# 各エポックごとの検証MSE
val_epochs = [l["epoch"] for l in logs if "val_mse" in l]
val_mse    = [l["val_mse"] for l in logs if "val_mse" in l]

plt.figure(1, figsize=(20, 5))  # 一度に1つのfigureを作成

# (1) トレーニング損失 vs step
plt.subplot(1,2,1)
plt.plot(train_steps, train_loss, label="train MSE")
plt.xlabel("Step")
plt.ylabel("MSE")
plt.title("Train Loss")
plt.grid(True)

# (2) 検証MSE vs epoch
plt.subplot(1,2,2)
plt.plot(val_epochs, val_mse, marker="o", label="val MSE", color="orange")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.title("Validation MSE")
plt.grid(True)

plt.tight_layout()
# グラフを表示（非ブロッキング）
plt.show(block=False)

# 同時にファイルとして保存
plt.savefig(f'{MODEL_SAVE_DIR}/training_curves.png', dpi=300, bbox_inches='tight')

print('Done. ', datetime.datetime.now(datetime.timezone(datetime.timedelta(hours = 9), 'JST')))
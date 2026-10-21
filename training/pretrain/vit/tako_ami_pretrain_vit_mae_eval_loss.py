# あとからeval_lossを計算するコード。結果を保存し、結果が保存してあるものはスキップする。
# version 1.0.0 20250724 first release

import os
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
from PIL import Image
from transformers import (
    ViTMAEForPreTraining,
    AutoImageProcessor,
    default_data_collator
)
import matplotlib.pyplot as plt
import json
import time
from concurrent.futures import ThreadPoolExecutor
import datetime

# 設定
MODEL_NAME = "facebook/vit-mae-base"
MODEL_SAVE_DIR = "/path/to/checkpoints/pretrain/vit/mae_pretrained_tako_ami_pret_vit_prod_20250722"
IMAGE_SIZE = 480
BATCH_SIZE = 16  # 評価時は小さめのバッチサイズ

# データセットパス
IMG_PRETRAIN_VALID_DIR = "/path/to/data/pretrain_ecg_480/valid"
IMG_PRETRAIN_ORIG_VALID_DIR = "/path/to/data/ecg_dataset_480/pt_valid"

# プロセッサーの初期化
processor = AutoImageProcessor.from_pretrained(MODEL_NAME, use_fast=True)

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

    def __len__(self): 
        return len(self.files)

    def __getitem__(self, idx):
        if idx in self.cache:
            return self.cache[idx]
            
        try:
            img = Image.open(self.files[idx]).convert("RGB")
            # 評価時はaugmentationなし
            if self.augment is not None:
                img = self.augment(img)
            # processor がリサイズ・正規化・tensor化
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

def create_dataset(dataset_path, processor):
    return PretrainDataset(dataset_path, processor, augment=None)

def calculate_eval_loss(model, valid_dataloader, device):
    """指定されたモデルでeval_lossを計算"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    print(f"  eval_loss計算中...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(valid_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"    バッチ {batch_idx}: 現在の平均損失 = {total_loss/num_batches:.4f}")
    
    avg_loss = total_loss / num_batches
    print(f"  eval_loss計算完了: {avg_loss:.4f}")
    return avg_loss

def find_checkpoints(model_dir):
    """保存されたチェックポイントを検索"""
    if not os.path.exists(model_dir):
        print(f"モデルディレクトリが存在しません: {model_dir}")
        return []
    
    checkpoints = []
    for item in os.listdir(model_dir):
        if item.startswith("checkpoint-"):
            checkpoints.append(item)
    
    if not checkpoints:
        print("チェックポイントが見つかりません")
        return []
    
    # ステップ数でソート
    checkpoints.sort(key=lambda x: int(x.split("-")[1]))
    return checkpoints

def load_existing_results(results_file):
    """既存の結果ファイルを読み込み"""
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                existing_results = json.load(f)
            print(f"既存の結果ファイルを読み込みました: {len(existing_results)}件")
            return existing_results
        except Exception as e:
            print(f"既存の結果ファイルの読み込みに失敗: {e}")
    return []

def save_checkpoint_result(checkpoint_name, result, results_file):
    """個別チェックポイントの結果を保存"""
    # 既存の結果を読み込み
    existing_results = load_existing_results(results_file)
    
    # 既存の結果から同じチェックポイントを削除（更新のため）
    existing_results = [r for r in existing_results if r['checkpoint'] != checkpoint_name]
    
    # 新しい結果を追加
    existing_results.append(result)
    
    # ステップ数でソート
    existing_results.sort(key=lambda x: x['step'])
    
    # 保存
    with open(results_file, 'w') as f:
        json.dump(existing_results, f, indent=2)
    
    print(f"  結果を保存しました: {results_file}")

def main():
    print("eval_loss計算開始...")
    
    # 結果ファイルのパス
    results_file = os.path.join(MODEL_SAVE_DIR, "eval_loss_results.json")
    
    # 既存の結果を読み込み
    existing_results = load_existing_results(results_file)
    existing_checkpoints = {r['checkpoint'] for r in existing_results}
    
    # デバイス設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")
    
    # 検証データセットの準備
    print("検証データセット初期化中...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        physio_valid_future = executor.submit(create_dataset, IMG_PRETRAIN_VALID_DIR, processor)
        orig_valid_future = executor.submit(create_dataset, IMG_PRETRAIN_ORIG_VALID_DIR, processor)
        
        physio_valid_ds = physio_valid_future.result()
        orig_valid_ds = orig_valid_future.result()
    
    valid_ds = ConcatDataset([physio_valid_ds, orig_valid_ds])
    print(f"検証データセット初期化完了: {time.time() - start_time:.2f}秒")
    print(f"検証データ数: {len(valid_ds)}")
    
    # データローダーの準備
    valid_dataloader = DataLoader(
        valid_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=default_data_collator,
        num_workers=8,
        pin_memory=True,
    )
    
    # チェックポイントを検索
    checkpoints = find_checkpoints(MODEL_SAVE_DIR)
    if not checkpoints:
        return
    
    print(f"見つかったチェックポイント数: {len(checkpoints)}")
    print(f"既に計算済みのチェックポイント数: {len(existing_checkpoints)}")
    
    # 計算が必要なチェックポイントを特定
    new_checkpoints = [cp for cp in checkpoints if cp not in existing_checkpoints]
    print(f"新しく計算が必要なチェックポイント数: {len(new_checkpoints)}")
    
    if not new_checkpoints:
        print("すべてのチェックポイントが既に計算済みです。")
    else:
        # 新しいチェックポイントのみ計算
        for i, checkpoint_name in enumerate(new_checkpoints):
            checkpoint_path = os.path.join(MODEL_SAVE_DIR, checkpoint_name)
            print(f"\n[{i+1}/{len(new_checkpoints)}] チェックポイント: {checkpoint_name}")
            
            try:
                # モデルを読み込み
                print(f"  モデル読み込み中: {checkpoint_path}")
                model = ViTMAEForPreTraining.from_pretrained(checkpoint_path)
                model.to(device)
                
                # eval_lossを計算
                eval_loss = calculate_eval_loss(model, valid_dataloader, device)
                
                # ステップ数とエポック数を推定
                step_number = int(checkpoint_name.split("-")[1])
                # エポック数の推定
                estimated_epoch = step_number / 13939
                
                result = {
                    "checkpoint": checkpoint_name,
                    "step": step_number,
                    "estimated_epoch": estimated_epoch,
                    "eval_loss": eval_loss
                }
                
                # 個別に結果を保存
                save_checkpoint_result(checkpoint_name, result, results_file)
                
                print(f"  結果: ステップ{step_number}, 推定エポック{estimated_epoch:.2f}, eval_loss={eval_loss:.4f}")
                
                # メモリ解放
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"  エラー: {e}")
                continue
    
    # 最終的な結果を読み込み
    final_results = load_existing_results(results_file)
    
    # 結果を表示
    print("\n=== eval_loss結果 ===")
    print("チェックポイント\t\tステップ\t推定エポック\teval_loss")
    print("-" * 80)
    for result in final_results:
        print(f"{result['checkpoint']}\t{result['step']}\t{result['estimated_epoch']:.2f}\t\t{result['eval_loss']:.4f}")
    
    # グラフ作成
    if len(final_results) > 1:
        steps = [r['step'] for r in final_results]
        eval_losses = [r['eval_loss'] for r in final_results]
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, eval_losses, 'bo-', linewidth=2, markersize=6)
        plt.xlabel('Step')
        plt.ylabel('Eval Loss')
        plt.title('Eval Loss vs Training Step')
        plt.grid(True, alpha=0.3)
        
        # グラフを保存
        plot_file = os.path.join(MODEL_SAVE_DIR, "eval_loss_curve.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"グラフを保存しました: {plot_file}")
        
        plt.show()
    
    print("\neval_loss計算終了")
    print('Done. ', datetime.datetime.now(datetime.timezone(datetime.timedelta(hours = 9), 'JST')))

if __name__ == "__main__":
    main()

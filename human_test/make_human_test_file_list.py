#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
画像ファイルランダム選択プログラム

このプログラムは以下の機能を提供します：
1. 指定したディレクトリ以下の任意の階層で、画像ファイルの直上のサブディレクトリ名が指定したものと一致する画像ファイル(.png)を検索し、絶対パスのリストを作成する
2. 上記の画像ファイルから、指定した数のファイルをランダムに選択してリストを作成し、csvファイルに出力する

使用方法:
    python make_human_test_file_list.py

設定:
    スクリプト内の以下の変数を変更してください：
    - TARGET_DIRECTORY: 検索対象のディレクトリパス
    - SUBDIRECTORY_LABEL_NAME: 画像ファイルの直上のサブディレクトリ名（デフォルト: "AMI"）
    - NUM_FILES_TO_SELECT: 選択するファイル数
    - OUTPUT_CSV: 出力するCSVファイル名
    - RANDOM_SEED: 乱数のシード値（再現性のため、デフォルト: 42）
"""

import os
import random
import csv
import glob
import argparse
from pathlib import Path
from typing import List, Optional


def find_image_files(directory_path: str, subdirectory_name: str = "AMI") -> List[str]:
    """
    指定したディレクトリ以下の任意の階層で、画像ファイルの直上のサブディレクトリ名が
    指定したものと一致する画像ファイル(.png)を検索し、絶対パスのリストを作成する
    
    Args:
        directory_path (str): 検索対象のディレクトリパス
        subdirectory_name (str): 画像ファイルの直上のサブディレクトリ名（デフォルト: "AMI"）
        
    Returns:
        List[str]: 画像ファイルの絶対パスのリスト
    """
    image_files = []
    
    # ディレクトリが存在するかチェック
    if not os.path.exists(directory_path):
        print(f"エラー: ディレクトリ '{directory_path}' が存在しません。")
        return image_files
    
    # 再帰的にすべての.pngファイルを検索
    pattern = os.path.join(directory_path, "**", "*.png")
    all_png_files = glob.glob(pattern, recursive=True)
    
    # 画像ファイルの直上のサブディレクトリ名が指定したものと一致するものをフィルタリング
    for file_path in all_png_files:
        # ファイルの親ディレクトリ（直上のサブディレクトリ）名を取得
        parent_dir_name = os.path.basename(os.path.dirname(file_path))
        
        # 直上のサブディレクトリ名が指定したものと一致するかチェック
        if parent_dir_name == subdirectory_name:
            image_files.append(file_path)
    
    # 絶対パスに変換
    image_files = [os.path.abspath(f) for f in image_files]
    
    print(f"検索対象ディレクトリ: {directory_path}")
    print(f"検索対象サブディレクトリ名（直上）: {subdirectory_name}")
    print(f"見つかった画像ファイル数: {len(image_files)}")
    
    return image_files


def random_select_files(image_files: List[str], num_files: int, seed: int = 42) -> List[str]:
    """
    画像ファイルのリストから指定した数のファイルをランダムに選択する
    
    Args:
        image_files (List[str]): 画像ファイルのパスのリスト
        num_files (int): 選択するファイル数
        seed (int): 乱数のシード値（デフォルト: 42）
        
    Returns:
        List[str]: ランダムに選択されたファイルのパスのリスト
    """
    if num_files > len(image_files):
        print(f"警告: 要求されたファイル数({num_files})が利用可能なファイル数({len(image_files)})を超えています。")
        print(f"利用可能なすべてのファイル({len(image_files)}個)を返します。")
        return image_files
    
    # シードを設定して再現性を確保
    random.seed(seed)
    print(f"乱数シードを設定しました: {seed}")
    
    # ランダムに選択
    selected_files = random.sample(image_files, num_files)
    
    print(f"ランダムに選択されたファイル数: {len(selected_files)}")
    
    return selected_files


def save_to_csv(selected_files: List[str], output_csv_path: str) -> bool:
    """
    選択されたファイルのリストをCSVファイルに出力する
    
    Args:
        selected_files (List[str]): 選択されたファイルのパスのリスト
        output_csv_path (str): 出力するCSVファイルのパス
        
    Returns:
        bool: 出力が成功したかどうか
    """
    try:
        with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            
            # ヘッダー行を追加
            writer.writerow(['index', 'file_path', 'file_name', 'directory'])
            
            # ファイル情報を書き込み
            for i, file_path in enumerate(selected_files, 1):
                file_name = os.path.basename(file_path)
                directory = os.path.dirname(file_path)
                writer.writerow([i, file_path, file_name, directory])
        
        print(f"CSVファイルが正常に出力されました: {output_csv_path}")
        return True
        
    except Exception as e:
        print(f"CSVファイルの出力中にエラーが発生しました: {e}")
        return False


def main():
    """
    メイン処理
    """
    # 設定
    TARGET_DIRECTORY = "/path/to/data/human_test/source_dir/"  # 検索対象ディレクトリを指定
    NUM_FILES_TO_SELECT = 50 #100  # 選択するファイル数
    OUTPUT_CSV = "/path/to/outputs/human_test/selected_images_ami_50.csv"  # 出力CSVファイル名
    SUBDIRECTORY_LABEL_NAME = "AMI"
    RANDOM_SEED = 42  # 乱数のシード値（再現性のため）

    print("=== 画像ファイルランダム選択プログラム ===")
    print()
    
    # 1. 画像ファイルの直上のサブディレクトリ名が指定したものと一致する画像ファイルを検索
    print("1. 画像ファイルの直上のサブディレクトリ名が指定したものと一致する画像ファイルを検索中...")
    image_files = find_image_files(TARGET_DIRECTORY, SUBDIRECTORY_LABEL_NAME)
    
    if not image_files:
        print("画像ファイルが見つかりませんでした。")
        return
    
    print()
    
    # 2. ランダムにファイルを選択
    print("2. ランダムにファイルを選択中...")
    selected_files = random_select_files(image_files, NUM_FILES_TO_SELECT, RANDOM_SEED)
    
    print()
    
    # 3. CSVファイルに出力
    print("3. CSVファイルに出力中...")
    if save_to_csv(selected_files, OUTPUT_CSV):
        print()
        print("=== 処理完了 ===")
        print(f"選択されたファイルの詳細は '{OUTPUT_CSV}' を確認してください。")
        
        # 選択されたファイルの一部を表示
        print("\n選択されたファイルの例:")
        for i, file_path in enumerate(selected_files[:5], 1):
            print(f"{i}. {os.path.basename(file_path)}")
        if len(selected_files) > 5:
            print(f"... 他 {len(selected_files) - 5} 個のファイル")


def parse_arguments():
    """
    コマンドライン引数を解析する
    """
    parser = argparse.ArgumentParser(
        description="画像ファイルの直上のサブディレクトリ名が指定したものと一致する画像ファイルを検索し、ランダムに選択してCSVファイルに出力するプログラム"
    )
    parser.add_argument(
        "-d", "--directory",
        type=str,
        help="検索対象のディレクトリパス"
    )
    parser.add_argument(
        "-s", "--subdirectory",
        type=str,
        default="AMI",
        help="画像ファイルの直上のサブディレクトリ名 (デフォルト: AMI)"
    )
    parser.add_argument(
        "-n", "--num-files",
        type=int,
        default=100,
        help="選択するファイル数 (デフォルト: 100)"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="selected_images.csv",
        help="出力するCSVファイル名 (デフォルト: selected_images.csv)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="乱数のシード値 (デフォルト: 42)"
    )
    
    return parser.parse_args()


def run_with_arguments():
    """
    コマンドライン引数を使用してプログラムを実行する
    """
    args = parse_arguments()
    
    # 設定
    target_directory = args.directory or "/path/to/your/image/directory"
    subdirectory_name = args.subdirectory
    num_files_to_select = args.num_files
    output_csv = args.output
    random_seed = args.seed
    
    print("=== 画像ファイルランダム選択プログラム ===")
    print(f"検索対象ディレクトリ: {target_directory}")
    print(f"検索対象サブディレクトリ名（直上）: {subdirectory_name}")
    print(f"選択するファイル数: {num_files_to_select}")
    print(f"出力CSVファイル: {output_csv}")
    print(f"乱数シード値: {random_seed}")
    print()
    
    # 1. 画像ファイルの直上のサブディレクトリ名が指定したものと一致する画像ファイルを検索
    print("1. 画像ファイルの直上のサブディレクトリ名が指定したものと一致する画像ファイルを検索中...")
    image_files = find_image_files(target_directory, subdirectory_name)
    
    if not image_files:
        print("画像ファイルが見つかりませんでした。")
        return
    
    print()
    
    # 2. ランダムにファイルを選択
    print("2. ランダムにファイルを選択中...")
    selected_files = random_select_files(image_files, num_files_to_select, random_seed)
    
    print()
    
    # 3. CSVファイルに出力
    print("3. CSVファイルに出力中...")
    if save_to_csv(selected_files, output_csv):
        print()
        print("=== 処理完了 ===")
        print(f"選択されたファイルの詳細は '{output_csv}' を確認してください。")


if __name__ == "__main__":
    # コマンドライン引数がある場合はそれを使用、ない場合はデフォルト設定を使用
    import sys
    if len(sys.argv) > 1:
        run_with_arguments()
    else:
        print("コマンドライン引数が指定されていないため、デフォルト設定を使用します。")
        print("設定を変更するには、スクリプト内の変数を編集するか、コマンドライン引数を使用してください。")
        print("使用方法: python make_human_test_file_list.py -d /path/to/directory -s ami -n 50 -o output.csv --seed 42")
        print()
        main()

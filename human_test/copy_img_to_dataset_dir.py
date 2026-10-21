# version 0.0.1 20250819
# version 1.0.0 20250819 うごく。

import os
import shutil
from pathlib import Path
import argparse

def copy_images_with_directory_prefix(image_list_path, output_dir):
    """
    画像ファイルのリストを読み込み、指定したディレクトリに画像をコピーして保存し、
    2つ上のディレクトリ名までファイル名の頭に"_"をつけて保存する
    
    Args:
        image_list_path (str): 画像ファイルのパスが記載されたテキストファイルのパス
        output_dir (str): 画像を保存する出力ディレクトリのパス
    """
    # 出力ディレクトリが存在しない場合は作成
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 画像ファイルのリストを読み込み
    with open(image_list_path, 'r', encoding='utf-8') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    copied_count = 0
    error_count = 0
    
    for image_path in image_paths:
        try:
            # 画像ファイルのパスをPathオブジェクトに変換
            source_path = Path(image_path)
            
            # 画像ファイルが存在するかチェック
            if not source_path.exists():
                print(f"警告: ファイルが存在しません: {image_path}")
                error_count += 1
                continue
            
            # 2つ上のディレクトリ名を取得
            parent_dir = source_path.parent
            grandparent_dir = parent_dir.parent
            
            # ディレクトリ名を取得（親ディレクトリと祖父ディレクトリの名前）
            parent_name = parent_dir.name
            grandparent_name = grandparent_dir.name
            
            # 元のファイル名を取得
            original_filename = source_path.name
            
            # 新しいファイル名を作成（祖父ディレクトリ名_親ディレクトリ名_元のファイル名）
            new_filename = f"{grandparent_name}_{parent_name}_{original_filename}"
            
            # 出力パスを作成
            output_file_path = output_path / new_filename
            
            # ファイルをコピー
            shutil.copy2(source_path, output_file_path)
            
            print(f"コピー完了: {new_filename}")
            copied_count += 1
            
        except Exception as e:
            print(f"エラー: {image_path} の処理中にエラーが発生しました: {e}")
            error_count += 1
    
    print(f"\n処理完了:")
    print(f"  成功: {copied_count} ファイル")
    print(f"  エラー: {error_count} ファイル")

def main():
    parser = argparse.ArgumentParser(description='画像ファイルをディレクトリ名付きでコピーする')
    parser.add_argument('image_list', help='画像ファイルのパスが記載されたテキストファイルのパス')
    parser.add_argument('output_dir', help='画像を保存する出力ディレクトリのパス')
    
    args = parser.parse_args()
    
    # 画像リストファイルが存在するかチェック
    if not os.path.exists(args.image_list):
        print(f"エラー: 画像リストファイルが存在しません: {args.image_list}")
        return
    
    copy_images_with_directory_prefix(args.image_list, args.output_dir)

if __name__ == "__main__":
    main()
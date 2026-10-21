#!/usr/bin/env python3
# coding: utf-8

# tako, AMIのECG診断の、人間のテスト。
# version 0.0.1 20250522
# version 0.0.2 20250718 minimal version
# version 0.0.3 20250719 randomに新しく50のdatasetで、無作為のテストを追加。
# version 0.0.4 20251228 test mode と start-phase オプションを追加。
# version 0.0.5 20251228 append_results 関数を修正。csvファイルの列の入力が漏れないよう、修正した。
# version 0.0.6 20251228 index_1 から index_49 までの列を追加, 問題を無作為の順番に変更。フェーズ1 - 3は同じ順番。
# version 0.0.7 20251231 画像を再表示できるように修正。画像をEnter, Spaceでもけせるように修正。phase 4を3 modeに変更した。
# version 0.0.8 20251231 英語表記にぬけがないように修正。画像サイズを修正。instructionを修正。
# version 1.0.0 20260101 1st 完成版
# version 1.0.1 20260101 seedにdefaultをいれており、phase 4のmodeが湖底になるため、seedを指定しない場合は現在時刻を使用するよう、修正。
# version 1.0.2 20260101 index_ordered_columnsのresponset timeがすべて保存されていないのを修正。重複しているものを修正。
#                        ファイルの保存をアトミックにするよう、修正。
# version 1.0.3 20260101 SensitivityとSpecificityを計算して、保存するよう、修正。
# version 1.0.4 20260101 テスト経験の有無を尋ねて記録するよう、修正。
# version 1.0.5 20260101 instructionを表示するよう、修正。
# version 1.0.6 20260103 indexがすくないと表示されるバグを修正。画像が壊れている、ないとおちるところを修正。index_1/index_2 の代わりに question_id/question_no/qid/case_id を使用するよう、修正。
# version 1.0.7 20260105 instructionが画像ウインドウを答えの入力で閉じれるように修正したものを反映していなかったのを反映した。
# version 1.0.8 20260108 instructionを修正。
# version 1.0.9 20260618 フェーズ4に層別ブロック無作為化を追加（デフォルト）。単純無作為は --randomization-mode simple。
#
# 使用例 / Usage Examples:
# ========================
# 1. 基本的な使用（全フェーズ実行、デフォルト設定）
#    Basic usage (all phases, default settings):
#    python tako_ami_human_test_with_random.py
#
# 2. チートシート画像を指定して実行
#    Run with cheat sheet image:
#    python tako_ami_human_test_with_random.py --cheat-sheet /path/to/cheat_sheet.png
#
# 3. テストモード（各フェーズの先頭5問のみ）
#    Test mode (only first 5 questions per phase):
#    python tako_ami_human_test_with_random.py --test
#
# 4. 特定のフェーズから開始（デバッグ用）
#    Start from specific phase (for debugging):
#    python tako_ami_human_test_with_random.py --start-phase 3
#
# 5. フェーズ4のグループ選択の重みを指定（group0:group1:group2 = 2:1:1）
#    Specify Phase 4 group selection weights (group0:group1:group2 = 2:1:1):
#    python tako_ami_human_test_with_random.py --phase4-weights 2 1 1
#
# 6. 複数のオプションを組み合わせて使用
#    Combine multiple options:
#    python tako_ami_human_test_with_random.py --test --cheat-sheet /path/to/cheat_sheet.png --seed 123
#
# コマンドライン引数 / Command-line Arguments:
# ============================================
# --test              : テストモード（各フェーズの先頭5問のみ実行）
#                      Test mode (only first 5 questions per phase)
# --seed SEED         : ランダムシード値（指定しない場合は現在時刻を使用）
#                      Random seed value (uses current time if not specified)
# --start-phase PHASE : 開始フェーズを指定（1-4、デフォルト: 1）
#                      Start phase (1-4, default: 1)
# --cheat-sheet PATH  : チートシート画像のパス（指定した場合のみフェーズ2、3、4で表示）
#                      Path to cheat sheet image (displayed in phases 2, 3, 4 if specified)
# --phase4-weights W0 W1 W2 : フェーズ4のグループ選択の重み（group0, group1, group2、デフォルト: 1 1 1）
#                              Phase 4 group selection weights (group0, group1, group2, default: 1 1 1)
# --randomization-mode MODE : フェーズ4の無作為化方式 (stratified_block|simple、デフォルト: stratified_block)
#                              Phase 4 randomization mode (stratified_block|simple, default: stratified_block)
# --randomization-table PATH : 層別ブロック無作為化の割付表CSV（デフォルト: stratified_block_random/randomization_table.csv）
# --randomization-state PATH : 層別ブロック無作為化の状態CSV（デフォルト: stratified_block_random/randomization_state.csv）
# --max-enrollment N        : 層別ブロック無作為化の最大割付人数（デフォルト: 100）
# --career-threshold N      : 層別化の職歴年数しきい値（>=N と <N、デフォルト: 10）

import os
import csv
import random
import fcntl
import datetime
import time
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import sys
import termios
import tty

RESULTS_SEQ_CSV = "/path/to/outputs/human_test/result/results_seq.csv"   # 回答を追記するファイル
RESULTS_RAND_CSV = "/path/to/outputs/human_test/result/results_rand.csv"   # 回答を追記するファイル
QUESTIONS_SEQ_CSV = "/path/to/data/human_test/question/ensemble_results_questions.csv"  # 事前に用意する: image_path,true_label,vit_grad_cam_path,convnextv2_grad_cam_path,ens_prob_AMI の列を持つ
QUESTIONS_RAND_CSV = "/path/to/data/human_test/question/ensemble_results_questions_2.csv"  # 事前に用意する: image_path,true_label,vit_grad_cam_path,convnextv2_grad_cam_path,ens_prob_AMI,weak_ensemble_bar_graph_path の列を持つ
STRATIFIED_BLOCK_RANDOM_DIR = "/path/to/outputs/human_test/stratified_block_random"
DEFAULT_RANDOMIZATION_TABLE = os.path.join(STRATIFIED_BLOCK_RANDOM_DIR, "randomization_table.csv")
DEFAULT_RANDOMIZATION_STATE = os.path.join(STRATIFIED_BLOCK_RANDOM_DIR, "randomization_state.csv")
RANDOMIZATION_STRATA = ["A", "B", "C", "D"]
STRATUM_LABELS = {
    "A": "cardiologist, career>=threshold",
    "B": "cardiologist, career<threshold",
    "C": "non-cardiologist, career>=threshold",
    "D": "non-cardiologist, career<threshold",
}

def determine_stratum(profession, years, career_threshold=10):
    """職種・職歴から層 (A-D) を判定する。"""
    is_cardiologist = profession == "cardiologist"
    years_int = int(years)
    senior = years_int >= career_threshold
    if is_cardiologist and senior:
        return "A"
    if is_cardiologist and not senior:
        return "B"
    if not is_cardiologist and senior:
        return "C"
    return "D"

def load_randomization_table(path):
    """割付表CSVを読み込み、層ごとの slot_id -> 割付情報 の辞書を返す。"""
    table_by_stratum = {stratum: {} for stratum in RANDOMIZATION_STRATA}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"stratum", "slot_id", "block_id", "position_in_block", "group"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise RuntimeError(
                f"{path} に必須列 {required} がありません / Required columns {required} are missing in {path}"
            )
        for row in reader:
            stratum = row["stratum"]
            if stratum not in table_by_stratum:
                raise RuntimeError(
                    f"未知の層 '{stratum}' が割付表にあります / Unknown stratum '{stratum}' in randomization table"
                )
            slot_id = int(row["slot_id"])
            table_by_stratum[stratum][slot_id] = {
                "group": int(row["group"]),
                "block_id": int(row["block_id"]),
                "position_in_block": int(row["position_in_block"]),
            }
    for stratum in RANDOMIZATION_STRATA:
        if not table_by_stratum[stratum]:
            raise RuntimeError(
                f"割付表に層 {stratum} の行がありません / No rows for stratum {stratum} in randomization table"
            )
    return table_by_stratum

def load_randomization_state(path):
    """状態CSVを読み込む。存在しない場合は初期状態を返す。"""
    state = {"total_assigned": 0, **{stratum: 1 for stratum in RANDOMIZATION_STRATA}}
    if not os.path.exists(path):
        return state
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row["key"]
            value = int(row["value"])
            if key == "total_assigned":
                state["total_assigned"] = value
            elif key in RANDOMIZATION_STRATA:
                state[key] = value
    return state

def save_randomization_state(path, state):
    """状態CSVをアトミックに保存する。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temp_path = path + ".tmp"
    with open(temp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["key", "value"])
        writer.writerow(["total_assigned", state["total_assigned"]])
        for stratum in RANDOMIZATION_STRATA:
            writer.writerow([stratum, state[stratum]])
    os.replace(temp_path, path)

def assign_stratified_block_group(table_by_stratum, state_path, stratum, max_enrollment):
    """
    層別ブロック無作為化で group を割り付ける。
    状態ファイルを排他ロックして更新する。
    """
    os.makedirs(os.path.dirname(state_path), exist_ok=True)
    lock_path = state_path + ".lock"
    with open(lock_path, "w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            state = load_randomization_state(state_path)
            if state["total_assigned"] >= max_enrollment:
                raise RuntimeError(
                    f"最大割付人数 ({max_enrollment}) に達しました。これ以上割り付けできません。"
                    f" / Maximum enrollment ({max_enrollment}) reached. Cannot assign more participants."
                )
            slot_id = state[stratum]
            stratum_table = table_by_stratum[stratum]
            if slot_id not in stratum_table:
                raise RuntimeError(
                    f"層 {stratum} の割付枠が不足しています (slot_id={slot_id})。"
                    f" / Insufficient randomization slots for stratum {stratum} (slot_id={slot_id})."
                )
            entry = stratum_table[slot_id]
            group = entry["group"]
            state[stratum] = slot_id + 1
            state["total_assigned"] += 1
            save_randomization_state(state_path, state)
            return group, {
                "randomization_mode": "stratified_block",
                "stratum": stratum,
                "stratum_label": STRATUM_LABELS[stratum],
                "randomization_slot_id": slot_id,
                "randomization_block_id": entry["block_id"],
            }
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

def get_key():
    """キー入力を読み取る（Enter, Space, 矢印キー、Backspaceなどを検出）"""
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
        # 矢印キーや特殊キーはエスケープシーケンスで始まる
        if ch == '\x1b':  # ESC
            ch = sys.stdin.read(1)
            if ch == '[':  # CSI (Control Sequence Introducer)
                ch = sys.stdin.read(1)
                if ch == 'A':  # 上矢印
                    return 'up'
                elif ch == 'B':  # 下矢印
                    return 'down'
                elif ch == 'C':  # 右矢印
                    return 'right'
                elif ch == 'D':  # 左矢印
                    return 'left'
        elif ch == '\r' or ch == '\n':  # Enter
            return 'enter'
        elif ch == ' ':  # Space
            return 'space'
        elif ch == '\x7f' or ch == '\b':  # Backspace
            return 'backspace'
        elif ch == '\x03':  # Ctrl+C
            return 'ctrl_c'
        else:
            return ch.lower()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def show_instructions():
    """言語を選択し、instructionを表示して同意を得る"""
    print("\n" + "="*70)
    print("言語を選択してください / Please select language")
    print("="*70)
    print("  1: 日本語 / Japanese")
    print("  2: English")
    while True:
        lang_choice = input("番号/number (1/2): ").strip()
        if lang_choice in ("1", "2"):
            break
        print("1 または 2 を入力してください / Please input 1 or 2")
    
    is_japanese = (lang_choice == "1")
    
    # ページごとにinstructionを表示
    if is_japanese:
        pages = [
            """日本語版 / Japanese Edition
======================================================================

ディスプレイの解像度が2560 x 1600ピクセル以上を推奨します。

これからクイズを行います。  
表示される心電図は、発症から14日以内の  
- たこつぼ症候群  
- ST上昇型急性前壁心筋梗塞(再潅流療法後を含む)  
です。  

表示された心電図が「たこつぼ症候群」のものか、「急性心筋梗塞」のものかを診断し、回答してください。

まず、「名前」, 「職種」, 「職歴」, およびこのクイズの実行が「初回かどうか」を入力します。  
この後、クイズがはじまります。

1. クイズがはじまると、心電図（フェーズ3以降ではAIモデルの予測も）を表示したウィンドウが表示されます。
    
2. 心電図が、「たこつぼ症候群」か「急性心筋梗塞」かを診断します。
    
3. 回答方法は2通りあります：
    
   【方法1】画像ウィンドウ内で直接回答する（推奨）
   - 画像ウィンドウがアクティブな状態で、「たこつぼ症候群」であれば 「t」, 「T」, 「0」を押し、急性心筋梗塞であれば 「a」, 「A」, 「1」を押します。
   - これにより、回答が入力され、自動的にウィンドウが閉じて次の問題に進みます。
   
   【方法2】画像ウィンドウを閉じてからコマンドラインで回答する
   - まず、この画像ウィンドウを「閉じます」。閉じる方法は、キーボードの「Space」キー、「Enter（Return）」キー、「Ctrl + W」、またはマウスで画像ウィンドウの「閉じるボタン」をクリックする、のいずれかです。
   - その後、コマンドラインに、「たこつぼ症候群」であれば 「t」, 「T」, 「0」を入力し、急性心筋梗塞であれば 「a」, 「A」, 「1」を入力して、「Enter（Return）」キーを押します。
    
4. 次の問題に移り、1. にもどります（次の画像が表示されます）。""",
            """クイズは4つのフェーズに分かれており、各フェーズで50枚の心電図を評価します。フェーズ1, 2, 3は同じ50枚、フェーズ4は異なる50枚です。

- フェーズ1: 心電図が表示されます。どちらの疾患かを診断し、回答してください。  
    50問が終了したらフェーズ2に移り、同様にフェーズ3、フェーズ4と進み、テストは終了します。
    
- フェーズ2: フェーズ1と同様ですが、チートシートを参照して回答しても構いません。
    
- フェーズ3: フェーズ2と同様ですが、心電図と同時にAIモデルの予測も表示されます。AIモデルは ViT、ConvNeXtV2-Base、MLP-Mixer の3モデルの予測をアンサンブル(ロジット平均)して出力します。このモデルの性能はたこつぼ症候群とAMIの1:1.8のデータセットにおいてROC-AUC 0.97, PR-AUC 0.98です(AMIを陽性とした場合)。さらに、2つのモデルの Grad-CAM（どこに注目しているかをしめす画像）も表示されます。診断して回答してください。
    
- フェーズ4: フェーズ1〜3とは異なる新しい心電図50枚を診断し、回答してください。画像ウインドウに表示される情報に加えて、チートシートも利用可能です。""",
            """次の内容に同意される場合は、「y / Y」 を入力するか、「Enter（Return）キー」を押してクイズに進みます。

DISCLAIMER:  
このクイズ内の心電図に関する情報は機密情報であり、FOR YOUR EYES ONLY, 関係者限り（閲覧者限り）とします。許可なく第三者に共有・転載は禁止・転載しないでください。スクリーンショット、写真撮影、録画、画面共有、SNS等への投稿を禁止します。
これらに反する行為が確認された場合は、後悔することになることを了承します。"""
        ]
    else:
        pages = [
            """English Edition
======================================================================

A screen resolution of 2560 x 1600 pixels or higher is recommended.

We will now conduct a quiz.  
The displayed electrocardiograms (ECGs) represent patients in the first 14 days following onset with:  
- Takotsubo syndrome  
- Acute anterior ST-segment elevation myocardial infarction (including those who received coronary reperfusion therapy)  

In this quiz, you will diagnose whether the displayed ECG shows "takotsubo syndrome" or "acute myocardial infarction (AMI)", and then submit your answer.

First, you will enter your "name", "job title", "work history", and whether this is your "first time" taking this quiz.  
After that, the quiz will begin.

1. When the quiz starts, a window will appear displaying the ECG (and from "Phase 3" onward, the AI models' predictions as well).
    
2. Diagnose whether the ECG is from takotsubo syndrome or acute myocardial infarction.
    
3. There are two ways to answer:
    
   [Method 1] Answer directly in the image window (recommended)
   - While the image window is active, press "t", "T", or "0" for takotsubo syndrome, or "a", "A", or "1" for acute myocardial infarction.
   - This will record your answer and automatically close the window to proceed to the next question.
   
   [Method 2] Close the window and answer in the command line
   - First, "close this window". You can do so by any of the following: pressing the "Space" key, pressing the "Enter (Return)" key, pressing "Ctrl+W", or clicking the window "close" button.
   - Then, in the command line, enter "t", "T", or "0" for takotsubo syndrome, or "a", "A", or "1" for acute myocardial infarction, and then press Enter (Return).
    
4. You will move on to the next question and return to step 1. The next image will be displayed.""",
            """The quiz is divided into 4 phases, and in each phase you will evaluate 50 ECGs. Phases 1, 2, and 3 use the same set of 50 ECGs, while Phase 4 uses a different set of 50 ECGs.

Phase 1: An ECG is displayed. Please diagnose which condition it represents and submit your answer.  
After you finish all 50 questions, you will proceed to Phase 2, and then similarly to Phase 3 and Phase 4 to complete the test.

Phase 2: Same as Phase 1, but you may refer to a cheat sheet while answering.

Phase 3: Same as Phase 2, but the AI model's predictions will be displayed along with the ECG. The AI output is an ensemble (logit averaging) of three models: ViT, ConvNeXtV2-Base, and MLP-Mixer. The performance of this model on a 1:1.8 dataset of takotsubo syndrome and AMI is ROC-AUC 0.97 and PR-AUC 0.98 (with AMI as positive). In addition, Grad-CAM visualizations (images showing which parts of the ECG the model is focusing on) from two of the models will also be displayed. Please diagnose and submit your answer.

Phase 4: Evaluate and answer 50 ECGs that are different from those in Phases 1–3. In addition to the on-screen display, you may also use the cheat sheet.""",
            """If you agree to the following, enter "y", "Y", or press "Enter (Return)" to proceed to the quiz.

DISCLAIMER:  
Information related to the ECGs in this quiz is confidential and FOR YOUR EYES ONLY. Do not share, copy, reproduce, or redistribute it to any third party without permission. Screenshots, photography, video recording, screen sharing, and posting on social media are strictly prohibited."""
        ]
    
    # 各ページを表示（キー入力で前後移動可能）
    current_page = 0
    while True:
        # 現在のページを表示
        print("\n" + "="*70)
        print(pages[current_page])
        print("="*70)
        
        # 最後のページで同意確認
        if current_page == len(pages) - 1:
            if is_japanese:
                print("\n同意する場合は \"y\" / \"Y\" を入力するか、Enter（Return）キーを押してください / If you agree, enter \"y\" / \"Y\" or press Enter (Return)")
                print("前のページに戻る: 上矢印キーまたはBackspaceキー / Go back: Up arrow or Backspace")
            else:
                print("\nIf you agree, enter \"y\", \"Y\", or press Enter")
                print("Go back: Up arrow or Backspace")
            
            agreed = False
            while True:
                key = get_key()
                if key == 'enter' or key == 'y':
                    # 同意確認
                    if key == 'y':
                        print("y")
                    agreement = 'y' if key == 'y' else ''
                    if agreement.lower() not in ("y", ""):
                        print("同意が必要です。プログラムを終了します。 / Agreement is required. Exiting the program.")
                        exit(1)
                    # 同意が確認された
                    agreed = True
                    break
                elif key == 'up' or key == 'backspace':
                    # 前のページに戻る
                    if current_page > 0:
                        current_page -= 1
                        break
                elif key == 'ctrl_c':
                    print("\nプログラムを終了します。 / Exiting the program.")
                    exit(1)
            # 同意が確認されたら、外側のループも終了
            if agreed:
                break
        else:
            # 最後のページ以外は、次のページに進むか前のページに戻る
            if is_japanese:
                print(f"\nページ {current_page + 1}/{len(pages)}")
                print("次のページ: Enter、Space、または下矢印キー / Next page: Enter, Space, or Down arrow")
                if current_page > 0:
                    print("前のページ: 上矢印キーまたはBackspaceキー / Previous page: Up arrow or Backspace")
            else:
                print(f"\nPage {current_page + 1}/{len(pages)}")
                print("Next page: Enter, Space, or Down arrow")
                if current_page > 0:
                    print("Previous page: Up arrow or Backspace")
            
            while True:
                key = get_key()
                if key == 'enter' or key == 'space' or key == 'down':
                    # 次のページに進む
                    current_page += 1
                    break
                elif key == 'up' or key == 'backspace':
                    # 前のページに戻る
                    if current_page > 0:
                        current_page -= 1
                        break
                elif key == 'ctrl_c':
                    print("\nプログラムを終了します。 / Exiting the program.")
                    exit(1)
    
    print("\nクイズを開始します / Starting the quiz...\n")

def prompt_user_info():
    name = input("名前を入力してください/Please enter your name: ").strip()
    prof_map_display = {"1":"循環器内科医/cardiologist", "2":"循環器内科医以外の医師/non-cardiologist", "3":"臨床検査技師/medical technologist", "4":"臨床検査技師以外のコメディカル/comedical", "5":"その他/other"}
    prof_map_save = {"1":"cardiologist", "2":"non-cardiologist", "3":"medical_technologist", "4":"comedical", "5":"other"}
    print("職種の番号を入力してください/Please select your profession by number:")
    for k,v in prof_map_display.items():
        print(f"  {k}: {v}")
    while True:
        c = input("番号/number (1/2/3/4/5): ").strip()
        if c in prof_map_save: break
        print("1, 2, 3, 4, または 5 を入力してください/Please input 1, 2, 3, 4, or 5")
    profession = prof_map_save[c]  # CSV保存用に英語のみを使用
    while True:
        years = input("職歴の長さ(年)を自然数で入力してください/Please enter your years of medical career (natural number): ").strip()
        if years.isdigit():  # 数字のみかチェック
            break
        print("数字(自然数)のみを入力してください/Please enter your years of medical career (natural number): ")
    # テスト経験の有無を尋ねる
    print("このテストは初めてですか？/Is this your first time taking this test?")
    print("  0: 初めて/First time")
    print("  1: 以前に行ったことがある/Have taken before")
    while True:
        first_time = input("番号/number (0/1): ").strip()
        if first_time in ("0", "1"):
            break
        print("0 または 1 を入力してください/Please input 0 or 1")
    is_first_time = int(first_time)  # 0: 初めて, 1: 以前に行ったことがある
    return name, profession, years, is_first_time

def load_questions(path):
    df = pd.read_csv(path)
    required = {"image_path", "true_label", "vit_grad_cam_path", "convnextv2_grad_cam_path", "ens_prob_AMI"}
    if not required.issubset(df.columns):
        raise RuntimeError(f"{path} に必須列 {required} がありません / Required columns {required} are missing in {path}")

    # 推奨: question_id / question_no / qid / case_id を優先して読む（後方互換で index_1/index_2 も許容）
    candidate_cols = ["question_id", "question_no", "qid", "case_id", "index_1", "index_2"]
    index_col = next((c for c in candidate_cols if c in df.columns), None)
    if index_col is None:
        raise RuntimeError(
            f"{path} にID列がありません / Missing ID column. "
            f"Expected one of: {candidate_cols}"
        )

    records = df.to_dict(orient="records")
    for record in records:
        record["index"] = int(record[index_col])      # 固定ID
        record["label"] = int(record["true_label"])
        record["pred_prob"] = float(record["ens_prob_AMI"])

    random.shuffle(records)
    return records

def run_phase(recs, phase, cheat_sheet_path=None):
    """
    phase=1: 画像のみ
    phase=2: 画像のみ（チートシート可）
    phase=3: 画像+attention map+予測確率（チートシート可）
    cheat_sheet_path: チートシート画像のパス（指定された場合のみ表示）
    """
    print(f"\n--- フェーズ {phase} 開始 ({len(recs)} 問) / Phase {phase} starts ({len(recs)} questions) ---")
    answers = []
    response_times = []

    cheat_sheet = None
    try:
        # チートシート画像を読み込む（フェーズ2、3で使用、パスが指定された場合のみ）
        if phase in (2, 3) and cheat_sheet_path:
            if os.path.exists(cheat_sheet_path):
                try:
                    cheat_sheet = Image.open(cheat_sheet_path)
                except Exception as e:
                    print(f"警告/Warning: チートシート画像が開けません: {cheat_sheet_path} ({e}) / Failed to open cheat sheet: {cheat_sheet_path} ({e})")
                    cheat_sheet = None
            else:
                print(f"警告/Warning: チートシート画像が見つかりません: {cheat_sheet_path} / Cheat sheet image not found: {cheat_sheet_path}")

        for i, r in enumerate(recs):
            img = None
            vit_att = None
            conv_att = None
            bar_graph = None
            fig = None  # 図を初期化

            try:
                # --- ここから「壊れた画像/パス不正ならスキップ」 ---
                try:
                    img = Image.open(r["image_path"])
                except Exception as e:
                    print(f"警告/Warning: 画像が開けないためスキップします [{i+1}/{len(recs)}]: {r.get('image_path')} ({e}) / Failed to open image, skipping: {r.get('image_path')} ({e})")
                    answers.append(None)
                    response_times.append(None)
                    continue

                if phase == 3:
                    try:
                        vit_att = Image.open(r["vit_grad_cam_path"])
                    except Exception as e:
                        print(f"警告/Warning: ViT Grad-CAMが開けないためスキップします [{i+1}/{len(recs)}]: {r.get('vit_grad_cam_path')} ({e}) / Failed to open ViT Grad-CAM, skipping: {r.get('vit_grad_cam_path')} ({e})")
                        answers.append(None)
                        response_times.append(None)
                        continue

                    try:
                        conv_att = Image.open(r["convnextv2_grad_cam_path"])
                    except Exception as e:
                        print(f"警告/Warning: ConvNeXtV2 Grad-CAMが開けないためスキップします [{i+1}/{len(recs)}]: {r.get('convnextv2_grad_cam_path')} ({e}) / Failed to open ConvNeXtV2 Grad-CAM, skipping: {r.get('convnextv2_grad_cam_path')} ({e})")
                        answers.append(None)
                        response_times.append(None)
                        continue

                    # weak_ensemble_bar_graph_pathが存在し、有効な値の場合は表示（開けなければ「棒グラフなし」で続行）
                    bar_graph_path = r.get("weak_ensemble_bar_graph_path")
                    if bar_graph_path and not pd.isna(bar_graph_path) and str(bar_graph_path).strip():
                        try:
                            bar_graph = Image.open(bar_graph_path)
                        except Exception as e:
                            print(f"警告/Warning: 棒グラフが開けないため棒グラフなしで続行 [{i+1}/{len(recs)}]: {bar_graph_path} ({e}) / Failed to open bar graph, continue without it: {bar_graph_path} ({e})")
                            bar_graph = None
                # --- ここまでスキップ処理 ---

                # 画像表示関数（再表示用）
                captured_answer = None  # 画像ウィンドウで入力された回答を保存

                def show_image():
                    nonlocal fig, captured_answer
                    captured_answer = None  # 再表示時はリセット

                    if phase == 3:
                        if bar_graph:
                            if cheat_sheet:
                                fig = plt.figure(figsize=(24, 16))
                                ax_img = plt.subplot2grid((4, 3), (0, 0), rowspan=4, fig=fig)
                                ax_img.imshow(img); ax_img.axis("off"); ax_img.set_title("Image", fontsize=18)
                                ax_bar = plt.subplot2grid((4, 3), (0, 1), colspan=2, fig=fig)
                                ax_bar.imshow(bar_graph); ax_bar.axis("off")
                                ax_bar.set_title(f"Weak Learners & Ensemble\nEns Prob: {r['pred_prob']:.2f}", fontsize=18)
                                ax_vit = plt.subplot2grid((4, 3), (1, 1), colspan=2, fig=fig)
                                ax_vit.imshow(vit_att); ax_vit.axis("off")
                                ax_vit.set_title("ViT Grad-CAM", fontsize=18)
                                ax_conv = plt.subplot2grid((4, 3), (2, 1), colspan=2, fig=fig)
                                ax_conv.imshow(conv_att); ax_conv.axis("off")
                                ax_conv.set_title("ConvNeXtV2 Grad-CAM", fontsize=18)
                                ax_cheat = plt.subplot2grid((4, 3), (3, 1), colspan=2, fig=fig)
                                ax_cheat.imshow(cheat_sheet); ax_cheat.axis("off")
                                ax_cheat.set_title("Cheat Sheet", fontsize=18)
                            else:
                                fig = plt.figure(figsize=(24, 14))
                                ax_img = plt.subplot2grid((3, 3), (0, 0), rowspan=3, fig=fig)
                                ax_img.imshow(img); ax_img.axis("off"); ax_img.set_title("Image", fontsize=18)
                                ax_bar = plt.subplot2grid((3, 3), (0, 1), colspan=2, fig=fig)
                                ax_bar.imshow(bar_graph); ax_bar.axis("off")
                                ax_bar.set_title(f"Weak Learners & Ensemble\nEns Prob: {r['pred_prob']:.2f}", fontsize=18)
                                ax_vit = plt.subplot2grid((3, 3), (1, 1), colspan=2, fig=fig)
                                ax_vit.imshow(vit_att); ax_vit.axis("off")
                                ax_vit.set_title("ViT Grad-CAM", fontsize=18)
                                ax_conv = plt.subplot2grid((3, 3), (2, 1), colspan=2, fig=fig)
                                ax_conv.imshow(conv_att); ax_conv.axis("off")
                                ax_conv.set_title("ConvNeXtV2 Grad-CAM", fontsize=18)
                        else:
                            if cheat_sheet:
                                fig = plt.figure(figsize=(24, 12))
                                ax_img = plt.subplot2grid((3, 3), (0, 0), rowspan=3, fig=fig)
                                ax_img.imshow(img); ax_img.axis("off"); ax_img.set_title("Image", fontsize=18)
                                ax_vit = plt.subplot2grid((3, 3), (0, 1), colspan=2, fig=fig)
                                ax_vit.imshow(vit_att); ax_vit.axis("off")
                                ax_vit.set_title("ViT Grad-CAM", fontsize=18)
                                ax_conv = plt.subplot2grid((3, 3), (1, 1), colspan=2, fig=fig)
                                ax_conv.imshow(conv_att); ax_conv.axis("off")
                                ax_conv.set_title(f"ConvNeXtV2 Grad-CAM\nEns Prob: {r['pred_prob']:.2f}", fontsize=18)
                                ax_cheat = plt.subplot2grid((3, 3), (2, 1), colspan=2, fig=fig)
                                ax_cheat.imshow(cheat_sheet); ax_cheat.axis("off")
                                ax_cheat.set_title("Cheat Sheet", fontsize=18)
                            else:
                                fig = plt.figure(figsize=(24, 10))
                                ax_img = plt.subplot2grid((2, 3), (0, 0), rowspan=2, fig=fig)
                                ax_img.imshow(img); ax_img.axis("off"); ax_img.set_title("Image", fontsize=18)
                                ax_vit = plt.subplot2grid((2, 3), (0, 1), colspan=2, fig=fig)
                                ax_vit.imshow(vit_att); ax_vit.axis("off")
                                ax_vit.set_title("ViT Grad-CAM", fontsize=18)
                                ax_conv = plt.subplot2grid((2, 3), (1, 1), colspan=2, fig=fig)
                                ax_conv.imshow(conv_att); ax_conv.axis("off")
                                ax_conv.set_title(f"ConvNeXtV2 Grad-CAM\nEns Prob: {r['pred_prob']:.2f}", fontsize=18)

                        plt.subplots_adjust(wspace=0.02, hspace=0.05)
                        plt.tight_layout(pad=0.5)

                    else:
                        if phase == 2 and cheat_sheet:
                            fig = plt.figure(figsize=(24, 12))
                            ax_img = plt.subplot(1, 2, 1)
                            ax_img.imshow(img); ax_img.axis("off"); ax_img.set_title("ECG Image", fontsize=12)
                            ax_cheat = plt.subplot(1, 2, 2)
                            ax_cheat.imshow(cheat_sheet); ax_cheat.axis("off"); ax_cheat.set_title("Cheat Sheet", fontsize=12)
                            plt.tight_layout(pad=2.0)
                        else:
                            fig = plt.figure(figsize=(24,12))
                            plt.imshow(img); plt.axis("off")

                    def on_key(event):
                        nonlocal captured_answer
                        if event.key in ('enter', 'return', ' '):
                            plt.close('all')
                        elif event.key.lower() in ('t', '0'):
                            captured_answer = '0'
                            print("t")
                            plt.close('all')
                        elif event.key.lower() in ('a', '1'):
                            captured_answer = '1'
                            print("a")
                            plt.close('all')

                    fig.canvas.mpl_connect('key_press_event', on_key)
                    plt.show(block=False)
                    while plt.get_fignums():
                        plt.pause(0.1)

                start_time = time.time()

                print(f"\n[{i+1}/{len(recs)}] 画像ウィンドウで回答キー（t/T/0 または a/A/1）を押すか、ウィンドウを閉じてコマンドラインで回答してください / Press answer key (t/T/0 or a/A/1) in image window, or close window to answer in command line")

                show_image()

                if captured_answer is None:
                    while True:
                        ans = input(f"[{i+1}/{len(recs)}] ラベルを入力 (takotsubo: 0 または t/T, AMI: 1 または a/A, 画像再表示: r/R/i/I) / Please input label (takotsubo: 0 or t/T, AMI: 1 or a/A, show image again: r/R/i/I): ").strip()
                        if ans.lower() in ("r", "i"):
                            print("画像を再表示します / Showing image again...")
                            show_image()
                            if captured_answer is not None:
                                ans = captured_answer
                                break
                            continue
                        elif ans.lower() in ("0", "1", "t", "a"):
                            break
                        else:
                            print("0, 1, t/T, a/A, r/R, または i/I を入力してください / Please input 0, 1, t/T, a/A, r/R, or i/I")
                else:
                    ans = captured_answer

                end_time = time.time()
                response_time = end_time - start_time

                if ans.lower() in ("t", "0"):
                    ans_value = 0
                else:
                    ans_value = 1

                answers.append(ans_value)
                response_times.append(response_time)

                print(f"回答時間/Response time: {response_time:.2f}秒/sec")

            finally:
                # 念のため matplotlib の figure も閉じる
                try:
                    plt.close('all')
                except Exception:
                    pass

                # ループ末尾で開いた画像を閉じる（ファイルディスクリプタ枯渇対策）
                try:
                    if img is not None:
                        img.close()
                    if vit_att is not None:
                        vit_att.close()
                    if conv_att is not None:
                        conv_att.close()
                    if bar_graph is not None:
                        bar_graph.close()
                except Exception:
                    pass

        # 正答率（回答できた分だけで計算）
        valid = [(a, r) for a, r in zip(answers, recs) if a in (0, 1)]
        if len(valid) == 0:
            acc = 0.0
        else:
            correct = [int(a == int(r["label"])) for a, r in valid]
            acc = sum(correct) / len(correct)

        # 平均回答時間（必要なら呼び出し側で使う）
        valid_times = [rt for rt in response_times if isinstance(rt, (int, float))]
        #avg_time = sum(valid_times) / len(valid_times) if valid_times else 0.0

        return answers, response_times, acc

    finally:
        # ループ外で開いたチートシートを閉じる
        if cheat_sheet is not None:
            try:
                cheat_sheet.close()
            except Exception:
                pass

def run_phase_4(recs, group, cheat_sheet_path=None):
    """
    phase=4: 無作為テスト
    group=0: チートシートのみ
    group=1: チートシート＋AI予測（Grad-CAMなし、weak_ensemble_bar_graph_pathのみ）
    group=2: チートシート＋AI予測（Grad-CAMあり）
    cheat_sheet_path: チートシート画像のパス（指定された場合のみ表示）
    """
    if group == 0:
        group_name = "チートシートのみ"
    elif group == 1:
        group_name = "チートシート + AI予測（Grad-CAMなし）"
    else:
        group_name = "チートシート + AI予測（Grad-CAMあり）"
    print(f"\n--- フェーズ 4 開始 ({len(recs)} 問) - {group_name} / Phase 4 starts ({len(recs)} questions) - {group_name} ---")

    answers = []
    response_times = []

    cheat_sheet = None
    try:
        if cheat_sheet_path:
            if os.path.exists(cheat_sheet_path):
                try:
                    cheat_sheet = Image.open(cheat_sheet_path)
                except Exception as e:
                    print(f"警告/Warning: チートシート画像が開けません: {cheat_sheet_path} ({e}) / Failed to open cheat sheet: {cheat_sheet_path} ({e})")
                    cheat_sheet = None
            else:
                print(f"警告/Warning: チートシート画像が見つかりません: {cheat_sheet_path} / Cheat sheet image not found: {cheat_sheet_path}")

        for i, r in enumerate(recs):
            img = None
            vit_att = None
            conv_att = None
            bar_graph = None
            fig = None

            try:
                # --- ここから「壊れた画像/パス不正ならスキップ」 ---
                try:
                    img = Image.open(r["image_path"])
                except Exception as e:
                    print(f"警告/Warning: 画像が開けないためスキップします [{i+1}/{len(recs)}]: {r.get('image_path')} ({e}) / Failed to open image, skipping: {r.get('image_path')} ({e})")
                    answers.append(None)
                    response_times.append(None)
                    continue

                if group == 1:
                    bar_graph_path = r.get("weak_ensemble_bar_graph_path")
                    if bar_graph_path and not pd.isna(bar_graph_path) and str(bar_graph_path).strip():
                        try:
                            bar_graph = Image.open(bar_graph_path)
                        except Exception as e:
                            print(f"警告/Warning: 棒グラフが開けないため棒グラフなしで続行 [{i+1}/{len(recs)}]: {bar_graph_path} ({e}) / Failed to open bar graph, continue without it: {bar_graph_path} ({e})")
                            bar_graph = None

                elif group == 2:
                    try:
                        vit_att = Image.open(r["vit_grad_cam_path"])
                    except Exception as e:
                        print(f"警告/Warning: ViT Grad-CAMが開けないためスキップします [{i+1}/{len(recs)}]: {r.get('vit_grad_cam_path')} ({e}) / Failed to open ViT Grad-CAM, skipping: {r.get('vit_grad_cam_path')} ({e})")
                        answers.append(None)
                        response_times.append(None)
                        continue

                    try:
                        conv_att = Image.open(r["convnextv2_grad_cam_path"])
                    except Exception as e:
                        print(f"警告/Warning: ConvNeXtV2 Grad-CAMが開けないためスキップします [{i+1}/{len(recs)}]: {r.get('convnextv2_grad_cam_path')} ({e}) / Failed to open ConvNeXtV2 Grad-CAM, skipping: {r.get('convnextv2_grad_cam_path')} ({e})")
                        answers.append(None)
                        response_times.append(None)
                        continue

                    bar_graph_path = r.get("weak_ensemble_bar_graph_path")
                    if bar_graph_path and not pd.isna(bar_graph_path) and str(bar_graph_path).strip():
                        try:
                            bar_graph = Image.open(bar_graph_path)
                        except Exception as e:
                            print(f"警告/Warning: 棒グラフが開けないため棒グラフなしで続行 [{i+1}/{len(recs)}]: {bar_graph_path} ({e}) / Failed to open bar graph, continue without it: {bar_graph_path} ({e})")
                            bar_graph = None
                # --- ここまでスキップ処理 ---

                captured_answer = None

                def show_image():
                    nonlocal fig, captured_answer
                    captured_answer = None

                    if group == 1:
                        if bar_graph:
                            if cheat_sheet:
                                fig = plt.figure(figsize=(24, 10))
                                ax_img = plt.subplot2grid((2, 3), (0, 0), rowspan=2, fig=fig)
                                ax_img.imshow(img); ax_img.axis("off"); ax_img.set_title("Image", fontsize=18)
                                ax_bar = plt.subplot2grid((2, 3), (0, 1), colspan=2, fig=fig)
                                ax_bar.imshow(bar_graph); ax_bar.axis("off")
                                ax_bar.set_title(f"Weak Learners & Ensemble\nEns Prob: {r['pred_prob']:.2f}", fontsize=18)
                                ax_cheat = plt.subplot2grid((2, 3), (1, 1), colspan=2, fig=fig)
                                ax_cheat.imshow(cheat_sheet); ax_cheat.axis("off")
                                ax_cheat.set_title("Cheat Sheet", fontsize=18)
                            else:
                                fig = plt.figure(figsize=(24, 8))
                                ax_img = plt.subplot2grid((1, 3), (0, 0), rowspan=1, fig=fig)
                                ax_img.imshow(img); ax_img.axis("off"); ax_img.set_title("Image", fontsize=18)
                                ax_bar = plt.subplot2grid((1, 3), (0, 1), colspan=2, fig=fig)
                                ax_bar.imshow(bar_graph); ax_bar.axis("off")
                                ax_bar.set_title(f"Weak Learners & Ensemble\nEns Prob: {r['pred_prob']:.2f}", fontsize=18)
                            plt.subplots_adjust(wspace=0.02, hspace=0.05)
                            plt.tight_layout(pad=0.5)
                        else:
                            if cheat_sheet:
                                fig = plt.figure(figsize=(24, 12))
                                ax_img = plt.subplot(1, 2, 1)
                                ax_img.imshow(img); ax_img.axis("off"); ax_img.set_title("ECG Image", fontsize=12)
                                ax_cheat = plt.subplot(1, 2, 2)
                                ax_cheat.imshow(cheat_sheet); ax_cheat.axis("off"); ax_cheat.set_title("Cheat Sheet", fontsize=12)
                                plt.tight_layout(pad=2.0)
                            else:
                                fig = plt.figure(figsize=(24, 12))
                                plt.imshow(img); plt.axis("off")

                    elif group == 2:
                        if bar_graph:
                            if cheat_sheet:
                                fig = plt.figure(figsize=(24, 16))
                                ax_img = plt.subplot2grid((4, 3), (0, 0), rowspan=4, fig=fig)
                                ax_img.imshow(img); ax_img.axis("off"); ax_img.set_title("Image", fontsize=18)
                                ax_bar = plt.subplot2grid((4, 3), (0, 1), colspan=2, fig=fig)
                                ax_bar.imshow(bar_graph); ax_bar.axis("off")
                                ax_bar.set_title(f"Weak Learners & Ensemble\nEns Prob: {r['pred_prob']:.2f}", fontsize=18)
                                ax_vit = plt.subplot2grid((4, 3), (1, 1), colspan=2, fig=fig)
                                ax_vit.imshow(vit_att); ax_vit.axis("off")
                                ax_vit.set_title("ViT Grad-CAM", fontsize=18)
                                ax_conv = plt.subplot2grid((4, 3), (2, 1), colspan=2, fig=fig)
                                ax_conv.imshow(conv_att); ax_conv.axis("off")
                                ax_conv.set_title("ConvNeXtV2 Grad-CAM", fontsize=18)
                                ax_cheat = plt.subplot2grid((4, 3), (3, 1), colspan=2, fig=fig)
                                ax_cheat.imshow(cheat_sheet); ax_cheat.axis("off")
                                ax_cheat.set_title("Cheat Sheet", fontsize=18)
                            else:
                                fig = plt.figure(figsize=(24, 14))
                                ax_img = plt.subplot2grid((3, 3), (0, 0), rowspan=3, fig=fig)
                                ax_img.imshow(img); ax_img.axis("off"); ax_img.set_title("Image", fontsize=18)
                                ax_bar = plt.subplot2grid((3, 3), (0, 1), colspan=2, fig=fig)
                                ax_bar.imshow(bar_graph); ax_bar.axis("off")
                                ax_bar.set_title(f"Weak Learners & Ensemble\nEns Prob: {r['pred_prob']:.2f}", fontsize=18)
                                ax_vit = plt.subplot2grid((3, 3), (1, 1), colspan=2, fig=fig)
                                ax_vit.imshow(vit_att); ax_vit.axis("off")
                                ax_vit.set_title("ViT Grad-CAM", fontsize=18)
                                ax_conv = plt.subplot2grid((3, 3), (2, 1), colspan=2, fig=fig)
                                ax_conv.imshow(conv_att); ax_conv.axis("off")
                                ax_conv.set_title("ConvNeXtV2 Grad-CAM", fontsize=18)
                            plt.subplots_adjust(wspace=0.02, hspace=0.05)
                            plt.tight_layout(pad=0.5)
                        else:
                            if cheat_sheet:
                                fig = plt.figure(figsize=(24, 12))
                                ax_img = plt.subplot2grid((3, 3), (0, 0), rowspan=3, fig=fig)
                                ax_img.imshow(img); ax_img.axis("off"); ax_img.set_title("Image", fontsize=18)
                                ax_vit = plt.subplot2grid((3, 3), (0, 1), colspan=2, fig=fig)
                                ax_vit.imshow(vit_att); ax_vit.axis("off")
                                ax_vit.set_title("ViT Grad-CAM", fontsize=18)
                                ax_conv = plt.subplot2grid((3, 3), (1, 1), colspan=2, fig=fig)
                                ax_conv.imshow(conv_att); ax_conv.axis("off")
                                ax_conv.set_title(f"ConvNeXtV2 Grad-CAM\nEns Prob: {r['pred_prob']:.2f}", fontsize=18)
                                ax_cheat = plt.subplot2grid((3, 3), (2, 1), colspan=2, fig=fig)
                                ax_cheat.imshow(cheat_sheet); ax_cheat.axis("off")
                                ax_cheat.set_title("Cheat Sheet", fontsize=18)
                            else:
                                fig = plt.figure(figsize=(24, 10))
                                ax_img = plt.subplot2grid((2, 3), (0, 0), rowspan=2, fig=fig)
                                ax_img.imshow(img); ax_img.axis("off"); ax_img.set_title("Image", fontsize=18)
                                ax_vit = plt.subplot2grid((2, 3), (0, 1), colspan=2, fig=fig)
                                ax_vit.imshow(vit_att); ax_vit.axis("off")
                                ax_vit.set_title("ViT Grad-CAM", fontsize=18)
                                ax_conv = plt.subplot2grid((2, 3), (1, 1), colspan=2, fig=fig)
                                ax_conv.imshow(conv_att); ax_conv.axis("off")
                                ax_conv.set_title(f"ConvNeXtV2 Grad-CAM\nEns Prob: {r['pred_prob']:.2f}", fontsize=18)
                            plt.subplots_adjust(wspace=0.02, hspace=0.05)
                            plt.tight_layout(pad=0.5)

                    else:
                        if cheat_sheet:
                            fig = plt.figure(figsize=(24, 12))
                            ax_img = plt.subplot(1, 2, 1)
                            ax_img.imshow(img); ax_img.axis("off"); ax_img.set_title("ECG Image", fontsize=12)
                            ax_cheat = plt.subplot(1, 2, 2)
                            ax_cheat.imshow(cheat_sheet); ax_cheat.axis("off"); ax_cheat.set_title("Cheat Sheet", fontsize=12)
                            plt.tight_layout(pad=2.0)
                        else:
                            fig = plt.figure(figsize=(24,12))
                            plt.imshow(img); plt.axis("off")

                    def on_key(event):
                        nonlocal captured_answer
                        if event.key in ('enter', 'return', ' '):
                            plt.close('all')
                        elif event.key.lower() in ('t', '0'):
                            captured_answer = '0'
                            print("t")
                            plt.close('all')
                        elif event.key.lower() in ('a', '1'):
                            captured_answer = '1'
                            print("a")
                            plt.close('all')

                    fig.canvas.mpl_connect('key_press_event', on_key)
                    plt.show(block=False)
                    while plt.get_fignums():
                        plt.pause(0.1)

                start_time = time.time()

                print(f"\n[{i+1}/{len(recs)}] 画像ウィンドウで回答キー（t/T/0 または a/A/1）を押すか、ウィンドウを閉じてコマンドラインで回答してください / Press answer key (t/T/0 or a/A/1) in image window, or close window to answer in command line")

                show_image()

                if captured_answer is None:
                    while True:
                        ans = input(f"[{i+1}/{len(recs)}] ラベルを入力 (takotsubo: 0 または t/T, AMI: 1 または a/A, 画像再表示: r/R/i/I) / Please input label (takotsubo: 0 or t/T, AMI: 1 or a/A, show image again: r/R/i/I): ").strip()
                        if ans.lower() in ("r", "i"):
                            print("画像を再表示します / Showing image again...")
                            show_image()
                            if captured_answer is not None:
                                ans = captured_answer
                                break
                            continue
                        elif ans.lower() in ("0", "1", "t", "a"):
                            break
                        else:
                            print("0, 1, t/T, a/A, r/R, または i/I を入力してください / Please input 0, 1, t/T, a/A, r/R, or i/I")
                else:
                    ans = captured_answer

                end_time = time.time()
                response_time = end_time - start_time

                if ans.lower() in ("t", "0"):
                    ans_value = 0
                else:
                    ans_value = 1

                answers.append(ans_value)
                response_times.append(response_time)

                print(f"回答時間/Response time: {response_time:.2f}秒/sec")

            finally:
                try:
                    plt.close('all')
                except Exception:
                    pass

                try:
                    if img is not None:
                        img.close()
                    if vit_att is not None:
                        vit_att.close()
                    if conv_att is not None:
                        conv_att.close()
                    if bar_graph is not None:
                        bar_graph.close()
                except Exception:
                    pass

        # 正答率（回答できた分だけで計算）
        valid = [(a, r) for a, r in zip(answers, recs) if a in (0, 1)]
        if len(valid) == 0:
            acc = 0.0
        else:
            correct = [int(a == int(r["label"])) for a, r in valid]
            acc = sum(correct) / len(correct)

        valid_times = [rt for rt in response_times if isinstance(rt, (int, float))]
        #avg_time = sum(valid_times) / len(valid_times) if valid_times else 0.0

        return answers, response_times, acc

    finally:
        if cheat_sheet is not None:
            try:
                cheat_sheet.close()
            except Exception:
                pass

def append_results(row, results_file):
    """
    結果をCSVファイルに追記する。
    既存のファイルがある場合、既存の列名と新しい行の列名をマージして、
    すべての列を含む完全なヘッダーで書き直す。
    インデックス関連の列はインデックス順に並べる。
    """
    new_header = list(row.keys())
    file_exists = os.path.exists(results_file)
    
    if file_exists:
        # 既存のファイルを読み込む
        existing_data = []
        existing_header = []
        try:
            with open(results_file, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                existing_header = reader.fieldnames or []
                existing_data = list(reader)
        except Exception as e:
            # ファイルが読み込めない場合は新規作成として扱う
            print(f"警告/Warning: 既存ファイルの読み込みに失敗しました。新規作成します: {e} / Failed to read existing file. Creating new file: {e}")
            file_exists = False
    
    # すべての列名を収集
    all_columns = set()
    if file_exists and existing_header:
        all_columns.update(existing_header)
    all_columns.update(new_header)
    
    # 基本情報列（固定の順序）
    basic_columns = ["timestamp", "name", "profession", "years", "is_first_time", "phase", "group", "group_name", "randomization_mode", "stratum", "stratum_label", "randomization_slot_id", "randomization_block_id", "accuracy", "sensitivity", "specificity", "avg_response_time"]
    basic_columns = [col for col in basic_columns if col in all_columns]
    
    # インデックス関連の列を抽出
    index_columns = {}
    other_columns = []

    prefix_index = "index_"
    prefix_time = "time_index_"
    prefix_correct = "correct_index_"

    for col in all_columns:
        if col.startswith(prefix_index) and col[len(prefix_index):].isdigit():
            idx = int(col[len(prefix_index):])
            index_columns.setdefault(idx, []).append(col)

        elif col.startswith(prefix_time) and col[len(prefix_time):].isdigit():
            idx = int(col[len(prefix_time):])
            index_columns.setdefault(idx, []).append(col)

        elif col.startswith(prefix_correct) and col[len(prefix_correct):].isdigit():
            idx = int(col[len(prefix_correct):])
            index_columns.setdefault(idx, []).append(col)

        elif col not in basic_columns:
            other_columns.append(col)
    
    # インデックス順にソートして列を構築
    # すべてのインデックスを収集（new_headerからも取得）
    all_indices = set()

    prefix_index = "index_"
    prefix_time = "time_index_"
    prefix_correct = "correct_index_"

    for col in all_columns:
        if col.startswith(prefix_index) and col[len(prefix_index):].isdigit():
            all_indices.add(int(col[len(prefix_index):]))

        elif col.startswith(prefix_time) and col[len(prefix_time):].isdigit():
            all_indices.add(int(col[len(prefix_time):]))

        elif col.startswith(prefix_correct) and col[len(prefix_correct):].isdigit():
            all_indices.add(int(col[len(prefix_correct):]))
    
    sorted_indices = sorted(all_indices)
    index_ordered_columns = []
    index_ordered_set = set()  # 重複を防ぐためのセット
    for idx in sorted_indices:
        # 各インデックスについて、index, correct_index, time_indexの順で配置（readability向上のため）
        # 列が存在するかどうかに関わらず、すべての列を追加する
        for prefix in ["index_", "correct_index_", "time_index_"]:
            col_name = f"{prefix}{idx}"
            if col_name in all_columns and col_name not in index_ordered_set:
                index_ordered_columns.append(col_name)
                index_ordered_set.add(col_name)
    
    # 最終的な列の順序: 基本情報列 + インデックス関連列（順序付き）+ その他の列
    # other_columnsから、index_ordered_columnsに既に含まれている列を除外
    other_columns_filtered = [col for col in other_columns if col not in index_ordered_set]
    merged_header = basic_columns + index_ordered_columns + sorted(other_columns_filtered)
    
    # 既存のデータに新しい行を追加
    if file_exists and existing_data:
        all_data = existing_data + [row]
    else:
        all_data = [row]
    
    # 完全なヘッダーでファイルを書き直す（アトミックな書き込みのため一時ファイルを使用）
    temp_file = results_file + ".tmp"
    try:
        with open(temp_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=merged_header)
            writer.writeheader()
            # 既存のデータと新しい行を書き込む（不足している列は空文字で埋める）
            for data_row in all_data:
                # すべての列が存在するようにする（不足している列は空文字）
                # データ型を保持するため、数値の場合は文字列に変換してから保存
                complete_row = {}
                for col in merged_header:
                    val = data_row.get(col, "")
                    # 数値の場合は文字列に変換（CSVでは文字列として保存）
                    if isinstance(val, (int, float)):
                        complete_row[col] = str(val)
                    elif val == "":
                        complete_row[col] = ""
                    else:
                        complete_row[col] = str(val) if val is not None else ""
                writer.writerow(complete_row)
        
        # 書き込みが成功したら、一時ファイルを元のファイルに置き換える（アトミック操作）
        os.replace(temp_file, results_file)
    except Exception as e:
        # エラーが発生した場合は、一時ファイルを削除してからエラーを再発生させる
        if os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass
        raise e

def save_phase_results(name, profession, years, is_first_time, phase, answers, response_times, acc, questions, results_file=RESULTS_SEQ_CSV):
    """各フェーズの結果を1行で保存（None=スキップは除外して指標計算、CSVは空欄で保存）"""
    ts = datetime.datetime.now().isoformat()

    tp = tn = fp = fn = 0
    valid_rts = []

    for a, q, rt in zip(answers, questions, response_times):
        if a not in (0, 1):
            continue
        true_label = int(q['label'])
        if isinstance(rt, (int, float)):
            valid_rts.append(rt)

        if true_label == 1 and a == 1:
            tp += 1
        elif true_label == 0 and a == 0:
            tn += 1
        elif true_label == 0 and a == 1:
            fp += 1
        elif true_label == 1 and a == 0:
            fn += 1

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    #valid_rts = [rt for rt in response_times if isinstance(rt, (int, float))]
    avg_rt = (sum(valid_rts) / len(valid_rts)) if valid_rts else 0.0

    row = {
        "timestamp": ts,
        "name": name,
        "profession": profession,
        "years": years,
        "is_first_time": is_first_time,
        "phase": phase,
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "avg_response_time": avg_rt,
    }

    data_list = list(zip(answers, questions, response_times))
    data_list.sort(key=lambda x: x[1]['index'])

    for a, q, rt in data_list:
        idx = q['index']
        if a in (0, 1):
            row[f"index_{idx}"] = a
            row[f"time_index_{idx}"] = rt if isinstance(rt, (int, float)) else ""
            row[f"correct_index_{idx}"] = 1 if a == int(q['label']) else 0
        else:
            # スキップは空欄で保存（列自体は append_results がヘッダを保持してくれる）
            row[f"index_{idx}"] = ""
            row[f"time_index_{idx}"] = ""
            row[f"correct_index_{idx}"] = ""

    append_results(row, results_file)

def save_phase_4_results(name, profession, years, is_first_time, group, answers, response_times, acc, questions, randomization_info=None):
    """フェーズ4の結果を1行で保存（None=スキップは除外して指標計算、CSVは空欄で保存）"""
    ts = datetime.datetime.now().isoformat()
    randomization_info = randomization_info or {}

    tp = tn = fp = fn = 0
    valid_rts = []

    for a, q, rt in zip(answers, questions, response_times):
        if a not in (0, 1):
            continue
        true_label = int(q['label'])
        if isinstance(rt, (int, float)):
            valid_rts.append(rt)

        if true_label == 1 and a == 1:
            tp += 1
        elif true_label == 0 and a == 0:
            tn += 1
        elif true_label == 0 and a == 1:
            fp += 1
        elif true_label == 1 and a == 0:
            fn += 1

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    #valid_rts = [rt for rt in response_times if isinstance(rt, (int, float))]
    avg_rt = (sum(valid_rts) / len(valid_rts)) if valid_rts else 0.0

    row = {
        "timestamp": ts,
        "name": name,
        "profession": profession,
        "years": years,
        "is_first_time": is_first_time,
        "phase": 4,
        "group": group,
        "group_name": "cheat_sheet_only" if group == 0 else ("cheat_sheet_ai_no_gradcam" if group == 1 else "cheat_sheet_ai_gradcam"),
        "randomization_mode": randomization_info.get("randomization_mode", ""),
        "stratum": randomization_info.get("stratum", ""),
        "stratum_label": randomization_info.get("stratum_label", ""),
        "randomization_slot_id": randomization_info.get("randomization_slot_id", ""),
        "randomization_block_id": randomization_info.get("randomization_block_id", ""),
        "accuracy": acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "avg_response_time": avg_rt,
    }

    data_list = list(zip(answers, questions, response_times))
    data_list.sort(key=lambda x: x[1]['index'])

    for a, q, rt in data_list:
        idx = q['index']
        if a in (0, 1):
            row[f"index_{idx}"] = a
            row[f"time_index_{idx}"] = rt if isinstance(rt, (int, float)) else ""
            row[f"correct_index_{idx}"] = 1 if a == int(q['label']) else 0
        else:
            row[f"index_{idx}"] = ""
            row[f"time_index_{idx}"] = ""
            row[f"correct_index_{idx}"] = ""

    append_results(row, RESULTS_RAND_CSV)

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description='tako, AMIのECG診断の、人間のテスト')
    parser.add_argument('--test', action='store_true', help='テストモード: 各フェーズの先頭5問のみ実行')
    parser.add_argument('--seed', type=int, default=None, help='ランダムシード値 (指定しない場合は現在時刻を使用)')
    parser.add_argument('--start-phase', type=int, choices=[1, 2, 3, 4], default=1, 
                       help='開始フェーズを指定 (1-4, default: 1)')
    parser.add_argument('--cheat-sheet', type=str, default=None,
                       help='チートシート画像のパス（指定した場合のみフェーズ2、3、4で表示）')
    parser.add_argument('--phase4-weights', type=int, nargs=3, default=[1, 1, 1],
                       metavar=('W0', 'W1', 'W2'),
                       help='フェーズ4のグループ選択の重み (group0, group1, group2) (default: 1 1 1)')
    parser.add_argument('--randomization-mode', type=str, choices=['stratified_block', 'simple'],
                       default='stratified_block',
                       help='フェーズ4の無作為化方式 (default: stratified_block)')
    parser.add_argument('--randomization-table', type=str, default=DEFAULT_RANDOMIZATION_TABLE,
                       help='層別ブロック無作為化の割付表CSV')
    parser.add_argument('--randomization-state', type=str, default=DEFAULT_RANDOMIZATION_STATE,
                       help='層別ブロック無作為化の状態CSV')
    parser.add_argument('--max-enrollment', type=int, default=100,
                       help='層別ブロック無作為化の最大割付人数 (default: 100)')
    parser.add_argument('--career-threshold', type=int, default=10,
                       help='層別化の職歴年数しきい値 (default: 10)')
    args = parser.parse_args()
    
    test_mode = args.test
    start_phase = args.start_phase
    cheat_sheet_path = args.cheat_sheet
    phase4_weights = args.phase4_weights
    randomization_mode = args.randomization_mode
    randomization_table_path = args.randomization_table
    randomization_state_path = args.randomization_state
    max_enrollment = args.max_enrollment
    career_threshold = args.career_threshold
    
    # フェーズ4の重みの検証
    if any(w < 0 for w in phase4_weights):
        raise ValueError("フェーズ4の重みは0以上の整数である必要があります / Phase 4 weights must be non-negative integers")
    if sum(phase4_weights) == 0:
        raise ValueError("フェーズ4の重みの合計が0です / Sum of Phase 4 weights cannot be zero")
    if max_enrollment < 1:
        raise ValueError("最大割付人数は1以上である必要があります / max-enrollment must be >= 1")
    if career_threshold < 0:
        raise ValueError("職歴年数しきい値は0以上である必要があります / career-threshold must be >= 0")
    if randomization_mode == 'stratified_block':
        if not os.path.exists(randomization_table_path):
            raise FileNotFoundError(
                f"割付表が見つかりません: {randomization_table_path}\n"
                f"generate_stratified_block_randomization_table.py を実行して作成してください。"
                f" / Randomization table not found: {randomization_table_path}"
            )
        randomization_table = load_randomization_table(randomization_table_path)
    else:
        randomization_table = None
    
    if test_mode:
        print("="*50)
        print("テストモード: 各フェーズの先頭5問のみ実行します / Test mode: Only first 5 questions per phase")
        print("="*50)
    
    if start_phase > 1:
        print("="*50)
        print(f"デバッグモード: フェーズ{start_phase}から開始します / Debug mode: Starting from phase {start_phase}")
        print("="*50)
    
    # 無作為化の再現性のためのseed設定
    if args.seed is None:
        # シード値が指定されていない場合は現在時刻を使用
        args.seed = int(time.time())
    random.seed(args.seed)  # 固定のseed値で再現性を確保
    print(f"ランダムシード/Random seed: {args.seed}")
    
    # Instructionを表示して同意を得る
    show_instructions()
    
    # 名前・職種・職歴・テスト経験の入力（結果保存のために常に必要）
    name, profession, years, is_first_time = prompt_user_info()

    # 結果変数の初期化
    ans1, time1, acc1 = None, None, None
    ans2, time2, acc2 = None, None, None
    ans3, time3, acc3 = None, None, None
    ans4, time4, acc4 = None, None, None
    group = None
    group_name = None
    group_name_en = None
    randomization_info = None

    # フェーズ1-3用の質問データ読み込み（フェーズ1-3を実行する場合のみ）
    if start_phase <= 3:
        questions = load_questions(QUESTIONS_SEQ_CSV)
        # テストモードの場合は先頭5問のみ
        if test_mode:
            questions = questions[:5]

    # フェーズ1
    if start_phase <= 1:
        print("Test phase 1: 画像ウィンドウ内の心電図を確認したら、ウィンドウを閉じて回答してください/Please check the ECG in the image window and close the window to answer.")
        print("画像ウィンドウを閉じる方法: Enterキー、スペースキー、またはマウスで×ボタンをクリック / How to close image window: Press Enter, Space, or click the × button with mouse")
        ans1, time1, acc1 = run_phase(questions, phase=1, cheat_sheet_path=cheat_sheet_path)
        save_phase_results(name, profession, years, is_first_time, 1, ans1, time1, acc1, questions)

    # フェーズ2
    if start_phase <= 2:
        input("チートシートを参照してOKなら Enter を押してください/Please refer to the cheat sheet and press Enter if you are OK.")
        print("画像ウィンドウを閉じる方法: Enterキー、スペースキー、またはマウスで×ボタンをクリック / How to close image window: Press Enter, Space, or click the × button with mouse")
        ans2, time2, acc2 = run_phase(questions, phase=2, cheat_sheet_path=cheat_sheet_path)
        save_phase_results(name, profession, years, is_first_time, 2, ans2, time2, acc2, questions)

    # フェーズ3
    if start_phase <= 3:
        input("ModelによるAMIのprobabilityが表示されます。OKなら Enter を押してください/The probability of AMI by the model will be displayed. Please press Enter if you are OK.")
        print("画像ウィンドウを閉じる方法: Enterキー、スペースキー、またはマウスで×ボタンをクリック / How to close image window: Press Enter, Space, or click the × button with mouse")
        ans3, time3, acc3 = run_phase(questions, phase=3, cheat_sheet_path=cheat_sheet_path)
        save_phase_results(name, profession, years, is_first_time, 3, ans3, time3, acc3, questions)

    # フェーズ4 - 無作為テスト
    if start_phase <= 4:
        print("\n=== フェーズ4: 最終テスト ===/Phase 4: Final test")
        # group=0: チートシートのみ, group=1: チートシート＋AI予測（Grad-CAMなし）, group=2: チートシート＋AI予測（Grad-CAMあり）
        if randomization_mode == 'stratified_block':
            stratum = determine_stratum(profession, years, career_threshold=career_threshold)
            group, randomization_info = assign_stratified_block_group(
                randomization_table,
                randomization_state_path,
                stratum,
                max_enrollment,
            )
            print(
                f"無作為化方式/Randomization mode: stratified_block "
                f"(stratum={stratum}, slot={randomization_info['randomization_slot_id']}, "
                f"block={randomization_info['randomization_block_id']}, group={group})"
            )
            print(
                f"最大割付人数/Max enrollment: {max_enrollment}, "
                f"職歴しきい値/Career threshold: {career_threshold} years"
            )
        else:
            group = random.choices([0, 1, 2], weights=phase4_weights)[0]
            randomization_info = {"randomization_mode": "simple"}
            print(f"無作為化方式/Randomization mode: simple")
            print(f"フェーズ4のグループ選択重み: {phase4_weights} (group0:group1:group2) / Phase 4 group selection weights: {phase4_weights}")
        if group == 0:
            group_name = "チートシートのみ"
            group_name_en = "Cheat sheet only"
        elif group == 1:
            group_name = "チートシート＋AI予測（Grad-CAMなし）"
            group_name_en = "Cheat sheet + AI prediction (no Grad-CAM)"
        else:  # group == 2
            group_name = "チートシート＋AI予測（Grad-CAMあり）"
            group_name_en = "Cheat sheet + AI prediction (with Grad-CAM)"
        #print(f"あなたは {group_name} グループに割り当てられました")
        
        # 無作為テスト用の質問データ読み込み
        questions_rand = load_questions(QUESTIONS_RAND_CSV)
        # テストモードの場合は先頭5問のみ
        if test_mode:
            questions_rand = questions_rand[:5]
        
        input(f"フェーズ4を開始します。準備ができたら Enter を押してください/Phase 4 starts. Please press Enter if you are ready.")
        print("画像ウィンドウを閉じる方法: Enterキー、スペースキー、またはマウスで×ボタンをクリック / How to close image window: Press Enter, Space, or click the × button with mouse")
        ans4, time4, acc4 = run_phase_4(questions_rand, group, cheat_sheet_path=cheat_sheet_path)
        save_phase_4_results(name, profession, years, is_first_time, group, ans4, time4, acc4, questions_rand, randomization_info=randomization_info)

    # 全フェーズの結果まとめ（実行されたフェーズのみ表示）
    print("\n" + "="*50)
    print("全フェーズの結果まとめ/Summary of all phases")
    print("="*50)
    
    # SensitivityとSpecificityを計算する関数（save_phase_resultsと同じロジック）
    def calc_sensitivity_specificity(answers, questions):
        """Sensitivity（感度）とSpecificity（特異度）を計算（None=スキップは除外）"""
        tp = tn = fp = fn = 0
        for a, q in zip(answers, questions):
            if a not in (0, 1):
                continue
            true_label = int(q['label'])
            if true_label == 1 and a == 1:
                tp += 1
            elif true_label == 0 and a == 0:
                tn += 1
            elif true_label == 0 and a == 1:
                fp += 1
            elif true_label == 1 and a == 0:
                fn += 1

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        return sensitivity, specificity

    def mean_ignore_none(xs):
        vals = [x for x in xs if isinstance(x, (int, float))]
        return (sum(vals) / len(vals)) if vals else 0.0

    if ans1 is not None:
        sens1, spec1 = calc_sensitivity_specificity(ans1, questions)
        print(f"フェーズ1 (画像のみ)/Phase 1 (Image only): 正答率/Accuracy {acc1*100:.1f}%, 感度/Sensitivity {sens1*100:.1f}%, 特異度/Specificity {spec1*100:.1f}%, 平均回答時間/Avg response time {mean_ignore_none(time1):.2f}秒/sec")
    if ans2 is not None:
        sens2, spec2 = calc_sensitivity_specificity(ans2, questions)
        print(f"フェーズ2 (チートシート可)/Phase 2 (Cheat sheet available): 正答率/Accuracy {acc2*100:.1f}%, 感度/Sensitivity {sens2*100:.1f}%, 特異度/Specificity {spec2*100:.1f}%, 平均回答時間/Avg response time {mean_ignore_none(time2):.2f}秒/sec")
    if ans3 is not None:
        sens3, spec3 = calc_sensitivity_specificity(ans3, questions)
        print(f"フェーズ3 (AI予測あり)/Phase 3 (AI prediction available): 正答率/Accuracy {acc3*100:.1f}%, 感度/Sensitivity {sens3*100:.1f}%, 特異度/Specificity {spec3*100:.1f}%, 平均回答時間/Avg response time {mean_ignore_none(time3):.2f}秒/sec")
    if ans4 is not None:
        sens4, spec4 = calc_sensitivity_specificity(ans4, questions_rand)
        print(f"フェーズ4 ({group_name})/Phase 4 ({group_name_en}): 正答率/Accuracy {acc4*100:.1f}%, 感度/Sensitivity {sens4*100:.1f}%, 特異度/Specificity {spec4*100:.1f}%, 平均回答時間/Avg response time {mean_ignore_none(time4):.2f}秒/sec")
    print("="*50)

    # 終了
    print("\nHave a nice day!")

if __name__ == "__main__":
    main()

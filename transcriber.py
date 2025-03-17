import os
import torch
import whisper
from whisper.utils import get_writer
import argparse
import time
import numpy as np
from pydub import AudioSegment
import datetime
import sys

def print_progress_bar(iteration, total, prefix='', suffix='', length=50, fill='█', print_end="\r"):
    """
    進捗バーを表示するための関数
    """
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=print_end)
    sys.stdout.flush()
    # 完了時は改行する
    if iteration == total:
        print()

def transcribe_audio(model, audio_path, output_dir=None, output_format="txt", language="ja", verbose=False):
    """
    音声ファイルを文字起こしします
    """
    start_time = time.time()
    print(f"文字起こし処理を開始します: {audio_path}")
    print(f"開始時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 出力ディレクトリの設定
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # ファイル名の取得
    file_name = os.path.basename(audio_path).split('.')[0]
    output_path = os.path.join(output_dir, file_name) if output_dir else file_name
    
    # メモリ使用量の表示
    if torch.cuda.is_available():
        print(f"GPU状態: {torch.cuda.device_count()}台利用可能")
        print(f"GPU使用メモリ: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        print(f"GPU合計メモリ: {torch.cuda.memory_reserved() / 1024 / 1024:.2f} MB")
    
    # Whisperモデルの推論オプション設定
    options = {
        "language": language,
        "task": "transcribe",
        "verbose": verbose,
        "word_timestamps": True,  # 単語ごとのタイムスタンプを取得
        "fp16": torch.cuda.is_available()  # GPUがある場合はfp16を使用
    }
    
    # 音声ファイルのサイズをチェック
    file_size = os.path.getsize(audio_path) / (1024 * 1024)  # MBに変換
    print(f"ファイルサイズ: {file_size:.2f} MB")
    
    # 通常の処理
    print(f"モデル {model.model.name} で文字起こしを開始します")
    result = model.transcribe(audio_path, **options)
    
    # 結果の保存
    if output_format is not None:
        writer = get_writer(output_format, output_dir)
        writer(result, output_path)
        print(f"文字起こし結果を保存しました: {output_path}.{output_format}")
    
    elapsed_time = time.time() - start_time
    print(f"処理時間: {elapsed_time:.2f}秒 ({elapsed_time / 60:.2f}分)")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="Whisperモデルを使用して音声ファイルを文字起こしします")
    parser.add_argument("--model", type=str, default="large-v3", help="使用するWhisperモデル (tiny, base, small, medium, large)")
    parser.add_argument("--model_dir", type=str, default=None, help="モデルを保存/読み込むディレクトリ")
    parser.add_argument("--language", type=str, default="ja", help="音声の言語 (例: ja, en)")
    parser.add_argument("--audio", type=str, required=True, help="文字起こしする音声ファイルのパス")
    parser.add_argument("--output_dir", type=str, default="output", help="出力ディレクトリ")
    parser.add_argument("--output_format", type=str, default="txt", help="出力フォーマット (txt, srt, vtt, tsv, json)")
    parser.add_argument("--device", type=str, default=None, help="使用するデバイス (cpu, cuda)")
    parser.add_argument("--verbose", action="store_true", help="詳細な出力を表示")
    
    args = parser.parse_args()
    
    # デバイスの設定
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"デバイス: {device}")
    
    # モデルのロード
    print(f"モデル '{args.model}' をロードしています...")
    model_load_start = time.time()
    
    if args.model_dir:
        model_path = os.path.join(args.model_dir, args.model)
        model = whisper.load_model(args.model, device=device, download_root=model_path)
    else:
        model = whisper.load_model(args.model, device=device)
    
    model_load_time = time.time() - model_load_start
    print(f"モデルのロード完了 ({model_load_time:.2f}秒)")
    
    # 音声ファイルの文字起こし
    result = transcribe_audio(model, args.audio, args.output_dir, args.output_format, args.language, args.verbose)
    
    # テキスト結果の表示
    print("\n文字起こし結果:")
    print(result["text"])

if __name__ == "__main__":
    main()
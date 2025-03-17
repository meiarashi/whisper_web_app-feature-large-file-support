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

def process_long_audio(audio_path, chunk_size_ms=30000, overlap_ms=2000):
    """
    長い音声ファイルを適切な長さのチャンクに分割して処理します
    chunk_size_ms: チャンクのサイズ（ミリ秒）
    overlap_ms: オーバーラップの長さ（ミリ秒）
    """
    print(f"長い音声ファイルを処理しています: {audio_path}")
    
    # 音声ファイルの読み込み
    audio = AudioSegment.from_file(audio_path)
    audio_length_ms = len(audio)
    audio_length_seconds = audio_length_ms / 1000
    
    print(f"音声の長さ: {audio_length_seconds:.2f}秒 ({datetime.timedelta(seconds=int(audio_length_seconds))})")
    
    # チャンクのリストを作成
    chunks = []
    for i in range(0, audio_length_ms, chunk_size_ms - overlap_ms):
        chunk_start = i
        chunk_end = min(i + chunk_size_ms, audio_length_ms)
        chunk = audio[chunk_start:chunk_end]
        chunks.append((chunk, chunk_start, chunk_end))
    
    print(f"音声を {len(chunks)} チャンクに分割しました")
    return chunks

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
    
    if file_size > 25:  # 25MB以上の場合は分割処理
        print(f"ファイルサイズが大きいため分割処理を行います: {file_size:.2f}MB")
        chunks = process_long_audio(audio_path)
        
        all_segments = []
        for i, (chunk, start_ms, end_ms) in enumerate(chunks):
            # 進捗バー表示
            progress_percentage = (i / len(chunks)) * 100
            print_progress_bar(i + 1, len(chunks), prefix=f'チャンク処理:', suffix=f'完了 ({i+1}/{len(chunks)})', length=40)
            
            # 一時ファイルに保存
            temp_path = f"temp_chunk_{i}.wav"
            chunk.export(temp_path, format="wav")
            
            print(f"\nチャンク {i+1}/{len(chunks)} を処理中... ({start_ms/1000:.2f}秒 - {end_ms/1000:.2f}秒)")
            chunk_start_time = time.time()
            
            # 文字起こし
            result = model.transcribe(temp_path, **options)
            
            chunk_elapsed_time = time.time() - chunk_start_time
            print(f"チャンク {i+1} 処理完了: {chunk_elapsed_time:.2f}秒")
            
            # セグメントの時間を調整
            for segment in result["segments"]:
                segment["start"] += start_ms / 1000
                segment["end"] += start_ms / 1000
            
            all_segments.extend(result["segments"])
            
            # 一時ファイルを削除
            os.remove(temp_path)
            
            # 残り時間の予測
            if i > 0:
                avg_time_per_chunk = (time.time() - start_time) / (i + 1)
                estimated_total_time = avg_time_per_chunk * len(chunks)
                remaining_time = estimated_total_time - (time.time() - start_time)
                print(f"推定残り時間: {datetime.timedelta(seconds=int(remaining_time))}")
        
        # 結果を統合
        result = {
            "text": " ".join(segment["text"] for segment in all_segments),
            "segments": sorted(all_segments, key=lambda x: x["start"]),
            "language": language
        }
    else:
        # 通常処理（小さいファイル）
        print("文字起こしを実行中...")
        process_start_time = time.time()
        
        # 進捗表示（インクリメンタルでの進捗は難しいので10秒ごとに経過時間を表示）
        def progress_reporter():
            thread_start_time = time.time()
            while True:
                time.sleep(10)  # 10秒ごとに更新
                elapsed = time.time() - thread_start_time
                print(f"処理中... 経過時間: {datetime.timedelta(seconds=int(elapsed))}")
                
                # メモリ使用量の表示
                if torch.cuda.is_available():
                    print(f"GPU使用メモリ: {torch.cuda.memory_allocated() / 1024 / 1024:.2f} MB")
        
        # 進捗レポーターは使用しない（シンプルな実装のため）
        # スレッドを使うと複雑になる可能性があります
        
        # 文字起こし実行
        result = model.transcribe(audio_path, **options)
        
        process_elapsed_time = time.time() - process_start_time
        print(f"文字起こし処理にかかった時間: {process_elapsed_time:.2f}秒")
    
    # 結果の出力
    if output_format:
        writer = get_writer(output_format, output_dir)
        writer(result, output_path)
    
    elapsed_time = time.time() - start_time
    print(f"文字起こしが完了しました: {elapsed_time:.2f}秒 ({datetime.timedelta(seconds=int(elapsed_time))})")
    print(f"終了時刻: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 結果のプレビュー
    preview_length = min(500, len(result["text"]))
    print(f"\n文字起こし結果プレビュー (最初の{preview_length}文字):")
    print(result["text"][:preview_length] + "...")
    
    return result

def main():
    parser = argparse.ArgumentParser(description="高性能日本語音声文字起こしツール")
    parser.add_argument("audio_path", type=str, help="音声ファイルのパス")
    parser.add_argument("--model", type=str, default="base", choices=["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"], help="使用するWhisperモデル")
    parser.add_argument("--output_dir", type=str, default="output", help="出力ディレクトリ")
    parser.add_argument("--output_format", type=str, default="all", help="出力形式 (txt, vtt, srt, tsv, json, all)")
    parser.add_argument("--device", type=str, default=None, help="使用するデバイス (cuda, cpu)")
    parser.add_argument("--verbose", action="store_true", help="詳細な出力を表示")
    
    args = parser.parse_args()
    
    # デバイスの設定
    if args.device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    print(f"使用デバイス: {device}")
    print(f"Whisperモデル: {args.model}")
    
    # Pythonのバージョンとライブラリ情報の表示
    print(f"Python バージョン: {sys.version}")
    print(f"PyTorch バージョン: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA バージョン: {torch.version.cuda}")
    
    print("\nモデルをロード中...")
    model_load_start = time.time()
    
    # モデルのロード
    model = whisper.load_model(args.model, device=device)
    
    model_load_time = time.time() - model_load_start
    print(f"モデルのロードが完了しました: {model_load_time:.2f}秒")
    
    # 文字起こしの実行
    result = transcribe_audio(
        model=model,
        audio_path=args.audio_path,
        output_dir=args.output_dir,
        output_format=args.output_format,
        language="ja",
        verbose=args.verbose
    )
    
    print("\n文字起こし処理が正常に完了しました!")

if __name__ == "__main__":
    main()
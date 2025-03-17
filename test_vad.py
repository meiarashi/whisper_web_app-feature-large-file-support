"""
Silero VAD処理のテストスクリプト
使用方法: python test_vad.py <音声ファイルパス>
"""

import os
import sys
import torch
import torchaudio
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import argparse

# プロジェクトのモジュールをインポート
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dental_processor import process_with_vad, load_silero_vad

def visualize_speech_segments(audio_tensor, timestamps, sample_rate=16000):
    """音声波形と検出された発話区間を可視化"""
    fig, ax = plt.subplots(figsize=(15, 5))
    
    # 時間軸（秒）
    time = np.arange(len(audio_tensor)) / sample_rate
    
    # 波形プロット
    ax.plot(time, audio_tensor, color='gray', alpha=0.5)
    
    # 発話区間をハイライト
    for ts in timestamps:
        start_time = ts['start'] / sample_rate
        end_time = ts['end'] / sample_rate
        duration = end_time - start_time
        ax.axvspan(start_time, end_time, color='green', alpha=0.3)
        ax.text(start_time + duration/2, 0, f"{duration:.2f}s", 
                horizontalalignment='center', verticalalignment='center')
    
    ax.set_title('音声波形と検出された発話区間')
    ax.set_xlabel('時間 (秒)')
    ax.set_ylabel('振幅')
    
    # 出力フォルダ
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    
    # ファイル名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f'speech_segments_{timestamp}.png')
    
    # 保存
    plt.tight_layout()
    plt.savefig(output_file)
    print(f"可視化結果を保存しました: {output_file}")
    
    # 可能ならプロットを表示
    try:
        plt.show()
    except:
        print("プロットの表示をスキップします（GUI環境がありません）")

def convert_to_wav_if_needed(file_path):
    """必要に応じてファイルをWAVに変換"""
    if file_path.lower().endswith(('.m4a', '.mp3', '.aac', '.ogg')):
        import tempfile
        import subprocess
        import os
        
        print(f"音声ファイルをWAVに変換中: {file_path}")
        
        # 一時ファイル名を生成
        temp_dir = tempfile.gettempdir()
        temp_wav = os.path.join(temp_dir, f"temp_{os.path.basename(file_path)}.wav")
        
        # FFmpegを使用して変換
        try:
            # プロジェクト内のFFmpegを探す
            ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
            if not os.path.exists(ffmpeg_path):
                ffmpeg_path = "ffmpeg"  # システムのFFmpegを使用
                
            command = [ffmpeg_path, "-i", file_path, "-ac", "1", "-ar", "16000", temp_wav, "-y"]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print(f"変換完了: {temp_wav}")
            return temp_wav
        except Exception as e:
            print(f"変換エラー: {e}")
            print("変換をスキップします")
            return file_path
    return file_path

def test_vad_processing(audio_file):
    """VAD処理をテストして結果を表示"""
    print(f"音声ファイル: {audio_file}")
    
    if not os.path.exists(audio_file):
        print(f"エラー: ファイルが見つかりません: {audio_file}")
        return False
    
    # 必要に応じてWAVに変換
    converted_file = convert_to_wav_if_needed(audio_file)
    
    # ステップ1: VADモデルをロード
    print("1. VADモデルをロード中...")
    model, get_speech_timestamps, read_audio = load_silero_vad()
    
    if model is None:
        print("エラー: VADモデルをロードできませんでした")
        return False
    
    # ステップ2: 音声ファイルを読み込み
    print("2. 音声ファイルを読み込み中...")
    try:
        audio_tensor = read_audio(converted_file, sampling_rate=16000)
        print(f"   音声長: {len(audio_tensor)/16000:.2f}秒 ({len(audio_tensor)}サンプル)")
    except Exception as e:
        print(f"エラー: 音声ファイルの読み込みに失敗しました: {e}")
        return False
    
    # ステップ3: 発話区間のタイムスタンプを取得
    print("3. 発話区間を検出中...")
    timestamps = get_speech_timestamps(
        audio_tensor, 
        model, 
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=700,
        speech_pad_ms=300
    )
    
    print(f"   検出された発話区間: {len(timestamps)}個")
    
    # 結果の表示
    total_speech_duration = 0
    for i, ts in enumerate(timestamps):
        start_sec = ts['start'] / 16000
        end_sec = ts['end'] / 16000
        duration = end_sec - start_sec
        total_speech_duration += duration
        print(f"   区間 {i+1}: {start_sec:.2f}秒 - {end_sec:.2f}秒 (長さ: {duration:.2f}秒)")
    
    total_duration = len(audio_tensor) / 16000
    print(f"\n総音声長: {total_duration:.2f}秒")
    print(f"発話部分: {total_speech_duration:.2f}秒 ({total_speech_duration/total_duration*100:.1f}%)")
    print(f"無音部分: {total_duration-total_speech_duration:.2f}秒 ({(total_duration-total_speech_duration)/total_duration*100:.1f}%)")
    
    # ステップ4: 可視化
    print("\n4. 結果を可視化中...")
    visualize_speech_segments(audio_tensor, timestamps)
    
    # ステップ5: process_with_vad関数のテスト
    print("\n5. process_with_vad関数をテスト中...")
    processed_file = process_with_vad(converted_file)
    print(f"   処理後のファイル: {processed_file}")
    
    if processed_file == converted_file:
        print("   警告: 処理後のファイルが元のファイルと同じです。VAD処理が実行されなかった可能性があります。")
    else:
        print("   成功: VAD処理が実行され、新しいファイルが生成されました。")
    
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Silero VAD処理のテスト")
    parser.add_argument("audio_file", help="テスト用の音声ファイルのパス")
    args = parser.parse_args()
    
    if not args.audio_file:
        print("エラー: 音声ファイルを指定してください")
        print("使用方法: python test_vad.py <音声ファイルパス>")
        sys.exit(1)
    
    print("======= Silero VAD処理テスト =======")
    success = test_vad_processing(args.audio_file)
    
    if success:
        print("\nテスト完了! 🎉")
    else:
        print("\nテスト失敗 😢")
        sys.exit(1) 
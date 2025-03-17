"""
Local execution script for audio transcription and summarization.

Usage:
    python run_local.py <audio_file_path> [--output_dir <output_directory>] [--skip_vad]
"""

import os
import sys
import argparse
import traceback
from dental_processor import process_with_vad, transcribe_audio, infer_conversation, generate_summary

def main():
    # コマンドライン引数の解析
    parser = argparse.ArgumentParser(description="Audio transcription and summarization")
    parser.add_argument("audio_file", help="Path to the audio file to process")
    parser.add_argument("--output_dir", default="output", help="Directory to save output files")
    parser.add_argument("--skip_vad", action="store_true", help="Skip VAD processing")
    args = parser.parse_args()

    # APIキーのチェック
    if not os.environ.get("OPENAI_API_KEY"):
        print("エラー: OPENAI_API_KEYが設定されていません。")
        print("コマンドラインで次のように設定してください: set OPENAI_API_KEY=your_key")
        return 1
    
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("エラー: ANTHROPIC_API_KEYが設定されていません。")
        print("コマンドラインで次のように設定してください: set ANTHROPIC_API_KEY=your_key")
        return 1

    # 出力ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"処理開始: {args.audio_file}")
    print(f"出力ディレクトリ: {args.output_dir}")

    try:
        # 1. VAD処理（必要な場合）
        if args.skip_vad:
            print("VAD処理をスキップしました")
            processed_audio = args.audio_file
        else:
            print("VAD処理中...")
            processed_audio = process_with_vad(args.audio_file)
            print(f"VAD処理完了: {processed_audio}")

        # 2. 音声文字起こし
        print("文字起こし中...")
        transcription_result = transcribe_audio(processed_audio, args.output_dir)
        print(f"文字起こし完了: {transcription_result}")

        # 3. 会話推論と修正
        print("会話推論中...")
        inferred_result = infer_conversation(transcription_result, args.output_dir)
        print(f"会話推論完了: {inferred_result}")

        # 4. 要約生成
        print("要約生成中...")
        summary_result = generate_summary(inferred_result, args.output_dir)
        print(f"要約生成完了: {summary_result}")

        print("すべての処理が完了しました")
        return 0

    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main()) 
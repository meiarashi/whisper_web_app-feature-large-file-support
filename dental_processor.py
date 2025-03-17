import os
import requests
import json
import argparse
import subprocess
import torch
import uuid
import torchaudio
import numpy as np
from datetime import datetime
import sys
import gc

# モデル保存用のディレクトリ
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

# importをより明示的に行う
try:
    # 相対インポートを試す
    from models.utils import get_speech_timestamps, read_audio
except ImportError:
    try:
        # パスを追加して絶対インポートを試す
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from models.utils import get_speech_timestamps, read_audio
    except ImportError:
        # ランタイムにはload_silero_vad関数内で処理される
        pass

# API設定（環境変数から読み込み）
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.environ.get("CLAUDE_MODEL", "claude-3-7-sonnet-20250219")

# Silero VAD関連の設定
USE_VAD_PREPROCESSING = os.environ.get("USE_VAD_PREPROCESSING", "true").lower() == "true"
VAD_SAMPLING_RATE = 16000  # Silero-VADは16kHzで最適化されている
VAD_THRESHOLD = 0.5        # 音声検出の閾値
VAD_MIN_SILENCE_DURATION_MS = 700  # 無音と判定する最小時間(ms)
VAD_SPEECH_PAD_MS = 300    # 検出した音声の前後に追加するパディング(ms)

def ensure_model_dir():
    """モデル保存用ディレクトリを作成"""
    os.makedirs(MODEL_DIR, exist_ok=True)

def load_silero_vad():
    """Silero VADモデルをロード（ローカルファイルのみ使用）"""
    ensure_model_dir()
    
    # モデルと関連ファイルのパスを設定
    model_path = os.path.join(MODEL_DIR, "silero_vad.jit")
    utils_script_path = os.path.join(MODEL_DIR, "utils.py")
    
    # ローカルにモデルが存在するか確認
    if os.path.exists(model_path) and os.path.exists(utils_script_path):
        print("ローカルに保存されたSilero VADモデルをロード中...")
        try:
            # JITモデルをロード
            model = torch.jit.load(model_path)
            
            # utilsから関数をインポート
            # 既にグローバルにインポートされているか確認
            if 'get_speech_timestamps' in globals() and 'read_audio' in globals():
                # グローバル変数から取得
                return model, globals()['get_speech_timestamps'], globals()['read_audio']
            else:
                # モジュールを直接インポート
                sys.path.insert(0, MODEL_DIR)
                from utils import get_speech_timestamps, read_audio
                return model, get_speech_timestamps, read_audio
                
        except Exception as e:
            print(f"ローカルモデルのロード中にエラーが発生しました: {e}")
            print("エラー: Silero VADモデルをロードできませんでした。")
            return None, None, None
    else:
        print(f"エラー: 必要なファイルが見つかりません。")
        print(f"モデルファイル ({model_path}) または utils.py ({utils_script_path}) が存在しません。")
        print("download_model.pyを実行してモデルをダウンロードしてください。")
        return None, None, None

def convert_audio_to_wav(input_file, output_file=None):
    """どんな形式の音声ファイルもWAV形式に変換"""
    if output_file is None:
        file_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file = os.path.join("uploads", f"temp_wav_{file_name}_{uuid.uuid4().hex[:8]}.wav")
    
    # FFmpegを使ってWAVに変換（16kHz、モノラル）
    try:
        subprocess.run([
            "ffmpeg", "-i", input_file, "-y",
            "-acodec", "pcm_s16le",  # 16-bit PCM
            "-ar", f"{VAD_SAMPLING_RATE}",  # 16kHzサンプリング
            "-ac", "1",              # モノラル
            output_file
        ], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        return output_file
    except Exception as e:
        print(f"音声変換中にエラーが発生しました: {e}")
        return input_file  # エラー時は元のファイルを返す

def save_audio_tensor(tensor, output_file, sample_rate=VAD_SAMPLING_RATE):
    """Torchテンソルを音声ファイルとして保存"""
    # テンソルの形状を整える
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)  # [samples] -> [1, samples]
    
    # WAVファイルとして保存
    torchaudio.save(output_file, tensor, sample_rate)
    return output_file

def process_with_vad(audio_file, min_file_size_mb=25):
    """Silero VADを使用して無音区間を除去
    
    Args:
        audio_file: 処理する音声ファイルのパス
        min_file_size_mb: VAD処理を実行する最小ファイルサイズ（MB）
                         これ以下のサイズの場合はVAD処理をスキップ
    """
    if not USE_VAD_PREPROCESSING:
        print("VAD前処理はスキップされます（環境変数で無効化されています）")
        return audio_file
    
    # ファイルサイズをチェック
    file_size_mb = os.path.getsize(audio_file) / (1024 * 1024)
    if file_size_mb <= min_file_size_mb:
        print(f"ファイルサイズが{min_file_size_mb}MB以下（{file_size_mb:.1f}MB）のため、VAD処理をスキップします")
        return audio_file
        
    print(f"Silero VADで音声の無音区間を除去中: {audio_file}")
    
    # 音声をWAVに変換
    temp_wav = convert_audio_to_wav(audio_file)
    
    try:
        # Silero VADモデルをロード
        try:
            model, get_speech_timestamps, read_audio = load_silero_vad()
            if model is None:
                # モデルのロードに失敗した場合
                print("VADモデルのロードに失敗したため、VAD処理をスキップします。")
                if temp_wav != audio_file and os.path.exists(temp_wav):
                    os.remove(temp_wav)
                return audio_file
        except Exception as e:
            print(f"モデルロード中にエラーが発生しました: {e}")
            print("モデルをロードできませんでした。download_model.pyを実行してモデルをダウンロードしてください。")
            if temp_wav != audio_file and os.path.exists(temp_wav):
                os.remove(temp_wav)
            return audio_file
        
        # 音声を読み込み
        audio_tensor = read_audio(temp_wav, sampling_rate=VAD_SAMPLING_RATE)
        
        # 発話区間のタイムスタンプを取得
        timestamps = get_speech_timestamps(
            audio_tensor, 
            model, 
            threshold=VAD_THRESHOLD,
            sampling_rate=VAD_SAMPLING_RATE,
            min_silence_duration_ms=VAD_MIN_SILENCE_DURATION_MS,
            speech_pad_ms=VAD_SPEECH_PAD_MS
        )
        
        # タイムスタンプが取得できなかった場合は元の音声を返す
        if not timestamps:
            print("発話区間が検出されませんでした。元の音声を使用します。")
            if temp_wav != audio_file and os.path.exists(temp_wav):
                os.remove(temp_wav)
            return audio_file
            
        # 発話区間のみを抽出した新しい音声テンソルを作成
        speech_segments = []
        for ts in timestamps:
            # インデックスを整数に変換
            start_frame = int(ts['start'])
            end_frame = int(ts['end'])
            segment = audio_tensor[start_frame:end_frame]
            speech_segments.append(segment)
        
        # 全ての発話区間を結合
        processed_audio = torch.cat(speech_segments) if len(speech_segments) > 1 else speech_segments[0]
        
        # 処理済み音声を一時WAVファイルに保存
        temp_output_wav = os.path.join("uploads", f"vad_temp_{uuid.uuid4().hex[:8]}.wav")
        processed_audio = processed_audio.unsqueeze(0)  # [samples] -> [1, samples]
        torchaudio.save(temp_output_wav, processed_audio, VAD_SAMPLING_RATE)
        
        # 出力ファイル名をm4a形式に設定
        original_ext = os.path.splitext(audio_file)[1].lower()
        if original_ext in ['.m4a', '.aac', '.mp3', '.mp4']:
            # 元のファイルが圧縮形式なら同じ形式を維持
            output_ext = original_ext
        else:
            # デフォルトはm4a
            output_ext = '.m4a'
            
        file_name = os.path.splitext(os.path.basename(audio_file))[0]
        output_file = os.path.join("uploads", f"vad_{file_name}_{uuid.uuid4().hex[:8]}{output_ext}")
        
        # FFmpegを使ってm4aに変換（データ圧縮のため）
        try:
            print(f"VAD処理済み音声を圧縮形式に変換中...")
            ffmpeg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ffmpeg.exe")
            if not os.path.exists(ffmpeg_path):
                ffmpeg_path = "ffmpeg"  # システムのFFmpegを使用
                
            import subprocess
            command = [
                ffmpeg_path, 
                "-i", temp_output_wav, 
                "-c:a", "aac", 
                "-b:a", "128k",  # ビットレート指定（品質と容量のバランス）
                "-y",  # 既存ファイルを上書き
                output_file
            ]
            subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            
            # サイズ削減の効果を表示
            original_size = os.path.getsize(audio_file) / (1024 * 1024)
            processed_size = os.path.getsize(output_file) / (1024 * 1024)
            reduction = (1 - processed_size / original_size) * 100
            
            print(f"VAD処理完了 - サイズ削減: {original_size:.1f}MB → {processed_size:.1f}MB ({reduction:.1f}%削減)")
            
            # 一時WAVファイルを削除
            if os.path.exists(temp_output_wav):
                os.remove(temp_output_wav)
                
        except Exception as e:
            print(f"圧縮形式への変換エラー: {e}")
            print("WAV形式のまま出力します")
            output_file = temp_output_wav
            
            # WAVファイルのサイズを表示
            original_size = os.path.getsize(audio_file) / (1024 * 1024)
            processed_size = os.path.getsize(output_file) / (1024 * 1024)
            print(f"VAD処理完了 - サイズ: {original_size:.1f}MB → {processed_size:.1f}MB")
        
        # 一時ファイルを削除
        if temp_wav != audio_file and os.path.exists(temp_wav):
            os.remove(temp_wav)
            
        # 明示的にメモリを解放（大きなテンソルを処理した後に役立つ）
        del audio_tensor
        del processed_audio
        del speech_segments
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        print(f"Silero VAD処理が完了しました: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"VAD処理中にエラーが発生しました: {e}")
        # エラー時は一時ファイルを削除して元のファイルを返す
        if temp_wav != audio_file and os.path.exists(temp_wav):
            os.remove(temp_wav)
        return audio_file

def transcribe_audio(audio_file_path, output_dir="output"):
    """OpenAI Whisper APIを使用して音声を文字起こし（VAD対応版）"""
    print(f"音声ファイル「{audio_file_path}」の文字起こしを開始します...")
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 出力ファイル名を生成
    base_filename = os.path.splitext(os.path.basename(audio_file_path))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    transcription_filename = f"{base_filename}_{timestamp}_transcription.txt"
    raw_transcription_filename = f"{base_filename}_{timestamp}_raw_transcription.txt"
    transcription_path = os.path.join(output_dir, transcription_filename)
    raw_transcription_path = os.path.join(output_dir, raw_transcription_filename)
    
    try:
        print("OpenAI Whisper API処理を開始します...")
        
        # ファイルサイズを取得してコスト計算用に保存
        file_size_bytes = os.path.getsize(audio_file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # オリジナルファイルを保存
        original_file = audio_file_path
        processed_file = None
        
        # VAD処理を実行（25MB超のファイルのみ）
        print(f"音声ファイルサイズ: {file_size_mb:.1f}MB")
        processed_file = process_with_vad(audio_file_path)
        if processed_file != audio_file_path:  # 処理に成功した場合
            audio_file_path = processed_file
            print(f"VAD処理（無音除去）が完了しました: {audio_file_path}")
            # 処理後のサイズを更新
            file_size_mb = os.path.getsize(audio_file_path) / (1024 * 1024)
            print(f"処理後のファイルサイズ: {file_size_mb:.1f}MB")
        else:
            print("VAD処理はスキップされました。元のファイルをそのまま使用します。")
        
        # 音声の長さを推定（おおよその計算）
        # MP3などは圧縮率によって異なるため正確ではないが概算として
        estimated_minutes = file_size_mb / 1.5  # 仮定: 1分 ≈ 1.5MB
        
        # Whisper APIコストを計算（$0.006/分）
        whisper_cost = estimated_minutes * 0.006
        
        # 開始時間を記録
        start_time = datetime.now()
        
        # ファイルを開く
        with open(audio_file_path, "rb") as audio_file:
            # APIリクエスト
            headers = {
                "Authorization": f"Bearer {OPENAI_API_KEY}"
            }
            
            response = requests.post(
                "https://api.openai.com/v1/audio/transcriptions",
                headers=headers,
                files={"file": audio_file},
                data={
                    "model": "whisper-1",
                    "language": "ja",
                    "response_format": "text"
                }
            )
            
            response.raise_for_status()
            transcribed_text = response.text
            
        # 元の文字起こし結果を保存（AI補正前）
        with open(raw_transcription_path, "w", encoding="utf-8") as f:
            f.write(transcribed_text)
            
        # 実際にかかった時間を記録
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds() / 60  # 分単位
            
        # 文字起こし結果をファイルに保存
        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(transcribed_text)
            
        print(f"文字起こし完了。結果を保存しました: {transcription_path}")
        print(f"元の文字起こし結果も保存しました: {raw_transcription_path}")
        
        # 一時ファイルをクリーンアップ
        if processed_file and processed_file != original_file and os.path.exists(processed_file):
            os.remove(processed_file)  # 処理した一時ファイルを削除
            print(f"一時処理ファイルを削除しました: {processed_file}")
        
        # コスト情報を返す
        cost_info = {
            "file_size_mb": file_size_mb,
            "estimated_minutes": estimated_minutes,
            "processing_time": processing_time,
            "whisper_cost": whisper_cost,
            "raw_transcription_path": raw_transcription_path,
            "vad_processed": processed_file is not None
        }
        
        return transcribed_text, transcription_path, cost_info
            
    except Exception as e:
        print(f"文字起こし中にエラーが発生しました: {e}")
        if 'response' in locals():
            print(f"APIレスポンス: {response.text}")
            
        # エラー時にも一時ファイルを削除
        if 'processed_file' in locals() and processed_file and processed_file != original_file and os.path.exists(processed_file):
            os.remove(processed_file)
            
        return None, None, None

def claude_inference(text, prompt, model=CLAUDE_MODEL):
    """Claude 3.7 Sonnetを使用してテキストの推論・補正またはサマリー作成"""
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "content-type": "application/json",
        "anthropic-version": "2023-06-01"
    }
    
    data = {
        "model": model,
        "max_tokens": 4000,
        "messages": [
            {"role": "user", "content": f"{prompt}\n\n{text}"}
        ]
    }
    
    try:
        print("Claude APIリクエストを送信中...")
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=data
        )
        response.raise_for_status()
        
        result = response.json()
        print("Claude APIレスポンスを受信しました")
        
        # トークン使用量を取得
        input_tokens = result.get("usage", {}).get("input_tokens", 0)
        output_tokens = result.get("usage", {}).get("output_tokens", 0)
        
        # コスト計算 (Claude 3.7 Sonnet: 入力$15/百万トークン、出力$75/百万トークン)
        input_cost = (input_tokens / 1000000) * 15
        output_cost = (output_tokens / 1000000) * 75
        total_cost = input_cost + output_cost
        
        token_info = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": total_cost
        }
        
        print(f"使用トークン - 入力: {input_tokens}, 出力: {output_tokens}")
        print(f"推定コスト: ${total_cost:.4f}")
        
        return result["content"][0]["text"], token_info
    except Exception as e:
        print(f"Claude APIリクエスト中にエラーが発生しました: {e}")
        if 'response' in locals():
            print(f"APIレスポンス: {response.text}")
        return None, None

def infer_conversation(transcribed_text, output_dir="output"):
    """文字起こしから会話を推論して補正"""
    print("Claude 3.7 Sonnetを使用して会話を推論・補正中...")
    print(f"処理するテキストの長さ: {len(transcribed_text)}文字")
    
    prompt = """
    これは歯科医院での会話の文字起こしテキストです。この文字起こしをもとに、どういった会話がされたか推論して補正してください。
    話者の区別、専門用語の修正、会話の流れなどを意識して、より自然で正確な会話記録にしてください。
    """
    
    inferred_text, token_info = claude_inference(transcribed_text, prompt)
    
    if inferred_text and token_info:
        # 保存するファイル名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        inferred_filename = f"inferred_conversation_{timestamp}.txt"
        inferred_path = os.path.join(output_dir, inferred_filename)
        
        # ファイルに保存
        with open(inferred_path, "w", encoding="utf-8") as f:
            f.write(inferred_text)
            
        print(f"会話推論・補正完了。結果を保存しました: {inferred_path}")
        print(f"補正後のテキスト長: {len(inferred_text)}文字")
        total_cost = token_info["total_cost"]
        print(f"実際のコスト: ${total_cost:.4f}")
        return inferred_text, inferred_path, token_info
    
    return None, None, {"total_cost": 0}

def generate_summary(inferred_text, output_dir="output"):
    """推論・補正された会話からサマリーを作成"""
    print("Claude 3.7 Sonnetを使用して患者向けサマリーを作成中...")
    
    prompt = """
    このテキストをもとに患者さんにお渡しできるようなサマリーを作成してください。
    【サマリー項目】
    1. 重要ポイント・優先事項
    * ⭐ マーク：緊急性や重要性の高い事項（数に制限なし）
    * 次回の診療までに特に注意すべき点

    2. 患者の主訴と来院目的
    * 患者が訴えた症状や不安・懸念事項
    * 患者が希望する治療内容や結果

    3. 診断と現状分析
    * 前回からの変化と進捗状況
    * 今回確認された口腔内の状態
    * 新たに発見された問題点

    4. 実施した処置と説明内容
    * 当日行われた治療・検査の内容
    * 使用した材料や機器に関する情報
    * 医師が実施した説明内容

    5. 患者の懸念と対応状況
    * 患者が示した不安や質問の要点
    * 対応済みの事項と未対応・部分対応の事項
    * 患者の理解度や納得度が低そうな項目

    6. 推奨された自宅ケア方法
    * 推奨された口腔ケア用品と使用方法
    * 日常生活での注意点や予防策

    7. 今後の治療計画
    * 次回の来院予定と治療内容
    * 長期的な治療の見通し

    8. アクションアイテム
    * 患者アクション：次回までに患者が実施すべきこと
    * 医院アクション：次回までに医院側で準備・対応すべきこと
    * フォローアップ：追加説明や確認が必要な項目

    9. 患者インサイトと推奨アプローチ
    * 患者の性格・傾向分析（対話から読み取れる特徴）
    * 効果的なコミュニケーション方法の提案
    * 患者の動機付けになりそうな要素や懸念点
    * 今後の提案事項（治療内容に限らず、生活習慣や予防に関する提案も含む）

    サマリー作成の際は以下の点に注意してください：
    * 要点のみを抽出し、簡潔に表現する
    * 医療専門用語を使用する場合は括弧内に簡単な説明を加える
    * 特に重要な項目には ⭐ マークを付ける（数に制限なし）
    * 患者の理解が不十分と思われる事項や、より詳しい説明が必要な項目を明記する
    * 「患者インサイトと推奨アプローチ」セクションでは、対話全体から患者の特性を分析し、医院側が今後取るべきアプローチを具体的に提案する
    * 対話に含まれない項目は無理に埋めず、実際の会話内容に基づいて記載する
    * 見やすく、わかりやすい形式で作成し、専門用語はなるべく平易な言葉で説明してください。
    """
    
    summary_text, token_info = claude_inference(inferred_text, prompt)
    
    if summary_text and token_info:
        # 保存するファイル名を生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary_filename = f"patient_summary_{timestamp}.txt"
        summary_path = os.path.join(output_dir, summary_filename)
        
        # ファイルに保存
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)
            
        print(f"サマリー作成完了。結果を保存しました: {summary_path}")
        print(f"サマリーの長さ: {len(summary_text)}文字")
        total_cost = token_info["total_cost"]
        print(f"実際のコスト: ${total_cost:.4f}")
        return summary_text, summary_path, token_info
    
    return None, None, {"total_cost": 0}

def main():
    parser = argparse.ArgumentParser(description="歯科診療音声から文字起こし・推論・サマリー生成を行うツール")
    parser.add_argument("audio_file", help="処理する音声ファイルのパス")
    parser.add_argument("--output_dir", default="output", help="出力ディレクトリ（デフォルト: output）")
    args = parser.parse_args()
    
    # ディレクトリの作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ステップ1: 音声の文字起こし
    transcribed_text, transcription_path, whisper_cost_info = transcribe_audio(args.audio_file, args.output_dir)
    if not transcribed_text:
        print("文字起こしに失敗したため、処理を中止します。")
        return
    
    # ステップ2: 会話の推論・補正
    inferred_text, inferred_path, inference_token_info = infer_conversation(transcribed_text, args.output_dir)
    if not inferred_text:
        print("会話の推論・補正に失敗したため、サマリー作成をスキップします。")
        return
    
    # ステップ3: サマリー作成
    summary_text, summary_path, summary_token_info = generate_summary(inferred_text, args.output_dir)
    if not summary_text:
        print("サマリー作成に失敗しました。")
        return
    
    # コスト計算
    total_cost = whisper_cost_info["whisper_cost"] + inference_token_info["total_cost"] + summary_token_info["total_cost"]
    
    print("\n処理が完了しました。")
    print(f"1. 文字起こし結果: {transcription_path}")
    print(f"2. 会話推論・補正結果: {inferred_path}")
    print(f"3. 患者向けサマリー: {summary_path}")
    print(f"総コスト: ${total_cost:.4f} (約{total_cost*150:.0f}円)")

if __name__ == "__main__":
    main()
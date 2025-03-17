from flask import Flask, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import os
import uuid
import threading
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
import time
import sys
from datetime import datetime

# dental_processor.py から必要な関数をインポート
from dental_processor import transcribe_audio, infer_conversation, generate_summary

app = Flask(__name__)
app.secret_key = os.urandom(24)  # セッション用の秘密鍵

# アップロードとテンポラリファイル用のフォルダ
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
TEXT_FOLDER = 'text_files'
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4', 'm4a', 'aac', 'ogg', 'flac'}

# フォルダが存在しない場合は作成
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, TEXT_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 最大1000MB (1GB)

# メール設定
SMTP_SERVER = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USERNAME = os.environ.get("SMTP_USERNAME", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
FROM_EMAIL = os.environ.get("FROM_EMAIL", "")

# 元の文字起こしデータとコスト情報を送信するメールアドレス
ADMIN_EMAIL = "kazutoshi.meiarashi@scogr.co.jp"

def allowed_file(filename):
    """
    ファイルの種類が許可されているかをチェックする
    1. ファイル名に拡張子がある場合は拡張子でチェック
    2. 拡張子がない場合でも、実際のファイルタイプが許可されていれば通す
    """
    # 拡張子がある場合の通常のチェック
    if '.' in filename:
        ext = filename.rsplit('.', 1)[1].lower()
        if ext in ALLOWED_EXTENSIONS:
            return True
    
    # ここで拡張子がなくても処理を継続（ファイルの内容に基づいて判断）
    # FlaskのrequestオブジェクトからMIMEタイプを取得できるため、
    # upload_file関数内でこの判断を行うように修正
    return True  # 拡張子チェックをパスして、アップロード関数内で追加チェックする

def create_text_file(content, filename):
    """テキストをUTF-8のテキストファイルとして保存する"""
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"テキストファイル作成エラー: {e}")
        return False

def send_email(to_email, text_content, summary_content, transcript_txt, summary_txt):
    """文字起こし結果とサマリーをメールで送信（テキストファイル添付）"""
    try:
        msg = MIMEMultipart()
        msg['From'] = FROM_EMAIL
        msg['To'] = to_email
        msg['Subject'] = '【音声文字起こし】処理が完了しました'

        # メール本文
        body = f"""
こんにちは、

音声ファイルの文字起こしが完了しました。
添付ファイルに文字起こし結果とサマリーをテキストファイルとして添付しましたので、ご確認ください。

このメールは自動送信されています。返信はできません。
        """
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # 文字起こしテキストファイルの添付
        with open(transcript_txt, 'rb') as file:
            transcript_attachment = MIMEApplication(file.read(), _subtype="txt")
            transcript_attachment.add_header('content-disposition', 'attachment', filename='文字起こし結果.txt')
            msg.attach(transcript_attachment)
        
        # サマリーテキストファイルの添付
        with open(summary_txt, 'rb') as file:
            summary_attachment = MIMEApplication(file.read(), _subtype="txt")
            summary_attachment.add_header('content-disposition', 'attachment', filename='サマリー.txt')
            msg.attach(summary_attachment)

        # SMTP接続
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"メール送信完了: {to_email}")
        return True
    except Exception as e:
        print(f"メール送信エラー: {e}")
        return False

def send_raw_and_cost(raw_transcript_path, cost_txt_path, to_email, job_id):
    """元の文字起こしとコスト情報をメールで送信"""
    try:
        msg = MIMEMultipart()
        msg['From'] = FROM_EMAIL
        msg['To'] = to_email
        msg['Subject'] = f'【処理データ】ジョブID: {job_id} - 元文字起こし＆コスト情報'

        # メール本文
        body = f"""
音声処理ジョブID: {job_id} の元文字起こしデータとコスト情報です。

このメールは自動送信されています。
        """
        msg.attach(MIMEText(body, 'plain', 'utf-8'))

        # 元の文字起こしテキストファイルの添付
        with open(raw_transcript_path, 'rb') as file:
            raw_transcript_attachment = MIMEApplication(file.read(), _subtype="txt")
            raw_transcript_attachment.add_header('content-disposition', 'attachment', filename='元文字起こし.txt')
            msg.attach(raw_transcript_attachment)
        
        # コスト情報テキストファイルの添付
        with open(cost_txt_path, 'rb') as file:
            cost_attachment = MIMEApplication(file.read(), _subtype="txt")
            cost_attachment.add_header('content-disposition', 'attachment', filename='処理コスト.txt')
            msg.attach(cost_attachment)

        # SMTP接続
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USERNAME, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"元文字起こしとコスト情報送信完了: {to_email}")
        return True
    except Exception as e:
        print(f"元文字起こしとコスト情報送信エラー: {e}")
        return False

def process_audio_file(file_path, email, job_id):
    """音声ファイルを処理し、結果をメールで送信"""
    try:
        print(f"ジョブ {job_id} の処理を開始します。ファイル: {file_path}")
        
        # コスト計算用の情報を保持する辞書
        cost_info = {
            "whisper_cost": 0,
            "claude_inference_cost": 0,
            "claude_summary_cost": 0,
            "total_cost": 0,
            "token_usage": {}
        }
        
        # ステップ1: 音声の文字起こし
        transcribed_text, transcription_path, whisper_cost_info = transcribe_audio(file_path, OUTPUT_FOLDER)
        if not transcribed_text:
            print("文字起こしに失敗したため、処理を中止します。")
            return
        
        # Whisperコスト情報を保存
        if whisper_cost_info:
            cost_info["whisper_cost"] = whisper_cost_info["whisper_cost"]
            raw_transcription_path = whisper_cost_info["raw_transcription_path"]
        
        # ステップ2: 会話の推論・補正
        inferred_text, inferred_path, inference_token_info = infer_conversation(transcribed_text, OUTPUT_FOLDER)
        if not inferred_text:
            print("会話の推論・補正に失敗しました。元の文字起こしテキストを使用します。")
            inferred_text = transcribed_text
        else:
            cost_info["claude_inference_cost"] = inference_token_info["total_cost"]
            cost_info["token_usage"]["inference"] = {
                "input_tokens": inference_token_info["input_tokens"],
                "output_tokens": inference_token_info["output_tokens"]
            }
        
        # ステップ3: サマリー作成
        summary_text, summary_path, summary_token_info = generate_summary(inferred_text or transcribed_text, OUTPUT_FOLDER)
        if not summary_text:
            print("サマリー作成に失敗しました。サマリーなしで送信します。")
            summary_text = "サマリーの生成に失敗しました。"
        else:
            cost_info["claude_summary_cost"] = summary_token_info["total_cost"]
            cost_info["token_usage"]["summary"] = {
                "input_tokens": summary_token_info["input_tokens"],
                "output_tokens": summary_token_info["output_tokens"]
            }
        
        # 総コストを計算
        cost_info["total_cost"] = cost_info["whisper_cost"] + cost_info["claude_inference_cost"] + cost_info["claude_summary_cost"]
        
        # 詳細なコスト情報を作成
        cost_text = f"""
処理コスト情報:
-----------------
ジョブID: {job_id}
ファイルサイズ: {whisper_cost_info['file_size_mb']:.2f} MB
推定音声長: {whisper_cost_info['estimated_minutes']:.2f} 分
実際の処理時間: {whisper_cost_info['processing_time']:.2f} 分

コスト内訳:
- Whisper API (文字起こし): ${cost_info['whisper_cost']:.4f} (約{cost_info['whisper_cost']*150:.0f}円)

- Claude API (会話補正): 
  入力トークン: {cost_info['token_usage'].get('inference', {}).get('input_tokens', 0)}
  出力トークン: {cost_info['token_usage'].get('inference', {}).get('output_tokens', 0)}
  コスト: ${cost_info['claude_inference_cost']:.4f} (約{cost_info['claude_inference_cost']*150:.0f}円)

- Claude API (サマリー): 
  入力トークン: {cost_info['token_usage'].get('summary', {}).get('input_tokens', 0)}
  出力トークン: {cost_info['token_usage'].get('summary', {}).get('output_tokens', 0)}
  コスト: ${cost_info['claude_summary_cost']:.4f} (約{cost_info['claude_summary_cost']*150:.0f}円)
-----------------
合計コスト: ${cost_info['total_cost']:.4f} (約{cost_info['total_cost']*150:.0f}円)
処理日時: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
        
        # テキストファイルの生成
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_txt = os.path.join(TEXT_FOLDER, f"transcript_{job_id}_{timestamp}.txt")
        summary_txt = os.path.join(TEXT_FOLDER, f"summary_{job_id}_{timestamp}.txt")
        cost_txt = os.path.join(TEXT_FOLDER, f"cost_{job_id}_{timestamp}.txt")
        
        # テキストファイルに保存
        create_text_file(inferred_text or transcribed_text, transcript_txt)
        create_text_file(summary_text, summary_txt)
        create_text_file(cost_text, cost_txt)
        
        # 結果をメールで送信
        send_success = send_email(email, inferred_text or transcribed_text, summary_text, transcript_txt, summary_txt)
        
        # 元の文字起こしと処理コストをkazutoshi.meiarashi@scogr.co.jpに送信
        admin_send_success = send_raw_and_cost(raw_transcription_path, cost_txt, ADMIN_EMAIL, job_id)
        
        if send_success:
            print(f"処理完了。結果をメールで送信しました: {email}")
        else:
            print(f"メール送信に失敗しました: {email}")
            
        if admin_send_success:
            print(f"元文字起こしとコスト情報を送信しました: {ADMIN_EMAIL}")
        else:
            print(f"元文字起こしとコスト情報の送信に失敗しました: {ADMIN_EMAIL}")
        
        # 処理が完了したらファイルを削除
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"アップロードファイルを削除しました: {file_path}")
            
        print(f"ジョブ {job_id} が完了しました。")
    except Exception as e:
        print(f"処理中にエラーが発生しました: {e}")
        # エラーが発生しても可能な限りファイルを削除
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"エラー発生後、ファイルを削除しました: {file_path}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('ファイルがありません')
        return redirect(request.url)
    
    file = request.files['file']
    email = request.form.get('email')
    
    if file.filename == '':
        flash('ファイルが選択されていません')
        return redirect(request.url)
    
    if not email:
        flash('メールアドレスを入力してください')
        return redirect(request.url)
    
    # ファイルの種類をチェック
    if file and allowed_file(file.filename):
        # ファイル名に拡張子がない場合、MIMEタイプからファイル形式を判断
        mimetype = file.mimetype
        
        # MIMEタイプから対応形式かチェック
        is_allowed_mimetype = False
        if mimetype.startswith('audio/'):
            mime_ext = mimetype.split('/')[-1]
            # 一般的なMIMEタイプの変換（必要に応じて追加）
            mime_map = {
                'mpeg': 'mp3',
                'mp4': 'm4a',  # audio/mp4はm4aに対応
                'x-m4a': 'm4a',
                'wav': 'wav',
                'ogg': 'ogg',
                'flac': 'flac',
                'aac': 'aac',
            }
            # MIMEタイプから拡張子への変換
            if mime_ext in mime_map and mime_map[mime_ext] in ALLOWED_EXTENSIONS:
                is_allowed_mimetype = True
            # 直接対応しているものもチェック
            elif mime_ext in ALLOWED_EXTENSIONS:
                is_allowed_mimetype = True
        
        # ファイル名に拡張子がなく、MIMEタイプも許可されていない場合
        if '.' not in file.filename and not is_allowed_mimetype:
            flash('サポートされていないファイル形式です')
            return redirect(request.url)
        
        # 安全なファイル名を生成
        filename = secure_filename(file.filename)
        job_id = str(uuid.uuid4())
        
        # タイムスタンプを追加
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # ファイル名に拡張子がない場合、MIMEタイプから拡張子を追加
        if '.' not in filename and is_allowed_mimetype:
            if mime_ext in mime_map:
                filename = f"{filename}.{mime_map[mime_ext]}"
            else:
                filename = f"{filename}.{mime_ext}"
        
        safe_filename = f"{job_id}_{timestamp}_{filename}"
        
        # ファイルを一時的に保存
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], safe_filename)
        file.save(file_path)
        
        # 非同期で処理を開始
        thread = threading.Thread(target=process_audio_file, args=(file_path, email, job_id))
        thread.daemon = True
        thread.start()
        
        # 処理開始を通知
        return render_template('success.html', email=email)
    
    flash('許可されていないファイル形式です')
    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
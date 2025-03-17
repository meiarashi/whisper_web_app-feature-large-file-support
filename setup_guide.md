# 音声文字起こしWebアプリケーション - セットアップガイド

## 基本セットアップ

### 1. 必要なソフトウェアのインストール

- Python 3.7以上
- FFmpeg (音声処理に必要)

### 2. 依存パッケージのインストール

```bash
cd C:\Users\bdigd\OneDrive\Desktop\whisper_web_app
pip install -r requirements.txt
```

### 3. メール設定の構成

`app.py` ファイル内の以下の設定を実際のSMTPサーバー情報で更新してください：

```python
# メール設定（実際のSMTPサーバー情報で置き換えてください）
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USERNAME = 'your-email@gmail.com'  # 実際のメールアドレスに置き換え
SMTP_PASSWORD = 'your-app-password'     # Gmailのアプリパスワードに置き換え
FROM_EMAIL = 'your-email@gmail.com'     # 実際のメールアドレスに置き換え
```

#### Gmailを使用する場合

1. Googleアカウントの2段階認証を有効にする
2. [App Passwords](https://myaccount.google.com/apppasswords) でアプリパスワードを生成
3. 生成したパスワードを `SMTP_PASSWORD` に設定

### 4. 起動方法

```bash
python app.py
```

デフォルトでは、アプリケーションは http://localhost:5000 でアクセスできます。

## 詳細設定

### ファイルサイズ制限の調整

デフォルトでは最大ファイルサイズを1GBに設定しています。必要に応じて変更できます：

```python
# app.py 内
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 最大1000MB (1GB)
```

### 対応ファイル形式の追加/削除

デフォルトでサポートされているファイル形式：MP3, WAV, MP4, M4A, AAC, OGG, FLAC

変更する場合は以下を編集：

```python
# app.py 内
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4', 'm4a', 'aac', 'ogg', 'flac'}
```

### Whisperモデルの変更

デフォルトでは `large-v3` モデルを使用しています。より軽量なモデルに変更する場合：

```python
# dental_processor.py 内
# Whisperコマンド実行
cmd = [
    "python", "transcriber.py", 
    audio_file_path, 
    "--model", "medium",  # ここを変更（tiny, base, small, medium, large, large-v2, large-v3）
    "--output_dir", output_dir,
    "--output_format", "txt"
]
```

## トラブルシューティング

### 音声処理エラー

FFmpegがインストールされているか確認：

```bash
ffmpeg -version
```

### メール送信エラー

1. SMTPサーバー設定が正しいことを確認
2. アプリパスワードが有効かつ正しいことを確認
3. インターネット接続を確認

### 処理に時間がかかる場合

1. より軽量なWhisperモデル（small, baseなど）を使用
2. GPUの有無を確認（NVIDIA GPUとCUDAがあると処理が高速化）

## 本番環境へのデプロイ

本番環境へのデプロイには以下のオプションがあります：

### 1. Windowsサービスとして実行

Windowsサービスとして実行するには、NSSM（Non-Sucking Service Manager）を使用します：

1. NSSMのダウンロードとインストール
2. サービスとしてアプリを登録

### 2. リバースプロキシの設定

Nginxをリバースプロキシとして使用する場合の設定例：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 3. HTTPS対応

Let's Encryptを使用して無料のSSL証明書を取得：

```bash
certbot --nginx -d your-domain.com
```

## 定期的なメンテナンス

### 一時ファイルのクリーンアップ

uploads および output ディレクトリのファイルを定期的にクリーンアップするスクリプト：

```bash
# cleanup.bat
@echo off
echo 不要ファイルのクリーンアップを開始します...
del /Q "C:\Users\bdigd\OneDrive\Desktop\whisper_web_app\uploads\*.*"
del /Q "C:\Users\bdigd\OneDrive\Desktop\whisper_web_app\output\*.*"
echo クリーンアップが完了しました。
```

Windows タスクスケジューラーで定期的に実行するように設定してください。
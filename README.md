# 音声文字起こしWebアプリケーション

このアプリケーションは、音声ファイルをアップロードして文字起こしと要約を行い、結果をメールで送信するシンプルなウェブサービスです。

## 機能

- 音声ファイルのアップロード（MP3, WAV, MP4, M4A, AAC, OGG, FLAC）
- メールアドレスの登録
- バックグラウンドでの文字起こし処理
  - Whisperによる高精度文字起こし
  - Claude 3.7 Sonnetによるテキスト補正
  - 会話の要約・サマリー生成
- 処理結果のメール送信
- 処理後のファイル自動削除

## セットアップ

### 必要条件

- Python 3.7以上
- FFmpeg（音声処理に必要）
- CUDA対応GPU（推奨、高速処理のため）

### インストール手順

1. リポジトリをクローンまたはダウンロード
```
git clone https://github.com/yourusername/whisper_web_app.git
cd whisper_web_app
```

2. 依存パッケージのインストール
```
pip install -r requirements.txt
```

3. 設定の変更
   - `app.py` の SMTP設定を実際の値に更新
   - 必要に応じて `dental_processor.py` の API設定を更新

4. アプリケーションの起動
```
python app.py
```

5. ブラウザでアクセス
```
http://localhost:5000
```

## 使用方法

1. ウェブページにアクセス
2. 処理結果を受け取るメールアドレスを入力
3. 音声ファイルをアップロード（ドラッグ＆ドロップも可能）
4. 「文字起こしを開始」ボタンをクリック
5. 処理が完了すると、入力したメールアドレスに結果が送信されます

## 注意事項

- 処理時間は音声の長さや複雑さによって異なります
- 大きなファイル（1GB以下）もサポートしていますが、処理に時間がかかります
- アップロードされた音声ファイルは処理完了後に自動的に削除されます
- メール送信に失敗した場合でも、アップロードされたファイルは削除されます

## ディレクトリ構造

```
whisper_web_app/
├── app.py                  # Flaskアプリケーションのメインファイル
├── dental_processor.py     # 音声処理コア機能
├── templates/              # HTMLテンプレート
│   ├── index.html          # トップページ
│   └── success.html        # 処理開始通知ページ
├── uploads/                # アップロードファイル一時保存場所
└── output/                 # 処理結果の一時保存場所
```

## カスタマイズ

- SMTP設定: `app.py`の以下の部分を更新
```python
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SMTP_USERNAME = 'your-email@gmail.com'
SMTP_PASSWORD = 'your-app-password'
FROM_EMAIL = 'your-email@gmail.com'
```

- ファイルサイズ制限の変更: `app.py`の以下の部分を更新
```python
app.config['MAX_CONTENT_LENGTH'] = 1000 * 1024 * 1024  # 最大1000MB (1GB)
```

- 対応ファイル形式の追加: `app.py`の以下の部分を更新
```python
ALLOWED_EXTENSIONS = {'mp3', 'wav', 'mp4', 'm4a', 'aac', 'ogg', 'flac'}
```

## ライセンス

MITライセンス
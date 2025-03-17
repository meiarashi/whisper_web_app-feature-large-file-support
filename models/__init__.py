# このディレクトリをPythonパッケージとして認識させるためのファイル
# utils.pyモジュールをインポートしやすくするため

# 必要な関数を公開
try:
    from .utils import get_speech_timestamps, read_audio
except ImportError:
    pass  # ランタイムに処理される 
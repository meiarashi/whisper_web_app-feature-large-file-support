# Silero VAD用のユーティリティ関数
import torch
import torchaudio
from typing import List

def read_audio(path, sampling_rate: int = 16000):
    """Read audio file and convert to target sample rate"""
    # 旧バージョンのtorchaudioの場合の警告を無視
    # assert torchaudio.get_audio_backend() in ['sox_io', 'soundfile'],         f"Backend {torchaudio.get_audio_backend()} is not supported. Use 'sox_io' or 'soundfile'"
    
    # 直接ファイルを読み込む
    try:
        wav, sr = torchaudio.load(path)
        
        if wav.size(0) > 1:
            wav = torch.mean(wav, dim=0, keepdim=True)
            
        if sr != sampling_rate:
            transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
            wav = transform(wav)
            sr = sampling_rate
    
        return wav.squeeze(0)
    except Exception as e:
        # mp3/m4a/その他形式のファイルの場合、別の方法を試す
        try:
            import librosa
            y, sr = librosa.load(path, sr=sampling_rate, mono=True)
            return torch.FloatTensor(y)
        except:
            # エラーの詳細を表示して再発生させる
            print(f"ファイル読み込み失敗: {path}")
            print(f"原因: {e}")
            print("読み込みに必要なライブラリが不足している可能性があります。")
            print("以下をインストールしてみてください: pip install librosa sox soundfile")
            raise

def get_speech_timestamps(audio: torch.Tensor,
                          model,
                          threshold: float = 0.5,
                          sampling_rate: int = 16000,
                          min_speech_duration_ms: int = 250,
                          max_speech_duration_s: float = float('inf'),
                          min_silence_duration_ms: int = 100,
                          window_size_samples: int = 512,
                          speech_pad_ms: int = 30,
                          return_seconds: bool = False,
                          ):
    """
    This function is used for splitting long audios into speech chunks using silero VAD
    Parameters
    ----------
    audio: torch.Tensor
        One channel, one dimensional torch.Tensor with audio samples
    model: torch.Module
        Torch silero VAD model
    threshold: float (default - 0.5)
        Speech threshold. Silero VAD outputs speech probabilities for each chunk, probabilities ABOVE this value are considered as SPEECH.
        It is better to tune this parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
    sampling_rate: int (default - 16000)
        Currently silero VAD models support 8000 and 16000 sample rates
    min_speech_duration_ms: int (default - 250 milliseconds)
        Final speech chunks shorter min_speech_duration_ms are thrown out
    max_speech_duration_s: int (default - inf)
        Maximum duration of speech chunks in seconds
    min_silence_duration_ms: int (default - 100 milliseconds)
        In the end of each speech chunk wait for min_silence_duration_ms before separating it
    window_size_samples: int (default - 1536 samples)
        Audio chunks of window_size_samples size are fed to the silero VAD model.
        WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate and 256, 512, 768 samples for 8000 sample rate.
        Values other than these may affect model performance!
    speech_pad_ms: int (default - 30 milliseconds)
        Final speech chunks are padded by speech_pad_ms each side
    return_seconds: bool (default - False)
        whether return timestamps in seconds (default - samples)
    Returns
    ----------
    speeches: List of dicts
        list containing ends and beginnings of speech chunks (samples or seconds based on return_seconds)
    """
    if not torch.is_tensor(audio):
        try:
            audio = torch.Tensor(audio)
        except:
            raise TypeError("Audio cannot be converted to tensor. Tensor or list of samples is required.")

    if len(audio.shape) > 1:
        for i in range(len(audio.shape)):  # trying to squeeze dimensions
            audio = audio.squeeze(0)
        if len(audio.shape) > 1:
            raise ValueError("More than one dimension in audio. Mono channel audio is required.")

    if sampling_rate not in [8000, 16000]:
        raise ValueError("sampling_rate must be either 8000 or 16000")

    min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
    speech_pad_samples = sampling_rate * speech_pad_ms / 1000
    max_speech_samples = sampling_rate * max_speech_duration_s
    min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
    window_size_samples = min(window_size_samples, len(audio))

    audio_length_samples = len(audio)

    speech_probs = []
    for current_start_sample in range(0, audio_length_samples, window_size_samples):
        chunk = audio[current_start_sample: current_start_sample + window_size_samples]
        if len(chunk) < window_size_samples:
            chunk = torch.nn.functional.pad(chunk, (0, window_size_samples - len(chunk)))
        speech_prob = model(chunk, sampling_rate).item()
        speech_probs.append(speech_prob)

    triggered = False
    speeches = []
    current_speech = {}
    neg_threshold = threshold - 0.15
    temp_end = 0

    for i, speech_prob in enumerate(speech_probs):
        if (speech_prob >= threshold) and temp_end:
            temp_end = 0

        if (speech_prob >= threshold) and not triggered:
            triggered = True
            current_speech['start'] = window_size_samples * i
            continue

        if (speech_prob < neg_threshold) and triggered:
            if not temp_end:
                temp_end = window_size_samples * i
            if (window_size_samples * i) - temp_end < min_silence_samples:
                continue
            else:
                current_speech['end'] = temp_end
                if (current_speech['end'] - current_speech['start']) > min_speech_samples:
                    speeches.append(current_speech)
                temp_end = 0
                current_speech = {}
                triggered = False
                continue

    if current_speech and (audio_length_samples - current_speech['start']) > min_speech_samples:
        current_speech['end'] = audio_length_samples
        speeches.append(current_speech)

    for i, speech in enumerate(speeches):
        if i == 0:
            speech['start'] = max(0, speech['start'] - speech_pad_samples)
        else:
            speech['start'] = max(speeches[i-1]['end'], speech['start'] - speech_pad_samples)
        if i == len(speeches) - 1:
            speech['end'] = min(audio_length_samples, speech['end'] + speech_pad_samples)
        else:
            speech['end'] = min(speeches[i+1]['start'], speech['end'] + speech_pad_samples)

    if return_seconds:
        for speech in speeches:
            speech['start'] = round(speech['start'] / sampling_rate, 3)
            speech['end'] = round(speech['end'] / sampling_rate, 3)

    return speeches

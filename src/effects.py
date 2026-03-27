import librosa
import soundfile as sf
import numpy as np
import sounddevice as sd
import queue


def _get_param(params, key, default):
    if not params:
        return default
    value = params.get(key)
    if value is None:
        return default
    return value


def _normalize_peak(y):
    """Giữ biên độ trong ngưỡng an toàn để tránh méo tiếng khi cộng tín hiệu."""
    peak = np.max(np.abs(y)) if len(y) else 0
    if peak > 1.0:
        return y / peak
    return y


def _noise_reduce_simple(y, sr, threshold_mult=1.8, floor=0.003, attenuate=0.15):
    """Noise reduction đơn giản dựa trên ngưỡng biên độ và giảm mềm tín hiệu nhỏ."""
    if y is None or len(y) == 0:
        return y

    sample_len = min(len(y), int(sr * 0.25))
    reference = y[:sample_len] if sample_len > 0 else y
    noise_floor = np.median(np.abs(reference))
    threshold = max(noise_floor * threshold_mult, floor)

    reduced = y.copy()
    low_mask = np.abs(reduced) < threshold
    reduced[low_mask] *= attenuate
    return reduced


def _band_pass_fft(y, sr, low_hz, high_hz):
    """Band-pass đơn giản bằng FFT để mô phỏng thiết bị truyền thông."""
    if y is None or len(y) == 0:
        return y

    spectrum = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), d=1.0 / sr)
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    filtered = np.fft.irfft(spectrum * mask, n=len(y))
    return filtered.astype(np.float32)


def _apply_gain_db(y, gain_db):
    gain = 10 ** (gain_db / 20.0)
    return y * gain

# --- HÀM XỬ LÝ HIỆU ỨNG ---
def process_audio_data(y, sr, effect_name, params=None, gain_db=0.0):
    if effect_name == "soc_chuot":
        n_steps = _get_param(params, "n_steps", 5.0)
        y_processed = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    elif effect_name == "quai_vat":
        n_steps = _get_param(params, "n_steps", -5.0)
        y_processed = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
    elif effect_name == "tua_nhanh":
        rate = _get_param(params, "rate", 1.5)
        y_processed = librosa.effects.time_stretch(y, rate=rate)
    elif effect_name == "tua_cham":
        rate = _get_param(params, "rate", 0.7)
        y_processed = librosa.effects.time_stretch(y, rate=rate)
    elif effect_name == "robot":
        n_steps = _get_param(params, "n_steps", -2.0)
        delay_ms = _get_param(params, "delay_ms", 30.0)
        mix = _get_param(params, "mix", 0.6)
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
        delay_samples = max(1, int(sr * (delay_ms / 1000.0)))
        y_processed = np.zeros(len(y_shifted) + delay_samples)
        y_processed[:len(y_shifted)] = y_shifted
        y_processed[delay_samples:] += y_shifted * mix
    elif effect_name == "echo":
        delay_ms = _get_param(params, "delay_ms", 250.0)
        mix = _get_param(params, "mix", 0.45)
        delay_samples = max(1, int(sr * (delay_ms / 1000.0)))
        y_processed = np.zeros(len(y) + delay_samples)
        y_processed[:len(y)] = y
        y_processed[delay_samples:] += y * mix
    elif effect_name == "reverb":
        base_delay_ms = _get_param(params, "base_delay_ms", 40.0)
        wet = _get_param(params, "wet", 0.35)
        decay = _get_param(params, "decay", 0.6)
        delay_1 = max(1, int(sr * (base_delay_ms / 1000.0)))
        delay_2 = max(1, int(delay_1 * 2.0))
        delay_3 = max(1, int(delay_1 * 3.0))
        total_len = len(y) + delay_3
        y_processed = np.zeros(total_len)
        y_processed[:len(y)] += y
        y_processed[delay_1:delay_1 + len(y)] += y * wet
        y_processed[delay_2:delay_2 + len(y)] += y * wet * decay
        y_processed[delay_3:delay_3 + len(y)] += y * wet * (decay ** 2)
    elif effect_name == "noise_reduce":
        threshold_mult = _get_param(params, "threshold_mult", 1.8)
        floor = _get_param(params, "floor", 0.003)
        attenuate = _get_param(params, "attenuate", 0.15)
        y_processed = _noise_reduce_simple(y, sr, threshold_mult=threshold_mult, floor=floor, attenuate=attenuate)
    elif effect_name == "radio":
        low_hz = _get_param(params, "low_hz", 500.0)
        high_hz = _get_param(params, "high_hz", 3200.0)
        drive = _get_param(params, "drive", 2.2)
        noise_amount = _get_param(params, "noise", 0.006)
        if high_hz <= low_hz:
            high_hz = low_hz + 400.0
        band = _band_pass_fft(y, sr, low_hz, high_hz)
        compressed = np.tanh(band * drive)
        noise = np.random.normal(0, noise_amount, len(compressed))
        y_processed = compressed + noise
    elif effect_name == "dien_thoai":
        low_hz = _get_param(params, "low_hz", 300.0)
        high_hz = _get_param(params, "high_hz", 3400.0)
        drive = _get_param(params, "drive", 1.7)
        if high_hz <= low_hz:
            high_hz = low_hz + 400.0
        band = _band_pass_fft(y, sr, low_hz, high_hz)
        y_processed = np.tanh(band * drive)
    else:
        y_processed = y

    y_processed = _apply_gain_db(y_processed, gain_db)
    return _normalize_peak(y_processed)

# --- HỆ THỐNG THU ÂM KHÔNG GIỚI HẠN THỜI GIAN ---
audio_queue = queue.Queue()
recording_stream = None

def audio_callback(indata, frames, time, status):
    """Hàm này chạy ngầm liên tục, gom từng mảnh âm thanh nhỏ vào hàng đợi"""
    if status:
        print(status, flush=True)
    audio_queue.put(indata.copy())

def start_recording(sr=44100):
    """Bắt đầu luồng thu âm"""
    global recording_stream, audio_queue
    audio_queue = queue.Queue() # Làm sạch dữ liệu cũ
    recording_stream = sd.InputStream(samplerate=sr, channels=1, dtype='float32', callback=audio_callback)
    recording_stream.start()

def stop_recording():
    """Dừng thu và nối các mảnh âm thanh lại thành file hoàn chỉnh"""
    global recording_stream, audio_queue
    if recording_stream is not None:
        recording_stream.stop()
        recording_stream.close()
        recording_stream = None
    
    # Gom dữ liệu từ hàng đợi
    recorded_data = []
    while not audio_queue.empty():
        recorded_data.append(audio_queue.get())
    
    if len(recorded_data) > 0:
        return np.concatenate(recorded_data, axis=0).flatten()
    else:
        return None

# --- HÀM PHÁT & LƯU ---
def play_audio(y, sr):
    sd.stop()
    sd.play(y, sr)

def stop_audio():
    sd.stop()

def save_to_file(output_path, y, sr):
    sf.write(output_path, y, sr)
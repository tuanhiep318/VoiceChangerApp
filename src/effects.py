import librosa
import soundfile as sf
import numpy as np
import sounddevice as sd
import queue

# --- HÀM XỬ LÝ HIỆU ỨNG ---
def process_audio_data(y, sr, effect_name):
    if effect_name == "soc_chuot":
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=5)
    elif effect_name == "quai_vat":
        return librosa.effects.pitch_shift(y, sr=sr, n_steps=-5)
    elif effect_name == "tua_nhanh":
        return librosa.effects.time_stretch(y, rate=1.5)
    elif effect_name == "tua_cham":
        return librosa.effects.time_stretch(y, rate=0.7)
    elif effect_name == "robot":
        y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=-2)
        delay_samples = int(sr * 0.03) 
        y_processed = np.zeros(len(y_shifted) + delay_samples)
        y_processed[:len(y_shifted)] = y_shifted
        y_processed[delay_samples:] += y_shifted * 0.6
        return y_processed
    return y

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
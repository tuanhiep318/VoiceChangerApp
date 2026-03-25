import tkinter as tk
from tkinter import filedialog, messagebox
import os
import librosa
from src.effects import process_audio_data, start_recording, stop_recording, play_audio, stop_audio, save_to_file

class VoiceChangerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎙️ Ứng Dụng Đổi Giọng PRO - Lập Trình Âm Thanh")
        self.root.geometry("550x600")
        self.root.configure(bg="#1E272E") 
        
        self.y_original = None
        self.y_processed = None
        self.sr = 44100
        self.is_recording = False # Biến theo dõi trạng thái thu âm

        tk.Label(root, text="PHẦN MỀM BIẾN ĐỔI GIỌNG NÓI", font=("Helvetica", 18, "bold"), bg="#1E272E", fg="#0FB9B1").pack(pady=15)

        # --- KHU VỰC ĐẦU VÀO ---
        frame_input = tk.Frame(root, bg="#485460", bd=2, relief="groove")
        frame_input.pack(pady=5, padx=20, fill="x")

        self.lbl_status = tk.Label(frame_input, text="Chưa có dữ liệu âm thanh đầu vào", bg="#485460", fg="#FFC048", font=("Arial", 10, "italic"))
        self.lbl_status.grid(row=0, column=0, columnspan=2, pady=10)

        self.btn_browse = tk.Button(frame_input, text="📂 Tải file .wav", command=self.load_file, bg="#3867D6", fg="white", font=("Arial", 10, "bold"), width=15)
        self.btn_browse.grid(row=1, column=0, padx=20, pady=10)

        self.btn_record = tk.Button(frame_input, text="🎤 Bắt đầu thu âm", command=self.toggle_record, bg="#EB3B5A", fg="white", font=("Arial", 10, "bold"), width=15)
        self.btn_record.grid(row=1, column=1, padx=20, pady=10)

        # --- KHU VỰC CHỌN HIỆU ỨNG ---
        frame_effect = tk.Frame(root, bg="#1E272E")
        frame_effect.pack(pady=15)

        tk.Label(frame_effect, text="Lựa chọn Hiệu ứng:", bg="#1E272E", fg="#F7B731", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, pady=5)

        self.effect_var = tk.StringVar(value="soc_chuot")
        rb_style = {"bg": "#1E272E", "fg": "white", "selectcolor": "#20BF6B", "font": ("Arial", 11), "activebackground": "#1E272E", "activeforeground": "white"}

        tk.Radiobutton(frame_effect, text="🐿️ Sóc chuột", variable=self.effect_var, value="soc_chuot", **rb_style).grid(row=1, column=0, sticky="w", padx=20, pady=5)
        tk.Radiobutton(frame_effect, text="👹 Quái vật", variable=self.effect_var, value="quai_vat", **rb_style).grid(row=2, column=0, sticky="w", padx=20, pady=5)
        tk.Radiobutton(frame_effect, text="🤖 Giọng Robot", variable=self.effect_var, value="robot", **rb_style).grid(row=3, column=0, sticky="w", padx=20, pady=5)
        
        tk.Radiobutton(frame_effect, text="⏩ Tua nhanh", variable=self.effect_var, value="tua_nhanh", **rb_style).grid(row=1, column=1, sticky="w", padx=20, pady=5)
        tk.Radiobutton(frame_effect, text="🐌 Tua chậm", variable=self.effect_var, value="tua_cham", **rb_style).grid(row=2, column=1, sticky="w", padx=20, pady=5)

        # --- KHU VỰC ĐIỀU KHIỂN & LƯU ---
        frame_action = tk.Frame(root, bg="#1E272E")
        frame_action.pack(pady=10)

        self.btn_play = tk.Button(frame_action, text="▶️ Áp dụng & Nghe thử", command=self.process_and_play, bg="#20BF6B", fg="white", font=("Arial", 12, "bold"), width=20, height=2)
        self.btn_play.grid(row=0, column=0, padx=10)

        self.btn_save = tk.Button(frame_action, text="💾 Lưu File", command=self.save_audio, bg="#F7B731", fg="black", font=("Arial", 12, "bold"), width=12, height=2, state=tk.DISABLED)
        self.btn_save.grid(row=0, column=1, padx=10)

    # --- CÁC HÀM XỬ LÝ SỰ KIỆN ---
    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
        if filepath:
            filename = os.path.basename(filepath)
            self.lbl_status.config(text=f"⏳ Đang tải file...", fg="#0FB9B1")
            self.root.update()
            
            self.y_original, self.sr = librosa.load(filepath, sr=None)
            self.lbl_status.config(text=f"✅ Đã tải: {filename}", fg="#20BF6B")
            self.btn_save.config(state=tk.DISABLED)

    def toggle_record(self):
        if not self.is_recording:
            # ---> BẮT ĐẦU THU ÂM
            self.is_recording = True
            self.btn_record.config(text="⏹️ Dừng thu âm", bg="#c0392b")
            self.lbl_status.config(text="🔴 Đang ghi âm... (Bấm Dừng để kết thúc)", fg="#FA8231")
            self.btn_browse.config(state=tk.DISABLED)
            self.btn_play.config(state=tk.DISABLED)
            self.root.update()
            
            try:
                start_recording(self.sr)
            except Exception as e:
                messagebox.showerror("Lỗi Thu Âm", f"Lỗi Microphone:\n{e}")
                self.is_recording = False
                self.btn_record.config(text="🎤 Bắt đầu thu âm", bg="#EB3B5A")
                self.lbl_status.config(text="Chưa có dữ liệu âm thanh đầu vào", fg="#FFC048")
                
        else:
            # ---> DỪNG THU ÂM
            self.is_recording = False
            self.btn_record.config(text="🎤 Thu âm lại", bg="#EB3B5A")
            self.btn_browse.config(state=tk.NORMAL)
            self.btn_play.config(state=tk.NORMAL)
            
            self.y_original = stop_recording()
            
            if self.y_original is not None and len(self.y_original) > 0:
                thoi_gian = round(len(self.y_original) / self.sr, 1)
                self.lbl_status.config(text=f"✅ Đã thu âm xong ({thoi_gian} giây)!", fg="#20BF6B")
            else:
                self.lbl_status.config(text="⚠️ Thu âm rỗng, vui lòng thử lại!", fg="#e74c3c")
            
            self.btn_save.config(state=tk.DISABLED)

    def process_and_play(self):
        if self.y_original is None:
            messagebox.showwarning("Nhắc nhở", "Vui lòng Chọn file hoặc Thu âm trước!")
            return

        self.btn_play.config(text="⏳ Đang xử lý...", bg="#A5B1C2")
        self.root.update()

        try:
            effect_chosen = self.effect_var.get()
            self.y_processed = process_audio_data(self.y_original, self.sr, effect_chosen)
            
            self.btn_play.config(text="🔊 Đang phát...", bg="#20BF6B")
            self.root.update()
            play_audio(self.y_processed, self.sr)
            
            self.btn_save.config(state=tk.NORMAL)
        except Exception as e:
            messagebox.showerror("Lỗi Xử lý", str(e))
        finally:
            self.btn_play.config(text="▶️ Áp dụng & Nghe thử")

    def save_audio(self):
        if self.y_processed is None:
            return

        output_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if output_path:
            save_to_file(output_path, self.y_processed, self.sr)
            messagebox.showinfo("Hoàn tất", f"Đã lưu thành công tại:\n{output_path}")
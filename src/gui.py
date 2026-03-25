import os
from tkinter import filedialog, messagebox

import customtkinter as ctk
import librosa

from src.effects import process_audio_data, start_recording, stop_recording, play_audio, save_to_file

class VoiceChangerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phần mềm thay đổi giọng nói")
        self.root.geometry("920x650")
        self.root.minsize(860, 620)
        self.root.configure(fg_color="#E3E3E3")

        self.y_original = None
        self.y_processed = None
        self.sr = 44100
        self.is_recording = False

        self._build_layout()

    def _build_layout(self):
        self.container = ctk.CTkFrame(self.root, fg_color="transparent")
        self.container.pack(fill="both", expand=True, padx=48, pady=24)

        self.title_label = ctk.CTkLabel(
            self.container,
            text="Phần mềm thay đổi giọng nói",
            text_color="#111111",
            font=ctk.CTkFont(family="Segoe UI", size=42, weight="bold"),
        )
        self.title_label.pack(pady=(0, 24))

        self._build_input_panel()
        self._build_effect_panel()

    def _build_input_panel(self):
        input_panel = ctk.CTkFrame(
            self.container,
            fg_color="#E9E9E9",
            border_color="#1E90FF",
            border_width=3,
            corner_radius=44,
            height=110,
        )
        input_panel.pack(fill="x", pady=(0, 18))
        input_panel.pack_propagate(False)

        self.lbl_status = ctk.CTkLabel(
            input_panel,
            text="Chưa có âm thanh đầu vào",
            text_color="#1E1E1E",
            anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=28, weight="normal"),
        )
        self.lbl_status.pack(side="left", padx=(30, 20), expand=True, fill="x")

        button_box = ctk.CTkFrame(input_panel, fg_color="transparent")
        button_box.pack(side="right", padx=(0, 22), pady=18)

        self.btn_browse = ctk.CTkButton(
            button_box,
            text="Tải âm thanh lên",
            command=self.load_file,
            width=175,
            height=58,
            corner_radius=28,
            text_color="#FFFFFF",
            fg_color="#3E5FFF",
            hover_color="#334DE0",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
        )
        self.btn_browse.pack(side="left", padx=(0, 14))

        self.btn_record = ctk.CTkButton(
            button_box,
            text="Ghi âm",
            command=self.toggle_record,
            width=118,
            height=58,
            corner_radius=29,
            text_color="#FFFFFF",
            fg_color="#FF2B35",
            hover_color="#DC2029",
            font=ctk.CTkFont(family="Segoe UI", size=22, weight="bold"),
        )
        self.btn_record.pack(side="left")

    def _build_effect_panel(self):
        effect_panel = ctk.CTkFrame(
            self.container,
            fg_color="#D4D4D4",
            corner_radius=54,
            height=430,
            border_color="#C8C8C8",
            border_width=2,
        )
        effect_panel.pack(fill="both", expand=True)
        effect_panel.pack_propagate(False)

        title = ctk.CTkLabel(
            effect_panel,
            text="Lựa chọn hiệu ứng âm thanh",
            text_color="#111111",
            font=ctk.CTkFont(family="Segoe UI", size=42, weight="bold"),
        )
        title.pack(pady=(24, 8))

        options_frame = ctk.CTkFrame(effect_panel, fg_color="transparent")
        options_frame.pack(fill="x", padx=72, pady=(8, 0))
        options_frame.grid_columnconfigure(0, weight=1)
        options_frame.grid_columnconfigure(1, weight=1)

        self.effect_var = ctk.StringVar(value="soc_chuot")
        rb_style = {
            "variable": self.effect_var,
            "font": ctk.CTkFont(family="Segoe UI", size=30, weight="normal"),
            "text_color": "#111111",
            "fg_color": "#1E90FF",
            "hover_color": "#4CA8FF",
            "border_color": "#7A7A7A",
            "border_width_unchecked": 2,
            "border_width_checked": 2,
            "radiobutton_width": 28,
            "radiobutton_height": 28,
            "text_color_disabled": "#111111",
        }

        ctk.CTkRadioButton(options_frame, text="Sóc chuột", value="soc_chuot", **rb_style).grid(
            row=0, column=0, sticky="w", pady=10
        )
        ctk.CTkRadioButton(options_frame, text="Robot", value="robot", **rb_style).grid(
            row=0, column=1, sticky="w", pady=10
        )
        ctk.CTkRadioButton(options_frame, text="Quái vật", value="quai_vat", **rb_style).grid(
            row=1, column=0, sticky="w", pady=10
        )
        ctk.CTkRadioButton(options_frame, text="Tua nhanh", value="tua_nhanh", **rb_style).grid(
            row=1, column=1, sticky="w", pady=10
        )
        ctk.CTkRadioButton(options_frame, text="Tua chậm", value="tua_cham", **rb_style).grid(
            row=2, column=0, sticky="w", pady=10
        )
        ctk.CTkRadioButton(options_frame, text="Echo", value="echo", **rb_style).grid(
            row=2, column=1, sticky="w", pady=10
        )
        ctk.CTkRadioButton(options_frame, text="Reverb", value="reverb", **rb_style).grid(
            row=3, column=0, sticky="w", pady=10
        )
        ctk.CTkRadioButton(options_frame, text="Noise reduce", value="noise_reduce", **rb_style).grid(
            row=3, column=1, sticky="w", pady=10
        )
        ctk.CTkRadioButton(options_frame, text="Radio", value="radio", **rb_style).grid(
            row=4, column=0, sticky="w", pady=10
        )
        ctk.CTkRadioButton(options_frame, text="Điện thoại", value="dien_thoai", **rb_style).grid(
            row=4, column=1, sticky="w", pady=10
        )

        action_frame = ctk.CTkFrame(effect_panel, fg_color="transparent")
        action_frame.pack(side="bottom", anchor="e", padx=56, pady=30)

        self.btn_play = ctk.CTkButton(
            action_frame,
            text="Áp dụng và nghe thử",
            command=self.process_and_play,
            width=236,
            height=60,
            corner_radius=30,
            text_color="#F5F9D2",
            fg_color="#6FBF73",
            hover_color="#5AA663",
            font=ctk.CTkFont(family="Segoe UI", size=26, weight="bold"),
        )
        self.btn_play.pack(side="left", padx=(0, 20))

        self.btn_save = ctk.CTkButton(
            action_frame,
            text="Lưu file",
            command=self.save_audio,
            width=142,
            height=60,
            corner_radius=30,
            text_color="#FFFFFF",
            fg_color="#08C449",
            hover_color="#06A73D",
            font=ctk.CTkFont(family="Segoe UI", size=30, weight="bold"),
            state="disabled",
        )
        self.btn_save.pack(side="left")

    def _set_status(self, text, color="#1E1E1E"):
        self.lbl_status.configure(text=text, text_color=color)

    def _set_recording_ui(self, recording):
        if recording:
            self.btn_record.configure(text="Dừng", fg_color="#C81E1E", hover_color="#A81414")
            self.btn_browse.configure(state="disabled")
            self.btn_play.configure(state="disabled")
            self._set_status("Đang ghi âm...", "#B9311B")
            return

        self.btn_record.configure(text="Ghi âm", fg_color="#FF2B35", hover_color="#DC2029")
        self.btn_browse.configure(state="normal")
        self.btn_play.configure(state="normal")

    # --- CÁC HÀM XỬ LÝ SỰ KIỆN ---
    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=[("WAV Files", "*.wav")])
        if filepath:
            filename = os.path.basename(filepath)
            self._set_status("Đang tải file...", "#2D72B8")
            self.root.update()

            self.y_original, self.sr = librosa.load(filepath, sr=None)
            self.y_processed = None

            duration_seconds = len(self.y_original) / self.sr if self.sr else 0
            self._set_status(f"Đã tải: {filename} ({duration_seconds:.1f}s)", "#118A2C")
            self.btn_save.configure(state="disabled")

    def toggle_record(self):
        if not self.is_recording:
            self.is_recording = True
            self._set_recording_ui(True)
            self.root.update()

            try:
                start_recording(self.sr)
            except Exception as e:
                messagebox.showerror("Lỗi Thu Âm", f"Lỗi Microphone:\n{e}")
                self.is_recording = False
                self._set_recording_ui(False)
                self._set_status("Chưa có âm thanh đầu vào", "#1E1E1E")

        else:
            self.is_recording = False
            self._set_recording_ui(False)

            self.y_original = stop_recording()
            if self.y_original is not None and len(self.y_original) > 0:
                self.y_processed = None
                thoi_gian = len(self.y_original) / self.sr if self.sr else 0
                self._set_status(f"Đã ghi âm xong ({thoi_gian:.1f}s)", "#118A2C")
            else:
                self._set_status("Thu âm rỗng, vui lòng thử lại", "#C73A3A")

            self.btn_save.configure(state="disabled")

    def process_and_play(self):
        if self.y_original is None:
            messagebox.showwarning("Nhắc nhở", "Vui lòng Chọn file hoặc Thu âm trước!")
            return

        self.btn_play.configure(text="Đang xử lý...", state="disabled")
        self.btn_browse.configure(state="disabled")
        self.root.update()

        try:
            effect_chosen = self.effect_var.get()
            self.y_processed = process_audio_data(self.y_original, self.sr, effect_chosen)

            self.btn_play.configure(text="Đang phát...")
            self.root.update()
            play_audio(self.y_processed, self.sr)

            self.btn_save.configure(state="normal")
            self._set_status("Đã áp dụng hiệu ứng và phát thử", "#118A2C")
        except Exception as e:
            messagebox.showerror("Lỗi Xử lý", str(e))
            self._set_status("Không thể xử lý âm thanh", "#C73A3A")
        finally:
            self.btn_play.configure(text="Áp dụng và nghe thử", state="normal")
            self.btn_browse.configure(state="normal")

    def save_audio(self):
        if self.y_processed is None:
            return

        output_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if output_path:
            save_to_file(output_path, self.y_processed, self.sr)
            messagebox.showinfo("Hoàn tất", f"Đã lưu thành công tại:\n{output_path}")
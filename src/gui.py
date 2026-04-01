import os
import queue
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import librosa
import numpy as np
import soundfile as sf

from src.effects import process_audio_data, start_recording, stop_recording, play_audio_stream, save_to_file

class VoiceChangerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phần mềm thay đổi giọng nói")
        self.base_width = 1080
        self.base_height = 720
        self._configure_window()
        self.root.minsize(820, 560)
        self.root.configure(fg_color="#E3E3E3")
        self._current_scale = 1.0

        self.y_original = None
        self.y_processed = None
        self.sr = 44100
        self.is_recording = False
        self.gain_db_var = ctk.DoubleVar(value=0.0)
        self.effect_param_vars = {}
        self.effect_param_specs = self._create_effect_param_specs()
        self.scalable_widgets = []
        self.scalable_sliders = []
        self.base_metrics = {}
        self.waveform_queue = queue.Queue(maxsize=8)
        self.waveform_after_id = None
        self.is_playing_audio = False
        self.played_samples = 0
        self.total_playback_samples = 0
        self.wave_palette_var = ctk.StringVar(value="Aurora")
        self.wave_smooth_var = ctk.DoubleVar(value=0.7)

        self._build_layout()
        self.root.bind("<Configure>", self._on_window_resize)

    def _configure_window(self):
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()

        width = min(self.base_width, max(900, screen_w - 80))
        height = min(self.base_height, max(600, screen_h - 120))

        x = max((screen_w - width) // 2, 0)
        y = max((screen_h - height) // 2, 0)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _create_effect_param_specs(self):
        return {
            "soc_chuot": [
                {"key": "n_steps", "label": "Pitch", "min": 0.0, "max": 12.0, "default": 5.0, "steps": 24, "suffix": " st"},
            ],
            "quai_vat": [
                {"key": "n_steps", "label": "Pitch", "min": -12.0, "max": 0.0, "default": -5.0, "steps": 24, "suffix": " st"},
            ],
            "robot": [
                {"key": "n_steps", "label": "Pitch", "min": -8.0, "max": 0.0, "default": -2.0, "steps": 16, "suffix": " st"},
                {"key": "delay_ms", "label": "Delay", "min": 5.0, "max": 80.0, "default": 30.0, "steps": 75, "suffix": " ms"},
                {"key": "mix", "label": "Mix", "min": 0.0, "max": 1.0, "default": 0.6, "steps": 100, "suffix": ""},
            ],
            "tua_nhanh": [
                {"key": "rate", "label": "Tốc độ", "min": 1.1, "max": 2.5, "default": 1.5, "steps": 140, "suffix": "x"},
            ],
            "tua_cham": [
                {"key": "rate", "label": "Tốc độ", "min": 0.4, "max": 0.95, "default": 0.7, "steps": 110, "suffix": "x"},
            ],
            "echo": [
                {"key": "delay_ms", "label": "Delay", "min": 80.0, "max": 600.0, "default": 250.0, "steps": 104, "suffix": " ms"},
                {"key": "mix", "label": "Mix", "min": 0.1, "max": 0.9, "default": 0.45, "steps": 80, "suffix": ""},
            ],
            "reverb": [
                {"key": "base_delay_ms", "label": "Base delay", "min": 20.0, "max": 120.0, "default": 40.0, "steps": 100, "suffix": " ms"},
                {"key": "wet", "label": "Wet", "min": 0.1, "max": 0.8, "default": 0.35, "steps": 70, "suffix": ""},
                {"key": "decay", "label": "Decay", "min": 0.1, "max": 0.8, "default": 0.6, "steps": 70, "suffix": ""},
            ],
            "noise_reduce": [
                {"key": "threshold_mult", "label": "Threshold", "min": 1.0, "max": 4.0, "default": 1.8, "steps": 120, "suffix": "x"},
                {"key": "floor", "label": "Min floor", "min": 0.001, "max": 0.02, "default": 0.003, "steps": 190, "suffix": ""},
                {"key": "attenuate", "label": "Attenuate", "min": 0.0, "max": 0.5, "default": 0.15, "steps": 100, "suffix": ""},
            ],
            "radio": [
                {"key": "low_hz", "label": "Low cut", "min": 100.0, "max": 1500.0, "default": 500.0, "steps": 140, "suffix": " Hz"},
                {"key": "high_hz", "label": "High cut", "min": 1800.0, "max": 5000.0, "default": 3200.0, "steps": 160, "suffix": " Hz"},
                {"key": "drive", "label": "Drive", "min": 1.0, "max": 4.0, "default": 2.2, "steps": 120, "suffix": ""},
                {"key": "noise", "label": "Noise", "min": 0.0, "max": 0.03, "default": 0.006, "steps": 120, "suffix": ""},
            ],
            "dien_thoai": [
                {"key": "low_hz", "label": "Low cut", "min": 100.0, "max": 1000.0, "default": 300.0, "steps": 180, "suffix": " Hz"},
                {"key": "high_hz", "label": "High cut", "min": 1500.0, "max": 4500.0, "default": 3400.0, "steps": 150, "suffix": " Hz"},
                {"key": "drive", "label": "Drive", "min": 1.0, "max": 3.0, "default": 1.7, "steps": 80, "suffix": ""},
            ],
        }

    def _build_layout(self):
        self.container = ctk.CTkFrame(self.root, fg_color="transparent")
        self.container.pack(fill="both", expand=True, padx=24, pady=18)

        self.title_label = ctk.CTkLabel(
            self.container,
            text="Phần mềm thay đổi giọng nói",
            text_color="#111111",
            font=ctk.CTkFont(family="Segoe UI", size=34, weight="bold"),
        )
        self.title_label.pack(pady=(0, 14))
        self.scalable_widgets.append((self.title_label, 34, "bold"))

        self._build_input_panel()
        self._build_effect_panel()

    def _build_input_panel(self):
        input_panel = ctk.CTkFrame(
            self.container,
            fg_color="#E9E9E9",
            border_color="#1E90FF",
            border_width=3,
            corner_radius=30,
        )
        input_panel.pack(fill="x", pady=(0, 12))

        self.lbl_status = ctk.CTkLabel(
            input_panel,
            text="Chưa có âm thanh đầu vào",
            text_color="#1E1E1E",
            anchor="w",
            font=ctk.CTkFont(family="Segoe UI", size=20, weight="normal"),
        )
        self.lbl_status.pack(side="left", padx=(20, 16), pady=14, expand=True, fill="x")
        self.scalable_widgets.append((self.lbl_status, 20, "normal"))

        button_box = ctk.CTkFrame(input_panel, fg_color="transparent")
        button_box.pack(side="right", padx=(0, 16), pady=10)

        self.btn_browse = ctk.CTkButton(
            button_box,
            text="Tải âm thanh lên",
            command=self.load_file,
            width=150,
            height=44,
            corner_radius=22,
            text_color="#FFFFFF",
            fg_color="#3E5FFF",
            hover_color="#334DE0",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
        )
        self.btn_browse.pack(side="left", padx=(0, 14))
        self.scalable_widgets.append((self.btn_browse, 16, "bold"))

        self.btn_record = ctk.CTkButton(
            button_box,
            text="Ghi âm",
            command=self.toggle_record,
            width=112,
            height=44,
            corner_radius=22,
            text_color="#FFFFFF",
            fg_color="#FF2B35",
            hover_color="#DC2029",
            font=ctk.CTkFont(family="Segoe UI", size=16, weight="bold"),
        )
        self.btn_record.pack(side="left")
        self.scalable_widgets.append((self.btn_record, 16, "bold"))

    def _build_effect_panel(self):
        effect_panel = ctk.CTkFrame(
            self.container,
            fg_color="#D4D4D4",
            corner_radius=32,
            border_color="#C8C8C8",
            border_width=2,
        )
        effect_panel.pack(fill="both", expand=True)
        self.effect_panel = effect_panel

        title = ctk.CTkLabel(
            effect_panel,
            text="Lựa chọn hiệu ứng âm thanh",
            text_color="#111111",
            font=ctk.CTkFont(family="Segoe UI", size=30, weight="bold"),
        )
        title.pack(pady=(14, 6))
        self.scalable_widgets.append((title, 30, "bold"))

        options_frame = ctk.CTkFrame(effect_panel, fg_color="transparent")
        options_frame.pack(fill="x", padx=40, pady=(4, 0))
        options_frame.grid_columnconfigure(0, weight=1)
        options_frame.grid_columnconfigure(1, weight=1)

        self.effect_var = ctk.StringVar(value="soc_chuot")
        rb_style = {
            "variable": self.effect_var,
            "font": ctk.CTkFont(family="Segoe UI", size=20, weight="normal"),
            "text_color": "#111111",
            "fg_color": "#1E90FF",
            "hover_color": "#4CA8FF",
            "border_color": "#7A7A7A",
            "border_width_unchecked": 2,
            "border_width_checked": 2,
            "radiobutton_width": 22,
            "radiobutton_height": 22,
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

        for widget in options_frame.winfo_children():
            self.scalable_widgets.append((widget, 20, "normal"))

        self.effect_var.trace_add("write", self._on_effect_changed)
        self._build_tuning_panel(effect_panel)
        self._build_waveform_panel(effect_panel)

        action_frame = ctk.CTkFrame(effect_panel, fg_color="transparent")
        action_frame.pack(side="bottom", anchor="e", padx=28, pady=16)

        self.btn_play = ctk.CTkButton(
            action_frame,
            text="Áp dụng và nghe thử",
            command=self.process_and_play,
            width=210,
            height=46,
            corner_radius=23,
            text_color="#F5F9D2",
            fg_color="#6FBF73",
            hover_color="#5AA663",
            font=ctk.CTkFont(family="Segoe UI", size=17, weight="bold"),
        )
        self.btn_play.pack(side="left", padx=(0, 20))
        self.scalable_widgets.append((self.btn_play, 17, "bold"))

        self.btn_save = ctk.CTkButton(
            action_frame,
            text="Lưu file",
            command=self.save_audio,
            width=130,
            height=46,
            corner_radius=23,
            text_color="#FFFFFF",
            fg_color="#08C449",
            hover_color="#06A73D",
            font=ctk.CTkFont(family="Segoe UI", size=17, weight="bold"),
            state="disabled",
        )
        self.btn_save.pack(side="left")
        self.scalable_widgets.append((self.btn_save, 17, "bold"))

        self.base_metrics = {
            self.btn_browse: {"width": 150, "height": 44, "corner_radius": 22},
            self.btn_record: {"width": 112, "height": 44, "corner_radius": 22},
            self.btn_play: {"width": 210, "height": 46, "corner_radius": 23},
            self.btn_save: {"width": 130, "height": 46, "corner_radius": 23},
            self.effect_panel: {"corner_radius": 32},
        }

    def _build_waveform_panel(self, parent):
        wave_panel = ctk.CTkFrame(parent, fg_color="#111111", corner_radius=20)
        wave_panel.pack(fill="x", padx=26, pady=(8, 6))

        wave_title = ctk.CTkLabel(
            wave_panel,
            text="Sóng âm khi phát",
            text_color="#EAF4FF",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
        )
        wave_title.pack(anchor="w", padx=14, pady=(10, 6))
        self.scalable_widgets.append((wave_title, 15, "bold"))

        self.wave_canvas = tk.Canvas(
            wave_panel,
            height=140,
            bg="#0B0F16",
            highlightthickness=0,
            bd=0,
        )
        self.wave_canvas.pack(fill="x", padx=12, pady=(0, 12))
        self.wave_canvas.bind("<Configure>", lambda _event: self._draw_waveform())

        controls_row = ctk.CTkFrame(wave_panel, fg_color="transparent")
        controls_row.pack(fill="x", padx=12, pady=(0, 8))

        palette_label = ctk.CTkLabel(
            controls_row,
            text="Màu:",
            text_color="#EAF4FF",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        )
        palette_label.pack(side="left", padx=(0, 6))
        self.scalable_widgets.append((palette_label, 12, "bold"))

        self.wave_palette_switch = ctk.CTkSegmentedButton(
            controls_row,
            values=["Aurora", "Neon", "Sunset"],
            variable=self.wave_palette_var,
            command=lambda _value: self._draw_waveform(),
            height=26,
            width=230,
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        )
        self.wave_palette_switch.pack(side="left", padx=(0, 14))
        self.scalable_widgets.append((self.wave_palette_switch, 12, "bold"))

        smooth_label = ctk.CTkLabel(
            controls_row,
            text="Độ mượt:",
            text_color="#EAF4FF",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="bold"),
        )
        smooth_label.pack(side="left", padx=(0, 6))
        self.scalable_widgets.append((smooth_label, 12, "bold"))

        self.wave_smooth_slider = ctk.CTkSlider(
            controls_row,
            from_=0.0,
            to=1.0,
            number_of_steps=20,
            variable=self.wave_smooth_var,
            width=140,
            progress_color="#4DD7FF",
            button_color="#26A9D4",
            command=lambda _value: self._draw_waveform(),
        )
        self.wave_smooth_slider.pack(side="left", padx=(0, 8))
        self.scalable_sliders.append(self.wave_smooth_slider)

        self.wave_smooth_value = ctk.CTkLabel(
            controls_row,
            text=f"{self.wave_smooth_var.get():.2f}",
            text_color="#D0E7FF",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="normal"),
            width=38,
            anchor="e",
        )
        self.wave_smooth_value.pack(side="left")
        self.scalable_widgets.append((self.wave_smooth_value, 12, "normal"))

        self.wave_progress = ctk.CTkProgressBar(wave_panel, height=10, corner_radius=6)
        self.wave_progress.pack(fill="x", padx=12, pady=(0, 6))
        self.wave_progress.set(0)

        time_row = ctk.CTkFrame(wave_panel, fg_color="transparent")
        time_row.pack(fill="x", padx=12, pady=(0, 10))

        self.lbl_time_current = ctk.CTkLabel(
            time_row,
            text="00:00.0",
            text_color="#AFC8E0",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="normal"),
        )
        self.lbl_time_current.pack(side="left")
        self.scalable_widgets.append((self.lbl_time_current, 12, "normal"))

        self.lbl_time_total = ctk.CTkLabel(
            time_row,
            text="00:00.0",
            text_color="#AFC8E0",
            font=ctk.CTkFont(family="Segoe UI", size=12, weight="normal"),
        )
        self.lbl_time_total.pack(side="right")
        self.scalable_widgets.append((self.lbl_time_total, 12, "normal"))

        self._draw_waveform()

    def _build_tuning_panel(self, parent):
        tuning_panel = ctk.CTkFrame(parent, fg_color="#E7E7E7", corner_radius=22)
        tuning_panel.pack(fill="x", padx=26, pady=(8, 4))
        self.tuning_panel = tuning_panel

        ctk.CTkLabel(
            tuning_panel,
            text="Tinh chỉnh âm thanh",
            text_color="#1A1A1A",
            font=ctk.CTkFont(family="Segoe UI", size=18, weight="bold"),
        ).pack(anchor="w", padx=18, pady=(12, 6))
        self.scalable_widgets.append((tuning_panel.winfo_children()[0], 18, "bold"))

        self.base_metrics[self.tuning_panel] = {"corner_radius": 22}

        self._add_slider_row(
            tuning_panel,
            label_text="Gain",
            var=self.gain_db_var,
            min_value=-18.0,
            max_value=18.0,
            steps=144,
            suffix=" dB",
        )

        self.effect_param_container = ctk.CTkFrame(tuning_panel, fg_color="transparent")
        self.effect_param_container.pack(fill="x", padx=12, pady=(2, 10))
        self._render_effect_controls(self.effect_var.get())

    def _add_slider_row(self, parent, label_text, var, min_value, max_value, steps, suffix):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=4)

        ctk.CTkLabel(
            row,
            text=label_text,
            width=110,
            anchor="w",
            text_color="#222222",
            font=ctk.CTkFont(family="Segoe UI", size=15, weight="bold"),
        ).pack(side="left", padx=(0, 8))
        label_widget = row.winfo_children()[-1]
        self.scalable_widgets.append((label_widget, 15, "bold"))

        value_label = ctk.CTkLabel(
            row,
            text=f"{var.get():.2f}{suffix}",
            width=86,
            anchor="e",
            text_color="#1A1A1A",
            font=ctk.CTkFont(family="Segoe UI", size=13, weight="normal"),
        )
        value_label.pack(side="right")
        self.scalable_widgets.append((value_label, 13, "normal"))

        slider = ctk.CTkSlider(
            row,
            from_=min_value,
            to=max_value,
            number_of_steps=steps,
            variable=var,
            progress_color="#1E90FF",
            button_color="#1677D9",
            command=lambda current: value_label.configure(text=f"{float(current):.2f}{suffix}"),
        )
        slider.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.scalable_sliders.append(slider)

    def _on_effect_changed(self, *_):
        self._render_effect_controls(self.effect_var.get())

    def _render_effect_controls(self, effect_name):
        for child in self.effect_param_container.winfo_children():
            child.destroy()

        self.effect_param_vars = {}
        specs = self.effect_param_specs.get(effect_name, [])
        if not specs:
            ctk.CTkLabel(
                self.effect_param_container,
                text="Hiệu ứng này không có thông số bổ sung",
                text_color="#444444",
                font=ctk.CTkFont(family="Segoe UI", size=13),
            ).pack(anchor="w", padx=6, pady=6)
            self.scalable_widgets.append((self.effect_param_container.winfo_children()[-1], 13, "normal"))
            return

        for spec in specs:
            key = spec["key"]
            var = ctk.DoubleVar(value=spec["default"])
            self.effect_param_vars[key] = var
            self._add_slider_row(
                self.effect_param_container,
                label_text=spec["label"],
                var=var,
                min_value=spec["min"],
                max_value=spec["max"],
                steps=spec["steps"],
                suffix=spec["suffix"],
            )

    def _on_window_resize(self, event):
        if event.widget is not self.root:
            return

        width = max(event.width, 1)
        height = max(event.height, 1)
        scale = min(width / self.base_width, height / self.base_height)
        scale = max(0.72, min(1.2, scale))

        if abs(scale - self._current_scale) < 0.04:
            return

        self._current_scale = scale
        self._apply_scale(scale)

    def _apply_scale(self, scale):
        for widget, base_size, weight in self.scalable_widgets:
            if not widget.winfo_exists():
                continue

            size = max(11, int(base_size * scale))
            widget.configure(font=ctk.CTkFont(family="Segoe UI", size=size, weight=weight))

        for widget, metrics in self.base_metrics.items():
            if not widget.winfo_exists():
                continue

            updated = {}
            for key, value in metrics.items():
                updated[key] = max(8, int(value * scale))
            widget.configure(**updated)

    def _collect_effect_params(self):
        return {key: float(var.get()) for key, var in self.effect_param_vars.items()}

    def _set_status(self, text, color="#1E1E1E"):
        self.lbl_status.configure(text=text, text_color=color)

    def _get_wave_palette(self):
        palettes = {
            "Aurora": {
                "bg": "#020726",
                "axis": "#122150",
                "hint": "#7D8CC0",
                "start": "#D13FFF",
                "mid": "#6A39FF",
                "end": "#57F3FF",
            },
            "Neon": {
                "bg": "#031914",
                "axis": "#13392C",
                "hint": "#78B8A1",
                "start": "#7DFF68",
                "mid": "#23E08A",
                "end": "#45FFF0",
            },
            "Sunset": {
                "bg": "#200B14",
                "axis": "#4A1F2F",
                "hint": "#C59AAB",
                "start": "#FF5A8F",
                "mid": "#FF7A59",
                "end": "#FFD35D",
            },
        }
        return palettes.get(self.wave_palette_var.get(), palettes["Aurora"])

    def _hex_to_rgb(self, color):
        color = color.lstrip("#")
        return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)

    def _rgb_to_hex(self, rgb):
        return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"

    def _mix_color(self, c1, c2, ratio):
        ratio = min(1.0, max(0.0, float(ratio)))
        r1, g1, b1 = self._hex_to_rgb(c1)
        r2, g2, b2 = self._hex_to_rgb(c2)
        mixed = (
            int(r1 + (r2 - r1) * ratio),
            int(g1 + (g2 - g1) * ratio),
            int(b1 + (b2 - b1) * ratio),
        )
        return self._rgb_to_hex(mixed)

    def _gradient_color(self, t, palette):
        t = min(1.0, max(0.0, float(t)))
        if t <= 0.5:
            local_t = t / 0.5
            return self._mix_color(palette["start"], palette["mid"], local_t)
        local_t = (t - 0.5) / 0.5
        return self._mix_color(palette["mid"], palette["end"], local_t)

    def _format_time(self, seconds):
        seconds = max(0.0, float(seconds))
        minutes = int(seconds // 60)
        remain = seconds - minutes * 60
        return f"{minutes:02d}:{remain:04.1f}"

    def _reset_playback_progress(self):
        self.played_samples = 0
        self.total_playback_samples = 0
        self.wave_progress.set(0)
        self.lbl_time_current.configure(text="00:00.0")
        self.lbl_time_total.configure(text="00:00.0")

    def _update_progress_ui(self):
        if self.sr <= 0:
            return

        total_seconds = self.total_playback_samples / self.sr if self.total_playback_samples else 0.0
        current_seconds = self.played_samples / self.sr
        progress = min(1.0, self.played_samples / self.total_playback_samples) if self.total_playback_samples else 0.0

        self.wave_progress.set(progress)
        self.lbl_time_current.configure(text=self._format_time(current_seconds))
        self.lbl_time_total.configure(text=self._format_time(total_seconds))

    def _draw_waveform(self, samples=None):
        if not hasattr(self, "wave_canvas"):
            return

        width = max(self.wave_canvas.winfo_width(), 10)
        height = max(self.wave_canvas.winfo_height(), 10)
        mid = height / 2
        palette = self._get_wave_palette()
        smoothness = float(self.wave_smooth_var.get())
        self.wave_smooth_value.configure(text=f"{smoothness:.2f}")
        self.wave_canvas.configure(bg=palette["bg"])
        self.wave_progress.configure(progress_color=palette["end"], fg_color=self._mix_color(palette["bg"], "#FFFFFF", 0.12))
        self.wave_smooth_slider.configure(
            progress_color=palette["end"],
            button_color=self._mix_color(palette["end"], "#FFFFFF", 0.25),
        )

        self.wave_canvas.delete("all")
        self.wave_canvas.create_line(0, mid, width, mid, fill=palette["axis"], width=1)

        if samples is None or len(samples) < 2:
            self.wave_canvas.create_text(
                width / 2,
                mid,
                text="Nhấn 'Áp dụng và nghe thử' để xem waveform realtime",
                fill=palette["hint"],
                font=("Segoe UI", 10),
            )
            return

        target_points = min(max(width, 90), 700)
        indexes = np.linspace(0, len(samples) - 1, target_points, dtype=int)
        reduced = np.asarray(samples[indexes], dtype=np.float32)

        envelope = np.abs(reduced)
        window = max(1, int(2 + smoothness * 26))
        if window > 1:
            kernel = np.ones(window, dtype=np.float32) / window
            envelope = np.convolve(envelope, kernel, mode="same")

        peak = float(np.max(envelope)) if len(envelope) else 0.0
        if peak < 1e-6:
            peak = 1.0
        envelope = np.clip(envelope / peak, 0.0, 1.0)
        envelope = np.power(envelope, 0.72)

        min_half = max(1.0, height * 0.012)
        max_half = height * 0.45

        for i, value in enumerate(envelope):
            x = (i / max(target_points - 1, 1)) * (width - 1)
            half = min_half + float(value) * (max_half - min_half)
            color = self._gradient_color(i / max(target_points - 1, 1), palette)
            glow_color = self._mix_color(color, palette["bg"], 0.62)

            self.wave_canvas.create_line(x, mid - half, x, mid + half, fill=glow_color, width=4)
            self.wave_canvas.create_line(x, mid - half, x, mid + half, fill=color, width=2)

    def _enqueue_wave_chunk(self, chunk):
        if chunk is None or len(chunk) == 0:
            return

        try:
            self.waveform_queue.put_nowait(chunk)
        except queue.Full:
            try:
                _ = self.waveform_queue.get_nowait()
            except queue.Empty:
                pass
            self.waveform_queue.put_nowait(chunk)

    def _start_waveform_loop(self):
        if self.waveform_after_id is None:
            self.waveform_after_id = self.root.after(30, self._update_waveform)

    def _update_waveform(self):
        latest_chunk = None
        consumed_samples = 0
        while True:
            try:
                latest_chunk = self.waveform_queue.get_nowait()
                consumed_samples += len(latest_chunk)
            except queue.Empty:
                break

        if latest_chunk is not None:
            self._draw_waveform(latest_chunk)
            self.played_samples += consumed_samples
            self._update_progress_ui()

        if self.is_playing_audio or (not self.waveform_queue.empty()):
            self.waveform_after_id = self.root.after(30, self._update_waveform)
        else:
            self.waveform_after_id = None

    def _on_audio_finished(self):
        self.root.after(0, self._finish_audio_ui)

    def _finish_audio_ui(self):
        self.is_playing_audio = False
        self.played_samples = self.total_playback_samples
        self._update_progress_ui()
        self.btn_play.configure(text="Áp dụng và nghe thử", state="normal")
        self.btn_browse.configure(state="normal")
        self._set_status("Đã phát xong bản xem thử", "#118A2C")

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

    def _get_audio_filetypes(self):
        common_ext = "*.wav *.mp3 *.mp4 *.m4a *.aac *.flac *.ogg *.opus *.wma *.aiff *.aif *.au"
        return [
            ("Audio files", common_ext),
            ("All files", "*.*"),
        ]

    def _load_audio_input(self, filepath):
        try:
            y, sr = librosa.load(filepath, sr=None, mono=True)
            return y.astype(np.float32), int(sr)
        except Exception:
            data, sr = sf.read(filepath, always_2d=False)
            if isinstance(data, np.ndarray) and data.ndim > 1:
                data = np.mean(data, axis=1)
            return np.asarray(data, dtype=np.float32), int(sr)

    # --- CÁC HÀM XỬ LÝ SỰ KIỆN ---
    def load_file(self):
        filepath = filedialog.askopenfilename(filetypes=self._get_audio_filetypes())
        if filepath:
            filename = os.path.basename(filepath)
            self._set_status("Đang tải file...", "#2D72B8")
            self.root.update()

            try:
                self.y_original, self.sr = self._load_audio_input(filepath)
                self.y_processed = None

                duration_seconds = len(self.y_original) / self.sr if self.sr else 0
                self._set_status(f"Đã tải: {filename} ({duration_seconds:.1f}s)", "#118A2C")
                self.btn_save.configure(state="disabled")
                self._reset_playback_progress()
                self._draw_waveform()
            except Exception as e:
                self.y_original = None
                self.y_processed = None
                self._set_status("Không thể đọc file âm thanh", "#C73A3A")
                ext = os.path.splitext(filename)[1].lower()
                extra_note = ""
                if ext == ".mp4":
                    extra_note = "\n\nLưu ý: để đọc .mp4, máy cần có backend giải mã (thường là FFmpeg)."
                messagebox.showerror(
                    "Lỗi đọc file",
                    f"Không thể đọc file {filename}.\nHãy thử định dạng khác hoặc kiểm tra codec."
                    f"{extra_note}\n\nChi tiết: {e}",
                )

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
                self._reset_playback_progress()
                self._draw_waveform()
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
            effect_params = self._collect_effect_params()
            gain_db = float(self.gain_db_var.get())
            self.y_processed = process_audio_data(
                self.y_original,
                self.sr,
                effect_chosen,
                params=effect_params,
                gain_db=gain_db,
            )

            self.btn_play.configure(text="Đang phát...")
            self.root.update()
            self.is_playing_audio = True
            self.played_samples = 0
            self.total_playback_samples = len(self.y_processed)
            self._update_progress_ui()
            self._start_waveform_loop()
            play_audio_stream(
                self.y_processed,
                self.sr,
                on_chunk=self._enqueue_wave_chunk,
                on_finished=self._on_audio_finished,
            )

            self.btn_save.configure(state="normal")
            self._set_status("Đang phát bản đã áp dụng hiệu ứng", "#118A2C")
        except Exception as e:
            messagebox.showerror("Lỗi Xử lý", str(e))
            self._set_status("Không thể xử lý âm thanh", "#C73A3A")
            self.btn_play.configure(text="Áp dụng và nghe thử", state="normal")
            self.btn_browse.configure(state="normal")

    def save_audio(self):
        if self.y_processed is None:
            return

        output_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if output_path:
            save_to_file(output_path, self.y_processed, self.sr)
            messagebox.showinfo("Hoàn tất", f"Đã lưu thành công tại:\n{output_path}")
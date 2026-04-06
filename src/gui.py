import os
import queue
import tkinter as tk
from tkinter import filedialog, messagebox

import customtkinter as ctk
import librosa
import numpy as np
import soundfile as sf

from src.effects import process_audio_data, start_recording, stop_recording, play_audio_stream, save_to_file, stop_audio

class VoiceChangerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Phần mềm thay đổi giọng nói")
        self.base_width = 1080
        self.base_height = 720
        self._configure_window()
        self.root.minsize(820, 560)
        self.root.configure(fg_color="#14161A")
        self._current_scale = 1.0

        self.y_original = None
        self.y_processed = None
        self.sr = 44100
        self.is_recording = False
        self.gain_db_var = ctk.DoubleVar(value=0.0)
        self.effect_param_vars = {}
        self.effect_param_specs = self._create_effect_param_specs()
        self.param_help_texts = self._create_param_help_texts()
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
        self.effect_buttons = {}
        self._tooltip_window = None
        self._tooltip_label = None
        self._tooltip_after_id = None
        self._tooltip_text = ""
        self.effect_options = [
            ("Sóc chuột", "soc_chuot"),
            ("Quái vật", "quai_vat"),
            ("Robot", "robot"),
            ("Tua nhanh", "tua_nhanh"),
            ("Tua chậm", "tua_cham"),
            ("Echo", "echo"),
            ("Reverb", "reverb"),
            ("Noise reduce", "noise_reduce"),
            ("Radio", "radio"),
            ("Điện thoại", "dien_thoai"),
        ]

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

    def _create_param_help_texts(self):
        return {
            "n_steps": "Điều chỉnh cao độ giọng. Giá trị lớn làm giọng cao hơn, nhỏ làm trầm hơn.",
            "delay_ms": "Thời gian trễ giữa âm gốc và âm lặp lại. Tăng để hiệu ứng vang rõ hơn.",
            "mix": "Tỉ lệ hòa trộn giữa âm gốc và hiệu ứng. Cao hơn nghĩa là hiệu ứng rõ hơn.",
            "rate": "Tốc độ phát lại âm thanh. Lớn hơn 1 là nhanh hơn, nhỏ hơn 1 là chậm hơn.",
            "base_delay_ms": "Độ trễ cơ bản của reverb, quyết định độ rộng không gian giả lập.",
            "wet": "Mức âm ướt (âm hiệu ứng) trong reverb. Tăng để âm vang nhiều hơn.",
            "decay": "Độ tắt dần của đuôi vang. Tăng để đuôi vang kéo dài hơn.",
            "threshold_mult": "Hệ số ngưỡng lọc nhiễu. Tăng để lọc mạnh hơn nhưng có thể mất chi tiết nhỏ.",
            "floor": "Ngưỡng sàn tối thiểu để nhận diện nhiễu nền khi tín hiệu quá nhỏ.",
            "attenuate": "Mức giảm biên độ phần tín hiệu bị xem là nhiễu. Cao hơn sẽ giảm mạnh hơn.",
            "low_hz": "Tần số cắt thấp. Âm dưới ngưỡng này sẽ bị giảm.",
            "high_hz": "Tần số cắt cao. Âm trên ngưỡng này sẽ bị giảm.",
            "drive": "Mức bão hòa/méo nhẹ. Tăng để âm dày và gắt hơn.",
            "noise": "Lượng nhiễu nền bổ sung để mô phỏng chất âm thiết bị cũ.",
        }

    def _show_tooltip(self):
        if not self._tooltip_text:
            return

        if self._tooltip_window is None or not self._tooltip_window.winfo_exists():
            self._tooltip_window = tk.Toplevel(self.root)
            self._tooltip_window.overrideredirect(True)
            self._tooltip_window.attributes("-topmost", True)
            self._tooltip_window.configure(bg="#111318")
            self._tooltip_label = ctk.CTkLabel(
                self._tooltip_window,
                text=self._tooltip_text,
                text_color="#ECEFF3",
                fg_color="#111318",
                corner_radius=8,
                justify="left",
                anchor="w",
                padx=10,
                pady=6,
                font=ctk.CTkFont(family="Bahnschrift", size=13, weight="normal"),
            )
            self._tooltip_label.pack(fill="both", expand=True)
        else:
            self._tooltip_label.configure(text=self._tooltip_text)

        pointer_x = self.root.winfo_pointerx()
        pointer_y = self.root.winfo_pointery()
        self._tooltip_window.geometry(f"+{pointer_x + 14}+{pointer_y + 12}")
        self._tooltip_window.deiconify()

    def _hide_tooltip(self):
        if self._tooltip_after_id is not None:
            try:
                self.root.after_cancel(self._tooltip_after_id)
            except Exception:
                pass
            self._tooltip_after_id = None

        if self._tooltip_window is not None and self._tooltip_window.winfo_exists():
            self._tooltip_window.withdraw()

    def _schedule_tooltip(self, text):
        self._tooltip_text = text or ""
        if not self._tooltip_text:
            self._hide_tooltip()
            return

        if self._tooltip_after_id is not None:
            try:
                self.root.after_cancel(self._tooltip_after_id)
            except Exception:
                pass
        self._tooltip_after_id = self.root.after(280, self._show_tooltip)

    def _bind_tooltip(self, widget, text):
        if widget is None or not text:
            return
        widget.bind("<Enter>", lambda _event, t=text: self._schedule_tooltip(t), add="+")
        widget.bind("<Leave>", lambda _event: self._hide_tooltip(), add="+")

    def _build_layout(self):
        self.container = ctk.CTkFrame(self.root, fg_color="transparent")
        self.container.pack(fill="both", expand=True, padx=22, pady=18)

        self.title_shell = ctk.CTkFrame(
            self.container,
            fg_color="#EDEDED",
            corner_radius=28,
            border_width=3,
            border_color="#FFFFFF",
            height=64,
        )
        self.title_shell.pack(fill="x", pady=(0, 12))
        self.title_shell.pack_propagate(False)

        ctk.CTkFrame(self.title_shell, width=40, fg_color="#131519", corner_radius=20).pack(
            side="left", padx=(8, 10), pady=8
        )
        ctk.CTkFrame(self.title_shell, width=40, fg_color="#131519", corner_radius=20).pack(
            side="right", padx=(10, 8), pady=8
        )

        self.title_label = ctk.CTkLabel(
            self.title_shell,
            text="PHẦN MỀM THAY ĐỔI GIỌNG NÓI",
            text_color="#0D1114",
            fg_color="#9EDF56",
            corner_radius=22,
            height=44,
            font=ctk.CTkFont(family="Bahnschrift", size=30, weight="bold"),
        )
        self.title_label.pack(fill="x", expand=True, padx=4, pady=8)
        self.scalable_widgets.append((self.title_label, 30, "bold"))

        self._build_input_panel()
        self._build_effect_panel()

    def _build_input_panel(self):
        input_panel = ctk.CTkFrame(
            self.container,
            fg_color="transparent",
        )
        input_panel.pack(fill="x", pady=(0, 12))

        status_box = ctk.CTkFrame(
            input_panel,
            fg_color="#EDEDED",
            corner_radius=20,
            border_width=2,
            border_color="#B5B5B5",
            height=42,
        )
        status_box.pack(side="left", fill="x", expand=True, padx=(0, 12))
        status_box.pack_propagate(False)

        self.lbl_status = ctk.CTkLabel(
            status_box,
            text="Chưa có âm thanh đầu vào",
            text_color="#6A6A6A",
            anchor="w",
            font=ctk.CTkFont(family="Bahnschrift", size=18, weight="normal"),
        )
        self.lbl_status.pack(side="left", padx=(16, 16), pady=8, expand=True, fill="x")
        self.scalable_widgets.append((self.lbl_status, 18, "normal"))

        button_box = ctk.CTkFrame(input_panel, fg_color="transparent")
        button_box.pack(side="right")

        self.btn_browse = ctk.CTkButton(
            button_box,
            text="Tải âm thanh lên",
            command=self.load_file,
            width=132,
            height=40,
            corner_radius=22,
            text_color="#08243F",
            fg_color="#78AEE8",
            hover_color="#6798CC",
            border_width=2,
            border_color="#2E5A9E",
            font=ctk.CTkFont(family="Bahnschrift", size=17, weight="bold"),
        )
        self.btn_browse.pack(side="left", padx=(0, 10))
        self.scalable_widgets.append((self.btn_browse, 17, "bold"))

        self.btn_record = ctk.CTkButton(
            button_box,
            text="Ghi âm",
            command=self.toggle_record,
            width=94,
            height=40,
            corner_radius=22,
            text_color="#2E1B00",
            fg_color="#FF910C",
            hover_color="#E67F00",
            border_width=2,
            border_color="#BE5D00",
            font=ctk.CTkFont(family="Bahnschrift", size=17, weight="bold"),
        )
        self.btn_record.pack(side="left")
        self.scalable_widgets.append((self.btn_record, 17, "bold"))

    def _build_effect_panel(self):
        effect_panel = ctk.CTkFrame(
            self.container,
            fg_color="transparent",
        )
        effect_panel.pack(fill="both", expand=True)
        self.effect_panel = effect_panel

        mid_panel = ctk.CTkFrame(effect_panel, fg_color="transparent")
        mid_panel.pack(fill="both", expand=True)
        mid_panel.grid_columnconfigure(0, weight=3)
        mid_panel.grid_columnconfigure(1, weight=2)
        mid_panel.grid_rowconfigure(0, weight=1)

        select_card = ctk.CTkFrame(mid_panel, fg_color="#2E3034", corner_radius=24)
        select_card.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=(0, 12))

        title = ctk.CTkLabel(
            select_card,
            text="Lựa chọn hiệu ứng âm thanh",
            text_color="#EFEFEF",
            font=ctk.CTkFont(family="Bahnschrift", size=24, weight="bold"),
        )
        title.pack(anchor="w", padx=20, pady=(14, 10))
        self.scalable_widgets.append((title, 24, "bold"))

        options_frame = ctk.CTkFrame(select_card, fg_color="transparent")
        options_frame.pack(fill="both", expand=True, padx=20, pady=(0, 18))
        options_frame.grid_columnconfigure(0, weight=1)
        options_frame.grid_columnconfigure(1, weight=1)

        self.effect_var = ctk.StringVar(value="soc_chuot")
        for idx, (label, value) in enumerate(self.effect_options):
            row = idx // 2
            col = idx % 2
            btn = ctk.CTkButton(
                options_frame,
                text=label,
                command=lambda effect=value: self._set_effect(effect),
                height=38,
                corner_radius=10,
                fg_color="#E7E7E7",
                hover_color="#CECECE",
                text_color="#212121",
                font=ctk.CTkFont(family="Bahnschrift", size=18, weight="normal"),
            )
            btn.grid(row=row, column=col, sticky="ew", padx=6, pady=6)
            self.effect_buttons[value] = btn
            self.scalable_widgets.append((btn, 18, "normal"))

        self._build_tuning_panel(mid_panel)

        self._build_waveform_panel(effect_panel)

        action_frame = ctk.CTkFrame(self.wave_panel, fg_color="transparent")
        action_frame.pack(anchor="ne", padx=16, pady=(10, 0))

        self.btn_play = ctk.CTkButton(
            action_frame,
            text="Phát",
            command=self.process_and_play,
            width=86,
            height=36,
            corner_radius=23,
            text_color="#203100",
            fg_color="#96DD47",
            hover_color="#82C53C",
            border_width=2,
            border_color="#588D1D",
            font=ctk.CTkFont(family="Bahnschrift", size=18, weight="bold"),
        )
        self.btn_play.pack(side="left", padx=(0, 10))
        self.scalable_widgets.append((self.btn_play, 18, "bold"))

        self.btn_save = ctk.CTkButton(
            action_frame,
            text="Lưu file",
            command=self.save_audio,
            width=90,
            height=36,
            corner_radius=23,
            text_color="#321A00",
            fg_color="#FF910C",
            hover_color="#E57E00",
            border_width=2,
            border_color="#C76700",
            font=ctk.CTkFont(family="Bahnschrift", size=18, weight="bold"),
            state="disabled",
        )
        self.btn_save.pack(side="left")
        self.scalable_widgets.append((self.btn_save, 18, "bold"))

        self._refresh_effect_buttons()
        self._render_effect_controls(self.effect_var.get())

        self.base_metrics = {
            self.btn_browse: {"width": 132, "height": 40, "corner_radius": 22},
            self.btn_record: {"width": 94, "height": 40, "corner_radius": 22},
            self.btn_play: {"width": 86, "height": 36, "corner_radius": 23},
            self.btn_save: {"width": 90, "height": 36, "corner_radius": 23},
        }

    def _build_waveform_panel(self, parent):
        wave_panel = ctk.CTkFrame(parent, fg_color="#2F3135", corner_radius=26)
        wave_panel.pack(fill="x", padx=0, pady=(0, 0), ipady=8)
        self.wave_panel = wave_panel

        wave_title = ctk.CTkLabel(
            wave_panel,
            text="Sóng âm khi phát",
            text_color="#EFEFEF",
            font=ctk.CTkFont(family="Bahnschrift", size=21, weight="bold"),
        )
        wave_title.pack(anchor="w", padx=24, pady=(8, 4))
        self.scalable_widgets.append((wave_title, 21, "bold"))

        self.wave_canvas = tk.Canvas(
            wave_panel,
            height=150,
            bg="#2A2D31",
            highlightthickness=0,
            bd=0,
        )
        self.wave_canvas.pack(fill="x", padx=20, pady=(2, 6))
        self.wave_canvas.bind("<Configure>", lambda _event: self._draw_waveform())

        self.wave_progress = ctk.CTkProgressBar(wave_panel, height=10, corner_radius=6)
        self.wave_progress.pack(fill="x", padx=20, pady=(0, 6))
        self.wave_progress.set(0)

        self._draw_waveform()

    def _build_tuning_panel(self, parent):
        tuning_panel = ctk.CTkFrame(parent, fg_color="#2E3034", corner_radius=24)
        tuning_panel.grid(row=0, column=1, sticky="nsew", padx=(10, 0), pady=(0, 12))
        self.tuning_panel = tuning_panel

        ctk.CTkLabel(
            tuning_panel,
            text="Tinh chỉnh âm thanh",
            text_color="#EFEFEF",
            font=ctk.CTkFont(family="Bahnschrift", size=24, weight="bold"),
        ).pack(anchor="w", padx=20, pady=(14, 10))
        self.scalable_widgets.append((tuning_panel.winfo_children()[0], 24, "bold"))

        self.base_metrics[self.tuning_panel] = {"corner_radius": 24}

        self._add_slider_row(
            tuning_panel,
            label_text="Gain",
            var=self.gain_db_var,
            min_value=-18.0,
            max_value=18.0,
            steps=144,
            suffix=" db",
        )

        ctk.CTkLabel(
            tuning_panel,
            text="Màu sóng",
            text_color="#EFEFEF",
            anchor="w",
            font=ctk.CTkFont(family="Bahnschrift", size=17, weight="normal"),
        ).pack(fill="x", padx=20, pady=(6, 4))

        self.wave_palette_switch = ctk.CTkSegmentedButton(
            tuning_panel,
            values=["Aurora", "Neon", "Sunset"],
            variable=self.wave_palette_var,
            command=lambda _value: self._draw_waveform(),
            height=30,
            font=ctk.CTkFont(family="Bahnschrift", size=14, weight="bold"),
            fg_color="#1B1D22",
            selected_color="#78AEE8",
            selected_hover_color="#6A9BD0",
            unselected_color="#F2F2F2",
            unselected_hover_color="#E2E2E2",
            text_color="#0F141A",
            text_color_disabled="#999999",
        )
        self.wave_palette_switch.pack(fill="x", padx=20, pady=(0, 8))
        self.scalable_widgets.append((self.wave_palette_switch, 14, "bold"))

        smooth_row = ctk.CTkFrame(tuning_panel, fg_color="transparent")
        smooth_row.pack(fill="x", padx=20, pady=(2, 8))

        smooth_label = ctk.CTkLabel(
            smooth_row,
            text="Độ mượt",
            text_color="#EFEFEF",
            font=ctk.CTkFont(family="Bahnschrift", size=17, weight="normal"),
        )
        smooth_label.pack(side="left")
        self.scalable_widgets.append((smooth_label, 17, "normal"))

        self.wave_smooth_value = ctk.CTkLabel(
            smooth_row,
            text=f"{self.wave_smooth_var.get():.2f}",
            text_color="#A5A8AD",
            font=ctk.CTkFont(family="Bahnschrift", size=16, weight="normal"),
        )
        self.wave_smooth_value.pack(side="right")
        self.scalable_widgets.append((self.wave_smooth_value, 16, "normal"))

        self.wave_smooth_slider = ctk.CTkSlider(
            tuning_panel,
            from_=0.0,
            to=1.0,
            number_of_steps=20,
            variable=self.wave_smooth_var,
            progress_color="#96DD47",
            fg_color="#ECECEC",
            button_color="#7FC73A",
            button_hover_color="#6BAD2F",
            command=lambda _value: self._draw_waveform(),
        )
        self.wave_smooth_slider.pack(fill="x", padx=20, pady=(0, 8))
        self.scalable_sliders.append(self.wave_smooth_slider)

        self.effect_param_container = ctk.CTkFrame(tuning_panel, fg_color="transparent")
        self.effect_param_container.pack(fill="x", padx=14, pady=(2, 10))

    def _add_slider_row(self, parent, label_text, var, min_value, max_value, steps, suffix, tooltip_text=""):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=20, pady=4)

        param_label = ctk.CTkLabel(
            row,
            text=label_text,
            width=110,
            anchor="w",
            text_color="#F0F0F0",
            font=ctk.CTkFont(family="Bahnschrift", size=17, weight="normal"),
        )
        param_label.pack(side="left", padx=(0, 8))
        self.scalable_widgets.append((param_label, 17, "normal"))

        value_label = ctk.CTkLabel(
            row,
            text=f"{var.get():.2f}{suffix}",
            width=86,
            anchor="e",
            text_color="#9EA3AA",
            font=ctk.CTkFont(family="Bahnschrift", size=16, weight="normal"),
        )
        value_label.pack(side="right")
        self.scalable_widgets.append((value_label, 16, "normal"))

        slider = ctk.CTkSlider(
            row,
            from_=min_value,
            to=max_value,
            number_of_steps=steps,
            variable=var,
            fg_color="#F2F2F2",
            progress_color="#96DD47",
            button_color="#80C63A",
            button_hover_color="#6DAE30",
            command=lambda current: value_label.configure(text=f"{float(current):.2f}{suffix}"),
        )
        slider.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.scalable_sliders.append(slider)

        self._bind_tooltip(param_label, tooltip_text)
        self._bind_tooltip(value_label, tooltip_text)
        self._bind_tooltip(slider, tooltip_text)

    def _set_effect(self, effect_name):
        self.effect_var.set(effect_name)
        self._on_effect_changed()

    def _refresh_effect_buttons(self):
        active_effect = self.effect_var.get()
        for effect, btn in self.effect_buttons.items():
            if effect == active_effect:
                btn.configure(
                    fg_color="#9EDF56",
                    hover_color="#91CE4F",
                    text_color="#182008",
                    border_width=0,
                )
            else:
                btn.configure(
                    fg_color="#ECECEC",
                    hover_color="#D5D5D5",
                    text_color="#202020",
                    border_width=0,
                )

    def _on_effect_changed(self, *_):
        self._refresh_effect_buttons()
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
                text_color="#B8BDC4",
                font=ctk.CTkFont(family="Bahnschrift", size=14),
            ).pack(anchor="w", padx=6, pady=6)
            self.scalable_widgets.append((self.effect_param_container.winfo_children()[-1], 14, "normal"))
            return

        for spec in specs:
            key = spec["key"]
            var = ctk.DoubleVar(value=spec["default"])
            self.effect_param_vars[key] = var
            tooltip_text = self.param_help_texts.get(key, "")
            self._add_slider_row(
                self.effect_param_container,
                label_text=spec["label"],
                var=var,
                min_value=spec["min"],
                max_value=spec["max"],
                steps=spec["steps"],
                suffix=spec["suffix"],
                tooltip_text=tooltip_text,
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
                "bg": "#2A2D31",
                "axis": "#41454A",
                "hint": "#80858E",
                "start": "#CFE45A",
                "mid": "#0F8AE4",
                "end": "#FF8E13",
            },
            "Neon": {
                "bg": "#23272C",
                "axis": "#3B4249",
                "hint": "#7B8794",
                "start": "#74FF4E",
                "mid": "#0ED2AF",
                "end": "#4CD9FF",
            },
            "Sunset": {
                "bg": "#2B2420",
                "axis": "#4E4139",
                "hint": "#A79383",
                "start": "#F2D55B",
                "mid": "#DE8A3E",
                "end": "#D2542E",
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
        if hasattr(self, "wave_progress"):
            self.wave_progress.set(0)
        if hasattr(self, "lbl_time_current"):
            self.lbl_time_current.configure(text="00:00.0")
        if hasattr(self, "lbl_time_total"):
            self.lbl_time_total.configure(text="00:00.0")

    def _update_progress_ui(self):
        if self.sr <= 0:
            return

        total_seconds = self.total_playback_samples / self.sr if self.total_playback_samples else 0.0
        current_seconds = self.played_samples / self.sr
        progress = min(1.0, self.played_samples / self.total_playback_samples) if self.total_playback_samples else 0.0

        if hasattr(self, "wave_progress"):
            self.wave_progress.set(progress)
        if hasattr(self, "lbl_time_current"):
            self.lbl_time_current.configure(text=self._format_time(current_seconds))
        if hasattr(self, "lbl_time_total"):
            self.lbl_time_total.configure(text=self._format_time(total_seconds))

    def _draw_waveform(self, samples=None):
        if not hasattr(self, "wave_canvas"):
            return

        width = max(self.wave_canvas.winfo_width(), 10)
        height = max(self.wave_canvas.winfo_height(), 10)
        mid = height / 2
        palette = self._get_wave_palette()
        smoothness = float(self.wave_smooth_var.get())
        if hasattr(self, "wave_smooth_value"):
            self.wave_smooth_value.configure(text=f"{smoothness:.2f}")
        self.wave_canvas.configure(bg=palette["bg"])
        if hasattr(self, "wave_progress"):
            self.wave_progress.configure(progress_color=palette["end"], fg_color=self._mix_color(palette["bg"], "#FFFFFF", 0.12))
        if hasattr(self, "wave_smooth_slider"):
            self.wave_smooth_slider.configure(
                progress_color="#96DD47",
                button_color="#80C63A",
            )

        self.wave_canvas.delete("all")
        self.wave_canvas.create_line(0, mid, width, mid, fill=palette["axis"], width=1)

        if samples is None or len(samples) < 2:
            self.wave_canvas.create_text(
                width / 2,
                mid,
                text="Nhấn 'Phát' để xem waveform realtime",
                fill=palette["hint"],
                font=("Bahnschrift", 11),
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
        if not self.is_playing_audio:
            return
        self.root.after(0, self._finish_audio_ui)

    def _finish_audio_ui(self):
        self.is_playing_audio = False
        self.played_samples = self.total_playback_samples
        self._update_progress_ui()
        self.btn_play.configure(text="Phát", state="normal")
        self.btn_browse.configure(state="normal")
        self._set_status("Đã phát xong bản xem thử", "#4F9F2F")

    def _set_recording_ui(self, recording):
        if recording:
            self.btn_record.configure(text="Dừng", fg_color="#D24728", hover_color="#BC3A1D", border_color="#8D2C15")
            self.btn_browse.configure(state="disabled")
            self.btn_play.configure(state="disabled")
            self._set_status("Đang ghi âm...", "#D16042")
            return

        self.btn_record.configure(text="Ghi âm", fg_color="#FF910C", hover_color="#E67F00", border_color="#BE5D00")
        self.btn_browse.configure(state="normal")
        self.btn_play.configure(state="normal")

    def _stop_playback(self, user_initiated=False):
        stop_audio()
        self.is_playing_audio = False

        while not self.waveform_queue.empty():
            try:
                self.waveform_queue.get_nowait()
            except queue.Empty:
                break

        if self.waveform_after_id is not None:
            try:
                self.root.after_cancel(self.waveform_after_id)
            except Exception:
                pass
            self.waveform_after_id = None

        self.btn_play.configure(text="Phát", state="normal")
        self.btn_browse.configure(state="normal")

        if user_initiated:
            self._set_status("Đã dừng phát", "#D9A82D")

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
        if self.is_playing_audio:
            self._stop_playback(user_initiated=False)

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
        if self.is_playing_audio:
            self._stop_playback(user_initiated=False)

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
        if self.is_playing_audio:
            self._stop_playback(user_initiated=True)
            return

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

            self.btn_play.configure(text="Dừng", state="normal")
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
            self._set_status("Đang phát bản đã áp dụng hiệu ứng", "#4F9F2F")
        except Exception as e:
            messagebox.showerror("Lỗi Xử lý", str(e))
            self._set_status("Không thể xử lý âm thanh", "#C73A3A")
            self.btn_play.configure(text="Phát", state="normal")
            self.btn_browse.configure(state="normal")

    def save_audio(self):
        if self.y_processed is None:
            return

        output_path = filedialog.asksaveasfilename(defaultextension=".wav", filetypes=[("WAV files", "*.wav")])
        if output_path:
            save_to_file(output_path, self.y_processed, self.sr)
            messagebox.showinfo("Hoàn tất", f"Đã lưu thành công tại:\n{output_path}")
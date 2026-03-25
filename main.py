import customtkinter as ctk
from src.gui import VoiceChangerApp

if __name__ == "__main__":
    # Khởi tạo cửa sổ chính của ứng dụng
    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
    root = ctk.CTk()
    
    # Gắn giao diện vào cửa sổ chính
    app = VoiceChangerApp(root)
    
    # Vòng lặp duy trì ứng dụng chạy liên tục
    root.mainloop()
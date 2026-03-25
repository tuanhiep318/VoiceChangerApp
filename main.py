import tkinter as tk
from src.gui import VoiceChangerApp

if __name__ == "__main__":
    # Khởi tạo cửa sổ chính của ứng dụng
    root = tk.Tk()
    
    # Gắn giao diện vào cửa sổ chính
    app = VoiceChangerApp(root)
    
    # Vòng lặp duy trì ứng dụng chạy liên tục
    root.mainloop()
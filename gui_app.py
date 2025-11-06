import customtkinter
from tkinter import filedialog
from PIL import Image
import torch
from ultralytics import YOLO
import os
from collections import Counter

# Import các thành phần cần thiết
from utils import ReUNet, val_test_transform
from main import process_image_pipeline

# --- 1. TẢI CÁC MODELS KHI KHỞI ĐỘNG ỨNG DỤNG ---
# Các model chỉ được tải một lần duy nhất để tối ưu hiệu suất.
print("Đang khởi tạo ứng dụng và tải models...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNET_MODEL_PATH = 'segmentation_chess/best_model.pth'
YOLO_MODEL_PATH = 'recognize_chess/best.pt'
if not os.path.exists(UNET_MODEL_PATH) or not os.path.exists(YOLO_MODEL_PATH):
    raise FileNotFoundError("Không tìm thấy file model. Vui lòng kiểm tra lại đường dẫn.")
# Tải model U-Net (segmentation)
unet_model = ReUNet(n_channels=3, n_classes=2).to(device)
unet_model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=device))
unet_model.eval()
# Tải model YOLO (object detection)
yolo_model = YOLO(YOLO_MODEL_PATH)
# Ánh xạ ID của các lớp sang tên quân cờ
class_names_list = ['BB', 'BK', 'BKN', 'BP', 'BQ', 'BR', 'WB', 'WK', 'WKN', 'WP', 'WQ', 'WR']
class_names_map = {index: name for index, name in enumerate(class_names_list)}
print("Tải mô hình hoàn tất. Ứng dụng sẵn sàng.")

class ChessApp(customtkinter.CTk):
    """Lớp chính của ứng dụng giao diện đồ họa (GUI)."""
    def __init__(self):
        super().__init__()

        # --- ĐỊNH NGHĨA GIAO DIỆN: MÀU SẮC & FONT CHỮ ---
        self.COLORS = {
            "bg_main": "#1A1B1E",       # Nền chính của cửa sổ
            "bg_frame": "#25262B",      # Nền cho các khung, card
            "accent": "#3B82F6",        # Màu nhấn cho các nút bấm chính
            "text_primary": "#F8F9FA",  # Màu chữ chính
            "text_secondary": "#909296" # Màu chữ phụ, chữ trạng thái
        }
        self.FONT_BODY = ("Segoe UI", 13)
        self.FONT_TITLE = ("Segoe UI", 16, "bold")
        self.FONT_SUBTITLE = ("Segoe UI", 13, "bold")

        # --- CẤU HÌNH CỬA SỔ CHÍNH ---
        self.title("Chess FEN-erator")
        self.geometry("1200x800")
        self.configure(fg_color=self.COLORS["bg_main"])

        # Biến lưu trữ dữ liệu
        self.original_pil = None
        self.processed_pil = None
        self.piece_data = []
        self._resize_job_id = None # ID cho tác vụ debounce khi resize

        # --- Cấu trúc Layout 3 phần: control, images, results ---
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure((0, 1), weight=1)

        # --- PHẦN 1: KHUNG ĐIỀU KHIỂN (TRÊN CÙNG) ---
        self.control_frame = customtkinter.CTkFrame(self, fg_color=self.COLORS["bg_frame"], corner_radius=10)
        self.control_frame.grid(row=0, column=0, columnspan=2, padx=20, pady=(20,10), sticky="ew")
        self.control_frame.grid_columnconfigure(1, weight=1)
        
        self.select_button = customtkinter.CTkButton(self.control_frame, text="Chọn ảnh bàn cờ", command=self.run_analysis, font=self.FONT_BODY, fg_color=self.COLORS["accent"], hover_color="#2563EB")
        self.select_button.grid(row=0, column=0, padx=15, pady=15)
        self.status_label = customtkinter.CTkLabel(self.control_frame, text="Vui lòng chọn một ảnh để bắt đầu.", text_color=self.COLORS["text_secondary"], font=self.FONT_BODY)
        self.status_label.grid(row=0, column=1, padx=15, pady=15, sticky="w")

        # --- PHẦN 2: KHU VỰC HIỂN THỊ ẢNH ---
        # Khung ảnh gốc (bên trái)
        self.left_frame = customtkinter.CTkFrame(self, fg_color=self.COLORS["bg_frame"], corner_radius=10)
        self.left_frame.grid(row=1, column=0, padx=(20, 10), pady=10, sticky="nsew")
        self.left_frame.grid_rowconfigure(1, weight=1)
        self.left_frame.grid_columnconfigure(0, weight=1)
        customtkinter.CTkLabel(self.left_frame, text="Ảnh Gốc", font=self.FONT_TITLE, text_color=self.COLORS["text_primary"]).grid(row=0, column=0, pady=15)
        self.original_image_label = customtkinter.CTkLabel(self.left_frame, text="")
        self.original_image_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Khung ảnh kết quả (bên phải)
        self.right_frame = customtkinter.CTkFrame(self, fg_color=self.COLORS["bg_frame"], corner_radius=10)
        self.right_frame.grid(row=1, column=1, padx=(10, 20), pady=10, sticky="nsew")
        self.right_frame.grid_rowconfigure(1, weight=1)
        self.right_frame.grid_columnconfigure(0, weight=1)
        self.result_title_label = customtkinter.CTkLabel(self.right_frame, text="Kết Quả", font=self.FONT_TITLE, text_color=self.COLORS["text_primary"])
        self.result_title_label.grid(row=0, column=0, pady=15)
        self.processed_image_label = customtkinter.CTkLabel(self.right_frame, text="")
        self.processed_image_label.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # --- PHẦN 3: KHUNG KẾT QUẢ FEN (DƯỚI CÙNG) ---
        self.bottom_frame = customtkinter.CTkFrame(self, fg_color=self.COLORS["bg_frame"], corner_radius=10)
        self.bottom_frame.grid(row=2, column=0, columnspan=2, padx=20, pady=(10, 20), sticky="ew")
        self.bottom_frame.grid_columnconfigure(1, weight=1)
        
        customtkinter.CTkLabel(self.bottom_frame, text="Chuỗi FEN", font=self.FONT_SUBTITLE, text_color=self.COLORS["text_primary"]).grid(row=0, column=0, padx=15, pady=15, sticky="w")
        self.fen_entry = customtkinter.CTkEntry(self.bottom_frame, font=self.FONT_BODY, border_color="#4A4A4A", fg_color="#2D2D2D")
        self.fen_entry.grid(row=0, column=1, padx=0, pady=15, sticky="ew")
        self.copy_fen_button = customtkinter.CTkButton(self.bottom_frame, text="Sao chép", width=100, command=self.copy_fen, font=self.FONT_BODY, fg_color=self.COLORS["accent"], hover_color="#2563EB")
        self.copy_fen_button.grid(row=0, column=2, padx=15, pady=15)

        # Gán sự kiện cho việc thay đổi kích thước cửa sổ
        self.bind("<Configure>", self.on_resize)
        
    def on_resize(self, event=None):
        """Sử dụng kỹ thuật 'debouncing' để chỉ gọi hàm resize sau khi người dùng ngừng thay đổi kích thước."""
        if self._resize_job_id: self.after_cancel(self._resize_job_id)
        self._resize_job_id = self.after(100, self._perform_resize)

    def _perform_resize(self):
        """Thực hiện việc vẽ lại các ảnh để vừa với kích thước khung mới."""
        self._resize_job_id = None
        self._update_image_display(self.original_image_label, self.original_pil)
        self._update_image_display(self.processed_image_label, self.processed_pil)

    def _update_image_display(self, label, pil_image):
        """Hàm phụ trợ để co dãn và hiển thị một ảnh PIL lên một CTkLabel."""
        if pil_image is None:
            label.configure(image=None)
            return
        w, h = label.winfo_width(), label.winfo_height()
        if w <= 1 or h <= 1: return
        img_copy = pil_image.copy()
        img_copy.thumbnail((w, h), Image.Resampling.LANCZOS)
        ctk_image = customtkinter.CTkImage(light_image=img_copy, size=img_copy.size)
        label.configure(image=ctk_image, text="")

    def copy_fen(self):
        """Sao chép chuỗi FEN trong ô textbox vào clipboard."""
        text = self.fen_entry.get()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.status_label.configure(text="Đã sao chép chuỗi FEN!")

    def run_analysis(self):
        """Hàm chính: Mở hộp thoại chọn file và chạy toàn bộ pipeline xử lý."""
        file_path = filedialog.askopenfilename(title="Chọn một ảnh bàn cờ", filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if not file_path: return

        self.status_label.configure(text=f"Đang xử lý: {os.path.basename(file_path)}...")
        self.update_idletasks() # Cập nhật giao diện ngay lập tức

        # Lưu ảnh gốc và chạy pipeline
        self.original_pil = Image.open(file_path)
        _, result_image_np, fen_string, piece_info = process_image_pipeline(
            image_path=file_path, unet_model=unet_model, yolo_model=yolo_model,
            device=device, transform=val_test_transform, class_map=class_names_map)

        # Hiển thị kết quả lên giao diện
        if result_image_np is not None:
            self.status_label.configure(text="Xử lý thành công!")
            self.result_title_label.configure(text="Kết Quả (nhấn để xem chi tiết)")
            self.processed_pil = Image.fromarray(result_image_np)
            self.piece_data = piece_info
            self._perform_resize() # Vẽ lại ảnh
            self.fen_entry.delete(0, "end")
            self.fen_entry.insert(0, fen_string)
            self.processed_image_label.bind("<Button-1>", self.show_popup_image)
        else:
            self.status_label.configure(text="Xử lý thất bại. Vui lòng thử ảnh khác.")
            self.processed_pil, self.piece_data = None, []
            self._perform_resize()

    def show_popup_image(self, event=None):
        """Hiển thị một cửa sổ pop-up với ảnh kết quả và bảng thống kê quân cờ."""
        if not self.processed_pil: return
        
        popup = customtkinter.CTkToplevel(self)
        popup.title("Ảnh kết quả và Thống kê quân cờ")
        popup.transient(self)
        popup.configure(fg_color=self.COLORS["bg_main"])
        popup.grid_columnconfigure(0, weight=3) # Cột ảnh lớn hơn
        popup.grid_columnconfigure(1, weight=2) # Cột thống kê nhỏ hơn
        popup.grid_rowconfigure(0, weight=1)

        # Cột trái: Ảnh
        img_copy = self.processed_pil.copy()
        img_copy.thumbnail((800, 800), Image.Resampling.LANCZOS)
        full_image = customtkinter.CTkImage(light_image=img_copy, size=img_copy.size)
        customtkinter.CTkLabel(popup, image=full_image, text="").grid(row=0, column=0, padx=20, pady=20, sticky="nsew")

        # Cột phải: Thống kê
        stats_frame = customtkinter.CTkFrame(popup, fg_color=self.COLORS["bg_frame"], corner_radius=10)
        stats_frame.grid(row=0, column=1, padx=(0, 20), pady=20, sticky="nsew")
        stats_frame.grid_columnconfigure(0, weight=1)
        stats_frame.grid_rowconfigure(2, weight=1)
        customtkinter.CTkLabel(stats_frame, text="Thống kê quân cờ", font=self.FONT_TITLE).grid(row=0, column=0, padx=15, pady=15, sticky="w")
        total_pieces = len(self.piece_data)
        customtkinter.CTkLabel(stats_frame, text=f"Tổng số quân cờ: {total_pieces}", font=self.FONT_SUBTITLE).grid(row=1, column=0, padx=15, pady=5, sticky="w")
        
        piece_counts = Counter(p['name'] for p in self.piece_data)
        white_pieces = {k: v for k, v in sorted(piece_counts.items()) if k.startswith('W')}
        black_pieces = {k: v for k, v in sorted(piece_counts.items()) if k.startswith('B')}
        stats_text = "--- QUÂN TRẮNG ---\n" + ("\n".join([f"{name}: {count}" for name, count in white_pieces.items()]) if white_pieces else "Không có.")
        stats_text += "\n\n--- QUÂN ĐEN ---\n" + ("\n".join([f"{name}: {count}" for name, count in black_pieces.items()]) if black_pieces else "Không có.")
        
        textbox = customtkinter.CTkTextbox(stats_frame, font=self.FONT_BODY, fg_color="#2D2D2D", activate_scrollbars=True)
        textbox.grid(row=2, column=0, sticky="nsew", padx=15, pady=15)
        textbox.insert("1.0", stats_text)
        textbox.configure(state="disabled")

        # Căn giữa và đặt kích thước cho popup
        popup.update_idletasks()
        w, h = min(img_copy.width + 500, 1600), min(img_copy.height + 100, 900)
        x = self.winfo_x() + (self.winfo_width() - w) // 2
        y = self.winfo_y() + (self.winfo_height() - h) // 2
        popup.geometry(f"{w}x{h}+{x}+{y}")
        popup.grab_set()

if __name__ == "__main__":
    customtkinter.set_appearance_mode("Dark")
    app = ChessApp()
    app.mainloop()

# Chess FEN-erator

Ứng dụng Desktop (Python + CustomTkinter) cho nhận diện bàn cờ và xuất FEN.

## Cài đặt

### 1) Yêu cầu
- Python 3.10+ (khuyến nghị 3.11)
- Windows (đã kiểm thử). Linux/MacOS có thể cần điều chỉnh UI
- (Tùy chọn) GPU CUDA để tăng tốc PyTorch

### 2) Tạo môi trường và cài thư viện
```bash
# Tạo virtual env
python -m venv .venv

# Kích hoạt
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/MacOS

# Cài thư viện
pip install -r requirements.txt
```

Lưu ý:
- Nếu dùng GPU, cài torch/torchvision theo hướng dẫn chính thức của PyTorch (phù hợp CUDA) thay vì bản CPU trong requirements.

### 3) Chuẩn bị model
- Đặt trọng số vào đúng vị trí:
  - Segmentation (U-Net): `segmentation_chess/best_model.pth`
  - Detection (YOLO): `recognize_chess/best.pt`
- Nếu tên file khác, chỉnh trong `gui_app.py`:
```python
UNET_MODEL_PATH = 'segmentation_chess/best_model.pth'
YOLO_MODEL_PATH = 'recognize_chess/best.pt'
```

### 4) Chạy ứng dụng
```bash
python gui_app.py
```
- Nhấn “Chọn ảnh bàn cờ” để chọn ảnh `.jpg/.jpeg/.png`.
- Ảnh gốc hiển thị bên trái; ảnh kết quả bên phải.
- Nhấn vào ảnh kết quả để mở cửa sổ xem lớn và thống kê.
- Chuỗi FEN hiển thị ở thanh dưới cùng, có nút “Sao chép”.

### 5) Gợi ý ảnh đầu vào
- Ảnh đủ sáng, ít bóng đổ, bàn cờ rõ; hạn chế che khuất rìa.
- Ứng dụng có nới vùng warp (padding 5%) để giảm cắt rìa, nhưng ảnh nguồn tốt sẽ cho kết quả ổn định hơn.

## Các file chính
- `gui_app.py`: Giao diện người dùng; tải model, chọn ảnh, hiển thị kết quả, pop-up thống kê.
- `main.py`: Pipeline xử lý (U-Net → warp → YOLO → ánh xạ → sinh FEN).
- `utils.py`: Định nghĩa ReUNet, `order_points`, và transform inference.
- `fen.py`: Hàm `board_to_fen(board_state)` chuyển ma trận 8x8 sang chuỗi FEN.
- `requirements.txt`: Danh sách thư viện cần cài.
- `recognize_chess/`: Thư mục chứa YOLO weights (`best.pt`).
- `segmentation_chess/`: Thư mục chứa U-Net weights (`best_model.pth`).

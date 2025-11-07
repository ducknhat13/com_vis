import torch
import torch.nn as nn
import timm
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# --- CÁC MODULE CON CỦA KIẾN TRÚC U-NET ---

class FirstFeature(nn.Module):
    """Lớp CNN đầu tiên để trích xuất các đặc trưng cơ bản từ ảnh đầu vào."""
    def __init__(self, in_channels):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU())

    def forward(self, X):
        return self.conv2d(X)

class FinalOutput(nn.Module):
    """Lớp CNN cuối cùng để tạo ra output có số kênh bằng số lớp cần phân loại."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, X):
        return self.conv2d(X)

class Decoder(nn.Module):
    """
    Một khối Decoder trong kiến trúc U-Net.
    Nó nhận feature map từ lớp trước, tăng kích thước (upsample),
    và kết hợp với feature map từ nhánh encoder tương ứng (skip connection).
    """
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Tích chập trên feature map được kết hợp từ lớp trước và skip connection
        self.conv2d = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
        # Phép tăng kích thước (Upsampling)
        self.up_sample = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, X, skip):
        X_up = self.up_sample(X)
        # Kết hợp (concatenate) theo chiều kênh (channel)
        X_cat = torch.cat([X_up, skip], dim=1)
        return self.conv2d(X_cat)

# --- KIẾN TRÚC U-NET CHÍNH ---

class ReUNet(nn.Module):
    """
    Kiến trúc U-Net hoàn chỉnh, sử dụng ResNet101 làm backbone (Encoder).
    - Encoder (Nhánh xuống): Sử dụng các lớp của ResNet101 đã được huấn luyện trước
      để trích xuất các đặc trưng phân cấp từ ảnh.
    - Decoder (Nhánh lên): Xây dựng lại segmentation map từ các đặc trưng đã trích xuất,
      kết hợp với thông tin từ encoder thông qua các skip connection.
    """
    def __init__(self, n_channels, n_classes):
        super().__init__()

        self.first_feature = FirstFeature(n_channels)
        # Sử dụng ResNet101 từ thư viện 'timm', chỉ lấy các feature map
        self.resnet = timm.create_model('resnet101', pretrained=True, features_only=True)

        # Các khối Decoder tương ứng với các cấp của Encoder
        # ResNet101 feature maps: (64, 256, 512, 1024, 2048)
        self.decoder1 = Decoder(2048, 1024, 1024)
        self.decoder2 = Decoder(1024, 512, 512)
        self.decoder3 = Decoder(512, 256, 256)
        self.decoder4 = Decoder(256, 64, 64)
        self.decoder5 = Decoder(64, n_channels, 32)  # Khối decoder cuối cùng kết hợp với ảnh gốc

        self.final_output = FinalOutput(32, n_classes)

    def forward(self, X):
        X1 = self.first_feature(X)

        # Encoder: Lấy các feature map từ ResNet
        E1, E2, E3, E4, E5 = self.resnet(X1)

        # Decoder: Xây dựng lại ảnh từ các feature map
        D1 = self.decoder1(E5, E4)
        D2 = self.decoder2(D1, E3)
        D3 = self.decoder3(D2, E2)
        D4 = self.decoder4(D3, E1)
        D5 = self.decoder5(D4, X) # Skip connection với ảnh đầu vào

        # Lớp output cuối cùng
        return self.final_output(D5)

# --- HÀM PHỤ TRỢ ---

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    Sắp xếp lại 4 điểm góc của một tứ giác theo thứ tự:
    trên-trái, trên-phải, dưới-phải, dưới-trái.
    """
    rect = np.zeros((4, 2), dtype="float32")
    
    # Điểm trên-trái có tổng (x+y) nhỏ nhất
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    # Điểm dưới-phải có tổng (x+y) lớn nhất
    rect[2] = pts[np.argmax(s)]

    # Điểm trên-phải có hiệu (x-y) lớn nhất
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmax(diff)]
    # Điểm dưới-trái có hiệu (x-y) nhỏ nhất
    rect[3] = pts[np.argmin(diff)]

    return rect

# --- CÁC PHÉP BIẾN ĐỔI ẢNH (TRANSFORMS) ---

size = 640
# Transform cho ảnh validation và test: chỉ resize và chuẩn hóa
val_test_transform = A.Compose([
    A.Resize(width=size, height=size),
    A.Normalize(mean=[0.5]*3, std=[0.5]*3),
    ToTensorV2()
], is_check_shapes=False)
<<<<<<< HEAD


def letterbox_for_pipeline(image, new_shape=(640, 640), color=(114, 114, 114)):
    """
    Thay đổi kích thước và đệm ảnh để đạt được hình dạng mong muốn mà không làm thay đổi tỷ lệ khung hình.

    Args:
        image (np.ndarray): Ảnh đầu vào.
        new_shape (tuple): Kích thước mong muốn (height, width).
        color (tuple): Màu sắc cho vùng đệm.

    Returns:
        tuple: Một tuple chứa (ảnh đã đệm, tỷ lệ co giãn, (đệm_w, đệm_h)).
    """
    h, w, _ = image.shape
    new_h, new_w = new_shape

    # Tính toán tỷ lệ co giãn
    ratio = min(new_h / h, new_w / w)
    
    # Kích thước mới
    scaled_w, scaled_h = int(w * ratio), int(h * ratio)
    
    # Thay đổi kích thước
    if h != scaled_h or w != scaled_w:
        resized_image = cv2.resize(image, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized_image = image
    
    # Tạo ảnh đã đệm
    padded_image = np.full((new_h, new_w, 3), color, dtype=np.uint8)
    
    # Tính toán lề đệm
    pad_h = (new_h - scaled_h) // 2
    pad_w = (new_w - scaled_w) // 2
    
    # Đặt ảnh đã thay đổi kích thước vào giữa
    padded_image[pad_h:pad_h + scaled_h, pad_w:pad_w + scaled_w] = resized_image
    
    return padded_image, ratio, (pad_w, pad_h)
=======
>>>>>>> 53e24a74f082b3e0459c7a9c6953b6d0f527ea50

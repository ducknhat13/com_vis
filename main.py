import cv2
import torch
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO
import os

# Import các thành phần từ utils.py và fen.py
from utils import ReUNet, order_points, letterbox_for_pipeline
from fen import board_to_fen

def process_image_pipeline(image_path: str, unet_model: torch.nn.Module, yolo_model: YOLO,
                             device: torch.device, transform: any, class_map: dict,
                             output_dir="output", visualize=True):
    """
    Pipeline xử lý ảnh bàn cờ dựa trên quy trình từ notebook.
    1. Lấy mask từ U-Net.
    2. Tìm 4 góc từ mask.
    3. Tính ma trận biến đổi M.
    4. Tạo ảnh letterbox cho YOLO và chạy nhận diện.
    5. Chuyển đổi tọa độ quân cờ về ảnh gốc.
    6. Ánh xạ các quân cờ lên bàn cờ 8x8 đã làm phẳng.
    7. Tạo chuỗi FEN.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Đọc ảnh gốc ---
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        print(f"Lỗi: Không thể đọc ảnh từ: {image_path}")
        return None, None, None, None
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # ============================================
    # BƯỚC 1: LẤY MASK TỪ U-NET
    # ============================================
    unet_model.eval()
    with torch.no_grad():
        augmented = transform(image=original_image_rgb)
        image_tensor = augmented["image"].unsqueeze(0).to(device)
        mask_hat = unet_model(image_tensor)
        mask_hat = F.interpolate(mask_hat, size=image_tensor.shape[-2:], mode='bilinear', align_corners=False)
        pred_mask = torch.argmax(mask_hat, dim=1)[0].cpu().numpy()

    # ============================================
    # BƯỚC 2: TÌM 4 GÓC TỪ MASK
    # ============================================
    mask_8bit = (pred_mask * 255).astype('uint8')
    contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"Lỗi: Không tìm thấy contour nào trong ảnh {image_path}")
        return None, None, None, None
    board_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(board_contour, True)
    corners = cv2.approxPolyDP(board_contour, epsilon, True)
    if len(corners) != 4:
        print(f"Cảnh báo: Không tìm thấy chính xác 4 góc (tìm thấy {len(corners)}).")
    
    # ============================================
    # BƯỚC 3: TÍNH TOÁN MA TRẬN BIẾN ĐỔI (M)
    # ============================================
    corners_2d = corners.reshape(-1, 2).astype('float32')
    src_points = order_points(corners_2d)
    width, height = 640, 640
    dst_points = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # ============================================
    # BƯỚC 4: LẤY DETECTIONS từ YOLO (trên ảnh đã letterbox)
    # ============================================
    # Tạo ảnh letterbox để giữ tỷ lệ khung hình
    image_letterboxed, ratio, (pad_w, pad_h) = letterbox_for_pipeline(original_image_rgb, new_shape=(width, height))
    detections = yolo_model.predict(image_letterboxed, iou=0.3, verbose=False)

    anchor_points_original_space = []
    labels = []
    for detection in detections[0].boxes:
        x_min, y_min, x_max, y_max = detection.xyxy[0].tolist()
        class_id = int(detection.cls[0].tolist())
        
        # Chuyển đổi tọa độ từ không gian letterbox về không gian ảnh gốc
        x_min_orig = (x_min - pad_w) / ratio
        y_min_orig = (y_min - pad_h) / ratio
        x_max_orig = (x_max - pad_w) / ratio
        y_max_orig = (y_max - pad_h) / ratio

        # Lấy điểm neo ở giữa bounding box
        anchor_x = int((x_min_orig + x_max_orig) / 2)
        anchor_y = int((y_min_orig + y_max_orig) / 2)
        
        anchor_points_original_space.append([anchor_x, anchor_y])
        labels.append(class_id)

    # ============================================
    # BƯỚC 5: ÁNH XẠ LÊN BÀN CỜ 8x8
    # ============================================
    warped_image = cv2.warpPerspective(original_image, M, (width, height))
    if not anchor_points_original_space:
        print("YOLO không phát hiện quân cờ nào.")
        empty_board = np.full((8, 8), "  ", dtype=object)
        fen = board_to_fen(empty_board)
        return empty_board, warped_image, fen, []

    np_anchor_points = np.array(anchor_points_original_space, dtype=np.float32).reshape(-1, 1, 2)
    transformed_points = cv2.perspectiveTransform(np_anchor_points, M)

    square_size = width / 8.0
    board_state = np.full((8, 8), "  ", dtype=object)
    vis_image_with_names = warped_image.copy()
    piece_info_list = []
    cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    rows = ['8', '7', '6', '5', '4', '3', '2', '1']

    for (point, class_id) in zip(transformed_points, labels):
        x, y = point[0]
        col_index = int(x // square_size)
        row_index = int(y // square_size)

        if 0 <= col_index < 8 and 0 <= row_index < 8:
            piece_name = class_map.get(class_id, "Unk")
            square_name = cols[col_index] + rows[row_index]
            
            if board_state[row_index, col_index] != "  ":
                print(f"⚠️ Cảnh báo: Ghi đè {board_state[row_index, col_index]} "
                      f"với {piece_name} tại ô [{row_index}, {col_index}]")
                board_state[row_index, col_index] += "*" # Đánh dấu ô bị ghi đè
            else:
                 board_state[row_index, col_index] = piece_name

            piece_info_list.append({
                "name": piece_name, "square": square_name, "coords": (x, y)
            })

            if visualize:
                cv2.circle(vis_image_with_names, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(vis_image_with_names, piece_name, (int(x) + 10, int(y) + 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # ============================================
    # BƯỚC 6: TẠO FEN VÀ LƯU ẢNH
    # ============================================
    final_fen = board_to_fen(board_state)
    
    output_image_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_image_path, cv2.cvtColor(vis_image_with_names, cv2.COLOR_RGB2BGR))
    print(f"Đã lưu ảnh kết quả tại: {output_image_path}")

    return board_state, vis_image_with_names, final_fen, piece_info_list

# Khối `if __name__ == '__main__':` dùng để chạy thử nghiệm pipeline trên một ảnh đơn lẻ
# khi thực thi trực tiếp file này. Logic chính của ứng dụng nằm ở `gui_app.py`.
if __name__ == '__main__':
    # Cài đặt và tải models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sử dụng thiết bị: {device}")
    UNET_MODEL_PATH = 'segmentation_chess/best_model.pth'
    YOLO_MODEL_PATH = 'recognize_chess/best.pt'
    if not os.path.exists(UNET_MODEL_PATH) or not os.path.exists(YOLO_MODEL_PATH):
        print("Lỗi: Không tìm thấy file model. Vui lòng kiểm tra lại đường dẫn.")
        exit()

    print("Đang load mô hình U-Net...")
    unet_model = ReUNet(n_channels=3, n_classes=2).to(device)
    unet_model.load_state_dict(torch.load(UNET_MODEL_PATH, map_location=device))
    unet_model.eval()

    print("Đang load mô hình YOLO...")
    yolo_model = YOLO(YOLO_MODEL_PATH)

    class_names_list = ['BB', 'BK', 'BKN', 'BP', 'BQ', 'BR', 'WB', 'WK', 'WKN', 'WP', 'WQ', 'WR']
    class_names_map = {index: name for index, name in enumerate(class_names_list)}
    print("Load mô hình hoàn tất.")

    # Đường dẫn đến ảnh ví dụ để thử nghiệm
    EXAMPLE_IMAGE_PATH = 'recognize_chess/content/test/images/IMG20210302182922_jpg.rf.0c45fed50ae0023ddb9ee60939f62982.jpg'
    if not os.path.exists(EXAMPLE_IMAGE_PATH):
        print(f"Lỗi: Không tìm thấy ảnh ví dụ tại '{EXAMPLE_IMAGE_PATH}'")
        exit()

    # Chạy pipeline
    final_board_state, final_image, final_fen, _ = process_image_pipeline(
        image_path=EXAMPLE_IMAGE_PATH,
        unet_model=unet_model,
        yolo_model=yolo_model,
        device=device,
        transform=val_test_transform,
        class_map=class_names_map,
        visualize=True
    )

    # In kết quả
    if final_board_state is not None:
        print("\n--- Trạng thái bàn cờ ---")
        print(final_board_state)
        print("\n--- Chuỗi FEN ---")
        print(final_fen)
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 10))
            plt.imshow(final_image)
            plt.title(f"Kết quả xử lý cho: {os.path.basename(EXAMPLE_IMAGE_PATH)}")
            plt.axis('off')
            plt.show()
        except ImportError:
            print("\nMatplotlib chưa được cài. Ảnh kết quả đã được lưu trong thư mục 'output'.")

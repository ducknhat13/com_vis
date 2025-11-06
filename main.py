import cv2
import torch
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO
import os

# Import các thành phần từ utils.py và fen.py
from utils import ReUNet, order_points, val_test_transform
from fen import board_to_fen

def process_image_pipeline(image_path: str, unet_model: torch.nn.Module, yolo_model: YOLO,
                             device: torch.device, transform: any, class_map: dict,
                             output_dir="output", visualize=True):
    """
    Pipeline hoàn chỉnh để xử lý một ảnh bàn cờ.

    Quy trình hoạt động:
    1.  **Đọc ảnh gốc.**
    2.  **U-Net (Tìm bàn cờ):** Dùng model U-Net để tạo segmentation mask, từ đó
        tìm ra 4 điểm góc của bàn cờ trên ảnh gốc.
    3.  **Warping (Làm phẳng):** Dựa trên 4 góc, tính toán ma trận biến đổi phối cảnh (perspective transform)
        để "làm phẳng" bàn cờ về góc nhìn từ trên xuống. Đồng thời, nới rộng các góc
        để tránh cắt mất quân cờ ở rìa.
    4.  **YOLO (Nhận diện quân cờ):** Dùng model YOLO trên **ảnh gốc** để phát hiện tất cả
        các quân cờ và lấy tọa độ bounding box của chúng.
    5.  **Ánh xạ & Tổng hợp:** "Ánh xạ" tọa độ của các quân cờ (tìm được trên ảnh gốc)
        lên ảnh đã làm phẳng để xác định vị trí của chúng trên bàn cờ 8x8.
    6.  **Tạo FEN:** Chuyển đổi trạng thái bàn cờ 8x8 thành chuỗi FEN.
    7.  **Trả về kết quả:** Trả về trạng thái bàn cờ, ảnh đã xử lý, chuỗi FEN, và
        thông tin chi tiết về từng quân cờ.

    Args:
        image_path (str): Đường dẫn đến file ảnh đầu vào.
        unet_model (torch.nn.Module): Model ReUNet đã được tải.
        yolo_model (YOLO): Model YOLO đã được tải.
        device (torch.device): Thiết bị để chạy model ('cuda' hoặc 'cpu').
        transform (any): Phép biến đổi (albumentations) cho ảnh đầu vào của U-Net.
        class_map (dict): Từ điển ánh xạ class ID của YOLO sang tên quân cờ.
        output_dir (str): Thư mục để lưu ảnh kết quả.
        visualize (bool): Nếu True, sẽ vẽ các nhãn lên ảnh kết quả.

    Returns:
        tuple: Một tuple chứa (board_state, warped_image_with_labels, fen_string, piece_info_list).
               Trả về (None, None, None, None) nếu có lỗi xảy ra.
    """
    # Tạo thư mục output nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    
    # --- BƯỚC 1: ĐỌC ẢNH GỐC ---
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_image is None:
        print(f"Lỗi: Không thể đọc ảnh từ đường dẫn: {image_path}")
        return None, None, None, None
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # --- BƯỚC 2: U-NET - TÌM 4 GÓC BÀN CỜ ---
    unet_model.eval()
    with torch.no_grad():
        augmented = transform(image=original_image)
        image_tensor = augmented["image"].unsqueeze(0).to(device)
        mask_hat = unet_model(image_tensor)
        mask_hat = F.interpolate(mask_hat, size=image_tensor.shape[-2:], mode='bilinear', align_corners=False)
        pred_mask = torch.argmax(mask_hat, dim=1)[0].cpu().numpy()

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
    
    # --- BƯỚC 3: TÍNH TOÁN MA TRẬN WARP (CÓ NỚI RỘNG) ---
    if corners.shape[1] == 1:
        corners_2d = corners.reshape(-1, 2)
    else:
        corners_2d = corners.astype('float32')
    src_points = order_points(corners_2d)

    # Nới rộng vùng warp để tránh cắt mất quân cờ ở rìa do hiệu ứng phối cảnh
    padding_percent = 0.05
    board_width = np.linalg.norm(src_points[0] - src_points[1])
    board_height = np.linalg.norm(src_points[0] - src_points[3])
    padding_pixels = (board_width + board_height) / 2 * padding_percent
    center = src_points.mean(axis=0)
    padded_src_points = []
    for point in src_points:
        vector = point - center
        normalized_vector = vector / np.linalg.norm(vector)
        new_point = point + normalized_vector * padding_pixels
        padded_src_points.append(new_point)
    padded_src_points = np.array(padded_src_points, dtype=np.float32)
    
    width, height = 640, 640
    dst_points = np.array([
        [0, 0], [width - 1, 0],
        [width - 1, height - 1], [0, height - 1]
    ], dtype="float32")
    M = cv2.getPerspectiveTransform(padded_src_points, dst_points)
    warped_image = cv2.warpPerspective(original_image, M, (width, height))

    # --- BƯỚC 4: YOLO - NHẬN DIỆN QUÂN CỜ TRÊN ẢNH GỐC ---
    detections = yolo_model.predict(source=image_path, verbose=False)

    anchor_points = []
    labels = []
    for detection in detections[0].boxes:
        x_min, y_min, x_max, y_max = detection.xyxy[0].tolist()
        class_id = int(detection.cls[0].tolist())
        # Lấy điểm neo ở giữa-đáy của bounding box
        anchor_x = int((x_min + x_max) / 2)
        anchor_y = int(y_max)
        anchor_points.append([anchor_x, anchor_y])
        labels.append(class_id)

    # --- BƯỚC 5: ÁNH XẠ TỌA ĐỘ VÀ TỔNG HỢP KẾT QUẢ ---
    if not anchor_points:
        print("YOLO không phát hiện quân cờ nào.")
        return np.full((8, 8), "  ", dtype=object), warped_image, "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", []

    np_anchor_points = np.array(anchor_points, dtype=np.float32).reshape(-1, 1, 2)
    # Dùng ma trận M để chuyển tọa độ các điểm neo từ ảnh gốc sang ảnh phẳng
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
            board_state[row_index, col_index] = piece_name
            square_name = cols[col_index] + rows[row_index]
            piece_info_list.append({
                "name": piece_name,
                "square": square_name,
                "coords": (x, y)
            })

            if visualize:
                # Vẽ nhãn và điểm đỏ lên ảnh kết quả
                cv2.putText(vis_image_with_names, piece_name, (int(x) -15, int(y) - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                cv2.circle(vis_image_with_names, (int(x), int(y)), 5, (255, 0, 0), -1)
    
    # --- BƯỚC 6: TẠO FEN VÀ LƯU ẢNH ---
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

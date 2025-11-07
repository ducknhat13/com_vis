import cv2
import torch
import numpy as np
import torch.nn.functional as F
from ultralytics import YOLO
import os

# Import các thành phần trợ giúp từ các file khác trong dự án.
# ReUNet: Kiến trúc model U-Net đã được định nghĩa.
# order_points: Hàm sắp xếp 4 điểm góc của một tứ giác.
# letterbox_for_pipeline: Hàm thay đổi kích thước ảnh và thêm padding (đệm) để giữ đúng tỷ lệ.
from utils import ReUNet, order_points, letterbox_for_pipeline
# board_to_fen: Hàm chuyển đổi ma trận trạng thái bàn cờ 8x8 thành chuỗi FEN.
from fen import board_to_fen

def process_image_pipeline(image_path: str, unet_model: torch.nn.Module, yolo_model: YOLO,
                             device: torch.device, transform: any, class_map: dict,
                             output_dir="output", visualize=True):
    """
    Pipeline xử lý ảnh theo logic tối ưu: YOLO trước -> Transform tọa độ -> Ánh xạ.
    Quy trình này đảm bảo tính nhất quán của các hệ tọa độ để có kết quả chính xác nhất.

    Args:
        image_path (str): Đường dẫn đến file ảnh.
        unet_model: Model U-Net đã được load.
        yolo_model: Model YOLO đã được load.
        device: 'cuda' hoặc 'cpu'.
        transform: Các phép biến đổi cho U-Net (chủ yếu là normalize và to tensor).
        class_map (dict): Từ điển ánh xạ class_id sang tên quân cờ.
        output_dir (str): Thư mục để lưu ảnh kết quả.
        visualize (bool): Cờ để quyết định có vẽ thông tin lên ảnh kết quả hay không.

    Returns:
        board_state (np.ndarray): Ma trận (8, 8) biểu diễn trạng thái bàn cờ.
        vis_image (np.ndarray): Ảnh bàn cờ đã làm phẳng và được vẽ các thông tin trực quan.
        final_fen (str): Chuỗi FEN tương ứng với trạng thái bàn cờ.
        piece_info_list (list): Danh sách thông tin chi tiết của các quân cờ.
    """
    # Tạo thư mục đầu ra nếu chưa tồn tại
    os.makedirs(output_dir, exist_ok=True)
    debug = False # Tắt các thông báo debug chi tiết

    # ===================================================
    # BƯỚC 1: ĐỌC ẢNH GỐC
    # ===================================================
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"❌ Không đọc được ảnh: {image_path}")
        return None, None, None, None

    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # ===================================================
    # BƯỚC 2: YOLO DETECT TỪ FILE GỐC (YOLO tự letterbox)
    # ===================================================
    detections = yolo_model.predict(source=image_path, conf=0.25, iou=0.3, verbose=False)

    num_pieces = len(detections[0].boxes) if detections else 0
    piece_detections = []
    if num_pieces > 0:
        for detection in detections[0].boxes:
            x_min, y_min, x_max, y_max = detection.xyxy[0].tolist()
            class_id = int(detection.cls[0].item())
            confidence = float(detection.conf[0].item())
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            piece_name = class_map.get(class_id, "??")
            piece_detections.append({
                'class_id': class_id, 'piece': piece_name, 'conf': confidence,
                'center': (center_x, center_y), 'bbox': (x_min, y_min, x_max, y_max)
            })

    # ===================================================
    # BƯỚC 3: U-NET TÌM BÀN CỜ TRÊN ẢNH LETTERBOX
    # ===================================================
    # Tạo ảnh letterbox thủ công để đảm bảo U-Net và YOLO làm việc trên cùng hệ tọa độ
    letterboxed_image, _, _, _ = letterbox_for_pipeline(original_rgb, 640)

    unet_model.eval()
    with torch.no_grad():
        augmented = transform(image=letterboxed_image)
        image_tensor = augmented["image"].unsqueeze(0).to(device)

        mask_hat = unet_model(image_tensor)
        mask_hat = F.interpolate(mask_hat, size=image_tensor.shape[-2:], mode='bilinear', align_corners=False)
        pred_mask = torch.argmax(mask_hat, dim=1)[0].cpu().numpy()

    # ===================================================
    # BƯỚC 4: TÌM 4 GÓC BÀN CỜ TỪ MASK
    # ===================================================
    mask_8bit = (pred_mask * 255).astype('uint8')
    contours, _ = cv2.findContours(mask_8bit, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("❌ U-Net không tìm thấy đường viền bàn cờ.")
        return None, None, None, None

    board_contour = max(contours, key=cv2.contourArea)
    epsilon = 0.02 * cv2.arcLength(board_contour, True)
    approx = cv2.approxPolyDP(board_contour, epsilon, True)

    if len(approx) != 4:
        points = approx.reshape(-1, 2)
        rect = cv2.minAreaRect(points)
        box = cv2.boxPoints(rect)
        approx = box.reshape(-1, 1, 2)

    corners = approx.reshape(-1, 2).astype(np.float32)
    src_points = order_points(corners)

    # ===================================================
    # BƯỚC 5: TÍNH MA TRẬN BIẾN ĐỔI PHỐI CẢNH (M)
    # ===================================================
    board_size = 640
    dst_points = np.array([[0, 0], [board_size - 1, 0], [board_size - 1, board_size - 1], [0, board_size - 1]], dtype=np.float32)
    
    # Ma trận này sẽ biến đổi từ không gian ảnh letterbox -> không gian bàn cờ phẳng.
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Tạo ảnh bàn cờ phẳng để trực quan hóa.
    warped_board = cv2.warpPerspective(letterboxed_image, M, (board_size, board_size))

    # ===================================================
    # BƯỚC 6: ÁNH XẠ TỌA ĐỘ YOLO LÊN BÀN CỜ PHẲNG
    # ===================================================
    square_size = board_size / 8.0
    square_detections = {}

    for det in piece_detections:
        # Tọa độ tâm của quân cờ trong không gian ảnh letterbox của YOLO.
        x_letterbox, y_letterbox = det['center']
        
        # Áp dụng ma trận M để chuyển tọa độ sang không gian bàn cờ phẳng.
        point = np.array([[[x_letterbox, y_letterbox]]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, M)
        x_board, y_board = transformed[0][0]
        
        # Tính toán vị trí hàng, cột trên lưới 8x8.
        col = int(x_board / square_size)
        row = int(y_board / square_size)

        # Lưu lại quân cờ nếu nó nằm trong bàn cờ.
        if 0 <= row < 8 and 0 <= col < 8:
            key = (row, col)
            if key not in square_detections:
                square_detections[key] = []
            square_detections[key].append({
                'piece': det['piece'], 'conf': det['conf'],
                'x': int(x_board), 'y': int(y_board)
            })

    # ===================================================
    # BƯỚC 7: XỬ LÝ TRÙNG LẶP VÀ TẠO TRẠNG THÁI BÀN CỜ
    # ===================================================
    board_state = np.full((8, 8), "  ", dtype=object)
    vis_image = warped_board.copy()
    piece_info_list = []
    cols = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    rows = ['8', '7', '6', '5', '4', '3', '2', '1']

    for (row, col), pieces in square_detections.items():
        if not pieces: continue
        
        # Chọn quân cờ có độ tự tin cao nhất trong ô.
        best = max(pieces, key=lambda x: x['conf'])
        board_state[row, col] = best['piece']

        # Thu thập thông tin quân cờ để trả về.
        x, y = best['x'], best['y']
        square_name = cols[col] + rows[row]
        piece_info_list.append({
            "name": best['piece'], "square": square_name, "coords": (x, y)
        })

        # Vẽ lên ảnh kết quả nếu được yêu cầu.
        if visualize:
            cv2.circle(vis_image, (x, y), 6, (0, 255, 0), -1)
            cv2.circle(vis_image, (x, y), 8, (0, 0, 0), 2)
            text = best['piece']
            cv2.putText(vis_image, text, (x + 10, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # ===================================================
    # BƯỚC 8: TẠO FEN, LƯU ẢNH VÀ TRẢ VỀ KẾT QUẢ
    # ===================================================
    final_fen = board_to_fen(board_state)
    
    # Lưu ảnh kết quả
    output_image_path = os.path.join(output_dir, os.path.basename(image_path))
    cv2.imwrite(output_image_path, cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR))
    print(f"Đã lưu ảnh kết quả tại: {output_image_path}")

    return board_state, vis_image, final_fen, piece_info_list

# Khối `if __name__ == '__main__':` dùng để chạy thử nghiệm pipeline trên một ảnh đơn lẻ
# khi thực thi trực tiếp file này. Logic chính của ứng dụng nằm ở `gui_app.py`.
if __name__ == '__main__':
    # --- THIẾT LẬP MÔI TRƯỜNG VÀ TẢI MODELS ---
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

    # Đường dẫn đến ảnh ví dụ để thử nghiệm.
    EXAMPLE_IMAGE_PATH = 'recognize_chess/content/test/images/IMG20210302182922_jpg.rf.0c45fed50ae0023ddb9ee60939f62982.jpg'
    if not os.path.exists(EXAMPLE_IMAGE_PATH):
        print(f"Lỗi: Không tìm thấy ảnh ví dụ tại '{EXAMPLE_IMAGE_PATH}'")
        exit()
    
    # Cần có transform từ utils.py để chạy độc lập.
    from utils import val_test_transform

    # --- CHẠY PIPELINE ---
    final_board_state, final_image, final_fen, _ = process_image_pipeline(
        image_path=EXAMPLE_IMAGE_PATH,
        unet_model=unet_model,
        yolo_model=yolo_model,
        device=device,
        transform=val_test_transform,
        class_map=class_names_map,
        visualize=True
    )

    # --- IN KẾT QUẢ RA CONSOLE VÀ HIỂN THỊ ẢNH ---
    if final_board_state is not None:
        print("\n--- Trạng thái bàn cờ ---")
        print(final_board_state)
        print("\n--- Chuỗi FEN ---")
        print(final_fen)
        try:
            # Thử hiển thị ảnh kết quả bằng matplotlib nếu đã cài.
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 10))
            plt.imshow(final_image)
            plt.title(f"Kết quả xử lý cho: {os.path.basename(EXAMPLE_IMAGE_PATH)}")
            plt.axis('off')
            plt.show()
        except ImportError:
            print("\nMatplotlib chưa được cài. Ảnh kết quả đã được lưu trong thư mục 'output'.")

import numpy as np

def board_to_fen(board_state: np.ndarray) -> str:
    """
    Chuyển đổi một mảng numpy 8x8 đại diện cho trạng thái bàn cờ sang chuỗi FEN.

    FEN (Forsyth-Edwards Notation) là một định dạng chuẩn để mô tả một thế cờ.
    - Quân trắng được biểu thị bằng chữ hoa (P, N, B, R, Q, K).
    - Quân đen được biểu thị bằng chữ thường (p, n, b, r, q, k).
    - Các ô trống liên tiếp được biểu thị bằng một con số (1-8).
    - Mỗi hàng được phân cách bởi một dấu '/'.

    Args:
        board_state (np.ndarray): Mảng numpy 8x8, mỗi phần tử là một chuỗi 
                                  đại diện cho quân cờ (ví dụ: 'WP', 'BK') 
                                  hoặc "  " cho ô trống.

    Returns:
        str: Chuỗi FEN hoàn chỉnh mô tả thế cờ.
    """
    fen_rows = []
    # Duyệt qua từng hàng của ma trận trạng thái bàn cờ
    for row in board_state:
        fen_row = ''
        empty_count = 0
        # Duyệt qua từng ô trong hàng
        for square in row:
            if square == '  ':  # Nếu là ô trống
                empty_count += 1
            else:
                # Nếu có các ô trống liền trước, ghi lại số lượng
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                
                # Chuyển đổi tên quân cờ sang ký hiệu FEN
                color = square[0]  # 'W' hoặc 'B'
                piece_type = square[1:] # 'P', 'KN', 'B', 'R', 'Q', 'K'
                
                # Mã (Knight) có ký hiệu 'N' trong FEN
                if piece_type == 'KN':
                    piece_char = 'N'
                else:
                    piece_char = piece_type[0]
                
                # Chuyển thành chữ hoa (Trắng) hoặc thường (Đen)
                if color == 'W':
                    fen_row += piece_char.upper()
                else:
                    fen_row += piece_char.lower()
        
        # Ghi lại số ô trống còn lại ở cuối hàng
        if empty_count > 0:
            fen_row += str(empty_count)
            
        fen_rows.append(fen_row)
    
    # Nối các hàng lại với nhau bằng dấu '/', và thêm các thông tin FEN mặc định
    # w: Lượt của quân Trắng đi
    # KQkq: Cả hai bên còn đủ quyền nhập thành
    # -: Không có nước bắt Tốt qua đường nào khả dụng
    # 0 1: Số nước đi nửa vời và số nước đi đầy đủ
    return "/".join(fen_rows) + " w KQkq - 0 1"

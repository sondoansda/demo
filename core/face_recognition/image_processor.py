import cv2
import numpy as np
from typing import Optional, Tuple

class ImagePreprocessor:
    def __init__(self):
        # Các tham số cho việc phát hiện khuôn mặt và chất lượng ảnh
        self.min_face_size = (100, 100)  # Kích thước tối thiểu cho khuôn mặt được phát hiện
        self.padding_percent = 0.2  # Phần đệm xung quanh khuôn mặt (20%)
        self.target_size = (224, 224)  # Kích thước chuẩn cho ảnh khuôn mặt
        self.min_brightness = 0.3  # Độ sáng trung bình tối thiểu (0-1)
        self.max_brightness = 0.85  # Độ sáng trung bình tối đa (0-1)
        self.min_contrast = 50  # Độ tương phản tối thiểu (độ lệch chuẩn)
        self.blur_threshold = 100  # Ngưỡng phương sai Laplacian cho việc phát hiện mờ
        self.face_angle_threshold = 30  # Góc nghiêng tối đa cho phép của khuôn mặt (độ)
        
    def preprocess_image(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Áp dụng tất cả các bước tiền xử lý cho một ảnh.
        Trả về None nếu ảnh không đạt tiêu chuẩn chất lượng.
        """
        if image is None or image.size == 0:
            return None
            
        # Chuyển đổi sang RGB nếu cần
        if len(image.shape) == 2:  # Ảnh xám
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # Ảnh RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            
        # Kiểm tra và điều chỉnh độ sáng/độ tương phản
        image = self._adjust_brightness_contrast(image)
        if image is None:
            return None
            
        # Kiểm tra độ mờ
        if not self._check_blur(image):
            return None
            
        # Thay đổi kích thước thành kích thước chuẩn
        image = cv2.resize(image, self.target_size)
        return image
        
    def _check_blur(self, image: np.ndarray) -> bool:
        """Kiểm tra xem ảnh có quá mờ không"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance >= self.blur_threshold
        
    def _adjust_brightness_contrast(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Điều chỉnh độ sáng và độ tương phản của ảnh nếu cần"""
        # Chuyển đổi sang không gian màu LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel = lab[:,:,0]
        
        # Tính toán độ sáng hiện tại
        current_brightness = l_channel.mean() / 255.0
        
        # Kiểm tra xem độ sáng có trong khoảng chấp nhận được không
        if current_brightness < self.min_brightness:
            # Tăng độ sáng
            alpha = 1.2  # Kiểm soát độ tương phản
            beta = 30    # Kiểm soát độ sáng
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        elif current_brightness > self.max_brightness:
            # Giảm độ sáng
            alpha = 0.8
            beta = -30
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
            
        # Kiểm tra độ tương phản
        if np.std(l_channel) < self.min_contrast:
            # Áp dụng cân bằng histogram cho kênh L
            lab[:,:,0] = cv2.equalizeHist(l_channel)
            image = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            
        return image
        
    def crop_face(self, image: np.ndarray, face_location: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """
        Cắt khuôn mặt từ ảnh với đệm và kiểm tra chất lượng
        """
        top, right, bottom, left = face_location
        
        # Kiểm tra kích thước khuôn mặt tối thiểu
        face_width = right - left
        face_height = bottom - top
        if face_width < self.min_face_size[0] or face_height < self.min_face_size[1]:
            return None
            
        # Thêm đệm
        height, width = image.shape[:2]
        padding_x = int(face_width * self.padding_percent)
        padding_y = int(face_height * self.padding_percent)
        
        # Đảm bảo tọa độ nằm trong giới hạn của ảnh
        top = max(0, top - padding_y)
        bottom = min(height, bottom + padding_y)
        left = max(0, left - padding_x)
        right = min(width, right + padding_x)
        
        # Cắt vùng khuôn mặt
        face_image = image[top:bottom, left:right]
        
        # Áp dụng tiền xử lý
        return self.preprocess_image(face_image)

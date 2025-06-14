# Đường dẫn: c:\Users\Dell\Desktop\demo\core\video\video_processor.py
import cv2
from core.face_recognition import FaceRecognition

class VideoProcessor:
    """Bộ xử lý video cho việc quét và nhận dạng khuôn mặt từ camera"""
    
    def __init__(self, db_manager=None):
        """Khởi tạo bộ xử lý video với camera mặc định"""
        self.cap = cv2.VideoCapture(0)
        self.face_recognizer = FaceRecognition(db_manager)

    def read_frame(self):
        """Đọc một khung hình từ camera"""
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        """Giải phóng tài nguyên camera"""
        if self.cap:
            self.cap.release()
import cv2
from core.face_recognition import FaceRecognition

class VideoProcessor:
    def __init__(self, db_manager=None):
        self.cap = cv2.VideoCapture(0)
        self.face_recognizer = FaceRecognition(db_manager)

    def read_frame(self):
        ret, frame = self.cap.read()
        return frame if ret else None

    def release(self):
        if self.cap:
            self.cap.release()
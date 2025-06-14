# Đường dẫn: c:\Users\Dell\Desktop\demo\core\face_recognition\embedding_manager.py
import face_recognition
import numpy as np
import os
from pathlib import Path
import cv2

class EmbeddingManager:
    def __init__(self, cache_path="data/train/embeddings_cache.npz"):
        self.cache_path = cache_path
        self.embeddings = {}
        self._load_cache()

    def _load_cache(self):
        """Tải các embedding đã có từ file cache nếu tồn tại"""
        if os.path.exists(self.cache_path):
            cached_data = np.load(self.cache_path, allow_pickle=True)
            self.embeddings = dict(cached_data.items())
        else:
            self.embeddings = {}

    def _save_cache(self):
        """Lưu các embedding hiện tại vào file cache"""
        np.savez(self.cache_path, **self.embeddings)    
        
    def update_embeddings_for_student(self, student_id):
        """Cập nhật embedding cho một sinh viên cụ thể bằng cách sử dụng ảnh khuôn mặt của họ"""
        student_dir = f"data/train/{student_id}"
        if not os.path.exists(student_dir):
            raise ValueError(f"No directory found for student {student_id}")

        # Get all image files in student directory
        image_files = [f for f in os.listdir(student_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            raise ValueError(f"No images found for student {student_id}")

        # Calculate embeddings for all images
        all_embeddings = []
        for img_file in image_files:
            img_path = os.path.join(student_dir, img_file)
            # Load and convert image
            image = face_recognition.load_image_file(img_path)
            
            # Get face encodings
            face_encodings = face_recognition.face_encodings(image)
            
            if face_encodings:
                # Use the first face found in the image
                all_embeddings.append(face_encodings[0])

        if all_embeddings:
            # Calculate average embedding for the student
            average_embedding = np.mean(all_embeddings, axis=0)
            # Update the embeddings dictionary
            self.embeddings[str(student_id)] = average_embedding
            # Save updated embeddings to cache
            self._save_cache()
            print(f"Updated embeddings for student {student_id}")
        else:
            print(f"No valid face encodings found for student {student_id}")

    def update_all_embeddings(self):
        """Update embeddings for all students in the data/train directory"""
        train_dir = "data/train"
        for student_id in os.listdir(train_dir):
            if os.path.isdir(os.path.join(train_dir, student_id)):
                try:
                    self.update_embeddings_for_student(student_id)
                except Exception as e:
                    print(f"Error updating embeddings for student {student_id}: {str(e)}")

    def get_embedding(self, student_id):
        """Get the embedding for a specific student"""
        return self.embeddings.get(str(student_id))

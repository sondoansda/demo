import cv2
import os
import face_recognition
import numpy as np
from pathlib import Path
from core.face_recognition.embedding_manager import EmbeddingManager
from core.face_recognition.image_processor import ImagePreprocessor

class VideoProcessor:
    def __init__(self):
        self.image_processor = ImagePreprocessor()
        self.min_face_confidence = 0.8  # Minimum confidence for face detection
        self.min_quality_score = 0.6    # Minimum quality score to save face
        self.max_angle = 30             # Maximum allowed face angle in degrees
        self.frame_buffer = []          # Buffer to store recent frames
        self.buffer_size = 5            # Number of frames to buffer
        
    def calculate_face_quality(self, face_image):
        """Calculate a quality score for the face image"""
        # Convert to grayscale for calculations
        gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        
        # Calculate metrics
        brightness = np.mean(gray) / 255.0  # Normalize to 0-1
        contrast = np.std(gray)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Weight the metrics
        quality_score = (
            0.4 * (1 - abs(brightness - 0.5))  # Prefer middle brightness
            + 0.3 * min(contrast / 100, 1.0)   # Higher contrast is better
            + 0.3 * min(sharpness / 500, 1.0)  # Higher sharpness is better
        )
        
        return quality_score
        
    def detect_and_process_face(self, frame):
        """
        Detect and process face from a frame.
        Returns processed face if it meets quality standards, None otherwise.
        """
        if frame is None:
            return None
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using CNN model for better accuracy
        face_locations = face_recognition.face_locations(rgb_frame, model="cnn")
        
        if not face_locations:
            return None
            
        # Process each detected face
        best_face = None
        best_quality = self.min_quality_score
        
        for face_location in face_locations:
            # Crop and preprocess face
            face_image = self.image_processor.crop_face(rgb_frame, face_location)
            if face_image is None:
                continue
                
            # Calculate face quality
            quality_score = self.calculate_face_quality(face_image)
            
            # Update best face if quality is better
            if quality_score > best_quality:
                best_quality = quality_score
                best_face = face_image
                
        return best_face
        
    def process_video(self, video_path, output_dir, skip_frames=1):
        """Process video and save high-quality face images"""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        video = cv2.VideoCapture(video_path)
        frame_count = 0
        saved_count = 0
        
        while True:
            success, frame = video.read()
            if not success:
                break
                
            # Skip frames if specified
            if frame_count % skip_frames != 0:
                frame_count += 1
                continue
                
            # Add frame to buffer
            self.frame_buffer.append(frame)
            if len(self.frame_buffer) > self.buffer_size:
                self.frame_buffer.pop(0)
                
            # Process current frame
            if len(self.frame_buffer) == self.buffer_size:
                # Use middle frame from buffer
                current_frame = self.frame_buffer[self.buffer_size // 2]
                face_image = self.detect_and_process_face(current_frame)
                
                if face_image is not None:
                    # Save face image
                    output_path = os.path.join(output_dir, f'frame_{saved_count}.jpg')
                    cv2.imwrite(output_path, cv2.cvtColor(face_image, cv2.COLOR_RGB2BGR))
                    saved_count += 1
                    
                    if saved_count % 10 == 0:
                        print(f"Saved {saved_count} faces")
                        
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames")
                
        video.release()
        print(f"\nCompleted! Processed {frame_count} frames and saved {saved_count} faces")
        return saved_count > 0

def main():
    processor = VideoProcessor()
    
    while True:
        video_path = input("Enter the path to the video file (or 'q' to quit): ")
        if video_path.lower() == 'q':
            break
            
        student_id = input("Enter student ID: ")
        output_dir = os.path.join("data", "train", str(student_id))
        
        # Process video
        success = processor.process_video(video_path, output_dir, skip_frames=2)
        
        if success:
            # Update embeddings
            embedding_manager = EmbeddingManager()
            embedding_manager.update_embeddings_for_student(student_id)
            print("Successfully updated embeddings cache!")
        else:
            print("No valid faces were detected in the video. Please try again with a different video.")

if __name__ == "__main__":
    main()

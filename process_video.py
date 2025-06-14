import cv2
import os
import face_recognition
from pathlib import Path
from core.face_recognition.embedding_manager import EmbeddingManager
from core.face_recognition.image_processor import ImagePreprocessor

def detect_and_crop_face(frame):
    """
    Detect and crop face from a frame.
    Returns cropped face if found and meets minimum size requirements, None otherwise.
    """
    # Convert BGR (OpenCV format) to RGB (face_recognition format)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face locations
    face_locations = face_recognition.face_locations(rgb_frame)
    
    if not face_locations:
        return None
        
    # Get the first face found
    top, right, bottom, left = face_locations[0]
    
    # Check if face size meets minimum requirements
    face_width = right - left
    face_height = bottom - top
    if face_width < min_face_size[0] or face_height < min_face_size[1]:
        return None
    
    # Add padding around face (20% on each side)
    height, width = frame.shape[:2]
    padding_x = int(face_width * 0.2)
    padding_y = int(face_height * 0.2)
    
    # Ensure padded coordinates are within image bounds
    top = max(0, top - padding_y)
    bottom = min(height, bottom + padding_y)
    left = max(0, left - padding_x)
    right = min(width, right + padding_x)
    
    # Crop and return the face region
    face_image = frame[top:bottom, left:right]
    return face_image

def process_video(video_path, output_dir, skip_frames=1):
    """
    Process a video file to extract faces and save them as individual images.
    
    Args:
        video_path (str): Path to the input video file
        output_dir (str): Directory to save the extracted faces
        skip_frames (int): Number of frames to skip (1 means process every frame)
    """    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Open the video file
    video = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0
    
    while True:
        # Read a frame
        success, frame = video.read()
        if not success:
            break
            
        # Skip frames if specified
        if frame_count % skip_frames != 0:
            frame_count += 1
            continue
            
        # Detect and crop face
        face_image = detect_and_crop_face(frame)
        
        if face_image is not None:
            # Save the face image
            output_path = os.path.join(output_dir, f'frame_{saved_count}.jpg')
            cv2.imwrite(output_path, face_image)
            saved_count += 1
            
            # Print progress every 10 saved faces
            if saved_count % 10 == 0:
                print(f"Saved {saved_count} faces")
            
        frame_count += 1
        
        # Print progress
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames, saved {saved_count} faces")
    
    video.release()
    print(f"\nCompleted! Processed {frame_count} frames and saved {saved_count} faces to {output_dir}")

def main():
    # Example usage
    video_path = input("Enter the path to the video file: ")
    student_id = input("Enter student ID: ")
    output_dir = os.path.join("data", "train", str(student_id))
    
    # Process video and save faces
    process_video(video_path, output_dir, skip_frames=2)  # Process every 2nd frame
    
    # Update embeddings for the new student
    embedding_manager = EmbeddingManager()
    embedding_manager.update_embeddings_for_student(student_id)
    print("Successfully updated embeddings cache!")

if __name__ == "__main__":
    main()

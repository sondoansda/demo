import cv2
import os
import numpy as np
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from datetime import datetime

class FaceRecognition:
    def __init__(self, db_manager):
        """
        Initialize the FaceRecognition class with MTCNN detector and FaceNet model.
        """
        self.db_manager = db_manager
        # Sử dụng thư mục train trong project thay vì .student_checkin
        self.train_data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'train')
        
        print(f"Loading models from: {self.train_data_dir}")
        
        # Initialize MTCNN with optimized parameters for face detection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.detector = MTCNN(
            image_size=160,  # Kích thước chuẩn cho FaceNet
            margin=20,       # Margin để có thêm context cho khuôn mặt
            keep_all=True,   # Luôn giữ tất cả các khuôn mặt phát hiện được
            min_face_size=60,  # Kích thước tối thiểu để lọc nhiễu
            thresholds=[0.6, 0.7, 0.7],  # Ngưỡng phát hiện cho từng bước
            factor=0.709,
            post_process=True,
            device=self.device,
            select_largest=False  # Không chỉ chọn khuôn mặt lớn nhất
        )
        
        # Initialize FaceNet model with VGGFace2 weights
        self.model = InceptionResnetV1(
            pretrained='vggface2',
            classify=False,
            device=self.device
        ).eval()
        
        # Adjust recognition parameters
        self.recognition_threshold = 0.5  # Ngưỡng khoảng cách tối đa (confidence > 50%)
        self.min_detection_confidence = 0.9  # Ngưỡng tin cậy phát hiện khuôn mặt
        self.min_similarity = 0.8  # Độ tương đồng tối thiểu (50%)
        self.max_distance_ratio = 0.5  # Tỉ lệ khoảng cách tối đa cho phép
        self.cap = None
        self.known_embeddings = {}
        self.load_known_embeddings()    
    def load_known_embeddings(self):
        """Load pre-computed embeddings from cache file or compute if not exists"""
        cache_file = os.path.join(self.train_data_dir, 'embeddings_cache.npz')
        
        # Get the root window for loading screen
        root = None
        if hasattr(self.db_manager, '_mainwindow'):
            root = self.db_manager._mainwindow

        loading_screen = None
        if root:
            from gui.loading_screen import LoadingScreen
            loading_screen = LoadingScreen(root)
            loading_screen.update_progress(0, "Đang tải dữ liệu khuôn mặt...")
            
        # Try to load from cache first
        if os.path.exists(cache_file):
            try:
                print("Loading embeddings from cache...")
                cached_data = np.load(cache_file, allow_pickle=True)
                self.known_embeddings = cached_data['embeddings'].item()
                if loading_screen:
                    loading_screen.update_progress(100, "Đã tải xong dữ liệu từ cache")
                    loading_screen.close()
                print(f"Loaded {len(self.known_embeddings)} embeddings from cache")
                return
            except Exception as e:
                print(f"Failed to load cache: {e}, will recompute embeddings")

        print("Đang tải dữ liệu khuôn mặt đã biết...")
        
        if not os.path.exists(self.train_data_dir):
            if loading_screen:
                loading_screen.close()
            print(f"Không tìm thấy thư mục train: {self.train_data_dir}")
            return
              # Chỉ load các thư mục số (ID sinh viên)
        student_dirs = [d for d in os.listdir(self.train_data_dir) 
                       if os.path.isdir(os.path.join(self.train_data_dir, d)) and d.isdigit()]
        
        total_students = len(student_dirs)
        for idx, student_id in enumerate(student_dirs):
            if loading_screen:
                progress = (idx / total_students) * 100
                loading_screen.update_progress(progress, f"Đang xử lý sinh viên {student_id}...")
                
            student_path = os.path.join(self.train_data_dir, student_id)
            embeddings = []
            face_images = [f for f in os.listdir(student_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            if not face_images:
                print(f"Không tìm thấy ảnh cho sinh viên {student_id}")
                continue
                
            print(f"Đang xử lý {len(face_images)} ảnh cho sinh viên {student_id}")
            
            for img_name in face_images:
                try:
                    img_path = os.path.join(student_path, img_name)
                    img = Image.open(img_path).convert('RGB')
                    
                    # Detect and align face
                    faces = self.detector(img)
                    if faces is not None and len(faces) > 0:
                        face_tensor = faces[0].unsqueeze(0).to(self.device)
                        
                        # Get embedding
                        with torch.no_grad():
                            embedding = self.model(face_tensor).cpu().numpy()
                        embeddings.append(embedding.reshape(-1))  # Flatten to 1D array
                        
                except Exception as e:
                    print(f"Lỗi xử lý ảnh {img_name}: {str(e)}")
                    continue            
            if embeddings:
                # Calculate mean embedding for the student
                mean_embedding = np.mean(embeddings, axis=0)
                # Normalize the embedding
                mean_embedding = mean_embedding / np.linalg.norm(mean_embedding)
                self.known_embeddings[student_id] = mean_embedding
                print(f"Đã tải thành công dữ liệu cho sinh viên {student_id}")
            else:
                print(f"Không thể trích xuất đặc trưng cho sinh viên {student_id}")
        
        if loading_screen:
            loading_screen.update_progress(100, f"Đã tải xong dữ liệu cho {len(self.known_embeddings)} sinh viên")
            loading_screen.close()
            
        # Save embeddings to cache
        cache_file = os.path.join(self.train_data_dir, 'embeddings_cache.npz')
        try:
            np.savez(cache_file, embeddings=self.known_embeddings)
            print("Đã lưu embedding vectors vào cache")
        except Exception as e:
            print(f"Lỗi khi lưu cache: {e}")
            
        print(f"Đã tải xong dữ liệu cho {len(self.known_embeddings)} sinh viên")

    def recognize_face(self, frame):
        """
        Perform face recognition on the given frame.
        Returns: List of (student_id, confidence, box) tuples for detected faces
        """
        try:
            # Tiền xử lý frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Phát hiện khuôn mặt với độ tin cậy
            boxes, probs = self.detector.detect(frame_pil)
            
            if boxes is None or len(boxes) == 0:
                print("Không tìm thấy khuôn mặt nào trong khung hình")
                return []

            print(f"Đã tìm thấy {len(boxes)} khuôn mặt trong khung hình")
            print(f"Độ tin cậy phát hiện từng khuôn mặt: {probs}")

            results = []
            # Detect và align faces
            faces = self.detector(frame_pil)
            if faces is None:
                print("Không thể trích xuất đặc trưng khuôn mặt")
                return []
            
            # Xử lý từng khuôn mặt phát hiện được
            for i, (box, prob, face) in enumerate(zip(boxes, probs, faces)):
                print(f"\nXử lý khuôn mặt thứ {i+1}:")
                
                if prob < self.min_detection_confidence:
                    print(f"- Bỏ qua do độ tin cậy thấp: {prob:.2f}")
                    continue
                    
                # Trích xuất đặc trưng
                face_tensor = face.unsqueeze(0).to(self.device)
                with torch.no_grad():
                    embedding = self.model(face_tensor).cpu().numpy()
                
                # Reshape và chuẩn hóa embedding
                embedding = embedding.reshape(-1)
                embedding = embedding / np.linalg.norm(embedding)
                
                if not self.known_embeddings:
                    print("- Chưa có dữ liệu khuôn mặt nào được lưu trong hệ thống")
                    continue                # So sánh với các khuôn mặt đã biết
                matches = []
                distances = []                # Tính toán khoảng cách với tất cả embedding đã biết
                for student_id, known_embedding in self.known_embeddings.items():
                    known_embedding = known_embedding.reshape(-1)
                    # Tính khoảng cách cosine
                    distance = 1 - np.dot(embedding, known_embedding)
                    print(f"- So sánh với ID {student_id}: {distance:.4f} (ngưỡng: {self.recognition_threshold})")
                    if distance < self.recognition_threshold:
                        matches.append((student_id, distance, box))
                        distances.append(distance)
                
                # Nếu có nhiều hơn 1 match, kiểm tra tỉ lệ khoảng cách
                best_match = None
                if matches:
                    best_idx = np.argmin(distances)
                    best_distance = distances[best_idx]                    # Kiểm tra xem có match nào quá gần với best_distance không
                    similar_matches = [
                        idx for idx, d in enumerate(distances)
                        if d <= best_distance * (1 + 0.2)  # Cho phép sai lệch 20%
                    ]
                    
                    # Chỉ chấp nhận nếu có duy nhất một match tốt nhất
                    if len(similar_matches) == 1:
                        student_id, distance, box = matches[best_idx]
                        confidence = 1 - distance
                        similarity = confidence * 100
                        best_match = (student_id, confidence, box)
                        print(f"- Tìm thấy match duy nhất: ID {student_id} (độ tương đồng: {similarity:.1f}%)")
                    else:
                        print(f"- Có {len(similar_matches)} khuôn mặt có độ tương đồng > {self.max_distance_ratio*100}%, bỏ qua")
                
                if best_match:
                    results.append(best_match)
                    print(f"- Đã nhận diện được ID: {best_match[0]} (độ tin cậy: {best_match[1]:.2%})")
                else:
                    print(f"- Không tìm thấy khuôn mặt phù hợp")
            
            print(f"\nTổng số khuôn mặt đã nhận diện được: {len(results)}")
            return results

        except Exception as e:
            print(f"Lỗi trong quá trình nhận diện: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

    def face_reg_on_camera(self, stop_event=None):
        """
        Run face recognition on camera feed and handle check-ins.
        stop_event: threading.Event to signal when to stop the camera feed
        """
        try:
            if self.cap is None:
                self.cap = cv2.VideoCapture(0)
                if not self.cap.isOpened():
                    print("Failed to open camera")
                    return
                    
            while True:
                if stop_event and stop_event.is_set():
                    break
                    
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to read frame")
                    break
                
                # Hiển thị số khuôn mặt đã phát hiện được
                cv2.putText(frame, f"Searching for faces...", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Perform recognition
                matches = self.recognize_face(frame)
                  # Handle successful recognitions and draw boxes
                for student_id, confidence, box in matches:
                    student_name = self.db_manager.get_student_name(student_id)
                    if student_name:
                        # Chỉ ghi nhận và hiển thị khi độ tin cậy > 50%
                        if confidence > 0.5:
                            # Update check-in in database
                            success = self.db_manager.add_recent_checkin(student_id, student_name)
                            
                            # Draw bounding box với màu khác nhau dựa trên kết quả
                            x1, y1, x2, y2 = map(int, box)
                            color = (0, 255, 0) if success else (0, 165, 255)  # Xanh lá nếu thành công, cam nếu thất bại
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Add text above box với thông tin chi tiết hơn
                            text = f"{student_name} (ID: {student_id})"
                            cv2.putText(frame, text, (x1, y1-25), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
                            # Add confidence và trạng thái dưới box
                            status = "Đã điểm danh" if success else "Đã điểm danh trước đó"
                            conf_text = f"{status} - {confidence:.1%}"
                            cv2.putText(frame, conf_text, (x1, y2+25), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        else:
                            # Vẽ box màu đỏ cho trường hợp độ tin cậy thấp
                            x1, y1, x2, y2 = map(int, box)
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                            cv2.putText(frame, f"Không đủ tin cậy ({confidence:.1%})", 
                                      (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display total faces found
                found_text = f"Found {len(matches)} face(s)"
                cv2.putText(frame, found_text, (10, frame.shape[0] - 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display ESC instruction
                cv2.putText(frame, "Press ESC to exit", (frame.shape[1] - 200, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Show frame
                cv2.imshow('Face Recognition', frame)
                
                # Check for ESC key
                if cv2.waitKey(1) & 0xFF == 27:
                    print("Camera stopped by user (ESC)")
                    break

        except Exception as e:
            print(f"Error in face recognition: {str(e)}")
            import traceback
            traceback.print_exc()
            
        finally:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            cv2.destroyAllWindows()    
    def add_training_face(self, student_id, frame):
        """
        Process and save a new training face image for a student.
        1. Detect faces in the frame
        2. Align and crop faces
        3. Save only the face regions as training data
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb)
        
        # Detect faces and get bounding boxes
        boxes, probs = self.detector.detect(frame_pil)
        if boxes is None or len(boxes) == 0:
            print("Không phát hiện được khuôn mặt nào")
            return False
            
        # Get aligned faces
        faces = self.detector(frame_pil)
        if faces is None:
            print("Không thể căn chỉnh khuôn mặt")
            return False
            
        student_dir = os.path.join(self.train_data_dir, student_id)
        os.makedirs(student_dir, exist_ok=True)
        
        saved_faces = 0
        for i, (box, prob, face) in enumerate(zip(boxes, probs, faces)):
            # Chỉ lưu khuôn mặt có độ tin cậy cao
            if prob < self.min_detection_confidence:
                print(f"Bỏ qua khuôn mặt {i+1} do độ tin cậy thấp: {prob:.2f}")
                continue
                
            try:
                # Chuẩn hóa khuôn mặt sang tensor
                face_tensor = face.unsqueeze(0)
                
                # Lưu ảnh khuôn mặt đã được cắt và căn chỉnh
                save_path = os.path.join(student_dir, f'face_{datetime.now().strftime("%Y%m%d_%H%M%S")}_{i}.jpg')
                
                # Chuyển tensor thành ảnh và lưu                face_img = face.permute(1, 2, 0).cpu().numpy()
                face_img = ((face_img * 127.5 + 127.5)).astype(np.uint8)
                cv2.imwrite(save_path, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                
                saved_faces += 1
                print(f"Đã lưu khuôn mặt {i+1} với độ tin cậy: {prob:.2f}")
            except Exception as e:
                print(f"Lỗi khi lưu khuôn mặt {i+1}: {str(e)}")
                continue
        
        if saved_faces > 0:
            # Cập nhật embeddings sau khi thêm ảnh mới
            self.load_known_embeddings()
            print(f"Đã lưu thành công {saved_faces} khuôn mặt")
            return True
        else:
            print("Không lưu được khuôn mặt nào")
            return False
            
            # Update known embeddings for this student
            self.load_known_embeddings()
            return True
        return False

# Module-level function for face recognition camera access
def face_reg_on_camera(db_manager=None):
    """
    Start face recognition on the camera feed.
    If db_manager is not provided, a new one will be created.
    """
    from data.database.db_manager import DatabaseManager
    if db_manager is None:
        db_manager = DatabaseManager()
    
    face_recognition = FaceRecognition(db_manager)
    return face_recognition.face_reg_on_camera()
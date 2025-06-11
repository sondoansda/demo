import tkinter as tk
from tkinter import ttk
import tkinter.filedialog as filedialog
from tkinter import messagebox
from data.database.db_manager import DatabaseManager
import os
import time
import cv2
import threading
from queue import Queue

class StudentForm(ttk.Frame):
    def __init__(self, parent, refresh_callback):
        super().__init__(parent)
        self.db_manager = DatabaseManager()
        self.db_manager._mainwindow = self.winfo_toplevel()
        self.refresh_callback = refresh_callback
        
        # Camera-related variables
        self.camera_thread = None
        self.stop_camera = threading.Event()
        self.frame_queue = Queue(maxsize=2)
        self.recording_done = threading.Event()
        self.recording_successful = False
        self.video_path = None

        self.student_id_label = ttk.Label(self, text="Student ID:")
        self.student_id_label.grid(row=0, column=0, padx=5, pady=5)
        self.student_id_entry = ttk.Entry(self)
        self.student_id_entry.grid(row=0, column=1, padx=5, pady=5)

        self.name_label = ttk.Label(self, text="Name:")
        self.name_label.grid(row=1, column=0, padx=5, pady=5)
        self.name_entry = ttk.Entry(self)
        self.name_entry.grid(row=1, column=1, padx=5, pady=5)

        self.video_path_label = ttk.Label(self, text="Video Path:")
        self.video_path_label.grid(row=2, column=0, padx=5, pady=5)
        self.video_path_entry = ttk.Entry(self)
        self.video_path_entry.grid(row=2, column=1, padx=5, pady=5)

        # Add buttons for additional functionality
        self.browse_button = ttk.Button(self, text="Browse Video", command=self.browse_video)
        self.browse_button.grid(row=3, column=0, padx=5, pady=5)

        self.add_student_button = ttk.Button(self, text="Add Student", command=self.add_student)
        self.add_student_button.grid(row=3, column=1, padx=5, pady=5)

        self.use_camera_button = ttk.Button(self, text="Use Camera", command=self.use_camera)
        self.use_camera_button.grid(row=4, column=0, columnspan=2, padx=5, pady=5)

    def get_values(self):
        return {
            "student_id": self.student_id_entry.get(),
            "name": self.name_entry.get(),
            "video_path": self.video_path_entry.get()
        }

    def clear(self):
        self.student_id_entry.delete(0, tk.END)
        self.name_entry.delete(0, tk.END)
        self.video_path_entry.delete(0, tk.END)

    def browse_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
        if video_path:
            self.video_path_entry.delete(0, tk.END)
            self.video_path_entry.insert(0, video_path)

    def add_student(self):
        values = self.get_values()
        if not values['student_id'] or not values['name'] or not values['video_path']:
            messagebox.showerror("Error", "Please fill all fields and provide a video")
            return

        try:
            self.db_manager.add_student(values['student_id'], values['name'], values['video_path'])
            messagebox.showinfo("Success", "Student added successfully")
            self.clear()
            self.refresh_callback()  # Refresh the student table
        except Exception as e:
            messagebox.showerror("Error", f"Failed to add student: {str(e)}")

    def process_camera_feed(self, video_path, width, height):
        try:
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
            if not cap.isOpened():
                return False

            out = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc(*'mp4v'),
                30,
                (width, height)
            )

            countdown_start = 3
            face_detected = False
            recording_time = 5
            start_time = None
            face_center_zone = None
            last_time = time.time()
            countdown_last_update = last_time

            while not self.stop_camera.is_set():
                current_time = time.time()
                if current_time - last_time < 1.0/30:
                    time.sleep(0.001)  # Short sleep instead of waitKey
                    continue
                last_time = current_time

                ret, frame = cap.read()
                if not ret:
                    break

                display_frame = frame.copy()
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)

                if len(faces) > 0:
                    (x, y, w, h) = faces[0]
                    face_center = (x + w//2, y + h//2)
                    
                    if face_center_zone is None:
                        zone_size = min(width, height) // 4
                        face_center_zone = (
                            width//2 - zone_size,
                            height//2 - zone_size,
                            width//2 + zone_size,
                            height//2 + zone_size
                        )
                    
                    cv2.rectangle(display_frame, 
                                (face_center_zone[0], face_center_zone[1]),
                                (face_center_zone[2], face_center_zone[3]),
                                (0, 255, 0), 2)
                    
                    if (face_center_zone[0] < face_center[0] < face_center_zone[2] and
                        face_center_zone[1] < face_center[1] < face_center_zone[3]):
                        face_detected = True
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.circle(display_frame, face_center, 5, (0, 255, 0), -1)
                    else:
                        face_detected = False
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 165, 255), 2)
                        cv2.circle(display_frame, face_center, 5, (0, 165, 255), -1)
                        cv2.putText(display_frame, "Center your face", 
                                  (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 165, 255), 2)
                else:
                    face_detected = False
                    cv2.putText(display_frame, "No face detected", 
                              (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                              0.7, (0, 0, 255), 2)

                if face_detected and start_time is None:
                    if countdown_start > 0:
                        if current_time - countdown_last_update >= 1.0:
                            countdown_start -= 1
                            countdown_last_update = current_time
                        cv2.putText(display_frame, f"Starting in {countdown_start+1}", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.7, (255, 255, 255), 2)
                    else:
                        start_time = current_time
                        cv2.putText(display_frame, "Recording...", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 255, 0), 2)
                elif start_time is not None:
                    elapsed = current_time - start_time
                    remaining = max(0, recording_time - elapsed)
                    if remaining > 0:
                        cv2.putText(display_frame, f"Recording... {remaining:.1f}s", 
                                  (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                  0.7, (0, 255, 0), 2)
                        out.write(frame)
                    else:
                        self.recording_successful = True
                        break

                # Put frame in queue for display
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except:
                        pass
                self.frame_queue.put((display_frame, face_detected, start_time is not None))

            cap.release()
            out.release()
            self.recording_done.set()

        except Exception as e:
            print(f"Camera thread error: {str(e)}")
            self.recording_done.set()
        finally:
            if cap is not None:
                cap.release()
            if out is not None:
                out.release()

    def update_display(self):
        if not self.stop_camera.is_set():
            try:
                if not self.frame_queue.empty():
                    frame, _, _ = self.frame_queue.get_nowait()
                    cv2.imshow("Registration", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:  # ESC key
                        self.stop_camera.set()
                
                if not self.recording_done.is_set():
                    self.after(10, self.update_display)
                else:
                    cv2.destroyAllWindows()
                    if self.recording_successful:
                        self.video_path_entry.delete(0, tk.END)
                        self.video_path_entry.insert(0, self.video_path)
                        messagebox.showinfo("Success", "Video recorded successfully!")
                    else:
                        if os.path.exists(self.video_path):
                            os.remove(self.video_path)
                        messagebox.showwarning("Warning", "Recording was cancelled or incomplete")
            except Exception as e:
                print(f"Display update error: {str(e)}")
                self.stop_camera.set()

    def use_camera(self):
        values = self.get_values()
        if not values['student_id'] or not values['name']:
            messagebox.showerror("Error", "Please enter Student ID and Name first")
            return

        try:
            # Create directory for training data
            train_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'train', values['student_id'])
            os.makedirs(train_dir, exist_ok=True)

            # Generate video file path
            self.video_path = os.path.join(train_dir, f'registration_{values["student_id"]}.mp4')
            
            # Reset control flags
            self.stop_camera.clear()
            self.recording_done.clear()
            self.recording_successful = False
            
            # Clear the frame queue
            while not self.frame_queue.empty():
                self.frame_queue.get_nowait()

            # Start camera thread
            self.camera_thread = threading.Thread(
                target=self.process_camera_feed,
                args=(self.video_path, 640, 480)
            )
            self.camera_thread.daemon = True
            self.camera_thread.start()

            # Start display update
            self.update_display()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
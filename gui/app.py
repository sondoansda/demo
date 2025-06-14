# Đường dẫn: c:\Users\Dell\Desktop\demo\gui\app.py
import tkinter as tk
from tkinter import ttk, messagebox
from data.database.db_manager import DatabaseManager
from gui.components.student_form import StudentForm
from gui.components.student_table import StudentTable
from gui.components.checkin_table import CheckinTable
import threading
from datetime import datetime
import cv2
from core.face_recognition import FaceRecognition

class StudentCheckInApp:
    def __init__(self, root):
        """
        Khởi tạo cửa sổ ứng dụng chính và các thành phần của nó.
        """
        self.root = root
        self.root.title("Hệ thống Điểm danh Sinh viên")

        # Khởi tạo các thành phần
        self.db_manager = DatabaseManager()
        self.face_recognition = FaceRecognition(self.db_manager)
        
        self._init_ui()
        self.refresh_tables()  # Cập nhật bảng khi khởi động
        
        # Gắn sự kiện đóng cửa sổ
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _init_ui(self):
        """
        Thiết lập giao diện người dùng, bao gồm bảng đăng ký sinh viên và bảng điểm danh.
        """
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Panel trái - Đăng ký sinh viên
        left_panel = ttk.LabelFrame(main_frame, text="Đăng ký Sinh viên", padding="5")
        left_panel.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.student_form = StudentForm(left_panel, self.refresh_tables)
        self.student_form.grid(row=0, column=0, padx=5, pady=5)

        # Panel phải - Các bảng
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Bảng sinh viên đã đăng ký
        students_frame = ttk.LabelFrame(right_panel, text="Sinh viên đã Đăng ký", padding="5")
        students_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.student_table = StudentTable(students_frame)
        self.student_table.pack(fill=tk.BOTH, expand=True)

        # Bảng điểm danh gần đây
        checkins_frame = ttk.LabelFrame(right_panel, text="Điểm danh Gần đây", padding="5")
        checkins_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.checkin_table = CheckinTable(checkins_frame, self.db_manager)
        self.checkin_table.pack(fill=tk.BOTH, expand=True)

        # Khung nút bấm
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=1, column=0, padx=5, pady=5)
        
        # Nút Điểm danh
        self.checkin_button = ttk.Button(button_frame, text="Điểm danh", command=self.start_checkin)
        self.checkin_button.pack(side='left', padx=5)

    def start_checkin(self):
        """
        Bắt đầu quá trình điểm danh bằng nhận diện khuôn mặt
        """
        self.checkin_button.configure(state='disabled')  # Vô hiệu hóa nút trong khi camera đang hoạt động
        
        try:
            self.face_recognition.face_reg_on_camera()
            # Cập nhật bảng check-in ngay sau khi camera đóng
            self.checkin_table.update_checkins()
        except Exception as e:
            messagebox.showerror("Lỗi", f"Điểm danh thất bại: {str(e)}")
        finally:
            self.checkin_button.configure(state='normal')  # Kích hoạt lại nút
            
    def refresh_tables(self):
        """
        Làm mới các bảng sinh viên và điểm danh với dữ liệu mới nhất từ cơ sở dữ liệu.
        """
        self.student_table.update_students(self.db_manager.get_students())
        self.checkin_table.update_checkins()  # Cập nhật bảng điểm danh
        # Lên lịch làm mới tiếp theo
        self.root.after(5000, self.refresh_tables)  # Làm mới mỗi 5 giây

    def on_closing(self):
        """Xử lý việc đóng ứng dụng"""
        try:
            cv2.destroyAllWindows()  # Đảm bảo đóng mọi cửa sổ camera đang mở
        finally:
            self.root.destroy()  # Đóng cửa sổ
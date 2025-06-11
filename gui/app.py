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
        Initialize the main application window and its components.
        """
        self.root = root
        self.root.title("Student Check-in System")

        # Initialize components
        self.db_manager = DatabaseManager()
        self.face_recognition = FaceRecognition(self.db_manager)
        
        self._init_ui()
        self.refresh_tables()  # Populate tables on startup
        
        # Bind window close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def _init_ui(self):
        """
        Set up the user interface, including student registration and check-in tables.
        """
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Left panel - Student registration
        left_panel = ttk.LabelFrame(main_frame, text="Student Registration", padding="5")
        left_panel.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.student_form = StudentForm(left_panel, self.refresh_tables)
        self.student_form.grid(row=0, column=0, padx=5, pady=5)

        # Right panel - Tables
        right_panel = ttk.Frame(main_frame)
        right_panel.grid(row=0, column=1, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        # Registered students table
        students_frame = ttk.LabelFrame(right_panel, text="Registered Students", padding="5")
        students_frame.grid(row=0, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.student_table = StudentTable(students_frame)
        self.student_table.pack(fill=tk.BOTH, expand=True)

        # Recent check-ins table
        checkins_frame = ttk.LabelFrame(right_panel, text="Recent Check-ins", padding="5")
        checkins_frame.grid(row=1, column=0, padx=5, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.checkin_table = CheckinTable(checkins_frame, self.db_manager)
        self.checkin_table.pack(fill=tk.BOTH, expand=True)

        # Add button frame
        button_frame = ttk.Frame(self.root)
        button_frame.grid(row=1, column=0, padx=5, pady=5)
        
        # Single Check-in button
        self.checkin_button = ttk.Button(button_frame, text="Check-in", command=self.start_checkin)
        self.checkin_button.pack(side='left', padx=5)

    def start_checkin(self):
        """
        Start the face recognition check-in process
        """
        self.checkin_button.configure(state='disabled')  # Disable button while camera is running
        
        try:
            self.face_recognition.face_reg_on_camera()
            # Cập nhật bảng check-in ngay sau khi camera đóng
            self.checkin_table.update_checkins()
        except Exception as e:
            messagebox.showerror("Error", f"Check-in failed: {str(e)}")
        finally:
            self.checkin_button.configure(state='normal')  # Re-enable button
            
    def refresh_tables(self):
        """
        Refresh the student and check-in tables with the latest data from the database.
        """
        self.student_table.update_students(self.db_manager.get_students())
        self.checkin_table.update_checkins()  # Update the check-in table
        # Schedule next refresh
        self.root.after(5000, self.refresh_tables)  # Refresh every 5 seconds

    def on_closing(self):
        """Handle application closing"""
        try:
            cv2.destroyAllWindows()  # Make sure to close any open camera windows
        finally:
            self.root.destroy()  # Close the window
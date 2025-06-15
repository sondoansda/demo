import sqlite3
import threading
import os
import cv2
import shutil
from datetime import datetime
from gui.loading_screen import LoadingScreen

class DatabaseManager:
    """Quản lý cơ sở dữ liệu SQLite cho hệ thống điểm danh"""
    
    def __init__(self, db_path="student_checkin.db"):
        """Khởi tạo kết nối đến cơ sở dữ liệu"""
        self.db_path = db_path
        self.local = threading.local()
        self._create_tables()
        self.loading_screen = None

    def _get_connection(self):
        """Lấy kết nối đến cơ sở dữ liệu theo thread"""
        if not hasattr(self.local, "connection"):
            self.local.connection = sqlite3.connect(self.db_path)
        return self.local.connection

    def _create_tables(self):
        """Tạo các bảng cần thiết trong cơ sở dữ liệu"""
        connection = self._get_connection()
        cursor = connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS students (
            student_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            video_path TEXT NOT NULL
        )''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS checkins (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            FOREIGN KEY(student_id) REFERENCES students(student_id)
        )''')
        connection.commit()

    def add_recent_checkin(self, student_id, name):
        """Thêm hoặc cập nhật bản ghi điểm danh cho hôm nay"""
        connection = None
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            today = now.strftime("%Y-%m-%d")
            
            # Kiểm tra xem sinh viên đã điểm danh hôm nay chưa
            cursor.execute("""
                SELECT id FROM checkins 
                WHERE student_id = ? AND date(timestamp) = ?
            """, (student_id, today))
            existing_checkin = cursor.fetchone()
            
            if existing_checkin:
                # Cập nhật thời gian điểm danh cho bản ghi đã tồn tại
                cursor.execute("""
                    UPDATE checkins 
                    SET timestamp = ?
                    WHERE id = ?
                """, (timestamp, existing_checkin[0]))
            else:
                # Tạo bản ghi điểm danh mới
                cursor.execute("""
                    INSERT INTO checkins (student_id, timestamp)
                    VALUES (?, ?)
                """, (student_id, timestamp))
                
            connection.commit()
            print(f"Check-in {'updated' if existing_checkin else 'saved'} for Student ID: {student_id}, Name: {name}")
            return True
        except sqlite3.Error as e:
            print(f"Error saving check-in: {e}")
            if connection:
                connection.rollback()
            return False

    def add_student(self, student_id, name, video_path):
        # Tạo và hiển thị màn hình tải
        if hasattr(self, '_mainwindow'):
            self.loading_screen = LoadingScreen(self._mainwindow)
            self.loading_screen.update_progress(0, "Đang lưu thông tin sinh viên...")
        
        try:
            # Lưu thông tin sinh viên vào cơ sở dữ liệu
            connection = self._get_connection()
            cursor = connection.cursor()
            cursor.execute('INSERT INTO students (student_id, name, video_path) VALUES (?, ?, ?)',
                        (student_id, name, video_path))
            connection.commit()

            if self.loading_screen:
                self.loading_screen.update_progress(20, "Đang xử lý khung hình video...")

            # Tạo thư mục cho dữ liệu đào tạo
            training_data_dir = os.path.join('data', 'train', student_id)
            os.makedirs(training_data_dir, exist_ok=True)

            # Đếm tổng số khung hình để tính toán tiến độ
            video_capture = cv2.VideoCapture(video_path)
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_capture.release()

            # Trích xuất khung hình từ video và lưu vào thư mục đào tạo
            video_capture = cv2.VideoCapture(video_path)
            frame_count = 0
            while True:
                ret, frame = video_capture.read()
                if not ret:
                    break
                frame_path = os.path.join(training_data_dir, f'frame_{frame_count}.jpg')
                cv2.imwrite(frame_path, frame)
                frame_count += 1
                
                if self.loading_screen and total_frames > 0:
                    progress = 20 + (60 * frame_count / total_frames)  # Tiến độ từ 20% đến 80%
                    self.loading_screen.update_progress(progress, f"Đang xử lý khung hình {frame_count}/{total_frames}")
            
            video_capture.release()

            if self.loading_screen:
                self.loading_screen.update_progress(80, "Đang đào tạo mô hình nhận diện khuôn mặt...")

            # Kích hoạt quá trình đào tạo
            self._train_model(student_id)

            if self.loading_screen:
                self.loading_screen.update_progress(100, "Hoàn thành!")
                self.loading_screen.close()

        except Exception as e:
            if self.loading_screen:
                self.loading_screen.close()
            raise e

    def _train_model(self, student_id):
        print(f"Đang đào tạo mô hình cho sinh viên {student_id}...")
        # Thêm logic đào tạo thực tế ở đây nếu cần
        import time
        time.sleep(2)  # Giả lập thời gian đào tạo

    def get_students(self):
        connection = self._get_connection()
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM students')
        return cursor.fetchall()

    def get_recent_checkins(self, limit=10):
        """Lấy danh sách điểm danh gần đây với tên sinh viên"""
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            cursor.execute('''
                SELECT c.student_id, s.name, c.timestamp 
                FROM checkins c 
                JOIN students s ON c.student_id = s.student_id 
                ORDER BY c.timestamp DESC 
                LIMIT ?
            ''', (limit,))
            return cursor.fetchall()
        except sqlite3.Error as e:
            print(f"Error getting recent check-ins: {e}")
            return []

    def get_student_name(self, student_id):
        """Lấy tên sinh viên từ cơ sở dữ liệu theo ID sinh viên"""
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            cursor.execute('SELECT name FROM students WHERE student_id = ?', (student_id,))
            result = cursor.fetchone()
            return result[0] if result else "Unknown"
        except sqlite3.Error as e:
            print(f"Error getting student name: {e}")
            return "Unknown"

    def delete_student(self, student_id):
        # Xóa thông tin sinh viên khỏi cơ sở dữ liệu
        connection = self._get_connection()
        cursor = connection.cursor()
        cursor.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
        connection.commit()

        # Xóa thư mục dữ liệu đào tạo của sinh viên
        training_data_dir = os.path.join('data', 'train', student_id)
        if os.path.exists(training_data_dir):
            shutil.rmtree(training_data_dir)
            print(f"Đã xóa dữ liệu đào tạo của sinh viên {student_id}.")
        else:
            print(f"Không tìm thấy dữ liệu đào tạo của sinh viên {student_id}.")

    def delete_checkin_history(self, days=None):
        """Xóa lịch sử điểm danh. Nếu có tham số days, chỉ xóa các bản ghi cũ hơn số ngày đó"""
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            
            if days is not None:
                # Xóa các bản ghi cũ hơn số ngày đã chỉ định
                cursor.execute("""
                    DELETE FROM checkins 
                    WHERE datetime(timestamp) < datetime('now', '-' || ? || ' days')
                """, (days,))
            else:
                # Xóa tất cả các bản ghi
                cursor.execute("DELETE FROM checkins")
                
            deleted_count = cursor.rowcount
            connection.commit()
            print(f"Đã xóa {deleted_count} bản ghi điểm danh")
            return deleted_count
        except sqlite3.Error as e:
            print(f"Error deleting check-in history: {e}")
            return 0

    def get_all_attendance(self):
        """Lấy tất cả các bản ghi điểm danh kèm thông tin sinh viên"""
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            
            query = """
                SELECT s.student_id, s.name, c.timestamp 
                FROM students s
                JOIN checkins c ON s.student_id = c.student_id
                ORDER BY c.timestamp DESC
            """
            
            cursor.execute(query)
            return cursor.fetchall()
            
        except sqlite3.Error as e:
            print(f"Error getting attendance data: {e}")
            return []
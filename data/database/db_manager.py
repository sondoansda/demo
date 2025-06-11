import sqlite3
import threading
import os
import cv2
import shutil
from datetime import datetime
from gui.loading_screen import LoadingScreen

class DatabaseManager:
    def __init__(self, db_path="student_checkin.db"):
        self.db_path = db_path
        self.local = threading.local()
        self._create_tables()
        self.loading_screen = None

    def _get_connection(self):
        if not hasattr(self.local, "connection"):
            self.local.connection = sqlite3.connect(self.db_path)
        return self.local.connection

    def _create_tables(self):
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
        """Add or update check-in record for today"""
        connection = None
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            now = datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            today = now.strftime("%Y-%m-%d")
            
            # Check if student already has a check-in today
            cursor.execute("""
                SELECT id FROM checkins 
                WHERE student_id = ? AND date(timestamp) = ?
            """, (student_id, today))
            existing_checkin = cursor.fetchone()
            
            if existing_checkin:
                # Update existing check-in time
                cursor.execute("""
                    UPDATE checkins 
                    SET timestamp = ?
                    WHERE id = ?
                """, (timestamp, existing_checkin[0]))
            else:
                # Create new check-in record
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
        # Create and show loading screen
        if hasattr(self, '_mainwindow'):
            self.loading_screen = LoadingScreen(self._mainwindow)
            self.loading_screen.update_progress(0, "Saving student information...")
        
        try:
            # Save student information to the database
            connection = self._get_connection()
            cursor = connection.cursor()
            cursor.execute('INSERT INTO students (student_id, name, video_path) VALUES (?, ?, ?)',
                        (student_id, name, video_path))
            connection.commit()

            if self.loading_screen:
                self.loading_screen.update_progress(20, "Processing video frames...")

            # Create directory for training data
            training_data_dir = os.path.join('data', 'train', student_id)
            os.makedirs(training_data_dir, exist_ok=True)

            # Count total frames for progress calculation
            video_capture = cv2.VideoCapture(video_path)
            total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            video_capture.release()

            # Extract frames from the video and save them to the training directory
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
                    progress = 20 + (60 * frame_count / total_frames)  # Progress from 20% to 80%
                    self.loading_screen.update_progress(progress, f"Processing frame {frame_count}/{total_frames}")
            
            video_capture.release()

            if self.loading_screen:
                self.loading_screen.update_progress(80, "Training face recognition model...")

            # Trigger training process
            self._train_model(student_id)

            if self.loading_screen:
                self.loading_screen.update_progress(100, "Completed!")
                self.loading_screen.close()

        except Exception as e:
            if self.loading_screen:
                self.loading_screen.close()
            raise e

    def _train_model(self, student_id):
        print(f"Training model for student {student_id}...")
        # Add actual training logic here if needed
        import time
        time.sleep(2)  # Simulate training time

    def get_students(self):
        connection = self._get_connection()
        cursor = connection.cursor()
        cursor.execute('SELECT * FROM students')
        return cursor.fetchall()

    def get_recent_checkins(self, limit=10):
        """Get recent check-ins with student names"""
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
        """Get student name from database by student ID"""
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
        # Delete student information from the database
        connection = self._get_connection()
        cursor = connection.cursor()
        cursor.execute('DELETE FROM students WHERE student_id = ?', (student_id,))
        connection.commit()

        # Remove the student's training data directory
        training_data_dir = os.path.join('data', 'train', student_id)
        if os.path.exists(training_data_dir):
            shutil.rmtree(training_data_dir)
            print(f"Deleted training data for student {student_id}.")
        else:
            print(f"No training data found for student {student_id}.")

    def delete_checkin_history(self, days=None):
        """Delete check-in history. If days is provided, only delete records older than that many days"""
        try:
            connection = self._get_connection()
            cursor = connection.cursor()
            
            if days is not None:
                # Delete records older than specified days
                cursor.execute("""
                    DELETE FROM checkins 
                    WHERE datetime(timestamp) < datetime('now', '-' || ? || ' days')
                """, (days,))
            else:
                # Delete all records
                cursor.execute("DELETE FROM checkins")
                
            deleted_count = cursor.rowcount
            connection.commit()
            print(f"Deleted {deleted_count} check-in records")
            return deleted_count
        except sqlite3.Error as e:
            print(f"Error deleting check-in history: {e}")
            return 0
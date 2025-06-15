# Đường dẫn: c:\Users\Dell\Desktop\demo\main.py
import tkinter as tk
from tkinter import ttk, messagebox
from gui.app import StudentCheckInApp
from core.evaluation.evaluator import FaceRecognitionEvaluator

def main():
    root = tk.Tk()
    app = StudentCheckInApp(root)
    
    # Thêm menu đánh giá
    menubar = tk.Menu(root)
    root.config(menu=menubar)
    
    evaluation_menu = tk.Menu(menubar, tearoff=0)
    menubar.add_cascade(label="Đánh giá", menu=evaluation_menu)
    
    def run_evaluation():
        """Chạy đánh giá model"""
        try:
            evaluator = FaceRecognitionEvaluator(app.face_recognition, app.db_manager)
            
            # Đánh giá với threshold mặc định
            metrics = evaluator.evaluate_recognition()
            
            # Phân tích các threshold khác nhau
            threshold_results = evaluator.threshold_analysis()
            
            messagebox.showinfo("Kết quả đánh giá", 
                              f"Đã hoàn thành đánh giá!\n\n"
                              f"Accuracy: {metrics['accuracy']:.4f}\n"
                              f"Precision: {metrics['precision']:.4f}\n"
                              f"Recall: {metrics['recall']:.4f}\n"
                              f"F1-score: {metrics['f1']:.4f}\n\n"
                              f"Kết quả chi tiết được lưu trong thư mục data/evaluation_results")
        except Exception as e:
            messagebox.showerror("Lỗi", f"Không thể chạy đánh giá:\n{str(e)}")
    
    evaluation_menu.add_command(label="Chạy đánh giá", command=run_evaluation)
    
    root.mainloop()

if __name__ == "__main__":
    main()
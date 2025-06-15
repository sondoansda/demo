import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os
from datetime import datetime
import sys

class FaceRecognitionEvaluator:
    def __init__(self, model, db_manager):
        """
        Khởi tạo evaluator
        
        Args:
            model: Model FaceRecognition cần đánh giá
            db_manager: Database manager để lấy dữ liệu test
        """
        self.model = model
        self.db_manager = db_manager
        self.predictions = []
        self.ground_truth = []
        self.scores = []
        
        # Tạo thư mục cho evaluation results
        self.results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                      'data', 'evaluation_results')
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
            
        self.logger = self.setup_logging()

    def setup_logging(self):
        """Cấu hình logging với hỗ trợ Unicode"""
        log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                              'logs')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        log_file = os.path.join(log_dir, f'face_recognition_eval_{datetime.now().strftime("%Y%m%d")}.log')
        
        # Cấu hình logger
        logger = logging.getLogger('face_recognition_eval')
        logger.setLevel(logging.INFO)
        
        # File handler with UTF-8 encoding
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        
        # Console handler with UTF-8 encoding
        if sys.platform == 'win32':
            # Windows specific setup
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
            ch = logging.StreamHandler(sys.stdout)
        else:
            ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Format
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                    datefmt='%Y-%m-%d %H:%M:%S')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Clear any existing handlers
        logger.handlers.clear()
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        # Prevent propagation to parent loggers
        logger.propagate = False
        
        return logger

    def evaluate_recognition(self, threshold=0.7):
        """
        Đánh giá model trên tập test
        
        Args:
            threshold: Ngưỡng cosine similarity để xác định match
        """
        self.logger.info("Starting evaluation with threshold=%.2f", threshold)
        
        # Reset lists
        self.predictions = []
        self.ground_truth = []
        self.scores = []
        
        # Lấy danh sách sinh viên để test
        students = self.db_manager.get_students()
        
        total = len(students)
        correct = 0
        
        for student in students:
            student_id = student[0]  # Assuming first column is student_id
            
            # Lấy embedding của student từ database
            true_embedding = self.model.get_embedding(student_id)
            
            if true_embedding is None:
                continue
                
            # Test với tất cả students khác
            for test_student in students:
                test_id = test_student[0]
                test_embedding = self.model.get_embedding(test_id)
                
                if test_embedding is None:
                    continue
                
                # Tính cosine similarity
                similarity = self.model.compute_similarity(true_embedding, test_embedding)
                
                # Lưu ground truth (1 nếu cùng ID, 0 nếu khác)
                self.ground_truth.append(1 if student_id == test_id else 0)
                
                # Lưu prediction dựa trên threshold
                self.predictions.append(1 if similarity >= threshold else 0)
                
                # Lưu similarity score để vẽ ROC
                self.scores.append(similarity)
                
                if (student_id == test_id and similarity >= threshold) or \
                   (student_id != test_id and similarity < threshold):
                    correct += 1
                    
        # Tính và log metrics
        metrics = self.calculate_metrics()
        
        self.logger.info("Evaluation results:")
        self.logger.info("Accuracy: %.4f", metrics['accuracy'])
        self.logger.info("Precision: %.4f", metrics['precision'])
        self.logger.info("Recall: %.4f", metrics['recall'])
        self.logger.info("F1-score: %.4f", metrics['f1'])
        
        # Vẽ và lưu visualization
        self.plot_confusion_matrix()
        self.plot_roc_curve()
        
        return metrics

    def calculate_metrics(self):
        """Tính toán các metrics"""
        return {
            'accuracy': accuracy_score(self.ground_truth, self.predictions),
            'precision': precision_score(self.ground_truth, self.predictions, average='binary'),
            'recall': recall_score(self.ground_truth, self.predictions, average='binary'),
            'f1': f1_score(self.ground_truth, self.predictions, average='binary')
        }

    def plot_confusion_matrix(self):
        """Vẽ confusion matrix"""
        cm = confusion_matrix(self.ground_truth, self.predictions)
        
        plt.figure(figsize=(10,8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Lưu plot
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'))
        plt.close()
        
        self.logger.info("Confusion matrix saved at: %s/confusion_matrix.png", self.results_dir)

    def plot_roc_curve(self):
        """Vẽ ROC curve"""
        fpr, tpr, _ = roc_curve(self.ground_truth, self.scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(10,8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        
        # Lưu plot
        plt.savefig(os.path.join(self.results_dir, 'roc_curve.png'))
        plt.close()
        
        self.logger.info("ROC curve saved at: %s/roc_curve.png", self.results_dir)

    def threshold_analysis(self, thresholds=[0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]):
        """Phân tích hiệu quả với các ngưỡng khác nhau"""
        results = []
        
        self.logger.info("Starting threshold analysis...")
        
        for threshold in thresholds:
            metrics = self.evaluate_recognition(threshold)
            results.append({
                'threshold': threshold,
                **metrics
            })
            
        # Vẽ biểu đồ so sánh
        metrics_names = ['accuracy', 'precision', 'recall', 'f1']
        plt.figure(figsize=(12, 6))
        
        for metric in metrics_names:
            values = [r[metric] for r in results]
            plt.plot(thresholds, values, marker='o', label=metric)
            
        plt.xlabel('Threshold')
        plt.ylabel('Score')
        plt.title('Metrics vs Threshold')
        plt.legend()
        plt.grid(True)
        
        # Lưu plot
        plt.savefig(os.path.join(self.results_dir, 'threshold_analysis.png'))
        plt.close()
        
        self.logger.info("Threshold analysis saved at: %s/threshold_analysis.png", self.results_dir)
        return results

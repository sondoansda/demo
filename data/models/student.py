class Student:
    def __init__(self, student_id, name, video_path):
        self.student_id = student_id
        self.name = name
        self.video_path = video_path

    def __repr__(self):
        return f"Student({self.student_id}, {self.name}, {self.video_path})"
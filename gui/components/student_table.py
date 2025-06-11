import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from data.database.db_manager import DatabaseManager

class StudentTable(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.db_manager = DatabaseManager()

        self.tree = ttk.Treeview(self, columns=("ID", "Name", "Video Path"), show="headings")
        self.tree.heading("ID", text="Student ID")
        self.tree.heading("Name", text="Name")
        self.tree.heading("Video Path", text="Video Path")
        self.tree.pack(fill=tk.BOTH, expand=True)

        delete_button = ttk.Button(self, text="Delete Student", command=self.delete_student)
        delete_button.pack(pady=5)

    def update_students(self, students):
        for row in self.tree.get_children():
            self.tree.delete(row)
        for student in students:
            self.tree.insert("", tk.END, values=student)

    def delete_student(self):
        selected_item = self.tree.selection()
        if not selected_item:
            messagebox.showerror("Error", "Please select a student to delete")
            return

        student_id = self.tree.item(selected_item, "values")[0]
        try:
            self.db_manager.delete_student(student_id)
            self.tree.delete(selected_item)
            messagebox.showinfo("Success", "Student deleted successfully")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete student: {str(e)}")
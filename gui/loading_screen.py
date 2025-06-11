import tkinter as tk
from tkinter import ttk

class LoadingScreen:
    def __init__(self, root):
        self.window = tk.Toplevel(root)
        self.window.title("Loading")
        self.window.geometry("300x150")
        self.window.transient(root)
        self.window.grab_set()

        self.label = ttk.Label(self.window, text="Initializing...", font=("Helvetica", 12))
        self.label.pack(pady=20)

        self.progress = ttk.Progressbar(self.window, length=200, mode='determinate')
        self.progress.pack(pady=10)

    def update_progress(self, value, text=None):
        if text:
            self.label.config(text=text)
        self.progress['value'] = value
        self.window.update()

    def close(self):
        self.window.grab_release()
        self.window.destroy()
# Đường dẫn: c:\Users\Dell\Desktop\demo\main.py
import tkinter as tk
from gui.app import StudentCheckInApp

def main():
    root = tk.Tk()
    app = StudentCheckInApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
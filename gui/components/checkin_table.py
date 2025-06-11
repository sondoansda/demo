import tkinter as tk
from tkinter import ttk, messagebox

class CheckinTable(ttk.Frame):
    def __init__(self, parent, db_manager):
        super().__init__(parent)
        self.db_manager = db_manager
        
        # Create a frame for the table
        self.table_frame = ttk.Frame(self)
        self.table_frame.pack(fill='both', expand=True)
        
        # Create treeview with scrollbar
        self.tree = ttk.Treeview(self.table_frame, columns=('ID', 'Name', 'Time'), show='headings')
        scrollbar = ttk.Scrollbar(self.table_frame, orient='vertical', command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        # Setup columns
        self.tree.heading('ID', text='Student ID')
        self.tree.heading('Name', text='Name')
        self.tree.heading('Time', text='Check-in Time')
        
        self.tree.column('ID', width=100)
        self.tree.column('Name', width=150)
        self.tree.column('Time', width=150)
        
        # Pack the treeview and scrollbar
        self.tree.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')

        # Create buttons frame
        self.buttons_frame = ttk.Frame(self)
        self.buttons_frame.pack(fill='x', pady=5)
        
        # Add Clear History button
        self.clear_button = ttk.Button(
            self.buttons_frame, 
            text="Clear Check-in History",
            command=self.confirm_clear_history
        )
        self.clear_button.pack(side='left', padx=5)
        
        # Schedule periodic updates
        self.update_checkins()
    
    def confirm_clear_history(self):
        """Show confirmation dialog before clearing history"""
        response = messagebox.askyesno(
            "Confirm Delete",
            "Are you sure you want to delete all check-in history?\nThis action cannot be undone.",
            icon='warning'
        )
        if response:
            deleted_count = self.db_manager.delete_checkin_history()
            messagebox.showinfo(
                "Success",
                f"Successfully deleted {deleted_count} check-in records."
            )
            self.update_checkins()  # Refresh the table
    
    def update_checkins(self):
        """Update the check-in table with recent entries"""
        try:
            # Clear existing items
            for item in self.tree.get_children():
                self.tree.delete(item)
            
            # Get recent check-ins
            checkins = self.db_manager.get_recent_checkins(limit=20)
            print(f"Retrieved {len(checkins)} recent check-ins")
            
            # Add new items
            for student_id, name, timestamp in checkins:
                self.tree.insert('', 0, values=(student_id, name, timestamp))  # Thêm vào đầu bảng
            
            # Update UI
            self.update_idletasks()
            
            # Schedule next update
            self.after(3000, self.update_checkins)  # Giảm thời gian cập nhật xuống 3 giây
        except Exception as e:
            print(f"Error updating check-in table: {e}")
            # Retry update after error
            self.after(3000, self.update_checkins)
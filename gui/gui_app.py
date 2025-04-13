import tkinter as tk
from tkinter import ttk
import threading
import cv2
from PIL import Image, ImageTk

# Backend hooks — replace with your actual start/stop functions
def start_detection():
    print("[INFO] Posture detection started")

def stop_detection():
    print("[INFO] Posture detection stopped")

class PostureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Posture Tracker")
        self.root.geometry("800x600")

        self.running = False
        self.cap = None

        # Top controls
        controls = ttk.Frame(root)
        controls.pack(pady=10)

        self.start_btn = ttk.Button(controls, text="Start", command=self.start)
        self.start_btn.grid(row=0, column=0, padx=10)

        self.stop_btn = ttk.Button(controls, text="Stop", command=self.stop, state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=10)

        self.settings_btn = ttk.Button(controls, text="Settings", command=self.open_settings)
        self.settings_btn.grid(row=0, column=2, padx=10)

        # Video display
        self.video_frame = ttk.Label(root)
        self.video_frame.pack()

    def start(self):
        self.running = True
        self.cap = cv2.VideoCapture(0)
        self.start_btn['state'] = 'disabled'
        self.stop_btn['state'] = 'normal'
        start_detection()
        threading.Thread(target=self.update_feed, daemon=True).start()

    def stop(self):
        self.running = False
        self.start_btn['state'] = 'normal'
        self.stop_btn['state'] = 'disabled'
        stop_detection()
        if self.cap:
            self.cap.release()
        self.video_frame.config(image='')

    def update_feed(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                continue
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_frame.imgtk = imgtk
            self.video_frame.config(image=imgtk)
        self.cap.release()

    def open_settings(self):
        settings_win = tk.Toplevel(self.root)
        settings_win.title("Settings")
        settings_win.geometry("300x200")
        ttk.Label(settings_win, text="(Settings config goes here)").pack(pady=20)

if __name__ == "__main__":
    root = tk.Tk()
    app = PostureApp(root)
    root.mainloop()

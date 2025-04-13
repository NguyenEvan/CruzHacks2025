
import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from pose.detector import init_pose
from posture_loop import process_frame_with_posture
from collections import deque


class SlouchDetectorApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Posture Detector")
        self.window.geometry("1200x680")
        self.running = False
        self.pose_model = init_pose()
        self.z_buffer = deque(maxlen=100)

        self.style = ttk.Style()
        self.style.theme_use("clam")
        self._configure_pastel_theme()

        self.setup_ui()

    #cream pastel background
    def _configure_pastel_theme(self):
        self.window.configure(bg="#fdf6f0")
        self.style.configure("TFrame", background="#fdf6f0")
        self.style.configure("TLabel", background="#fdf6f0", foreground="#4a4a4a", font=("Segoe UI", 10))
        self.style.configure("TButton",
                             background="#ffeaa7",
                             foreground="#4a4a4a",
                             font=("Segoe UI", 10),
                             padding=6)
        self.style.map("TButton", background=[("active", "#fab1a0")])

    def setup_ui(self):
        self.main_frame = ttk.Frame(self.window, padding=10)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        #video feed
        self.video_label = ttk.Label(self.main_frame, anchor="center", relief=tk.SOLID, borderwidth=2)
        self.video_label.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="nsew")

        #sidebar
        self.sidebar = tk.Text(
            self.main_frame,
            width=45,
            wrap=tk.WORD,
            font=("Segoe UI", 11),
            bg="#c8d6e5",     #pastel blue
            fg="#4a4a4a",     #soft dark gray color text
            insertbackground="#4a4a4a",
            relief=tk.FLAT,
            padx=10,
            pady=10
        )
        self.sidebar.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="nsew")
        self.sidebar.insert(tk.END, "🧠 Waiting for feedback...\n")
        self.sidebar.config(state=tk.DISABLED)

        self.main_frame.columnconfigure(0, weight=3)
        self.main_frame.columnconfigure(1, weight=2)
        self.main_frame.rowconfigure(0, weight=1)

        #buttons
        self.button_frame = ttk.Frame(self.window, padding=(10, 5))
        self.button_frame.pack(fill=tk.X)

        self.start_button = ttk.Button(self.button_frame, text="▶ Start", command=self.start_camera)
        self.start_button.pack(side=tk.LEFT, padx=10)

        self.stop_button = ttk.Button(self.button_frame, text="⏹ Stop", command=self.stop_camera)
        self.stop_button.pack(side=tk.LEFT, padx=10)

    def start_camera(self):
        if not self.running:
            self.cap = cv2.VideoCapture(0)
            self.running = True
            self.update_frame()

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()
        self.video_label.config(image='')
        self.sidebar.config(state=tk.NORMAL)
        self.sidebar.delete(1.0, tk.END)
        self.sidebar.insert(tk.END, "⛔ Camera stopped.\n")
        self.sidebar.config(state=tk.DISABLED)

    def update_frame(self):
        try:
            if self.running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    processed_frame, z_score, gemini_feedback = process_frame_with_posture(
                        frame, self.pose_model, self.z_buffer
                    )

                    frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(frame_rgb)
                    image_tk = ImageTk.PhotoImage(image)

                    self.video_label.configure(image=image_tk)
                    self.video_label.image = image_tk

                    self.z_buffer.append((z_score, image))

                    if gemini_feedback:
                        self.sidebar.config(state=tk.NORMAL)
                        self.sidebar.delete(1.0, tk.END)
                        self.sidebar.insert(tk.END, "🧠 Gemini Feedback:\n\n")
                        self.sidebar.insert(tk.END, gemini_feedback.strip())
                        self.sidebar.config(state=tk.DISABLED)

            if self.running:
                self.window.after(15, self.update_frame)
        except Exception as e:
            print(f"[update_frame ERROR] {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = SlouchDetectorApp(root)
    root.mainloop()

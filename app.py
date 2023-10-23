import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk  # Import PIL modules

import numpy as np
from yolov4 import *

bike_detect = yolo_helmet_v4()
camera = cv2.VideoCapture(0)
frame_width = int(camera.get(3))
frame_height = int(camera.get(4))

size = (frame_width, frame_height)
result = cv2.VideoWriter('filename.avi',
                         cv2.VideoWriter_fourcc(*'MJPG'),
                         10, size)


class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Camera App")

        self.cap = cv2.VideoCapture(0)
        self.video_frame = tk.Label(self.root)
        self.root.geometry("500x500")
        self.video_frame.pack()

        self.start_button = ttk.Button(
            self.root, text="Start Camera", command=self.start_camera)
        self.stop_button = ttk.Button(
            self.root, text="Stop Camera", command=self.stop_camera)
        self.start_button.pack()
        self.stop_button.pack()
        self.stop_button.config(state="disabled")

        self.running = False

    def start_camera(self):
        self.start_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.running = True
        self.show_camera_feed()

    def stop_camera(self):
        self.start_button.config(state="normal")
        self.stop_button.config(state="disabled")
        self.running = False
        result.release()

    def show_camera_feed(self):

        if self.running:
            ret, frame = self.cap.read()
            if ret:
                bike_detect.detect_helmet_V4(frame)
                result.write(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                photo = ImageTk.PhotoImage(image=Image.fromarray(frame))
                self.video_frame.config(image=photo)
                self.video_frame.image = photo
                self.root.after(10, self.show_camera_feed)


if __name__ == "__main__":
    root = tk.Tk()
    app = CameraApp(root)
    root.mainloop()

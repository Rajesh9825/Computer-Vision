import cv2
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk

class FaceDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Detector App")
        self.root.geometry("800x600")
        
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.cap = None  # For webcam
        self.running = False  # Track whether webcam is streaming
        
        self.label = tk.Label(root, text="Face Detector App", font=("Arial", 24))
        self.label.pack(pady=10)

        self.video_label = tk.Label(root)
        self.video_label.pack()

        self.start_button = tk.Button(root, text="Start", command=self.start_detection, font=("Arial", 14), bg="green", fg="white")
        self.start_button.pack(side=tk.LEFT, padx=50, pady=20)

        self.stop_button = tk.Button(root, text="Stop", command=self.stop_detection, font=("Arial", 14), bg="red", fg="white")
        self.stop_button.pack(side=tk.RIGHT, padx=50, pady=20)

    def start_detection(self):
        if not self.running:
            self.cap = cv2.VideoCapture(1)
            self.running = True
            self.detect_faces()

    def stop_detection(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.video_label.config(image='') # Clear image from GUI

    def detect_faces(self):
        if self.running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert frame to grayscale (face detection works better this way).
                faces = self.face_cascade.detectMultiScale(gray, 1.5, 4)
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)

                # Convert image to Tkinter compatible format
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
                img = Image.fromarray(img)  # Convert to PIL Image
                imgtk = ImageTk.PhotoImage(image=img)  # Convert to Tkinter image
                self.video_label.imgtk = imgtk  # Store reference
                self.video_label.configure(image=imgtk)  # Display in label


            # Repeat this method after 10 ms
            self.root.after(40, self.detect_faces)  # This schedules detect_faces() to run again after 10 milliseconds (like a loop, without freezing the GUI).


    #When the user closes the app window, make sure to stop webcam and exit properly.
    def on_closing(self):
        self.stop_detection()
        self.root.destroy()

# Create window and run app
root = tk.Tk() # Create the Tkinter app window.
app = FaceDetectorApp(root) # Initialize your app class.
root.protocol("WM_DELETE_WINDOW", app.on_closing) # Attach a callback to close properly.
root.mainloop() # Run the main Tkinter loop.

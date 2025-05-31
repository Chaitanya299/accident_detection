import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf

# Define class labels (update as per your training data)
class_labels = ["Accident", "Non Accident"]

# Load the saved model (ensure the correct path to your saved model file)
try:
    model = tf.keras.models.load_model("model_accident_classifier.h5")  # Replace with the actual path
except FileNotFoundError:
    print("Model file not found. Please ensure 'model_accident_classifier.h5' exists in the working directory.")
    exit()

def classify_image(file_path):
    """Classify an image."""
    image = Image.open(file_path)
    image = image.resize((250, 250))  # Resize to match the input size of your model
    image = np.array(image) / 255.0  # Normalize image
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Perform prediction
    prediction = model.predict(image)
    class_index = np.argmax(prediction)
    class_name = class_labels[class_index]

    classification_label.config(text=f"Classification: {class_name}")

def upload_image():
    """Handle image upload and classification."""
    file_path = filedialog.askopenfilename(filetypes=[("Image files", ".jpg;.jpeg;*.png")])
    if file_path:
        display_image(file_path)
        classify_image(file_path)

def display_image(file_path):
    """Display the uploaded image in the GUI."""
    image = Image.open(file_path)
    image = image.resize((300, 300), Image.Resampling.LANCZOS)
    photo = ImageTk.PhotoImage(image)
    image_label.config(image=photo)
    image_label.image = photo

def classify_frame(frame):
    """Classify a single frame of a video."""
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    frame = cv2.resize(frame, (250, 250))  # Resize to match the input size of your model
    frame = np.array(frame) / 255.0  # Normalize frame
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension

    # Perform prediction
    prediction = model.predict(frame)
    class_index = np.argmax(prediction)
    class_name = class_labels[class_index]

    classification_label.config(text=f"Classification: {class_name}")

def upload_video():
    """Handle video upload and frame-by-frame classification."""
    file_path = filedialog.askopenfilename(filetypes=[("Video files", ".mp4;.avi")])  # Add valid video formats
    if file_path:
        display_video(file_path)

def display_video(file_path):
    """Process and classify frames from a video."""
    cap = cv2.VideoCapture(file_path)

    # Get the frame rate of the video
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    skip_frames = int(frame_rate / 5)  # Skip frames to process video at 5fps
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            classify_frame(frame)

            # Convert the frame to Pillow image and display it
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img = img.resize((300, 300))
            photo = ImageTk.PhotoImage(image=img)
            image_label.config(image=photo)
            image_label.image = photo

            root.update_idletasks()
            root.update()

        frame_count += 1 

    cap.release()

# GUI setup
root = tk.Tk()
root.title("Image and Video Classifier")

label = tk.Label(root, text="Upload an image or video for classification:")
label.pack(pady=10)

upload_image_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_image_button.pack(pady=5)

upload_video_button = tk.Button(root, text="Upload Video", command=upload_video)
upload_video_button.pack(pady=5)

image_label = tk.Label(root)
image_label.pack(pady=5)

classification_label = tk.Label(root)
classification_label.pack(pady=5)

root.mainloop()

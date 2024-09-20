# Image-Object-Detection-using-MobileNet-SSD
This Python application utilizes OpenCV's DNN module and a pre-trained MobileNet-SSD model to detect objects in images. The application features a graphical user interface (GUI) built with Tkinter that allows users to upload an image for detection.
import cv2
import numpy as np
import os
import urllib.request
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Model files
model_file = "mobilenet_iter_73000.caffemodel"
prototxt_file = "deploy.prototxt"

# URLs for downloading model files
model_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/mobilenet_iter_73000.caffemodel"
prototxt_url = "https://github.com/chuanqi305/MobileNet-SSD/raw/master/deploy.prototxt"

def download_file(url, filename):
    print(f"Downloading {filename}...")
    urllib.request.urlretrieve(url, filename)
    print(f"{filename} downloaded successfully.")

# Check if model files exist, download if not
if not os.path.exists(model_file):
    download_file(model_url, model_file)
if not os.path.exists(prototxt_file):
    download_file(prototxt_url, prototxt_file)

# Load the model
net = cv2.dnn.readNetFromCaffe(prototxt_file, model_file)

# Define a list of class labels
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

def detect_objects(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx]}: {confidence * 100:.2f}%"
            cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(image, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return image

def upload_and_detect():
    file_path = filedialog.askopenfilename()
    if file_path:
        image = cv2.imread(file_path)
        if image is None:
            print(f"Error loading image {file_path}")
            return
        result_image = detect_objects(image)
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_image)
        result_image = ImageTk.PhotoImage(result_image)
        
        result_label.config(image=result_image)
        result_label.image = result_image

# Create the main window
root = tk.Tk()
root.title("Image Object Detection")

# Create and pack the upload button
upload_button = tk.Button(root, text="Upload and Detect", command=upload_and_detect)
upload_button.pack(pady=10)

# Create and pack the result label
result_label = tk.Label(root)
result_label.pack(pady=10)

root.mainloop()

# flood_Segmentation_project
Segmentation of flood-hit areas using U-Net model

# Flood Area Segmentation Project

This project aims to create a segmentation model that can accurately identify water regions in images of flood-hit areas. Such models can be used for flood surveys, better decision-making, and planning.

## Introduction

Floods can cause significant damage and pose serious risks to human lives and property. Rapid identification of water-affected areas is crucial for effective disaster management and response. This project uses a U-Net model to perform image segmentation, identifying floodwater in images. The dataset contains images of flood-hit areas and corresponding masks showing the water regions.

## Tools

The project utilizes the following tools and libraries:
- Python
- OpenCV
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Label Studio (for data annotation)
- tqdm (for progress bars)
- scikit-learn (for dataset splitting)

## Dataset

The dataset contains 290 images of flood-hit areas with self-annotated masks. The masks were created using Label Studio, an open-source data labeling software. The dataset is split into training and validation sets to train and evaluate the segmentation model.

## U-Net Model

The U-Net architecture used in this project is a convolutional neural network designed for biomedical image segmentation, but it is effective for general image segmentation tasks. The architecture consists of an encoder (downsampling path) and a decoder (upsampling path), with skip connections between corresponding layers of the encoder and decoder.

## Results

The model is trained for 10 epochs and evaluated on a validation set. The training and validation loss curves are plotted to monitor the model's performance. The model's predictions are visualized by overlaying the predicted masks on the original images.

## Authors

- Taiseer Almedani
- Mohammad Aldwaish

## Usage

To use the pre-trained model to segment flood areas in a video, run the following script:

```python
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('model.h5')

# Threshold value
threshold = 0.5

# Define a color for the masks (e.g., red)
mask_color = np.array([255, 0, 0], dtype=np.uint8)  # Red color

# Function to preprocess frames
def preprocess_frame(frame, target_size=(128, 128)):
    frame = cv2.resize(frame, target_size)
    frame = frame.astype(np.float32) / 255.0
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    return frame

# Function to postprocess the predicted mask
def postprocess_mask(mask, original_shape):
    mask = mask.squeeze()
    mask = (mask > threshold).astype(np.uint8)
    mask = cv2.resize(mask, (original_shape[1], original_shape[0]))
    return mask

# Function to overlay the mask on the original image
def overlay_mask_on_frame(frame, mask):
    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    colored_mask[mask == 1] = mask_color
    overlay_frame = cv2.addWeighted(frame, 1, colored_mask, 0.5, 0)
    return overlay_frame

# OpenCV VideoCapture
cap = cv2.VideoCapture('1.mp4')  # Change to 0 for the default camera

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    preprocessed_frame = preprocess_frame(frame)

    # Predict the mask
    predicted_mask = model.predict(preprocessed_frame)
    
    # Postprocess the mask
    mask = postprocess_mask(predicted_mask, frame.shape[:2])

    # Overlay the mask on the original frame
    overlay_frame = overlay_mask_on_frame(frame, mask)

    # Display the frame
    cv2.imshow('Video with Mask', overlay_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

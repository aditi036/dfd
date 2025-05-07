import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.xception import preprocess_input
from mtcnn import MTCNN
import time
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Parameters
TIME_STEPS = 30  # Number of frames per video
HEIGHT, WIDTH = 299, 299

# Load the model
def build_model(lstm_hidden_size=256, num_classes=2, dropout_rate=0.5):
    inputs = layers.Input(shape=(TIME_STEPS, HEIGHT, WIDTH, 3))
    base_model = keras.applications.Xception(weights='imagenet', include_top=False, pooling='avg')
    x = layers.TimeDistributed(base_model)(inputs)
    x = layers.LSTM(lstm_hidden_size)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = keras.Model(inputs, outputs)
    return model

# Load the model weights
model_path = 'Phase 2_Epoch6.keras'  # Path to the model
model = build_model()
model.load_weights(model_path)

# Function to extract faces from video
def extract_faces_from_video(video_path, num_frames=TIME_STEPS):
    detector = MTCNN()
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)

    idx = 0
    success = True
    faces_detected = False  # Flag to track if any faces are detected
    while success and len(frames) < num_frames:
        success, frame = cap.read()
        if not success:
            break
        if idx in frame_indices:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = detector.detect_faces(frame_rgb)
            if detections:
                faces_detected = True
                x, y, width, height = detections[0]['box']
                x, y = max(0, x), max(0, y)
                x2, y2 = x + width, y + height
                face = frame_rgb[y:y2, x:x2]
                face_image = Image.fromarray(face).resize((WIDTH, HEIGHT))
                face_array = np.array(face_image)
                face_array = preprocess_input(face_array)
                frames.append(face_array)
            else:
                # If no face is detected, continue but do not add to frames
                pass
        idx += 1

    cap.release()

    # If no faces were detected at all, raise an error
    if not faces_detected:
        raise ValueError("No faces detected in the video. Please upload a video with faces.")

    # Ensure the correct number of frames
    if len(frames) < num_frames:
        last_frame = frames[-1] if frames else np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32)
        frames += [last_frame] * (num_frames - len(frames))

    video_array = np.array(frames)
    video_array = np.expand_dims(video_array, axis=0)
    return video_array, frames  # Return individual frames too for display

# Function to make predictions
def make_prediction(video_file):
    # Save the uploaded file temporarily
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())

    # Process the video and get the preprocessed frames
    video_array, frames = extract_faces_from_video("temp_video.mp4", num_frames=TIME_STEPS)

    # Make prediction
    predictions = model.predict(video_array)
    predicted_class = np.argmax(predictions, axis=1)[0]
    probabilities = predictions[0]

    class_names = ['Real', 'Fake']
    return class_names[predicted_class], probabilities, frames

# Streamlit UI
st.title("VARS: Video Analysis for Realness & Synthetic Detection")
st.write("Welcome to VARS, a DeepFake detection tool that uses advanced deep learning models to analyze videos and predict if they contain real or synthetically generated content.")

st.header("How VARS Works")
st.markdown("""
- **Step 1**: VARS extracts key frames from the uploaded video.
- **Step 2**: Each frame is analyzed to locate faces using a face detection model.
- **Step 3**: The frames are processed and fed into a neural network that classifies the video as 'Real' or 'Fake'.
""")

st.write("Upload a video to predict if it's real or fake.")

# File uploader for video
video_file = st.file_uploader("Upload Video", type=["mp4", "mov", "avi"])

if video_file is not None:
    # Start timer
    start_time = time.time()

    # Initialize progress bar
    progress_bar = st.progress(0)
    
    with st.spinner("Processing video... This may take a few moments."):
        # Sequential status updates during the video processing phase
        st.info("Extracting frames from the video...")
        progress_bar.progress(25)
        time.sleep(1)
        
        st.info("Detecting faces in the frames...")
        progress_bar.progress(50)
        time.sleep(1)
        
        st.info("Preprocessing frames for model input...")
        progress_bar.progress(75)
        time.sleep(1)
        
        st.info("Running prediction on the video...")
        progress_bar.progress(100)
        time.sleep(1)
        
        try:
            # Make prediction
            predicted_class, probabilities, frames = make_prediction(video_file)
        
        except ValueError as e:
            st.error(str(e))  # Show error message to the user
            st.stop()  # Stop execution

    # End timer
    end_time = time.time()
    processing_time = end_time - start_time

    # Display prediction results
    st.subheader("Prediction Results")
    st.write(f"**Predicted Class**: {predicted_class}")
    st.write(f"**Class Probabilities**: Real: {probabilities[0]:.4f}, Fake: {probabilities[1]:.4f}")
    st.write(f"**Processing Time**: {processing_time:.2f} seconds")

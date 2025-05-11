import streamlit as st
import os
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.xception import preprocess_input
from mtcnn import MTCNN
import time
import pandas as pd
import matplotlib.pyplot as plt
import base64
# TensorFlow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Parameters
TIME_STEPS = 30  # Frames per video
HEIGHT, WIDTH = 299, 299

# Model builder
def build_model(lstm_hidden_size=256, num_classes=2, dropout_rate=0.5):
    inputs = layers.Input(shape=(TIME_STEPS, HEIGHT, WIDTH, 3))
    base_model = tf.keras.applications.Xception(weights='imagenet', include_top=False, pooling='avg')
    x = layers.TimeDistributed(base_model)(inputs)
    x = layers.LSTM(lstm_hidden_size)(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.Model(inputs, outputs)
    return model

# Load model
model_path = 'COMBINED_best_Phase1.keras'
model = build_model()
model.load_weights(model_path)

def preprocess_image(image):
    """
    Preprocess image for model input
    
    Args:
        image (PIL.Image or numpy.ndarray): Input image
    
    Returns:
        numpy.ndarray: Preprocessed image
    """
    # Convert to numpy array if it's a PIL Image
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Resize to expected input size
    image = cv2.resize(image, (WIDTH, HEIGHT))
    
    # Preprocess for Xception model
    image = preprocess_input(image)
    
    return image

def extract_faces_from_video(video_path, num_frames=TIME_STEPS, skip_frames=0):
    """
    Extract faces from video with more robust frame selection
    
    Args:
        video_path (str): Path to the video file
        num_frames (int): Number of frames to extract
        skip_frames (int): Number of initial frames to skip
    
    Returns:
        tuple: (video_array, frames) or (None, None) if no faces detected
    """
    detector = MTCNN()
    cap = cv2.VideoCapture(video_path)
    
    # Get total frame count
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Validate input parameters
    skip_frames = max(0, min(skip_frames, frame_count - num_frames))
    
    # Calculate frame indices to sample
    frame_indices = np.linspace(skip_frames, frame_count - 1, num_frames, dtype=int)
    
    frames = []
    processed_frames = []
    
    for idx in range(frame_count):
        success, frame = cap.read()
        if not success:
            break
        
        # Check if this frame should be processed
        if idx in frame_indices:
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces
            detections = detector.detect_faces(frame_rgb)
            
            if detections:
                # Get the first detected face
                x, y, width, height = detections[0]['box']
                x, y = max(0, x), max(0, y)
                x2, y2 = x + width, y + height
                
                # Extract face
                face = frame_rgb[y:y2, x:x2]
                
                # Convert to PIL Image and preprocess
                face_image = Image.fromarray(face)
                processed_face = preprocess_image(face_image)
                
                frames.append(processed_face)
            else:
                # If no face detected, use a zero array
                frames.append(np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32))
            
            # Stop if we have collected enough frames
            if len(frames) == num_frames:
                break
    
    cap.release()
    
    # If not enough frames were found, pad with the last frame or zeros
    while len(frames) < num_frames:
        if frames:
            frames.append(frames[-1])  # Pad with the last frame
        else:
            frames.append(np.zeros((HEIGHT, WIDTH, 3), dtype=np.float32))
    
    # Convert to numpy array and expand dimensions
    video_array = np.expand_dims(np.array(frames), axis=0)
    
    return video_array, frames

def make_prediction(video_file):
    """
    Make prediction on the uploaded video file
    
    Args:
        video_file: Uploaded video file object
    
    Returns:
        tuple: (predicted_class, probabilities, frames) or (None, None, None) if error
    """
    try:
        # Ensure the directory exists
        os.makedirs('temp', exist_ok=True)
        
        # Save the uploaded file
        temp_video_path = os.path.join('temp', 'temp_video.mp4')
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())
        
        # Extract faces and video array
        video_array, frames = extract_faces_from_video(temp_video_path)
        
        # Validate the video array
        if video_array is None or video_array.shape[1] != TIME_STEPS:
            st.error("Unable to process video. Please ensure the video contains clear, visible faces.")
            return None, None, None
        
        # Make prediction
        predictions = model.predict(video_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        probabilities = predictions[0]
        
        return predicted_class, probabilities, frames
    
    except Exception as e:
        st.error(f"An error occurred while processing the video: {str(e)}")
        return None, None, None
    finally:
        # Clean up temporary file
        if os.path.exists(temp_video_path):
            os.remove(temp_video_path)

# (Keep all the previous imports and functions)

# Streamlit UI
st.set_page_config(page_title="VARS - Video Analysis", layout="wide")
st.markdown("<style>h1{font-size: 45px !important;}</style>", unsafe_allow_html=True)

# Create two columns for header and main content
header_col1, header_col2 = st.columns([1, 1])

def get_base64_image(file_path):
    with open(file_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Path to the uploaded image
image_path = "Image2.png"  # Ensure this is the correct path to your saved image

# Convert image to Base64
image_base64 = get_base64_image(image_path)

# Header Section with Image
with header_col1:
    image = Image.open("Image2.png")
    desired_height = 300  # Reduced height
    aspect_ratio = image.width / image.height
    new_width = int(desired_height * aspect_ratio)
    resized_image = image.resize((new_width, desired_height))
    # st.image(resized_image, use_container_width=True)

# Title and Description
with header_col2:
    st.markdown(
    """
    <style>
    .header-container {
        position: relative;
        text-align: center;
        color: white;
        font-family: 'Arial', sans-serif;
    }
    .header-image {
        width: 100%;
        height: 300px;
        object-fit: cover;
    }
    .header-text {
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        font-size: 50px;
        font-weight: bold;
        text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.7);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# HTML content for the header
st.markdown(
    f"""
    <div class="header-container">
        <img src="data:image/png;base64,{image_base64}" class="header-image" />
        <div class="header-text">VARS: Video Analysis for Real & Synthetic Detection</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.title("How It Works")
st.sidebar.markdown(
    """
1. 📤 **Upload Video:** 
   - Choose a video file (mp4, mov, avi) 
   (make sure suspected video is withing 1 second and doesn't have multiple faces)
2. 🔍 **Process Frames:** 
   - Detect and analyze faces
3. 🤖 **AI Analysis:** 
   - Predict 'Real' or 'Fake'
4. 📊 **Detailed Results:** 
   - View probabilities and insights
"""
)
# st.sidebar.info("**Note:** This app is for educational purposes only. The results may not be 100% accurate.")

# Upload video
st.subheader("🎥 Upload Your Video")
video_file = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi"], label_visibility="collapsed")
st.markdown(
    """
    <style>
    .fixed-height-col {
        height: 500px; /* Set the height you want */
        display: flex;
        justify-content: center;
        align-items: center;
        border: 1px solid #ccc; /* Optional: Adds a border for visual distinction */
        padding: 10px; /* Optional: Adds padding */
    }
    </style>
    """,
    unsafe_allow_html=True,
)
if video_file is not None:
    # Create columns to make the layout more compact
    video_col, results_col = st.columns([1, 1])
    
    # Video Display
    with video_col:
        st.subheader("Uploaded Video")
        st.video(video_file)
    
    # Processing and Results
    with results_col:
        st.subheader("Analysis")
        start_time = time.time()

        # Loading animation
        with st.spinner("🚀 Processing video... Please wait!"):
            predicted_class, probabilities, frames = make_prediction(video_file)

        if predicted_class is None:  # No faces detected
            st.error("No faces detected in the uploaded video. Please upload a different video.")
        else:
            end_time = time.time()
            processing_time = end_time - start_time

            # Display results
            if predicted_class == 0:
                st.success("The video is classified as **Real**!")
            else:
                st.error("The video is classified as **Fake**!")

            st.write(f"**Prediction Confidence:**")
            st.progress(int(probabilities[predicted_class] * 100))

    # Detailed Results Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Probabilities", "🖼️ Frame Previews", "⏱️ Processing Time"])

    with tab1:
        st.subheader("Class Probabilities")
        st.bar_chart({"Real": [probabilities[0]], "Fake": [probabilities[1]]})

    with tab2:
        st.subheader("Frame Previews")
        st.write("Key frames analyzed during the process:")
        cols = st.columns(5)
        for i, frame in enumerate(frames[:10]):
            frame = np.clip(frame, 0, 1)
            frame = (frame * 255).astype(np.uint8)
            with cols[i % 5]:
                st.image(frame, caption=f"Frame {i+1}", use_container_width=True)

    with tab3:
        st.subheader("Processing Details")
        st.write(f"**Time Taken:** {processing_time:.2f} seconds")

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import joblib
import av
import os
import time

from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# Import the new Tasks API components
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ============================================================
# Load model
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "gesture detection model.pkl"))
encoder = joblib.load(os.path.join(BASE_DIR, "gesture detection encoder.pkl"))


# ============================================================
# Video Processor
# ============================================================

class GestureProcessor(VideoProcessorBase):

    def __init__(self):
        # Fix file path issues by building an absolute path to the model
        model_path = os.path.join(BASE_DIR, "hand_landmarker.task")
        
        if not os.path.isfile(model_path) or os.path.getsize(model_path) == 0:
            raise FileNotFoundError(
                f"MediaPipe model is missing or empty: {model_path}"
            )

        # Store latest detection results globally within the processor
        self.latest_results = None
        self.gesture = "No hand detected"

        # Callback function required for LIVE_STREAM mode
        def save_result(result, _output_image, timestamp_ms):
            # The Tasks API invokes this callback asynchronously. Keep the
            # complete result from the latest frame so recv() can use the
            # same 21 normalized (x, y) landmarks used by model.ipynb.
            self.latest_results = (result, timestamp_ms)

        # Configure the modern HandLandmarker Options
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.LIVE_STREAM,  # Must be LIVE_STREAM for continuous frame processing
            result_callback=save_result,
            num_hands=1
        )
        
        # Initialize the new detector
        self.detector = vision.HandLandmarker.create_from_options(options)


    def recv(self, frame):
        # WebRTC frame → OpenCV image (BGR)
        image = frame.to_ndarray(format="bgr24")

        # BGR → RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert OpenCV/NumPy RGB image to MediaPipe Image format
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # MediaPipe detection via the Tasks API (requires a timestamp in milliseconds)
        timestamp_ms = int(time.time() * 1000)
        self.detector.detect_async(mp_image, timestamp_ms)

        # Fetch latest async results
        result_data = self.latest_results
        results = result_data[0] if result_data else None

        # ====================================================
        # If hand detected
        # ====================================================
        if results and results.hand_landmarks:
            # results.hand_landmarks is a list of lists containing the 21 landmarks
            hand_landmarks = results.hand_landmarks[0]

            # Extract 21 (x,y) landmarks from the new object structure
            landmarks = np.array([
                [lm.x, lm.y]
                for lm in hand_landmarks
            ], dtype=np.float32)

            # 21 × 2 → 42 features
            features = landmarks.flatten().reshape(1, -1)

            # Random Forest prediction
            prediction = model.predict(features)[0]

            # Convert encoded label → gesture name
            self.gesture = encoder.inverse_transform([prediction])[0]

            # Manual visual anchors: Draw simple custom dots and lines 
            h, w, _ = image.shape
            
            # Map landmarks to pixel space
            points = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks]
            
            # Draw joints (lines) based on standard skeletal structures
            connections = [
                (0,1), (1,2), (2,3), (3,4),       # Thumb
                (0,5), (5,6), (6,7), (7,8),       # Index
                (9,10), (10,11), (11,12),         # Middle
                (13,14), (14,15), (15,16),        # Ring
                (0,17), (17,18), (18,19), (19,20),# Pinky
                (5,9), (9,13), (13,17)            # Palm base
            ]
            
            for start, end in connections:
                if start < len(points) and end < len(points):
                    cv2.line(image, points[start], points[end], (0, 255, 0), 2)
                    
            for pt in points:
                cv2.circle(image, pt, 5, (0, 0, 255), -1)

            # Display prediction on video
            cv2.putText(
                image,
                f"Gesture: {self.gesture}",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                3
            )
        else:
            self.gesture = "No hand detected"
            cv2.putText(
                image,
                "No hand detected",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                2
            )

        # Return processed frame
        return av.VideoFrame.from_ndarray(image, format="bgr24")


# ============================================================
# Streamlit UI
# ============================================================

st.set_page_config(
    page_title="Hand Gesture Recognition",
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 Live Hand Gesture Recognition")

st.write(
    "Show your hand to the camera. "
    "The Random Forest model will recognize your gesture in real time."
)


# ============================================================
# Start webcam
# ============================================================

webrtc_streamer(
    key="gesture-recognition",
    video_processor_factory=GestureProcessor,
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    },
)

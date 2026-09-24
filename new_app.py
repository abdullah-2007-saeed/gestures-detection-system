import streamlit as st
import numpy as np
import joblib
from PIL import Image
import mediapipe as mp

# ============================================================
# 1. Load Trained Models
# ============================================================
@st.cache_resource
def load_models():
    model = joblib.load("gesture detection model.pkl")
    encoder = joblib.load("gesture detection encoder.pkl")
    return model, encoder

model, encoder = load_models()

# Initialize MediaPipe Hand solution cleanly via its basic Python utility
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, 
    max_num_hands=1, 
    min_detection_confidence=0.5
)

# ============================================================
# 2. Streamlit UI Layout
# ============================================================
st.set_page_config(page_title="Hand Gesture Recognition", page_icon="🤟")
st.title("🤟 Hand Gesture Recognition")

camera_img = st.camera_input("Take a snapshot of your hand gesture")

if camera_img:
    # Open the image file using Pillow
    img = Image.open(camera_img)
    
    # Convert Pillow image to a standard RGB numpy array for MediaPipe
    rgb_image = np.array(img.convert('RGB'))
    
    # Process the frame to get landmarks
    results = hands.process(rgb_image)
    
    if results.multi_hand_landmarks:
        # Get coordinates for the first hand found
        hand_landmarks = results.multi_hand_landmarks[0]
        
        # Extract the 21 (x, y) points exactly like your training dataset did
        landmarks = []
        for lm in hand_landmarks.landmark:
            landmarks.extend([lm.x, lm.y])
            
        # Convert list to array and shape as 1 row with 42 columns
        features = np.array(landmarks).reshape(1, -1)
        
        # Run prediction on your 42 features
        prediction = model.predict(features)[0]
        gesture_name = encoder.inverse_transform([prediction])[0]
        
        st.success(f"### Detected Gesture: **{gesture_name}**")
    else:
        st.warning("No hand detected. Please position your hand clearly in front of the camera.")

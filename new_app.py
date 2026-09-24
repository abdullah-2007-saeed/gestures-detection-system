import streamlit as st
import numpy as np
import joblib
from PIL import Image

# Load your existing Random Forest model (Keep your .pkl files!)
model = joblib.load("gesture detection model.pkl")
encoder = joblib.load("gesture detection encoder.pkl")

st.title("🤟 Hand Gesture Recognition")

# Use Streamlit's native camera input
camera_img = st.camera_input("Take a snapshot of your hand gesture")

if camera_img:
    # 1. Open the image using Pillow instead of OpenCV
    img = Image.open(camera_img)
    
    # 2. Convert to grayscale and resize it to match whatever shape you need
    # (For example, if your model was trained on 64x64 pixel images)
    img_resized = img.convert('L').resize((64, 64))
    
    # 3. Convert to a flat numpy array for your Random Forest model
    img_array = np.array(img_resized).flatten().reshape(1, -1)
    
    # 4. Predict
    prediction = model.predict(img_array)[0]
    gesture_name = encoder.inverse_transform([prediction])[0]
    
    st.success(f"### Detected Gesture: **{gesture_name}**")

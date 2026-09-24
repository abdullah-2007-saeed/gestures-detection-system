import streamlit as st
import numpy as np
import joblib
import streamlit.components.v1 as components

# ============================================================
# 1. Page Configuration & Title
# ============================================================
st.set_page_config(
    page_title="Browser Hand Gesture Recognition", 
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 Real-Time Hand Gesture Recognition")
st.write(
    "This app tracks your hand coordinates directly inside your browser. "
    "This bypasses heavy cloud server library conflicts and feeds the "
    "coordinates instantly into your Random Forest model!"
)

# ============================================================
# 2. Load Your Existing Random Forest Model
# ============================================================
@st.cache_resource
def load_models():
    model = joblib.load("gesture detection model.pkl")
    encoder = joblib.load("gesture detection encoder.pkl")
    return model, encoder

try:
    model, encoder = load_models()
except Exception as e:
    st.error(f"Failed to load model or encoder files. Check if they are in your repository: {e}")
    st.stop()

# ============================================================
# 3. Embed Browser-Side Tracking HTML & JavaScript
# ============================================================
html_code = """
<div style="position: relative; display: flex; justify-content: center; align-items: center;">
  <video id="webcam" autoplay playsinline style="width: 100%; max-width: 500px; transform: scaleX(-1); border-radius: 10px; background-color: #222;"></video>
  <canvas id="output_canvas" style="position: absolute; width: 100%; max-width: 500px; transform: scaleX(-1); pointer-events: none;"></canvas>
</div>

<!-- Load MediaPipe dependencies via secure CDNs directly into the user's browser -->
<script src="https://jsdelivr.net" crossorigin="anonymous"></script>
<script src="https://jsdelivr.net" crossorigin="anonymous"></script>

<script>
  const videoElement = document.getElementById('webcam');
  const canvasElement = document.getElementById('output_canvas');
  const canvasCtx = canvasElement.getContext('2d');

  function onResults(results) {
    // Clear canvas for the new frame draw cycle
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      // Pick the first tracked hand
      const hand = results.multiHandLandmarks[0];
      
      // Draw red tracking landmark dots for visual feedback
      for (const point of hand) {
        canvasCtx.beginPath();
        canvasCtx.arc(point.x * canvasElement.width, point.y * canvasElement.height, 4, 0, 2 * Math.PI);
        canvasCtx.fillStyle = '#FF0000';
        canvasCtx.fill();
      }

      // Flatten 21 landmarks into an array of 42 numeric features (x1, y1, x2, y2...)
      let flatFeatures = [];
      hand.forEach(lm => {
        flatFeatures.push(lm.x);
        flatFeatures.push(lm.y);
      });

      // Securely pass the 42 coordinate array back into the Streamlit app state
      window.parent.postMessage({
        type: 'streamlit:setComponentValue',
        value: flatFeatures
      }, '*');
    } else {
      // Tell Python no hands are visible right now
      window.parent.postMessage({
        type: 'streamlit:setComponentValue',
        value: null
      }, '*');
    }
    canvasCtx.restore();
  }

  // Initialize MediaPipe Hands in the client browser
  const hands = new Hands({locateFile: (file) => `https://jsdelivr.net{file}`});
  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
  hands.onResults(onResults);

  // Bind browser native camera hardware feed
  const camera = new Camera(videoElement, {
    onFrame: async () => {
      if (videoElement.videoWidth > 0) {
        canvasElement.width = videoElement.videoWidth;
        canvasElement.height = videoElement.videoHeight;
        await hands.send({image: videoElement});
      }
    },
    width: 640,
    height: 480
  });
  camera.start();
</script>
"""

# Render the frontend tracker inside Streamlit
receiver = components.html(html_code, height=400)

# ============================================================
# 4. Handle Python Machine Learning Predictions
# ============================================================
st.subheader("Prediction Output")

# Ensure receiver has actual landmark data array and is not empty or a basic type string
if receiver is not None and isinstance(receiver, list) and len(receiver) == 42:
    try:
        # Convert incoming browser coordinate array to float32 NumPy array
        features = np.array(receiver, dtype=np.float32).reshape(1, -1)
        
        # Run Random Forest prediction using the 42 extracted points
        prediction = model.predict(features)[0]
        gesture_name = encoder.inverse_transform([prediction])[0]
        
        # Display the result to the user
        st.success(f"### Detected Gesture: **{gesture_name}**")
        
    except Exception as e:
        st.info("Aligning incoming gesture structural maps...")
else:
    # Safe structural fallbacks for when hands drop out of frame view
    st.info("👋 Please allow camera access and position your hand inside the webcam feed area.")

import streamlit as st
import numpy as np
import joblib
import streamlit.components.v1 as components

# ============================================================
# 1. Load Your Existing Random Forest Model
# ============================================================
@st.cache_resource
def load_models():
    model = joblib.load("gesture detection model.pkl")
    encoder = joblib.load("gesture detection encoder.pkl")
    return model, encoder

model, encoder = load_models()

# Set up Streamlit Page
st.set_page_config(page_title="Browser Hand Gesture Recognition", page_icon="🤟")
st.title("🤟 Browser Hand Gesture Recognition")
st.write("Processing hand landmarks directly inside your browser to bypass cloud server constraints.")

# Create a placeholder to communicate with JavaScript
if "js_landmarks" not in st.session_state:
    st.session_state.js_landmarks = None

# ============================================================
# 2. Embed Browser-Side Tracking HTML & JavaScript
# ============================================================
html_code = """
<div style="position: relative;">
  <video id="webcam" autoplay playsinline style="width: 100%; max-width: 500px; transform: scaleX(-1); border-radius: 10px;"></video>
  <canvas id="output_canvas" style="position: absolute; left: 0; top: 0; width: 100%; max-width: 500px; transform: scaleX(-1);"></canvas>
</div>

<!-- Load MediaPipe via CDNs directly into the browser -->
<script src="https://jsdelivr.net" crossorigin="anonymous"></script>
<script src="https://jsdelivr.net" crossorigin="anonymous"></script>

<script>
  const videoElement = document.getElementById('webcam');
  const canvasElement = document.getElementById('output_canvas');
  const canvasCtx = canvasElement.getContext('2d');

  function onResults(results) {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const hand = results.multiHandLandmarks[0];
      
      // Draw minimal visual anchors for user confirmation
      for (const landmarks of results.multiHandLandmarks) {
        for (const point of landmarks) {
          canvasCtx.beginPath();
          canvasCtx.arc(point.x * canvasElement.width, point.y * canvasElement.height, 5, 0, 2 * Math.PI);
          canvasCtx.fillStyle = '#FF0000';
          canvasCtx.fill();
        }
      }

      // Flatten 21 landmarks into an array of 42 numeric features (x1, y1, x2, y2...)
      let flatFeatures = [];
      hand.forEach(lm => {
        flatFeatures.push(lm.x);
        flatFeatures.push(lm.y);
      });

      // Send the 42 features directly up to Streamlit Python
      window.parent.postMessage({
        type: 'streamlit:setComponentValue',
        value: flatFeatures
      }, '*');
    } else {
      window.parent.postMessage({
        type: 'streamlit:setComponentValue',
        value: null
      }, '*');
    }
    canvasCtx.restore();
  }

  const hands = new Hands({locateFile: (file) => `https://jsdelivr.net{file}`});
  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5,
    minTrackingConfidence: 0.5
  });
  hands.onResults(onResults);

  const camera = new Camera(videoElement, {
    onFrame: async () => {
      canvasElement.width = videoElement.videoWidth;
      canvasElement.height = videoElement.videoHeight;
      await hands.send({image: videoElement});
    },
    width: 640,
    height: 480
  });
  camera.start();
</script>
"""

# Render the frontend tracker inside the app
receiver = components.html(html_code, height=380)

# ============================================================
# 3. Handle Python Machine Learning Predictions
# ============================================================
if receiver is not None:
    # receiver contains the flat array of 42 landmarks sent from JS
    features = np.array(receiver).reshape(1, -1)
    
    if features.shape[1] == 42:
        prediction = model.predict(features)[0]
        gesture_name = encoder.inverse_transform([prediction])[0]
        st.success(f"## Predicted Gesture: **{gesture_name}**")
else:
    st.info("Please allow camera access. Place your hand clearly in front of the browser feed.")

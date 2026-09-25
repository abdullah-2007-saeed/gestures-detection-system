import streamlit as st
import numpy as np
import joblib
import streamlit.components.v1 as components

# ============================================================
# 1. Page Layout Configuration
# ============================================================
st.set_page_config(
    page_title="Pure Web Hand Recognition", 
    page_icon="🤟",
    layout="centered"
)

st.title("🤟 Pure Web Hand Gesture Recognition")
st.write("Upload an image below. Hand landmark extraction runs securely inside your browser—no OpenCV or MediaPipe is installed on this server!")

# ============================================================
# 2. Load Your Existing Random Forest Model Safely
# ============================================================
@st.cache_resource
def load_models():
    model = joblib.load("gesture detection model.pkl")
    encoder = joblib.load("gesture detection encoder.pkl")
    return model, encoder

try:
    model, encoder = load_models()
except Exception as e:
    st.error(f"Failed to find required model or encoder files: {e}")
    st.stop()

# ============================================================
# 3. HTML & JavaScript Browser Tracker Component
# ============================================================
html_code = """
<div style="font-family: sans-serif; display: flex; flex-direction: column; align-items: center; gap: 15px;">
  <!-- Native Browser File Input Button -->
  <input type="file" id="image_loader" accept="image/*" style="padding: 10px; border: 1px dashed #444; border-radius: 5px; width: 100%; max-width: 400px; background: #111; color: #fff; cursor: pointer;">
  
  <div style="position: relative; display: flex; justify-content: center; align-items: center; margin-top: 10px;">
    <img id="source_image" style="display: none; width: 100%; max-width: 500px; border-radius: 10px;" />
    <canvas id="output_canvas" style="position: absolute; left: 0; top: 0; width: 100%; max-width: 500px; pointer-events: none;"></canvas>
  </div>
</div>

<!-- Load MediaPipe framework entirely inside the client browser tab -->
<script src="https://jsdelivr.net" crossorigin="anonymous"></script>

<script>
  const fileInput = document.getElementById('image_loader');
  const imageElement = document.getElementById('source_image');
  const canvasElement = document.getElementById('output_canvas');
  const canvasCtx = canvasElement.getContext('2d');

  // Initialize MediaPipe Hands in the client browser
  const hands = new Hands({locateFile: (file) => `https://jsdelivr.net{file}`});
  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.5
  });
  
  hands.onResults((results) => {
    canvasCtx.save();
    canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
    
    if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
      const hand = results.multiHandLandmarks[0];
      
      // Draw simple landmark dots on the browser canvas for visual feedback
      for (const point of hand) {
        canvasCtx.beginPath();
        canvasCtx.arc(point.x * canvasElement.width, point.y * canvasElement.height, 5, 0, 2 * Math.PI);
        canvasCtx.fillStyle = '#00FF00';
        canvasCtx.fill();
      }

      // Convert the landmarks to a flat array of 42 numeric coordinates
      let flatArray = [];
      hand.forEach(lm => {
        flatArray.push(lm.x);
        flatArray.push(lm.y);
      });

      // Pass the 42 raw numbers safely to Streamlit
      window.parent.postMessage({
        type: 'streamlit:setComponentValue',
        value: flatArray
      }, '*');
    } else {
      // Send null if no hand was found in the photo
      window.parent.postMessage({
        type: 'streamlit:setComponentValue',
        value: "no_hand"
      }, '*');
    }
    canvasCtx.restore();
  });

  // Handle local file uploads inside the browser window
  fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      imageElement.src = event.target.result;
      imageElement.style.display = 'block';
      
      imageElement.onload = async () => {
        canvasElement.width = imageElement.clientWidth;
        canvasElement.height = imageElement.clientHeight;
        canvasElement.style.width = imageElement.clientWidth + 'px';
        canvasElement.style.height = imageElement.clientHeight + 'px';
        
        // Feed image data to browser-side MediaPipe
        await hands.send({image: imageElement});
      };
    };
    reader.readAsDataURL(file);
  });
</script>
"""

# Render the frontend container frame widget
receiver = components.html(html_code, height=450)

# ============================================================
# 4. Handle Python Machine Learning Predictions
# ============================================================
st.subheader("Classification Outcome")

if receiver is not None:
    if isinstance(receiver, list) and len(receiver) == 42:
        try:
            # Convert raw coordinate numbers directly into a NumPy input row
            features = np.array(receiver, dtype=np.float32).reshape(1, -1)
            
            # Run Random Forest prediction using the 42 coordinate features
            prediction = model.predict(features)[0]
            gesture_name = encoder.inverse_transform([prediction])[0]
            
            st.success(f"## Classified Gesture: **{gesture_name}**")
        except Exception as e:
            st.info("Reading incoming structural array matrix...")
    elif receiver == "no_hand":
        st.warning("⚠️ Could not trace hand outline landmarks in this photo. Ensure your hand is clearly visible.")
else:
    st.info("📁 Please click the upload button above to choose a gesture image file.")

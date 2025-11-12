import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model once (cached)
@st.cache_resource
def load_model():
    return ResNet50(weights='imagenet')

model = load_model()

# Initialize session state
if "show_results" not in st.session_state:
    st.session_state.show_results = False

st.title("🖼️ What Am I Looking At?")
st.write("Upload an image and I’ll tell you what’s in it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# --- Show image and results ---
if uploaded_file is not None and not st.session_state.show_results:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Classify Image"):
        # Preprocess the image
        img = img.resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        predictions = decode_predictions(preds, top=5)[0]

        st.session_state.predictions = predictions
        st.session_state.image = img
        st.session_state.show_results = True
        st.rerun()

# --- Show classification results ---
if st.session_state.show_results:
    st.image(st.session_state.image, caption="Uploaded Image", use_column_width=True)
    st.subheader("✅ Top 5 Predictions:")

    for i, (imagenet_id, label, score) in enumerate(st.session_state.predictions):
        st.write(f"{i+1}. **{label}** — {round(score * 100, 2)}%")

    if st.button("🔁 Try another image"):
        # Proper reset
        st.session_state.clear()
        st.rerun()

import streamlit as st
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# Load model once
@st.cache_resource
def load_model():
    model = ResNet50(weights='imagenet')
    return model

model = load_model()

# App title and instructions
st.title("🖼️ What Am I Looking At?")
st.write("Upload an image and I’ll tell you what’s in it!")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display image
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=True)

    # Predict button
    if st.button("🔍 Classify Image"):
        # Preprocess the image
        img = img.resize((224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        preds = model.predict(x)
        predictions = decode_predictions(preds, top=5)[0]

        st.subheader("✅ Top 5 Predictions:")
        for i, (imagenet_id, label, score) in enumerate(predictions):
            st.write(f"{i+1}. **{label}** — {round(score * 100, 2)}%")

    # Try another image button (Reset)
    if st.button("🔁 Try another image"):
        st.session_state.clear()
        st.rerun()

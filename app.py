
import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

st.set_page_config(page_title="Image Classifier", page_icon="🔍", layout="centered")

# ---- Title & Instructions ----
st.title("🖼️ What Am I Looking At?")
st.write("Upload an image and I'll tell you what’s in it!")

# ---- Load Pretrained Model ----
@st.cache_resource
def load_model():
    model = tf.keras.applications.ResNet50(weights="imagenet")
    return model

model = load_model()

# ---- Prediction Function ----
def predict_image(image):
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    preds = model.predict(img_array)
    decoded_preds = tf.keras.applications.resnet50.decode_predictions(preds, top=5)[0]
    return decoded_preds

# ---- File Upload ----
uploaded_file = st.file_uploader("📁 Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image Preview", use_column_width=True)
    st.write("🔍 Classifying... please wait...")
    
    predictions = predict_image(image)
    
    st.subheader("Top 5 Predictions:")
    for (_, label, prob) in predictions:
        st.write(f"**{label}** — {prob*100:.2f}%")

   if st.button("🔁 Try another image"):
    st.session_state.clear()
    st.rerun()
       
else:
    st.info("Please upload an image to begin.")

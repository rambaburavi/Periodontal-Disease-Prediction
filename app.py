import streamlit as st
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from PIL import Image as PILImage

# === Configuration ===
model_path = r'C:\Users\Rambabu\OneDrive\Desktop\periodontal\periodontal_model.h5'
img_height, img_width = 128, 128

# === Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(model_path)
    return model

model = load_model()

# === Streamlit App UI
st.set_page_config(page_title="Periodontal Disease Classifier", layout="centered")
st.title(" Periodontal Disease Detection")
st.markdown("Upload a **dental panoramic X-ray** to detect signs of periodontal disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Save image to disk temporarily
    temp_file_path = "temp_uploaded_image.jpg"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Show image
    st.image(PILImage.open(temp_file_path), caption="Uploaded Image", use_container_width=True)

    # Preprocess and predict (no validation)
    img = image.load_img(temp_file_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)

    prediction = model.predict(img_array)[0][0]
    confidence = float(prediction)

    # Show prediction and confidence
    if confidence >= 0.5:
        st.success(f"🧠 Prediction: **Periodontal Disease** ({confidence * 100:.2f}% confidence)")
        st.progress(int(confidence * 100))
    else:
        st.success(f"🦷 Prediction: **Non Periodontal Disease** ({(1 - confidence) * 100:.2f}% confidence)")
        st.progress(int((1 - confidence) * 100))

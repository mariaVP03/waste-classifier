import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Constants
MODEL_PATH = 'waste_model.h5'
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

@st.cache(allow_output_mutation=True)
def load_model_cached():
    return load_model(MODEL_PATH)

def preprocess_image(image):
    image = image.resize((224, 224))  # MobileNetV2 input size
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict(image):
    model = load_model_cached()
    processed = preprocess_image(image)
    predictions = model.predict(processed)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    return CLASS_NAMES[class_idx], confidence

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Waste Prediction", "About Us"])

# Page 1: Introduction
if page == "Introduction":
    st.title("♻️ Waste Classification")
    st.markdown("""
    Welcome! This app uses a deep learning model based on **MobileNetV2** to identify the category of waste from an uploaded image.
    
    The possible categories are:
    - **Cardboard**
    - **Glass**
    - **Metal**
    - **Paper**
    - **Plastic**
    - **Trash**

    👉 Go to **Waste Prediction** in the sidebar to get started.
    """)

# Page 2: Waste Prediction
elif page == "Waste Prediction":
    st.title("Upload Your Waste Image")
    st.write("Upload a photo of a waste item. The model will predict its category.")

    uploaded = st.file_uploader("Upload an image", type=['jpg', 'png'])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            label, confidence = predict(img)
            st.success(f"Prediction: **{label}**")
            st.write(f"Confidence: **{confidence:.2f}**")

# Page 3: About Us
elif page == "About Us":
    st.title("👋Meet the Team")
    st.markdown("""
    This project was developed by:

    - **Uxia Lojo Miranda**
    - **Joy Zohn**
    - **Joel James Alarde**
    - **Maria Vazquez Pinedo**
    - **Sarina Ratnabhas**

    We're passionate about using AI to build meaningful, sustainable solutions. 🌍
    """)

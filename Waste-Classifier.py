import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model

# Constants
MODEL_PATH = 'waste_model.h5'
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
SAMPLE_IMAGE_PATH = 'sample_image.jpg'
TEAM_IMAGE_PATH = 'team_photo.jpg'

@st.cache(allow_output_mutation=True)
def load_model_cached():
    return load_model(MODEL_PATH)

def preprocess_image(image):
    image = image.resize((224, 224))  # MobileNetV2 default input size
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict(image):
    model = load_model_cached()
    processed = preprocess_image(image)
    predictions = model.predict(processed)
    class_idx = np.argmax(predictions[0])
    return CLASS_NAMES[class_idx], predictions[0][class_idx]

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Introduction", "Waste Prediction", "About Us"])

# Page 1: Introduction
if page == "Introduction":
    st.title("♻️ Waste Classification")
    st.markdown("""
    Welcome to our waste classification tool!  
    This application uses a **MobileNetV2** deep learning model fine-tuned on a dataset of garbage categories:
    
    - **Cardboard**
    - **Glass**
    - **Metal**
    - **Paper**
    - **Plastic**
    - **Trash**
    
    
    Use the **sidebar** to navigate through the app.
    """)

# Page 2: Waste Prediction
elif page == "Waste Prediction":
    st.title("Upload Your Image")
    st.write("Upload an image of waste to predict its category.")

    uploaded = st.file_uploader("Choose an image", type=['jpg', 'png'])

    if uploaded:
        img = Image.open(uploaded)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            label, confidence = predict(img)
            st.success(f"Prediction: **{label}** with confidence **{confidence:.2f}**")

    st.markdown("Or try with a sample image:")
    if st.button("Predict Sample Image"):
        sample = Image.open(SAMPLE_IMAGE_PATH)
        st.image(sample, caption="Sample Image", use_column_width=True)
        label, confidence = predict(sample)
        st.success(f"Prediction: **{label}** with confidence **{confidence:.2f}**")

# Page 3: About Us
elif page == "About Us":
    st.title("Meet the Team 👋")
    st.markdown("""
    This project was built with love by:
    
    - **Uxia Logo**
    - **Joy Zohn**
    - **James Alarde**
    - **Maria Vazquez**
    
    We are passionate about AI, sustainability, and using technology for good.
    """)
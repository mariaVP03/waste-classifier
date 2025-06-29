import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import model_from_json

# Constants
CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

@st.cache(allow_output_mutation=True)
def load_model_cached():
    with open("network.json", "r") as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights("weights.hdf5")
    return model

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(image)
    img_array = tf.keras.applications.resnet50.preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def predict(image):
    model = load_model_cached()
    processed = preprocess_image(image)
    predictions = model.predict(processed)
    class_idx = np.argmax(predictions[0])
    confidence = predictions[0][class_idx]
    return CLASS_NAMES[class_idx], confidence

def main():
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Introduction", "Waste Prediction", "About Us"])

    # Page 1: Introduction
    if page == "Introduction":
        st.title("♻️ Waste Classification")

        st.subheader("📊Business Case: The Problem")
        st.markdown("""
        By **2030**, the world is expected to generate over **2.6 billion tons of waste** every year.  
        Current waste management systems are inefficient and suffer from:

        - **Manual human sorting** that handles only 1–2 items per second
        - **High processing costs** of **$50–$100 per ton**
        - **Contamination and classification error rates** between **15–25%**

        These inefficiencies lead to:
        - ⚠️ **High contamination rates**
        - 💸 **Increased processing costs**
        - ♻️ **Reduced recycling efficiency**
        """)

        st.subheader("💡What Our Project Does")
        st.markdown("""
        To address these challenges, our team built an **AI-powered waste classifier** based on **ResNet50**.

        The app uses computer vision to classify household or industrial waste images into six key categories:

        - **Cardboard**
        - **Glass**
        - **Metal**
        - **Paper**
        - **Plastic**
        - **Trash**

        This makes sorting faster, cheaper, and more reliable.
        """)

        st.subheader("✅ How It Solves the Problem")
        st.markdown("""
        - **Automates classification**, eliminating the need for slow human sorting
        - **Improves accuracy**, reducing contamination and error rates
        - **Cuts costs** by optimizing the processing chain
        - **Boosts recycling efficiency**, contributing to sustainability goals

        In short, our solution leverages AI to make waste management **smarter, greener, and more scalable**.
        """)

    # Page 2: Waste Prediction
    elif page == "Waste Prediction":
        st.title("Waste Classification")
        st.write("Upload a photo or take one with your webcam. The model will predict the category and give you eco information.")

        input_method = st.radio("Select input method:", ["📁 Upload from device", "📷 Take a photo"])

        img = None

        if input_method == "📁 Upload from device":
            uploaded = st.file_uploader("Upload an image", type=['jpg', 'png'])
            if uploaded:
                img = Image.open(uploaded)
                st.image(img, caption="Uploaded Image", use_column_width=True)

        elif input_method == "📷 Take a photo":
            cam = st.camera_input("Take a photo")
            if cam:
                img = Image.open(cam)
                st.image(img, caption="Captured Photo", use_column_width=True)

        if img and st.button("🔍 Predict Waste Category"):
            label, confidence = predict(img)
            st.success(f"Prediction: **{label.upper()}**")
            st.write(f"Confidence: **{confidence:.2f}**")

            waste_info = {
                "cardboard": {
                    "impact": "Cardboard has a mixed environmental impact...",
                    "bin": "🟦Blue bin",
                    "link": "https://www.gwp.co.uk/guides/how-is-cardboard-recycled/"
                },
                "glass": {
                    "impact": "Glass production and disposal have both positive and negative environmental impacts...",
                    "bin": "🟩Green bin",
                    "link": "https://www.recyclenow.com/how-to-recycle/glass-recycling"
                },
                "metal": {
                    "impact": "Metal production and use have significant and widespread impacts...",
                    "bin": "🟨Yellow bin",
                    "link": "https://www.anis-trend.com/recycling-metals-5-simple-steps/"
                },
                "paper": {
                    "impact": "The environmental impact of paper production and use is multifaceted...",
                    "bin": "🟦Blue bin",
                    "link": "https://www.recyclenow.com/how-to-recycle/paper-recycling"
                },
                "plastic": {
                    "impact": "Plastic pollution has far-reaching and detrimental effects...",
                    "bin": "🟨Yellow bin",
                    "link": "https://www.recyclenow.com/how-to-recycle/plastic-recycling"
                },
                "trash": {
                    "impact": "Non-recyclable trash significantly harms the environment...",
                    "bin": "🟫Brown bin",
                    "link": "https://northlondonheatandpower.london/alternative-ways-treat-non-recyclable-waste"
                },
            }

            info = waste_info.get(label.lower())

            if info:
                st.markdown(f"### ♻️ Environmental Info for: {label.title()}")
                st.write(f"🌍 **Impact:** {info['impact']}")
                st.write(f"🗑️ **Recycle in:** {info['bin']}")
                st.markdown(f"[🔗 Learn more about {label.lower()} recycling]({info['link']})")

    # Page 3: About Us
    elif page == "About Us":
        st.title("👋 Meet the Team")

        st.markdown("### Uxia Lojo")
        st.image("Uxia.jpeg", width=180)

        st.markdown("### Joy Zhong")
        st.image("Joy.jpeg", width=180)

        st.markdown("### James Alarde")
        st.image("James.jpeg", width=180)

        st.markdown("### Maria Vazquez")
        st.image("Maria.jpeg", width=180)

        st.markdown("### Sarina Ratnabhas")
        st.image("Sarina.jpeg", width=180)

        st.markdown("""
        We are a team passionate about using AI to build meaningful, sustainable solutions that tackle real-world problems like waste management. 🌍
        """)

if __name__ == "__main__":
    main()




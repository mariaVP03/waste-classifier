import streamlit as st
from PIL import Image
import tensorflow as tf
import numpy as np

CLASS_NAMES = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

@st.cache(allow_output_mutation=True)
def load_model_cached():
    model = tf.keras.models.load_model("weights.hdf5")
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

# Página 1: Introducción
def page_intro():
    st.title("♻ Waste Classification")
    st.subheader("📊 Business Case: The Problem")
    st.markdown("""
    By 2030, the world is expected to generate over *2.6 billion tons of waste* per year.  
    Current waste management systems are inefficient:
    - Manual sorting is slow
    - Processing costs are high ($50–$100/ton)
    - Error rates reach 15–25%

    Result: contamination, high costs, and poor recycling efficiency.
    """)

    st.subheader("💡 What Our Project Does")
    st.markdown("""
    Our AI app classifies waste into:
    - Cardboard
    - Glass
    - Metal
    - Paper
    - Plastic
    - Trash

    Using ResNet50, we make sorting faster, cheaper, and more accurate.
    """)

    st.subheader("✅ How It Helps")
    st.markdown("""
    - Automates classification
    - Improves accuracy
    - Reduces contamination
    - Boosts recycling and sustainability
    """)

# Página 2: Clasificación
def page_predict():
    st.title("🧪 Waste Prediction")
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
        st.success(f"Prediction: {label.upper()}")
        st.write(f"Confidence: {confidence:.2f}")

        # Info ambiental
        waste_info = {
            "cardboard": {
                "impact": "Cardboard is recyclable, but production pollutes water and emits methane in landfills.",
                "bin": "🟦 Blue bin",
                "link": "https://www.gwp.co.uk/guides/how-is-cardboard-recycled/"
            },
            "glass": {
                "impact": "Glass is 100% recyclable, but heavy and energy-intensive to produce.",
                "bin": "🟩 Green bin",
                "link": "https://www.recyclenow.com/how-to-recycle/glass-recycling"
            },
            "metal": {
                "impact": "Metal production causes habitat loss, pollution, and CO₂ emissions.",
                "bin": "🟨 Yellow bin",
                "link": "https://www.anis-trend.com/recycling-metals-5-simple-steps/"
            },
            "paper": {
                "impact": "Paper production leads to deforestation, water/air pollution and methane from waste.",
                "bin": "🟦 Blue bin",
                "link": "https://www.recyclenow.com/how-to-recycle/paper-recycling"
            },
            "plastic": {
                "impact": "Plastic pollution affects oceans, wildlife and releases toxic chemicals.",
                "bin": "🟨 Yellow bin",
                "link": "https://www.recyclenow.com/how-to-recycle/plastic-recycling"
            },
            "trash": {
                "impact": "Non-recyclable waste releases gases, leaches toxins, and harms biodiversity.",
                "bin": "🟫 Brown bin",
                "link": "https://northlondonheatandpower.london/alternative-ways-treat-non-recyclable-waste"
            }
        }

        info = waste_info.get(label.lower())
        if info:
            st.markdown(f"### ♻ Environmental Info for: {label.title()}")
            st.write(f"🌍 Impact: {info['impact']}")
            st.write(f"🗑 Recycle in: {info['bin']}")
            st.markdown(f"[🔗 Learn more about {label.lower()} recycling]({info['link']})")

# Página 3: About Us
def page_about():
    st.title("👋 Meet the Team")

    for name, image in [
        ("Uxia Lojo", "Uxia.jpeg"),
        ("Joy Zhong", "Joy.jpeg"),
        ("James Alarde", "James.jpeg"),
        ("Maria Vazquez", "Maria.jpeg"),
        ("Sarina Ratnabhas", "Sarina.jpeg")
    ]:
        st.markdown(f"### {name}")
        st.image(image, width=180)

    st.markdown("We’re passionate about using AI for sustainability and solving real-world challenges 🌍.")

# Sidebar navigation
def render_sidebar():
    st.sidebar.title("Navigation")
    return st.sidebar.radio("Go to", ["Introduction", "Waste Prediction", "About Us"])

# Main routing
if _name_ == "_main_":
    current_page = render_sidebar()

    if current_page == "Introduction":
        page_intro()
    elif current_page == "Waste Prediction":
        page_predict()
    elif current_page == "About Us":
        page_about()
    else:
        page_intro()




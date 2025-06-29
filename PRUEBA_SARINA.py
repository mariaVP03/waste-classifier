import streamlit as st
from tensorflow.keras.models import model_from_json
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

# ----------------------
# Cached model loader
# ----------------------
@st.cache_resource
def load_model_cached():
    with open('network.json', 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights('weights.hdf5')
    return model

model = load_model_cached()

# Class labels for prediction
class_labels = ['metal', 'glass', 'paper', 'trash', 'cardboard', 'plastic']

# ----------------------
# Image preprocessing for ResNet50
# ----------------------
def prepare_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

# ----------------------
# Main page with navigation
# ----------------------
def main():
    st.title("Waste Classification App")
    menu = ["Home", "Upload and Predict", "About"]
    choice = st.sidebar.selectbox("Navigation", menu)

    if choice == "Home":
        st.header("Welcome to the Waste Classification App")
        st.write("Upload an image of waste, and the model will predict its category to support recycling and waste management initiatives.")

    elif choice == "Upload and Predict":
        st.header("Upload an Image for Prediction")
        uploaded_file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])

        if uploaded_file is not None:
            try:
                img = Image.open(uploaded_file).convert('RGB')
                st.image(img, caption='Uploaded Image', use_column_width=True)
                st.write("Predicting...")

                processed_image = prepare_image(img)
                predictions = model.predict(processed_image)
                predicted_class = np.argmax(predictions, axis=1)[0]
                confidence = np.max(predictions)

                st.success(f"Predicted Class: {class_labels[predicted_class]}")
                st.info(f"Confidence: {confidence * 100:.2f}%")

                confidences = predictions[0]
                for i, label in enumerate(class_labels):
                    st.write(f"{label}: {confidences[i] * 100:.2f}%")

            except Exception as e:
                st.error(f"Error processing image: {e}")

    elif choice == "About":
        st.header("About This App")
        st.write("This waste classification app was developed using a ResNet50 model with transfer learning and fine-tuning to help sort waste efficiently for sustainability efforts.")

if _name_ == '_main_':
    main()
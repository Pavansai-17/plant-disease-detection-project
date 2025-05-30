import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the model
model = load_model("PDDS.keras")

st.title("🌿 Plant Disease Detection System")

uploaded_file = st.file_uploader("Upload an image of a plant leaf", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image for prediction
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)

    st.write("Prediction:", prediction)

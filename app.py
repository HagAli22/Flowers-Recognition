import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# -------------------------------
# Load model
# -------------------------------
model = tf.keras.models.load_model("flowers.keras")

# نفس ترتيب الكلاسات اللي اتعلمها ImageDataGenerator
class_names = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌸 Flower Classification App")
st.write("Upload a flower image and the model will predict its class.")

# Upload image
uploaded_file = st.file_uploader("Choose a flower image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess
    img_resized = image.resize((128, 128))
    img_array = np.array(img_resized) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)  # add batch dimension

    # Prediction
    predictions = model.predict(img_array)
    pred_class = np.argmax(predictions[0])
    confidence = predictions[0][pred_class]

    # Show results
    st.subheader(f"Prediction: **{class_names[pred_class]}** 🌼")
    st.write(f"Confidence: {confidence:.2f}")

    # Plot probabilities
    st.bar_chart({class_names[i]: predictions[0][i] for i in range(len(class_names))})

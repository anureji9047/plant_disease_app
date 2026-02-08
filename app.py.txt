import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# load model
model = tf.keras.models.load_model("plant_disease_model1.h5")

classes = ["Healthy", "Early Blight", "Late Blight"]

st.title("Tomato Leaf Disease Detection")

uploaded_file = st.file_uploader("Upload leaf image", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((128,128))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    result = classes[np.argmax(prediction)]

    st.success("Prediction: " + result)

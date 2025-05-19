import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource(allow_output_mutation=True)
def load_model():
    model = tf.keras.models.load_model('cheating_gru_model(0.88ac, 0.97 val).keras') # Assuming a .h5 model file
    return model

def preprocess_image(image_data, model_input_size=(64, 64)): # Adjust size as needed
    image = Image.open(image_data).convert('RGB') # Ensure RGB
    image = ImageOps.fit(image, model_input_size, Image.Resampling.LANCZOS)
    img_array = np.asarray(image)
    img_array = img_array / 255.0 # Normalize pixel values (common practice)
    img_reshape = np.expand_dims(img_array, axis=0)
    return img_reshape

def predict(processed_image, model):
    prediction = model.predict(processed_image)
    return prediction

st.title("Cheating vs. Normal Detection System")

file = st.file_uploader("Upload an image for detection...", type=["jpg", "png", "jpeg"])

if file is not None:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    model = load_model()
    processed_image = preprocess_image(file)
    prediction = predict(processed_image)

    class_names = ['Normal', 'Cheating'] # Order should match your model's output
    predicted_class_index = np.argmax(prediction)
    confidence = prediction[0][predicted_class_index]

    st.subheader("Prediction:")
    st.write(f"Predicted Class: **{class_names[predicted_class_index]}**")
    st.write(f"Confidence: **{confidence:.4f}**")

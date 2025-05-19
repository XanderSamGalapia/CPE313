import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load your trained GRU model once and cache it
@st.cache_resource
def load_gru_model():
    model = load_model('/Users/xantinegalapia/Documents/fpfp/ffinalproj/cheating_gru_model(0.88ac, 0.97 val).keras')  # Put your model filename here
    return model

model = load_gru_model()

st.title("GRU Model Prediction")

# Input text box for user to enter data
input_data = st.text_area("Enter your input sequence as comma-separated numbers:")

if st.button("Predict"):
    try:
        # Convert string input into a numpy array
        data = np.array([float(x) for x in input_data.split(",")])
        
        # Reshape input to (1, sequence_length, features)
        # Here assuming 1 feature per timestep; adjust if yours differs
        data = data.reshape(1, -1, 1)
        
        # Run prediction
        prediction = model.predict(data)
        
        st.write("Prediction:", prediction)
    except Exception as e:
        st.error(f"Error: {e}")

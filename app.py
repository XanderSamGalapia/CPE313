import streamlit as st
from tensorflow.keras.models import load_model
import numpy as np

# Load your trained GRU model once and cache it
@st.cache_resource
def load_gru_model():
    model = load_model('cheating_gru_model(0.88ac, 0.97 val).keras')  # Put your model filename here
    return model

model = load_gru_model()

st.title("GRU Model Prediction")

# Input text box for user to enter data
input_data = st.text_area("Enter your input sequence as comma-separated numbers:")

if st.button("Predict"):
    try:
        # Convert string input into a numpy array
        data = np.array([float(x) for x in input_data.split(",")])
        
        # Check if the input data has the correct shape.  This is crucial.
        #  The GRU model expects input of shape (batch_size, sequence_length, num_features).
        #  You MUST determine the correct sequence_length and num_features that your
        #  model was trained on.  I'm making a guess here, and this is the most likely
        #  place for an error if the reshaping is wrong.
        sequence_length = 30  #  <-- REPLACE WITH THE ACTUAL SEQUENCE LENGTH YOUR MODEL EXPECTS
        num_features = 1      #  <-- REPLACE WITH THE ACTUAL NUMBER OF FEATURES PER TIMESTEP
        
        if data.size != sequence_length * num_features:
            st.error(f"Error: Input data must contain {sequence_length * num_features} numbers.")
            st.stop()
            
        # Reshape input to (1, sequence_length, num_features)
        data = data.reshape(1, sequence_length, num_features)
        
        # Run prediction
        prediction = model.predict(data)
        
        st.write("Prediction:", prediction)
    except ValueError:
        st.error("Error: Please enter numbers separated by commas.")
    except Exception as e:
        st.error(f"Error: {e}")

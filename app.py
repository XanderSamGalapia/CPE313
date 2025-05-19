import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO  # Import YOLO
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the YOLO model for pose estimation
@st.cache_resource
def load_yolo_model(weights_path="weights/best.pt"):
    """Loads the YOLO model for pose estimation."""
    try:
        yolo_model = YOLO(weights_path)
        return yolo_model
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return None

# Load the GRU model for action/cheating detection
@st.cache_resource
def load_gru_model(model_path='cheating_gru_model(0.88ac, 0.97 val).keras'):
    """Loads the GRU model."""
    try:
        gru_model = load_model(model_path)
        return gru_model
    except Exception as e:
        st.error(f"Error loading GRU model: {e}")
        return None

yolo_model = load_yolo_model()
gru_model = load_gru_model()

st.title("Pose Estimation and Action Detection")
st.subheader("Upload a video or image for analysis")

input_type = st.radio("Select Input Type", ["Image", "Video"])

if input_type == "Image":
    uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption="Uploaded Image", use_column_width=True)

        if st.button("Analyze Image"):
            if yolo_model is not None and gru_model is not None:
                try:
                    # 1. Pose Estimation with YOLO
                    yolo_results = yolo_model.predict(image_rgb)
                    keypoints_list = []

                    for result in yolo_results:
                        keypoints = result.keypoints.data.cpu().numpy()
                        if len(keypoints) > 0:
                            # Use the first person's keypoints
                            person_keypoints = keypoints[0]
                            keypoints_list.append(person_keypoints[:, :2].flatten())  # Store x, y
                        else:
                            keypoints_list.append(np.zeros(17 * 2))  # 17 keypoints, 2 coords each

                    if not keypoints_list:
                         st.warning("No poses detected in the image.")
                    else:
                        # 2. Prepare Data for GRU (Reshape)
                        sequence_length = 30  # Example sequence length, adjust as needed
                        num_features = 34  # 17 keypoints * 2 coordinates
                        
                        # Pad or truncate keypoints_list to the desired sequence length
                        if len(keypoints_list) < sequence_length:
                            padding = np.zeros((sequence_length - len(keypoints_list), num_features))
                            keypoints_sequence = np.concatenate((keypoints_list, padding), axis=0)
                        elif len(keypoints_list) > sequence_length:
                            keypoints_sequence = keypoints_list[:sequence_length]
                        else:
                            keypoints_sequence = np.array(keypoints_list)
                            
                        # Reshape for GRU (1, sequence_length, num_features)
                        gru_input = keypoints_sequence.reshape(1, sequence_length, num_features)
                        
                        # 3. GRU Prediction
                        gru_prediction = gru_model.predict(gru_input)
                        st.write("GRU Prediction:", gru_prediction)  # Output raw prediction

                        #  interpret the prediction (example - adjust based on your model)
                        predicted_class = np.argmax(gru_prediction)
                        if predicted_class == 0:
                            st.success("Action: Normal")
                        else:
                            st.error("Action: Cheating/Suspicious")
                        
                        # Visualize Pose (optional)
                        for result in yolo_results:
                            kps = result.keypoints.data.cpu().numpy()
                            if len(kps) > 0:
                                for i in range(kps.shape[1]):
                                    x, y, confidence = kps[0, i] #use first person
                                    if confidence > 0.5:
                                        cv2.circle(image, (int(x), int(y)), 5, (0, 255, 0), -1)
                                        cv2.putText(image, str(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
                        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption="Pose Estimation", use_column_width=True)


                except Exception as e:
                    st.error(f"Error during processing: {e}")
            else:
                st.error("YOLO and/or GRU model not loaded.")

elif input_type == "Video":
    uploaded_video = st.file_uploader("Upload a video...", type=["mp4", "mov", "avi"])
    if uploaded_video is not None:
        video_bytes = uploaded_video.read()
        st.video(video_bytes)

        if st.button("Analyze Video"):
            if yolo_model is not None and gru_model is not None:
                try:
                    # Save the video to a temporary file
                    with open("temp_video.mp4", "wb") as temp_file:
                        temp_file.write(video_bytes)
                    video_path = "temp_video.mp4"

                    # 1. Video Capture and Keypoint Extraction
                    cap = cv2.VideoCapture(video_path)
                    keypoints_list = []
                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        yolo_results = yolo_model.predict(frame_rgb)
                        frame_keypoints = []

                        for result in yolo_results:
                            keypoints = result.keypoints.data.cpu().numpy()
                            if len(keypoints) > 0:
                                # Use the first person's keypoints
                                person_keypoints = keypoints[0]
                                frame_keypoints.append(person_keypoints[:, :2].flatten())
                            else:
                                frame_keypoints.append(np.zeros(17 * 2))
                        keypoints_list.append(frame_keypoints)
                    cap.release()

                    # 2. Prepare Data for GRU
                    sequence_length = 30  # Example, adjust as needed
                    num_features = 34
                    
                    # Pad or truncate
                    if len(keypoints_list) < sequence_length:
                        padding = np.zeros((sequence_length - len(keypoints_list), num_features))
                        keypoints_sequence = np.concatenate((keypoints_list, padding), axis=0)
                    elif len(keypoints_list) > sequence_length:
                         keypoints_sequence = keypoints_list[:sequence_length]
                    else:
                        keypoints_sequence = np.array(keypoints_list)
                    
                    gru_input = keypoints_sequence.reshape(1, sequence_length, num_features)

                    # 3. GRU Prediction
                    gru_predictions = gru_model.predict(gru_input)
                    st.write("GRU Predictions:", gru_predictions)

                    predicted_class = np.argmax(gru_predictions)
                    if predicted_class == 0:
                        st.success("Action: Normal")
                    else:
                        st.error("Action: Cheating/Suspicious")
                        

                except Exception as e:
                    st.error(f"Error processing video: {e}")
            else:
                st.error("YOLO and/or GRU model not loaded.")

import streamlit as st
import tempfile
import cv2
import numpy as np
import tensorflow as tf  # Import TensorFlow

# Upload video file
uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'mov'])

# Define the columns
col1, col2 = st.columns(2)

# Rest of your code...

if uploaded_file is not None:

    # Rendering the video 
    with col1: 
        st.info('The video below displays the uploaded video')
        video_bytes = uploaded_file.read() 
        st.video(video_bytes)



    with col2: 
        st.info('This is all the machine learning model sees when making a prediction')

        # Save the uploaded file to a temporary location
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(video_bytes)
        tfile.close()

        # Read the video frames into a list
        cap = cv2.VideoCapture(tfile.name)
        frames = []
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == False:
                break
            frames.append(frame)
        cap.release()

        # Convert the list of frames into a numpy array
        frames_np = np.array(frames)

        # Convert the numpy array into a TensorFlow tensor
        video_tensor = tf.convert_to_tensor(frames_np)

        video, annotations = load_data(video_tensor)

        st.info('This is the output of the machine learning model as tokens')
        model = load_model()
        yhat = model.predict(tf.expand_dims(video, axis=0))
        decoder = tf.keras.backend.ctc_decode(yhat, [75], greedy=True)[0][0].numpy()
        st.text(decoder)

        # Convert prediction to text
        st.info('Decode the raw tokens into words')
        converted_prediction = tf.strings.reduce_join(num_to_char(decoder)).numpy().decode('utf-8')
        st.text(converted_prediction)
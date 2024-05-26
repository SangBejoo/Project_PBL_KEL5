import os
import io
import base64
import cv2
import dlib
import math
import numpy as np
import tensorflow as tf
from collections import deque
from flask import Flask, render_template, Response, redirect, url_for
from flask_socketio import SocketIO, emit
import sys
import logging
import subprocess

# Add the data_collection directory to the system path
sys.path.append(os.path.join(os.path.dirname(__file__), 'data_collection'))
from constants import *

# Configure logging
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)s %(message)s',
                    handlers=[
                        logging.FileHandler("app.log"),
                        logging.StreamHandler()
                    ])

app = Flask(__name__)
socketio = SocketIO(app)

label_dict = {6: 'hello', 5: 'dog', 10: 'my', 12: 'you', 9: 'lips', 3: 'cat', 11: 'read', 0: 'a', 4: 'demo', 7: 'here', 8: 'is', 1: 'bye', 2: 'can'}
input_shape = (TOTAL_FRAMES, 80, 112, 3)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Conv3D(16, (3, 3, 3), activation='relu', input_shape=input_shape),
    tf.keras.layers.MaxPooling3D((2, 2, 2)),
    tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu'),
    tf.keras.layers.MaxPooling3D((2, 2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(label_dict), activation='softmax')
])

script_dir = os.path.dirname(os.path.realpath(__file__))
weights_path = os.path.join(script_dir, "model_weights.h5")
predictor_path = os.path.join(script_dir, "face_weights.dat")

model.load_weights(weights_path, by_name=True)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

cap = cv2.VideoCapture(0)
curr_word_frames = []
not_talking_counter = 0
first_word = True
labels = []
past_word_frames = deque(maxlen=PAST_BUFFER_SIZE)
ending_buffer_size = 5
predicted_word_label = None
draw_prediction = False
spoken_already = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/streamlit')
def run_streamlit():
    # Stop any existing Streamlit process first
    subprocess.run(["pkill", "-f", "streamlit"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Start Streamlit process
    streamlit_proc = subprocess.Popen(["streamlit", "run", "streamlite.py", "--server.port", "8501"])
    return redirect("http://localhost:8501")

def generate_frames():
    global curr_word_frames, not_talking_counter, draw_prediction, count, predicted_word_label, spoken_already, past_word_frames

    count = 0
    while True:
        success, frame = cap.read()
        if not success:
            logging.error("Failed to capture frame")
            break
        else:
            gray = cv2.cvtColor(src=frame, code=cv2.COLOR_BGR2GRAY)
            faces = detector(gray)

            for face in faces:
                landmarks = predictor(image=gray, box=face)
                mouth_top = (landmarks.part(51).x, landmarks.part(51).y)
                mouth_bottom = (landmarks.part(57).x, landmarks.part(57).y)
                lip_distance = math.hypot(mouth_bottom[0] - mouth_top[0], mouth_bottom[1] - mouth_top[1])
                lip_left = landmarks.part(48).x
                lip_right = landmarks.part(54).x
                lip_top = landmarks.part(50).y
                lip_bottom = landmarks.part(58).y

                width_diff = LIP_WIDTH - (lip_right - lip_left)
                height_diff = LIP_HEIGHT - (lip_bottom - lip_top)
                pad_left = width_diff // 2
                pad_right = width_diff - pad_left
                pad_top = height_diff // 2
                pad_bottom = height_diff - pad_top

                pad_left = min(pad_left, lip_left)
                pad_right = min(pad_right, frame.shape[1] - lip_right)
                pad_top = min(pad_top, lip_top)
                pad_bottom = min(pad_bottom, frame.shape[0] - lip_bottom)

                lip_frame = frame[lip_top - pad_top:lip_bottom + pad_bottom, lip_left - pad_left:lip_right + pad_right]
                lip_frame = cv2.resize(lip_frame, (LIP_WIDTH, LIP_HEIGHT))
                lip_frame_lab = cv2.cvtColor(lip_frame, cv2.COLOR_BGR2LAB)
                l_channel, a_channel, b_channel = cv2.split(lip_frame_lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(3, 3))
                l_channel_eq = clahe.apply(l_channel)
                lip_frame_eq = cv2.merge((l_channel_eq, a_channel, b_channel))
                lip_frame_eq = cv2.cvtColor(lip_frame_eq, cv2.COLOR_LAB2BGR)
                lip_frame_eq = cv2.GaussianBlur(lip_frame_eq, (7, 7), 0)
                lip_frame_eq = cv2.bilateralFilter(lip_frame_eq, 5, 75, 75)
                kernel = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
                lip_frame_eq = cv2.filter2D(lip_frame_eq, -1, kernel)
                lip_frame_eq = cv2.GaussianBlur(lip_frame_eq, (5, 5), 0)
                lip_frame = lip_frame_eq

                if lip_distance > 45:
                    cv2.putText(frame, "Talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    logging.debug("Detected talking")
                    curr_word_frames.append(lip_frame)
                    not_talking_counter = 0
                    draw_prediction = False
                else:
                    cv2.putText(frame, "Not talking", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    logging.debug("Detected not talking")
                    not_talking_counter += 1
                    if not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE == TOTAL_FRAMES:
                        curr_word_frames = list(past_word_frames) + curr_word_frames
                        curr_data = np.array([curr_word_frames[:input_shape[0]]])
                        prediction = model.predict(curr_data)
                        prob_per_class = []
                        for i in range(len(prediction[0])):
                            prob_per_class.append((prediction[0][i], label_dict[i]))
                        sorted_probs = sorted(prob_per_class, key=lambda x: x[0], reverse=True)
                        predicted_class_index = np.argmax(prediction)
                        while label_dict[predicted_class_index] in spoken_already:
                            prediction[0][predicted_class_index] = 0
                            predicted_class_index = np.argmax(prediction)
                        predicted_word_label = label_dict[predicted_class_index]
                        logging.info(f"Predicted word: {predicted_word_label}")
                        spoken_already.append(predicted_word_label)
                        draw_prediction = True
                        count = 0
                        curr_word_frames = []
                        not_talking_counter = 0

                        # Emit the prediction to the client
                        socketio.emit('prediction', {'word': predicted_word_label})

                    elif not_talking_counter < NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE < TOTAL_FRAMES and len(curr_word_frames) > VALID_WORD_THRESHOLD:
                        curr_word_frames.append(lip_frame)
                        not_talking_counter = 0
                    elif len(curr_word_frames) < VALID_WORD_THRESHOLD or (not_talking_counter >= NOT_TALKING_THRESHOLD and len(curr_word_frames) + PAST_BUFFER_SIZE > TOTAL_FRAMES):
                        logging.debug("Clearing current word frames")
                        curr_word_frames = []
                    past_word_frames.append(lip_frame.tolist())
                    if len(past_word_frames) > PAST_BUFFER_SIZE:
                        past_word_frames.pop(0)

            if draw_prediction and count < 20:
                count += 1
                cv2.putText(frame, predicted_word_label, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000)

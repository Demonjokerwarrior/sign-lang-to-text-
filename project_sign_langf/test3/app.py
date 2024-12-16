import os
import json
import math
import cv2
import numpy as np
import pyttsx3
from flask import Flask, render_template, Response, request, jsonify
import mediapipe as mp

# Flask app setup
app = Flask(__name__)

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Directory for saving gesture images and JSON files
output_dir = "gestures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Landmarks for the finger tips
FINGER_TIP_INDICES = [4, 8, 12, 16, 20]

# Text-to-Speech engine setup
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 0.9)  # Volume level

# Global variables
recording = False
gesture_name = None
recognized_gesture = None

# Function to save landmarks to a JSON file
def save_landmarks_to_json(hand_landmarks, gesture_name):
    landmarks = []
    for landmark in hand_landmarks.landmark:
        landmarks.append({
            "x": landmark.x,
            "y": landmark.y,
            "z": landmark.z
        })

    gesture_dir = os.path.join(output_dir, gesture_name)
    if not os.path.exists(gesture_dir):
        os.makedirs(gesture_dir)

    count = len(os.listdir(gesture_dir))
    json_filename = os.path.join(gesture_dir, f"{count + 1}.json")
    with open(json_filename, 'w') as json_file:
        json.dump(landmarks, json_file, indent=4)

# Function to extract normalized landmarks
def extract_landmarks(hand_landmarks):
    return [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]

# Function to compare approximate positions of finger tips
def compare_finger_tips(user_landmarks, saved_landmarks, threshold=0.05):
    for tip_index in FINGER_TIP_INDICES:
        u_landmark = user_landmarks[tip_index]
        s_landmark = saved_landmarks[tip_index]

        dist = math.sqrt((u_landmark["x"] - s_landmark["x"])**2 +
                         (u_landmark["y"] - s_landmark["y"])**2 +
                         (u_landmark["z"] - s_landmark["z"])**2)
        if dist > threshold:
            return False
    return True

# Function to check landmarks against saved gestures
def check_gesture(user_landmarks):
    for gesture_name in os.listdir(output_dir):
        gesture_dir = os.path.join(output_dir, gesture_name)
        if not os.path.isdir(gesture_dir):
            continue

        for filename in os.listdir(gesture_dir):
            if filename.endswith(".json"):
                with open(os.path.join(gesture_dir, filename), "r") as f:
                    saved_landmarks = json.load(f)

                if compare_finger_tips(user_landmarks, saved_landmarks):
                    return gesture_name
    return None

# Function to generate video frames
def generate_frames():
    global recording, gesture_name, recognized_gesture
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            height, width, _ = frame.shape
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(canvas, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    if recording and gesture_name:
                        save_landmarks_to_json(hand_landmarks, gesture_name)

                    # For testing, check gesture
                    user_landmarks = extract_landmarks(hand_landmarks)
                    matched_gesture = check_gesture(user_landmarks)

                    if matched_gesture:
                        recognized_gesture = matched_gesture
                        sentence = f"The recognized gesture is {matched_gesture}"  # Form the sentence
                        cv2.putText(canvas, sentence, (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                        # Speak the full sentence
                        engine.say(sentence)
                        engine.runAndWait()

            ret, buffer = cv2.imencode('.jpg', canvas)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    camera.release()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/train')
def train():
    return render_template('train.html')

@app.route('/test')
def test():
    return render_template('test.html')

@app.route('/set_gesture_name', methods=['POST'])
def set_gesture_name():
    global gesture_name
    data = request.get_json()
    gesture_name = data.get('gesture_name', '').strip()

    if gesture_name:
        return jsonify({"status": "success", "gesture_name": gesture_name})
    else:
        return jsonify({"status": "error", "message": "Invalid gesture name"})

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global recording
    recording = True
    return '', 204

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording
    recording = False
    return '', 204

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

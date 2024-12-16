import os
import json
import math
import cv2
import numpy as np
from flask import Flask, render_template, Response, request, jsonify
import mediapipe as mp

app = Flask(__name__)

# Mediapipe Hands setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Directory for saving gesture images and JSON files
output_dir = "gestures"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

recording = False
gesture_name = None

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

    # Save the landmarks to a JSON file
    count = len(os.listdir(gesture_dir))
    json_filename = os.path.join(gesture_dir, f"{count + 1}.json")
    with open(json_filename, 'w') as json_file:
        json.dump(landmarks, json_file, indent=4)

# Function to compare two sets of landmarks
def compare_landmarks(user_landmarks, saved_landmarks):
    # Calculate the Euclidean distance between each corresponding point in the landmarks
    threshold = 0.02  # This threshold determines how closely the landmarks should match
    for u_landmark, s_landmark in zip(user_landmarks, saved_landmarks):
        dist = math.sqrt((u_landmark["x"] - s_landmark["x"])**2 +
                         (u_landmark["y"] - s_landmark["y"])**2 +
                         (u_landmark["z"] - s_landmark["z"])**2)
        if dist > threshold:
            return False
    return True

# Function to generate video frames
def generate_frames():
    global recording, gesture_name
    camera = cv2.VideoCapture(0)

    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Process frame for hand landmarks
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # Create a black canvas for drawing landmarks and connections
            height, width, _ = frame.shape
            canvas = np.zeros((height, width, 3), dtype=np.uint8)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw hand landmarks and connections
                    mp_drawing.draw_landmarks(canvas, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Save landmarks as JSON if recording
                    if recording and gesture_name:
                        save_landmarks_to_json(hand_landmarks, gesture_name)

            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', canvas)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_gesture_name', methods=['POST'])
def set_gesture_name():
    global gesture_name
    data = request.get_json()  # Fetch JSON data
    gesture_name = data.get('gesture_name', '').strip()  # Get gesture_name from JSON

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

@app.route('/compare_gesture', methods=['POST'])
def compare_gesture():
    global gesture_name
    user_landmarks = request.get_json().get('landmarks')  # User's hand landmarks from the front end

    if not user_landmarks or not gesture_name:
        return jsonify({"status": "error", "message": "Invalid landmarks or no gesture name set"})

    # Load the pre-saved gesture landmarks for comparison
    gesture_dir = os.path.join(output_dir, gesture_name)
    saved_landmarks_list = []

    for filename in os.listdir(gesture_dir):
        if filename.endswith(".json"):
            with open(os.path.join(gesture_dir, filename), "r") as f:
                saved_landmarks_list.append(json.load(f))

    if not saved_landmarks_list:
        return jsonify({"status": "error", "message": "No saved landmarks found for this gesture"})

    # Compare user landmarks with all saved landmarks for this gesture
    for saved_landmarks in saved_landmarks_list:
        if compare_landmarks(user_landmarks, saved_landmarks):
            return jsonify({"status": "success", "match": True})

    return jsonify({"status": "success", "match": False})

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/recognition')
def recognition():
    return render_template('recognition.html')


if __name__ == '__main__':
    app.run(debug=True)

import cv2
import mediapipe as mp
import os
import json
import math
import subprocess
import webbrowser
import numpy as np

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Directory where gesture landmarks are stored
output_dir = "gestures"

# Landmarks for the finger tips (thumb, index, middle, ring, pinky)
FINGER_TIP_INDICES = [4, 8, 12, 16, 20]

# Function to compare approximate positions of finger tips
def compare_finger_tips(user_landmarks, saved_landmarks, threshold=0.05):
    """
    Compare only the positions of the five finger tips for approximate matching.
    """
    for tip_index in FINGER_TIP_INDICES:
        u_landmark = user_landmarks[tip_index]
        s_landmark = saved_landmarks[tip_index]

        dist = math.sqrt((u_landmark["x"] - s_landmark["x"])**2 +
                         (u_landmark["y"] - s_landmark["y"])**2 +
                         (u_landmark["z"] - s_landmark["z"])**2)
        if dist > threshold:
            return False  # Finger tip positions are not close enough
    return True

# Function to check landmarks against saved gestures
def check_gesture(user_landmarks):
    for gesture_name in os.listdir(output_dir):
        gesture_dir = os.path.join(output_dir, gesture_name)
        if not os.path.isdir(gesture_dir):
            continue

        # Load all saved landmarks for the gesture
        for filename in os.listdir(gesture_dir):
            if filename.endswith(".json"):
                with open(os.path.join(gesture_dir, filename), "r") as f:
                    saved_landmarks = json.load(f)

                # Compare the finger tips for an approximate match
                if compare_finger_tips(user_landmarks, saved_landmarks):
                    return gesture_name  # Return the matched gesture name
    return None  # No match found

# Function to extract normalized landmarks
def extract_landmarks(hand_landmarks):
    return [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in hand_landmarks.landmark]

# Function to display the gesture name using figlet
def display_with_figlet(gesture_name):
    subprocess.run(["figlet", gesture_name])

# Function to search Google using Firefox
def search_google(gesture_name):
    url = gesture_name
    webbrowser.get("firefox").open(url)

# Main video feed processing
def main():
    cap = cv2.VideoCapture(0)  # Open webcam
    print("Press 'q' to exit.")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Create a blank canvas to display the landmarks
        height, width, _ = frame.shape
        canvas = np.zeros((height, width, 3), dtype=np.uint8)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Extract landmarks for comparison
                user_landmarks = extract_landmarks(hand_landmarks)

                # Compare the landmarks with saved gestures
                matched_gesture = check_gesture(user_landmarks)

                if matched_gesture:
                    print(f"Matched Gesture: {matched_gesture}")
                    # Use figlet to display the name
                    display_with_figlet(matched_gesture)
                    # Open Firefox and search on Google
                    search_google(matched_gesture)
                else:
                    print("No matching gesture found.")

                # Draw the landmarks on the canvas
                mp_drawing.draw_landmarks(canvas, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display the canvas
        cv2.imshow("Landmark Tracking", canvas)

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if not os.path.exists(output_dir):
        print(f"Gesture directory '{output_dir}' does not exist. Create it and add gesture JSON files.")
    else:
        main()

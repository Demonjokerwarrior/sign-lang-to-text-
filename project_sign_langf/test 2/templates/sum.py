import os
import json
import math

# Directory where gestures are stored
output_dir = "gestures"

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

# Function to check a user's gesture against saved gestures
def check_gesture(user_landmarks, gesture_name):
    gesture_dir = os.path.join(output_dir, gesture_name)

    if not os.path.exists(gesture_dir):
        print(f"Gesture '{gesture_name}' does not exist.")
        return False

    saved_landmarks_list = []
    for filename in os.listdir(gesture_dir):
        if filename.endswith(".json"):
            with open(os.path.join(gesture_dir, filename), "r") as f:
                saved_landmarks_list.append(json.load(f))

    if not saved_landmarks_list:
        print(f"No saved landmarks found for gesture '{gesture_name}'.")
        return False

    # Compare user's landmarks with each saved landmark
    for saved_landmarks in saved_landmarks_list:
        if compare_landmarks(user_landmarks, saved_landmarks):
            print(f"Gesture '{gesture_name}' matched successfully.")
            return True

    print(f"Gesture '{gesture_name}' did not match.")
    return False

# Example Usage
if __name__ == "__main__":
    # Example user landmarks to compare (replace this with actual data)
    example_user_landmarks = [
        {"x": 0.5, "y": 0.5, "z": 0.1},  # Replace these values with real hand landmark data
        {"x": 0.6, "y": 0.6, "z": 0.2},
        # ... more landmarks
    ]

    # Gesture name to check
    gesture_name = input("Enter the gesture name to check: ").strip()

    # Perform gesture check
    result = check_gesture(example_user_landmarks, gesture_name)
    if result:
        print("Gesture recognition successful!")
    else:
        print("Gesture recognition failed.")

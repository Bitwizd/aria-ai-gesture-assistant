import cv2
import time
import mediapipe as mp
import csv
import os

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Model Path
# -----------------------------
MODEL_PATH = "models/hand_landmarker.task"


# -----------------------------
# Dataset Path
# -----------------------------
DATASET_DIR = "dataset"
DATASET_FILE = os.path.join(DATASET_DIR, "hand_gestures.csv")

os.makedirs(DATASET_DIR, exist_ok=True)


# -----------------------------
# Gesture Classes
# -----------------------------
GESTURES = {
    "0": "Open Palm",
    "1": "Fist",
    "2": "Thumbs Up",
    "3": "Peace",
    "4": "Pointing"
}


# -----------------------------
# Create Dataset File
# -----------------------------
if not os.path.exists(DATASET_FILE):

    with open(DATASET_FILE, "w", newline="") as file:

        writer = csv.writer(file)

        header = []

        for i in range(1, 22):
            header.extend([
                f"x{i}",
                f"y{i}",
                f"z{i}"
            ])

        header.append("label")

        writer.writerow(header)


# -----------------------------
# Initialize Hand Landmarker
# -----------------------------
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1
)

landmarker = vision.HandLandmarker.create_from_options(
    options
)


# -----------------------------
# Initialize Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("Could not access the webcam.")
    exit()

print("Camera started successfully.")
print()
print("Dataset Collection Controls:")
print("0 = Open Palm")
print("1 = Fist")
print("2 = Thumbs Up")
print("3 = Peace")
print("4 = Pointing")
print("Q = Quit")
print()


# -----------------------------
# Current Recording State
# -----------------------------
current_label = None
current_gesture = "Not Recording"


# -----------------------------
# Sample Counters
# -----------------------------
sample_counts = {
    "0": 0,
    "1": 0,
    "2": 0,
    "3": 0,
    "4": 0
}


# -----------------------------
# Count Existing Samples
# -----------------------------
if os.path.exists(DATASET_FILE):

    with open(DATASET_FILE, "r", newline="") as file:

        reader = csv.DictReader(file)

        for row in reader:

            label = row["label"]

            if label in sample_counts:
                sample_counts[label] += 1


# -----------------------------
# FPS Calculation
# -----------------------------
previous_time = time.time()


# -----------------------------
# Main Camera Loop
# -----------------------------
while True:

    ret, frame = cap.read()

    if not ret:

        print("Failed to capture frame.")
        break


    # -----------------------------
    # Get Frame Dimensions
    # -----------------------------
    height, width, channels = frame.shape


    # -----------------------------
    # Convert BGR → RGB
    # -----------------------------
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )


    # -----------------------------
    # Convert to MediaPipe Image
    # -----------------------------
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )


    # -----------------------------
    # Run Hand Landmarker
    # -----------------------------
    timestamp_ms = int(time.time() * 1000)

    result = landmarker.detect_for_video(
        mp_image,
        timestamp_ms
    )


    # -----------------------------
    # Process Hand
    # -----------------------------
    if result.hand_landmarks:

        hand_landmarks = result.hand_landmarks[0]


        # -----------------------------
        # Wrist Reference
        # -----------------------------
        wrist_x = hand_landmarks[0].x
        wrist_y = hand_landmarks[0].y
        wrist_z = hand_landmarks[0].z


        # -----------------------------
        # Extract Normalized Features
        # -----------------------------
        landmarks = []

        for landmark in hand_landmarks:

            x = landmark.x - wrist_x
            y = landmark.y - wrist_y
            z = landmark.z - wrist_z

            landmarks.extend([
                x,
                y,
                z
            ])


        # -----------------------------
        # Save Sample
        # -----------------------------
        if current_label is not None:

            with open(
                DATASET_FILE,
                "a",
                newline=""
            ) as file:

                writer = csv.writer(file)

                row = landmarks + [current_label]

                writer.writerow(row)

            sample_counts[current_label] += 1


        # -----------------------------
        # Draw Landmark Points
        # -----------------------------
        for landmark in hand_landmarks:

            x = int(landmark.x * width)
            y = int(landmark.y * height)

            cv2.circle(
                frame,
                (x, y),
                5,
                (0, 255, 0),
                -1
            )


        # -----------------------------
        # Draw Landmark Connections
        # -----------------------------
        connections = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 4),

            (0, 5),
            (5, 6),
            (6, 7),
            (7, 8),

            (0, 9),
            (9, 10),
            (10, 11),
            (11, 12),

            (0, 13),
            (13, 14),
            (14, 15),
            (15, 16),

            (0, 17),
            (17, 18),
            (18, 19),
            (19, 20),

            (5, 9),
            (9, 13),
            (13, 17)
        ]


        for start, end in connections:

            x1 = int(
                hand_landmarks[start].x * width
            )

            y1 = int(
                hand_landmarks[start].y * height
            )

            x2 = int(
                hand_landmarks[end].x * width
            )

            y2 = int(
                hand_landmarks[end].y * height
            )

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


    # -----------------------------
    # Calculate FPS
    # -----------------------------
    current_time = time.time()

    time_difference = current_time - previous_time

    if time_difference > 0:

        fps = 1 / time_difference

    else:

        fps = 0

    previous_time = current_time


    # -----------------------------
    # Display Recording Status
    # -----------------------------
    if current_label is not None:

        status_text = (
            f"RECORDING: {current_gesture}"
        )

        status_color = (0, 0, 255)

    else:

        status_text = "NOT RECORDING"

        status_color = (0, 255, 0)


    cv2.putText(
        frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        status_color,
        2
    )


    # -----------------------------
    # Display FPS
    # -----------------------------
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )


    # -----------------------------
    # Display Sample Counts
    # -----------------------------
    cv2.putText(
        frame,
        f"Palm: {sample_counts['0']}",
        (20, 115),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Fist: {sample_counts['1']}",
        (20, 145),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Thumb: {sample_counts['2']}",
        (20, 175),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Peace: {sample_counts['3']}",
        (20, 205),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        f"Pointing: {sample_counts['4']}",
        (20, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # -----------------------------
    # Display Camera
    # -----------------------------
    cv2.imshow(
        "Hand Gesture Dataset Collection",
        frame
    )


    # -----------------------------
    # Keyboard Controls
    # -----------------------------
    key = cv2.waitKey(1) & 0xFF


    if key == ord("0"):

        current_label = "0"
        current_gesture = GESTURES["0"]

        print(
            f"Recording {current_gesture}..."
        )


    elif key == ord("1"):

        current_label = "1"
        current_gesture = GESTURES["1"]

        print(
            f"Recording {current_gesture}..."
        )


    elif key == ord("2"):

        current_label = "2"
        current_gesture = GESTURES["2"]

        print(
            f"Recording {current_gesture}..."
        )


    elif key == ord("3"):

        current_label = "3"
        current_gesture = GESTURES["3"]

        print(
            f"Recording {current_gesture}..."
        )


    elif key == ord("4"):

        current_label = "4"
        current_gesture = GESTURES["4"]

        print(
            f"Recording {current_gesture}..."
        )


    elif key == ord("q"):

        print()
        print("Dataset collection stopped.")

        break


# -----------------------------
# Cleanup
# -----------------------------
cap.release()

cv2.destroyAllWindows()

landmarker.close()


# -----------------------------
# Final Dataset Summary
# -----------------------------
print()
print("Dataset Summary")
print("-----------------------------")

for label, gesture in GESTURES.items():

    print(
        f"{gesture}: {sample_counts[label]} samples"
    )

print()
print(
    f"Dataset saved to: {DATASET_FILE}"
)
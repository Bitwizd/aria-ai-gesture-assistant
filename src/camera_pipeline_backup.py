import cv2
import time
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -----------------------------
# Model Path
# -----------------------------
MODEL_PATH = "models/hand_landmarker.task"


# -----------------------------
# Initialize Hand Landmarker
# -----------------------------
base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2
)

landmarker = vision.HandLandmarker.create_from_options(options)


# -----------------------------
# Initialize Webcam
# -----------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not access the webcam.")
    exit()

print("Camera started successfully.")


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

    # Get frame dimensions
    height, width, channels = frame.shape

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Convert OpenCV image to MediaPipe Image
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Run Hand Landmarker
    timestamp_ms = int(time.time() * 1000)

    result = landmarker.detect_for_video(
    mp_image,
    timestamp_ms
)

    # -----------------------------
    # Draw Hand Landmarks
    # -----------------------------
    for hand_landmarks in result.hand_landmarks:

        # Draw each landmark
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

        # Draw connections between landmarks
        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (0, 9), (9, 10), (10, 11), (11, 12),
            (0, 13), (13, 14), (14, 15), (15, 16),
            (0, 17), (17, 18), (18, 19), (19, 20),
            (5, 9), (9, 13), (13, 17)
        ]

        for start, end in connections:

            x1 = int(hand_landmarks[start].x * width)
            y1 = int(hand_landmarks[start].y * height)

            x2 = int(hand_landmarks[end].x * width)
            y2 = int(hand_landmarks[end].y * height)

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

    fps = 1 / (current_time - previous_time)

    previous_time = current_time


    # -----------------------------
    # Display Information
    # -----------------------------
    cv2.putText(
        frame,
        f"Width: {width} Height: {height} Channels: {channels}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

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
    # Display Camera
    # -----------------------------
    cv2.imshow("Hand Landmark Pipeline", frame)


    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# -----------------------------
# Cleanup
# -----------------------------
cap.release()
cv2.destroyAllWindows()

landmarker.close()
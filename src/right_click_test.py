import cv2
import mediapipe as mp
import pandas as pd
import joblib
import pyautogui
import math
import time


MODEL_PATH = "models/gesture_classifier.pkl"
LANDMARK_MODEL = "models/hand_landmarker.task"

GESTURES = {
    0: "OPEN PALM",
    1: "FIST",
    2: "THUMBS UP",
    3: "PEACE",
    4: "POINTING"
}

FEATURE_NAMES = []

for i in range(1, 22):
    FEATURE_NAMES.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])

model = joblib.load(MODEL_PATH)

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=LANDMARK_MODEL
    ),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

timestamp = 0
right_click_done = False

PINCH_THRESHOLD = 0.08

print()
print("RIGHT CLICK TEST")
print("-----------------------------")
print("Make PEACE ✌️")
print("Then pinch thumb + index 🤏")
print("Hold for 0.5 seconds")
print("-----------------------------")
print("Q = EXIT")
print()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp += 33

    result = landmarker.detect_for_video(
        image,
        timestamp
    )

    status = "NO HAND"

    if result.hand_landmarks:

        landmarks = result.hand_landmarks[0]

        wrist = landmarks[0]

        features = []

        for landmark in landmarks:

            features.extend([
                landmark.x - wrist.x,
                landmark.y - wrist.y,
                landmark.z - wrist.z
            ])

        features = pd.DataFrame(
            [features],
            columns=FEATURE_NAMES
        )

        prediction = model.predict(features)

        label = int(prediction[0])

        gesture = GESTURES.get(
            label,
            "UNKNOWN"
        )

        thumb = landmarks[4]
        index = landmarks[8]

        distance = math.sqrt(
            (thumb.x - index.x) ** 2 +
            (thumb.y - index.y) ** 2
        )

        pinch = distance < PINCH_THRESHOLD

        if label == 3 and pinch:

            status = "RIGHT CLICK READY"

            if not right_click_done:

                time.sleep(0.5)

                pyautogui.rightClick()

                right_click_done = True

                status = "RIGHT CLICK!"

        elif label == 3:

            status = "PEACE - NOW PINCH"

            right_click_done = False

        else:

            status = gesture

            right_click_done = False


    cv2.putText(
        frame,
        status,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (80, 80, 255),
        2
    )

    cv2.putText(
        frame,
        "PEACE + PINCH = RIGHT CLICK",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (180, 70, 20),
        1
    )

    cv2.imshow(
        "ARIA Right Click Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
landmarker.close()

print("Right-click test stopped.")
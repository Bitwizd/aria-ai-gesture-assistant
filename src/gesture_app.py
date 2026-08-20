import cv2
import mediapipe as mp
import numpy as np
import joblib

from gesture_effects import apply_effect


# ---------------------------------
# Load trained ML model
# ---------------------------------

MODEL_PATH = "models/gesture_classifier.pkl"

model = joblib.load(MODEL_PATH)

print("Gesture classifier loaded successfully.")


# ---------------------------------
# Gesture labels
# ---------------------------------

GESTURE_NAMES = {
    0: "Open Palm",
    1: "Fist",
    2: "Thumbs Up",
    3: "Peace",
    4: "Pointing",
}


# ---------------------------------
# MediaPipe setup
# ---------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
)


# ---------------------------------
# Start webcam
# ---------------------------------

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


print("Camera started successfully.")
print("Press Q to quit.")


# ---------------------------------
# Run hand detection
# ---------------------------------

with HandLandmarker.create_from_options(options) as landmarker:

    while True:

        ret, frame = cap.read()
        
        if not ret:

            print("ERROR: Could not read frame.")
            break


        # ---------------------------------
        # Convert BGR → RGB
        # ---------------------------------

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # ---------------------------------
        # Create MediaPipe image
        # ---------------------------------

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ---------------------------------
        # Detect hand
        # ---------------------------------

        result = landmarker.detect(mp_image)

        prediction_text = "No hand detected"

        hand = None


        # ---------------------------------
        # Extract landmarks
        # ---------------------------------

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # ---------------------------------
            # Wrist-relative coordinates
            # ---------------------------------

            wrist_x = hand[0].x
            wrist_y = hand[0].y
            wrist_z = hand[0].z

            features = []


            for landmark in hand:

                x = landmark.x - wrist_x
                y = landmark.y - wrist_y
                z = landmark.z - wrist_z

                features.extend([
                    x,
                    y,
                    z
                ])


            # ---------------------------------
            # Convert to numpy
            # ---------------------------------

            features = np.array(
                features,
                dtype=np.float32
            ).reshape(1, -1)


            # ---------------------------------
            # Predict gesture
            # ---------------------------------

            prediction = model.predict(features)[0]

            prediction_text = GESTURE_NAMES.get(
                int(prediction),
                str(prediction)
            )


        # ---------------------------------
        # Display gesture name
        # ---------------------------------

        cv2.putText(
            frame,
            f"Gesture: {prediction_text}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


        # ---------------------------------
        # Apply hand-positioned effect
        # ---------------------------------

        if hand is not None:

            frame = apply_effect(
                frame,
                prediction_text,
                hand
            )


        # ---------------------------------
        # Display camera
        # ---------------------------------

        cv2.imshow(
            "AI Gesture Effects",
            frame
        )


        # ---------------------------------
        # Quit
        # ---------------------------------

        if cv2.waitKey(1) & 0xFF == ord("q"):

            break


# ---------------------------------
# Cleanup
# ---------------------------------

cap.release()

cv2.destroyAllWindows()

print("AI Gesture Effects stopped.")
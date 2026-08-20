import cv2
import mediapipe as mp
import math

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode

HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="models/hand_landmarker.task"
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

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

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

        thumb = landmarks[4]
        index = landmarks[8]

        distance = math.sqrt(
            (thumb.x - index.x) ** 2 +
            (thumb.y - index.y) ** 2
        )

        if distance < 0.06:
            status = "PINCH DETECTED"
        else:
            status = "OPEN"

        cv2.putText(
            frame,
            f"Distance: {distance:.3f}",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    cv2.putText(
        frame,
        status,
        (30, 90),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2
    )

    cv2.imshow(
        "ARIA Pinch Test",
        frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
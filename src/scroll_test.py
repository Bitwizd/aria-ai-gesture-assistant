import cv2
import mediapipe as mp


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


landmarker = HandLandmarker.create_from_options(
    options
)


cap = cv2.VideoCapture(0)

timestamp = 0
previous_y = None


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

        # Index fingertip
        index = landmarks[8]

        current_y = index.y


        if previous_y is not None:

            movement = current_y - previous_y

            if movement < -0.015:
                status = "SCROLL UP"

            elif movement > 0.015:
                status = "SCROLL DOWN"

            else:
                status = "HOLD"


        previous_y = current_y


        # Draw fingertip
        px = int(index.x * frame.shape[1])
        py = int(index.y * frame.shape[0])

        cv2.circle(
            frame,
            (px, py),
            8,
            (80, 80, 255),
            -1
        )


    cv2.putText(
        frame,
        status,
        (30, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (80, 80, 255),
        2
    )


    cv2.putText(
        frame,
        "PEACE: MOVE UP / DOWN",
        (30, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 70, 20),
        1
    )


    cv2.imshow(
        "ARIA Scroll Test",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()
landmarker.close()
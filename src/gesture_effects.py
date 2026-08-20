import cv2


# ---------------------------------
# Convert normalized landmark
# coordinates to pixel coordinates
# ---------------------------------

def landmark_to_pixel(landmark, width, height):

    x = int(landmark.x * width)
    y = int(landmark.y * height)

    return x, y


# ---------------------------------
# Apply gesture effect
# ---------------------------------

def apply_effect(frame, gesture_name, hand_landmarks):

    height, width = frame.shape[:2]

    # Important hand landmarks
    wrist = hand_landmarks[0]
    thumb_tip = hand_landmarks[4]
    index_tip = hand_landmarks[8]
    middle_tip = hand_landmarks[12]

    wrist_x, wrist_y = landmark_to_pixel(
        wrist,
        width,
        height
    )

    thumb_x, thumb_y = landmark_to_pixel(
        thumb_tip,
        width,
        height
    )

    index_x, index_y = landmark_to_pixel(
        index_tip,
        width,
        height
    )

    middle_x, middle_y = landmark_to_pixel(
        middle_tip,
        width,
        height
    )


    # ---------------------------------
    # Open Palm
    # ---------------------------------

    if gesture_name == "Open Palm":

        cv2.circle(
            frame,
            (wrist_x, wrist_y),
            90,
            (0, 255, 0),
            5
        )


    # ---------------------------------
    # Fist
    # ---------------------------------

    elif gesture_name == "Fist":

        cv2.circle(
            frame,
            (wrist_x, wrist_y),
            80,
            (0, 0, 255),
            5
        )


    # ---------------------------------
    # Thumbs Up
    # ---------------------------------

    elif gesture_name == "Thumbs Up":

        cv2.putText(
            frame,
            "THUMBS UP!",
            (thumb_x - 80, thumb_y - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 0, 0),
            3
        )


    # ---------------------------------
    # Peace
    # ---------------------------------

    elif gesture_name == "Peace":

        cv2.putText(
            frame,
            "PEACE!",
            (index_x - 70, index_y - 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            3
        )


    # ---------------------------------
    # Pointing
    # ---------------------------------

    elif gesture_name == "Pointing":

        cv2.circle(
            frame,
            (index_x, index_y),
            45,
            (255, 255, 255),
            3
        )

        cv2.line(
            frame,
            (index_x - 60, index_y),
            (index_x + 60, index_y),
            (255, 255, 255),
            2
        )

        cv2.line(
            frame,
            (index_x, index_y - 60),
            (index_x, index_y + 60),
            (255, 255, 255),
            2
        )


    return frame
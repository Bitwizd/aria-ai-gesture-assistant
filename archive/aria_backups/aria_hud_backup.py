import cv2
import mediapipe as mp
import numpy as np
import joblib
import math
import time


# ============================================================
# ARIA HOLOGRAPHIC HUD
# ============================================================

MODEL_PATH = "models/gesture_classifier.pkl"
LANDMARK_MODEL = "models/hand_landmarker.task"


# ============================================================
# GESTURES
# ============================================================

GESTURE_NAMES = {
    0: "OPEN PALM",
    1: "FIST",
    2: "THUMBS UP",
    3: "PEACE",
    4: "POINTING",
}


# ============================================================
# COLOUR PALETTE
#
# Based on your reference image:
# black / navy / electric cyan / white / teal
# ============================================================

BLACK = (2, 3, 8)
NAVY = (12, 18, 38)
DEEP_NAVY = (20, 30, 60)

CYAN = (255, 210, 40)
BRIGHT_CYAN = (255, 245, 80)
SOFT_CYAN = (180, 150, 35)

WHITE = (245, 245, 245)
SILVER = (180, 185, 195)

TEAL = (180, 210, 70)

GREY = (80, 90, 105)


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(MODEL_PATH)

print("ARIA gesture model loaded successfully.")


# ============================================================
# MEDIAPIPE
# ============================================================

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=LANDMARK_MODEL
    ),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1,
)


# ============================================================
# DRAWING FUNCTIONS
# ============================================================

def draw_text(
    frame,
    text,
    position,
    size=0.5,
    color=WHITE,
    thickness=1
):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        size,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_glow_circle(
    frame,
    center,
    radius,
    color,
    thickness=1
):
    glow = frame.copy()

    cv2.circle(
        glow,
        center,
        radius,
        color,
        thickness + 6,
        cv2.LINE_AA
    )

    frame[:] = cv2.addWeighted(
        frame,
        0.80,
        glow,
        0.20,
        0
    )

    cv2.circle(
        frame,
        center,
        radius,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_ring(
    frame,
    center,
    radius,
    color,
    thickness=1,
    start=0,
    end=360
):
    cv2.ellipse(
        frame,
        center,
        (radius, radius),
        0,
        start,
        end,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_ticks(
    frame,
    center,
    radius,
    count,
    color,
    tick_length=8,
    thickness=1,
    offset=0
):
    cx, cy = center

    for i in range(count):

        angle = math.radians(
            (360 / count) * i + offset
        )

        x1 = int(
            cx + radius * math.cos(angle)
        )

        y1 = int(
            cy + radius * math.sin(angle)
        )

        x2 = int(
            cx + (radius + tick_length)
            * math.cos(angle)
        )

        y2 = int(
            cy + (radius + tick_length)
            * math.sin(angle)
        )

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
            cv2.LINE_AA
        )


def draw_hexagon(
    frame,
    center,
    radius,
    color,
    thickness=1
):
    points = []

    for i in range(6):

        angle = math.radians(
            60 * i - 30
        )

        x = int(
            center[0] +
            radius * math.cos(angle)
        )

        y = int(
            center[1] +
            radius * math.sin(angle)
        )

        points.append((x, y))

    points = np.array(
        points,
        dtype=np.int32
    )

    cv2.polylines(
        frame,
        [points],
        True,
        color,
        thickness,
        cv2.LINE_AA
    )


def draw_corner_brackets(
    frame,
    margin=20,
    length=35
):
    h, w = frame.shape[:2]

    # Top left
    cv2.line(
        frame,
        (margin, margin),
        (margin + length, margin),
        CYAN,
        1
    )

    cv2.line(
        frame,
        (margin, margin),
        (margin, margin + length),
        CYAN,
        1
    )

    # Top right
    cv2.line(
        frame,
        (w - margin, margin),
        (w - margin - length, margin),
        CYAN,
        1
    )

    cv2.line(
        frame,
        (w - margin, margin),
        (w - margin, margin + length),
        CYAN,
        1
    )

    # Bottom left
    cv2.line(
        frame,
        (margin, h - margin),
        (margin + length, h - margin),
        CYAN,
        1
    )

    cv2.line(
        frame,
        (margin, h - margin),
        (margin, h - margin - length),
        CYAN,
        1
    )

    # Bottom right
    cv2.line(
        frame,
        (w - margin, h - margin),
        (w - margin - length, h - margin),
        CYAN,
        1
    )

    cv2.line(
        frame,
        (w - margin, h - margin),
        (w - margin, h - margin - length),
        CYAN,
        1
    )


def draw_target(
    frame,
    x,
    y
):
    # Outer glow
    glow = frame.copy()

    cv2.circle(
        glow,
        (x, y),
        35,
        BRIGHT_CYAN,
        5,
        cv2.LINE_AA
    )

    frame[:] = cv2.addWeighted(
        frame,
        0.75,
        glow,
        0.25,
        0
    )

    cv2.circle(
        frame,
        (x, y),
        35,
        BRIGHT_CYAN,
        1,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        (x, y),
        20,
        CYAN,
        1,
        cv2.LINE_AA
    )

    cv2.circle(
        frame,
        (x, y),
        4,
        WHITE,
        -1,
        cv2.LINE_AA
    )

    cv2.line(
        frame,
        (x - 48, y),
        (x - 15, y),
        CYAN,
        1
    )

    cv2.line(
        frame,
        (x + 15, y),
        (x + 48, y),
        CYAN,
        1
    )

    cv2.line(
        frame,
        (x, y - 48),
        (x, y - 15),
        CYAN,
        1
    )

    cv2.line(
        frame,
        (x, y + 15),
        (x, y + 48),
        CYAN,
        1
    )


def draw_hand(
    frame,
    hand,
    width,
    height
):
    # MediaPipe hand connections
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

    points = []

    for landmark in hand:

        x = int(
            landmark.x * width
        )

        y = int(
            landmark.y * height
        )

        points.append((x, y))

        cv2.circle(
            frame,
            (x, y),
            3,
            BRIGHT_CYAN,
            -1,
            cv2.LINE_AA
        )

    for a, b in connections:

        cv2.line(
            frame,
            points[a],
            points[b],
            SOFT_CYAN,
            1,
            cv2.LINE_AA
        )


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():

    print("ERROR: Could not open webcam.")
    exit()


print()
print("==========================================")
print("          ARIA HOLOGRAPHIC HUD")
print("==========================================")
print()
print("POINTING   = Virtual targeting cursor")
print("OPEN PALM  = Scan mode")
print("FIST       = Lock / activate")
print("PEACE      = Secondary mode")
print("THUMBS UP  = Confirm")
print("Q          = Exit")
print()


# ============================================================
# WINDOW
# ============================================================

WINDOW_NAME = "ARIA // HOLOGRAPHIC INTERFACE"

cv2.namedWindow(
    WINDOW_NAME,
    cv2.WINDOW_NORMAL
)


# ============================================================
# STATE
# ============================================================

start_time = time.time()

smooth_cursor_x = 640
smooth_cursor_y = 360

gesture = "NO HAND"
confidence = 0.0
action = "STANDBY"

previous_gesture = "NO HAND"

action_flash_until = 0


# ============================================================
# MAIN LOOP
# ============================================================

with HandLandmarker.create_from_options(
    options
) as landmarker:

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Could not read frame."
            )

            break


        height, width = frame.shape[:2]


        # ====================================================
        # DARKEN CAMERA
        # ====================================================

        dark_layer = np.zeros_like(frame)

        dark_layer[:] = (
            4,
            6,
            15
        )

        frame = cv2.addWeighted(
            frame,
            0.38,
            dark_layer,
            0.62,
            0
        )


        # ====================================================
        # RGB FOR MEDIAPIPE
        # ====================================================

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )


        # ====================================================
        # DETECT HAND
        # ====================================================

        result = landmarker.detect(
            mp_image
        )

        hand = None

        gesture = "NO HAND"

        confidence = 0.0

        action = "STANDBY"


        # ====================================================
        # HAND FOUND
        # ====================================================

        if result.hand_landmarks:

            hand = result.hand_landmarks[0]


            # ------------------------------------------------
            # SAME PREPROCESSING AS YOUR WORKING MODEL
            # ------------------------------------------------

            wrist_x = hand[0].x
            wrist_y = hand[0].y
            wrist_z = hand[0].z

            features = []

            for landmark in hand:

                x = (
                    landmark.x -
                    wrist_x
                )

                y = (
                    landmark.y -
                    wrist_y
                )

                z = (
                    landmark.z -
                    wrist_z
                )

                features.extend([
                    x,
                    y,
                    z
                ])


            features = np.array(
                features,
                dtype=np.float32
            ).reshape(1, -1)


            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            prediction = model.predict(
                features
            )[0]

            gesture = GESTURE_NAMES.get(
                int(prediction),
                "UNKNOWN"
            )


            # ------------------------------------------------
            # CONFIDENCE
            # ------------------------------------------------

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = (
                    model.predict_proba(
                        features
                    )
                )

                confidence = float(
                    np.max(
                        probabilities[0]
                    )
                )


            # =================================================
            # GESTURE ACTIONS
            # =================================================

            if gesture == "POINTING":

                index_tip = hand[8]

                target_x = int(
                    index_tip.x * width
                )

                target_y = int(
                    index_tip.y * height
                )

                smooth_cursor_x = (
                    smooth_cursor_x * 0.75
                    +
                    target_x * 0.25
                )

                smooth_cursor_y = (
                    smooth_cursor_y * 0.75
                    +
                    target_y * 0.25
                )

                action = "TARGETING"


            elif gesture == "OPEN PALM":

                action = "SCANNING"


            elif gesture == "FIST":

                action = "LOCKED"


            elif gesture == "PEACE":

                action = "SECONDARY MODE"


            elif gesture == "THUMBS UP":

                action = "CONFIRMED"


            # ------------------------------------------------
            # Detect gesture changes
            # ------------------------------------------------

            if gesture != previous_gesture:

                action_flash_until = (
                    time.time() + 0.35
                )

            previous_gesture = gesture


        # ====================================================
        # OUTER HUD
        # ====================================================

        draw_corner_brackets(
            frame
        )


        # ====================================================
        # TOP HEADER
        # ====================================================

        draw_text(
            frame,
            "ARIA",
            (45, 55),
            0.95,
            BRIGHT_CYAN,
            2
        )

        draw_text(
            frame,
            "HOLOGRAPHIC INTELLIGENCE INTERFACE",
            (115, 53),
            0.45,
            WHITE,
            1
        )

        draw_text(
            frame,
            "99%",
            (width - 65, 35),
            0.45,
            TEAL,
            1
        )

        draw_text(
            frame,
            "ONLINE",
            (width - 82, 55),
            0.38,
            TEAL,
            1
        )


        # ====================================================
        # CENTRAL HUD
        # ====================================================

        center_x = width // 2
        center_y = height // 2

        center = (
            center_x,
            center_y
        )

        elapsed = (
            time.time() -
            start_time
        )

        rotation = (
            elapsed * 25
        ) % 360


        # Outer rings

        draw_ring(
            frame,
            center,
            185,
            DEEP_NAVY,
            1
        )

        draw_ring(
            frame,
            center,
            175,
            SOFT_CYAN,
            1
        )

        draw_ring(
            frame,
            center,
            160,
            CYAN,
            2,
            20,
            340
        )

        draw_ring(
            frame,
            center,
            145,
            SILVER,
            1,
            0,
            300
        )

        draw_ring(
            frame,
            center,
            125,
            DEEP_NAVY,
            2
        )

        draw_ring(
            frame,
            center,
            110,
            SOFT_CYAN,
            1
        )


        # Rotating ticks

        draw_ticks(
            frame,
            center,
            180,
            72,
            SILVER,
            5,
            1,
            rotation
        )

        draw_ticks(
            frame,
            center,
            155,
            36,
            CYAN,
            7,
            1,
            -rotation * 1.5
        )


        # Central hexagon

        draw_hexagon(
            frame,
            center,
            70,
            CYAN,
            1
        )

        draw_hexagon(
            frame,
            center,
            55,
            DEEP_NAVY,
            1
        )


        # Central core

        draw_glow_circle(
            frame,
            center,
            28,
            BRIGHT_CYAN,
            2
        )

        cv2.circle(
            frame,
            center,
            7,
            WHITE,
            -1
        )


        # ====================================================
        # RADAR SWEEP
        # ====================================================

        radar_angle = (
            elapsed * 80
        ) % 360

        angle = math.radians(
            radar_angle
        )

        radar_x = int(
            center_x +
            150 * math.cos(angle)
        )

        radar_y = int(
            center_y +
            150 * math.sin(angle)
        )

        cv2.line(
            frame,
            center,
            (radar_x, radar_y),
            CYAN,
            1,
            cv2.LINE_AA
        )


        # ====================================================
        # TARGET CURSOR
        # ====================================================

        if gesture == "POINTING":

            cursor_x = int(
                smooth_cursor_x
            )

            cursor_y = int(
                smooth_cursor_y
            )

            draw_target(
                frame,
                cursor_x,
                cursor_y
            )

            draw_text(
                frame,
                "TARGET",
                (
                    cursor_x + 50,
                    cursor_y - 10
                ),
                0.4,
                BRIGHT_CYAN,
                1
            )


        # ====================================================
        # HAND LANDMARKS
        # ====================================================

        if hand is not None:

            draw_hand(
                frame,
                hand,
                width,
                height
            )


        # ====================================================
        # LEFT INFORMATION PANEL
        # ====================================================

        left_x = 40
        left_y = 135

        draw_text(
            frame,
            "SYSTEM",
            (left_x, left_y),
            0.55,
            BRIGHT_CYAN,
            1
        )

        draw_text(
            frame,
            "VISION CORE",
            (left_x, left_y + 35),
            0.40,
            WHITE,
            1
        )

        draw_text(
            frame,
            "ONLINE",
            (left_x + 120, left_y + 35),
            0.40,
            TEAL,
            1
        )

        draw_text(
            frame,
            "NEURAL LINK",
            (left_x, left_y + 60),
            0.40,
            WHITE,
            1
        )

        draw_text(
            frame,
            "ACTIVE",
            (left_x + 120, left_y + 60),
            0.40,
            TEAL,
            1
        )

        draw_text(
            frame,
            "HAND TRACK",
            (left_x, left_y + 85),
            0.40,
            WHITE,
            1
        )

        draw_text(
            frame,
            "ACTIVE" if hand else "SEARCHING",
            (left_x + 120, left_y + 85),
            0.40,
            TEAL if hand else GREY,
            1
        )


        # ====================================================
        # RIGHT INFORMATION PANEL
        # ====================================================

        right_x = width - 255
        right_y = 135

        draw_text(
            frame,
            "GESTURE ANALYSIS",
            (right_x, right_y),
            0.50,
            BRIGHT_CYAN,
            1
        )

        draw_text(
            frame,
            "INPUT",
            (right_x, right_y + 35),
            0.38,
            GREY,
            1
        )

        draw_text(
            frame,
            gesture,
            (right_x + 65, right_y + 35),
            0.38,
            WHITE,
            1
        )

        draw_text(
            frame,
            "CONFIDENCE",
            (right_x, right_y + 60),
            0.38,
            GREY,
            1
        )

        draw_text(
            frame,
            f"{confidence * 100:.1f}%",
            (right_x + 105, right_y + 60),
            0.38,
            CYAN,
            1
        )

        draw_text(
            frame,
            "ACTION",
            (right_x, right_y + 85),
            0.38,
            GREY,
            1
        )

        draw_text(
            frame,
            action,
            (right_x + 65, right_y + 85),
            0.38,
            BRIGHT_CYAN,
            1
        )


        # ====================================================
        # GESTURE CHANGE FLASH
        # ====================================================

        if time.time() < action_flash_until:

            cv2.rectangle(
                frame,
                (25, 25),
                (width - 25, height - 25),
                CYAN,
                1
            )


        # ====================================================
        # BOTTOM COMMAND SYSTEM
        # ====================================================

        command_y = height - 90

        cv2.line(
            frame,
            (40, command_y),
            (width - 40, command_y),
            DEEP_NAVY,
            1
        )

        commands = [
            ("POINT", "TARGET"),
            ("PALM", "SCAN"),
            ("FIST", "LOCK"),
            ("PEACE", "MODE"),
            ("THUMB", "CONFIRM")
        ]

        x_positions = [
            55,
            175,
            290,
            410,
            540
        ]

        for i, (gesture_name, command) in enumerate(
            commands
        ):

            draw_text(
                frame,
                gesture_name,
                (
                    x_positions[i],
                    command_y + 28
                ),
                0.35,
                CYAN,
                1
            )

            draw_text(
                frame,
                command,
                (
                    x_positions[i],
                    command_y + 48
                ),
                0.32,
                GREY,
                1
            )


        # ====================================================
        # FOOTER
        # ====================================================

        draw_text(
            frame,
            "ARIA CORE // MK-I",
            (40, height - 25),
            0.35,
            GREY,
            1
        )

        draw_text(
            frame,
            "Q : EXIT",
            (width - 85, height - 25),
            0.35,
            GREY,
            1
        )


        # ====================================================
        # SHOW
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            frame
        )


        # ====================================================
        # QUIT
        # ====================================================

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("ARIA holographic HUD stopped.")
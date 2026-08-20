import cv2
from aria_commands import ARIACommandEngine
import mediapipe as mp
import pandas as pd
import joblib
import pyautogui
import math
import time
import threading
import sounddevice as sd
import speech_recognition as sr
import webbrowser
import urllib.parse


# ============================================================
# ARIA — FINAL GESTURE + VOICE INTERFACE
# ============================================================

MODEL_PATH = "models/gesture_classifier.pkl"
LANDMARK_MODEL = "models/hand_landmarker.task"


# ============================================================
# COLORS — NAVY + RED
# ============================================================

NAVY = (180, 70, 20)
NAVY_DARK = (100, 40, 10)
RED = (80, 80, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# ============================================================
# GESTURES
# ============================================================

GESTURES = {
    0: "OPEN PALM",
    1: "FIST",
    2: "THUMBS UP",
    3: "PEACE",
    4: "POINTING"
}


# ============================================================
# SCREEN
# ============================================================

SCREEN_W, SCREEN_H = pyautogui.size()

SMOOTHING = 0.28

previous_x = SCREEN_W // 2
previous_y = SCREEN_H // 2


# ============================================================
# PINCH / CLICK SETTINGS
# ============================================================

PINCH_THRESHOLD = 0.075

LEFT_CLICK_HOLD = 0.35
DOUBLE_CLICK_HOLD = 0.35

pinch_start_time = None
thumbs_start_time = None

left_click_done = False
double_click_done = False


# ============================================================
# SCROLL SETTINGS
# ============================================================

previous_scroll_y = None

SCROLL_DEADZONE = 0.006
SCROLL_SPEED = 4

last_scroll_time = 0
SCROLL_INTERVAL = 0.08


# ============================================================
# ARIA COMMAND ENGINE
# ============================================================

command_engine = ARIACommandEngine()


# ============================================================
# SYSTEM STATE
# ============================================================

aria_paused = False
aria_running = True

status_text = "SYSTEM READY"
status_until = 0

voice_status = "VOICE STANDBY"

state_lock = threading.Lock()


# ============================================================
# MODEL
# ============================================================

model = joblib.load(MODEL_PATH)


FEATURE_NAMES = []

for i in range(1, 22):

    FEATURE_NAMES.extend([
        f"x{i}",
        f"y{i}",
        f"z{i}"
    ])


# ============================================================
# MEDIAPIPE
# ============================================================

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


landmarker = HandLandmarker.create_from_options(
    options
)


# ============================================================
# STATUS
# ============================================================

def set_status(text, duration=1.0):

    global status_text
    global status_until

    status_text = text
    status_until = time.time() + duration


# ============================================================
# VOICE STATUS
# ============================================================

def set_voice_status(text):

    global voice_status

    with state_lock:
        voice_status = text


# ============================================================
# VOICE COMMAND HANDLER
# ============================================================

def handle_voice_command(command):

    global aria_paused

    command = command.lower().strip()

    print(f"[VOICE] {command}")


    # --------------------------------------------------------
    # TOLERANT WAKE WORD
    # --------------------------------------------------------

    wake_words = [
        "aria",
        "area",
        "arya",
        "ariah"
    ]

    found_wake_word = None

    for word in wake_words:

        if word in command:

            found_wake_word = word
            break


    if found_wake_word is None:

        return


    command = command.replace(
        found_wake_word,
        "",
        1
    ).strip()


    if not command:

        set_voice_status(
            "ARIA LISTENING"
        )

        return


    # ========================================================
    # PAUSE
    # ========================================================

    if (
        "pause" in command
        or "stop system" in command
    ):

        aria_paused = True

        set_status(
            "SYSTEM PAUSED",
            2
        )

        set_voice_status(
            "VOICE: PAUSED"
        )

        return


    # ========================================================
    # RESUME
    # ========================================================

    if (
        "resume" in command
        or "continue" in command
        or "start system" in command
    ):

        aria_paused = False

        set_status(
            "SYSTEM RESUMED",
            2
        )

        set_voice_status(
            "VOICE: RESUMED"
        )

        return


    # ========================================================
    # ARIA COMMAND ENGINE
    # ========================================================
    #
    # This comes BEFORE the generic browser command so that:
    #
    # "ARIA open Safari"
    #
    # actually launches Safari through aria_commands.py.
    #
    # ========================================================

    response = command_engine.execute(command)

    if not response.startswith("I don't know that command yet"):

        set_status(
            response.upper(),
            2
        )

        set_voice_status(
            "VOICE: " + response.upper()
        )

        print("[ARIA ACTION]", response)

        return


    # ========================================================
    # OPEN BROWSER
    # ========================================================

    if (
        "open browser" in command
        or "open chrome" in command
    ):

        webbrowser.open(
            "https://www.google.com"
        )

        set_status(
            "OPENING BROWSER",
            2
        )

        set_voice_status(
            "VOICE: BROWSER"
        )

        return


    # ========================================================
    # YOUTUBE
    # ========================================================

    if "youtube" in command:

        search_text = command

        for phrase in [
            "search youtube",
            "search on youtube",
            "youtube"
        ]:

            search_text = search_text.replace(
                phrase,
                ""
            )


        search_text = search_text.strip()


        if search_text:

            encoded = urllib.parse.quote_plus(
                search_text
            )

            url = (
                "https://www.youtube.com/results?search_query="
                + encoded
            )

        else:

            url = "https://www.youtube.com"


        webbrowser.open(url)

        set_status(
            "OPENING YOUTUBE",
            2
        )

        set_voice_status(
            "VOICE: YOUTUBE"
        )

        return


    # ========================================================
    # GOOGLE SEARCH
    # ========================================================

    if (
        "search for" in command
        or "google" in command
        or "search" in command
    ):

        search_text = command

        for phrase in [
            "search for",
            "search",
            "google"
        ]:

            search_text = search_text.replace(
                phrase,
                ""
            )


        search_text = search_text.strip()


        if search_text:

            encoded = urllib.parse.quote_plus(
                search_text
            )

            url = (
                "https://www.google.com/search?q="
                + encoded
            )

            webbrowser.open(url)

            set_status(
                "GOOGLE SEARCH",
                2
            )

            set_voice_status(
                "VOICE: SEARCH"
            )

        return


    # ========================================================
    # SCROLL UP
    # ========================================================

    if (
        "scroll up" in command
        or "go up" in command
        or "move up" in command
    ):

        pyautogui.scroll(6)

        set_status(
            "VOICE SCROLL UP",
            1
        )

        set_voice_status(
            "VOICE: SCROLL UP"
        )

        return


    # ========================================================
    # SCROLL DOWN
    # ========================================================

    if (
        "scroll down" in command
        or "go down" in command
        or "move down" in command
    ):

        pyautogui.scroll(-6)

        set_status(
            "VOICE SCROLL DOWN",
            1
        )

        set_voice_status(
            "VOICE: SCROLL DOWN"
        )

        return


    # ========================================================
    # UNKNOWN
    # ========================================================

    set_status(
        "VOICE COMMAND NOT RECOGNIZED",
        1.5
    )


# ============================================================
# VOICE LISTENER
# ============================================================

def voice_listener():

    global aria_running

    recognizer = sr.Recognizer()

    # Better recognition settings
    recognizer.energy_threshold = 250

    recognizer.dynamic_energy_threshold = True

    recognizer.pause_threshold = 0.8

    recognizer.phrase_threshold = 0.25

    recognizer.non_speaking_duration = 0.5


    SAMPLE_RATE = 16000

    RECORD_SECONDS = 4


    print()
    print("[VOICE] ARIA voice system started.")
    print("[VOICE] Say: ARIA + command")
    print("[VOICE] Example: ARIA scroll down")
    print()


    while aria_running:

        try:

            set_voice_status(
                "VOICE LISTENING"
            )


            recording = sd.rec(
                int(
                    RECORD_SECONDS *
                    SAMPLE_RATE
                ),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="int16"
            )


            sd.wait()


            audio_data = sr.AudioData(
                recording.tobytes(),
                SAMPLE_RATE,
                2
            )


            set_voice_status(
                "VOICE PROCESSING"
            )


            try:

                command = recognizer.recognize_google(
                    audio_data,
                    language="en-IN"
                )

                handle_voice_command(
                    command
                )


            except sr.UnknownValueError:

                set_voice_status(
                    "VOICE LISTENING"
                )


            except sr.RequestError:

                set_voice_status(
                    "VOICE INTERNET ERROR"
                )

                time.sleep(1)


        except Exception as error:

            print(
                f"[VOICE ERROR] {error}"
            )

            set_voice_status(
                "VOICE ERROR"
            )

            time.sleep(1)


# ============================================================
# FEATURES
# ============================================================

def create_features(landmarks):

    wrist = landmarks[0]

    features = []

    for landmark in landmarks:

        features.extend([
            landmark.x - wrist.x,
            landmark.y - wrist.y,
            landmark.z - wrist.z
        ])


    return pd.DataFrame(
        [features],
        columns=FEATURE_NAMES
    )


# ============================================================
# CURSOR POSITION
# ============================================================

def get_cursor_position(landmarks):

    index_tip = landmarks[8]

    x = index_tip.x
    y = index_tip.y

    x = max(0.0, min(1.0, x))
    y = max(0.0, min(1.0, y))


    return (
        int(x * SCREEN_W),
        int(y * SCREEN_H)
    )


# ============================================================
# SMOOTH CURSOR
# ============================================================

def smooth_cursor(x, y):

    global previous_x
    global previous_y

    previous_x = int(
        previous_x +
        (x - previous_x) * SMOOTHING
    )

    previous_y = int(
        previous_y +
        (y - previous_y) * SMOOTHING
    )


    return previous_x, previous_y


# ============================================================
# PINCH
# ============================================================

def get_pinch_distance(landmarks):

    thumb = landmarks[4]

    index = landmarks[8]


    return math.sqrt(
        (thumb.x - index.x) ** 2 +
        (thumb.y - index.y) ** 2
    )


# ============================================================
# HUD
# ============================================================

def draw_hud(
    frame,
    gesture,
    hand_detected,
    pinch_active
):

    h, w, _ = frame.shape


    overlay = frame.copy()


    cv2.rectangle(
        overlay,
        (0, 0),
        (w, 145),
        BLACK,
        -1
    )


    frame = cv2.addWeighted(
        overlay,
        0.62,
        frame,
        0.38,
        0
    )


    # ========================================================
    # TITLE
    # ========================================================

    cv2.putText(
        frame,
        "ARIA",
        (30, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.05,
        RED,
        2,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        "ADVANCED REMOTE INTERFACE ASSISTANT",
        (30, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        NAVY,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # GESTURE STATUS
    # ========================================================

    if aria_paused:

        gesture_display = "SYSTEM PAUSED"

    elif hand_detected:

        gesture_display = gesture

    else:

        gesture_display = "NO HAND"


    cv2.putText(
        frame,
        gesture_display,
        (w - 330, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        RED if aria_paused else WHITE,
        2,
        cv2.LINE_AA
    )


    # ========================================================
    # VOICE STATUS
    # ========================================================

    with state_lock:

        current_voice_status = voice_status


    cv2.putText(
        frame,
        current_voice_status,
        (w - 330, 72),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        NAVY,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # SYSTEM STATUS
    # ========================================================

    if time.time() < status_until:

        cv2.putText(
            frame,
            status_text,
            (30, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            RED,
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # CONTROLS
    # ========================================================

    cv2.putText(
        frame,
        "POINT=MOVE   PINCH=LEFT CLICK   PEACE=SCROLL",
        (30, h - 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        NAVY,
        1,
        cv2.LINE_AA
    )


    cv2.putText(
        frame,
        "THUMBS=DOUBLE CLICK   P=PAUSE   Q=EXIT",
        (30, h - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        RED,
        1,
        cv2.LINE_AA
    )


    # ========================================================
    # CORNER MARKERS
    # ========================================================

    length = 20


    cv2.line(
        frame,
        (0, 0),
        (length, 0),
        RED,
        2
    )


    cv2.line(
        frame,
        (0, 0),
        (0, length),
        RED,
        2
    )


    cv2.line(
        frame,
        (w - length, 0),
        (w, 0),
        RED,
        2
    )


    cv2.line(
        frame,
        (w, 0),
        (w, length),
        RED,
        2
    )


    return frame


# ============================================================
# START VOICE THREAD
# ============================================================

voice_thread = threading.Thread(
    target=voice_listener,
    daemon=True
)

voice_thread.start()


# ============================================================
# CAMERA
# ============================================================

cap = cv2.VideoCapture(0)


if not cap.isOpened():

    print("ERROR: Camera not available.")

    aria_running = False

    exit()


# ============================================================
# START MESSAGE
# ============================================================

print()
print("================================================")
print("                  ARIA ONLINE")
print("================================================")
print("GESTURE + VOICE SYSTEM ACTIVE")
print()
print("VOICE EXAMPLES:")
print("  ARIA open browser")
print("  ARIA open Safari")
print("  ARIA open VS Code")
print("  ARIA open Terminal")
print("  ARIA search YouTube cats")
print("  ARIA scroll down")
print("  ARIA scroll up")
print("  ARIA pause system")
print("  ARIA resume system")
print()
print("GESTURES:")
print("  POINTING     -> MOVE CURSOR")
print("  PINCH        -> LEFT CLICK")
print("  PEACE        -> SCROLL")
print("  THUMBS UP    -> DOUBLE CLICK")
print()
print("P = pause/resume")
print("Q = exit")
print("================================================")
print()


# ============================================================
# MAIN LOOP
# ============================================================

timestamp = 0


while True:

    ret, frame = cap.read()


    if not ret:

        break


    # No horizontal mirror effect


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


    gesture = "UNKNOWN"

    hand_detected = False

    pinch_active = False


    # ========================================================
    # HAND DETECTED
    # ========================================================

    if result.hand_landmarks:

        hand_detected = True

        landmarks = result.hand_landmarks[0]


        # ----------------------------------------------------
        # LANDMARKS
        # ----------------------------------------------------

        for landmark in landmarks:

            px = int(
                landmark.x *
                frame.shape[1]
            )

            py = int(
                landmark.y *
                frame.shape[0]
            )


            cv2.circle(
                frame,
                (px, py),
                3,
                NAVY,
                -1
            )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        features = create_features(
            landmarks
        )


        prediction = model.predict(
            features
        )


        label = int(
            prediction[0]
        )


        gesture = GESTURES.get(
            label,
            "UNKNOWN"
        )


        # ----------------------------------------------------
        # PINCH
        # ----------------------------------------------------

        distance = get_pinch_distance(
            landmarks
        )


        pinch_active = (
            distance < PINCH_THRESHOLD
        )


        # ====================================================
        # ACTIONS
        # ====================================================

        if not aria_paused:


            # =================================================
            # POINTING = MOVE CURSOR
            # =================================================

            if label == 4:

                cursor_x, cursor_y = (
                    get_cursor_position(
                        landmarks
                    )
                )


                cursor_x, cursor_y = (
                    smooth_cursor(
                        cursor_x,
                        cursor_y
                    )
                )


                pyautogui.moveTo(
                    cursor_x,
                    cursor_y,
                    duration=0
                )


            # =================================================
            # PEACE = SCROLL
            # =================================================

            if label == 3:

                current_y = landmarks[8].y


                if previous_scroll_y is not None:

                    movement = (
                        current_y
                        -
                        previous_scroll_y
                    )


                    now = time.time()


                    # MOVE HAND UP
                    if (
                        movement
                        <
                        -SCROLL_DEADZONE
                    ):

                        if (
                            now
                            -
                            last_scroll_time
                            >
                            SCROLL_INTERVAL
                        ):

                            pyautogui.scroll(
                                SCROLL_SPEED
                            )

                            last_scroll_time = now

                            set_status(
                                "SCROLL UP",
                                0.15
                            )


                    # MOVE HAND DOWN
                    elif (
                        movement
                        >
                        SCROLL_DEADZONE
                    ):

                        if (
                            now
                            -
                            last_scroll_time
                            >
                            SCROLL_INTERVAL
                        ):

                            pyautogui.scroll(
                                -SCROLL_SPEED
                            )

                            last_scroll_time = now

                            set_status(
                                "SCROLL DOWN",
                                0.15
                            )


                previous_scroll_y = current_y


            else:

                previous_scroll_y = None


            # =================================================
            # LEFT CLICK — PINCH
            # =================================================

            if (
                pinch_active
                and label != 3
                and label != 2
            ):

                if pinch_start_time is None:

                    pinch_start_time = time.time()


                held = (
                    time.time()
                    -
                    pinch_start_time
                )


                if (
                    held >= LEFT_CLICK_HOLD
                    and not left_click_done
                ):

                    pyautogui.click()

                    left_click_done = True

                    set_status(
                        "LEFT CLICK!",
                        0.8
                    )


            else:

                pinch_start_time = None

                left_click_done = False


            # =================================================
            # THUMBS UP = DOUBLE CLICK
            # =================================================

            if label == 2:

                if thumbs_start_time is None:

                    thumbs_start_time = time.time()


                held = (
                    time.time()
                    -
                    thumbs_start_time
                )


                if (
                    held >= DOUBLE_CLICK_HOLD
                    and not double_click_done
                ):

                    pyautogui.doubleClick(
                        interval=0.12
                    )

                    double_click_done = True

                    set_status(
                        "DOUBLE CLICK!",
                        0.8
                    )


            else:

                thumbs_start_time = None

                double_click_done = False


    else:

        # ====================================================
        # NO HAND
        # ====================================================

        pinch_start_time = None

        thumbs_start_time = None

        left_click_done = False

        double_click_done = False

        previous_scroll_y = None


    # ========================================================
    # HUD
    # ========================================================

    frame = draw_hud(
        frame,
        gesture,
        hand_detected,
        pinch_active
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        "ARIA — Gesture + Voice",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


    elif key == ord("p"):

        aria_paused = not aria_paused


        if aria_paused:

            set_status(
                "SYSTEM PAUSED",
                2
            )

        else:

            set_status(
                "SYSTEM RESUMED",
                2
            )


# ============================================================
# CLEANUP
# ============================================================

aria_running = False

cap.release()

cv2.destroyAllWindows()

landmarker.close()


print()
print("ARIA stopped safely.")
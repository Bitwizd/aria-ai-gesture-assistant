## What ARIA Can Do


The goal of ARIA is simple: **control parts of your computer naturally, without constantly reaching for the keyboard or mouse.**




ARIA is a real-time hands-free computer interaction system that combines voice commands with hand gesture recognition. You can speak commands such as **“open Safari,” “open Notes,” or “open Terminal,”** and ARIA will recognize the command and perform the action. Through the camera, it also tracks your hand and recognizes gestures such as thumbs up, thumbs down, peace, fist, pointing, and pinch, along with hand movements for scrolling and interaction.

### Supported Interactions

| Input               | What ARIA Does                |
| ------------------- | ----------------------------- |
| 🎙️ “Open Safari”   | Opens Safari                  |
| 🎙️ “Open Notes”    | Opens Notes                   |
| 🎙️ “Open Terminal” | Opens Terminal                |
| 🎙️ “Close Safari”  | Closes Safari                 |
| 👍 Thumbs Up        | Recognizes gesture            |
| 👎 Thumbs Down      | Recognizes gesture            |
| ✌️ Peace            | Recognizes gesture            |
| ✊ Fist              | Recognizes gesture            |
| ☝️ Pointing         | Recognizes gesture            |
| 🤏 Pinch            | Recognizes gesture            |
| 🖐️ Hand movement   | Enables interaction/scrolling |
| ⬆️ Move hand up     | Scroll up                     |
| ⬇️ Move hand down   | Scroll down                   |


# 🧠 How ARIA Works

ARIA follows a real-time computer vision pipeline:

```text
📷 WEBCAM
    ↓
👁️ HAND DETECTION
    ↓
📍 21 HAND LANDMARKS
    ↓
🔢 FEATURE EXTRACTION
    ↓
🧠 ML CLASSIFIER
    ↓
🎯 GESTURE RECOGNITION
    ↓
⚡ ARIA COMMAND
    ↓
🖥️ ACTION + HUD


---

## 📁 Project Structure

```text
aria-ai-gesture-assistant/
│
├── 📂 dataset/
│   └── hand_gestures.csv
│
├── 📂 models/
│   ├── gesture_classifier.pkl
│   └── hand_landmarker.task
│
├── 📂 src/
│   ├── aria_commands.py
│   ├── aria_hud.py
│   ├── camera_pipeline.py
│   ├── gesture_app.py
│   ├── gesture_effects.py
│   ├── predict_live.py
│   ├── train_model.py
│   ├── webcam_test.py
│   ├── pinch_test.py
│   ├── right_click_test.py
│   └── scroll_test.py
│
├── 📂 archive/
│   └── aria_backups/
│
├── 📄 requirements.txt
├── 📄 README.md
└── 📄 .gitignore

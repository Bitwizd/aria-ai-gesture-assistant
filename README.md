# 🤖 ARIA — AI Gesture & Voice Assistant

### A hands-free, multimodal interface powered by Computer Vision, Machine Learning & Voice

> **ARIA lets you interact with a computer using your hands and voice — turning a webcam into an intelligent interface.**

<p align="center">

👋 Gesture Recognition · 🎙️ Voice Interaction · 👁️ Computer Vision · 🧠 Machine Learning · ⚡ Real-Time HUD

</p>

---

## ✨ What is ARIA?

ARIA is a real-time **human-computer interaction system** that combines computer vision, machine learning, gesture recognition, and voice interaction into one interface.

Instead of relying only on a mouse and keyboard:

**👋 Show a gesture → 🧠 ARIA recognizes it → ⚡ ARIA performs an action**

**🎙️ Speak a command → 🧠 ARIA processes it → 💬 ARIA responds**

The project explores how **vision + voice can create more natural ways to interact with computers.**

---

## 🚀 What ARIA Can Do

| Capability | What it does |
|---|---|
| 👋 **Hand Tracking** | Detects hand landmarks in real time |
| 🧠 **Gesture Recognition** | Classifies custom hand gestures using ML |
| 🎙️ **Voice Interaction** | Accepts spoken commands |
| 🖥️ **Interactive HUD** | Displays real-time ARIA feedback |
| 🖱️ **Gesture Controls** | Enables gesture-driven interaction |
| 📜 **Scrolling** | Supports gesture-based scrolling |
| ✨ **Visual Effects** | Provides futuristic interface feedback |
| 🛡️ **Safety Layer** | Controls command execution |

---

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
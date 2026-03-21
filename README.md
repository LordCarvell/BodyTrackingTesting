# Full Body Tracker

A real-time body tracking tool for webcam footage. Built with Python, OpenCV, and MediaPipe. It overlays a full skeleton on your body, tracks every finger joint on both hands, maps a 468-point mesh across your face, and blurs detected faces all at the same time.

> **Author:** LordCarvell

---

## Features

* **Full body skeleton** - tracks all 33 MediaPipe pose landmarks including shoulders, elbows, wrists, hips, knees, ankles, heels, toes, and facial anchor points
* **Hand tracking** - 21 landmarks per hand covering every finger knuckle and tip, for both hands simultaneously. Left hand is drawn in green, right hand in orange
* **Face mesh** - 468-point facial landmark mesh with tessellation and contour highlights covering eyes, eyebrows, nose, lips, cheeks and forehead
* **Face blur** - automatically detects and blurs faces using Haar cascade detection. Blur strength is adjustable on the fly
* **Toggle controls** - each layer (skeleton, hands, face mesh, blur) can be switched on and off independently while the app is running
* **Fullscreen mode** - one key press to go fullscreen and back
* **Status overlay** - live readout of face count, blur strength, and which layers are currently active

---

## Requirements

* Python 3.9 or newer
* pip packages:

```
mediapipe
opencv-python
```

---

## Installation

**1. Clone or download the repo**

```
git clone https://github.com/LordCarvell/Body-Tracker.git
cd body-tracker
```

Or just download `main.py` on its own.

**2. Install the dependencies**

```
pip install mediapipe opencv-python
```

**3. Run it**

```
python main.py
```

On first run it will download three model files into the same folder as the script. About 64 MB total and only happens once.

---

## How to Use

1. Run the script. Models download automatically if not already present
2. Point your webcam at yourself
3. The skeleton, hand tracking, and face mesh will appear straight away
4. Use the keyboard controls below to adjust what is shown

---

## Controls

| Key | Action |
|-----|--------|
| `q` | Quit |
| `f` | Toggle fullscreen |
| `b` | Toggle face blur on/off |
| `s` | Toggle body skeleton on/off |
| `h` | Toggle hand tracking on/off |
| `m` | Toggle face mesh on/off |
| `+` / `=` | Increase blur strength |
| `-` | Decrease blur strength |

---

## Project Structure

```
body-tracker/
├── main.py          # whole application in one file
├── pose_landmarker_heavy.task    # downloaded automatically on first run
├── hand_landmarker.task          # downloaded automatically on first run
├── face_landmarker.task          # downloaded automatically on first run
└── README.md
```

---

## Tracking Layers

| Layer | Points | Colour |
|-------|--------|--------|
| Body skeleton | 33 landmarks | Red |
| Left hand | 21 landmarks | Green |
| Right hand | 21 landmarks | Orange |
| Face mesh | 468 landmarks | Cyan |

---

## Known Issues

* **First run is slow** - the three model files are about 64 MB and get downloaded on first run. After that startup is instant
* **Face blur uses Haar cascade** - the blur relies on OpenCV's frontal face detector which can miss faces at sharp angles or in poor lighting. The face mesh layer uses a separate MediaPipe model and is generally more robust
* **Hand tracking drops at distance** - hands further than roughly a metre from the camera may not be detected reliably
* **Heavy model needs a decent CPU** - the pose landmarker uses the heavy model for better accuracy. On slower machines you may notice frame rate drops if all three layers are active at once. Toggling off the layers you do not need will help
* **mp.solutions is not used** - older MediaPipe tutorials use `mp.solutions.holistic` which was removed in newer versions. This script uses only the Tasks API so it works fine with current MediaPipe

---

## Roadmap

* Configurable visibility threshold per layer
* Save annotated video to file
* Multi-person pose tracking
* Pixel segmentation / background removal
* Joint angle readouts (elbow, knee, shoulder)
* Pose comparison / mirroring mode

---

## Built With

* [MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide) - pose, hand, and face landmark models
* [OpenCV](https://opencv.org/) - camera capture, drawing, and face detection

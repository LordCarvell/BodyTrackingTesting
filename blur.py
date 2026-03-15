# SECTION 1 - IMPORTS #

import cv2
import sys
import urllib.request
import os

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# SECTION 2 - MODEL URLS / PATHS #

POSE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
)
POSE_MODEL_PATH = "pose_landmarker_heavy.task"

HAND_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_MODEL_PATH = "hand_landmarker.task"

FACE_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
FACE_MODEL_PATH = "face_landmarker.task"


def download_model(url, path, label):
    if not os.path.exists(path):
        print(f"Downloading {label}...")
        urllib.request.urlretrieve(url, path)
        print(f"  Done: {path}")


# SECTION 3 - SKELETON CONNECTIONS #

# Body: all 33 MediaPipe pose landmarks fully connected
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # Neck
    (0, 11), (0, 12),
    # Shoulders
    (11, 12),
    # Left arm
    (11, 13), (13, 15),
    (15, 17), (15, 19), (15, 21),
    (17, 19),
    # Right arm
    (12, 14), (14, 16),
    (16, 18), (16, 20), (16, 22),
    (18, 20),
    # Torso
    (11, 23), (12, 24), (23, 24),
    # Left leg
    (23, 25), (25, 27),
    (27, 29), (29, 31), (27, 31),
    # Right leg
    (24, 26), (26, 28),
    (28, 30), (30, 32), (28, 32),
]

# Hand: 21 landmarks, finger-by-finger connections
HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm
    (5, 9), (9, 13), (13, 17),
]

# Face mesh: key contour index pairs (eyes, lips, face oval, irises)
FACE_CONTOUR_CONNECTIONS = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109,
]

FACE_MESH_CONNECTIONS = [
    # Right eye
    (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),
    (155,133),(33,246),(246,161),(161,160),(160,159),(159,158),(158,157),
    (157,173),(173,133),
    # Left eye
    (362,382),(382,381),(381,380),(380,374),(374,373),(373,390),(390,249),
    (249,263),(362,398),(398,384),(384,385),(385,386),(386,387),(387,388),
    (388,466),(466,263),
    # Right eyebrow
    (46,53),(53,52),(52,65),(65,55),(70,63),(63,105),(105,66),(66,107),
    # Left eyebrow
    (276,283),(283,282),(282,295),(295,285),(300,293),(293,334),(334,296),(296,336),
    # Nose
    (168,6),(6,197),(197,195),(195,5),(5,4),(4,1),(1,19),(19,94),(94,2),
    (98,97),(97,2),(2,326),(326,327),(327,294),
    (44,1),(1,274),
    # Lips outer
    (61,146),(146,91),(91,181),(181,84),(84,17),(17,314),(314,405),(405,321),
    (321,375),(375,291),(61,185),(185,40),(40,39),(39,37),(37,0),(0,267),
    (267,269),(269,270),(270,409),(409,291),
    # Lips inner
    (78,95),(95,88),(88,178),(178,87),(87,14),(14,317),(317,402),(402,318),
    (318,324),(324,308),(78,191),(191,80),(80,81),(81,82),(82,13),(13,312),
    (312,311),(311,310),(310,415),(415,308),
    # Cheeks / jaw
    (127,34),(34,139),(139,156),(156,143),(143,111),(111,117),(117,118),
    (118,119),(119,120),(120,121),(121,128),(128,245),
    (356,454),(454,323),(323,361),(361,288),(288,397),(397,365),(365,379),
    (379,378),(378,400),(400,377),(377,152),
    # Forehead
    (10,151),(151,9),(9,8),(8,168),
]

# Colors (BGR)
BODY_COLOR      = (0,   0, 220)
JOINT_COLOR     = (0,   0, 255)
L_HAND_COLOR    = (0, 220,   0)
R_HAND_COLOR    = (0, 160, 255)
FACE_MESH_COLOR = (220, 220,  0)
FACE_CONT_COLOR = (255, 255, 100)

LINE_THICKNESS  = 2
JOINT_RADIUS    = 5
VIS_THRESHOLD   = 0.4


# SECTION 4 - DRAW BODY SKELETON #

def draw_skeleton(frame, detection_result):
    if not detection_result or not detection_result.pose_landmarks:
        return
    h, w = frame.shape[:2]
    for pose_landmarks in detection_result.pose_landmarks:
        for (a, b) in POSE_CONNECTIONS:
            if a >= len(pose_landmarks) or b >= len(pose_landmarks):
                continue
            pa, pb = pose_landmarks[a], pose_landmarks[b]
            if pa.visibility < VIS_THRESHOLD or pb.visibility < VIS_THRESHOLD:
                continue
            x1, y1 = int(pa.x * w), int(pa.y * h)
            x2, y2 = int(pb.x * w), int(pb.y * h)
            cv2.line(frame, (x1, y1), (x2, y2), BODY_COLOR, LINE_THICKNESS, cv2.LINE_AA)
        for idx in set(i for pair in POSE_CONNECTIONS for i in pair):
            if idx >= len(pose_landmarks):
                continue
            p = pose_landmarks[idx]
            if p.visibility < VIS_THRESHOLD:
                continue
            cx, cy = int(p.x * w), int(p.y * h)
            cv2.circle(frame, (cx, cy), JOINT_RADIUS, JOINT_COLOR, -1, cv2.LINE_AA)


# SECTION 5 - DRAW HANDS #

def draw_hand(frame, hand_landmarks_list, handedness_list):
    if not hand_landmarks_list:
        return
    h, w = frame.shape[:2]
    for i, hand_lms in enumerate(hand_landmarks_list):
        side = "Right"
        if handedness_list and i < len(handedness_list):
            side = handedness_list[i][0].display_name
        color = L_HAND_COLOR if side == "Left" else R_HAND_COLOR
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in hand_lms]
        for (a, b) in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], color, LINE_THICKNESS, cv2.LINE_AA)
        for pt in pts:
            cv2.circle(frame, pt, 4, color, -1, cv2.LINE_AA)


# SECTION 6 - DRAW FACE MESH #

def draw_face_mesh(frame, face_result):
    if not face_result or not face_result.face_landmarks:
        return
    h, w = frame.shape[:2]
    for face_lms in face_result.face_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_lms]
        for (a, b) in FACE_MESH_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b], FACE_MESH_COLOR, 1, cv2.LINE_AA)
        for idx in FACE_CONTOUR_CONNECTIONS:
            if idx < len(pts):
                cv2.circle(frame, pts[idx], 2, FACE_CONT_COLOR, -1, cv2.LINE_AA)


# SECTION 7 - FACE BLUR #

def apply_face_blur(frame, faces, blur_strength=99):
    for (x, y, w, h) in faces:
        padding = int(0.2 * w)
        x1 = max(0, x - padding)
        y1 = max(0, y - padding)
        x2 = min(frame.shape[1], x + w + padding)
        y2 = min(frame.shape[0], y + h + padding)
        face_region = frame[y1:y2, x1:x2]
        k = blur_strength if blur_strength % 2 == 1 else blur_strength + 1
        blurred = cv2.GaussianBlur(face_region, (k, k), 0)
        frame[y1:y2, x1:x2] = blurred
    return frame


# SECTION 8 - MAIN #

def main():
    download_model(POSE_MODEL_URL, POSE_MODEL_PATH, "Pose model (heavy, ~25 MB)")
    download_model(HAND_MODEL_URL, HAND_MODEL_PATH, "Hand model (~9 MB)")
    download_model(FACE_MODEL_URL, FACE_MODEL_PATH, "Face mesh model (~30 MB)")

    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    if face_cascade.empty():
        print("Error: Could not load face cascade classifier.")
        sys.exit(1)

    pose_landmarker = mp_vision.PoseLandmarker.create_from_options(
        mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=POSE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    hand_landmarker = mp_vision.HandLandmarker.create_from_options(
        mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    face_landmarker = mp_vision.FaceLandmarker.create_from_options(
        mp_vision.FaceLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        sys.exit(1)

    WIN_NAME = "Full Body Tracker"
    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN_NAME, 1280, 720)

    print("\nRunning. Controls:")
    print("  q      - Quit")
    print("  f      - Toggle fullscreen")
    print("  +/-    - Increase/decrease blur strength")
    print("  b      - Toggle face blur on/off")
    print("  s      - Toggle body skeleton on/off")
    print("  h      - Toggle hand tracking on/off")
    print("  m      - Toggle face mesh on/off\n")

    blur_strength    = 99
    blur_enabled     = True
    skeleton_enabled = True
    hands_enabled    = True
    mesh_enabled     = True
    fullscreen       = False
    frame_index      = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        frame_index += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        pose_result = None
        hand_result = None
        face_result = None

        if skeleton_enabled:
            pose_result = pose_landmarker.detect_for_video(mp_image, frame_index)
        if hands_enabled:
            hand_result = hand_landmarker.detect_for_video(mp_image, frame_index)
        if mesh_enabled:
            face_result = face_landmarker.detect_for_video(mp_image, frame_index)

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )

        if skeleton_enabled and pose_result:
            draw_skeleton(frame, pose_result)
        if mesh_enabled and face_result:
            draw_face_mesh(frame, face_result)
        if hands_enabled and hand_result:
            draw_hand(frame, hand_result.hand_landmarks, hand_result.handedness)

        if blur_enabled and len(faces) > 0:
            frame = apply_face_blur(frame, faces, blur_strength)
        elif not blur_enabled:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        hand_count = len(hand_result.hand_landmarks) if hand_result else 0
        status = (
            f"Faces:{len(faces)}  "
            f"Blur:{'ON' if blur_enabled else 'OFF'}({blur_strength})  "
            f"Body:{'ON' if skeleton_enabled else 'OFF'}  "
            f"Hands:{'ON' if hands_enabled else 'OFF'}({hand_count})  "
            f"Mesh:{'ON' if mesh_enabled else 'OFF'}"
        )
        cv2.putText(frame, status, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        cv2.imshow(WIN_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key in (ord("+"), ord("=")):
            blur_strength = min(151, blur_strength + 10)
            print(f"Blur strength: {blur_strength}")
        elif key == ord("-"):
            blur_strength = max(11, blur_strength - 10)
            print(f"Blur strength: {blur_strength}")
        elif key == ord("b"):
            blur_enabled = not blur_enabled
            print(f"Blur {'enabled' if blur_enabled else 'disabled'}")
        elif key == ord("s"):
            skeleton_enabled = not skeleton_enabled
            print(f"Skeleton {'enabled' if skeleton_enabled else 'disabled'}")
        elif key == ord("h"):
            hands_enabled = not hands_enabled
            print(f"Hand tracking {'enabled' if hands_enabled else 'disabled'}")
        elif key == ord("m"):
            mesh_enabled = not mesh_enabled
            print(f"Face mesh {'enabled' if mesh_enabled else 'disabled'}")
        elif key == ord("f"):
            fullscreen = not fullscreen
            prop = cv2.WND_PROP_FULLSCREEN
            mode = cv2.WINDOW_FULLSCREEN if fullscreen else cv2.WINDOW_NORMAL
            cv2.setWindowProperty(WIN_NAME, prop, mode)
            print(f"Fullscreen {'ON' if fullscreen else 'OFF'}")

    cap.release()
    pose_landmarker.close()
    hand_landmarker.close()
    face_landmarker.close()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
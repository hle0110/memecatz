"""
Everything that reads signals off the webcam frame: the 7-class emotion CNN,
the face analyzer (52-point blendshapes), and hand gesture recognition.
Combined into one module since they're all "look at the frame and produce
tags/scores". mood.py is what actually combines their output into a final mood.

Face and hands use MediaPipe's tasks API, which every mediapipe release from
0.10.21 on provides. The older mp.solutions API was removed in 0.10.30, so
nothing here depends on it.
"""

import os
import json
import math
import time
import cv2
import numpy as np
import mediapipe as mp
import requests

def _load_tflite_interpreter():
    """
    The emotion CNN only needs a TFLite interpreter. ai-edge-litert is the small
    maintained one (about 15MB, works on Mac, Windows and Linux). tflite-runtime
    and tensorflow are accepted too if either is already installed. Only
    EmotionDetector needs this, FaceAnalyzer and HandGestureRecognizer must keep
    working without any of them.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        from tflite_runtime.interpreter import Interpreter
        return Interpreter
    except ImportError:
        pass
    try:
        import tensorflow
        return tensorflow.lite.Interpreter
    except ImportError:
        return None


TFLiteInterpreter = _load_tflite_interpreter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # project root (this file lives at the top level)
ASSETS_DIR = os.path.join(BASE_DIR, "assets")                    # bundled with the code
MODEL_CACHE_DIR = os.path.join(BASE_DIR, "model_cache")          # downloaded once on first run, cached after that


def _dist(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


# ============================================================================
# Emotion CNN (7-class FER, bundled TFLite model)
# ============================================================================

EMOTION_PADDING = 40
EMOTION_TARGET_SIZE = (64, 64)
EMOTION_OFFSET_X = 10
EMOTION_OFFSET_Y = 10

EMOTION_LABELS = {
    0: "angry",
    1: "disgust",
    2: "fear",
    3: "happy",
    4: "sad",
    5: "surprise",
    6: "neutral",
}

EMOTION_MODEL_PATH = os.path.join(ASSETS_DIR, "emotion_model_quantized.tflite")


class EmotionDetector:
    def __init__(self, model_path=EMOTION_MODEL_PATH):
        if TFLiteInterpreter is None:
            raise ImportError("a TFLite runtime is required for the emotion CNN (pip install ai-edge-litert)")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"emotion model not found at {model_path}")

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("failed to load the haar cascade face detector")

        self.interpreter = TFLiteInterpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    @staticmethod
    def _to_square(box):
        x, y, w, h = box
        if h > w:
            diff = h - w
            x -= diff // 2
            w += diff
        elif w > h:
            diff = w - h
            y -= diff // 2
            h += diff
        return int(x), int(y), int(w), int(h)

    @staticmethod
    def _pad(gray):
        row, col = gray.shape[:2]
        bottom = gray[row - 2:row, 0:col]
        mean = cv2.mean(bottom)[0]
        return cv2.copyMakeBorder(gray, EMOTION_PADDING, EMOTION_PADDING, EMOTION_PADDING, EMOTION_PADDING,
                                   cv2.BORDER_CONSTANT, value=[mean])

    def find_faces(self, frame_bgr):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50), flags=cv2.CASCADE_SCALE_IMAGE
        )
        return gray, faces

    def detect(self, frame_bgr):
        gray, faces = self.find_faces(frame_bgr)
        if len(faces) == 0:
            return None

        face_box = max(faces, key=lambda box: box[2] * box[3])
        x, y, w, h = self._to_square(face_box)
        padded_gray = self._pad(gray)

        x1 = x - EMOTION_OFFSET_X + EMOTION_PADDING
        x2 = x + w + EMOTION_OFFSET_X + EMOTION_PADDING
        y1 = y - EMOTION_OFFSET_Y + EMOTION_PADDING
        y2 = y + h + EMOTION_OFFSET_Y + EMOTION_PADDING
        x1 = max(0, x1)
        y1 = max(0, y1)

        face_crop = padded_gray[y1:y2, x1:x2]
        if face_crop.size == 0:
            return None

        face_resized = cv2.resize(face_crop, EMOTION_TARGET_SIZE).astype("float32")
        face_norm = (face_resized / 255.0 - 0.5) * 2.0
        face_input = np.expand_dims(np.expand_dims(face_norm, -1), 0).astype("float32")

        self.interpreter.set_tensor(self.input_details[0]["index"], face_input)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details[0]["index"])[0]

        top_index = int(np.argmax(output))
        emotion = EMOTION_LABELS[top_index]
        confidence = float(output[top_index])

        return {
            "emotion": emotion,
            "confidence": confidence,
            "box": (int(face_box[0]), int(face_box[1]), int(face_box[2]), int(face_box[3])),
            "scores": {EMOTION_LABELS[i]: float(score) for i, score in enumerate(output)},
        }


# ============================================================================
# Model downloads (face and hand landmark models, cached after the first run)
# ============================================================================

MODEL_DOWNLOAD_TIMEOUT_SECONDS = 20
DOWNLOAD_RETRY_COOLDOWN_SECONDS = 3600
DOWNLOAD_STATUS_PATH = os.path.join(MODEL_CACHE_DIR, "download_status.json")

# Pinned model versions, so a new upload can never change behaviour silently.
BLENDSHAPE_MODEL_PATH = os.path.join(MODEL_CACHE_DIR, "face_landmarker.task")
BLENDSHAPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
HAND_MODEL_PATH = os.path.join(MODEL_CACHE_DIR, "hand_landmarker.task")
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


def _read_download_status():
    if not os.path.isfile(DOWNLOAD_STATUS_PATH):
        return {}
    try:
        with open(DOWNLOAD_STATUS_PATH, "r") as handle:
            data = json.load(handle)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _write_download_status(status):
    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        with open(DOWNLOAD_STATUS_PATH, "w") as handle:
            json.dump(status, handle)
    except OSError:
        pass


def ensure_model(path, url):
    """Downloads a model file once. Returns True if it is available locally.

    A failed download is not retried for an hour, so an offline start stays fast.
    """
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return True

    key = os.path.basename(path)
    status = _read_download_status()
    last_attempt = status.get(key, {}).get("last_attempt", 0) if isinstance(status.get(key), dict) else 0
    if (time.time() - last_attempt) < DOWNLOAD_RETRY_COOLDOWN_SECONDS:
        return False

    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        response = requests.get(url, timeout=MODEL_DOWNLOAD_TIMEOUT_SECONDS, stream=True)
        response.raise_for_status()
        tmp_path = path + ".part"
        with open(tmp_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 16):
                if chunk:
                    handle.write(chunk)
        os.replace(tmp_path, path)
        status[key] = {"last_attempt": time.time(), "last_success": True}
        _write_download_status(status)
        return True
    except (requests.RequestException, OSError):
        status[key] = {"last_attempt": time.time(), "last_success": False}
        _write_download_status(status)
        return False


def ensure_blendshape_model():
    return ensure_model(BLENDSHAPE_MODEL_PATH, BLENDSHAPE_MODEL_URL)


def ensure_hand_model():
    return ensure_model(HAND_MODEL_PATH, HAND_MODEL_URL)


def _to_mp_image(frame_bgr):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


# ============================================================================
# Face analysis: 52-point blendshapes
# ============================================================================

CALIBRATION_FRAMES = 12
BASELINE_DRIFT_ALPHA = 0.01
BLENDSHAPE_DRIFT_THRESHOLD = 0.07
MAX_GUEST_FACES = 3
GUEST_MATCH_DISTANCE_RATIO = 0.6
GUEST_STALE_SECONDS = 2.0

BLENDSHAPE_NAMES = (
    "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
    "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
    "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
    "jawOpen", "jawForward", "jawLeft", "jawRight",
    "mouthClose", "mouthFunnel", "mouthPucker",
    "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
    "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
    "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
    "mouthUpperUpLeft", "mouthUpperUpRight", "mouthShrugLower", "mouthShrugUpper",
    "mouthRollLower", "mouthRollUpper", "mouthLeft", "mouthRight",
    "noseSneerLeft", "noseSneerRight",
)


class _BlendshapeBackend:
    engine_name = "blendshapes"

    def __init__(self, num_faces=1, min_detection_confidence=0.5):
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core.base_options import BaseOptions

        options = mp_vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=BLENDSHAPE_MODEL_PATH),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=num_faces,
            min_face_detection_confidence=min_detection_confidence,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def process(self, frame_bgr):
        height, width = frame_bgr.shape[:2]
        result = self._landmarker.detect(_to_mp_image(frame_bgr))
        return self.parse_result(result, width, height)

    @staticmethod
    def parse_result(result, width, height):
        if not result.face_blendshapes or not result.face_landmarks:
            return None

        faces = []
        count = min(len(result.face_blendshapes), len(result.face_landmarks))
        for i in range(count):
            scores = {category.category_name: category.score for category in result.face_blendshapes[i]}
            raw_features = {name: scores.get(name, 0.0) for name in BLENDSHAPE_NAMES}

            landmarks = result.face_landmarks[i]
            xs = [lm.x * width for lm in landmarks]
            ys = [lm.y * height for lm in landmarks]
            x1 = max(0, int(min(xs)) - 10)
            y1 = max(0, int(min(ys)) - 10)
            x2 = min(width, int(max(xs)) + 10)
            y2 = min(height, int(max(ys)) + 10)
            box = (x1, y1, x2 - x1, y2 - y1)

            faces.append({"raw_features": raw_features, "box": box})

        return faces if faces else None

    def close(self):
        self._landmarker.close()


class FaceAnalyzer:
    """Blendshape face analysis with per-person calibration.

    If the face model cannot be downloaded, engine is None and analyze()
    returns None. The rest of the app keeps running on the emotion model and
    hand gestures.
    """

    def __init__(self, max_faces=4, min_detection_confidence=0.5):
        self._backend = None
        self.engine = None
        self.error = None

        if ensure_blendshape_model():
            try:
                self._backend = _BlendshapeBackend(num_faces=max_faces, min_detection_confidence=min_detection_confidence)
                self.engine = self._backend.engine_name
            except Exception as exc:
                self._backend = None
                self.error = str(exc)
        else:
            self.error = "face model not downloaded (offline?), retrying within an hour"

        self.feature_keys = BLENDSHAPE_NAMES

        self.baseline = None
        self.calibrating = False
        self.calibration_samples = []
        self.calibration_started_at = None
        self._guest_tracks = []

    def start_calibration(self):
        self.calibrating = True
        self.calibration_samples = []
        self.calibration_started_at = time.time()
        self.baseline = None

    def calibration_progress(self):
        if not self.calibrating:
            return 1.0
        return min(1.0, len(self.calibration_samples) / CALIBRATION_FRAMES)

    def _drift_baseline(self, baseline, deltas, raw_features):
        magnitude = sum(abs(v) for v in deltas.values()) / max(1, len(deltas))
        if magnitude < BLENDSHAPE_DRIFT_THRESHOLD:
            for key in self.feature_keys:
                baseline[key] = baseline[key] * (1 - BASELINE_DRIFT_ALPHA) + raw_features[key] * BASELINE_DRIFT_ALPHA

    def _process_guests(self, guest_faces):
        now = time.time()
        results = []
        used_tracks = set()

        for face in guest_faces:
            box = face["box"]
            raw_features = face["raw_features"]
            if box is None:
                continue
            cx = box[0] + box[2] / 2.0
            cy = box[1] + box[3] / 2.0
            diag = math.hypot(box[2], box[3]) or 1.0

            track = None
            for candidate in self._guest_tracks:
                if id(candidate) in used_tracks:
                    continue
                dist = math.hypot(candidate["center"][0] - cx, candidate["center"][1] - cy)
                if dist < diag * GUEST_MATCH_DISTANCE_RATIO:
                    track = candidate
                    break

            if track is None:
                track = {"baseline": dict(raw_features), "center": (cx, cy), "last_seen": now}
                self._guest_tracks.append(track)
            else:
                track["center"] = (cx, cy)
                track["last_seen"] = now
            used_tracks.add(id(track))

            deltas = {key: raw_features[key] - track["baseline"][key] for key in self.feature_keys}
            self._drift_baseline(track["baseline"], deltas, raw_features)

            tags = _blendshape_tags_from_deltas(deltas)
            results.append({"box": box, "tags": tags})

        self._guest_tracks = [t for t in self._guest_tracks if now - t["last_seen"] < GUEST_STALE_SECONDS]
        return results

    @property
    def available(self):
        return self._backend is not None

    def analyze(self, frame_bgr):
        if self._backend is None:
            return None
        faces = self._backend.process(frame_bgr)
        if not faces:
            return None

        faces_sorted = sorted(
            faces, key=lambda f: (f["box"][2] * f["box"][3]) if f["box"] else 0, reverse=True
        )
        primary = faces_sorted[0]
        raw_features = primary["raw_features"]
        box = primary["box"]

        if self.calibrating:
            self.calibration_samples.append(raw_features)
            if len(self.calibration_samples) >= CALIBRATION_FRAMES:
                self.baseline = {
                    key: float(np.median([sample[key] for sample in self.calibration_samples]))
                    for key in self.feature_keys
                }
                self.calibrating = False

        deltas = None
        if self.baseline is not None:
            deltas = {key: raw_features[key] - self.baseline[key] for key in self.feature_keys}
            self._drift_baseline(self.baseline, deltas, raw_features)

        secondary_faces = self._process_guests(faces_sorted[1 : 1 + MAX_GUEST_FACES])

        return {
            "box": box,
            "raw_features": raw_features,
            "deltas": deltas,
            "engine": self.engine,
            "secondary_faces": secondary_faces,
        }

    def close(self):
        if self._backend is not None:
            self._backend.close()


def _blendshape_tags_from_deltas(deltas):
    tags = {}

    def avg(*keys):
        return sum(deltas[k] for k in keys) / len(keys)

    def clip(value, span):
        return max(0.0, min(1.0, value / span))

    jaw_open = deltas["jawOpen"]
    if jaw_open > 0.18:
        tags["jaw_drop"] = clip(jaw_open, 0.5)

    smile = avg("mouthSmileLeft", "mouthSmileRight")
    smile_asymmetry = abs(deltas["mouthSmileLeft"] - deltas["mouthSmileRight"])
    if smile > 0.12 and not (jaw_open > 0.35):
        tags["smile"] = clip(smile, 0.55)
    if smile_asymmetry > 0.15 and smile > 0.05:
        tags["smirk"] = clip(smile_asymmetry, 0.35)

    frown = avg("mouthFrownLeft", "mouthFrownRight")
    if frown > 0.1:
        tags["frown"] = clip(frown, 0.4)

    brow_raise = avg("browInnerUp", "browOuterUpLeft", "browOuterUpRight")
    if brow_raise > 0.15:
        tags["brow_raise"] = clip(brow_raise, 0.55)

    brow_furrow = avg("browDownLeft", "browDownRight")
    if brow_furrow > 0.15:
        tags["brow_furrow"] = clip(brow_furrow, 0.5)

    brow_asymmetry = abs(deltas["browOuterUpLeft"] - deltas["browOuterUpRight"])
    if brow_asymmetry > 0.2 and brow_raise < 0.3:
        tags["skeptical"] = clip(brow_asymmetry, 0.45)

    squint = avg("eyeSquintLeft", "eyeSquintRight")
    if squint > 0.15:
        tags["squint"] = clip(squint, 0.45)

    blink_asymmetry = abs(deltas["eyeBlinkLeft"] - deltas["eyeBlinkRight"])
    if blink_asymmetry > 0.3:
        tags["wink"] = clip(blink_asymmetry, 0.6)

    eye_wide = avg("eyeWideLeft", "eyeWideRight")
    if eye_wide > 0.15:
        tags["eye_wide"] = clip(eye_wide, 0.4)

    sneer = avg("noseSneerLeft", "noseSneerRight")
    if sneer > 0.12:
        tags["sneer"] = clip(sneer, 0.4)

    if deltas["cheekPuff"] > 0.15:
        tags["cheek_puff"] = clip(deltas["cheekPuff"], 0.4)

    pucker = avg("mouthPucker", "mouthFunnel")
    if pucker > 0.15:
        tags["pucker"] = clip(pucker, 0.45)

    return tags


def tags_from_deltas(result_or_deltas):
    """Expression tags from a FaceAnalyzer result or a plain deltas dict."""
    if result_or_deltas is None:
        return {}
    if isinstance(result_or_deltas, dict) and "engine" in result_or_deltas:
        deltas = result_or_deltas.get("deltas")
    else:
        deltas = result_or_deltas
    if deltas is None:
        return {}
    return _blendshape_tags_from_deltas(deltas)


# ============================================================================
# Hand gesture recognition
# ============================================================================

WRIST = 0
THUMB_MCP = 2
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_PIP = 6
INDEX_TIP = 8
MIDDLE_MCP = 9
MIDDLE_PIP = 10
MIDDLE_TIP = 12
RING_MCP = 13
RING_PIP = 14
RING_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_TIP = 20

FINGER_JOINTS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_TIP),
}

GESTURE_TO_TAGS = {
    "thumbs_up": {"approval": 1.0, "happy": 0.5},
    "thumbs_down": {"disapproval": 1.0, "sad": 0.4},
    "open_palm": {"surprise": 0.5, "stop": 1.0},
    "fist": {"angry": 0.7, "determined": 0.6},
    "peace": {"happy": 0.6, "chill": 1.0},
    "pointing": {"suspicious": 0.6, "focused": 0.5},
}


def _finger_extended(points, mcp_idx, pip_idx, tip_idx, wrist_idx=WRIST):
    wrist = points[wrist_idx]
    tip_dist = _dist(wrist, points[tip_idx])
    pip_dist = _dist(wrist, points[pip_idx])
    return tip_dist > pip_dist * 1.08


def _thumb_extended(points):
    wrist = points[WRIST]
    tip_dist = _dist(wrist, points[THUMB_TIP])
    mcp_dist = _dist(wrist, points[THUMB_MCP])
    return tip_dist > mcp_dist * 1.25


def classify_gesture(points):
    finger_state = {
        name: _finger_extended(points, mcp, pip, tip)
        for name, (mcp, pip, tip) in FINGER_JOINTS.items()
    }
    thumb_state = _thumb_extended(points)
    extended_count = sum(finger_state.values()) + (1 if thumb_state else 0)

    palm_center_y = (points[WRIST][1] + points[MIDDLE_MCP][1]) / 2.0

    if thumb_state and not any(finger_state.values()):
        if points[THUMB_TIP][1] < palm_center_y - 15:
            return "thumbs_up"
        if points[THUMB_TIP][1] > palm_center_y + 15:
            return "thumbs_down"

    if extended_count >= 5:
        return "open_palm"

    if extended_count == 0:
        return "fist"

    if finger_state["index"] and finger_state["middle"] and not finger_state["ring"] and not finger_state["pinky"]:
        return "peace"

    if finger_state["index"] and not finger_state["middle"] and not finger_state["ring"] and not finger_state["pinky"]:
        return "pointing"

    return None


class HandGestureRecognizer:
    """Hand landmarks through MediaPipe's HandLandmarker task.

    If the hand model cannot be downloaded, analyze() returns no detections
    and the app runs without gestures.
    """

    def __init__(self, max_hands=2, min_detection_confidence=0.6, min_presence_confidence=0.5):
        self._landmarker = None
        self.error = None
        if not ensure_hand_model():
            self.error = "hand model not downloaded (offline?), retrying within an hour"
            return
        try:
            from mediapipe.tasks.python import vision as mp_vision
            from mediapipe.tasks.python.core.base_options import BaseOptions

            options = mp_vision.HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=HAND_MODEL_PATH),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_hands=max_hands,
                min_hand_detection_confidence=min_detection_confidence,
                min_hand_presence_confidence=min_presence_confidence,
            )
            self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            self._landmarker = None
            self.error = str(exc)

    @property
    def available(self):
        return self._landmarker is not None

    def analyze(self, frame_bgr):
        if self._landmarker is None:
            return []
        height, width = frame_bgr.shape[:2]
        result = self._landmarker.detect(_to_mp_image(frame_bgr))

        detections = []
        for hand_landmarks in result.hand_landmarks or []:
            points = {i: (lm.x * width, lm.y * height) for i, lm in enumerate(hand_landmarks)}
            gesture = classify_gesture(points)
            xs = [p[0] for p in points.values()]
            ys = [p[1] for p in points.values()]
            box = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))
            detections.append({"gesture": gesture, "box": box, "points": points})

        return detections

    def close(self):
        if self._landmarker is not None:
            self._landmarker.close()


def tags_from_gestures(detections):
    combined = {}
    for detection in detections:
        gesture = detection.get("gesture")
        if gesture is None:
            continue
        for tag, score in GESTURE_TO_TAGS.get(gesture, {}).items():
            combined[tag] = max(combined.get(tag, 0.0), score)
    return combined

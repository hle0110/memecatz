"""
Everything that reads signals off the webcam frame: the 7-class emotion CNN,
the face analyzer (52-point blendshapes or a geometric fallback), and hand
gesture recognition. Combined into one module since they're all "look at the
frame and produce tags/scores" — mood.py is what actually combines their
output into a final mood.
"""

import os
import json
import math
import time
import cv2
import numpy as np
import mediapipe as mp
import requests

try:
    import tensorflow as tf
except ImportError:
    tf = None  # only EmotionDetector needs this; FaceAnalyzer/HandGestureRecognizer must still work without it

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
        if tf is None:
            raise ImportError("tensorflow is required for the emotion CNN (pip install tensorflow)")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"emotion model not found at {model_path}")

        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("failed to load the haar cascade face detector")

        self.interpreter = tf.lite.Interpreter(model_path=model_path)
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
# Face analysis: 52-point blendshapes (preferred) or a geometric fallback
# ============================================================================

BLENDSHAPE_MODEL_PATH = os.path.join(MODEL_CACHE_DIR, "face_landmarker.task")
DOWNLOAD_STATUS_PATH = os.path.join(MODEL_CACHE_DIR, "download_status.json")
BLENDSHAPE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)
MODEL_DOWNLOAD_TIMEOUT_SECONDS = 20
DOWNLOAD_RETRY_COOLDOWN_SECONDS = 3600

CALIBRATION_FRAMES = 12
BASELINE_DRIFT_ALPHA = 0.01
DRIFT_THRESHOLDS = {"geometric": 0.05, "blendshapes": 0.07}
MAX_GUEST_FACES = 3
GUEST_MATCH_DISTANCE_RATIO = 0.6
GUEST_STALE_SECONDS = 2.0

RIGHT_EYE_OUTER = 33
RIGHT_EYE_INNER = 133
RIGHT_EYE_TOP = 159
RIGHT_EYE_BOTTOM = 145
LEFT_EYE_OUTER = 263
LEFT_EYE_INNER = 362
LEFT_EYE_TOP = 386
LEFT_EYE_BOTTOM = 374
MOUTH_RIGHT = 61
MOUTH_LEFT = 291
LIP_TOP = 13
LIP_BOTTOM = 14
RIGHT_BROW = 105
LEFT_BROW = 334
FOREHEAD = 10
CHIN = 152
NOSE_TIP = 4

GEOMETRIC_FEATURE_KEYS = (
    "mouth_aperture",
    "mouth_width",
    "smile_curve",
    "corner_asymmetry",
    "eyebrow_raise",
    "eye_aperture_right",
    "eye_aperture_left",
)

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


def _read_download_status():
    if not os.path.isfile(DOWNLOAD_STATUS_PATH):
        return {}
    try:
        with open(DOWNLOAD_STATUS_PATH, "r") as handle:
            return json.load(handle)
    except (json.JSONDecodeError, OSError):
        return {}


def _write_download_status(status):
    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        with open(DOWNLOAD_STATUS_PATH, "w") as handle:
            json.dump(status, handle)
    except OSError:
        pass


def ensure_blendshape_model():
    if os.path.isfile(BLENDSHAPE_MODEL_PATH) and os.path.getsize(BLENDSHAPE_MODEL_PATH) > 0:
        return True

    status = _read_download_status()
    last_attempt = status.get("last_attempt", 0)
    if (time.time() - last_attempt) < DOWNLOAD_RETRY_COOLDOWN_SECONDS:
        return False

    try:
        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
        response = requests.get(BLENDSHAPE_MODEL_URL, timeout=MODEL_DOWNLOAD_TIMEOUT_SECONDS, stream=True)
        response.raise_for_status()
        tmp_path = BLENDSHAPE_MODEL_PATH + ".part"
        with open(tmp_path, "wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 16):
                if chunk:
                    handle.write(chunk)
        os.replace(tmp_path, BLENDSHAPE_MODEL_PATH)
        _write_download_status({"last_attempt": time.time(), "last_success": True})
        return True
    except (requests.RequestException, OSError):
        _write_download_status({"last_attempt": time.time(), "last_success": False})
        return False


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
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
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


class _GeometricBackend:
    engine_name = "geometric"

    def __init__(self, max_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    @staticmethod
    def _points(landmarks, width, height):
        return {
            "right_eye_outer": (landmarks[RIGHT_EYE_OUTER].x * width, landmarks[RIGHT_EYE_OUTER].y * height),
            "right_eye_inner": (landmarks[RIGHT_EYE_INNER].x * width, landmarks[RIGHT_EYE_INNER].y * height),
            "right_eye_top": (landmarks[RIGHT_EYE_TOP].x * width, landmarks[RIGHT_EYE_TOP].y * height),
            "right_eye_bottom": (landmarks[RIGHT_EYE_BOTTOM].x * width, landmarks[RIGHT_EYE_BOTTOM].y * height),
            "left_eye_outer": (landmarks[LEFT_EYE_OUTER].x * width, landmarks[LEFT_EYE_OUTER].y * height),
            "left_eye_inner": (landmarks[LEFT_EYE_INNER].x * width, landmarks[LEFT_EYE_INNER].y * height),
            "left_eye_top": (landmarks[LEFT_EYE_TOP].x * width, landmarks[LEFT_EYE_TOP].y * height),
            "left_eye_bottom": (landmarks[LEFT_EYE_BOTTOM].x * width, landmarks[LEFT_EYE_BOTTOM].y * height),
            "mouth_right": (landmarks[MOUTH_RIGHT].x * width, landmarks[MOUTH_RIGHT].y * height),
            "mouth_left": (landmarks[MOUTH_LEFT].x * width, landmarks[MOUTH_LEFT].y * height),
            "lip_top": (landmarks[LIP_TOP].x * width, landmarks[LIP_TOP].y * height),
            "lip_bottom": (landmarks[LIP_BOTTOM].x * width, landmarks[LIP_BOTTOM].y * height),
            "right_brow": (landmarks[RIGHT_BROW].x * width, landmarks[RIGHT_BROW].y * height),
            "left_brow": (landmarks[LEFT_BROW].x * width, landmarks[LEFT_BROW].y * height),
        }

    @staticmethod
    def _raw_features(pts):
        iod = _dist(pts["right_eye_outer"], pts["left_eye_outer"])
        if iod < 1e-6:
            iod = 1.0

        corner_avg_y = (pts["mouth_right"][1] + pts["mouth_left"][1]) / 2.0

        features = {
            "mouth_aperture": _dist(pts["lip_top"], pts["lip_bottom"]) / iod,
            "mouth_width": _dist(pts["mouth_right"], pts["mouth_left"]) / iod,
            "smile_curve": (pts["lip_top"][1] - corner_avg_y) / iod,
            "corner_asymmetry": (pts["mouth_right"][1] - pts["mouth_left"][1]) / iod,
            "eyebrow_raise": (
                (_dist(pts["right_brow"], pts["right_eye_top"]) + _dist(pts["left_brow"], pts["left_eye_top"]))
                / 2.0
                / iod
            ),
            "eye_aperture_right": _dist(pts["right_eye_top"], pts["right_eye_bottom"]) / iod,
            "eye_aperture_left": _dist(pts["left_eye_top"], pts["left_eye_bottom"]) / iod,
        }
        return features, iod

    @staticmethod
    def _bounding_box(pts, width, height):
        xs = [p[0] for p in pts.values()]
        ys = [p[1] for p in pts.values()]
        x1 = max(0, int(min(xs)) - 20)
        y1 = max(0, int(min(ys)) - 40)
        x2 = min(width, int(max(xs)) + 20)
        y2 = min(height, int(max(ys)) + 20)
        return x1, y1, x2 - x1, y2 - y1

    def process(self, frame_bgr):
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        faces = []
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            pts = self._points(landmarks, width, height)
            raw_features, _ = self._raw_features(pts)
            box = self._bounding_box(pts, width, height)
            faces.append({"raw_features": raw_features, "box": box})

        return faces

    def close(self):
        self._mesh.close()


class FaceAnalyzer:
    def __init__(self, max_faces=4, min_detection_confidence=0.5, min_tracking_confidence=0.5, prefer_blendshapes=True):
        self._backend = None
        self.engine = None

        if prefer_blendshapes and ensure_blendshape_model():
            try:
                self._backend = _BlendshapeBackend(num_faces=max_faces, min_detection_confidence=min_detection_confidence)
                self.engine = self._backend.engine_name
            except Exception:
                self._backend = None

        if self._backend is None:
            self._backend = _GeometricBackend(
                max_faces=max_faces,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )
            self.engine = self._backend.engine_name

        self.feature_keys = BLENDSHAPE_NAMES if self.engine == "blendshapes" else GEOMETRIC_FEATURE_KEYS

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
        threshold = DRIFT_THRESHOLDS.get(self.engine, 0.06)
        if magnitude < threshold:
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

            if self.engine == "blendshapes":
                tags = _blendshape_tags_from_deltas(deltas)
            else:
                tags = _geometric_tags_from_deltas(deltas)
            results.append({"box": box, "tags": tags})

        self._guest_tracks = [t for t in self._guest_tracks if now - t["last_seen"] < GUEST_STALE_SECONDS]
        return results

    def analyze(self, frame_bgr):
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
        self._backend.close()


def _geometric_tags_from_deltas(deltas):
    tags = {}

    jaw_drop_active = deltas["mouth_aperture"] > 0.22
    if jaw_drop_active:
        tags["jaw_drop"] = min(1.0, deltas["mouth_aperture"] / 0.4)

    smile_score = max(0.0, deltas["smile_curve"]) + max(0.0, deltas["mouth_width"]) * 0.5
    if smile_score > 0.05 and not (jaw_drop_active and deltas["mouth_aperture"] > 0.3):
        tags["smile"] = min(1.0, smile_score / 0.18)

    if deltas["smile_curve"] < -0.045:
        tags["frown"] = min(1.0, -deltas["smile_curve"] / 0.12)

    if abs(deltas["corner_asymmetry"]) > 0.035 and smile_score < 0.09:
        tags["smirk"] = min(1.0, abs(deltas["corner_asymmetry"]) / 0.09)

    if deltas["eyebrow_raise"] > 0.055:
        tags["brow_raise"] = min(1.0, deltas["eyebrow_raise"] / 0.13)
    elif deltas["eyebrow_raise"] < -0.03:
        tags["brow_furrow"] = min(1.0, -deltas["eyebrow_raise"] / 0.08)

    eye_r = deltas["eye_aperture_right"]
    eye_l = deltas["eye_aperture_left"]
    both_narrow = eye_r < -0.035 and eye_l < -0.035
    if both_narrow:
        tags["squint"] = min(1.0, -((eye_r + eye_l) / 2.0) / 0.1)
    elif abs(eye_r - eye_l) > 0.05:
        tags["wink"] = min(1.0, abs(eye_r - eye_l) / 0.12)

    return tags


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
    if result_or_deltas is None:
        return {}

    if isinstance(result_or_deltas, dict) and "engine" in result_or_deltas:
        deltas = result_or_deltas.get("deltas")
        engine = result_or_deltas.get("engine")
    else:
        deltas = result_or_deltas
        engine = "geometric" if deltas is not None and "mouth_aperture" in deltas else "blendshapes"

    if deltas is None:
        return {}

    if engine == "blendshapes":
        return _blendshape_tags_from_deltas(deltas)
    return _geometric_tags_from_deltas(deltas)


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
    def __init__(self, max_hands=2, min_detection_confidence=0.6, min_tracking_confidence=0.5):
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def analyze(self, frame_bgr):
        height, width = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = self._hands.process(rgb)

        detections = []
        if not results.multi_hand_landmarks:
            return detections

        for hand_landmarks in results.multi_hand_landmarks:
            points = {i: (lm.x * width, lm.y * height) for i, lm in enumerate(hand_landmarks.landmark)}
            gesture = classify_gesture(points)
            xs = [p[0] for p in points.values()]
            ys = [p[1] for p in points.values()]
            box = (int(min(xs)), int(min(ys)), int(max(xs) - min(xs)), int(max(ys) - min(ys)))
            detections.append({"gesture": gesture, "box": box, "points": points})

        return detections

    def close(self):
        self._hands.close()


def tags_from_gestures(detections):
    combined = {}
    for detection in detections:
        gesture = detection.get("gesture")
        if gesture is None:
            continue
        for tag, score in GESTURE_TO_TAGS.get(gesture, {}).items():
            combined[tag] = max(combined.get(tag, 0.0), score)
    return combined

import os
import json
import time
import cv2
import numpy as np

try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False

MATCH_TOLERANCE = 0.6


class FaceIdentityManager:
    def __init__(self, profiles_dir):
        self.available = FACE_RECOGNITION_AVAILABLE
        self.profiles_dir = profiles_dir
        self.profiles_path = os.path.join(profiles_dir, "profiles.json")
        self.profiles = []

        if self.available:
            self._load()

    def _load(self):
        if not os.path.isfile(self.profiles_path):
            return
        try:
            with open(self.profiles_path, "r") as handle:
                self.profiles = json.load(handle)
        except (json.JSONDecodeError, OSError):
            self.profiles = []

    def _save(self):
        try:
            os.makedirs(self.profiles_dir, exist_ok=True)
            with open(self.profiles_path, "w") as handle:
                json.dump(self.profiles, handle)
        except OSError:
            pass

    def _encode(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb, model="hog")
        if not locations:
            return None
        encodings = face_recognition.face_encodings(rgb, known_face_locations=locations)
        if not encodings:
            return None
        return encodings[0]

    def identify(self, frame_bgr):
        if not self.available or not self.profiles:
            return None

        try:
            encoding = self._encode(frame_bgr)
        except Exception:
            return None

        if encoding is None:
            return None

        known_encodings = [np.array(profile["encoding"]) for profile in self.profiles]
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_index = int(np.argmin(distances))
        if distances[best_index] <= MATCH_TOLERANCE:
            return self.profiles[best_index]
        return None

    def enroll(self, frame_bgr, name, engine, baseline):
        if not self.available:
            return False

        try:
            encoding = self._encode(frame_bgr)
        except Exception:
            encoding = None

        if encoding is None:
            return False

        profile = {
            "name": name,
            "encoding": encoding.tolist(),
            "engine": engine,
            "baseline": baseline,
            "updated_at": time.time(),
        }
        self.profiles = [existing for existing in self.profiles if existing["name"] != name]
        self.profiles.append(profile)
        self._save()
        return True

    def update_baseline(self, name, engine, baseline):
        for profile in self.profiles:
            if profile["name"] == name:
                profile["engine"] = engine
                profile["baseline"] = baseline
                profile["updated_at"] = time.time()
                self._save()
                return True
        return False

    def next_guest_name(self):
        existing = {profile["name"] for profile in self.profiles}
        index = 1
        while f"guest_{index}" in existing:
            index += 1
        return f"guest_{index}"

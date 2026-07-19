import os
import sys
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def check_opencv():
    print(f"opencv version: {cv2.__version__}")
    return True


def check_cascade():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    ok = not cascade.empty()
    print(f"face cascade loaded: {ok}")
    return ok


def check_emotion_model():
    try:
        import tensorflow as tf
    except ImportError as error:
        print(f"tensorflow import failed: {error}")
        return False

    try:
        from vision import EMOTION_MODEL_PATH as model_path
    except ImportError as error:
        print(f"vision import failed: {error}")
        return False

    if not os.path.isfile(model_path):
        print(f"model file missing at {model_path}")
        return False

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    dummy_input = np.zeros((1, 64, 64, 1), dtype=np.float32)
    interpreter.set_tensor(input_details[0]["index"], dummy_input)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])
    ok = output.shape == (1, 7)
    print(f"emotion cnn runs, output shape: {output.shape}")
    return ok


def check_mediapipe():
    try:
        import mediapipe as mp
    except ImportError as error:
        print(f"mediapipe import failed: {error}")
        return False

    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
    face_mesh.close()
    hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=2)
    hands.close()
    print(f"mediapipe version: {mp.__version__}, face mesh + hands initialized ok")
    return True


def check_face_analyzer():
    try:
        from vision import FaceAnalyzer
    except ImportError as error:
        print(f"vision import failed: {error}")
        return False

    analyzer = FaceAnalyzer()
    if analyzer.engine == "blendshapes":
        print("52-point blendshape model ready (best quality, downloaded/cached locally)")
    else:
        print("blendshape model unavailable right now (no internet on first run, or blocked network),"
              " using the built-in geometric fallback analyzer instead. this is not an error."
              " it retries automatically about once an hour, or delete model_cache/ to retry immediately.")
    analyzer.close()
    return True


def check_face_identity():
    try:
        from identity import FaceIdentityManager
    except ImportError as error:
        print(f"identity import failed: {error}")
        return False

    manager = FaceIdentityManager(os.path.join(BASE_DIR, "profiles"))
    if manager.available:
        print(f"face_recognition installed, multi-user profiles enabled ({len(manager.profiles)} saved profile(s))")
    else:
        print("face_recognition not installed, multi-user profiles disabled (this is optional and fine)."
              " see README for how to enable it.")
    return True


def check_reaction_source():
    try:
        from reactions import ReactionSource
    except ImportError as error:
        print(f"reactions import failed: {error}")
        return False

    cache_dir = os.path.join(BASE_DIR, "reaction_cache")
    source = ReactionSource(cache_dir)
    print("reaction source:", source.describe_source())

    pick = source.pick(["happy"])
    if pick["source"] == "unavailable":
        print("no real cat reaction available yet (no internet, or no GIPHY_API_KEY/CAT_API_KEY reachable)."
              " this is not a crash: the app shows a plain 'connecting...' panel with your caption text"
              " until it can reach the internet. it never substitutes fake or generated imagery.")
    else:
        print(f"fetched a real sample reaction from '{pick['source']}': {pick['name']}")
    return True


def check_caption_engine():
    try:
        from captions import CaptionEngine
    except ImportError as error:
        print(f"captions import failed: {error}")
        return False

    engine = CaptionEngine()
    if engine.enabled:
        print("OPENAI_API_KEY detected, live machine learning caption generation enabled")
    else:
        print("no OPENAI_API_KEY set, using the built-in static caption bank (this is fine, app still works)")
    return True


def check_vision_mood():
    try:
        from captions import VisionMoodAnalyzer
    except ImportError as error:
        print(f"captions import failed: {error}")
        return False

    analyzer = VisionMoodAnalyzer()
    if analyzer.enabled:
        print("OPENAI_API_KEY detected, vision-based mood boost enabled (richer nuance from a vision-capable model)")
    else:
        print("no OPENAI_API_KEY set, vision mood boost disabled (this is optional and fine, uses the same key as captions)")
    return True


def check_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("webcam did not open, check camera permissions and that no other app is using it")
        return False
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("webcam opened but did not return a frame")
        return False
    print(f"webcam ok, frame shape: {frame.shape}")
    return True


def main():
    checks = [
        ("opencv", check_opencv),
        ("face cascade", check_cascade),
        ("emotion cnn", check_emotion_model),
        ("mediapipe (face mesh + hands)", check_mediapipe),
        ("face analyzer (blendshape model)", check_face_analyzer),
        ("face identity (optional)", check_face_identity),
        ("real cat reaction source", check_reaction_source),
        ("caption engine", check_caption_engine),
        ("vision mood boost (optional)", check_vision_mood),
        ("webcam", check_webcam),
    ]

    results = {}
    for name, check in checks:
        print(f"\n--- checking {name} ---")
        try:
            results[name] = check()
        except Exception as error:
            print(f"{name} check raised an error: {error}")
            results[name] = False

    print("\n=== summary ===")
    all_passed = True
    for name, passed in results.items():
        status = "ok" if passed else "FAILED"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False

    if all_passed:
        print("\neverything looks good, run: python run.py (or ./run.sh / run.bat)")
    else:
        print("\nfix the failed checks above before running run.py")
        sys.exit(1)


if __name__ == "__main__":
    main()

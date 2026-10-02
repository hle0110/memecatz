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
        from vision import EMOTION_MODEL_PATH as model_path, TFLiteInterpreter
    except ImportError as error:
        print(f"vision import failed: {error}")
        return False

    if TFLiteInterpreter is None:
        print("no TFLite runtime found, install one with: pip install ai-edge-litert")
        return False

    print(f"tflite runtime: {TFLiteInterpreter.__module__.split('.')[0]}")

    if not os.path.isfile(model_path):
        print(f"model file missing at {model_path}")
        return False

    interpreter = TFLiteInterpreter(model_path=model_path)
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
        from mediapipe.tasks.python import vision as mp_vision  # noqa: F401
    except ImportError as error:
        print(f"mediapipe import failed: {error}")
        return False
    print(f"mediapipe version: {mp.__version__}, tasks API available")
    return True


def check_face_analyzer():
    try:
        from vision import FaceAnalyzer
    except ImportError as error:
        print(f"vision import failed: {error}")
        return False

    analyzer = FaceAnalyzer()
    try:
        if not analyzer.available:
            print(f"face model unavailable: {analyzer.error}."
                  " the app still runs on the emotion model and gestures."
                  " it retries about once an hour, or delete model_cache/ to retry now.")
            return False
        analyzer.analyze(np.zeros((480, 640, 3), dtype=np.uint8))
        print("52-point blendshape model ready and runs on a test frame")
        return True
    finally:
        analyzer.close()


def check_hand_gestures():
    try:
        from vision import HandGestureRecognizer
    except ImportError as error:
        print(f"vision import failed: {error}")
        return False

    recognizer = HandGestureRecognizer()
    try:
        if not recognizer.available:
            print(f"hand model unavailable: {recognizer.error}. the app still runs without gestures.")
            return False
        recognizer.analyze(np.zeros((480, 640, 3), dtype=np.uint8))
        print("hand landmark model ready and runs on a test frame")
        return True
    finally:
        recognizer.close()


def check_face_identity():
    try:
        from identity import FaceIdentityManager
    except ImportError as error:
        print(f"identity import failed: {error}")
        return False

    manager = FaceIdentityManager(os.path.join(BASE_DIR, "profiles"))
    if manager.available:
        print(f"face_recognition installed, saved profiles available with --save-profile"
              f" ({len(manager.profiles)} saved profile(s))")
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
        print("OPENAI_API_KEY detected, live caption generation enabled")
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
        print("OPENAI_API_KEY detected, vision mood boost enabled (adds extra nuance to mood detection)")
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
    skip_webcam = "--no-webcam" in sys.argv  # for CI machines, which have no camera
    checks = [
        ("opencv", check_opencv),
        ("face cascade", check_cascade),
        ("emotion cnn", check_emotion_model),
        ("mediapipe", check_mediapipe),
        ("face analyzer (blendshape model)", check_face_analyzer),
        ("hand gestures (hand landmark model)", check_hand_gestures),
        ("face identity (optional)", check_face_identity),
        ("real reaction source", check_reaction_source),
        ("caption engine", check_caption_engine),
        ("vision mood boost (optional)", check_vision_mood),
    ]
    if not skip_webcam:
        checks.append(("webcam", check_webcam))

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

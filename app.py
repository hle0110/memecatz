import os
import sys
import time
import argparse
import threading
import cv2
import numpy as np

from vision import EmotionDetector, FaceAnalyzer, tags_from_deltas, HandGestureRecognizer, tags_from_gestures
from mood import combine, top_tags
from reactions import ReactionSource, render as render_reaction, render_frames as render_reaction_frames
from captions import CaptionEngine, VisionMoodAnalyzer
from identity import FaceIdentityManager
from utils import fit_to_panel, compute_fit

BASE_DIR = os.path.dirname(os.path.abspath(__file__))       # project root (this file lives at the top level)
REACTION_CACHE_DIR = os.path.join(BASE_DIR, "reaction_cache")
PROFILES_DIR = os.path.join(BASE_DIR, "profiles")
SNAPSHOTS_DIR = os.path.join(BASE_DIR, "snapshots")

PANEL_WIDTH = 640
PANEL_HEIGHT = 480
CAMERA_INDEX = 0

DETECTION_INTERVAL_SECONDS = 0.15
MOOD_SWITCH_COOLDOWN = 1.8
SAME_MOOD_ROTATE_SECONDS = 7.0
MOOD_TOP_LIMIT = 3
MOOD_SMOOTHING_ALPHA = 0.35
REACTION_TRANSITION_SECONDS = 0.35
SNAPSHOT_FLASH_SECONDS = 1.2

UNAVAILABLE_PLACEHOLDER = "connecting for a real reaction..."


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.latest_frame = None
        self.running = True
        self.face_box = None
        self.secondary_faces = []
        self.top_mood_tags = []
        self.gesture_label = None
        self.calibrating = True
        self.calibration_progress = 0.0
        self.caption_source = "static"
        self.reaction_source = "unavailable"
        self.recalibrate_requested = False


def crop_face(frame, box, padding_ratio=0.25):
    if box is None:
        return None
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(frame.shape[1], x + w + pad_x)
    y2 = min(frame.shape[0], y + h + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def render_pick(pick, caption):
    """Returns (panels, durations). A still image is a single panel."""
    if pick.get("frames"):
        panels = render_reaction_frames(pick["frames"], caption, PANEL_WIDTH, PANEL_HEIGHT,
                                        attribution=pick.get("attribution"))
        return panels, list(pick["durations"])
    panel = render_reaction(
        pick["image"], caption, PANEL_WIDTH, PANEL_HEIGHT,
        placeholder_text=UNAVAILABLE_PLACEHOLDER,
        attribution=pick.get("attribution"),
    )
    return [panel], [0.0]


def current_animation_frame(panels, durations, start, now):
    """The panel to show at time now for an animation that started at start."""
    if not panels:
        return None
    if len(panels) == 1:
        return panels[0]
    total = sum(durations)
    if total <= 0:
        return panels[0]
    t = (now - start) % total
    for panel, duration in zip(panels, durations):
        if t < duration:
            return panel
        t -= duration
    return panels[-1]


def smooth_mood_vector(previous_vector, current_vector, alpha=MOOD_SMOOTHING_ALPHA):
    """
    Exponential moving average over the mood vector so a single noisy frame
    can't cause the ranked mood (and HUD readout) to jump around. Higher
    alpha reacts faster; lower alpha is smoother but slower to catch up.
    """
    all_tags = set(previous_vector) | set(current_vector)
    smoothed = {}
    for tag in all_tags:
        previous = previous_vector.get(tag, 0.0)
        current = current_vector.get(tag, 0.0)
        smoothed[tag] = previous + alpha * (current - previous)
    return smoothed


def get_current_reaction_panel(reaction_holder, now=None):
    """
    Crossfades from the previous reaction panel to the current one over
    REACTION_TRANSITION_SECONDS instead of a hard cut, whenever the reaction
    just switched.
    """
    now = time.time() if now is None else now
    # Read the current set once; the worker thread swaps in a whole new tuple.
    panels, durations, start = reaction_holder.get("current", (None, None, 0.0))
    panel = current_animation_frame(panels, durations, start, now)
    if panel is None:
        return np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)

    previous = reaction_holder.get("previous_panel")
    if previous is None:
        return panel

    start = reaction_holder.get("transition_start", 0.0)
    elapsed = now - start
    if elapsed >= REACTION_TRANSITION_SECONDS or elapsed < 0:
        return panel

    t = max(0.0, min(1.0, elapsed / REACTION_TRANSITION_SECONDS))
    return cv2.addWeighted(previous, 1.0 - t, panel, t, 0)


def switch_reaction(state, reaction_source, caption_engine, reaction_holder, mood_tags):
    """Picks, captions and renders the next reaction, then swaps it in."""
    try:
        pick = reaction_source.pick(mood_tags, exclude_key=reaction_holder.get("key"))
        caption = caption_engine.generate(pick["key"], pick["name"], mood_tags)
        panels, durations = render_pick(pick, caption)
    except Exception as error:
        print(f"could not switch reaction: {error}")
        return

    render_done = time.time()
    reaction_holder["previous_panel"] = get_current_reaction_panel(reaction_holder, render_done)
    reaction_holder["current"] = (panels, durations, render_done)
    reaction_holder["transition_start"] = render_done
    reaction_holder["key"] = pick["key"]
    reaction_holder["name"] = pick["name"]
    reaction_holder["source"] = pick["source"]

    with state.lock:
        state.caption_source = caption.get("source", "static")
        state.reaction_source = pick["source"]


def detection_worker(state, emotion_detector, face_analyzer, hand_recognizer, reaction_source, caption_engine,
                      vision_analyzer, reaction_holder):
    last_switch_time = 0.0
    last_rotate_time = time.time()
    last_mood_key = None
    smoothed_vector = {}
    switch_thread = None

    while state.running:
        with state.lock:
            frame = None if state.latest_frame is None else state.latest_frame.copy()
            recalibrate = state.recalibrate_requested
            state.recalibrate_requested = False

        # Recalibration is started here, on the thread that uses the analyzer,
        # so the calibration samples are never touched from two threads.
        if recalibrate and face_analyzer.available:
            face_analyzer.start_calibration()
            smoothed_vector = {}

        if frame is None:
            time.sleep(0.05)
            continue

        try:
            face_result = face_analyzer.analyze(frame)
        except Exception:
            face_result = None

        with state.lock:
            state.calibrating = face_analyzer.calibrating
            state.calibration_progress = face_analyzer.calibration_progress()
            state.face_box = face_result["box"] if face_result else None
            state.secondary_faces = face_result.get("secondary_faces", []) if face_result else []

        if face_analyzer.calibrating:
            time.sleep(DETECTION_INTERVAL_SECONDS)
            continue

        try:
            emotion_result = emotion_detector.detect(frame)
        except Exception:
            emotion_result = None

        try:
            hand_detections = hand_recognizer.analyze(frame)
        except Exception:
            hand_detections = []

        au_tags = tags_from_deltas(face_result) if face_result else {}
        gesture_tags = tags_from_gestures(hand_detections)
        fer_scores = emotion_result["scores"] if emotion_result else None

        gesture_label = None
        for detection in hand_detections:
            if detection.get("gesture"):
                gesture_label = detection["gesture"]
                break

        preliminary_vector = combine(fer_scores=fer_scores, au_tags=au_tags, gesture_tags=gesture_tags)
        preliminary_tags = [tag for tag, _ in top_tags(preliminary_vector, limit=MOOD_TOP_LIMIT)]

        face_crop = crop_face(frame, face_result["box"]) if face_result else None
        try:
            vision_tags = vision_analyzer.analyze(face_crop, preliminary_tags)
        except Exception:
            vision_tags = {}

        mood_vector = combine(fer_scores=fer_scores, au_tags=au_tags, gesture_tags=gesture_tags, vision_tags=vision_tags)
        smoothed_vector = smooth_mood_vector(smoothed_vector, mood_vector)
        ranked = top_tags(smoothed_vector, limit=MOOD_TOP_LIMIT)
        mood_tags = [tag for tag, _ in ranked]

        with state.lock:
            state.top_mood_tags = ranked
            state.gesture_label = gesture_label

        now = time.time()
        mood_key = mood_tags[0] if mood_tags else "neutral"
        mood_changed = mood_key != last_mood_key
        cooldown_elapsed = (now - last_switch_time) > MOOD_SWITCH_COOLDOWN
        rotate_due = (now - last_rotate_time) > SAME_MOOD_ROTATE_SECONDS

        switch_due = (mood_changed and cooldown_elapsed) or (not mood_changed and rotate_due)
        if switch_due and (switch_thread is None or not switch_thread.is_alive()):
            # Downloading a reaction and asking for a caption can take a second
            # or more, so it runs beside detection instead of pausing it.
            switch_thread = threading.Thread(
                target=switch_reaction,
                args=(state, reaction_source, caption_engine, reaction_holder, list(mood_tags)),
                daemon=True,
            )
            switch_thread.start()
            last_switch_time = now
            last_rotate_time = now
            last_mood_key = mood_key

        time.sleep(DETECTION_INTERVAL_SECONDS)


def draw_face_box(panel, frame_shape, box):
    if box is None:
        return
    scale, x_off, y_off, _, _ = compute_fit(frame_shape, PANEL_WIDTH, PANEL_HEIGHT)
    x, y, w, h = box
    x1 = int(x * scale) + x_off
    y1 = int(y * scale) + y_off
    x2 = int((x + w) * scale) + x_off
    y2 = int((y + h) * scale) + y_off
    cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 255, 120), 2)


def draw_secondary_faces(panel, frame_shape, secondary_faces):
    if not secondary_faces:
        return
    scale, x_off, y_off, _, _ = compute_fit(frame_shape, PANEL_WIDTH, PANEL_HEIGHT)
    for face in secondary_faces:
        box = face.get("box")
        if box is None:
            continue
        x, y, w, h = box
        x1 = int(x * scale) + x_off
        y1 = int(y * scale) + y_off
        x2 = int((x + w) * scale) + x_off
        y2 = int((y + h) * scale) + y_off
        cv2.rectangle(panel, (x1, y1), (x2, y2), (255, 160, 0), 2)
        tags = face.get("tags") or {}
        top_tag = max(tags.items(), key=lambda item: item[1])[0] if tags else "neutral"
        cv2.putText(panel, top_tag, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 160, 0), 2, cv2.LINE_AA)


def draw_hud(panel, mood_tags, gesture_label, fps_value, caption_source, calibrating, calibration_progress, engine_label,
             vision_enabled, reaction_source_label):
    y = 28
    if calibrating:
        cv2.putText(panel, "getting ready...", (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 220, 255), 2, cv2.LINE_AA)
        bar_width = int((PANEL_WIDTH - 40) * calibration_progress)
        cv2.rectangle(panel, (20, y + 12), (20 + bar_width, y + 24), (0, 220, 255), -1)
        cv2.rectangle(panel, (20, y + 12), (PANEL_WIDTH - 20, y + 24), (255, 255, 255), 1)
        return

    if mood_tags:
        text = "  ".join(f"{tag} {score:.2f}" for tag, score in mood_tags)
        cv2.putText(panel, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 120), 2, cv2.LINE_AA)

    if gesture_label:
        cv2.putText(panel, f"gesture: {gesture_label}", (20, y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 200, 0), 2, cv2.LINE_AA)

    vision_label = "on" if vision_enabled else "off"
    footer = (f"FPS {fps_value:.0f}  face: {engine_label}  vision: {vision_label}  "
              f"reactions: {reaction_source_label}  captions: {caption_source}  q quit  c recalibrate  s snapshot")
    cv2.putText(panel, footer, (20, PANEL_HEIGHT - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_snapshot_flash(panel, flash_deadline, now=None):
    """Brief on-screen confirmation so 's' feels acknowledged, not just a console print."""
    now = time.time() if now is None else now
    if now >= flash_deadline:
        return
    text = "snapshot saved"
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cx = panel.shape[1] // 2
    x1, y1 = cx - text_w // 2 - 14, 14
    x2, y2 = cx + text_w // 2 + 14, 14 + text_h + 18
    cv2.rectangle(panel, (x1, y1), (x2, y2), (0, 150, 0), -1)
    cv2.putText(panel, text, (cx - text_w // 2, y2 - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255, 255, 255), 2, cv2.LINE_AA)


def save_snapshot(combined_frame):
    os.makedirs(SNAPSHOTS_DIR, exist_ok=True)
    filename = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.png"
    path = os.path.join(SNAPSHOTS_DIR, filename)
    cv2.imwrite(path, combined_frame)
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="MemeCatz: your face, their reaction.")
    parser.add_argument("--animal", choices=["cat", "dog"], default="cat", help="react with cats or dogs")
    parser.add_argument("--giphy-key", default=None, help="Giphy key for mood matched reactions (or set GIPHY_API_KEY)")
    parser.add_argument("--openai-key", default=None, help="OpenAI key for live captions (or set OPENAI_API_KEY)")
    parser.add_argument("--save-profile", action="store_true",
                        help="remember your calibration for next time (stores a face encoding in profiles/)")
    parser.add_argument("--refresh-dataset", action="store_true", help="clear the saved backup photos and fetch new ones")
    return parser.parse_args()


def main():
    args = parse_args()
    window_name = f"MemeCatz ({args.animal.capitalize()} Mode)"

    try:
        emotion_detector = EmotionDetector()
    except Exception as error:
        print(f"failed to initialize emotion detector: {error}")
        sys.exit(1)

    face_analyzer = FaceAnalyzer()
    if face_analyzer.available:
        print("face analysis: 52-point blendshape model")
    else:
        print(f"face analysis: off ({face_analyzer.error}). running on the emotion model and hand gestures.")
    hand_recognizer = HandGestureRecognizer()
    print("hand gestures:", "on" if hand_recognizer.available else f"off ({hand_recognizer.error})")

    print(f"connecting to real {args.animal} reaction sources...")
    reaction_source = ReactionSource(REACTION_CACHE_DIR, animal=args.animal, giphy_api_key=args.giphy_key,
                                      force_refresh=args.refresh_dataset)
    print("reactions:", reaction_source.describe_source())

    caption_engine = CaptionEngine(api_key=args.openai_key)
    print("captions:", "live text-generation model" if caption_engine.enabled else "static fallback bank (set OPENAI_API_KEY to enable live captions)")

    vision_analyzer = VisionMoodAnalyzer(api_key=args.openai_key)
    print("vision mood boost:", "enabled (adds extra nuance to mood detection)" if vision_analyzer.enabled
          else "disabled (set OPENAI_API_KEY to enable it, same key used for captions)")

    identity_manager = FaceIdentityManager(PROFILES_DIR)
    if not identity_manager.available:
        print("saved profiles: off (install face_recognition to use --save-profile)")
    elif args.save_profile:
        print("saved profiles: on, your calibration will be remembered on this computer")
    else:
        print("saved profiles: reading existing ones only (add --save-profile to save yours)")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("could not open webcam, check that it is connected and not in use by another app")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # keep capture latency low; ignored if the backend doesn't support it
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # many USB webcams hit a much higher FPS in MJPG mode; ignored if unsupported
    cap.set(cv2.CAP_PROP_FPS, 30)  # ask for a faster capture rate where the backend allows it

    state = SharedState()

    matched_profile = None
    if identity_manager.available and identity_manager.profiles:
        # The first frames from many webcams are dark while exposure settles,
        # so give it a moment and try a few frames before giving up.
        for attempt in range(20):
            ok, priming_frame = cap.read()
            if not ok:
                break
            if attempt >= 8 and attempt % 4 == 0:
                matched_profile = identity_manager.identify(cv2.flip(priming_frame, 1))
                if matched_profile is not None:
                    break

    needs_enrollment = False
    active_profile_name = None
    if not face_analyzer.available:
        face_analyzer.calibrating = False
    elif matched_profile is not None and matched_profile.get("engine") == face_analyzer.engine:
        face_analyzer.baseline = matched_profile["baseline"]
        face_analyzer.calibrating = False
        active_profile_name = matched_profile["name"]
        print(f"welcome back, {matched_profile['name']}! loaded your saved calibration.")
    else:
        if matched_profile is not None:
            print(f"found a profile for {matched_profile['name']} but the face engine changed, recalibrating.")
        face_analyzer.start_calibration()
        # A face encoding is biometric data, so it is only stored when asked.
        needs_enrollment = identity_manager.available and args.save_profile

    initial_pick = reaction_source.pick(["neutral"])
    # The first caption comes from the built-in bank so startup never waits on the API.
    initial_caption = caption_engine.generate(initial_pick["key"], initial_pick["name"], ["neutral"], live=False)
    initial_panels, initial_durations = render_pick(initial_pick, initial_caption)
    reaction_holder = {
        "key": initial_pick["key"],
        "current": (initial_panels, initial_durations, time.time()),
        "previous_panel": None,
        "transition_start": 0.0,
        "name": initial_pick["name"],
        "source": initial_pick["source"],
    }
    state.reaction_source = initial_pick["source"]

    worker = threading.Thread(
        target=detection_worker,
        args=(state, emotion_detector, face_analyzer, hand_recognizer, reaction_source, caption_engine,
              vision_analyzer, reaction_holder),
        daemon=True,
    )
    worker.start()

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, PANEL_WIDTH * 2, PANEL_HEIGHT)

    fps_time = time.time()
    fps_counter = 0
    fps_value = 0.0
    # Start the shared flag from the real state, so a loaded profile is not
    # mistaken for a calibration that just finished.
    state.calibrating = face_analyzer.calibrating
    was_calibrating = face_analyzer.calibrating
    enrolled_this_session = not needs_enrollment
    snapshot_flash_deadline = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("lost connection to the webcam")
                break

            frame = cv2.flip(frame, 1)

            with state.lock:
                state.latest_frame = frame.copy()
                face_box = state.face_box
                secondary_faces = state.secondary_faces
                mood_tags = state.top_mood_tags
                gesture_label = state.gesture_label
                calibrating = state.calibrating
                calibration_progress = state.calibration_progress
                caption_source = state.caption_source
                reaction_source_label = state.reaction_source

            if was_calibrating and not calibrating and needs_enrollment and not enrolled_this_session:
                guest_name = identity_manager.next_guest_name()
                if identity_manager.enroll(frame, guest_name, face_analyzer.engine, face_analyzer.baseline):
                    active_profile_name = guest_name
                    print(f"saved a new calibration profile as '{guest_name}'"
                          f" (rename it any time by editing profiles/profiles.json)")
                enrolled_this_session = True
            elif (was_calibrating and not calibrating and args.save_profile and active_profile_name
                  and face_analyzer.baseline is not None):
                # A recalibration (c) replaces the saved calibration too.
                if identity_manager.update_baseline(active_profile_name, face_analyzer.engine,
                                                    dict(face_analyzer.baseline)):
                    print(f"updated the saved calibration for '{active_profile_name}'")
            was_calibrating = calibrating

            webcam_panel = fit_to_panel(frame, PANEL_WIDTH, PANEL_HEIGHT)
            reaction_panel = get_current_reaction_panel(reaction_holder)
            # Snapshots get the picture without boxes, scores or menus.
            clean_combined = np.hstack((webcam_panel, reaction_panel))
            webcam_panel = webcam_panel.copy()
            draw_face_box(webcam_panel, frame.shape, face_box)
            draw_secondary_faces(webcam_panel, frame.shape, secondary_faces)

            fps_counter += 1
            if time.time() - fps_time >= 1.0:
                fps_value = fps_counter / (time.time() - fps_time)
                fps_counter = 0
                fps_time = time.time()

            draw_hud(webcam_panel, mood_tags, gesture_label, fps_value, caption_source, calibrating, calibration_progress,
                     face_analyzer.engine or "off", vision_analyzer.enabled, reaction_source_label)

            combined = np.hstack((webcam_panel, reaction_panel))
            draw_snapshot_flash(combined, snapshot_flash_deadline)
            cv2.imshow(window_name, combined)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                break
            if key == ord("c"):
                with state.lock:
                    state.recalibrate_requested = True
            if key == ord("s"):
                path = save_snapshot(clean_combined)
                snapshot_flash_deadline = time.time() + SNAPSHOT_FLASH_SECONDS
                print(f"saved snapshot to {os.path.relpath(path, BASE_DIR)}")
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        state.running = False
        cap.release()
        cv2.destroyAllWindows()
        face_analyzer.close()
        hand_recognizer.close()


if __name__ == "__main__":
    main()

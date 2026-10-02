"""
Desktop checks that need the installed requirements but no webcam, network,
or model download. Run with: python tests/test_desktop.py
"""

import io
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import captions  # noqa: E402
from reactions import decode_animation, render, render_frames, GIPHY_MAX_FRAMES  # noqa: E402


def _gif(frame_count, duration_ms):
    frames = [Image.new("RGB", (40, 30), (i * 7 % 256, 80, 160)) for i in range(frame_count)]
    buf = io.BytesIO()
    frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    return buf.getvalue()


def test_decode_animation_keeps_timing():
    frames, durations = decode_animation(_gif(10, 50))
    assert len(frames) == 10
    assert frames[0].shape == (30, 40, 3)
    assert abs(sum(durations) - 0.5) < 1e-6


def test_long_animation_is_thinned_without_changing_speed():
    frames, durations = decode_animation(_gif(GIPHY_MAX_FRAMES * 2, 40))
    assert len(frames) == GIPHY_MAX_FRAMES
    assert abs(sum(durations) - GIPHY_MAX_FRAMES * 2 * 0.04) < 1e-6


def test_zero_duration_frames_get_a_sane_default():
    _, durations = decode_animation(_gif(4, 0))
    assert all(d >= 0.02 for d in durations)


def test_render_shapes():
    still = render(np.zeros((100, 120, 3), np.uint8), {"top": "TOP", "bottom": "BOTTOM"}, 640, 480)
    assert still.shape == (480, 640, 3) and still.dtype == np.uint8
    frames = render_frames([np.zeros((50, 50, 3), np.uint8)] * 3, {"top": "A", "bottom": ""}, 640, 480,
                           attribution="Powered By GIPHY")
    assert len(frames) == 3 and frames[0].shape == (480, 640, 3)
    assert frames[0].max() > 200, "caption text is drawn"


def test_placeholder_when_no_image():
    panel = render(None, {"top": "WAIT", "bottom": ""}, 640, 480, placeholder_text="connecting...")
    assert panel.shape == (480, 640, 3)


def test_animation_frame_timing():
    from app import current_animation_frame
    panels = ["p0", "p1", "p2"]
    durations = [0.1, 0.2, 0.3]
    assert current_animation_frame(panels, durations, 0.0, 0.05) == "p0"
    assert current_animation_frame(panels, durations, 0.0, 0.15) == "p1"
    assert current_animation_frame(panels, durations, 0.0, 0.45) == "p2"
    assert current_animation_frame(panels, durations, 0.0, 0.65) == "p0"  # loops
    assert current_animation_frame(["only"], [0.0], 0.0, 9.0) == "only"


def test_static_caption_avoids_immediate_repeat():
    for mood, bank in captions.STATIC_CAPTIONS.items():
        assert len(bank) >= 2, mood
        previous = None
        for _ in range(30):
            caption = captions._static_caption(mood)
            pair = (caption["top"], caption["bottom"])
            assert pair != previous, mood
            previous = pair


def _fake_vision(replies):
    """A VisionMoodAnalyzer whose API calls return the given replies in order."""
    import threading
    analyzer = captions.VisionMoodAnalyzer(api_key="test-key")
    assert analyzer.enabled and analyzer.client is not None
    assert analyzer.client.timeout == captions.OPENAI_TIMEOUT_SECONDS
    assert analyzer.client.max_retries == captions.OPENAI_MAX_RETRIES
    queue = list(replies)
    done = threading.Event()

    def call(prompt, image_b64):
        reply = queue.pop(0)
        done.set()
        return reply

    analyzer._call_api = call
    return analyzer, done


def _vision_step(analyzer, done):
    import time
    done.clear()
    analyzer._last_call_time = 0.0
    analyzer.analyze(np.full((60, 60, 3), 128, np.uint8))
    assert done.wait(2)
    for _ in range(100):
        if not analyzer._busy:
            break
        time.sleep(0.01)
    return analyzer.analyze(None)


def test_vision_tags_clear_on_neutral_answer():
    analyzer, done = _fake_vision(['{"tags": ["smug"]}', '{"tags": []}'])
    assert _vision_step(analyzer, done) == {"smug": 1.0}
    assert _vision_step(analyzer, done) == {}, "a neutral answer must clear old tags"


def test_vision_keeps_tags_on_unreadable_reply_then_expires():
    analyzer, done = _fake_vision(['{"tags": ["chill"]}', "sorry, I can't"])
    assert _vision_step(analyzer, done) == {"chill": 1.0}
    assert _vision_step(analyzer, done) == {"chill": 1.0}
    analyzer._last_tags_time -= captions.VISION_TAG_MAX_AGE_SECONDS + 1
    assert analyzer.analyze(None) == {}, "old tags expire"


def test_vision_call_does_not_block():
    import threading
    import time
    analyzer = captions.VisionMoodAnalyzer(api_key="test-key")
    release = threading.Event()
    analyzer._call_api = lambda prompt, image: (release.wait(5), '{"tags": ["happy"]}')[1]
    start = time.time()
    analyzer.analyze(np.full((60, 60, 3), 128, np.uint8))
    assert time.time() - start < 0.5, "analyze returned without waiting for the API"
    release.set()


def test_damaged_profiles_file_is_skipped_not_fatal():
    import json
    import tempfile
    import identity
    with tempfile.TemporaryDirectory() as folder:
        good = {"name": "guest_1", "encoding": [0.0] * 128, "engine": "blendshapes", "baseline": {}}
        with open(os.path.join(folder, "profiles.json"), "w") as handle:
            json.dump([good, {"name": "broken"}, "junk", {"encoding": [1] * 128}], handle)
        manager = identity.FaceIdentityManager.__new__(identity.FaceIdentityManager)
        manager.available = True
        manager.profiles_dir = folder
        manager.profiles_path = os.path.join(folder, "profiles.json")
        manager.profiles = []
        manager._load()
        assert [p["name"] for p in manager.profiles] == ["guest_1"]
        with open(manager.profiles_path, "w") as handle:
            handle.write('{"not": "a list"}')
        manager._load()
        assert manager.profiles == []


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("\nALL DESKTOP TESTS PASSED")


if __name__ == "__main__":
    main()

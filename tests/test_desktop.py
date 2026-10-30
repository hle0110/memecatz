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


def test_giphy_rotates_phrases_with_depth_limits():
    import tempfile
    import reactions
    calls = []

    class Reply:
        status_code = 200
        def raise_for_status(self):
            pass
        def json(self):
            return {"data": [{"id": f"g{len(calls)}x{i}", "images": {"fixed_height": {"url": "https://media.giphy.com/x.gif"}}}
                             for i in range(3)]}

    def fake_get(url, params=None, **kwargs):
        calls.append(dict(params or {}))
        return Reply()

    real_get = reactions.requests.get
    reactions.requests.get = fake_get
    try:
        with tempfile.TemporaryDirectory() as folder:
            dataset = reactions.AnimalReactionDataset(folder, animal="dog", giphy_api_key="test-key")
            phrases = reactions.queries_for_mood("annoyed", "dog")
            for _ in range(len(phrases) + 1):
                dataset._giphy_entries["annoyed"] = []
                assert dataset.get_for_mood("annoyed")
    finally:
        reactions.requests.get = real_get
    expected = list(phrases) + [phrases[0]]
    assert [(c["q"], c["limit"]) for c in calls] == expected
    assert all(c["offset"] == 0 for c in calls)


def _pet_library_folder(folder):
    """A pet folder with one animated WebP and one mp4, made from plain test frames."""
    import cv2
    os.makedirs(os.path.join(folder, "dog"))
    frames = [Image.new("RGB", (48, 32), (i * 40, 90, 150)) for i in range(4)]
    frames[0].save(os.path.join(folder, "dog", "stop-1.webp"), format="WEBP", save_all=True,
                   append_images=frames[1:], duration=100, loop=0)
    writer = cv2.VideoWriter(os.path.join(folder, "dog", "stop-2.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (48, 32))
    for i in range(30):
        writer.write(np.full((32, 48, 3), i * 8, dtype=np.uint8))
    writer.release()
    item = {"credit": "Mochi, filmed by the owner", "permission": "owner"}
    return {"dog": {
        "stop": [dict(item, file="dog/stop-1.webp", type="webp"), dict(item, file="dog/stop-2.mp4", type="mp4")],
        "sad": [dict(item, file="dog/missing.webp", type="webp")],
    }}


def test_pet_library_comes_first_then_giphy():
    import tempfile
    import reactions
    with tempfile.TemporaryDirectory() as folder:
        pets = os.path.join(folder, "pets")
        library = _pet_library_folder(pets)
        real_dir = reactions.PET_LIBRARY_DIR
        reactions.PET_LIBRARY_DIR = pets
        try:
            source = reactions.ReactionSource(os.path.join(folder, "cache"), animal="dog")
            source.pet_library = library
            giphy_moods = []
            source.dataset.take_for_mood = lambda mood, exclude_key=None: giphy_moods.append(mood)
            source.dataset.get_general = lambda: []
            first, second, third = source.pick(["stop"]), source.pick(["stop"]), source.pick(["stop"], exclude_key="pet:dog/stop-1.webp")
            missing = source.pick(["sad"])
        finally:
            reactions.PET_LIBRARY_DIR = real_dir
    assert [first["key"], second["key"], third["key"]] == ["pet:dog/stop-1.webp", "pet:dog/stop-2.mp4", "pet:dog/stop-2.mp4"]
    assert first["source"] == "pet_library" and first["attribution"] == "Pet clip: Mochi, filmed by the owner"
    assert len(first["frames"]) == 4 and abs(sum(first["durations"]) - 0.4) < 1e-6
    assert second["frames"] and len(second["frames"]) <= GIPHY_MAX_FRAMES and abs(sum(second["durations"]) - 1.5) < 1e-6
    assert giphy_moods == ["sad"], "only the mood whose library file is missing asks Giphy"
    assert missing["source"] == "unavailable"


def _pet_clips():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dev"))
    import pet_clips
    return pet_clips


def test_checked_in_pet_library_passes_clip_checks():
    import reactions
    pet_clips = _pet_clips()
    library = reactions.load_pet_library()
    assert sorted(library["cat"]) == ["approval", "stop"]
    assert sorted(library["dog"]) == ["approval", "sad", "stop", "suspicious"]
    rows = []
    for animal in ("cat", "dog"):
        for mood, items in library[animal].items():
            assert len(reactions.library_entries(library, mood, animal)) == len(items), f"{animal} {mood} has a malformed item"
    for animal, mood, entry in pet_clips.library_clips(library):
        info = pet_clips.measure(entry)
        assert not pet_clips.problems(entry, info), pet_clips.problems(entry, info)
        assert reactions.load_pet_media(entry)[0], f"{entry['file']} does not play"
        rows.append((animal, mood, entry, info))
    with Image.open(pet_clips.PREVIEW_PATH) as preview:
        shown = preview.text.get("clips")
    assert shown == pet_clips.clip_list(rows), "docs/pet_library_preview.png is stale: run python dev/pet_clips.py --preview"


def test_clip_checks_catch_bad_clips():
    import tempfile
    import reactions
    pet_clips = _pet_clips()

    def webp(folder, name, size, count, ms, loop):
        frames = [Image.new("RGB", size, (i * 9 % 256, 60, 120)) for i in range(count)]
        frames[0].save(os.path.join(folder, name), format="WEBP", save_all=True, append_images=frames[1:], duration=ms, loop=loop)
        return {"file": name, "type": "webp", "key": f"pet:{name}", "credit": "x", "permission": "owner"}

    with tempfile.TemporaryDirectory() as folder:
        real_dir = reactions.PET_LIBRARY_DIR
        reactions.PET_LIBRARY_DIR = folder
        try:
            good = webp(folder, "good.webp", (480, 270), 45, 67, 0)
            assert pet_clips.problems(good, pet_clips.measure(good)) == []
            cases = {
                "wide.webp": ((1280, 720), 45, 67, 0, "width 1280"),
                "short.webp": ((480, 270), 10, 67, 0, " s long"),
                "slow.webp": ((480, 270), 12, 250, 0, " fps"),
                "once.webp": ((480, 270), 45, 67, 1, "loop count 1"),
            }
            for name, (size, count, ms, loop, expected) in cases.items():
                entry = webp(folder, name, size, count, ms, loop)
                found = pet_clips.problems(entry, pet_clips.measure(entry))
                assert any(expected in p for p in found), (name, found)
        finally:
            reactions.PET_LIBRARY_DIR = real_dir


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print("\nALL DESKTOP TESTS PASSED")


if __name__ == "__main__":
    main()

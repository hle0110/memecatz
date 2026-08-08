import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from mood import combine, top_tags, primary_tag


def test_neutral_default():
    mood = combine(fer_scores={"neutral": 0.95, "happy": 0.02, "sad": 0.01, "angry": 0.01,
                                "surprise": 0.005, "fear": 0.003, "disgust": 0.002})
    assert primary_tag(mood) == "neutral"


def test_smile_beats_neutral():
    fer_scores = {"neutral": 0.6, "happy": 0.35, "sad": 0.02, "angry": 0.01,
                  "surprise": 0.01, "fear": 0.005, "disgust": 0.005}
    au_tags = {"smile": 0.85}
    mood = combine(fer_scores=fer_scores, au_tags=au_tags)
    assert primary_tag(mood) == "happy"


def test_gesture_dominates():
    fer_scores = {"neutral": 0.9, "happy": 0.03, "sad": 0.02, "angry": 0.02,
                  "surprise": 0.01, "fear": 0.01, "disgust": 0.01}
    gesture_tags = {"approval": 1.0, "happy": 0.5}
    mood = combine(fer_scores=fer_scores, gesture_tags=gesture_tags)
    assert primary_tag(mood) == "approval"


def test_angry_combo():
    fer_scores = {"neutral": 0.3, "angry": 0.55, "sad": 0.05, "happy": 0.02,
                  "surprise": 0.02, "fear": 0.03, "disgust": 0.03}
    au_tags = {"brow_furrow": 0.7}
    mood = combine(fer_scores=fer_scores, au_tags=au_tags)
    tags = top_tags(mood)
    assert tags[0][0] == "angry"


def test_top_tags_limit_and_floor():
    mood = {"happy": 0.9, "surprise": 0.5, "mischief": 0.13, "sad": 0.02}
    tags = top_tags(mood, limit=2)
    assert len(tags) == 2
    assert tags[0][0] == "happy" and tags[1][0] == "surprise"


def test_vision_tags_add_nuance():
    fer_scores = {"neutral": 0.3, "happy": 0.3, "sad": 0.1, "angry": 0.1,
                  "surprise": 0.1, "fear": 0.05, "disgust": 0.05}
    au_tags = {"smirk": 0.4}
    vision_tags = {"mocking": 1.0}
    mood = combine(fer_scores=fer_scores, au_tags=au_tags, vision_tags=vision_tags)
    tags = top_tags(mood)
    assert tags[0][0] == "mocking"


ALL_TESTS = [
    test_neutral_default,
    test_smile_beats_neutral,
    test_gesture_dominates,
    test_angry_combo,
    test_top_tags_limit_and_floor,
    test_vision_tags_add_nuance,
]

if __name__ == "__main__":
    for test in ALL_TESTS:
        test()
        print(f"{test.__name__}: OK")
    print("\nALL MOOD TESTS PASSED")

"""
Checks that the web app (web/*.js) and the desktop app agree: same captions,
same Giphy search phrases, same mood math, same gesture rules, same emotion
crop rules, and the same security policy in worker.js and web/_headers.

Needs Node 22 or newer, and the desktop requirements (it imports vision.py).
Run with: python tests/test_web_parity.py
"""

import json
import os
import random
import re
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from mood import combine, top_tags, AU_TAG_TO_MOOD  # noqa: E402
from captions import STATIC_CAPTIONS  # noqa: E402
from reactions import MOOD_QUERIES, DOG_BREEDS, queries_for_mood  # noqa: E402
from vision import classify_gesture, GESTURE_TO_TAGS, BLENDSHAPE_NAMES, EmotionDetector, EMOTION_PADDING, EMOTION_OFFSET_X, EMOTION_OFFSET_Y  # noqa: E402

FER = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
GESTURE_TAGS = sorted({t for m in GESTURE_TO_TAGS.values() for t in m})


def _random_mood_cases(rng, n=400):
    cases = []
    for _ in range(n):
        case = {}
        if rng.random() < 0.8:
            case["ferScores"] = {f: round(rng.random(), 3) for f in FER}
        if rng.random() < 0.8:
            case["auTags"] = {a: round(rng.random(), 3) for a in AU_TAG_TO_MOOD if rng.random() < 0.4}
        if rng.random() < 0.4:
            case["gestureTags"] = {g: round(rng.random(), 3) for g in GESTURE_TAGS if rng.random() < 0.3}
        if rng.random() < 0.2:
            case["visionTags"] = {"mocking": round(rng.random(), 3)}
        cases.append(case)
    return cases


def _hand(wrist, mcp, pip, tip, thumb_mcp, thumb_tip, middle_mcp_y):
    """21 hand points from a simple description, enough for the rules."""
    pts = [(0.0, 0.0)] * 21
    pts[0] = wrist
    pts[2] = thumb_mcp
    pts[4] = thumb_tip
    for (m, p, t), (mi, pi, ti) in zip(zip(mcp, pip, tip), [(5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20)]):
        pts[mi], pts[pi], pts[ti] = m, p, t
    pts[9] = (pts[9][0], middle_mcp_y)
    return [list(p) for p in pts]


def _random_hands(rng, n=600):
    hands = []
    for _ in range(n):
        wx, wy = rng.uniform(200, 440), rng.uniform(300, 420)
        mcp, pip, tip = [], [], []
        for k in range(4):
            mx, my = wx - 45 + 30 * k, wy - rng.uniform(70, 110)
            px, py = mx + rng.uniform(-8, 8), my - rng.uniform(20, 40)
            ext = rng.random() < 0.5
            tx, ty = (px + rng.uniform(-6, 6), py - rng.uniform(15, 45)) if ext else (px + rng.uniform(-10, 10), py + rng.uniform(10, 45))
            mcp.append((mx, my)); pip.append((px, py)); tip.append((tx, ty))
        tmx, tmy = wx - 50, wy - 40
        thumb_out = rng.random() < 0.5
        ang = rng.uniform(0, 6.283)
        dist = rng.uniform(80, 140) if thumb_out else rng.uniform(40, 70)
        ttx, tty = wx + dist * np.cos(ang), wy + dist * np.sin(ang)
        hands.append(_hand((wx, wy), mcp, pip, tip, (tmx, tmy), (ttx, tty), mcp[1][1]))
    return hands


def _run_node(request):
    proc = subprocess.run(
        ["node", os.path.join(ROOT, "tests", "web_dump.mjs")],
        input=json.dumps(request), capture_output=True, text=True, cwd=ROOT, timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)
    return json.loads(proc.stdout)


def main():
    rng = random.Random(1234)
    mood_cases = _random_mood_cases(rng)
    hands = _random_hands(rng)
    boxes = [[rng.randint(-30, 600), rng.randint(-30, 440), rng.randint(20, 300), rng.randint(20, 300)] for _ in range(300)]
    pixels = [[rng.randint(0, 255) for _ in range(3)] for _ in range(2000)]
    js = _run_node({"moodCases": mood_cases, "gestureCases": hands, "boxes": boxes, "pixels": pixels})
    failures = []

    def check(name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name + ("" if ok else f": {detail}"))
        if not ok:
            failures.append(name)

    py_captions = {k: [list(p) for p in v] for k, v in STATIC_CAPTIONS.items()}
    check("captions identical", py_captions == js["captions"])
    as_lists = {a: {m: [list(p) for p in ps] for m, ps in t.items()} for a, t in MOOD_QUERIES.items()}
    check("giphy search phrases identical", as_lists == js["moodQueries"])
    check("query fallback identical",
          {m: [list(p) for p in queries_for_mood(m, "dog")] for m in js["queries"]} == js["queries"])
    bad_depth = [(a, m, p) for a, t in MOOD_QUERIES.items() for m, ps in t.items() for p, d in ps
                 if not (isinstance(d, int) and 1 <= d <= 25) or a not in p.split()]
    check("every phrase names its animal and has a sane depth", not bad_depth, bad_depth)
    check("dog breed list identical", list(DOG_BREEDS) == js["dogBreeds"])
    check("gesture tag map identical", GESTURE_TO_TAGS == js["gestureToTags"])
    check("expression tag map identical", AU_TAG_TO_MOOD == js["auTagToMood"])
    check("blendshape names identical", list(BLENDSHAPE_NAMES) == js["featureKeys"])

    reachable = set(FER) | {t for m in AU_TAG_TO_MOOD.values() for t in m} | set(GESTURE_TAGS)
    missing_captions = sorted(t for t in reachable if t not in STATIC_CAPTIONS)
    missing_queries = sorted(f"{a}:{t}" for a in ("cat", "dog") for t in reachable if t not in MOOD_QUERIES[a])
    check("every reachable mood has captions", not missing_captions, missing_captions)
    check("every reachable mood has a search phrase", not missing_queries, missing_queries)

    bad = 0
    for case, res in zip(mood_cases, js["moods"]):
        v = combine(fer_scores=case.get("ferScores"), au_tags=case.get("auTags"),
                    gesture_tags=case.get("gestureTags"), vision_tags=case.get("visionTags"))
        if v != res["vector"] or [list(t) for t in top_tags(v, 3, 0.12)] != res["top"]:
            bad += 1
    check(f"mood math identical on {len(mood_cases)} random cases", bad == 0, f"{bad} differ")

    py_gestures = [classify_gesture({i: tuple(p) for i, p in enumerate(h)}) for h in hands]
    diff = sum(1 for a, b in zip(py_gestures, js["gestures"]) if a != b)
    kinds = sorted({g for g in py_gestures if g})
    check(f"gesture rules identical on {len(hands)} random hands ({', '.join(kinds)})", diff == 0, f"{diff} differ")

    sq_bad = crop_bad = 0
    for box, sq, crop in zip(boxes, js["squares"], js["crops"]):
        x, y, w, h = EmotionDetector._to_square(tuple(box))
        if [x, y, w, h] != sq:
            sq_bad += 1
        padded = np.zeros((480 + 2 * EMOTION_PADDING, 640 + 2 * EMOTION_PADDING), dtype=np.uint8)
        x1 = max(0, x - EMOTION_OFFSET_X + EMOTION_PADDING)
        y1 = max(0, y - EMOTION_OFFSET_Y + EMOTION_PADDING)
        x2 = x + w + EMOTION_OFFSET_X + EMOTION_PADDING
        y2 = y + h + EMOTION_OFFSET_Y + EMOTION_PADDING
        region = padded[y1:y2, x1:x2]  # numpy clips the far edges
        if [x1, y1, x1 + region.shape[1], y1 + region.shape[0]] != crop:
            crop_bad += 1
    check("emotion square box identical", sq_bad == 0, f"{sq_bad} differ")
    check("emotion crop rectangle identical", crop_bad == 0, f"{crop_bad} differ")

    arr = np.array([[p[::-1] for p in pixels]], dtype=np.uint8)  # RGB -> BGR for cv2
    cv_gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)[0].tolist()
    check("gray conversion identical to cv2", cv_gray == js["grays"],
          f"{sum(1 for a, b in zip(cv_gray, js['grays']) if a != b)} differ")

    check("time-scaled smoothing keeps the reference rate", abs(js["alphaSame"] - 0.35) < 1e-12)
    check("two half steps equal one full step", abs(js["alphaTwoHalves"] - 0.35) < 1e-12)

    with open(os.path.join(ROOT, "worker.js")) as handle:
        worker = handle.read()
    with open(os.path.join(ROOT, "web", "_headers")) as handle:
        headers = handle.read()
    m = re.search(r"const CSP = \[(.*?)\]\.join", worker, re.S)
    worker_csp = "; ".join(re.findall(r'"([^"]+)"', m.group(1))) if m else None
    hdr = re.search(r"Content-Security-Policy: (.+)", headers)
    check("security policy same in worker.js and web/_headers", hdr is not None and worker_csp == hdr.group(1).strip())
    hosts = set(re.findall(r"https://[a-z0-9.*-]+", hdr.group(1))) if hdr else set()
    web_src = "".join(open(os.path.join(ROOT, "web", f)).read() for f in os.listdir(os.path.join(ROOT, "web")) if f.endswith(".js"))
    used = set(re.findall(r"https://([a-z0-9.-]+)", web_src))
    unlisted = sorted(h for h in used if not any(
        p == f"https://{h}" or (p.startswith("https://*.") and h.endswith(p[len("https://*"):])) for p in hosts))
    check("every host the page calls is allowed by the policy", not unlisted, unlisted)

    print(f"\n{len(failures)} failed" if failures else "\nall checks passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

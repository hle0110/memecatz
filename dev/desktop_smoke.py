"""
Runs the desktop app end to end without a webcam or a window: the camera is
replaced by dev/output/media/cam.mp4, the window by a frame recorder, and
keys are pressed on a timer (c at 12s, s at 16s, q at 26s).

Setup (once): python dev/make_test_media.py
Run from the repo root with the app's environment, for example:
  .venv/bin/python dev/desktop_smoke.py              backup photos, no keys
  .venv/bin/python dev/desktop_smoke.py --mock-giphy  fake Giphy search results, real Giphy media

Saves a few frames and the snapshot under dev/output/desktop/ to look at.
"""

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "output", "desktop")
VIDEO = os.path.join(HERE, "output", "media", "cam.mp4")
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import requests  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--mock-giphy", action="store_true")
args = parser.parse_args()
if not os.path.exists(VIDEO):
    raise SystemExit("missing test video, run: python dev/make_test_media.py")

real_capture = cv2.VideoCapture


class FakeCamera:
    def __init__(self, *_):
        self.capture = real_capture(VIDEO)

    def isOpened(self):
        return self.capture.isOpened()

    def set(self, *_):
        return True

    def read(self):
        ok, frame = self.capture.read()
        if not ok:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        time.sleep(1 / 15)
        # The app mirrors every frame, so pre-mirror to end up like a webcam.
        return ok, cv2.flip(frame, 1)

    def release(self):
        self.capture.release()


started = time.time()
frames = []
keys = {12: "c", 16: "s", 26: "q"}
pressed = set()


def imshow(_name, image):
    if len(frames) % 30 == 0:
        frames.append((time.time() - started, image.copy()))
    else:
        frames.append((time.time() - started, None))


def wait_key(_delay):
    elapsed = time.time() - started
    for at, key in keys.items():
        if elapsed >= at and at not in pressed:
            pressed.add(at)
            print(f"[smoke] pressing {key} at {elapsed:.1f}s")
            return ord(key)
    return -1


cv2.VideoCapture = FakeCamera
cv2.imshow = imshow
cv2.waitKey = wait_key
cv2.namedWindow = lambda *a, **k: None
cv2.resizeWindow = lambda *a, **k: None
cv2.destroyAllWindows = lambda: None
cv2.getWindowProperty = lambda *a: 1

searches = []
if args.mock_giphy:
    os.environ["GIPHY_API_KEY"] = "TESTKEY"
    real_get = requests.get
    ids = ["JIX9t2j0ZTN9S", "mlvseq9yvZhba", "VbnUQpnihPSIgIXuZv", "3oriO0OEd9QIDdllqo"]

    class FakeSearch:
        status_code = 200

        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": i, "images": {"fixed_height": {"url": f"https://media.giphy.com/media/{i}/200.gif"}}}
                             for i in ids]}

    def fake_get(url, *a, **k):
        if url.startswith("https://api.giphy.com"):
            searches.append(k.get("params", {}).get("q"))
            return FakeSearch()
        return real_get(url, *a, **k)

    requests.get = fake_get

import reactions  # noqa: E402
if args.mock_giphy:
    reactions.requests.get = requests.get
import app  # noqa: E402

os.makedirs(OUT, exist_ok=True)
app.SNAPSHOTS_DIR = os.path.join(OUT, "snapshots")
app.REACTION_CACHE_DIR = os.path.join(OUT, "reaction_cache")
sys.argv = ["app.py"]
app.main()

shown = [f for f in frames if f[1] is not None]
if len(frames) > 1:
    print(f"[smoke] {len(frames)} frames shown, about {len(frames) / (frames[-1][0] - frames[0][0]):.1f} fps")
for elapsed, image in shown[::3]:
    cv2.imwrite(os.path.join(OUT, f"frame_{int(elapsed):02d}s.jpg"), image)
print("[smoke] frames and snapshot saved under", OUT)
if args.mock_giphy:
    print("[smoke] giphy searches:", searches)

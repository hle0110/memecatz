"""
Builds a fake webcam video for the smoke tests: a smiling face for 8 seconds,
then the same face with a thumbs up, peace sign, pointing finger, and fist.

Sources (downloaded once into dev/output/media/):
  justin.jpg from github.com/justinshenk/fer (MIT), a smiling face
  hand gesture photos from storage.googleapis.com/mediapipe-assets
Needs ffmpeg on PATH and Pillow.

Run from the repo root: python dev/make_test_media.py
Writes dev/output/media/cam.y4m (for Chromium) and cam.mp4 (for the desktop test).
"""

import os
import subprocess
import urllib.request

from PIL import Image

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "media")
SOURCES = {
    "justin.jpg": "https://raw.githubusercontent.com/justinshenk/fer/master/justin.jpg",
    "thumb_up.jpg": "https://storage.googleapis.com/mediapipe-assets/thumb_up.jpg",
    "victory.jpg": "https://storage.googleapis.com/mediapipe-assets/victory.jpg",
    "pointing_up.jpg": "https://storage.googleapis.com/mediapipe-assets/pointing_up.jpg",
    "fist.jpg": "https://storage.googleapis.com/mediapipe-assets/fist.jpg",
}
TIMELINE = [("face", 8), ("thumb_up", 4), ("face", 2), ("victory", 4), ("face", 2), ("pointing_up", 4), ("face", 2), ("fist", 4)]


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, url in SOURCES.items():
        path = os.path.join(OUT, name)
        if not os.path.exists(path):
            urllib.request.urlretrieve(url, path)

    face = Image.open(os.path.join(OUT, "justin.jpg")).convert("RGB")
    canvas = lambda: Image.new("RGB", (640, 480), (70, 80, 95))  # noqa: E731
    frame = canvas()
    frame.paste(face.resize((360, 363)), (140, 60))
    frame.save(os.path.join(OUT, "f_face.png"))
    for gesture in ("thumb_up", "victory", "pointing_up", "fist"):
        frame = canvas()
        frame.paste(face.resize((300, 302)), (20, 90))
        hand = Image.open(os.path.join(OUT, f"{gesture}.jpg")).convert("RGB")
        hand.thumbnail((300, 330))
        frame.paste(hand, (330, 80))
        frame.save(os.path.join(OUT, f"f_{gesture}.png"))

    concat = os.path.join(OUT, "timeline.txt")
    with open(concat, "w") as handle:
        for name, seconds in TIMELINE:
            handle.write(f"file 'f_{name}.png'\nduration {seconds}\n")
        handle.write(f"file 'f_{TIMELINE[-1][0]}.png'\n")

    y4m = os.path.join(OUT, "cam.y4m")
    mp4 = os.path.join(OUT, "cam.mp4")
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "concat", "-i", concat,
                    "-vf", "fps=15,format=yuv420p", "-pix_fmt", "yuv420p", y4m], check=True, cwd=OUT)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", y4m, "-c:v", "libx264",
                    "-pix_fmt", "yuv420p", "-crf", "20", mp4], check=True)
    print("wrote", y4m)
    print("wrote", mp4)


if __name__ == "__main__":
    main()

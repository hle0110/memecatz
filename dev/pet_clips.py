"""
Checks every clip in the pet library (web/pets/library.json) and draws a
preview sheet of them all.

  python dev/pet_clips.py            check every clip, exit 1 on a problem
  python dev/pet_clips.py --preview  also write docs/pet_library_preview.png

tests/test_desktop.py runs the same checks, so a clip that breaks them cannot
merge. The loop seam (how far the last frame is from the first, compared with
a normal step between frames) is reported but never fails a clip: some moods,
like a high five, cannot loop smoothly.
"""

import argparse
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402
from PIL.PngImagePlugin import PngInfo  # noqa: E402

import reactions  # noqa: E402

MIN_WIDTH, MAX_WIDTH = 360, 640
MAX_HEIGHT = 640
MIN_SECONDS, MAX_SECONDS = 2.0, 4.0
MIN_FPS, MAX_FPS = 10.0, 30.0
MAX_BYTES = 500 * 1024
SEAM_WARN = 3.0
PREVIEW_PATH = os.path.join(ROOT, "docs", "pet_library_preview.png")
FONT_PATH = os.path.join(ROOT, "assets", "DejaVuSans-Bold.ttf")


def library_clips(library=None):
    """(animal, mood, entry) for every valid item, in file order."""
    library = reactions.load_pet_library() if library is None else library
    out = []
    for animal in ("cat", "dog"):
        table = library.get(animal) if isinstance(library, dict) else None
        for mood in (table or {}):
            out.extend((animal, mood, entry) for entry in reactions.library_entries(library, mood, animal))
    return out


def _all_frames(entry):
    """Every frame and its duration, without the playback thinning."""
    path = os.path.join(reactions.PET_LIBRARY_DIR, *entry["file"].split("/"))
    if entry["type"] == "mp4":
        return reactions.decode_video(path, max_frames=reactions.PET_MAX_VIDEO_FRAMES)
    with open(path, "rb") as handle:
        return reactions.decode_animation(handle.read(), max_frames=10 ** 6)


def _seam_ratio(frames):
    """Last-to-first difference over the median frame-to-frame difference."""
    small = [cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 36)).astype(np.float32) for f in frames]
    steps = [float(np.mean(np.abs(b - a))) for a, b in zip(small, small[1:])]
    typical = float(np.median(steps)) if steps else 0.0
    seam = float(np.mean(np.abs(small[0] - small[-1])))
    return seam / typical if typical > 0 else 0.0


def measure(entry):
    """Size, timing, loop setting and seam of one clip, plus its frames."""
    path = os.path.join(reactions.PET_LIBRARY_DIR, *entry["file"].split("/"))
    frames, durations = _all_frames(entry)
    info = {"bytes": os.path.getsize(path), "frames": frames or []}
    if not frames:
        return info
    seconds = float(sum(durations))
    info.update({
        "width": frames[0].shape[1],
        "height": frames[0].shape[0],
        "count": len(frames),
        "seconds": seconds,
        "fps": len(frames) / seconds if seconds > 0 else 0.0,
        "seam": _seam_ratio(frames),
    })
    if entry["type"] == "webp":
        with Image.open(path) as image:
            info["loop"] = image.info.get("loop")
    return info


def problems(entry, info):
    """Everything about a clip that breaks the rules in docs/PET_LIBRARY.md."""
    if not info["frames"]:
        return [f"{entry['file']} does not decode"]
    found = []
    if not MIN_WIDTH <= info["width"] <= MAX_WIDTH:
        found.append(f"width {info['width']} px, keep it {MIN_WIDTH} to {MAX_WIDTH} (about 480)")
    if info["height"] > MAX_HEIGHT:
        found.append(f"height {info['height']} px, keep it {MAX_HEIGHT} or less")
    if not MIN_SECONDS <= info["seconds"] <= MAX_SECONDS:
        found.append(f"{info['seconds']:.2f} s long, keep it {MIN_SECONDS:g} to {MAX_SECONDS:g} s")
    if not MIN_FPS <= info["fps"] <= MAX_FPS:
        found.append(f"{info['fps']:.1f} fps, keep it {MIN_FPS:g} to {MAX_FPS:g}")
    if info["bytes"] >= MAX_BYTES:
        found.append(f"{info['bytes']} bytes, keep it under {MAX_BYTES}")
    if entry["type"] == "webp" and info.get("loop") != 0:
        found.append(f"loop count {info.get('loop')}, it must be 0 (loops forever)")
    return [f"{entry['file']}: {p}" for p in found]


def summary(animal, mood, entry, info):
    if not info["frames"]:
        return f"{animal} {mood}  {entry['file']}  does not decode"
    seam = f"seam {info['seam']:.1f}x" + (" (visible jump)" if info["seam"] > SEAM_WARN else "")
    return (f"{animal} {mood}  {entry['file']}  {info['width']}x{info['height']}  {info['seconds']:.2f} s  "
            f"{info['fps']:.0f} fps  {info['bytes'] / 1024:.0f} KB  {seam}")


def clip_list(rows):
    """The clips a preview shows, stored in the PNG so a test can spot a stale one."""
    return ",".join(f"{animal}/{mood}/{entry['file']}" for animal, mood, entry, _ in rows)


def write_preview(rows, path=PREVIEW_PATH):
    """One row per clip: first, middle and last frame, with its numbers."""
    tile_w, tile_h, label_h, pad = 240, 135, 44, 8
    width = 3 * tile_w + 4 * pad
    sheet = Image.new("RGB", (width, max(1, len(rows)) * (tile_h + label_h + pad) + pad), (18, 19, 26))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype(FONT_PATH, 13)
    if not rows:
        draw.text((pad, pad), "the pet library is empty", font=font, fill=(220, 220, 220))
    for r, (animal, mood, entry, info) in enumerate(rows):
        top = pad + r * (tile_h + label_h + pad)
        frames = info["frames"]
        picks = [frames[0], frames[len(frames) // 2], frames[-1]] if frames else []
        for c, frame in enumerate(picks):
            tile = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            tile.thumbnail((tile_w, tile_h))
            sheet.paste(tile, (pad + c * (tile_w + pad), top))
        line = summary(animal, mood, entry, info)
        draw.text((pad, top + tile_h + 6), line, font=font, fill=(235, 235, 235))
        draw.text((pad, top + tile_h + 24), f"Pet clip: {entry['credit']}  ({entry['permission']})", font=font, fill=(160, 166, 180))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    meta = PngInfo()
    meta.add_text("clips", clip_list(rows))
    sheet.save(path, optimize=True, pnginfo=meta)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--preview", action="store_true", help=f"write {os.path.relpath(PREVIEW_PATH, ROOT)}")
    args = parser.parse_args()

    rows, failures = [], []
    for animal, mood, entry in library_clips():
        info = measure(entry)
        rows.append((animal, mood, entry, info))
        failures.extend(problems(entry, info))
        print(summary(animal, mood, entry, info))
    if not rows:
        print("the pet library is empty")
    if args.preview:
        print(f"wrote {os.path.relpath(write_preview(rows), ROOT)}")
    for line in failures:
        print(f"PROBLEM {line}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()

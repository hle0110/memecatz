"""
Shows, as contact sheets, exactly what the app's Giphy phrases return today,
so the MOOD_QUERIES table in reactions.py and web/reactions.js can be checked
by eye. Every result the app can show is drawn with a green label; a few
results just past each phrase's depth are drawn with a red label, to judge
whether a depth could go up or must come down.

What to look for: real cats or dogs only. No drawings, cartoons, 3D or
generated renders, people, or other animals, and the mood must read clearly.

Run from the repo root (Pillow and requests are in requirements.txt):
  python dev/giphy_audit.py --key YOUR_GIPHY_KEY --animal cat
  python dev/giphy_audit.py --key YOUR_GIPHY_KEY --animal dog --moods happy,sad

One API call per distinct phrase (cat has about 30, dog about 35). A beta key
allows 100 calls an hour, shared with the live site, so audit one animal per
hour. Results are kept only in dev/output/audit for this review.
"""

import argparse
import concurrent.futures as cf
import io
import json
import os
import sys
import urllib.parse
import urllib.request

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from reactions import MOOD_QUERIES  # noqa: E402

OUT = os.path.join(HERE, "output", "audit")
FONT = os.path.join(os.path.dirname(HERE), "assets", "DejaVuSans-Bold.ttf")
TW, TH, COLS, PER_SHEET = 160, 140, 8, 48


def search(key, phrase, limit):
    url = "https://api.giphy.com/v1/gifs/search?" + urllib.parse.urlencode(
        {"api_key": key, "q": phrase, "limit": limit, "offset": 0, "rating": "g", "lang": "en"})
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.load(response)["data"]


def thumb(item):
    still = item["images"].get("fixed_width_still", {}).get("url") or item["images"]["fixed_width_small_still"]["url"]
    try:
        with urllib.request.urlopen(still, timeout=20) as response:
            image = Image.open(io.BytesIO(response.read())).convert("RGB")
    except Exception:
        image = Image.new("RGB", (200, 150), (120, 0, 0))
    image.thumbnail((TW, TH - 18))
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", default=os.environ.get("GIPHY_API_KEY"))
    parser.add_argument("--animal", choices=["cat", "dog"], default="cat")
    parser.add_argument("--moods", help="comma separated, default all")
    parser.add_argument("--extra", type=int, default=3, help="results to show past each depth")
    parser.add_argument("--max-calls", type=int, default=60)
    args = parser.parse_args()
    if not args.key:
        raise SystemExit("pass --key or set GIPHY_API_KEY")

    table = MOOD_QUERIES[args.animal]
    moods = args.moods.split(",") if args.moods else list(table)
    phrases = []
    for mood in moods:
        for phrase, depth in table[mood]:
            if phrase not in [p for p, _ in phrases]:
                phrases.append((phrase, depth))
    if len(phrases) > args.max_calls:
        raise SystemExit(f"{len(phrases)} phrases would need {len(phrases)} calls, over --max-calls {args.max_calls}")

    tiles = []
    for phrase, depth in phrases:
        for i, item in enumerate(search(args.key, phrase, depth + args.extra)):
            tiles.append((f"{phrase} #{i + 1}", i < depth, item))
    print(f"{len(phrases)} API calls, {len(tiles)} results")

    os.makedirs(OUT, exist_ok=True)
    font = ImageFont.truetype(FONT, 11)
    with cf.ThreadPoolExecutor(16) as pool:
        images = list(pool.map(thumb, [t[2] for t in tiles]))
    for n in range(0, len(tiles), PER_SHEET):
        chunk = list(zip(tiles[n:n + PER_SHEET], images[n:n + PER_SHEET]))
        rows = (len(chunk) + COLS - 1) // COLS
        sheet = Image.new("RGB", (COLS * (TW + 2), rows * (TH + 2)), (200, 200, 200))
        draw = ImageDraw.Draw(sheet)
        for k, ((label, used, _), image) in enumerate(chunk):
            x, y = (k % COLS) * (TW + 2), (k // COLS) * (TH + 2)
            sheet.paste(Image.new("RGB", (TW, TH), "white"), (x, y))
            sheet.paste(image, (x + (TW - image.width) // 2, y + 18))
            draw.rectangle([x, y, x + TW - 1, y + 15], fill=(30, 140, 60) if used else (190, 40, 40))
            draw.text((x + 3, y + 1), label[:24], fill="white", font=font)
        path = os.path.join(OUT, f"{args.animal}_{n // PER_SHEET + 1}.jpg")
        sheet.save(path, quality=86)
        print("wrote", path)


if __name__ == "__main__":
    main()

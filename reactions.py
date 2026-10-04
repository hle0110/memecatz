"""
Everything about getting a real reaction on screen: real cat or dog content
from Giphy (mood matched, animated) and a zero-setup real photo backup (The Cat
API / Dog CEO API), picking one for the current mood, and rendering the caption
onto it. No generated, drawn, or fake imagery is ever used as a substitute.

Giphy content follows Giphy's integration rules: search results and media are
kept in memory for the current session only, never written to disk, and used
in the order Giphy returns them.
"""

import io
import os
import glob
import json
import random
import shutil
import time
import hashlib
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageSequence

from utils import fit_to_panel

GIPHY_SEARCH_URL = "https://api.giphy.com/v1/gifs/search"
CAT_API_SEARCH_URL = "https://api.thecatapi.com/v1/images/search"
CAT_API_DEMO_KEY = "DEMO-API-KEY"
DOG_BREED_URL = "https://dog.ceo/api/breed/{breed}/images/random/{count}"
REQUEST_TIMEOUT_SECONDS = 8
FETCH_RETRY_COOLDOWN = 1800
GIPHY_RETRY_COOLDOWN = 30
RESULTS_PER_MOOD = 12
GENERAL_KEY = "_general"

# Animated reactions are decoded into memory, so cap their size.
GIPHY_MAX_BYTES = 8 * 1024 * 1024
GIPHY_MAX_FRAMES = 48
GIF_DEFAULT_FRAME_SECONDS = 0.1
GIF_MIN_FRAME_SECONDS = 0.02

# Pet breeds only. Dog CEO's "any breed" endpoint also returns wild canids such
# as African wild dogs and dholes. Every entry was checked against
# https://dog.ceo/api/breeds/list/all and matches web/reactions.js.
DOG_BREEDS = (
    "beagle", "boxer", "bulldog/french", "cavapoo", "chihuahua", "cockapoo", "collie/border",
    "corgi/cardigan", "dachshund", "dalmatian", "frise/bichon", "german/shepherd", "havanese",
    "husky", "labrador", "malamute", "maltese", "papillon", "pembroke", "pomeranian",
    "poodle/toy", "pug", "retriever/golden", "samoyed", "shiba", "shihtzu",
)

# Giphy search phrases per mood, the same as MOOD_QUERIES in web/reactions.js.
# Each phrase was chosen by looking at what Giphy actually returns for it:
# real animals showing that mood, not drawings or people. The number is how
# many of its top results to use. Past that point a phrase's results drift
# off-mood or to cartoons, so the app moves on to the next phrase instead of
# digging deeper.
# Checked by eye in October 2026. Giphy's results change over time, so it is
# worth looking at them again every few months.
MOOD_QUERIES = {
    "cat": {
        "happy": [("cat smiling", 12), ("happy cat", 7)],
        "sad": [("crying cat", 11), ("sad cat", 11)],
        "angry": [("angry cat", 12)],
        "surprise": [("shocked cat", 12), ("surprised cat", 12)],
        "fear": [("scared cat", 12)],
        "disgust": [("disgusted cat", 7)],
        "neutral": [("cat stare", 11)],
        "smug": [("cat smirk", 10)],
        "confused": [("cat huh", 12), ("confused cat", 5)],
        "mischief": [("cat knocking things off", 12), ("sneaky cat", 9)],
        "annoyed": [("annoyed cat", 12), ("unimpressed cat", 12), ("grumpy cat", 12)],
        "approval": [("cat thumbs up", 3), ("cat smiling", 12)],
        "disapproval": [("cat side eye", 9)],
        "chill": [("relaxed cat", 12)],
        "suspicious": [("suspicious cat", 5), ("cat side eye", 9)],
        "triumph": [("cat winning", 3), ("cat smiling", 12)],
        "anxious": [("scared cat", 12)],
        "bored": [("bored cat", 12), ("cat yawning", 11)],
        "mocking": [("cat laughing", 10), ("sassy cat", 12)],
        "stop": [("cat high five", 3), ("cat no", 4), ("cat says no", 3)],
        "determined": [("serious cat", 12), ("cat butt wiggle", 10)],
        "focused": [("cat butt wiggle", 10), ("serious cat", 12)],
    },
    "dog": {
        "happy": [("dog smiling", 12), ("happy dog", 5)],
        "sad": [("sad dog", 5), ("dog crying", 2), ("depressed dog", 1)],
        "angry": [("angry dog", 12)],
        "surprise": [("shocked dog", 10), ("surprised dog", 9)],
        "fear": [("scared dog", 9)],
        "disgust": [("disgusted dog", 7)],
        "neutral": [("dog stare", 12), ("dog staring", 12)],
        "smug": [("smug dog", 6)],
        "confused": [("dog head tilt", 12), ("confused dog", 12)],
        "mischief": [("dog zoomies", 10), ("guilty dog", 2)],
        "annoyed": [("dog side eye", 5), ("unimpressed dog", 3), ("dog eye roll", 2)],
        "approval": [("dog yes", 4), ("dog thumbs up", 2), ("dog smiling", 12)],
        "disapproval": [("dog side eye", 5), ("dog judging", 6)],
        "chill": [("lazy dog", 10), ("sleepy dog", 3)],
        "suspicious": [("suspicious dog", 2), ("dog side eye", 5)],
        "triumph": [("excited dog", 12), ("dog winning", 3)],
        "anxious": [("nervous dog", 5), ("scared dog", 9)],
        "bored": [("bored dog", 12), ("dog yawning", 6)],
        "mocking": [("dog grin", 5), ("dog smiling", 12)],
        "stop": [("dog high five", 8)],
        "determined": [("serious dog", 12)],
        "focused": [("serious dog", 12), ("dog staring", 12)],
    },
}


def queries_for_mood(mood_tag, animal="cat"):
    """[(phrase, depth), ...] for a mood, with a plain fallback for unknown moods."""
    table = MOOD_QUERIES["dog" if animal == "dog" else "cat"]
    return table.get(mood_tag) or [(f"{mood_tag.replace('_', ' ')} {animal}", 8)]


def decode_animation(data, max_frames=GIPHY_MAX_FRAMES):
    """GIF bytes -> (list of BGR frames, list of frame durations in seconds).

    Long animations are thinned evenly to max_frames, with the dropped frames'
    time added to the kept ones so playback speed stays the same.
    """
    with Image.open(io.BytesIO(data)) as image:
        frames = []
        durations = []
        for frame in ImageSequence.Iterator(image):
            seconds = frame.info.get("duration", 0) / 1000.0
            if seconds < GIF_MIN_FRAME_SECONDS:
                seconds = GIF_DEFAULT_FRAME_SECONDS
            frames.append(cv2.cvtColor(np.array(frame.convert("RGB")), cv2.COLOR_RGB2BGR))
            durations.append(seconds)

    if not frames:
        return None, None
    if len(frames) > max_frames:
        step = len(frames) / float(max_frames)
        kept_frames, kept_durations = [], []
        for i in range(max_frames):
            lo, hi = int(i * step), int((i + 1) * step)
            kept_frames.append(frames[lo])
            kept_durations.append(sum(durations[lo:hi]))
        frames, durations = kept_frames, kept_durations
    return frames, durations


class AnimalReactionDataset:
    """
    Real cat/dog reaction content from public APIs, never bundled, generated,
    or fake imagery. Giphy's search API is queried per detected mood (e.g.
    "confused cat") for mood matched GIFs and needs a free Giphy API key
    (optional). A real photo backup (The Cat API for cats, the Dog CEO API for
    dogs) is used when Giphy isn't configured or reachable; it works with zero
    signup, though it isn't mood matched.

    Backup photos are cached on disk so repeat launches don't re-fetch. Giphy
    results stay in memory only. Every network call degrades gracefully: no key,
    no internet, or a failed request just means no image yet, never a fake one.
    """

    def __init__(self, cache_dir, animal="cat", giphy_api_key=None, cat_api_key=None):
        self.animal = animal if animal in ("cat", "dog") else "cat"
        self.cache_dir = os.path.join(cache_dir, self.animal)
        self.images_dir = os.path.join(self.cache_dir, "images")
        self.giphy_api_key = giphy_api_key or os.environ.get("GIPHY_API_KEY")
        self.cat_api_key = cat_api_key or os.environ.get("CAT_API_KEY") or CAT_API_DEMO_KEY
        self.giphy_enabled = bool(self.giphy_api_key)
        self._giphy_entries = {}      # mood -> entries in Giphy's order, this session only
        self._giphy_phrase_index = {}  # mood -> which of its phrases to search next
        self._giphy_failed_at = {}
        self._general = None
        self._status = self._read_status()
        self._remove_stored_giphy_content()

    # ---- disk cache for the backup photos -------------------------------

    def _status_path(self):
        return os.path.join(self.cache_dir, "fetch_status.json")

    def _read_status(self):
        path = self._status_path()
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r") as handle:
                data = json.load(handle)
            return data if isinstance(data, dict) else {}
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_status(self):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self._status_path(), "w") as handle:
                json.dump(self._status, handle)
        except OSError:
            pass

    def _remove_stored_giphy_content(self):
        """Earlier versions saved Giphy stills and search results to disk."""
        for path in glob.glob(os.path.join(self.images_dir, "giphy_*")):
            try:
                os.remove(path)
            except OSError:
                pass
        for path in glob.glob(os.path.join(self.cache_dir, "*.json")):
            name = os.path.basename(path)
            if name in ("fetch_status.json", f"{GENERAL_KEY}.json"):
                continue
            try:
                os.remove(path)
            except OSError:
                pass
        stale = [key for key in self._status if key.startswith("giphy:")]
        if stale:
            for key in stale:
                del self._status[key]
            self._write_status()

    def _general_cache_path(self):
        return os.path.join(self.cache_dir, f"{GENERAL_KEY}.json")

    def _load_general_cache(self):
        path = self._general_cache_path()
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(data, list):
            return None
        valid = [e for e in data if isinstance(e, dict) and os.path.isfile(e.get("local_path", ""))]
        return valid if valid else None

    def _save_general_cache(self, entries):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self._general_cache_path(), "w") as handle:
                json.dump(entries, handle)
        except OSError:
            pass

    def _should_retry(self, status_key):
        info = self._status.get(status_key, {})
        if info.get("last_success"):
            return True
        return (time.time() - info.get("last_attempt", 0)) > FETCH_RETRY_COOLDOWN

    def _mark_status(self, status_key, success):
        self._status[status_key] = {"last_attempt": time.time(), "last_success": success}
        self._write_status()

    @staticmethod
    def _download(url, dest_path):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            with open(dest_path, "wb") as handle:
                handle.write(response.content)
            return True
        except (requests.RequestException, OSError):
            return False

    # ---- Giphy, in memory only ------------------------------------------

    def _fetch_giphy(self, mood_tag):
        if not self.giphy_enabled:
            return None
        if time.time() - self._giphy_failed_at.get(mood_tag, 0) < GIPHY_RETRY_COOLDOWN:
            return None

        phrases = queries_for_mood(mood_tag, animal=self.animal)
        index = self._giphy_phrase_index.get(mood_tag, 0) % len(phrases)
        query, depth = phrases[index]
        params = {
            "api_key": self.giphy_api_key,
            "q": query,
            "limit": depth,
            "offset": 0,
            "rating": "g",
            "lang": "en",
        }
        try:
            response = requests.get(GIPHY_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            if response.status_code in (401, 403):
                print(f"giphy rejected the key ({response.status_code}), using photos instead")
                self.giphy_enabled = False
                return None
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            self._giphy_failed_at[mood_tag] = time.time()
            return None

        data = payload.get("data", []) if isinstance(payload, dict) else []
        entries = []
        for item in data:
            gif_id = item.get("id") if isinstance(item, dict) else None
            fixed = (item.get("images") or {}).get("fixed_height") or {} if gif_id else {}
            media_url = fixed.get("url")
            if not gif_id or not media_url:
                continue
            entries.append({
                "key": f"giphy:{gif_id}",
                "name": f"{mood_tag} {self.animal} reaction",
                "media_url": media_url,
                "source": "giphy",
                "attribution": "Powered By GIPHY",
            })

        # Next time, the next phrase. After the last one, back to the first.
        self._giphy_phrase_index[mood_tag] = index + 1
        if not entries:
            self._giphy_failed_at[mood_tag] = time.time()
            return None
        return entries

    def get_for_mood(self, mood_tag):
        """Giphy entries for a mood, in Giphy's order. Fetches more when empty."""
        queue = self._giphy_entries.get(mood_tag)
        if not queue:
            fetched = self._fetch_giphy(mood_tag)
            if fetched:
                self._giphy_entries[mood_tag] = fetched
                queue = fetched
        return queue or None

    def take_for_mood(self, mood_tag, exclude_key=None):
        """Removes and returns the next Giphy entry for the mood, or None."""
        queue = self.get_for_mood(mood_tag)
        if not queue:
            return None
        for i, entry in enumerate(queue):
            if entry["key"] != exclude_key:
                return queue.pop(i)
        return None

    @staticmethod
    def load_giphy_media(entry):
        """Downloads one Giphy GIF into memory. Returns (frames, durations) or (None, None)."""
        try:
            response = requests.get(entry["media_url"], timeout=REQUEST_TIMEOUT_SECONDS, stream=True)
            response.raise_for_status()
            chunks = []
            size = 0
            for chunk in response.iter_content(chunk_size=1 << 16):
                size += len(chunk)
                if size > GIPHY_MAX_BYTES:
                    return None, None
                chunks.append(chunk)
            return decode_animation(b"".join(chunks))
        except (requests.RequestException, OSError, ValueError):
            return None, None

    # ---- backup photos ----------------------------------------------------

    def _fetch_cat_api(self):
        status_key = "cat_api"
        if not self._should_retry(status_key):
            return None

        os.makedirs(self.images_dir, exist_ok=True)
        headers = {"x-api-key": self.cat_api_key}
        params = {"limit": RESULTS_PER_MOOD}
        try:
            response = requests.get(CAT_API_SEARCH_URL, headers=headers, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            self._mark_status(status_key, False)
            return None

        entries = []
        items = payload if isinstance(payload, list) else []
        for item in items:
            image_id = item.get("id") if isinstance(item, dict) else None
            url = item.get("url") if image_id else None
            if not image_id or not url:
                continue

            extension = os.path.splitext(url)[1] or ".jpg"
            local_path = os.path.join(self.images_dir, f"catapi_{image_id}{extension}")
            if not os.path.isfile(local_path) and not self._download(url, local_path):
                continue

            entries.append({
                "key": f"catapi:{image_id}",
                "name": "real cat photo",
                "local_path": local_path,
                "source": "cat_api",
                "attribution": None,
            })

        if not entries:
            self._mark_status(status_key, False)
            return None

        self._mark_status(status_key, True)
        self._save_general_cache(entries)
        return entries

    def _fetch_dog_api(self):
        status_key = "dog_api"
        if not self._should_retry(status_key):
            return None

        os.makedirs(self.images_dir, exist_ok=True)
        urls = []
        for breed in random.sample(DOG_BREEDS, 3):
            try:
                response = requests.get(DOG_BREED_URL.format(breed=breed, count=4), timeout=REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError):
                continue
            if payload.get("status") == "success" and isinstance(payload.get("message"), list):
                urls.extend(payload["message"])

        entries = []
        for url in urls:
            image_id = hashlib.md5(url.encode("utf-8")).hexdigest()[:12]
            extension = os.path.splitext(url)[1] or ".jpg"
            local_path = os.path.join(self.images_dir, f"dogapi_{image_id}{extension}")
            if not os.path.isfile(local_path) and not self._download(url, local_path):
                continue
            entries.append({
                "key": f"dogapi:{image_id}",
                "name": "real dog photo",
                "local_path": local_path,
                "source": "dog_api",
                "attribution": None,
            })

        if not entries:
            self._mark_status(status_key, False)
            return None

        self._mark_status(status_key, True)
        self._save_general_cache(entries)
        return entries

    def get_general(self):
        if self._general:
            return self._general
        entries = self._load_general_cache()
        if entries is None:
            entries = self._fetch_dog_api() if self.animal == "dog" else self._fetch_cat_api()
        if entries:
            self._general = entries
        return entries

    def describe_source(self):
        backup_name = "Dog CEO API" if self.animal == "dog" else "The Cat API"
        if self.giphy_enabled:
            return f"real {self.animal} reactions via Giphy (mood matched, animated), real {self.animal} photos via {backup_name} as backup"
        return f"real {self.animal} photos via {backup_name} (set GIPHY_API_KEY for mood matched reactions instead of generic photos)"

    def clear_cache(self):
        if os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        self._giphy_entries = {}
        self._giphy_phrase_index = {}
        self._general = None
        self._status = {}


class ReactionSource:
    def __init__(self, cache_dir, animal="cat", giphy_api_key=None, cat_api_key=None, force_refresh=False):
        self.dataset = AnimalReactionDataset(cache_dir, animal=animal, giphy_api_key=giphy_api_key, cat_api_key=cat_api_key)
        if force_refresh:
            self.dataset.clear_cache()

    def describe_source(self):
        return self.dataset.describe_source()

    def pick(self, mood_tags, exclude_key=None):
        """
        A reaction for the strongest mood that has one: Giphy first, walking
        down the ranked moods, then a backup photo. Returns a dict with "image"
        (first frame) and, for animations, "frames" and "durations".
        """
        moods = list(mood_tags) if mood_tags else ["neutral"]
        primary_mood = moods[0]

        for mood in moods:
            for _ in range(3):  # skip a GIF that fails to download or decode
                entry = self.dataset.take_for_mood(mood, exclude_key=exclude_key)
                if entry is None:
                    break
                frames, durations = self.dataset.load_giphy_media(entry)
                if frames:
                    return {
                        "key": entry["key"],
                        "name": entry["name"],
                        "image": frames[0],
                        "frames": frames if len(frames) > 1 else None,
                        "durations": durations if len(frames) > 1 else None,
                        "source": entry["source"],
                        "attribution": entry["attribution"],
                        "tags": [mood],
                    }

        entries = self.dataset.get_general()
        if entries:
            candidates = entries
            if exclude_key is not None and len(entries) > 1:
                filtered = [entry for entry in entries if entry["key"] != exclude_key]
                if filtered:
                    candidates = filtered
            candidates = list(candidates)
            random.shuffle(candidates)
            for choice in candidates[:3]:
                image = cv2.imread(choice["local_path"])
                if image is not None:
                    return {
                        "key": choice["key"],
                        "name": choice["name"],
                        "image": image,
                        "frames": None,
                        "durations": None,
                        "source": choice["source"],
                        "attribution": choice.get("attribution"),
                        "tags": [primary_mood],
                    }

        return {
            "key": f"unavailable:{primary_mood}",
            "name": f"{primary_mood} reaction",
            "image": None,
            "frames": None,
            "durations": None,
            "source": "unavailable",
            "attribution": None,
            "tags": [primary_mood],
        }


# ============================================================================
# Rendering: draws the caption onto the picked reaction image
# ============================================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
FONT_PATH = os.path.join(ASSETS_DIR, "DejaVuSans-Bold.ttf")

MAX_FONT_SIZE_RATIO = 0.11
MIN_FONT_SIZE = 16
TEXT_MARGIN_RATIO = 0.04
OUTLINE_WIDTH = 3
ATTRIBUTION_FONT_SIZE = 14


def _wrap_text(draw, text, font, max_width):
    words = text.split()
    if not words:
        return [""]

    lines = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        bbox = draw.textbbox((0, 0), candidate, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _fit_lines(draw, text, max_width, start_size):
    size = start_size
    while size > MIN_FONT_SIZE:
        font = ImageFont.truetype(FONT_PATH, size)
        lines = _wrap_text(draw, text, font, max_width)
        widest = max(draw.textbbox((0, 0), line, font=font)[2] for line in lines)
        if widest <= max_width and len(lines) <= 3:
            return font, lines
        size -= 3
    font = ImageFont.truetype(FONT_PATH, MIN_FONT_SIZE)
    return font, _wrap_text(draw, text, font, max_width)


def _draw_outlined_text(draw, xy, text, font, fill=(255, 255, 255), outline=(0, 0, 0)):
    x, y = xy
    for dx in range(-OUTLINE_WIDTH, OUTLINE_WIDTH + 1):
        for dy in range(-OUTLINE_WIDTH, OUTLINE_WIDTH + 1):
            if dx != 0 or dy != 0:
                draw.text((x + dx, y + dy), text, font=font, fill=outline)
    draw.text((x, y), text, font=font, fill=fill)


def _draw_caption_block(draw, text, canvas_width, anchor_y, start_size, max_width):
    if not text:
        return
    font, lines = _fit_lines(draw, text.upper(), max_width, start_size)
    line_height = font.size + 6

    y = anchor_y
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_width = bbox[2] - bbox[0]
        x = (canvas_width - line_width) // 2
        _draw_outlined_text(draw, (x, y), line, font)
        y += line_height


def _block_height(draw, text, max_width, start_size):
    if not text:
        return 0
    font, lines = _fit_lines(draw, text.upper(), max_width, start_size)
    return (font.size + 6) * len(lines)


def caption_overlay(caption, panel_width, panel_height, attribution=None):
    """The caption and attribution drawn once on a transparent layer.

    Returns (bgr, alpha) with alpha in 0..1, shaped (h, w, 1), so the same text
    can be laid over every frame of an animation cheaply.
    """
    layer = Image.new("RGBA", (panel_width, panel_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)

    start_size = max(MIN_FONT_SIZE, int(panel_height * MAX_FONT_SIZE_RATIO))
    margin = int(panel_height * TEXT_MARGIN_RATIO)
    max_width = int(panel_width * (1 - 2 * TEXT_MARGIN_RATIO))

    top_text = (caption or {}).get("top", "")
    bottom_text = (caption or {}).get("bottom", "")

    if top_text:
        _draw_caption_block(draw, top_text, panel_width, margin, start_size, max_width)

    if bottom_text:
        block_height = _block_height(draw, bottom_text, max_width, start_size)
        start_y = panel_height - margin - block_height
        _draw_caption_block(draw, bottom_text, panel_width, start_y, start_size, max_width)

    if attribution:
        attr_font = ImageFont.truetype(FONT_PATH, ATTRIBUTION_FONT_SIZE)
        bbox = draw.textbbox((0, 0), attribution, font=attr_font)
        attr_width = bbox[2] - bbox[0]
        _draw_outlined_text(
            draw,
            (panel_width - attr_width - 10, panel_height - ATTRIBUTION_FONT_SIZE - 8),
            attribution,
            attr_font,
            fill=(220, 220, 220),
        )

    rgba = np.array(layer)
    bgr = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_RGB2BGR).astype(np.float32)
    alpha = (rgba[:, :, 3:4].astype(np.float32)) / 255.0
    return bgr, alpha


def _apply_overlay(canvas_bgr, overlay):
    bgr, alpha = overlay
    blended = canvas_bgr.astype(np.float32) * (1.0 - alpha) + bgr * alpha
    return np.clip(blended + 0.5, 0, 255).astype(np.uint8)


def render(reaction_bgr, caption, panel_width, panel_height, placeholder_text="loading reaction...", attribution=None):
    canvas = fit_to_panel(reaction_bgr, panel_width, panel_height, placeholder_text=placeholder_text)
    overlay = caption_overlay(caption, panel_width, panel_height, attribution=attribution)
    return _apply_overlay(canvas, overlay)


def render_frames(frames_bgr, caption, panel_width, panel_height, attribution=None):
    """Every frame of an animation fitted to the panel with the same caption."""
    overlay = caption_overlay(caption, panel_width, panel_height, attribution=attribution)
    return [_apply_overlay(fit_to_panel(frame, panel_width, panel_height), overlay) for frame in frames_bgr]

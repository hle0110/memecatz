"""
Everything about getting a real reaction on screen: fetching/caching real
cat or dog content from Giphy (mood-matched) and a zero-setup real-photo
backup (The Cat API / Dog CEO API), picking one for the current mood, and
rendering the caption onto it. No generated, drawn, or AI-created imagery
is ever used as a substitute.
"""

import os
import json
import random
import shutil
import time
import hashlib
import requests
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from utils import fit_to_panel

GIPHY_SEARCH_URL = "https://api.giphy.com/v1/gifs/search"
CAT_API_SEARCH_URL = "https://api.thecatapi.com/v1/images/search"
CAT_API_DEMO_KEY = "DEMO-API-KEY"
DOG_API_URL = "https://dog.ceo/api/breeds/image/random"
REQUEST_TIMEOUT_SECONDS = 8
FETCH_RETRY_COOLDOWN = 1800
RESULTS_PER_MOOD = 12
GENERAL_KEY = "_general"

MOOD_QUERY_TEMPLATES = {
    "happy": "happy {animal}",
    "sad": "sad {animal}",
    "angry": "angry {animal}",
    "surprise": "surprised {animal}",
    "fear": "scared {animal}",
    "disgust": "disgusted {animal}",
    "neutral": "{animal} staring blankly",
    "smug": "smug {animal}",
    "confused": "confused {animal}",
    "mischief": "mischievous {animal}",
    "annoyed": "annoyed {animal}",
    "approval": "{animal} nodding approval",
    "disapproval": "{animal} side eye disapproval",
    "chill": "relaxed {animal}",
    "suspicious": "suspicious {animal}",
    "triumph": "{animal} victory",
    "anxious": "nervous {animal}",
    "bored": "bored {animal}",
    "mocking": "sassy {animal}",
    "stop": "{animal} stop paw",
    "determined": "determined {animal}",
    "focused": "focused {animal}",
}


def query_for_mood(mood_tag, animal="cat"):
    template = MOOD_QUERY_TEMPLATES.get(mood_tag)
    if template is None:
        return f"{mood_tag.replace('_', ' ')} {animal}"
    return template.format(animal=animal)


class AnimalReactionDataset:
    """
    Real cat/dog reaction content, sourced live from two public APIs (no
    bundled, generated, or AI-created imagery at any point):

    - Giphy's search API, queried per detected mood (e.g. "confused cat"),
      for mood-matched reaction GIFs. Requires a free Giphy API key (optional).
    - A real-photo backup (The Cat API for cats, the Dog CEO API for dogs)
      when Giphy isn't configured/reachable or has nothing for a given mood.
      Works out of the box with zero signup, though it isn't mood-matched.

    Everything downloaded is cached locally so repeat launches don't re-fetch,
    and every network call degrades gracefully: no key, no internet, or a
    failed request just means no image is available yet, never a fake one.
    """

    def __init__(self, cache_dir, animal="cat", giphy_api_key=None, cat_api_key=None):
        self.animal = animal if animal in ("cat", "dog") else "cat"
        self.cache_dir = os.path.join(cache_dir, self.animal)
        self.images_dir = os.path.join(self.cache_dir, "images")
        self.giphy_api_key = giphy_api_key or os.environ.get("GIPHY_API_KEY")
        self.cat_api_key = cat_api_key or os.environ.get("CAT_API_KEY") or CAT_API_DEMO_KEY
        self.giphy_enabled = bool(self.giphy_api_key)
        self._mood_cache = {}
        self._status = self._read_status()

    def _status_path(self):
        return os.path.join(self.cache_dir, "fetch_status.json")

    def _read_status(self):
        path = self._status_path()
        if not os.path.isfile(path):
            return {}
        try:
            with open(path, "r") as handle:
                return json.load(handle)
        except (json.JSONDecodeError, OSError):
            return {}

    def _write_status(self):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self._status_path(), "w") as handle:
                json.dump(self._status, handle)
        except OSError:
            pass

    def _mood_cache_path(self, mood_tag):
        return os.path.join(self.cache_dir, f"{mood_tag}.json")

    def _load_mood_cache(self, mood_tag):
        path = self._mood_cache_path(mood_tag)
        if not os.path.isfile(path):
            return None
        try:
            with open(path, "r") as handle:
                data = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return None
        valid = [entry for entry in data if os.path.isfile(entry.get("local_path", ""))]
        return valid if valid else None

    def _save_mood_cache(self, mood_tag, entries):
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            with open(self._mood_cache_path(mood_tag), "w") as handle:
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

    def _download(self, url, dest_path):
        try:
            response = requests.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            with open(dest_path, "wb") as handle:
                handle.write(response.content)
            return True
        except (requests.RequestException, OSError):
            return False

    def _fetch_giphy(self, mood_tag):
        status_key = f"giphy:{mood_tag}"
        if not self.giphy_enabled or not self._should_retry(status_key):
            return None

        os.makedirs(self.images_dir, exist_ok=True)
        params = {
            "api_key": self.giphy_api_key,
            "q": query_for_mood(mood_tag, animal=self.animal),
            "limit": RESULTS_PER_MOOD,
            "rating": "g",
            "lang": "en",
        }
        try:
            response = requests.get(GIPHY_SEARCH_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
            response.raise_for_status()
            payload = response.json()
        except (requests.RequestException, ValueError):
            self._mark_status(status_key, False)
            return None

        entries = []
        for item in payload.get("data", []):
            gif_id = item.get("id")
            images = item.get("images", {})
            still = images.get("fixed_height_still") or images.get("original_still") or {}
            still_url = still.get("url")
            if not gif_id or not still_url:
                continue

            local_path = os.path.join(self.images_dir, f"giphy_{gif_id}.jpg")
            if not os.path.isfile(local_path) and not self._download(still_url, local_path):
                continue

            entries.append({
                "key": f"giphy:{gif_id}",
                "name": f"{mood_tag} {self.animal} reaction",
                "local_path": local_path,
                "source": "giphy",
                "attribution": "Powered By GIPHY",
            })

        if not entries:
            self._mark_status(status_key, False)
            return None

        self._mark_status(status_key, True)
        self._save_mood_cache(mood_tag, entries)
        return entries

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
            image_id = item.get("id")
            url = item.get("url")
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
        self._save_mood_cache(GENERAL_KEY, entries)
        return entries

    def _fetch_dog_api(self):
        status_key = "dog_api"
        if not self._should_retry(status_key):
            return None

        os.makedirs(self.images_dir, exist_ok=True)
        entries = []
        for _ in range(RESULTS_PER_MOOD):
            try:
                response = requests.get(DOG_API_URL, timeout=REQUEST_TIMEOUT_SECONDS)
                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError):
                break

            if payload.get("status") != "success" or not payload.get("message"):
                break

            url = payload["message"]
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
        self._save_mood_cache(GENERAL_KEY, entries)
        return entries

    def get_for_mood(self, mood_tag):
        if mood_tag in self._mood_cache:
            return self._mood_cache[mood_tag]

        entries = self._load_mood_cache(mood_tag)
        if entries is None:
            entries = self._fetch_giphy(mood_tag)

        if entries:
            self._mood_cache[mood_tag] = entries
            return entries
        return None

    def get_general(self):
        if GENERAL_KEY in self._mood_cache:
            return self._mood_cache[GENERAL_KEY]

        entries = self._load_mood_cache(GENERAL_KEY)
        if entries is None:
            entries = self._fetch_dog_api() if self.animal == "dog" else self._fetch_cat_api()

        if entries:
            self._mood_cache[GENERAL_KEY] = entries
            return entries
        return None

    def describe_source(self):
        backup_name = "Dog CEO API" if self.animal == "dog" else "The Cat API"
        if self.giphy_enabled:
            return f"real {self.animal} reactions via Giphy (mood-matched), real {self.animal} photos via {backup_name} as backup"
        return f"real {self.animal} photos via {backup_name} (set GIPHY_API_KEY for mood-matched reactions instead of generic photos)"

    def clear_cache(self):
        if os.path.isdir(self.cache_dir):
            shutil.rmtree(self.cache_dir, ignore_errors=True)
        self._mood_cache = {}
        self._status = {}


class ReactionSource:
    def __init__(self, cache_dir, animal="cat", giphy_api_key=None, cat_api_key=None, force_refresh=False):
        self.dataset = AnimalReactionDataset(cache_dir, animal=animal, giphy_api_key=giphy_api_key, cat_api_key=cat_api_key)
        if force_refresh:
            self.dataset.clear_cache()

    def describe_source(self):
        return self.dataset.describe_source()

    def pick(self, mood_tags, exclude_key=None):
        primary_mood = mood_tags[0] if mood_tags else "neutral"

        entries = self.dataset.get_for_mood(primary_mood)
        if not entries:
            entries = self.dataset.get_general()

        if entries:
            candidates = entries
            if exclude_key is not None and len(entries) > 1:
                filtered = [entry for entry in entries if entry["key"] != exclude_key]
                if filtered:
                    candidates = filtered

            choice = random.choice(candidates)
            image = cv2.imread(choice["local_path"])
            if image is not None:
                return {
                    "key": choice["key"],
                    "name": choice["name"],
                    "image": image,
                    "source": choice["source"],
                    "attribution": choice.get("attribution"),
                    "tags": [primary_mood],
                }

        return {
            "key": f"unavailable:{primary_mood}",
            "name": f"{primary_mood} reaction",
            "image": None,
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


def render(reaction_bgr, caption, panel_width, panel_height, placeholder_text="loading reaction...", attribution=None):
    canvas = fit_to_panel(reaction_bgr, panel_width, panel_height, placeholder_text=placeholder_text)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_image)

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

    rendered_rgb = np.array(pil_image)
    return cv2.cvtColor(rendered_rgb, cv2.COLOR_RGB2BGR)

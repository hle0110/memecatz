"""
Both OpenAI-backed enrichment features live here: live caption generation
(CaptionEngine) and the optional vision mood boost (VisionMoodAnalyzer).
Grouped together since they share the same API key and the same
rate-limit/cache/fallback pattern, just for different purposes.
"""

import os
import json
import random
import time
import re
import base64
import threading
import cv2

# ============================================================================
# Live captions
# ============================================================================

# A slow or unreachable API must never hold up the app for long. The OpenAI
# client's own default is a 10 minute timeout with 2 retries.
OPENAI_TIMEOUT_SECONDS = 10
OPENAI_MAX_RETRIES = 1

CAPTION_DEFAULT_MODEL = "gpt-4o-mini"
CAPTION_MIN_SECONDS_BETWEEN_CALLS = 4.0
CAPTION_MAX_TOKENS = 150

STATIC_CAPTIONS = {
    "happy": [
        ("BIG MOOD", "NO NOTES"),
        ("THAT'S A YES", "FROM ME"),
        ("LIVING MY", "BEST LIFE"),
        ("TODAY IS", "A GOOD DAY"),
        ("SMILING", "FOR NO REASON"),
        ("PURE", "JOY"),
    ],
    "sad": [
        ("NOT VIBING", "RIGHT NOW"),
        ("PAUSE", "NOT OKAY"),
        ("NEED A HUG", "IMMEDIATELY"),
        ("IT'S FINE", "IT'S NOT FINE"),
        ("CANCEL", "TODAY"),
        ("EMOTIONALLY", "BUFFERING"),
    ],
    "angry": [
        ("SEND TWEET", ""),
        ("ABOUT TO", "SNAP"),
        ("WHO DID THIS", ""),
        ("I'M CALM", "I'M SO CALM"),
        ("DO NOT", "TEST ME"),
        ("COUNTING", "TO TEN"),
    ],
    "surprise": [
        ("WAIT WHAT", ""),
        ("PLOT TWIST", "INCOMING"),
        ("DID THAT", "JUST HAPPEN"),
        ("HOLD", "THE PHONE"),
        ("EXCUSE ME", "??"),
        ("NOBODY SAW", "THAT COMING"),
    ],
    "fear": [
        ("NOPE NOPE NOPE", ""),
        ("RUN", "IT'S OVER"),
        ("I HEARD", "A NOISE"),
        ("WHAT WAS", "THAT"),
        ("NOT TODAY", ""),
        ("HIDE", "EVERYTHING"),
    ],
    "disgust": [
        ("THE EW FACTOR", "IS HIGH"),
        ("HARD PASS", ""),
        ("WHO MADE", "THIS"),
        ("SMELLS", "SUSPICIOUS"),
        ("I NEED", "A MINUTE"),
        ("NO THANK", "YOU"),
    ],
    "neutral": [
        ("OKAY.", ""),
        ("PROCESSING", "..."),
        ("NO THOUGHTS", "HEAD EMPTY"),
        ("JUST", "EXISTING"),
        ("STARING", "INTO SPACE"),
        ("LOADING", "OPINION..."),
    ],
    "smug": [
        ("ROLL SAFE", "THINK ABOUT IT"),
        ("KNEW IT", "ALL ALONG"),
        ("TOLD YOU", "SO"),
        ("TOO EASY", ""),
        ("CALCULATED", ""),
        ("I'M KIND OF", "A GENIUS"),
    ],
    "confused": [
        ("WAIT", "WHAT JUST HAPPENED"),
        ("HOLD ON", "LET ME THINK"),
        ("I HAVE", "QUESTIONS"),
        ("THE MATH", "ISN'T MATHING"),
        ("SAY THAT", "AGAIN?"),
        ("HUH", ""),
    ],
    "mischief": [
        ("OH IT'S ON", ""),
        ("WATCH THIS", ""),
        ("I HAVE", "A PLAN"),
        ("NOBODY", "TELL MOM"),
        ("HEHEHE", ""),
        ("CHAOS", "MODE ON"),
    ],
    "annoyed": [
        ("HERE WE GO", "AGAIN"),
        ("NOT THIS", "AGAIN"),
        ("I'M SO TIRED", "OF THIS"),
        ("SIGH", ""),
        ("CAN WE", "NOT"),
        ("WHY IS IT", "ALWAYS ME"),
    ],
    "approval": [
        ("SEAL OF", "APPROVAL"),
        ("TAKE MY", "UPVOTE"),
        ("WE LOVE", "TO SEE IT"),
        ("TEN OUT OF", "TEN"),
        ("CERTIFIED", "GOOD"),
        ("YES", "CHEF"),
    ],
    "disapproval": [
        ("HARD NO", ""),
        ("ABSOLUTELY NOT", ""),
        ("I DON'T", "LIKE THIS"),
        ("DENIED", ""),
        ("THAT'S A NO", "FROM ME"),
        ("TRY", "AGAIN"),
    ],
    "chill": [
        ("ALL GOOD", "HERE"),
        ("VIBES ONLY", ""),
        ("NO STRESS", ""),
        ("TAKING IT", "EASY"),
        ("COOL CALM", "COLLECTED"),
        ("JUST", "CHILLING"),
    ],
    "suspicious": [
        ("SOMETHING'S", "NOT RIGHT"),
        ("I'M WATCHING", "YOU"),
        ("SUS", ""),
        ("I SEE", "WHAT YOU DID"),
        ("EXPLAIN", "YOURSELF"),
        ("TRUST", "NOBODY"),
    ],
    "triumph": [
        ("NAILED IT", ""),
        ("W TAKEN", ""),
        ("CHAMPION", "ENERGY"),
        ("VICTORY", "IS MINE"),
        ("UNDEFEATED", ""),
        ("GG", ""),
    ],
    "anxious": [
        ("THIS IS FINE", "PROBABLY"),
        ("KEEP IT", "TOGETHER"),
        ("OVERTHINKING", "AGAIN"),
        ("DID I", "LOCK THE DOOR"),
        ("DEEP", "BREATHS"),
        ("STRESS LEVEL", "HIGH"),
    ],
    "bored": [
        ("STILL WAITING", ""),
        ("ANY DAY NOW", ""),
        ("IS IT OVER", "YET"),
        ("SO", "BORED"),
        ("NAP", "TIME?"),
        ("WAKE ME UP", "WHEN IT'S DONE"),
    ],
    "mocking": [
        ("SURE, BUDDY", ""),
        ("OKAY THERE", "CHAMP"),
        ("WOW", "SO IMPRESSIVE"),
        ("THAT'S", "CUTE"),
        ("MHM", "SURE"),
        ("GOOD FOR", "YOU I GUESS"),
    ],
    "stop": [
        ("STOP", "RIGHT THERE"),
        ("HOLD UP", ""),
        ("NOT ONE", "MORE STEP"),
        ("THAT'S", "ENOUGH"),
        ("PAUSE", ""),
        ("HALT", ""),
    ],
    "determined": [
        ("LET'S", "GO"),
        ("LOCKED IN", ""),
        ("NOTHING", "CAN STOP ME"),
        ("GAME FACE", "ON"),
        ("WATCH ME", "WORK"),
        ("NO DAYS", "OFF"),
    ],
    "focused": [
        ("LOCKED", "IN"),
        ("DO NOT", "DISTURB"),
        ("IN THE", "ZONE"),
        ("ONE JOB", ""),
        ("CONCENTRATING", ""),
        ("EYES ON", "THE PRIZE"),
    ],
}


_last_static_caption = None


def _static_caption(mood_tag):
    """Picks from the static bank, avoiding an immediate repeat of the last pick."""
    global _last_static_caption
    options = STATIC_CAPTIONS.get(mood_tag, STATIC_CAPTIONS["neutral"])
    if _last_static_caption in options and len(options) > 1:
        options = [option for option in options if option != _last_static_caption]
    choice = random.choice(options)
    _last_static_caption = choice
    top, bottom = choice
    return {"top": top, "bottom": bottom, "source": "static"}


class CaptionEngine:
    def __init__(self, api_key=None, model=CAPTION_DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = None
        self.enabled = bool(self.api_key)
        self._cache = {}
        self._last_call_time = 0.0

        if self.enabled:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key, timeout=OPENAI_TIMEOUT_SECONDS,
                                            max_retries=OPENAI_MAX_RETRIES)
            except Exception:
                self.enabled = False
                self.client = None

    def _call_api(self, prompt):
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=CAPTION_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    @staticmethod
    def _parse_caption(text):
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None

        top = str(payload.get("top", "")).strip().upper()
        bottom = str(payload.get("bottom", "")).strip().upper()
        if not top and not bottom:
            return None
        return {"top": top, "bottom": bottom}

    def _build_prompt(self, template_name, mood_tags):
        tags_text = ", ".join(mood_tags) if mood_tags else "neutral"
        return (
            f"Write a short, funny meme-style caption for a real animal reaction photo/gif ('{template_name}'). "
            f"The person's current detected mood/expression tags are: {tags_text}. "
            "Reply with ONLY a JSON object like "
            '{"top": "TOP TEXT", "bottom": "BOTTOM TEXT"}. '
            "Keep each line under 6 words, punchy, in classic meme caps style. "
            "Bottom can be an empty string if the joke only needs one line."
        )

    def generate(self, template_id, template_name, mood_tags, live=True):
        """A caption for the reaction. live=False skips the API, for startup."""
        primary = mood_tags[0] if mood_tags else "neutral"
        cache_key = (template_id, primary)
        if cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            cached["source"] = "cache"
            return cached

        if not self.enabled or not live:
            return _static_caption(primary)

        now = time.time()
        if now - self._last_call_time < CAPTION_MIN_SECONDS_BETWEEN_CALLS:
            return _static_caption(primary)

        self._last_call_time = now

        try:
            prompt = self._build_prompt(template_name, mood_tags)
            raw_text = self._call_api(prompt)
            parsed = self._parse_caption(raw_text)
            if parsed is None:
                return _static_caption(primary)
            parsed["source"] = "live"
            self._cache[cache_key] = {"top": parsed["top"], "bottom": parsed["bottom"]}
            return parsed
        except Exception:
            return _static_caption(primary)


# ============================================================================
# Optional vision mood boost
# ============================================================================

VISION_DEFAULT_MODEL = "gpt-4o-mini"
VISION_MIN_SECONDS_BETWEEN_CALLS = 4.0
VISION_MAX_TOKENS = 120
VISION_JPEG_QUALITY = 80
# Tags describe the face at the moment the photo was taken, so they expire.
VISION_TAG_MAX_AGE_SECONDS = 10.0

MOOD_VOCABULARY = (
    "happy", "sad", "angry", "surprise", "fear", "disgust", "neutral",
    "smug", "confused", "mischief", "annoyed", "approval", "disapproval",
    "chill", "suspicious", "triumph", "anxious", "bored", "mocking",
)


class VisionMoodAnalyzer:
    def __init__(self, api_key=None, model=VISION_DEFAULT_MODEL):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.model = model
        self.client = None
        self.enabled = bool(self.api_key)
        self._last_call_time = 0.0
        self._last_tags = {}
        self._last_tags_time = 0.0
        self._lock = threading.Lock()
        self._busy = False

        if self.enabled:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key, timeout=OPENAI_TIMEOUT_SECONDS,
                                            max_retries=OPENAI_MAX_RETRIES)
            except Exception:
                self.enabled = False
                self.client = None

    @staticmethod
    def _encode_frame(face_crop_bgr):
        ok, buffer = cv2.imencode(".jpg", face_crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, VISION_JPEG_QUALITY])
        if not ok:
            return None
        return base64.b64encode(buffer).decode("ascii")

    @staticmethod
    def _build_prompt(current_tags):
        tags_text = ", ".join(current_tags) if current_tags else "none yet"
        vocabulary_text = ", ".join(MOOD_VOCABULARY)
        return (
            "Look at this cropped webcam face photo. "
            f"The current rule-based mood guess is: {tags_text}. "
            f"Pick up to 2 tags from this exact list that best describe the expression: {vocabulary_text}. "
            'Reply with ONLY a JSON object like {"tags": ["smug", "mischief"]}. '
            "Only include tags you are reasonably confident about; use fewer if unsure, "
            'or {"tags": []} if the expression is plainly neutral.'
        )

    def _call_api(self, prompt, image_b64):
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=VISION_MAX_TOKENS,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                ],
            }],
        )
        return response.choices[0].message.content

    @staticmethod
    def _parse_tags(text):
        """Tag dict from the reply, {} for a plain neutral answer, None if unreadable."""
        match = re.search(r"\{.*\}", text or "", re.DOTALL)
        if not match:
            return None
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None

        raw_tags = payload.get("tags", [])
        if not isinstance(raw_tags, list):
            return None

        tags = {}
        for tag in raw_tags:
            name = str(tag).strip().lower()
            if name in MOOD_VOCABULARY:
                tags[name] = 1.0
        return tags

    def _current_tags(self):
        with self._lock:
            if time.time() - self._last_tags_time > VISION_TAG_MAX_AGE_SECONDS:
                return {}
            return dict(self._last_tags)

    def _run(self, image_b64, prompt):
        try:
            new_tags = self._parse_tags(self._call_api(prompt, image_b64))
        except Exception:
            new_tags = None
        with self._lock:
            # An unreadable reply keeps the old tags until they expire. A clear
            # answer, including "nothing special", replaces them.
            if new_tags is not None:
                self._last_tags = new_tags
                self._last_tags_time = time.time()
            self._busy = False

    def analyze(self, face_crop_bgr, current_tags=None):
        """Returns the latest tags at once. A new request, at most every few
        seconds, runs in the background so detection never waits on it."""
        if not self.enabled:
            return {}

        now = time.time()
        with self._lock:
            start = (not self._busy
                     and now - self._last_call_time >= VISION_MIN_SECONDS_BETWEEN_CALLS
                     and face_crop_bgr is not None and face_crop_bgr.size > 0)
            if start:
                self._busy = True
                self._last_call_time = now

        if start:
            image_b64 = self._encode_frame(face_crop_bgr)
            if image_b64 is None:
                with self._lock:
                    self._busy = False
            else:
                prompt = self._build_prompt(current_tags or [])
                threading.Thread(target=self._run, args=(image_b64, prompt), daemon=True).start()

        return self._current_tags()

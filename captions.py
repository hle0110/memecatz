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
import cv2

# ============================================================================
# Live captions
# ============================================================================

CAPTION_DEFAULT_MODEL = "gpt-4o-mini"
CAPTION_MIN_SECONDS_BETWEEN_CALLS = 4.0
CAPTION_MAX_TOKENS = 150

STATIC_CAPTIONS = {
    "happy": [("BIG MOOD", "NO NOTES"), ("THAT'S A YES", "FROM ME")],
    "sad": [("NOT VIBING", "RIGHT NOW"), ("PAUSE", "NOT OKAY")],
    "angry": [("SEND TWEET", ""), ("ABOUT TO", "SNAP")],
    "surprise": [("WAIT WHAT", ""), ("PLOT TWIST", "INCOMING")],
    "fear": [("NOPE NOPE NOPE", ""), ("RUN", "IT'S OVER")],
    "disgust": [("THE EW FACTOR", "IS HIGH"), ("HARD PASS", "")],
    "neutral": [("OKAY.", ""), ("PROCESSING", "...")],
    "smug": [("ROLL SAFE", "THINK ABOUT IT"), ("KNEW IT", "ALL ALONG")],
    "confused": [("WAIT", "WHAT JUST HAPPENED"), ("HOLD ON", "LET ME THINK")],
    "mischief": [("OH IT'S ON", ""), ("WATCH THIS", "")],
    "annoyed": [("HERE WE GO", "AGAIN"), ("NOT THIS", "AGAIN")],
    "approval": [("SEAL OF", "APPROVAL"), ("TAKE MY", "UPVOTE")],
    "disapproval": [("HARD NO", ""), ("ABSOLUTELY NOT", "")],
    "chill": [("ALL GOOD", "HERE"), ("VIBES ONLY", "")],
    "suspicious": [("SOMETHING'S", "NOT RIGHT"), ("I'M WATCHING", "YOU")],
    "triumph": [("NAILED IT", ""), ("W TAKEN", "")],
    "anxious": [("THIS IS FINE", "PROBABLY"), ("KEEP IT", "TOGETHER")],
    "bored": [("STILL WAITING", ""), ("ANY DAY NOW", "")],
    "mocking": [("SURE, BUDDY", ""), ("OKAY THERE", "CHAMP")],
}


def _static_caption(mood_tag):
    options = STATIC_CAPTIONS.get(mood_tag, STATIC_CAPTIONS["neutral"])
    top, bottom = random.choice(options)
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
                self.client = openai.OpenAI(api_key=self.api_key)
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

    def generate(self, template_id, template_name, mood_tags):
        primary = mood_tags[0] if mood_tags else "neutral"
        cache_key = (template_id, primary)
        if cache_key in self._cache:
            cached = dict(self._cache[cache_key])
            cached["source"] = "cache"
            return cached

        if not self.enabled:
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
            parsed["source"] = "ml"
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

        if self.enabled:
            try:
                import openai
                self.client = openai.OpenAI(api_key=self.api_key)
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
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return {}
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

        raw_tags = payload.get("tags", [])
        if not isinstance(raw_tags, list):
            return {}

        tags = {}
        for tag in raw_tags:
            name = str(tag).strip().lower()
            if name in MOOD_VOCABULARY:
                tags[name] = 1.0
        return tags

    def analyze(self, face_crop_bgr, current_tags=None):
        if not self.enabled:
            return {}

        now = time.time()
        if now - self._last_call_time < VISION_MIN_SECONDS_BETWEEN_CALLS:
            return self._last_tags

        if face_crop_bgr is None or face_crop_bgr.size == 0:
            return self._last_tags

        self._last_call_time = now

        try:
            image_b64 = self._encode_frame(face_crop_bgr)
            if image_b64 is None:
                return self._last_tags
            prompt = self._build_prompt(current_tags or [])
            raw_text = self._call_api(prompt, image_b64)
            new_tags = self._parse_tags(raw_text)
            if new_tags:
                self._last_tags = new_tags
            return self._last_tags
        except Exception:
            return self._last_tags

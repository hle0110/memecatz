from collections import defaultdict

FER_WEIGHT = 0.5
AU_WEIGHT = 1.0
GESTURE_WEIGHT = 1.2
VISION_WEIGHT = 1.4
NEUTRAL_SUPPRESSION = 0.3
NEUTRAL_SUPPRESSION_THRESHOLD = 0.15
TOP_TAG_FLOOR = 0.12

AU_TAG_TO_MOOD = {
    "smile": {"happy": 1.0},
    "frown": {"sad": 0.7, "disgust": 0.25},
    "smirk": {"smug": 1.0, "mischief": 0.4},
    "jaw_drop": {"surprise": 1.0},
    "brow_raise": {"surprise": 0.8},
    "brow_furrow": {"angry": 0.8, "annoyed": 0.5},
    "squint": {"disgust": 0.5, "suspicious": 0.6},
    "wink": {"mischief": 1.0},
    "eye_wide": {"surprise": 0.6},
    "sneer": {"disgust": 0.7, "mocking": 0.4},
    "skeptical": {"suspicious": 0.7, "smug": 0.4},
    "cheek_puff": {"mischief": 0.8},
    "pucker": {"confused": 0.5, "mischief": 0.2},
}


def combine(fer_scores=None, au_tags=None, gesture_tags=None, vision_tags=None):
    mood = defaultdict(float)

    if fer_scores:
        for tag, score in fer_scores.items():
            mood[tag] += score * FER_WEIGHT

    if au_tags:
        for au_tag, score in au_tags.items():
            for mood_tag, weight in AU_TAG_TO_MOOD.get(au_tag, {}).items():
                mood[mood_tag] += score * weight * AU_WEIGHT

    if gesture_tags:
        for tag, score in gesture_tags.items():
            mood[tag] += score * GESTURE_WEIGHT

    if vision_tags:
        for tag, score in vision_tags.items():
            mood[tag] += score * VISION_WEIGHT

    non_neutral_peak = max((v for k, v in mood.items() if k != "neutral"), default=0.0)
    if "neutral" in mood and non_neutral_peak > NEUTRAL_SUPPRESSION_THRESHOLD:
        mood["neutral"] *= NEUTRAL_SUPPRESSION

    return dict(mood)


def top_tags(mood_vector, limit=3, floor=TOP_TAG_FLOOR):
    ranked = sorted(mood_vector.items(), key=lambda item: item[1], reverse=True)
    filtered = [(tag, score) for tag, score in ranked if score >= floor]
    if not filtered:
        return [("neutral", mood_vector.get("neutral", 1.0))]
    return filtered[:limit]


def primary_tag(mood_vector):
    return top_tags(mood_vector, limit=1)[0][0]

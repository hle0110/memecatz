// Direct port of mood.py. Same weights, same thresholds, same ranking rules.

export const FER_WEIGHT = 0.5;
export const AU_WEIGHT = 1.0;
export const GESTURE_WEIGHT = 1.2;
export const VISION_WEIGHT = 1.4;
export const NEUTRAL_SUPPRESSION = 0.3;
export const NEUTRAL_SUPPRESSION_THRESHOLD = 0.15;
export const TOP_TAG_FLOOR = 0.12;

export const AU_TAG_TO_MOOD = {
  smile: { happy: 1.0 },
  frown: { sad: 0.7, disgust: 0.25 },
  smirk: { smug: 1.0, mischief: 0.4 },
  jaw_drop: { surprise: 1.0 },
  brow_raise: { surprise: 0.8 },
  brow_furrow: { angry: 0.8, annoyed: 0.5 },
  squint: { disgust: 0.5, suspicious: 0.6 },
  wink: { mischief: 1.0 },
  eye_wide: { surprise: 0.6 },
  sneer: { disgust: 0.7, mocking: 0.4 },
  skeptical: { suspicious: 0.7, smug: 0.4 },
  cheek_puff: { mischief: 0.8 },
  pucker: { confused: 0.5, mischief: 0.2 },
};

export function combine({ ferScores = null, auTags = null, gestureTags = null, visionTags = null } = {}) {
  const mood = {};
  const add = (tag, amount) => {
    mood[tag] = (mood[tag] || 0) + amount;
  };

  if (ferScores) {
    for (const [tag, score] of Object.entries(ferScores)) add(tag, score * FER_WEIGHT);
  }

  if (auTags) {
    for (const [auTag, score] of Object.entries(auTags)) {
      const mapped = AU_TAG_TO_MOOD[auTag] || {};
      for (const [moodTag, weight] of Object.entries(mapped)) add(moodTag, score * weight * AU_WEIGHT);
    }
  }

  if (gestureTags) {
    for (const [tag, score] of Object.entries(gestureTags)) add(tag, score * GESTURE_WEIGHT);
  }

  if (visionTags) {
    for (const [tag, score] of Object.entries(visionTags)) add(tag, score * VISION_WEIGHT);
  }

  let nonNeutralPeak = 0.0;
  for (const [k, v] of Object.entries(mood)) {
    if (k !== "neutral" && v > nonNeutralPeak) nonNeutralPeak = v;
  }
  if ("neutral" in mood && nonNeutralPeak > NEUTRAL_SUPPRESSION_THRESHOLD) {
    mood.neutral *= NEUTRAL_SUPPRESSION;
  }

  return mood;
}

export function topTags(moodVector, limit = 3, floor = TOP_TAG_FLOOR) {
  const ranked = Object.entries(moodVector).sort((a, b) => b[1] - a[1]);
  const filtered = ranked.filter(([, score]) => score >= floor);
  if (filtered.length === 0) {
    return [["neutral", moodVector.neutral !== undefined ? moodVector.neutral : 1.0]];
  }
  return filtered.slice(0, limit);
}

export function primaryTag(moodVector) {
  return topTags(moodVector, 1)[0][0];
}

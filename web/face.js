// Port of the blendshape half of vision.py: expression tags, calibration,
// baseline drift, and hand gesture classification.

export const CALIBRATION_FRAMES = 12;
export const BASELINE_DRIFT_ALPHA = 0.01;
export const BLENDSHAPE_DRIFT_THRESHOLD = 0.07;

// The blendshape names the tag rules below read. MediaPipe emits 52; these are
// the ones the expression rules actually use.
export const FEATURE_KEYS = [
  "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
  "cheekPuff",
  "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
  "jawOpen",
  "mouthFunnel", "mouthPucker",
  "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
  "noseSneerLeft", "noseSneerRight",
];

export function blendshapeTagsFromDeltas(deltas) {
  const tags = {};
  const avg = (...keys) => keys.reduce((s, k) => s + deltas[k], 0) / keys.length;
  const clip = (value, span) => Math.max(0.0, Math.min(1.0, value / span));

  const jawOpen = deltas.jawOpen;
  if (jawOpen > 0.18) tags.jaw_drop = clip(jawOpen, 0.5);

  const smile = avg("mouthSmileLeft", "mouthSmileRight");
  const smileAsymmetry = Math.abs(deltas.mouthSmileLeft - deltas.mouthSmileRight);
  if (smile > 0.12 && !(jawOpen > 0.35)) tags.smile = clip(smile, 0.55);
  if (smileAsymmetry > 0.15 && smile > 0.05) tags.smirk = clip(smileAsymmetry, 0.35);

  const frown = avg("mouthFrownLeft", "mouthFrownRight");
  if (frown > 0.1) tags.frown = clip(frown, 0.4);

  const browRaise = avg("browInnerUp", "browOuterUpLeft", "browOuterUpRight");
  if (browRaise > 0.15) tags.brow_raise = clip(browRaise, 0.55);

  const browFurrow = avg("browDownLeft", "browDownRight");
  if (browFurrow > 0.15) tags.brow_furrow = clip(browFurrow, 0.5);

  const browAsymmetry = Math.abs(deltas.browOuterUpLeft - deltas.browOuterUpRight);
  if (browAsymmetry > 0.2 && browRaise < 0.3) tags.skeptical = clip(browAsymmetry, 0.45);

  const squint = avg("eyeSquintLeft", "eyeSquintRight");
  if (squint > 0.15) tags.squint = clip(squint, 0.45);

  const blinkAsymmetry = Math.abs(deltas.eyeBlinkLeft - deltas.eyeBlinkRight);
  if (blinkAsymmetry > 0.3) tags.wink = clip(blinkAsymmetry, 0.6);

  const eyeWide = avg("eyeWideLeft", "eyeWideRight");
  if (eyeWide > 0.15) tags.eye_wide = clip(eyeWide, 0.4);

  const sneer = avg("noseSneerLeft", "noseSneerRight");
  if (sneer > 0.12) tags.sneer = clip(sneer, 0.4);

  if (deltas.cheekPuff > 0.15) tags.cheek_puff = clip(deltas.cheekPuff, 0.4);

  const pucker = avg("mouthPucker", "mouthFunnel");
  if (pucker > 0.15) tags.pucker = clip(pucker, 0.45);

  return tags;
}

function median(values) {
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

export class FaceCalibrator {
  constructor() {
    this.baseline = null;
    this.calibrating = false;
    this.samples = [];
  }

  startCalibration() {
    this.calibrating = true;
    this.samples = [];
    this.baseline = null;
  }

  progress() {
    if (!this.calibrating) return 1.0;
    return Math.min(1.0, this.samples.length / CALIBRATION_FRAMES);
  }

  // raw is an object of blendshape name to score for the current frame.
  // Returns the delta object once calibrated, or null while still calibrating.
  update(raw) {
    if (this.calibrating) {
      this.samples.push(raw);
      if (this.samples.length >= CALIBRATION_FRAMES) {
        this.baseline = {};
        for (const key of FEATURE_KEYS) {
          this.baseline[key] = median(this.samples.map((s) => s[key] || 0));
        }
        this.calibrating = false;
      }
    }

    if (this.baseline === null) return null;

    const deltas = {};
    for (const key of FEATURE_KEYS) deltas[key] = (raw[key] || 0) - this.baseline[key];
    this._drift(deltas, raw);
    return deltas;
  }

  _drift(deltas, raw) {
    const vals = Object.values(deltas);
    const magnitude = vals.reduce((s, v) => s + Math.abs(v), 0) / Math.max(1, vals.length);
    if (magnitude < BLENDSHAPE_DRIFT_THRESHOLD) {
      for (const key of FEATURE_KEYS) {
        this.baseline[key] =
          this.baseline[key] * (1 - BASELINE_DRIFT_ALPHA) + (raw[key] || 0) * BASELINE_DRIFT_ALPHA;
      }
    }
  }
}

// Hand gestures, ported from the joint rules in vision.py.
export const GESTURE_TO_TAGS = {
  thumbs_up: { approval: 1.0, happy: 0.5 },
  thumbs_down: { disapproval: 1.0, sad: 0.4 },
  open_palm: { surprise: 0.5, stop: 1.0 },
  fist: { angry: 0.7, determined: 0.6 },
  peace: { happy: 0.6, chill: 1.0 },
  pointing: { suspicious: 0.6, focused: 0.5 },
};

const FINGER_JOINTS = { index: [5, 6, 8], middle: [9, 10, 12], ring: [13, 14, 16], pinky: [17, 18, 20] };

const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

function fingerExtended(lm, joints) {
  const [mcp, pip, tip] = joints;
  return dist(lm[tip], lm[0]) > dist(lm[pip], lm[0]) && dist(lm[tip], lm[0]) > dist(lm[mcp], lm[0]);
}

function thumbExtended(lm) {
  return dist(lm[4], lm[0]) > dist(lm[2], lm[0]) * 1.2;
}

export function classifyGesture(landmarks) {
  if (!landmarks || landmarks.length < 21) return null;
  const ext = {};
  for (const [name, joints] of Object.entries(FINGER_JOINTS)) ext[name] = fingerExtended(landmarks, joints);
  const thumb = thumbExtended(landmarks);
  const count = Object.values(ext).filter(Boolean).length;

  if (count === 0 && thumb) {
    return landmarks[4].y < landmarks[0].y ? "thumbs_up" : "thumbs_down";
  }
  if (count === 4) return "open_palm";
  if (count === 0 && !thumb) return "fist";
  if (ext.index && ext.middle && !ext.ring && !ext.pinky) return "peace";
  if (ext.index && !ext.middle && !ext.ring && !ext.pinky) return "pointing";
  return null;
}

export function tagsFromGestures(gestures) {
  const tags = {};
  for (const g of gestures) {
    const mapped = GESTURE_TO_TAGS[g] || {};
    for (const [tag, score] of Object.entries(mapped)) {
      tags[tag] = Math.max(tags[tag] || 0, score);
    }
  }
  return tags;
}

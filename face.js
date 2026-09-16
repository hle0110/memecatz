// Port of the blendshape half of vision.py: expression tags, calibration,
// baseline drift, and hand gesture classification.

// Threshold scaling. At 1.0 the tag rules behave exactly like the desktop app.
// The browser build has no emotion model feeding the mood vector, so the
// blendshape rules carry all the signal alone and need to be more sensitive.
let SENSITIVITY = 1.0;
export function setSensitivity(value) { SENSITIVITY = Math.max(0.2, Math.min(4.0, value)); }
export function getSensitivity() { return SENSITIVITY; }

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
  // A higher sensitivity lowers every trigger threshold proportionally.
  const t = (threshold) => threshold / SENSITIVITY;

  const jawOpen = deltas.jawOpen;
  if (jawOpen > t(0.18)) tags.jaw_drop = clip(jawOpen, 0.5);

  const smile = avg("mouthSmileLeft", "mouthSmileRight");
  const smileAsymmetry = Math.abs(deltas.mouthSmileLeft - deltas.mouthSmileRight);
  if (smile > t(0.12) && !(jawOpen > 0.35)) tags.smile = clip(smile, 0.55);
  if (smileAsymmetry > t(0.15) && smile > t(0.05)) tags.smirk = clip(smileAsymmetry, 0.35);

  const frown = avg("mouthFrownLeft", "mouthFrownRight");
  if (frown > t(0.1)) tags.frown = clip(frown, 0.4);

  const browRaise = avg("browInnerUp", "browOuterUpLeft", "browOuterUpRight");
  if (browRaise > t(0.15)) tags.brow_raise = clip(browRaise, 0.55);

  const browFurrow = avg("browDownLeft", "browDownRight");
  if (browFurrow > t(0.15)) tags.brow_furrow = clip(browFurrow, 0.5);

  const browAsymmetry = Math.abs(deltas.browOuterUpLeft - deltas.browOuterUpRight);
  if (browAsymmetry > t(0.2) && browRaise < 0.3) tags.skeptical = clip(browAsymmetry, 0.45);

  const squint = avg("eyeSquintLeft", "eyeSquintRight");
  if (squint > t(0.15)) tags.squint = clip(squint, 0.45);

  const blinkAsymmetry = Math.abs(deltas.eyeBlinkLeft - deltas.eyeBlinkRight);
  if (blinkAsymmetry > t(0.3)) tags.wink = clip(blinkAsymmetry, 0.6);

  const eyeWide = avg("eyeWideLeft", "eyeWideRight");
  if (eyeWide > t(0.15)) tags.eye_wide = clip(eyeWide, 0.4);

  const sneer = avg("noseSneerLeft", "noseSneerRight");
  if (sneer > t(0.12)) tags.sneer = clip(sneer, 0.4);

  if (deltas.cheekPuff > t(0.15)) tags.cheek_puff = clip(deltas.cheekPuff, 0.4);

  const pucker = avg("mouthPucker", "mouthFunnel");
  if (pucker > t(0.15)) tags.pucker = clip(pucker, 0.45);

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


// ---------------------------------------------------------------------------
// Continuous expression reading.
//
// blendshapeTagsFromDeltas above is the faithful desktop port: it thresholds,
// so a signal is either absent or present. That was correct on desktop where an
// emotion model carried most of the mood signal, but on its own it throws away
// the precision MediaPipe gives us. mouthSmileLeft arrives as a smooth 0..1
// value every frame; thresholding flattens it to yes or no.
//
// This version keeps the value. A faint smile produces a faint happy, a broad
// one produces a strong happy, and everything in between is represented. The
// only hard cut is a small deadzone that rejects tracker jitter, and past that
// the response is a smoothstep so there is no cliff at the onset.
// ---------------------------------------------------------------------------

export const JITTER_DEADZONE = 0.04;

function ramp(value, span) {
  const top = Math.max(span / SENSITIVITY, JITTER_DEADZONE + 1e-6);
  const t = Math.max(0, Math.min(1, (value - JITTER_DEADZONE) / (top - JITTER_DEADZONE)));
  return t * t * (3 - 2 * t); // smoothstep, zero slope at both ends
}

// Returns every expression signal as a continuous 0..1 strength. Signals at rest
// come back as 0 rather than being omitted, so callers see the full picture.
export function expressionStrengths(deltas) {
  const avg = (...keys) => keys.reduce((s, k) => s + (deltas[k] || 0), 0) / keys.length;
  const d = (k) => deltas[k] || 0;

  const jawOpen = d("jawOpen");
  const smile = avg("mouthSmileLeft", "mouthSmileRight");
  const smileAsymmetry = Math.abs(d("mouthSmileLeft") - d("mouthSmileRight"));
  const frown = avg("mouthFrownLeft", "mouthFrownRight");
  const browRaise = avg("browInnerUp", "browOuterUpLeft", "browOuterUpRight");
  const browFurrow = avg("browDownLeft", "browDownRight");
  const browAsymmetry = Math.abs(d("browOuterUpLeft") - d("browOuterUpRight"));
  const squint = avg("eyeSquintLeft", "eyeSquintRight");
  const blinkAsymmetry = Math.abs(d("eyeBlinkLeft") - d("eyeBlinkRight"));
  const eyeWide = avg("eyeWideLeft", "eyeWideRight");
  const sneer = avg("noseSneerLeft", "noseSneerRight");
  const pucker = avg("mouthPucker", "mouthFunnel");

  // A wide open jaw drags the mouth corners, which reads as a false smile.
  const jawSmileDamp = 1 - Math.max(0, Math.min(1, (jawOpen - 0.3) / 0.3));

  const out = {
    jaw_drop: ramp(jawOpen, 0.5),
    smile: ramp(smile, 0.55) * jawSmileDamp,
    smirk: ramp(smileAsymmetry, 0.35),
    frown: ramp(frown, 0.4),
    brow_raise: ramp(browRaise, 0.55),
    brow_furrow: ramp(browFurrow, 0.5),
    // Asymmetric brows only read as skeptical when both are not simply raised.
    skeptical: ramp(browAsymmetry, 0.45) * (1 - ramp(browRaise, 0.55)),
    squint: ramp(squint, 0.45),
    wink: ramp(blinkAsymmetry, 0.6),
    eye_wide: ramp(eyeWide, 0.4),
    sneer: ramp(sneer, 0.4),
    cheek_puff: ramp(d("cheekPuff"), 0.4),
    pucker: ramp(pucker, 0.45),
  };

  for (const k of Object.keys(out)) {
    if (!Number.isFinite(out[k]) || out[k] <= 0) out[k] = 0;
  }
  return out;
}

// How neutral the face is: 1 when nothing is happening, falling towards 0 as any
// expression takes over. This replaces the baseline the emotion model used to
// contribute on desktop, so neutral competes properly instead of winning by default.
export function neutralStrength(strengths) {
  let peak = 0;
  for (const v of Object.values(strengths)) if (v > peak) peak = v;
  return Math.max(0, Math.min(1, 1 - peak));
}

// Drops the zeros, for display and for feeding the mood engine.
export function activeOnly(strengths, floor = 0.02) {
  const out = {};
  for (const [k, v] of Object.entries(strengths)) if (v > floor) out[k] = v;
  return out;
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

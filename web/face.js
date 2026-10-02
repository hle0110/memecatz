// Port of the face and hand half of vision.py: calibration, baseline drift,
// continuous expression strengths, and hand gesture classification.
//
// The desktop app runs one detection step about every 0.15 seconds. The page
// processes every new camera frame instead, so anything that the desktop app
// applies "once per step" is scaled by elapsed time here. That keeps the
// behaviour the same on a 30 fps webcam, a 60 fps webcam, or a slow phone.

// Threshold scaling. At 1.0 the expression ramps use the desktop spans exactly.
let SENSITIVITY = 1.0;
export function setSensitivity(value) { SENSITIVITY = Math.max(0.2, Math.min(4.0, value)); }
export function getSensitivity() { return SENSITIVITY; }

// Calibration needs both a minimum number of real frames and a minimum amount
// of time. The desktop app collects 12 samples at about 0.15 s apart, so about
// 1.8 s. Using the same duration here keeps one blink or twitch from becoming
// part of the neutral baseline.
export const CALIBRATION_MIN_SAMPLES = 12;
export const CALIBRATION_SECONDS = 1.8;

export const BASELINE_DRIFT_ALPHA = 0.01;        // per desktop detection step
export const DRIFT_REFERENCE_SECONDS = 0.15;     // desktop detection interval
export const BLENDSHAPE_DRIFT_THRESHOLD = 0.07;

// Same 43 names vision.py tracks. The drift check averages over all of them, so
// the list has to match for the 0.07 threshold to mean the same thing.
export const FEATURE_KEYS = [
  "browDownLeft", "browDownRight", "browInnerUp", "browOuterUpLeft", "browOuterUpRight",
  "cheekPuff", "cheekSquintLeft", "cheekSquintRight",
  "eyeBlinkLeft", "eyeBlinkRight", "eyeSquintLeft", "eyeSquintRight", "eyeWideLeft", "eyeWideRight",
  "jawOpen", "jawForward", "jawLeft", "jawRight",
  "mouthClose", "mouthFunnel", "mouthPucker",
  "mouthSmileLeft", "mouthSmileRight", "mouthFrownLeft", "mouthFrownRight",
  "mouthDimpleLeft", "mouthDimpleRight", "mouthStretchLeft", "mouthStretchRight",
  "mouthPressLeft", "mouthPressRight", "mouthLowerDownLeft", "mouthLowerDownRight",
  "mouthUpperUpLeft", "mouthUpperUpRight", "mouthShrugLower", "mouthShrugUpper",
  "mouthRollLower", "mouthRollUpper", "mouthLeft", "mouthRight",
  "noseSneerLeft", "noseSneerRight",
];

// An update rate defined per reference interval, converted to elapsed time.
// rateAdjustedAlpha(a, ref, ref) === a, and two half steps equal one full step.
export function rateAdjustedAlpha(alpha, elapsedSeconds, referenceSeconds) {
  if (!(elapsedSeconds > 0)) return 0;
  return 1 - Math.pow(1 - alpha, elapsedSeconds / referenceSeconds);
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
    this.startedAt = null;
    this.lastUpdate = null;
  }

  startCalibration() {
    this.calibrating = true;
    this.samples = [];
    this.baseline = null;
    this.startedAt = null;
    this.lastUpdate = null;
  }

  // nowMs is a millisecond timestamp, for example performance.now().
  progress(nowMs) {
    if (!this.calibrating) return 1.0;
    if (this.startedAt === null) return 0.0;
    const bySamples = this.samples.length / CALIBRATION_MIN_SAMPLES;
    const byTime = (nowMs - this.startedAt) / (CALIBRATION_SECONDS * 1000);
    return Math.max(0, Math.min(1.0, bySamples, byTime));
  }

  // raw: blendshape name -> score for one new camera frame.
  // Returns deltas from the baseline once calibrated, otherwise null.
  update(raw, nowMs) {
    if (this.calibrating) {
      if (this.startedAt === null) this.startedAt = nowMs;
      this.samples.push(raw);
      const elapsed = (nowMs - this.startedAt) / 1000;
      if (this.samples.length >= CALIBRATION_MIN_SAMPLES && elapsed >= CALIBRATION_SECONDS) {
        this.baseline = {};
        for (const key of FEATURE_KEYS) {
          this.baseline[key] = median(this.samples.map((s) => s[key] || 0));
        }
        this.calibrating = false;
        this.samples = [];
        this.lastUpdate = nowMs;
      }
    }

    if (this.baseline === null) return null;

    const deltas = {};
    for (const key of FEATURE_KEYS) deltas[key] = (raw[key] || 0) - this.baseline[key];

    const dt = this.lastUpdate === null ? 0 : Math.min(1.0, (nowMs - this.lastUpdate) / 1000);
    this.lastUpdate = nowMs;
    this._drift(deltas, raw, dt);
    return deltas;
  }

  _drift(deltas, raw, dtSeconds) {
    const vals = Object.values(deltas);
    const magnitude = vals.reduce((s, v) => s + Math.abs(v), 0) / Math.max(1, vals.length);
    if (magnitude >= BLENDSHAPE_DRIFT_THRESHOLD) return;
    const a = rateAdjustedAlpha(BASELINE_DRIFT_ALPHA, dtSeconds, DRIFT_REFERENCE_SECONDS);
    if (a <= 0) return;
    for (const key of FEATURE_KEYS) {
      this.baseline[key] = this.baseline[key] * (1 - a) + (raw[key] || 0) * a;
    }
  }
}


// ---------------------------------------------------------------------------
// Continuous expression reading.
//
// The desktop rules threshold each signal, so it is either absent or present.
// Here the value is kept: a faint smile produces a faint happy and a broad one
// a strong happy. The only hard cut is a small deadzone that rejects tracker
// jitter, and past that the response is a smoothstep so there is no cliff.
// ---------------------------------------------------------------------------

export const JITTER_DEADZONE = 0.04;

function ramp(value, span) {
  const top = Math.max(span / SENSITIVITY, JITTER_DEADZONE + 1e-6);
  const t = Math.max(0, Math.min(1, (value - JITTER_DEADZONE) / (top - JITTER_DEADZONE)));
  return t * t * (3 - 2 * t); // smoothstep, zero slope at both ends
}

// Every expression signal as a continuous 0..1 strength. Signals at rest come
// back as 0 rather than being omitted, so callers see the full picture.
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

// How neutral the face is: 1 when nothing is happening, falling towards 0 as
// any expression takes over. Used only while the emotion model is not loaded.
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


// ---------------------------------------------------------------------------
// Hand gestures. Same rules and constants as classify_gesture in vision.py,
// computed in pixels like the desktop app. MediaPipe gives x as a fraction of
// the width and y as a fraction of the height, so distances in those raw units
// would squash one axis on any non-square camera.
// ---------------------------------------------------------------------------

export const GESTURE_TO_TAGS = {
  thumbs_up: { approval: 1.0, happy: 0.5 },
  thumbs_down: { disapproval: 1.0, sad: 0.4 },
  open_palm: { surprise: 0.5, stop: 1.0 },
  fist: { angry: 0.7, determined: 0.6 },
  peace: { happy: 0.6, chill: 1.0 },
  pointing: { suspicious: 0.6, focused: 0.5 },
};

const WRIST = 0;
const THUMB_MCP = 2;
const THUMB_TIP = 4;
const MIDDLE_MCP = 9;
const FINGER_JOINTS = { index: [5, 6, 8], middle: [9, 10, 12], ring: [13, 14, 16], pinky: [17, 18, 20] };

const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

export function toPixels(landmarks, width, height) {
  return landmarks.map((p) => [p.x * width, p.y * height]);
}

// points: 21 [x, y] pairs in pixels.
export function classifyGesturePoints(points) {
  if (!points || points.length < 21) return null;
  const wrist = points[WRIST];
  const fingers = {};
  for (const [name, [, pip, tip]] of Object.entries(FINGER_JOINTS)) {
    fingers[name] = dist(wrist, points[tip]) > dist(wrist, points[pip]) * 1.08;
  }
  const thumb = dist(wrist, points[THUMB_TIP]) > dist(wrist, points[THUMB_MCP]) * 1.25;
  const anyFinger = Object.values(fingers).some(Boolean);
  const extended = Object.values(fingers).filter(Boolean).length + (thumb ? 1 : 0);
  const palmCenterY = (wrist[1] + points[MIDDLE_MCP][1]) / 2.0;

  if (thumb && !anyFinger) {
    if (points[THUMB_TIP][1] < palmCenterY - 15) return "thumbs_up";
    if (points[THUMB_TIP][1] > palmCenterY + 15) return "thumbs_down";
  }
  if (extended >= 5) return "open_palm";
  if (extended === 0) return "fist";
  if (fingers.index && fingers.middle && !fingers.ring && !fingers.pinky) return "peace";
  if (fingers.index && !fingers.middle && !fingers.ring && !fingers.pinky) return "pointing";
  return null;
}

export function classifyGesture(landmarks, width, height) {
  if (!landmarks || landmarks.length < 21 || !width || !height) return null;
  return classifyGesturePoints(toPixels(landmarks, width, height));
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

// MemeCatz for the browser. Everything runs on your own machine: the camera
// stream never leaves the page, and the only network calls are for the cat and
// dog photos.

import { combine, topTags, primaryTag } from "./mood.js";
import {
  FaceCalibrator, expressionStrengths, neutralStrength, activeOnly,
  classifyGesture, tagsFromGestures, setSensitivity,
} from "./face.js";
import { ReactionSource } from "./reactions.js";
import { captionFor } from "./captions.js";

// Face runs every frame for responsiveness. Hands change far more slowly and
// cost a second model invocation on the main thread, so they run less often.
const HAND_INTERVAL_MS = 100;
const MOOD_SWITCH_COOLDOWN_MS = 1800;
const SAME_MOOD_ROTATE_MS = 7000;
const MOOD_TOP_LIMIT = 3;

// Tuning for the browser build. The desktop app had an emotion model feeding the
// mood vector alongside these signals; here the blendshape rules are on their own,
// so it reads expressions more eagerly and reacts faster.
// Continuous strengths are inherently smoother than on/off tags, so the filter
// can react faster without getting jumpy.
const MOOD_SMOOTHING_ALPHA = 0.6;   // desktop uses 0.35
const MOOD_FLOOR = 0.06;            // desktop uses 0.12
const DEFAULT_SENSITIVITY = 1.3;

const WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm";
const FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const el = (id) => document.getElementById(id);
const video = el("webcam");
const reactionImg = el("reaction");
const reactionImgNext = el("reaction-next");
const placeholder = el("placeholder");
const capTop = el("cap-top");
const capBottom = el("cap-bottom");
const statusLine = el("status");
const moodLine = el("mood");
const startBtn = el("start");
const recalBtn = el("recalibrate");
const animalSel = el("animal");
const stage = el("stage");
const debugPanel = el("debug");
const sensSlider = el("sensitivity");
const sensValue = el("sens-value");

const calibrator = new FaceCalibrator();
const reactions = new ReactionSource(animalSel ? animalSel.value : "cat");

let faceLandmarker = null;
let handLandmarker = null;
let running = false;
let smoothed = {};
let lastHandDetect = 0;
let lastGestureTags = {};
let overlayCtx = null;
let lastSwitch = 0;
let lastRotate = 0;
let lastMoodKey = null;
let currentKey = null;
let showingPrimary = true;
let lastDeltas = null;
let debugOn = new URLSearchParams(location.search).has("debug");

// Exponential moving average over the mood vector, same as smooth_mood_vector.
export function smoothMoodVector(previous, current, alpha = MOOD_SMOOTHING_ALPHA) {
  const out = {};
  const tags = new Set([...Object.keys(previous), ...Object.keys(current)]);
  for (const tag of tags) {
    const p = previous[tag] || 0;
    const c = current[tag] || 0;
    out[tag] = p + alpha * (c - p);
  }
  return out;
}

function setStatus(text) {
  if (statusLine) statusLine.textContent = text;
}

async function loadModels() {
  setStatus("loading the face and hand models, this takes a few seconds the first time");
  const { FilesetResolver, FaceLandmarker, HandLandmarker } = await import(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/vision_bundle.mjs"
  );
  const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

  faceLandmarker = await FaceLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: FACE_MODEL, delegate: "GPU" },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
  });

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: HAND_MODEL, delegate: "GPU" },
    runningMode: "VIDEO",
    numHands: 2,
  });
}

async function startCamera() {
  const stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

function blendshapesToObject(categories) {
  const out = {};
  for (const c of categories) out[c.categoryName] = c.score;
  return out;
}

function swapReaction(url) {
  // Crossfade by fading the incoming image in over the outgoing one.
  const incoming = showingPrimary ? reactionImgNext : reactionImg;
  const outgoing = showingPrimary ? reactionImg : reactionImgNext;
  incoming.onload = () => {
    incoming.classList.add("visible");
    outgoing.classList.remove("visible");
    placeholder.classList.add("hidden");
    showingPrimary = !showingPrimary;
  };
  incoming.onerror = () => {
    console.warn("reaction image failed to load, keeping the current one");
  };
  incoming.src = url;
}

function updateReaction(moodTags) {
  const now = Date.now();
  const moodKey = moodTags.length ? moodTags[0] : "neutral";
  const changed = moodKey !== lastMoodKey;
  const cooled = now - lastSwitch > MOOD_SWITCH_COOLDOWN_MS;
  const rotateDue = now - lastRotate > SAME_MOOD_ROTATE_MS;

  if (!((changed && cooled) || (!changed && rotateDue))) return;

  const pick = reactions.pick(currentKey);
  if (!pick) {
    // Nothing real available yet. Say so rather than showing something fake.
    placeholder.classList.remove("hidden");
    placeholder.textContent = "connecting for a real reaction...";
    return;
  }

  const caption = captionFor(moodKey);
  capTop.textContent = caption.top;
  capBottom.textContent = caption.bottom;
  swapReaction(pick.url);

  currentKey = pick.key;
  lastSwitch = now;
  lastRotate = now;
  lastMoodKey = moodKey;
}


// Live readout of what the face is actually producing. Open with ?debug=1 or the
// Show detail button. This is the thing to read when moods feel wrong.
function renderDebug(deltas, auTags, gestureTags, vector, ranked, neutral) {
  if (!debugOn || !debugPanel) return;

  const fmt = (obj, n = 6) =>
    Object.entries(obj)
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, n)
      .map(([k, v]) => `${k} ${v.toFixed(3)}`)
      .join("\n") || "(none)";

  debugPanel.textContent =
    "biggest blendshape deltas\n" + (deltas ? fmt(deltas) : "(calibrating)") +
    "\n\nexpression strengths\n" + fmt(auTags, 8) +
    "\n\nneutral strength\n" + (neutral === undefined ? "?" : neutral.toFixed(3)) +
    "\n\ngesture tags\n" + fmt(gestureTags) +
    "\n\nmood this frame\n" + fmt(vector) +
    "\n\nsmoothed and ranked\n" + (ranked.map(([t, s]) => `${t} ${s.toFixed(3)}`).join("\n") || "(none)") +
    "\n\ncaption mood: " + (ranked.length ? ranked[0][0] : "neutral");
}

function drawMesh(landmarks) {
  if (!overlayCtx) return;
  const c = overlayCtx.canvas;
  overlayCtx.clearRect(0, 0, c.width, c.height);
  if (!landmarks || !landmarks.length) return;

  // Mirrored to match the flipped video so the dots sit on your face.
  overlayCtx.fillStyle = "rgba(120, 224, 143, 0.75)";
  for (const p of landmarks) {
    const x = (1 - p.x) * c.width;
    const y = p.y * c.height;
    overlayCtx.fillRect(x, y, 1.6, 1.6);
  }
}

function loop() {
  if (!running) return;
  const now = Date.now();

  if (video.readyState >= 2) {
    const ts = performance.now();
    let auTags = {};
    let strengths = null;
    let neutral = 1;

    // Face: every frame.
    try {
      const faceResult = faceLandmarker.detectForVideo(video, ts);
      const shapes = faceResult.faceBlendshapes && faceResult.faceBlendshapes[0];
      drawMesh(faceResult.faceLandmarks && faceResult.faceLandmarks[0]);

      if (shapes && shapes.categories) {
        const raw = {};
        for (const cat of shapes.categories) raw[cat.categoryName] = cat.score;
        const deltas = calibrator.update(raw);
        lastDeltas = deltas;
        if (deltas) {
          strengths = expressionStrengths(deltas);
          auTags = activeOnly(strengths);
          neutral = neutralStrength(strengths);
        } else {
          setStatus(`getting ready... ${Math.round(calibrator.progress() * 100)}%`);
        }
      }
    } catch (err) {
      console.warn("face frame skipped:", err.message);
    }

    // Hands: throttled.
    if (now - lastHandDetect >= HAND_INTERVAL_MS) {
      lastHandDetect = now;
      try {
        const handResult = handLandmarker.detectForVideo(video, ts);
        const gestures = [];
        for (const lm of handResult.landmarks || []) {
          const g = classifyGesture(lm);
          if (g) gestures.push(g);
        }
        lastGestureTags = tagsFromGestures(gestures);
      } catch (err) {
        console.warn("hand frame skipped:", err.message);
      }
    }

    if (!calibrator.calibrating && calibrator.baseline && strengths) {
      // Neutral rides in through the emotion-score slot, which is what the
      // desktop build used it for, so the existing neutral suppression applies.
      const vector = combine({
        ferScores: { neutral },
        auTags,
        gestureTags: lastGestureTags,
      });
      smoothed = smoothMoodVector(smoothed, vector);
      const ranked = topTags(smoothed, MOOD_TOP_LIMIT, MOOD_FLOOR);
      const tags = ranked.map(([t]) => t);

      moodLine.textContent = ranked.map(([t, s]) => `${t} ${s.toFixed(2)}`).join("   ");
      setStatus(reactions.describeSource());
      updateReaction(tags);
      renderDebug(lastDeltas, auTags, lastGestureTags, vector, ranked, neutral);
    }
  }

  requestAnimationFrame(loop);
}

async function start() {
  startBtn.disabled = true;
  try {
    setStatus("asking for camera permission");
    await startCamera();
    await loadModels();
    const overlay = el("overlay");
    if (overlay) {
      overlay.width = video.videoWidth || 640;
      overlay.height = video.videoHeight || 480;
      overlayCtx = overlay.getContext("2d");
    }
    calibrator.startCalibration();
    reactions.refill();
    running = true;
    stage.classList.add("live");
    recalBtn.disabled = false;
    setStatus("hold still for a second while it calibrates");
    loop();
  } catch (err) {
    startBtn.disabled = false;
    setStatus(`could not start: ${err.message}`);
    console.error(err);
  }
}

setSensitivity(DEFAULT_SENSITIVITY);
if (sensSlider) {
  sensSlider.value = String(DEFAULT_SENSITIVITY);
  if (sensValue) sensValue.textContent = DEFAULT_SENSITIVITY.toFixed(1);
  sensSlider.addEventListener("input", () => {
    const v = parseFloat(sensSlider.value);
    setSensitivity(v);
    if (sensValue) sensValue.textContent = v.toFixed(1);
  });
}

const debugBtn = el("toggle-debug");
if (debugBtn) {
  if (debugOn && debugPanel) debugPanel.classList.remove("hidden");
  debugBtn.addEventListener("click", () => {
    debugOn = !debugOn;
    if (debugPanel) debugPanel.classList.toggle("hidden", !debugOn);
    debugBtn.textContent = debugOn ? "Hide detail" : "Show detail";
  });
}

if (startBtn) startBtn.addEventListener("click", start);
if (recalBtn)
  recalBtn.addEventListener("click", () => {
    calibrator.startCalibration();
    smoothed = {};
    setStatus("recalibrating, hold a neutral face");
  });
if (animalSel)
  animalSel.addEventListener("change", () => {
    reactions.setAnimal(animalSel.value);
    reactions.refill();
    lastMoodKey = null;
  });

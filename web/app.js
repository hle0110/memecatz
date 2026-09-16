// MemeCatz for the browser. Everything runs on your own machine: the camera
// stream never leaves the page, and the only network calls are for the cat and
// dog photos.

import { combine, topTags, primaryTag } from "./mood.js";
import { FaceCalibrator, blendshapeTagsFromDeltas, classifyGesture, tagsFromGestures } from "./face.js";
import { ReactionSource } from "./reactions.js";
import { captionFor } from "./captions.js";

// Same tuning constants as the desktop app.
const DETECTION_INTERVAL_MS = 150;
const MOOD_SWITCH_COOLDOWN_MS = 1800;
const SAME_MOOD_ROTATE_MS = 7000;
const MOOD_TOP_LIMIT = 3;
const MOOD_SMOOTHING_ALPHA = 0.35;

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

const calibrator = new FaceCalibrator();
const reactions = new ReactionSource(animalSel ? animalSel.value : "cat");

let faceLandmarker = null;
let handLandmarker = null;
let running = false;
let smoothed = {};
let lastDetect = 0;
let lastSwitch = 0;
let lastRotate = 0;
let lastMoodKey = null;
let currentKey = null;
let showingPrimary = true;

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

function loop() {
  if (!running) return;
  const now = Date.now();

  if (now - lastDetect >= DETECTION_INTERVAL_MS && video.readyState >= 2) {
    lastDetect = now;
    const ts = performance.now();

    let auTags = {};
    let gestureTags = {};

    try {
      const faceResult = faceLandmarker.detectForVideo(video, ts);
      const shapes = faceResult.faceBlendshapes && faceResult.faceBlendshapes[0];
      if (shapes && shapes.categories) {
        const raw = blendshapesToObject(shapes.categories);
        const deltas = calibrator.update(raw);
        if (deltas) {
          auTags = blendshapeTagsFromDeltas(deltas);
        } else {
          setStatus(`getting ready... ${Math.round(calibrator.progress() * 100)}%`);
        }
      }
    } catch (err) {
      console.warn("face detection frame skipped:", err.message);
    }

    try {
      const handResult = handLandmarker.detectForVideo(video, ts);
      const gestures = [];
      for (const lm of handResult.landmarks || []) {
        const g = classifyGesture(lm);
        if (g) gestures.push(g);
      }
      gestureTags = tagsFromGestures(gestures);
    } catch (err) {
      console.warn("hand detection frame skipped:", err.message);
    }

    if (!calibrator.calibrating && calibrator.baseline) {
      const vector = combine({ auTags, gestureTags });
      smoothed = smoothMoodVector(smoothed, vector);
      const ranked = topTags(smoothed, MOOD_TOP_LIMIT);
      const tags = ranked.map(([t]) => t);

      moodLine.textContent = ranked.map(([t, s]) => `${t} ${s.toFixed(2)}`).join("   ");
      setStatus(reactions.describeSource());
      updateReaction(tags);
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

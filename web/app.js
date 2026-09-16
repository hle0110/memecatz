// MemeCatz for the browser. The camera stream never leaves the page. Network
// calls are for the reaction media (Giphy through this site's own proxy, or
// The Cat API / Dog CEO), and for the MediaPipe and LiteRT runtimes from a CDN.

import { combine, topTags } from "./mood.js";
import {
  FaceCalibrator, expressionStrengths, neutralStrength, activeOnly,
  classifyGesture, tagsFromGestures, setSensitivity,
} from "./face.js";
import { ReactionSource } from "./reactions.js";
import { captionFor } from "./captions.js";
import { EmotionDetector } from "./emotion.js";

// Face runs every frame for responsiveness. Hands change far more slowly and
// cost a second model invocation on the main thread, so they run less often.
const HAND_INTERVAL_MS = 100;
const EMOTION_INTERVAL_MS = 100;
const MOOD_SWITCH_COOLDOWN_MS = 1800;
const SAME_MOOD_ROTATE_MS = 7000;
const MOOD_TOP_LIMIT = 3;

// Tuning for the browser build. Expression strengths are continuous rather than
// on/off, which is inherently smoother, so the filter can react faster than the
// desktop app without getting jumpy.
const MOOD_SMOOTHING_ALPHA = 0.6;   // desktop uses 0.35
const MOOD_FLOOR = 0.06;            // desktop uses 0.12
const DEFAULT_SENSITIVITY = 1.3;

// Pinned to major version 1 so a future breaking release cannot take the site down.
const MEDIAPIPE_MODULE = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1/vision_bundle.mjs";
const WASM_ROOT = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1/wasm";
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
let lastEmotionDetect = 0;
let lastFerScores = null;
let emotionBusy = false;
const emotion = new EmotionDetector();
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

const MOOD_WORDS = {
  happy: "happy", sad: "sad", angry: "angry", surprise: "surprised", fear: "scared",
  disgust: "grossed out", neutral: "neutral", smug: "smug", confused: "confused",
  mischief: "up to something", annoyed: "annoyed", approval: "approving",
  disapproval: "disapproving", chill: "chill", suspicious: "suspicious",
  triumph: "triumphant", anxious: "anxious", bored: "bored", mocking: "unimpressed",
  stop: "saying stop", determined: "determined", focused: "focused",
};

function friendlyMood(ranked) {
  if (!ranked.length) return "";
  const [tag, score] = ranked[0];
  const word = MOOD_WORDS[tag] || tag.replace(/_/g, " ");
  const strength = score > 0.7 ? "very " : score > 0.35 ? "" : "a little ";
  return `you look ${strength}${word}`;
}

function friendlyError(err) {
  const name = err && err.name;
  if (name === "NotAllowedError" || name === "PermissionDeniedError")
    return "camera permission was denied. allow it in your browser's site settings, then press Start again.";
  if (name === "NotFoundError" || name === "DevicesNotFoundError")
    return "no camera was found on this device.";
  if (name === "NotReadableError" || name === "TrackStartError")
    return "the camera is in use by another app. close it and press Start again.";
  return `could not start: ${err && err.message ? err.message : err}`;
}

function setStatus(text) {
  if (statusLine) statusLine.textContent = text;
}

async function loadModels() {
  setStatus("loading the face and hand models, this takes a few seconds the first time");
  const { FilesetResolver, FaceLandmarker, HandLandmarker } = await import(MEDIAPIPE_MODULE);
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

  const pick = reactions.pick(moodTags, currentKey);
  if (!pick) {
    // Nothing real available yet. Say so rather than showing something fake.
    placeholder.classList.remove("hidden");
    placeholder.textContent = "connecting for a real reaction...";
    return;
  }

  const caption = captionFor(moodKey);
  capTop.textContent = caption.top;
  capBottom.textContent = caption.bottom;
  const attr = el("attribution");
  if (attr) attr.textContent = pick.attribution || "";
  swapReaction(pick.url);

  currentKey = pick.key;
  lastSwitch = now;
  lastRotate = now;
  lastMoodKey = moodKey;
}


// Live readout of what the face is actually producing. Open with ?debug=1 or the
// Show detail button. This is the thing to read when moods feel wrong.
function renderDebug(deltas, auTags, gestureTags, vector, ranked, neutral, fer) {
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
    "\n\nemotion model (7 classes)\n" + (fer ? fmt(fer, 7) : "(not loaded, using face signals only)") +
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
      const landmarks = faceResult.faceLandmarks && faceResult.faceLandmarks[0];
      drawMesh(landmarks);

      // Emotion model, throttled. Its 7 scores feed the mood engine exactly as
      // they do on desktop. If it never loaded, lastFerScores stays null.
      // Never start a second inference while one is running: the model and
      // its scratch canvas are shared state.
      if (emotion.ready && landmarks && !emotionBusy && now - lastEmotionDetect >= EMOTION_INTERVAL_MS) {
        lastEmotionDetect = now;
        emotionBusy = true;
        emotion.detect(video, landmarks)
          .then((scores) => { if (scores) lastFerScores = scores; })
          .finally(() => { emotionBusy = false; });
      }

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
      // With the emotion model running, its real 7-class output takes the slot
      // it has on desktop. Without it, a synthetic neutral keeps the vector honest.
      const ferScores = lastFerScores ? lastFerScores : { neutral };
      const vector = combine({
        ferScores,
        auTags,
        gestureTags: lastGestureTags,
      });
      smoothed = smoothMoodVector(smoothed, vector);
      const ranked = topTags(smoothed, MOOD_TOP_LIMIT, MOOD_FLOOR);
      const tags = ranked.map(([t]) => t);

      moodLine.textContent = friendlyMood(ranked);
      setStatus(reactions.describeSource() + (emotion.ready ? "  |  emotion model on" : ""));
      updateReaction(tags);
      renderDebug(lastDeltas, auTags, lastGestureTags, vector, ranked, neutral, lastFerScores);
    }
  }

  requestAnimationFrame(loop);
}

async function start() {
  startBtn.disabled = true;
  startBtn.textContent = "Starting...";
  document.body.classList.add("loading");
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
    reactions.warm(["neutral", "happy"]);
    running = true;
    stage.classList.add("live");
    recalBtn.disabled = false;
    setStatus("hold still for a second while it calibrates");
    startBtn.textContent = "Running";
    loop();
  } catch (err) {
    startBtn.disabled = false;
    startBtn.textContent = "Start camera";
    setStatus(friendlyError(err));
    console.error(err);
  } finally {
    document.body.classList.remove("loading");
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
document.addEventListener("keydown", (e) => {
  if (e.key === "c" && running && !e.metaKey && !e.ctrlKey && e.target === document.body) {
    recalBtn.click();
  }
});
if (recalBtn)
  recalBtn.addEventListener("click", () => {
    calibrator.startCalibration();
    smoothed = {};
    setStatus("recalibrating, hold a neutral face");
  });
if (animalSel)
  animalSel.addEventListener("change", () => {
    reactions.setAnimal(animalSel.value);
    reactions.warm(["neutral", "happy"]);
    lastMoodKey = null;
  });

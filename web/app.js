// MemeCatz for the browser. The camera stream never leaves the page. Network
// calls are for the reaction media (Giphy, or The Cat API / Dog CEO) and for
// the MediaPipe and LiteRT runtimes from a CDN.

import { combine, topTags } from "./mood.js";
import {
  FaceCalibrator, expressionStrengths, neutralStrength, activeOnly,
  classifyGesture, tagsFromGestures, setSensitivity, rateAdjustedAlpha,
} from "./face.js";
import { ReactionSource } from "./reactions.js";
import { captionFor } from "./captions.js";
import { EmotionDetector } from "./emotion.js";
import {
  PROMPTS, ROUND_LIMIT_MS, RoundTimer, dailyPrompts, dailyNumber, parsePromptList, isDateString,
  utcDateString, scoreRun, formatSeconds, shareText, recordResult, currentStreak, isOfficialDone,
  normalizeStore, bestRoundIndex, faceOnlyVector, roundSignal,
} from "./challenge.js";

// Face runs on every new camera frame. Hands and the emotion model change more
// slowly and each cost a model run on the main thread, so they are throttled.
const HAND_INTERVAL_MS = 100;
const EMOTION_INTERVAL_MS = 100;
const MOOD_SWITCH_COOLDOWN_MS = 1800;
const SAME_MOOD_ROTATE_MS = 7000;
const MOOD_TOP_LIMIT = 3;
const FACE_LOST_MS = 1000;

// Mood smoothing, defined per 1/30 s and scaled by real elapsed time so a
// 60 fps camera and a 15 fps phone camera feel the same. Expression strengths
// are continuous, so this can react faster than the desktop app (0.35 per
// 0.15 s step) without getting jumpy.
const MOOD_SMOOTHING_ALPHA = 0.6;
const MOOD_SMOOTHING_REFERENCE_S = 1 / 30;
const MOOD_FLOOR = 0.06;            // desktop uses 0.12
const DEFAULT_SENSITIVITY = 1.3;

// Exact versions, so a new release can never change the site without a deploy.
const MEDIAPIPE_VERSION = "1.0.1";
const MEDIAPIPE_MODULE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/vision_bundle.mjs`;
const WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}/wasm`;
const FACE_MODEL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";
const HAND_MODEL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

const params = new URLSearchParams(location.search);
const FORCE_CPU = params.has("cpu");
// Testing hooks for the daily challenge: ?prompts=happy,approval sets the
// prompts (such runs are always practice) and ?date=2026-10-04 sets the day.
const PROMPT_OVERRIDE = parsePromptList(params.get("prompts"));
const DATE_OVERRIDE = isDateString(params.get("date")) ? params.get("date") : null;
const STORE_KEY = "memecatz.daily";
const COUNTDOWN_MS = 3000;
const RESULT_MS = 3500;
const RETRY_PICK_MS = 300;

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
const stopBtn = el("stop");
const recalBtn = el("recalibrate");
const snapBtn = el("snapshot");
const animalSel = el("animal");
const stage = el("stage");
const debugPanel = el("debug");
const sensSlider = el("sensitivity");
const sensValue = el("sens-value");
const attributionEl = el("attribution");

const calibrator = new FaceCalibrator();
const reactions = new ReactionSource(animalSel ? animalSel.value : "cat");
const emotion = new EmotionDetector();

let vision = null;
let faceLandmarker = null;
let handLandmarker = null;
let delegateUsed = null;
let running = false;
let stream = null;
let frameHandle = null;
let lastVideoTime = -1;
let lastFrameAt = null;
let smoothed = {};
let smoothedFace = {};        // face signals only, for challenge face rounds
let lastHandDetect = 0;
let lastEmotionDetect = 0;
let lastFerScores = null;
let emotionBusy = false;
let faceEpoch = 0;            // bumped when the face is lost, drops late results
let lastGestureTags = {};
let lastGestures = [];        // hand shapes from the latest hand check
let overlayCtx = null;
let lastSwitch = 0;
let lastRotate = 0;
let lastMoodKey = null;
let currentEntry = null;      // last picked, may still be loading
let currentCaption = null;
let displayedEntry = null;    // what is on screen right now
let displayedCaption = null;
let showingPrimary = true;
let lastFaceSeen = 0;
let missedFaceFrames = 0;
let debugOn = params.has("debug");
let mode = "free";
let challenge = null;         // the daily run in progress
let lastRun = null;           // the last finished run, for sharing

// Exponential moving average over the mood vector, same as smooth_mood_vector.
export function smoothMoodVector(previous, current, alpha) {
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
  stop: "like you're saying stop", determined: "determined", focused: "focused",
};

function friendlyMood(ranked) {
  if (!ranked.length) return "";
  const [tag, score] = ranked[0];
  if (tag === "neutral") return "you look calm";
  const word = MOOD_WORDS[tag] || tag.replace(/_/g, " ");
  const strength = score > 0.7 ? "very " : score > 0.35 ? "" : "a little ";
  return `you look ${word.startsWith("like ") ? "" : strength}${word}`;
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

let lastStatus = "";
function setStatus(text) {
  if (statusLine && text !== lastStatus) {
    statusLine.textContent = text;
    lastStatus = text;
  }
}

let lastMoodText = "";
function setMoodText(text) {
  if (moodLine && text !== lastMoodText) {
    moodLine.textContent = text;
    lastMoodText = text;
  }
}

// GPU first, CPU if the GPU path is unavailable on this device or browser.
async function createWithFallback(Task, options) {
  const delegates = FORCE_CPU ? ["CPU"] : ["GPU", "CPU"];
  let lastErr = null;
  for (const delegate of delegates) {
    try {
      const task = await Task.createFromOptions(vision, {
        ...options,
        baseOptions: { ...options.baseOptions, delegate },
      });
      return { task, delegate };
    } catch (err) {
      lastErr = err;
      console.warn(`${delegate} delegate failed, trying the next option:`, err && err.message ? err.message : err);
    }
  }
  throw lastErr;
}

async function loadModels() {
  if (faceLandmarker && handLandmarker) return;
  setStatus("loading the face and hand models, this takes a few seconds the first time");
  const { FilesetResolver, FaceLandmarker, HandLandmarker } = await import(MEDIAPIPE_MODULE);
  vision = await FilesetResolver.forVisionTasks(WASM_ROOT);

  const face = await createWithFallback(FaceLandmarker, {
    baseOptions: { modelAssetPath: FACE_MODEL },
    runningMode: "VIDEO",
    numFaces: 1,
    outputFaceBlendshapes: true,
  });
  faceLandmarker = face.task;
  delegateUsed = face.delegate;

  const hand = await createWithFallback(HandLandmarker, {
    baseOptions: { modelAssetPath: HAND_MODEL },
    runningMode: "VIDEO",
    numHands: 2,
    minHandDetectionConfidence: 0.6,
  });
  handLandmarker = hand.task;
}

async function startCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

function stopCamera() {
  if (stream) {
    for (const track of stream.getTracks()) track.stop();
    stream = null;
  }
  video.srcObject = null;
}

function swapReaction(entry, caption) {
  // Crossfade by fading the incoming image in over the outgoing one.
  const incoming = showingPrimary ? reactionImgNext : reactionImg;
  const outgoing = showingPrimary ? reactionImg : reactionImgNext;
  incoming.onload = () => {
    // Caption, attribution and image change together, once the image is ready.
    capTop.textContent = caption.top;
    capBottom.textContent = caption.bottom;
    if (attributionEl) attributionEl.textContent = entry.attribution || "";
    displayedEntry = entry;
    displayedCaption = caption;
    incoming.classList.add("visible");
    outgoing.classList.remove("visible");
    placeholder.classList.add("hidden");
    showingPrimary = !showingPrimary;
  };
  incoming.onerror = () => {
    // A CORS-mode load can fail if the image host skips its CORS header on a
    // cached copy. Try once more as a normal image: it still shows, it just
    // cannot go into a snapshot.
    if (entry.cors) {
      entry.cors = false;
      incoming.removeAttribute("crossorigin");
      incoming.src = entry.url;
      return;
    }
    console.warn("reaction image failed to load, keeping the current one");
  };
  // Sources that send CORS headers are loaded in CORS mode so a snapshot can
  // include them. This has to be set before src.
  if (entry.cors) incoming.crossOrigin = "anonymous";
  else incoming.removeAttribute("crossorigin");
  incoming.alt = entry.name || "a real cat or dog reacting";
  incoming.src = entry.url;
}

function updateReaction(moodTags, now) {
  const moodKey = moodTags.length ? moodTags[0] : "neutral";
  const changed = moodKey !== lastMoodKey;
  const cooled = now - lastSwitch > MOOD_SWITCH_COOLDOWN_MS;
  const rotateDue = now - lastRotate > SAME_MOOD_ROTATE_MS;

  if (!((changed && cooled) || (!changed && rotateDue))) return;

  const pick = reactions.pick(moodTags, currentEntry ? currentEntry.key : null);
  if (!pick) {
    // Nothing real available yet. Say so rather than showing something fake.
    if (!currentEntry) {
      placeholder.classList.remove("hidden");
      placeholder.textContent = "connecting for a real reaction...";
    }
    return;
  }

  currentCaption = captionFor(moodKey, currentCaption);
  swapReaction(pick, currentCaption);

  currentEntry = pick;
  lastSwitch = now;
  lastRotate = now;
  lastMoodKey = moodKey;
}

// Live readout of what the face is producing. Open with ?debug=1 or the Show
// detail button. This is the thing to read when moods feel wrong.
function renderDebug(info) {
  if (!debugOn || !debugPanel) return;

  const fmt = (obj, n = 6) =>
    Object.entries(obj || {})
      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
      .slice(0, n)
      .map(([k, v]) => `${k} ${v.toFixed(3)}`)
      .join("\n") || "(none)";

  const emotionState = emotion.ready ? (info.fer ? fmt(info.fer, 7) : "(waiting for a face)")
    : emotion.failed ? "(could not load, using face signals only)" : "(loading)";

  debugPanel.textContent =
    `runtime: ${delegateUsed || "?"}  camera: ${video.videoWidth}x${video.videoHeight}  fps: ${info.fps.toFixed(0)}` +
    "\n\nbiggest blendshape deltas\n" + (info.deltas ? fmt(info.deltas) : "(calibrating)") +
    "\n\nexpression strengths\n" + fmt(info.auTags, 8) +
    "\n\nemotion model (7 classes)\n" + emotionState +
    "\n\nneutral strength (used only without the emotion model)\n" + info.neutral.toFixed(3) +
    "\n\ngesture tags\n" + fmt(info.gestureTags) +
    "\n\nmood this frame\n" + fmt(info.vector) +
    "\n\nsmoothed and ranked\n" + (info.ranked.map(([t, s]) => `${t} ${s.toFixed(3)}`).join("\n") || "(none)") +
    "\n\ncaption mood: " + (info.ranked.length ? info.ranked[0][0] : "neutral");
}

function drawMesh(landmarks) {
  if (!overlayCtx) return;
  const c = overlayCtx.canvas;
  // Phones can change the camera resolution when rotated; keep the overlay in step.
  if (video.videoWidth && (c.width !== video.videoWidth || c.height !== video.videoHeight)) {
    c.width = video.videoWidth;
    c.height = video.videoHeight;
  }
  overlayCtx.clearRect(0, 0, c.width, c.height);
  if (!landmarks || !landmarks.length) return;

  // Mirrored to match the flipped video so the dots sit on your face.
  overlayCtx.fillStyle = "rgba(120, 224, 143, 0.75)";
  for (const p of landmarks) {
    overlayCtx.fillRect((1 - p.x) * c.width, p.y * c.height, 1.6, 1.6);
  }
}

let fpsValue = 0;
function processFrame(nowMs) {
  const dt = lastFrameAt === null ? 0 : (nowMs - lastFrameAt) / 1000;
  lastFrameAt = nowMs;
  if (dt > 0) fpsValue = fpsValue ? fpsValue * 0.9 + (1 / dt) * 0.1 : 1 / dt;

  const now = Date.now();
  let auTags = {};
  let strengths = null;
  let neutral = 1;
  let deltas = null;
  let faceFound = false;

  try {
    const faceResult = faceLandmarker.detectForVideo(video, nowMs);
    const shapes = faceResult.faceBlendshapes && faceResult.faceBlendshapes[0];
    const landmarks = faceResult.faceLandmarks && faceResult.faceLandmarks[0];
    drawMesh(landmarks);

    // Emotion model, throttled, never two runs at once: the model and its
    // scratch canvas are shared state.
    if (emotion.ready && landmarks && !emotionBusy && now - lastEmotionDetect >= EMOTION_INTERVAL_MS) {
      lastEmotionDetect = now;
      emotionBusy = true;
      const epoch = faceEpoch;
      emotion.detect(video, landmarks)
        .then((scores) => { if (scores && epoch === faceEpoch) lastFerScores = scores; })
        .catch((err) => console.warn("emotion frame skipped:", err && err.message ? err.message : err))
        .finally(() => { emotionBusy = false; });
    }

    if (shapes && shapes.categories) {
      faceFound = true;
      lastFaceSeen = now;
      missedFaceFrames = 0;
      const raw = {};
      for (const cat of shapes.categories) raw[cat.categoryName] = cat.score;
      deltas = calibrator.update(raw, nowMs);
      if (deltas) {
        strengths = expressionStrengths(deltas);
        auTags = activeOnly(strengths);
        neutral = neutralStrength(strengths);
      }
    }
  } catch (err) {
    console.warn("face frame skipped:", err.message);
  }

  if (now - lastHandDetect >= HAND_INTERVAL_MS) {
    lastHandDetect = now;
    try {
      const handResult = handLandmarker.detectForVideo(video, nowMs);
      const gestures = [];
      for (const lm of handResult.landmarks || []) {
        const g = classifyGesture(lm, video.videoWidth, video.videoHeight);
        if (g) gestures.push(g);
      }
      lastGestures = gestures;
      lastGestureTags = tagsFromGestures(gestures);
    } catch (err) {
      console.warn("hand frame skipped:", err.message);
    }
  }

  if (!faceFound) {
    lastFerScores = null;
    faceEpoch += 1;
    missedFaceFrames += 1;
    // Needs several frames in a row, so a slow device that takes over a second
    // per frame does not flash this message while your face is in view.
    if (missedFaceFrames >= 3 && now - lastFaceSeen > FACE_LOST_MS) {
      setStatus("can't see your face. face the camera with good light.");
      setMoodText("");
    }
    challengeTick(null, [], false);
    return;
  }

  if (calibrator.calibrating) {
    setStatus(`hold a relaxed face while it calibrates... ${Math.round(calibrator.progress(nowMs) * 100)}%`);
    challengeTick(null, [], false);
    return;
  }
  if (!strengths) { challengeTick(null, [], false); return; }

  // The emotion model's 7 scores fill the same slot they fill on desktop.
  // Without the model, a synthetic neutral keeps the vector honest.
  const ferScores = lastFerScores ? lastFerScores : { neutral };
  const vector = combine({ ferScores, auTags, gestureTags: lastGestureTags });
  const alpha = rateAdjustedAlpha(MOOD_SMOOTHING_ALPHA, dt, MOOD_SMOOTHING_REFERENCE_S);
  smoothed = smoothMoodVector(smoothed, vector, alpha);
  const ranked = topTags(smoothed, MOOD_TOP_LIMIT, MOOD_FLOOR);
  smoothedFace = smoothMoodVector(smoothedFace, faceOnlyVector(ferScores, auTags), alpha);
  const tags = ranked.map(([t]) => t);

  setMoodText(friendlyMood(ranked));
  const emotionNote = emotion.ready ? "  |  emotion model on" : emotion.failed ? "" : "  |  emotion model loading";
  setStatus(reactions.describeSource() + emotionNote);
  if (challenge) challengeTick(topTags(smoothedFace, MOOD_TOP_LIMIT, MOOD_FLOOR), lastGestures, true);
  else if (mode === "free") updateReaction(tags, now);
  renderDebug({ deltas, auTags, gestureTags: lastGestureTags, vector, ranked, neutral, fer: lastFerScores, fps: fpsValue });
}

// Runs once per new camera frame. requestVideoFrameCallback is used where the
// browser has it; elsewhere animation frames are checked for a new video time.
function scheduleNext() {
  if (!running) return;
  if ("requestVideoFrameCallback" in HTMLVideoElement.prototype) {
    frameHandle = { kind: "vfc", id: video.requestVideoFrameCallback(onVideoFrame) };
  } else {
    frameHandle = { kind: "raf", id: requestAnimationFrame(onAnimationFrame) };
  }
}

function onVideoFrame(nowMs) {
  if (!running) return;
  if (video.readyState >= 2) processFrame(nowMs);
  scheduleNext();
}

function onAnimationFrame(nowMs) {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    processFrame(nowMs);
  }
  scheduleNext();
}

function cancelFrames() {
  if (!frameHandle) return;
  if (frameHandle.kind === "vfc" && video.cancelVideoFrameCallback) video.cancelVideoFrameCallback(frameHandle.id);
  if (frameHandle.kind === "raf") cancelAnimationFrame(frameHandle.id);
  frameHandle = null;
}

function setRunningUi(isRunning) {
  stage.classList.toggle("live", isRunning);
  recalBtn.disabled = !isRunning;
  if (stopBtn) stopBtn.disabled = !isRunning;
  if (snapBtn) snapBtn.disabled = !isRunning;
  startBtn.disabled = isRunning;
  startBtn.textContent = isRunning ? "Running" : "Start camera";
}

async function start() {
  startBtn.disabled = true;
  startBtn.textContent = "Starting...";
  document.body.classList.add("loading");
  // The emotion model loads alongside everything else and switches on when ready.
  emotion.load();
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
    smoothed = {};
    smoothedFace = {};
    lastFrameAt = null;
    lastVideoTime = -1;
    lastFaceSeen = Date.now();
    missedFaceFrames = 0;
    reactions.warm(["neutral", "happy"]);
    running = true;
    setRunningUi(true);
    setStatus("hold a relaxed face while it calibrates...");
    scheduleNext();
  } catch (err) {
    stopCamera();
    setRunningUi(false);
    setStatus(friendlyError(err));
    console.error(err);
  } finally {
    document.body.classList.remove("loading");
  }
}

function stop() {
  abortChallenge();
  running = false;
  cancelFrames();
  stopCamera();
  if (overlayCtx) overlayCtx.clearRect(0, 0, overlayCtx.canvas.width, overlayCtx.canvas.height);
  setRunningUi(false);
  setMoodText("");
  setStatus("camera off");
}

// ---------------------------------------------------------------------------
// Snapshot: your mirrored webcam frame next to the reaction, with the caption,
// saved as a PNG on your device. Nothing is uploaded.
// ---------------------------------------------------------------------------

function drawCover(ctx, source, sw, sh, dx, dy, dw, dh, mirror = false) {
  const scale = Math.max(dw / sw, dh / sh);
  const w = sw * scale, h = sh * scale;
  ctx.save();
  ctx.beginPath();
  ctx.rect(dx, dy, dw, dh);
  ctx.clip();
  if (mirror) {
    ctx.translate(dx + dw, dy);
    ctx.scale(-1, 1);
    ctx.drawImage(source, (dw - w) / 2, (dh - h) / 2, w, h);
  } else {
    ctx.drawImage(source, dx + (dw - w) / 2, dy + (dh - h) / 2, w, h);
  }
  ctx.restore();
}

function drawCaptionLine(ctx, text, cx, y, maxWidth, baseline) {
  if (!text) return;
  let size = 44;
  ctx.font = `800 ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  while (ctx.measureText(text).width > maxWidth && size > 18) {
    size -= 2;
    ctx.font = `800 ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  }
  ctx.textAlign = "center";
  ctx.textBaseline = baseline;
  ctx.lineJoin = "round";
  ctx.lineWidth = Math.max(4, size / 7);
  ctx.strokeStyle = "#000";
  ctx.fillStyle = "#fff";
  ctx.strokeText(text, cx, y);
  ctx.fillText(text, cx, y);
}

function takeSnapshot() {
  if (!running || !video.videoWidth) return;
  const W = 640, H = 480;
  const canvas = document.createElement("canvas");
  canvas.width = W * 2;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#1c1e28";
  ctx.fillRect(0, 0, W * 2, H);
  drawCover(ctx, video, video.videoWidth, video.videoHeight, 0, 0, W, H, true);

  const shown = showingPrimary ? reactionImg : reactionImgNext;
  let note = "";
  if (displayedEntry && displayedEntry.cors && shown.complete && shown.naturalWidth) {
    drawCover(ctx, shown, shown.naturalWidth, shown.naturalHeight, W, 0, W, H);
  } else if (displayedEntry) {
    note = "this photo's host does not allow saving it";
  }

  if (displayedCaption) {
    drawCaptionLine(ctx, displayedCaption.top, W + W / 2, 16, W - 32, "top");
    drawCaptionLine(ctx, displayedCaption.bottom, W + W / 2, H - 16, W - 32, "bottom");
  }
  ctx.font = "13px system-ui, sans-serif";
  ctx.textBaseline = "bottom";
  ctx.fillStyle = "rgba(255,255,255,0.85)";
  if (displayedEntry && displayedEntry.attribution) {
    ctx.textAlign = "right";
    ctx.fillText(displayedEntry.attribution, W * 2 - 10, H - 64);
  }
  if (note) {
    ctx.textAlign = "center";
    ctx.fillText(note, W + W / 2, H / 2);
  }

  try {
    canvas.toBlob((blob) => {
      if (!blob) { setStatus("could not save the snapshot"); return; }
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      const stamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
      a.href = url;
      a.download = `memecatz-${stamp}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
    }, "image/png");
  } catch (err) {
    console.warn("snapshot failed:", err.message);
    setStatus("could not save the snapshot");
  }
}

// ---------------------------------------------------------------------------
// Daily challenge. The rules live in challenge.js; this part drives them with
// the live mood and shows the prompts, reactions and results.
// ---------------------------------------------------------------------------

const modeFreeBtn = el("mode-free");
const modeDailyBtn = el("mode-daily");
const dailyPanel = el("daily");
const dailyInfo = el("daily-info");
const dailyStartBtn = el("daily-start");
const skipBtn = el("skip");
const chOverlay = el("ch-overlay");
const chRound = el("ch-round");
const chBig = el("ch-big");
const chHint = el("ch-hint");
const chBars = el("ch-bars");
const chTime = el("ch-time");
const chHold = el("ch-hold");
const resultsPanel = el("results");
const shareTextBox = el("share-text");
const shareStatus = el("share-status");
const shareFace = el("share-face");

const PASS_LINES = ["nailed it!", "purrfect!", "textbook!", "too easy!", "the cat approves!"];
const SITE_URL = `${location.origin}/`;

function moodWord(tag) {
  if (tag === "neutral") return "calm";
  return MOOD_WORDS[tag] || tag.replace(/_/g, " ");
}

function setText(node, text) {
  if (node && node.textContent !== text) node.textContent = text;
}

function challengeDate() {
  return DATE_OVERRIDE || utcDateString(new Date());
}

// Browser storage can be missing or blocked (private windows, strict
// settings). The challenge still works then, it just remembers nothing.
function loadStore() {
  try {
    return normalizeStore(JSON.parse(localStorage.getItem(STORE_KEY) || "null"));
  } catch (err) {
    return normalizeStore(null);
  }
}

function saveStore(store) {
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(store));
  } catch (err) {
    console.warn("could not save the daily result:", err && err.message ? err.message : err);
  }
}

function renderDailyInfo() {
  const date = challengeDate();
  const store = loadStore();
  const parts = [`Daily #${dailyNumber(date)}`];
  if (PROMPT_OVERRIDE) parts.push("test prompts, practice only");
  const streak = currentStreak(store, date);
  if (streak) parts.push(`streak ${streak}`);
  if (store.best) parts.push(`best ${store.best.passes}/${store.best.total} · ${formatSeconds(store.best.totalMs)} s`);
  const done = isOfficialDone(store, date);
  if (done) parts.push(`today ${store.official.passes}/${store.official.total} · ${formatSeconds(store.official.totalMs)} s`);
  setText(dailyInfo, parts.join("  ·  "));
  setText(dailyStartBtn, done || PROMPT_OVERRIDE ? "Play again (practice)" : "Play today's challenge");
}

function setMode(next) {
  if (next === mode) return;
  if (challenge) abortChallenge();
  mode = next;
  modeFreeBtn.setAttribute("aria-pressed", String(mode === "free"));
  modeDailyBtn.setAttribute("aria-pressed", String(mode === "daily"));
  dailyPanel.classList.toggle("hidden", mode !== "daily");
  if (mode === "daily") renderDailyInfo();
  else {
    resultsPanel.classList.add("hidden");
    lastMoodKey = null;
    lastSwitch = 0;
  }
}

function showOverlay({ round = "", big = "", hint = "", counting = false, bars = false, phase = "", prompt = "" }) {
  chOverlay.classList.remove("hidden");
  chOverlay.classList.toggle("counting", counting);
  chOverlay.dataset.phase = phase;
  chOverlay.dataset.prompt = prompt;
  setText(chRound, round);
  setText(chBig, big);
  setText(chHint, hint);
  chBars.classList.toggle("hidden", !bars);
}

function hideOverlay() {
  chOverlay.classList.add("hidden");
  chOverlay.dataset.phase = "";
  chOverlay.dataset.prompt = "";
}

function roundLabel(c) {
  return `round ${c.index + 1} of ${c.prompts.length}`;
}

async function startDaily() {
  if (challenge) return;
  const date = challengeDate();
  const prompts = PROMPT_OVERRIDE || dailyPrompts(date);
  challenge = {
    date,
    number: dailyNumber(date),
    prompts,
    practice: !!PROMPT_OVERRIDE || isOfficialDone(loadStore(), date),
    index: 0,
    phase: "waiting",
    phaseStart: performance.now(),
    timer: null,
    rounds: [],
    lastPickTry: 0,
  };
  resultsPanel.classList.add("hidden");
  dailyStartBtn.disabled = true;
  skipBtn.classList.remove("hidden");
  // Reactions for every target, fetched now so each pass shows one at once.
  reactions.warm(prompts);
  if (!displayedEntry) placeholder.textContent = "your reactions show up here";
  showOverlay({ round: `Daily #${challenge.number}`, big: "get ready", hint: "hold a relaxed face first", phase: "waiting" });
  if (!running) {
    await start();
    if (!running) abortChallenge();
  } else if (!calibrator.calibrating) {
    beginCountdown(performance.now());
  }
}

function abortChallenge() {
  if (!challenge) return;
  challenge = null;
  hideOverlay();
  skipBtn.classList.add("hidden");
  dailyStartBtn.disabled = false;
  lastMoodKey = null;
  lastSwitch = 0;
}

function beginCountdown(now) {
  challenge.phase = "countdown";
  challenge.phaseStart = now;
}

function beginRound(now, faceRanked, gestures, faceVisible) {
  const c = challenge;
  c.phase = "round";
  c.phaseStart = now;
  c.timer = new RoundTimer(c.prompts[c.index]);
  c.timer.update(now, roundSignal(c.prompts[c.index], faceRanked, gestures), faceVisible);
}

// A small, mirrored still of the camera at the end of a round. It stays in the
// page and is used only if "include my face" is ticked for the share image.
function captureFace() {
  if (!video.videoWidth) return null;
  const canvas = document.createElement("canvas");
  canvas.width = 480;
  canvas.height = 540;
  drawCover(canvas.getContext("2d"), video, video.videoWidth, video.videoHeight, 0, 0, 480, 540, true);
  return canvas;
}

function roundMessage(r) {
  if (r.passed) return `${PASS_LINES[Math.floor(Math.random() * PASS_LINES.length)]} ${formatSeconds(r.ms)} s`;
  if (r.skipped) return "skipped, on to the next one";
  if (!r.detected) return "so close! hold it a little longer";
  return `close one! you looked ${moodWord(r.detected)} instead`;
}

function finishRound(outcome, now) {
  const c = challenge;
  const round = {
    ...outcome,
    mood: outcome.passed ? outcome.prompt : (outcome.detected || "neutral"),
    message: "",
    entry: null,
    caption: null,
    face: captureFace(),
  };
  round.message = roundMessage(round);
  c.rounds.push(round);
  c.phase = "result";
  c.phaseStart = now;
  c.lastPickTry = 0;
  // The previous round's reaction must not stand in for this one.
  reactionImg.classList.remove("visible");
  reactionImgNext.classList.remove("visible");
  setText(capTop, "");
  setText(capBottom, "");
  if (attributionEl) attributionEl.textContent = "";
  placeholder.classList.remove("hidden");
  placeholder.textContent = "connecting for a real reaction...";
  showRoundReaction(round, now);
}

function showRoundReaction(round, now) {
  if (round.entry || now - challenge.lastPickTry < RETRY_PICK_MS) return;
  challenge.lastPickTry = now;
  const pick = reactions.pick([round.mood], currentEntry ? currentEntry.key : null);
  if (!pick) return;
  round.entry = pick;
  round.caption = captionFor(round.mood, currentCaption);
  currentEntry = pick;
  currentCaption = round.caption;
  swapReaction(pick, round.caption);
}

// faceRanked: smoothed face-only moods; gestures: the latest hand shapes.
function challengeTick(faceRanked, gestures, faceVisible) {
  const c = challenge;
  if (!c) return;
  const now = performance.now();
  const prompt = c.prompts[c.index];
  const info = PROMPTS[prompt];

  if (c.phase === "waiting") {
    if (calibrator.calibrating || !faceVisible) return;
    beginCountdown(now);
  }

  if (c.phase === "countdown") {
    const left = COUNTDOWN_MS - (now - c.phaseStart);
    if (left > 0) {
      showOverlay({ round: roundLabel(c), big: String(Math.ceil(left / 1000)), hint: `next: ${info.text}`, counting: true, phase: "countdown", prompt });
      return;
    }
    beginRound(now, faceRanked, gestures, faceVisible);
  }

  if (c.phase === "round") {
    const outcome = c.timer.update(now, roundSignal(prompt, faceRanked, gestures), faceVisible);
    showOverlay({
      round: roundLabel(c),
      big: info.text,
      hint: c.timer.paused ? "paused, can't see your face" : info.hint,
      bars: true,
      phase: "round",
      prompt,
    });
    chTime.style.width = `${(c.timer.remainingMs / ROUND_LIMIT_MS) * 100}%`;
    chHold.style.width = `${c.timer.holdProgress * 100}%`;
    if (outcome) finishRound(outcome, now);
    return;
  }

  if (c.phase === "result") {
    const round = c.rounds[c.rounds.length - 1];
    showOverlay({ round: roundLabel(c), big: round.passed ? "😺" : "⬛", hint: round.message, phase: "result", prompt });
    showRoundReaction(round, now);
    if (now - c.phaseStart < RESULT_MS) return;
    c.index += 1;
    if (c.index < c.prompts.length) beginCountdown(now);
    else finishChallenge();
  }
}

function skipRound() {
  const c = challenge;
  if (!c || (c.phase !== "round" && c.phase !== "countdown")) return;
  const now = performance.now();
  if (c.phase === "countdown") beginRound(now, null, [], false);
  finishRound(c.timer.skip(), now);
}

function finishChallenge() {
  const c = challenge;
  const score = scoreRun(c.rounds);
  const result = {
    date: c.date,
    number: c.number,
    passes: score.passes,
    total: score.total,
    totalMs: Math.round(score.totalMs),
    marks: c.rounds.map((r) => r.passed),
    prompts: c.prompts,
  };
  const recorded = recordResult(loadStore(), result, { practice: c.practice });
  if (recorded.official) saveStore(recorded.store);
  lastRun = { ...result, practice: !recorded.official, rounds: c.rounds };
  challenge = null;
  hideOverlay();
  skipBtn.classList.add("hidden");
  dailyStartBtn.disabled = false;
  renderDailyInfo();
  renderResults(lastRun);
}

function renderResults(run) {
  const flawless = run.passes === run.total;
  setText(el("res-title"), `MemeCatz Daily #${run.number}${run.practice ? " (practice)" : ""}`);
  setText(el("res-score"), `${run.passes}/${run.total} · ${formatSeconds(run.totalMs)} s`);
  const streak = currentStreak(loadStore(), run.date);
  const note = run.practice
    ? "practice run, your first run of the day is the one that counts"
    : `${flawless ? "flawless! " : ""}streak ${streak} ${streak === 1 ? "day" : "days"}, see you tomorrow`;
  setText(el("res-note"), note);

  const list = el("res-rounds");
  list.textContent = "";
  for (const r of run.rounds) {
    const li = document.createElement("li");
    li.append(`${r.passed ? "😺" : "⬛"} ${PROMPTS[r.prompt].text} `);
    const detail = document.createElement("span");
    detail.className = "res-detail";
    detail.textContent = r.passed ? `${formatSeconds(r.ms)} s` : r.message;
    li.append(detail);
    list.append(li);
  }
  shareTextBox.value = shareText(run, SITE_URL);
  setText(shareStatus, "");
  resultsPanel.classList.remove("hidden");
}

async function copyShare() {
  const text = shareTextBox.value;
  try {
    await navigator.clipboard.writeText(text);
    setText(shareStatus, "copied!");
  } catch (err) {
    shareTextBox.select();
    setText(shareStatus, "press Ctrl+C or Cmd+C to copy");
  }
}

function loadCorsImage(url) {
  return new Promise((resolve) => {
    const img = new Image();
    const timer = setTimeout(() => resolve(null), 8000);
    img.crossOrigin = "anonymous";
    img.onload = () => { clearTimeout(timer); resolve(img); };
    img.onerror = () => { clearTimeout(timer); resolve(null); };
    img.src = url;
  });
}

function drawText(ctx, text, x, y, size, color, weight = 700, align = "center", maxWidth = Infinity) {
  ctx.font = `${weight} ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  while (ctx.measureText(text).width > maxWidth && size > 12) {
    size -= 1;
    ctx.font = `${weight} ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  }
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = "middle";
  ctx.fillText(text, x, y);
}

// 1080x1080 PNG: score, the best round's reaction (only images whose host
// allows drawing them, so not The Cat API photos), and each round.
async function saveShareImage() {
  if (!lastRun) return;
  const run = lastRun;
  setText(shareStatus, "making your image...");
  const S = 1080;
  const canvas = document.createElement("canvas");
  canvas.width = S;
  canvas.height = S;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#12131a";
  ctx.fillRect(0, 0, S, S);

  drawText(ctx, `MemeCatz Daily #${run.number}${run.practice ? " (practice)" : ""}`, S / 2, 80, 54, "#f2f3f7", 800);
  drawText(ctx, `${run.passes}/${run.total} · ${formatSeconds(run.totalMs)} s`, S / 2, 170, 84, "#78e08f", 800);

  const idx = bestRoundIndex(run.rounds, (r) => r.entry && r.entry.cors);
  const img = idx >= 0 ? await loadCorsImage(run.rounds[idx].entry.url) : null;
  const withFace = shareFace.checked;
  const faceRound = idx >= 0 ? run.rounds[idx] : run.rounds[bestRoundIndex(run.rounds)];
  const face = withFace && faceRound ? faceRound.face : null;
  const top = 250, h = 540;
  const box = face ? { x: 550, w: 470 } : { x: 60, w: 960 };
  ctx.fillStyle = "#1c1e28";
  ctx.fillRect(60, top, 960, h);
  if (face) drawCover(ctx, face, face.width, face.height, 60, top, 470, h);
  if (img) {
    drawCover(ctx, img, img.naturalWidth, img.naturalHeight, box.x, top, box.w, h);
    const r = run.rounds[idx];
    if (r.caption) {
      drawCaptionLine(ctx, r.caption.top, box.x + box.w / 2, top + 16, box.w - 32, "top");
      drawCaptionLine(ctx, r.caption.bottom, box.x + box.w / 2, top + h - 16, box.w - 32, "bottom");
    }
    if (r.entry.attribution) drawText(ctx, r.entry.attribution, box.x + box.w - 12, top + h + 24, 22, "#9aa0b4", 600, "right");
  } else {
    drawText(ctx, "no reaction image could be saved this time", box.x + box.w / 2, top + h / 2, 26, "#9aa0b4", 500);
  }

  const n = run.rounds.length;
  const tileW = 180, gap = (960 - n * tileW) / Math.max(1, n - 1);
  run.rounds.forEach((r, i) => {
    const x = 60 + i * (tileW + gap);
    const y = 840;
    ctx.fillStyle = r.passed ? "#78e08f" : "#2e3142";
    ctx.fillRect(x, y, tileW, 140);
    const color = r.passed ? "#0f1a12" : "#f2f3f7";
    drawText(ctx, PROMPTS[r.prompt].text, x + tileW / 2, y + 48, 26, color, 700, "center", tileW - 16);
    drawText(ctx, r.passed ? `${formatSeconds(r.ms)} s` : r.skipped ? "skipped" : "missed", x + tileW / 2, y + 96, 30, color, 800);
  });
  drawText(ctx, SITE_URL.replace(/^https?:\/\//, "").replace(/\/$/, ""), S / 2, 1035, 26, "#9aa0b4", 500);

  try {
    canvas.toBlob((blob) => {
      if (!blob) { setText(shareStatus, "could not make the image"); return; }
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `memecatz-daily-${run.number}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      setText(shareStatus, "image saved");
    }, "image/png");
  } catch (err) {
    console.warn("share image failed:", err.message);
    setText(shareStatus, "could not make the image");
  }
}

modeFreeBtn.addEventListener("click", () => setMode("free"));
modeDailyBtn.addEventListener("click", () => setMode("daily"));
dailyStartBtn.addEventListener("click", startDaily);
skipBtn.addEventListener("click", skipRound);
el("copy-share").addEventListener("click", copyShare);
el("save-share").addEventListener("click", saveShareImage);

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------

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
  if (debugOn && debugPanel) {
    debugPanel.classList.remove("hidden");
    debugBtn.textContent = "Hide detail";
  }
  debugBtn.addEventListener("click", () => {
    debugOn = !debugOn;
    if (debugPanel) debugPanel.classList.toggle("hidden", !debugOn);
    debugBtn.textContent = debugOn ? "Hide detail" : "Show detail";
  });
}

function recalibrate() {
  if (!running) return;
  calibrator.startCalibration();
  smoothed = {};
  smoothedFace = {};
  setStatus("recalibrating, hold a relaxed face");
}

startBtn.addEventListener("click", start);
if (stopBtn) stopBtn.addEventListener("click", stop);
recalBtn.addEventListener("click", recalibrate);
if (snapBtn) snapBtn.addEventListener("click", takeSnapshot);

// Keyboard shortcuts: c recalibrates, s saves a snapshot. Ignored while typing
// in a field or using a slider or menu.
document.addEventListener("keydown", (e) => {
  if (e.metaKey || e.ctrlKey || e.altKey || e.repeat) return;
  const tag = e.target && e.target.tagName;
  if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") return;
  if (e.key === "c") recalibrate();
  if (e.key === "s" && running) takeSnapshot();
});

if (animalSel)
  animalSel.addEventListener("change", () => {
    reactions.setAnimal(animalSel.value);
    reactions.warm(challenge ? challenge.prompts : ["neutral", "happy"]);
    lastMoodKey = null;
    lastRotate = 0;
  });

// Free the camera if the tab is closed or navigated away.
window.addEventListener("pagehide", () => { if (running) stop(); });

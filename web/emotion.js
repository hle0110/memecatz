// The 7-class emotion model from the desktop app, running in the browser
// through LiteRT.js. Same .tflite file, same crop rules, same 7 outputs.
//
// The face box comes from MediaPipe's landmarks instead of the desktop app's
// Haar cascade. The square, offset, and padding rules below are applied to it
// exactly as vision.py applies them to the cascade box. The final resize is
// done by the browser rather than cv2.resize, so individual pixels can differ
// slightly from the desktop app.

const LITERT_VERSION = "2.5.3";
// jsDelivr's +esm build rewrites the package's bare "@litertjs/wasm-utils"
// import into a URL. The plain dist/index.js cannot load in a browser without
// a bundler, which is why the emotion model never ran before.
const LITERT_MODULE = `https://cdn.jsdelivr.net/npm/@litertjs/core@${LITERT_VERSION}/+esm`;
const LITERT_WASM = `https://cdn.jsdelivr.net/npm/@litertjs/core@${LITERT_VERSION}/wasm/`;
const MODEL_URL = "./emotion_model_quantized.tflite";

export const EMOTION_LABELS = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"];

// Mirrors vision.py.
export const EMOTION_PADDING = 40;
export const EMOTION_OFFSET = 10;
export const TARGET = 64;

// Pure helpers, kept separate so they can be checked in Node without a browser.

export function toSquare(box) {
  let [x, y, w, h] = box;
  if (h > w) {
    const diff = h - w;
    x -= Math.floor(diff / 2);
    w += diff;
  } else if (w > h) {
    const diff = w - h;
    y -= Math.floor(diff / 2);
    h += diff;
  }
  return [Math.trunc(x), Math.trunc(y), Math.trunc(w), Math.trunc(h)];
}

// The crop rectangle in padded-frame coordinates, as vision.py computes it.
// numpy slicing silently stops at the edge of the padded frame, so the far
// edges are clipped to it here too. That can make the crop non-square, which
// cv2.resize then stretches, and the browser does the same.
export function cropRect(squareBox, frameWidth, frameHeight) {
  const [x, y, w, h] = squareBox;
  const paddedW = frameWidth + 2 * EMOTION_PADDING;
  const paddedH = frameHeight + 2 * EMOTION_PADDING;
  const x1 = Math.max(0, x - EMOTION_OFFSET + EMOTION_PADDING);
  const y1 = Math.max(0, y - EMOTION_OFFSET + EMOTION_PADDING);
  const x2 = Math.min(paddedW, x + w + EMOTION_OFFSET + EMOTION_PADDING);
  const y2 = Math.min(paddedH, y + h + EMOTION_OFFSET + EMOTION_PADDING);
  return [x1, y1, x2, y2];
}

// cv2.COLOR_BGR2GRAY, reproduced exactly. OpenCV does this in fixed point
// with the 0.299 / 0.587 / 0.114 weights scaled by 2^15, summed as integers,
// then rounded by adding half and shifting. Checked against cv2 on all 16.7
// million possible pixels (OpenCV 4.11 and 4.14 on x86).
export function grayOf(r, g, b) {
  return (r * 9798 + g * 19235 + b * 3735 + 16384) >> 15;
}

// (pixel / 255 - 0.5) * 2, the model's expected [-1, 1] range.
export function normalize(gray) {
  return (gray / 255.0 - 0.5) * 2.0;
}

// rgba pixel buffer of TARGET x TARGET -> Float32Array of TARGET*TARGET in [-1,1]
export function rgbaToModelInput(rgba) {
  const n = TARGET * TARGET;
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    out[i] = normalize(grayOf(rgba[o], rgba[o + 1], rgba[o + 2]));
  }
  return out;
}

// Mean gray value of an rgba buffer, rounded the way cv2 stores a float
// border value into an 8-bit image.
export function meanGray(rgba) {
  let sum = 0;
  const n = rgba.length / 4;
  for (let i = 0; i < n; i++) {
    const o = i * 4;
    sum += grayOf(rgba[o], rgba[o + 1], rgba[o + 2]);
  }
  return n ? Math.round(sum / n) : 128;
}

// Landmark list (normalized x,y) -> pixel box [x, y, w, h].
export function boxFromLandmarks(landmarks, width, height) {
  let minX = 1, minY = 1, maxX = 0, maxY = 0;
  for (const p of landmarks) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  const x = Math.floor(minX * width);
  const y = Math.floor(minY * height);
  const w = Math.ceil((maxX - minX) * width);
  const h = Math.ceil((maxY - minY) * height);
  return [x, y, w, h];
}

export class EmotionDetector {
  constructor() {
    this.model = null;
    this.ready = false;
    this.failed = false;
    this.loading = null;
    this.canvas = null;
    this.ctx = null;
    this.rowCanvas = null;
    this.rowCtx = null;
    this.Tensor = null;
  }

  // Safe to call more than once; later calls share the first load.
  load() {
    if (!this.loading) this.loading = this._load();
    return this.loading;
  }

  async _load() {
    try {
      const lib = await import(LITERT_MODULE);
      await lib.loadLiteRt(LITERT_WASM);
      this.model = await lib.loadAndCompile(MODEL_URL, { accelerator: "wasm" });
      this.Tensor = lib.Tensor;

      const inputs = this.model.getInputDetails ? this.model.getInputDetails() : null;
      if (inputs && inputs[0] && inputs[0].shape) {
        const s = Array.from(inputs[0].shape).join("x");
        if (s !== `1x${TARGET}x${TARGET}x1`) throw new Error(`unexpected model input shape ${s}`);
      }

      this.canvas = document.createElement("canvas");
      this.canvas.width = TARGET;
      this.canvas.height = TARGET;
      this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
      this.rowCanvas = document.createElement("canvas");
      this.rowCtx = this.rowCanvas.getContext("2d", { willReadFrequently: true });
      this.ready = true;
    } catch (err) {
      this.failed = true;
      console.warn("emotion model unavailable, continuing on face signals only:", err && err.message ? err.message : err);
    }
  }

  // vision.py pads the frame with the mean gray of its bottom two rows.
  _padValue(video, vw, vh) {
    if (this.rowCanvas.width !== vw) this.rowCanvas.width = vw;
    if (this.rowCanvas.height !== 2) this.rowCanvas.height = 2;
    this.rowCtx.drawImage(video, 0, vh - 2, vw, 2, 0, 0, vw, 2);
    return meanGray(this.rowCtx.getImageData(0, 0, vw, 2).data);
  }

  // Returns {label: score} for the 7 classes, or null if not ready / no face.
  async detect(video, landmarks) {
    if (!this.ready || !landmarks || !landmarks.length) return null;
    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (!vw || !vh) return null;

    const square = toSquare(boxFromLandmarks(landmarks, vw, vh));
    const [px1, py1, px2, py2] = cropRect(square, vw, vh);
    const sw = px2 - px1;
    const sh = py2 - py1;
    if (sw <= 0 || sh <= 0) return null;

    // Crop origin in video coordinates (may be negative, inside the padding).
    const sx = px1 - EMOTION_PADDING;
    const sy = py1 - EMOTION_PADDING;

    const ctx = this.ctx;
    const pad = this._padValue(video, vw, vh);
    ctx.fillStyle = `rgb(${pad},${pad},${pad})`;
    ctx.fillRect(0, 0, TARGET, TARGET);

    // Draw the part of the crop that lies inside the real frame on top. The
    // pixel data is never mirrored; only the on-screen video is, in CSS.
    const vx1 = Math.max(0, sx), vy1 = Math.max(0, sy);
    const vx2 = Math.min(vw, sx + sw), vy2 = Math.min(vh, sy + sh);
    if (vx2 > vx1 && vy2 > vy1) {
      const dx = ((vx1 - sx) / sw) * TARGET;
      const dy = ((vy1 - sy) / sh) * TARGET;
      const dw = ((vx2 - vx1) / sw) * TARGET;
      const dh = ((vy2 - vy1) / sh) * TARGET;
      ctx.drawImage(video, vx1, vy1, vx2 - vx1, vy2 - vy1, dx, dy, dw, dh);
    }

    const input = rgbaToModelInput(ctx.getImageData(0, 0, TARGET, TARGET).data);

    let tensor = null, outputs = null;
    try {
      tensor = new this.Tensor(input, [1, TARGET, TARGET, 1]);
      outputs = await this.model.run(tensor);
      const out = Array.isArray(outputs) ? outputs[0] : outputs;
      const data = await out.data();
      const scores = {};
      for (let i = 0; i < EMOTION_LABELS.length; i++) scores[EMOTION_LABELS[i]] = data[i];
      return scores;
    } catch (err) {
      console.warn("emotion inference skipped:", err && err.message ? err.message : err);
      return null;
    } finally {
      try { tensor && tensor.delete && tensor.delete(); } catch (_) {}
      try {
        const arr = Array.isArray(outputs) ? outputs : outputs ? [outputs] : [];
        for (const o of arr) o.delete && o.delete();
      } catch (_) {}
    }
  }
}

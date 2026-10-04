# Architecture

Both apps run the same pipeline: read signals from the camera, combine them into a mood, pick a real reaction for that mood, put a caption on it.

```
camera frame
  ├─ face landmarks + 52 blendshapes ─> calibration and baseline ─> expression strengths ─┐
  ├─ 7-class emotion model on the face crop ──────────────────────────────────────────────┤
  ├─ hand landmarks ─> gesture (thumbs up, peace, open palm, fist, pointing) ─────────────┼─> mood.combine ─> smoothing ─> top moods
  └─ desktop only, optional: OpenAI vision tags (needs a key) ────────────────────────────┘
top moods ─> reaction (Giphy phrase table, else backup photo) + caption ─> screen
```

## Files

| Desktop | Web | Role |
|---|---|---|
| `app.py` | `web/app.js` | Main loop, UI, timing, snapshots |
| `vision.py` | `web/face.js`, `web/emotion.js` | Face, emotion model, hands, gestures |
| `mood.py` | `web/mood.js` | Combines signals into a mood vector; bit-identical |
| `reactions.py` | `web/reactions.js` | Giphy phrase table, backup photos, caption rendering (desktop) |
| `captions.py` | `web/captions.js` | Static caption bank (22 moods, 6 each), identical; desktop also has optional live captions |
| | `web/challenge.js` | Daily challenge rules: prompts per date, round timer, scoring, saved results, share text (web only) |
| `identity.py` | | Optional saved calibration profiles (`--save-profile`, needs face_recognition) |
| `run.py`, `run.sh`, `run.bat` | | Launcher: private `.venv`, installs requirements, repairs a broken venv |
| `check_setup.py` | | Checks every component; `--no-webcam` for CI; exits 1 on failure |
| | `worker.js`, `wrangler.toml`, `web/_headers` | Cloudflare Worker, routing, security headers |

`assets/` holds the emotion model (`emotion_model_quantized.tflite`, also copied in `web/`), the caption font, and licenses.

## Signals

**Face.** MediaPipe FaceLandmarker (`face_landmarker/float16/1`) gives 52 blendshapes. The first 12 samples over at least 1.8 seconds become the person's neutral baseline (median). After that, blendshapes are read as differences from the baseline, and the baseline slowly drifts toward the current face while the face is near neutral (alpha 0.01 per 0.15 s, only when the average difference is under 0.07). The desktop turns differences into on/off tags with thresholds; the web turns them into continuous 0 to 1 strengths with a small jitter deadzone and a smoothstep, scaled by the Sensitivity slider (default 1.3).

**Emotion model.** mini_XCEPTION, 64x64 grayscale, 7 classes (angry, disgust, fear, happy, sad, surprise, neutral). The desktop finds the face with a Haar cascade; the web uses the face landmark box. Both then make the box square, add a 10 px offset, pad the frame by 40 px with the mean gray of its bottom two rows, crop, resize, convert to gray, and scale to [-1, 1]. The web gray conversion reproduces OpenCV exactly: `(r*9798 + g*19235 + b*3735 + 16384) >> 15`, checked on all 16.7 million pixels. The resize itself differs slightly (browser vs cv2). On web it runs every 100 ms with a guard against overlapping runs, through LiteRT.js loaded from `https://cdn.jsdelivr.net/npm/@litertjs/core@2.5.3/+esm`.

**Hands.** MediaPipe HandLandmarker (`hand_landmarker/float16/1`), every 100 ms on web, every detection step on desktop. Gesture rules are identical on both and computed in pixels (MediaPipe's normalized x and y would squash one axis on a 4:3 camera). A finger is extended when its tip is farther from the wrist than its PIP joint times 1.08; the thumb when its tip is farther than its MCP times 1.25. Thumbs up or down needs only the thumb out, with the tip 15 px above or below the palm center. Open palm is all five extended, fist is none, peace is index and middle, pointing is index only.

| Gesture | Tags |
|---|---|
| thumbs_up | approval 1.0, happy 0.5 |
| thumbs_down | disapproval 1.0, sad 0.4 |
| open_palm | stop 1.0, surprise 0.5 |
| fist | angry 0.7, determined 0.6 |
| peace | chill 1.0, happy 0.6 |
| pointing | suspicious 0.6, focused 0.5 |

## Mood

`combine()` weights the sources (gestures by 1.2) into one vector. The web smooths it with alpha 0.6 per 1/30 s, scaled by real elapsed time; the desktop uses 0.35 per 0.15 s step. The top three moods above a floor (web 0.06, desktop 0.12) are ranked. A new reaction is picked when the top mood changes (after a 1.8 s cooldown) or every 7 s while it stays the same. Moods reachable on web: the 7 emotion classes plus smug, confused, mischief, annoyed, approval, disapproval, chill, suspicious, stop, determined, focused. Triumph, anxious, bored and mocking come only from the desktop's optional vision tags.

## Reactions

Giphy first, mood matched, using the phrase table in `docs/REACTIONS.md`. Backup when Giphy is not configured, rejected the key, or is failing: The Cat API (random cats) or Dog CEO (random photos from a list of 26 pet breeds, since its any-breed endpoint includes wild dogs). Never both at once while Giphy works. Web shows Giphy's animated WebP in an `<img>`; desktop downloads the GIF into memory, decodes up to 48 frames, and plays it with the caption drawn on each frame. Giphy content always carries "Powered By GIPHY".

## Web specifics

`web/app.js` processes each new camera frame (`requestVideoFrameCallback`, with an animation-frame fallback). The face runs every frame; hands and emotion are throttled. MediaPipe tries the GPU delegate and falls back to CPU; `?cpu=1` forces CPU. Other URL options: `?debug=1` opens the detail panel. Keys: c recalibrates, s saves a snapshot. The snapshot is a 1280x480 PNG of the mirrored camera and the reaction with its caption, made in the page and downloaded; nothing is uploaded. Captions and image change together when the new image has loaded.

## Daily challenge (web only)

A toggle above the camera switches between Free play (everything above) and Daily challenge. The desktop app has no challenge mode.

**Prompts.** Everyone gets the same five prompts on the same UTC date: three of the face prompts happy, surprise, angry, sad, neutral ("Poker face!") and two of the gesture prompts approval (thumbs up), chill (peace), stop (open palm), suspicious (pointing), with no repeats. The date string seeds an FNV-1a hash and a mulberry32 generator, which shuffle each list, take three and two, and shuffle the five together. Fist is not a prompt. Confused is left out of this first version because on the web it can only come from a pucker, which makes it very hard to reach. The daily number is the count of days since `LAUNCH_DATE` in `web/challenge.js`, with that day as #1.

**Rounds.** Calibration runs first, as in free play. Each round is a 3, 2, 1 countdown that names the next prompt, then the prompt in large text with a 6 second clock. Each kind of round reads its own signal, so neither can be passed with the other. A face round uses a face-only mood (`faceOnlyVector`: the emotion model and the expression strengths, no gesture tags), smoothed like the free play mood, and passes when the target is the top ranked face-only mood with a score of at least 0.35. A gesture round passes when the latest hand check classified the matching shape (approval thumbs_up, chill peace, stop open_palm, suspicious pointing); the mood vector plays no part. Either way the condition has to hold for 0.8 seconds without a break. The clock and the hold both stop while the face is not visible (or during recalibration), and a lost face restarts the hold. A stalled frame counts as at most 1 second. Skip ends the round as a miss. A pass shows a reaction for the target mood; a miss shows one for what was shown longest during the round ("you looked happy instead"): the top face-only mood, or in a gesture round the mood of a different hand shape if one was made (a peace sign in a thumbs up round reads as chill). If only the target was seen and was too weak, it shows neutral. Reactions come from the same `ReactionSource.pick`, and the five target moods are fetched as soon as the run starts. Free play reactions pause during a run.

**Score and storage.** The score is passes out of five, then total time, where a miss counts as 6 seconds. The first finished run of a date is the official one; later runs that day are practice. `localStorage` key `memecatz.daily` keeps the day's official result, the best official result (more passes, then less time), and the streak (consecutive days with an official run). Every read and write is wrapped so blocked storage only means nothing is remembered.

**Sharing.** The results panel lists each round and gives the text `MemeCatz Daily #N P/5 · T s`, a row of 😺 (pass) and ⬛ (miss), and the site address, with a Copy button. Save image makes a 1080x1080 PNG in the page with the score, the best round's reaction with its caption, and a tile per round. Only reactions that can be drawn into a canvas are used, so The Cat API photos are skipped. "include my face" (off by default) adds a still of the camera taken at the end of that round; these stills stay in page memory and are never stored or uploaded.

**Testing hooks.** `?prompts=happy,approval,chill,suspicious,angry` replaces the day's prompts (unknown names or repeats are ignored, and such runs are always practice). `?date=YYYY-MM-DD` sets the day, for the prompts, the number and the streak.

## Worker and hosting

`wrangler.toml`: `main = "worker.js"`, assets from `./web` bound as `ASSETS`, `not_found_handling = "404-page"`. Cloudflare serves any file that exists in `web/` directly, with headers from `web/_headers` (CSP, Permissions-Policy camera=self, Referrer-Policy, nosniff). Only other paths reach `worker.js`, which answers `GET /api/config` with `{"giphyKey": ...}` (from the secret `GIPHY_API_KEY`, or null) and `no-store`, returns 404 for other `/api/*`, and passes everything else to the assets (the 404 page). The CSP allows scripts from `self` and `cdn.jsdelivr.net` with `'wasm-unsafe-eval'`, connections to jsDelivr, `storage.googleapis.com` (models), `api.giphy.com`, `api.thecatapi.com`, `dog.ceo`, and images from Giphy, The Cat API's S3 host, and `images.dog.ceo`. A new external host must be added in both `worker.js` and `web/_headers`.

## Desktop specifics

Python 3.9 to 3.14. Requirements: `opencv-contrib-python`, `mediapipe>=0.10.21,<0.11` (0.10.30 and newer install on any Python 3; Intel Macs only have 0.10.21, so Python 3.12 or older there), `numpy`, `ai-edge-litert` (tensorflow on Python 3.9 and Intel Macs), `Pillow`, `requests`, `openai<2`. Detection runs on a worker thread every 0.15 s; reaction switching (download, caption, render) runs on its own thread so detection never pauses. Optional keys: `GIPHY_API_KEY` (or `--giphy-key`), `OPENAI_API_KEY` (or `--openai-key`) for live captions and the vision mood boost, both with 10 s timeouts and run in the background. Flags: `--animal dog`, `--save-profile`, `--refresh-dataset`. Models download once into `model_cache/`; backup photos cache in `reaction_cache/`; Giphy content is never written to disk.

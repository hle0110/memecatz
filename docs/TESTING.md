# Testing

Run the fast checks after every change, the smoke tests after anything that touches the camera, models, reactions, or the page, and say plainly what was not tested.

## Automated tests

| Command | Needs | Covers |
|---|---|---|
| `python tests/test_mood.py` | Python only | Mood engine |
| `node --test tests/web_reactions.test.mjs` | Node 22+ | Web reaction logic with a fake fetch: Giphy order, WebP, attribution, no proxy, phrase rotation and depth limits, dog phrases, missing key, rejected key, rate limit fallback, config network errors |
| `node --test tests/web_challenge.test.mjs` | Node 22+ | Daily challenge: same date same prompts, different dates differ, three face and two gesture prompts with no repeats, daily number, override parsing, hold timer pass, threshold, broken hold, face lost pause, frame stalls, skip, a fist not passing "Look angry!", a smile not passing any gesture round, gesture rounds passing on the hand shape, poker face, scoring and tie break, share text, official versus practice, best, streak, damaged saved data, best round for the share image |
| `python tests/test_desktop.py` | requirements | GIF decoding and timing, caption rendering, animation frame timing, caption bank, vision tags clearing and expiring, non-blocking vision calls, damaged profiles file, Giphy phrase rotation |
| `python tests/test_web_parity.py` | requirements, Node 22+ | Web and desktop identical: captions, phrase table, dog breeds, gesture and expression maps, blendshape names, mood math (400 random cases), gesture rules (600 random hands), emotion square and crop, gray conversion vs cv2, time-scaled smoothing, CSP in `worker.js` equals `web/_headers`, every host the page calls is allowed by the CSP |
| `python check_setup.py --no-webcam` | requirements | Every desktop component loads and runs on a blank frame; exits 1 on any failure |
| `for f in web/*.js worker.js tests/*.mjs; do node --check "$f"; done` | Node | Syntax |

CI runs all of these (`.github/workflows/ci.yml`): a quick job, then a full job on Python 3.10, 3.12 and 3.14 on Ubuntu, which first installs `libegl1` and `libgles2` (MediaPipe's native library links against them and Ubuntu runners do not have them).

## Smoke tests with a fake camera

These run the real apps end to end with a recorded "webcam". Build the test video once (needs ffmpeg):

```
python dev/make_test_media.py
```

It downloads a smiling face photo (from the MIT `fer` project) and MediaPipe's sample hand photos, and writes `dev/output/media/cam.y4m` and `cam.mp4`: face for 8 s, then thumbs up, peace, pointing and fist, each with the face, separated by face-only gaps. Since the face is already smiling during calibration, expect the emotion model to read happy (about 0.93) while the expression strengths stay near zero.

**Web.** Serve the site with `npx wrangler dev` (real Worker, `_headers`, and 404 handling), then:

```
pip install playwright && python -m playwright install chromium
python dev/browser_smoke.py --mock-giphy --snapshot
python dev/browser_smoke.py --giphy-key YOUR_KEY --animal dog
```

It prints the status line, mood, caption, emotion scores and gestures every 3 seconds, the Giphy searches made, and any page errors. Expect "emotion model on", happy around 0.93, then approval, chill, suspicious and angry as the gestures appear. Runs use `?cpu=1` because headless Chromium has no GPU (the GPU path works but is very slow in software). WebGPU and NPU warnings from LiteRT in headless runs are expected. `--mock-giphy` fakes Giphy's search answers but loads real Giphy media; `--giphy-key` uses real searches and costs API calls.

**Daily challenge.** The looping video cannot line up with rounds, which start whenever the models finish loading, so `--challenge` swaps the camera for a canvas that shows the still frame matching each round's prompt (the smiling face, thumbs up, peace, pointing) and the plain face between rounds:

```
python dev/browser_smoke.py --mock-giphy --challenge
python dev/browser_smoke.py --mock-giphy --challenge --prompts none --date 2026-10-06 --runs 2 --reload --face
```

The first uses `?prompts=happy,approval,chill,suspicious,angry`. Expect the first four to pass in about 0.9 s and angry to miss with "you looked happy instead", since there is no angry frame. The second plays the real prompts for that date twice: the first run is official and is saved, the second is practice and changes nothing, and after the reload the panel shows the streak, best and today's result. Each run saves the share image to `dev/output/daily_share.png` (or `daily_share_face.png`). `--camera video` plays the challenge against the looping video instead; gesture rounds then pass only when the video happens to show that gesture.

**Desktop.** With the app's environment (for example `.venv/bin/python`):

```
python dev/desktop_smoke.py
python dev/desktop_smoke.py --mock-giphy
```

The camera is replaced by `cam.mp4` and the window by a recorder; it presses c at 12 s, s at 16 s and q at 26 s, prints the Giphy searches, and saves frames and the snapshot under `dev/output/desktop/` to look at. Expect about 14 fps in the sandbox.

**Giphy phrases.** See `docs/REACTIONS.md` for `dev/giphy_audit.py`.

## Other useful checks

Live site after a deploy: compare served files with the repo (`curl -s https://memecatz.lhieu020304.workers.dev/app.js | cmp - web/app.js`), check `/api/config` returns the key, `/nope` returns the 404 page, and response headers include `content-security-policy`.

Python versions: `uv venv -p 3.14 /tmp/env && VIRTUAL_ENV=/tmp/env uv pip install -r requirements.txt` gives a clean install to test against. Resolution on other platforms can be checked without installing: `uv pip compile --python-version 3.12 --python-platform x86_64-apple-darwin requirements.txt`.

Firefox can be driven by Playwright too, with `firefox_user_prefs={"media.navigator.streams.fake": True, "media.navigator.permission.disabled": True}`, but its fake camera has no face, so it only proves loading and the security policy.

## Never tested

Real webcam and real face, phones, Safari, Windows, Intel Macs, saved profiles with a real face_recognition install, real OpenAI calls.

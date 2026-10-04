# Status

Snapshot as of October 4, 2026. Check `git log --oneline -5` first, since anything below about pushes may be out of date.

## Where things stand

`main` is at `a30c9bf` (commit message "33"), which added the hand-picked reaction phrase table (`docs/REACTIONS.md`) on top of `44a45bf`. All four CI jobs are green on it, and the live site serves exactly the files in `web/` from that commit, with security headers, the 404 page, the emotion model running, and Giphy enabled.

Cloudflare settings: the secret `GIPHY_API_KEY` is set and confirmed working (the site's `/api/config` returns it). The build variable `SKIP_DEPENDENCY_INSTALL` = `1` was being added under Settings, Builds; it is not yet confirmed. Without it, every build installs the desktop app's Python packages for nothing.

## Daily challenge

Web only, launched October 4, 2026 (`LAUNCH_DATE`, Daily #1), in `web/challenge.js`, wired into `web/app.js`, see `docs/ARCHITECTURE.md`. Face rounds score from the face alone and gesture rounds from the classified hand shape, so a fist cannot pass "Look angry!" and a smile cannot pass a hand round. Best result counts official runs only. Tested with the Node tests and the headless smoke runs in `docs/TESTING.md`: gesture and smile rounds pass, misses name the detected mood, official versus practice, reload keeps the streak, Skip, switching mode or pressing Stop mid run, blocked storage, the share image with and without the face, and The Cat API photos left out of the image. Not tested: any of it with a real face (in particular whether surprise, angry, sad and poker face are comfortable to reach within 6 seconds), the Copy button against a real clipboard, phones and Safari.

## Never tested

A real webcam with a real face, real phones, Safari, Windows, and Intel Macs. Everything else was tested with a fake camera video (see `docs/TESTING.md`). Firefox was checked only for loading and the security policy, because its fake camera has no face.

## Known limits

1. **Thin moods.** Giphy has few real, on-mood results for some moods, so these repeat sooner: stop (about 10 cat, 8 dog), cat approval (3 plus "cat smiling"), cat triumph (3 plus "cat smiling"), dog sad (8), dog suspicious (2 plus "dog side eye"), dog annoyed (10).
2. **Giphy results drift.** The phrase table was checked by eye in October 2026. Re-run `dev/giphy_audit.py` every few months.
3. **Giphy beta key** allows 100 calls an hour for everyone. A public launch needs the production key (apply on the Giphy developer dashboard; it has a fee).
4. **Cat photo snapshots.** The Cat API's image host sends no CORS header, so a snapshot with a backup cat photo saves the caption and a note instead of the photo. Giphy and Dog CEO images work.
5. **MediaPipe telemetry** tries to reach a Google logging host; the security policy blocks it, which leaves one console error per visit. Harmless, and better for privacy.
6. **Emotion model license.** The 7-class model (mini_XCEPTION from the MIT `fer` package) was trained on FER2013. Check that dataset's terms before any commercial use.

## History (what was fixed and why)

| Area | Change |
|---|---|
| Desktop crash | Fresh installs pulled mediapipe 0.10.35, which removed `mp.solutions`; the app crashed at launch. Face and hands moved to the tasks API. Python 3.13 and 3.14 now work. |
| Intel Mac | Could not install at all (no ai-edge-litert wheel). Now uses tensorflow and mediapipe 0.10.21 there, Python 3.12 or older. |
| Web emotion model | Was never loaded, and LiteRT could not load anyway (bare import). Fixed with `emotion.load()` and the jsDelivr `+esm` build. |
| Parity | Gesture rules were not a true port (thresholds, deadzone, aspect ratio); gray conversion differed from cv2. Both now exact and tested. |
| Frame rate | Calibration, drift and smoothing were per frame on web; now time based. |
| Giphy | The old worker proxied and cached Giphy, which Giphy forbids. Now the browser calls Giphy directly; desktop keeps Giphy media in memory only and plays it animated. |
| Precision | One generic phrase per mood returned cartoons, people and generated cats. Replaced by a checked table of phrases with depth limits (`docs/REACTIONS.md`). |
| Site | Security headers, 404 page, favicon, privacy and how-it-works text, Stop and Save snapshot buttons, "can't see your face" message, accessibility basics. |
| Desktop | Opt-in saved profiles (`--save-profile`), clean snapshots, background reaction switching, OpenAI timeouts and non-blocking vision calls, a launcher that repairs a broken `.venv`. |
| CI | Quick job plus full installs on Python 3.10, 3.12, 3.14; Linux needs `libegl1 libgles2`. |

## Backlog (ideas, none started)

| Idea | Notes |
|---|---|
| Custom domain | Workers Custom Domains need the domain's DNS on Cloudflare. Cloudflare Registrar is simplest; a domain bought at IONOS works after moving its nameservers to Cloudflare. |
| Giphy production key | Needed before sharing the site widely. |
| Clip recording | MediaRecorder on a composite canvas; same CORS limits as snapshots. |
| Session gallery | Best reactions of this visit, kept in the page only. |
| Opt-in accuracy feedback | "Right mood?" thumbs, storing only the mood vector and answer (Workers KV or D1). Needs a privacy note. |
| Own curated library | Real, licensed, mood-tagged photos or clips served from the site, for exact matching with no rate limit. |
| Cookieless analytics | Cloudflare Web Analytics. |
| Social preview image | `og:image` once the final domain is known. |
| Smarter backup photos | Pexels search (free, 200 requests an hour, 20,000 a month, attribution required) instead of random Cat API photos, only while Giphy is rate limited. |
| Any-animal mode | Tested: Giphy "animal" searches mostly return cats anyway, so low value. |

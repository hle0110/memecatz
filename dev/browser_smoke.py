"""
Runs the web app in headless Chromium with a fake webcam and prints what it
does: status line, mood, caption, emotion scores, gestures, Giphy searches,
and any page errors or blocked requests.

Setup (once): pip install playwright && python -m playwright install chromium
              python dev/make_test_media.py
Serve the site first, for example: npx wrangler dev   (http://localhost:8787)

Examples, from the repo root:
  python dev/browser_smoke.py                                  local site, photos only
  python dev/browser_smoke.py --mock-giphy                     fake Giphy, no key used
  python dev/browser_smoke.py --giphy-key KEY --animal dog     real Giphy searches
  python dev/browser_smoke.py --url https://memecatz.lhieu020304.workers.dev --gpu
  python dev/browser_smoke.py --mock-giphy --challenge         daily challenge, test prompts
  python dev/browser_smoke.py --mock-giphy --challenge --prompts none --date 2026-10-06

With --challenge the camera shows the still frame that matches each round's
prompt (smiling face, thumbs up, peace, pointing), because the looping video
cannot line up with rounds that start whenever the models finish loading.
Angry has no matching frame, so that round shows the smiling face and should
end as a miss. --camera video keeps the looping video instead.

Headless Chromium has no real GPU, so the default adds ?cpu=1. The GPU path
works but is very slow under software rendering.
"""

import argparse
import asyncio
import base64
import json
import os
import time

from playwright.async_api import async_playwright

HERE = os.path.dirname(os.path.abspath(__file__))
VIDEO = os.path.join(HERE, "output", "media", "cam.y4m")
# MediaPipe and LiteRT log their normal startup chatter through console.error
# and console.warn. Lines starting with these are not problems.
RUNTIME_NOISE = ("INFO:", "WARNING: [npu_registry", "W0", "W1", "I0", "I1", "[.WebGL", "GL Driver Message")
MOCK_IDS = ["JIX9t2j0ZTN9S", "mlvseq9yvZhba", "VbnUQpnihPSIgIXuZv", "3oriO0OEd9QIDdllqo"]
VIDEO_ORDER = "happy,approval,chill,suspicious,angry"
PROMPT_FRAMES = {"approval": "thumb_up", "chill": "victory", "suspicious": "pointing_up"}

# Replaces the camera with a canvas that shows the frame for the current
# challenge round, read from the prompt overlay's data attributes. Between
# rounds it shows the plain face.
FOLLOW_PROMPTS = """(() => {
  const frames = %s;
  const map = %s;
  const images = {};
  for (const [name, src] of Object.entries(frames)) { const i = new Image(); i.src = src; images[name] = i; }
  navigator.mediaDevices.getUserMedia = async () => {
    const canvas = document.createElement('canvas');
    canvas.width = 640; canvas.height = 480;
    const ctx = canvas.getContext('2d');
    const draw = () => {
      const o = document.getElementById('ch-overlay');
      const name = o && o.dataset.phase === 'round' ? (map[o.dataset.prompt] || 'face') : 'face';
      const img = images[name];
      if (img.complete && img.naturalWidth) ctx.drawImage(img, 0, 0, 640, 480);
    };
    draw();
    setInterval(draw, 1000 / 15);
    return canvas.captureStream(15);
  };
})();"""

READ_CHALLENGE = """() => {
  const o = document.getElementById('ch-overlay');
  const results = document.getElementById('results');
  return {
    phase: o.dataset.phase || '', prompt: o.dataset.prompt || '',
    big: document.getElementById('ch-big').textContent, hint: document.getElementById('ch-hint').textContent,
    info: document.getElementById('daily-info').textContent,
    done: !results.classList.contains('hidden'),
    title: document.getElementById('res-title').textContent,
    score: document.getElementById('res-score').textContent,
    note: document.getElementById('res-note').textContent,
    rounds: [...document.querySelectorAll('#res-rounds li')].map(li => li.textContent),
    share: document.getElementById('share-text').value,
    stored: localStorage.getItem('memecatz.daily'),
    image: ([...document.querySelectorAll('img.reaction.visible')].map(i => i.src)[0] || ''),
    caption: document.getElementById('cap-top').textContent + ' / ' + document.getElementById('cap-bottom').textContent,
  };
}"""

READ_STATE = """() => ({
  status: document.getElementById('status').textContent,
  mood: document.getElementById('mood').textContent,
  caption: document.getElementById('cap-top').textContent + ' / ' + document.getElementById('cap-bottom').textContent,
  debug: (document.getElementById('debug') || {}).textContent || '',
  image: ([...document.querySelectorAll('img.reaction.visible')].map(i => i.src)[0] || '')
})"""


def section(debug, start, end):
    if start not in debug:
        return ""
    part = debug.split(start, 1)[1]
    return part.split(end, 1)[0].strip().replace("\n", " | ") if end in part else part.strip()


async def main(args):
    if not os.path.exists(VIDEO):
        raise SystemExit("missing test video, run: python dev/make_test_media.py")
    url = args.url.rstrip("/") + "/?debug=1" + ("" if args.gpu else "&cpu=1")
    if args.challenge and args.prompts != "none":
        url += "&prompts=" + args.prompts
    if args.date:
        url += "&date=" + args.date
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=args.chrome or None, args=[
            "--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream",
            f"--use-file-for-fake-video-capture={VIDEO}", "--enable-unsafe-swiftshader"])
        context = await browser.new_context(permissions=["camera"], accept_downloads=True)
        page = await context.new_page()
        if args.challenge and args.camera == "prompts":
            media = os.path.dirname(VIDEO)
            frames = {}
            for name in ("face", "thumb_up", "victory", "pointing_up"):
                with open(os.path.join(media, f"f_{name}.png"), "rb") as handle:
                    frames[name] = "data:image/png;base64," + base64.b64encode(handle.read()).decode()
            await page.add_init_script(FOLLOW_PROMPTS % (json.dumps(frames), json.dumps(PROMPT_FRAMES)))
        problems, searches = [], []
        page.on("pageerror", lambda e: problems.append(f"page error: {e}"))
        page.on("console", lambda m: problems.append(f"console {m.type}: {m.text}")
                if m.type in ("error", "warning") and not m.text.lstrip().startswith(RUNTIME_NOISE) else None)
        page.on("request", lambda r: searches.append(r.url) if "api.giphy.com" in r.url else None)

        if args.giphy_key or args.mock_giphy:
            key = args.giphy_key or "TESTKEY"
            async def config(route):
                await route.fulfill(status=200, content_type="application/json", body=json.dumps({"giphyKey": key}))
            await page.route("**/api/config", config)
        if args.mock_giphy:
            async def giphy(route):
                data = [{"id": g, "title": "t", "images": {"fixed_height": {
                    "url": f"https://media.giphy.com/media/{g}/200.gif",
                    "webp": f"https://media.giphy.com/media/{g}/200.webp"}}} for g in MOCK_IDS]
                await route.fulfill(status=200, content_type="application/json", body=json.dumps({"data": data}))
            await page.route("https://api.giphy.com/**", giphy)

        response = await page.goto(url)
        print("page", response.status, "| csp header:", "yes" if response.headers.get("content-security-policy") else "no")
        if args.animal == "dog":
            await page.select_option("#animal", "dog")
        if args.challenge:
            await run_challenge(page, args)
            for run in range(1, args.runs):
                print(f"--- run {run + 1}")
                await page.click("#daily-start")
                await run_challenge(page, args, started=True)
            if args.reload:
                await page.reload()
                await page.click("#mode-daily")
                print("after reload:", (await page.evaluate(READ_CHALLENGE))["info"])
            print("giphy searches:", [u.split("q=")[1].split("&")[0] for u in searches if "q=" in u])
            print("problems:" if problems else "no page errors or warnings")
            for line in dict.fromkeys(problems):
                print("  ", line[:200])
            await browser.close()
            return
        await page.click("#start")

        start = time.time()
        while time.time() - start < (args.seconds or 45):
            await asyncio.sleep(3)
            s = await page.evaluate(READ_STATE)
            fer = section(s["debug"], "emotion model (7 classes)", "neutral strength")
            gestures = section(s["debug"], "gesture tags", "mood this frame")
            print(f"t={time.time() - start:3.0f}s {s['status'][:60]!r} mood={s['mood']!r} caption={s['caption']!r}")
            print(f"       emotion: {fer[:90]}  gestures: {gestures[:50]}")

        if args.snapshot:
            async with page.expect_download(timeout=10000) as download:
                await page.click("#snapshot")
            path = os.path.join(HERE, "output", "browser_snapshot.png")
            await (await download.value).save_as(path)
            print("snapshot saved to", path)

        print("giphy searches:", [u.split("q=")[1].split("&")[0] for u in searches if "q=" in u])
        print("problems:" if problems else "no page errors or warnings")
        for line in dict.fromkeys(problems):
            print("  ", line[:200])
        await browser.close()


async def run_challenge(page, args, started=False):
    if not started:
        await page.click("#mode-daily")
        print("daily panel:", (await page.evaluate(READ_CHALLENGE))["info"])
        await page.click("#daily-start")
    start = time.time()
    last = None
    while time.time() - start < (args.seconds or 150):
        await asyncio.sleep(0.25)
        s = await page.evaluate(READ_CHALLENGE)
        if s["done"]:
            break
        key = (s["phase"], s["prompt"], s["big"] if s["phase"] != "countdown" else "", s["hint"] if s["phase"] == "result" else "")
        if key != last:
            last = key
            mood = await page.evaluate("document.getElementById('mood').textContent")
            print(f"t={time.time() - start:5.1f}s {s['phase'] or '-':9} {s['prompt']:10} {s['big']!r} {s['hint']!r} mood={mood!r}")
    else:
        print("challenge did not finish in time")
        return
    print("result:", s["title"], "|", s["score"], "|", s["note"])
    for line in s["rounds"]:
        print("   ", line)
    print("share text:", s["share"].replace("\n", " / "))
    print("reaction shown:", s["image"][:80], "| caption:", s["caption"])
    print("stored:", s["stored"])
    if args.face:
        await page.check("#share-face")
    async with page.expect_download(timeout=15000) as download:
        await page.click("#save-share")
    path = os.path.join(HERE, "output", f"daily_share{'_face' if args.face else ''}.png")
    await (await download.value).save_as(path)
    print("share image saved to", path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8787")
    parser.add_argument("--animal", choices=["cat", "dog"], default="cat")
    parser.add_argument("--seconds", type=int, help="free play length (default 45), or the time limit for one challenge run (default 150)")
    parser.add_argument("--gpu", action="store_true", help="do not add ?cpu=1")
    parser.add_argument("--giphy-key", help="answer /api/config with this key, searches go to real Giphy")
    parser.add_argument("--mock-giphy", action="store_true", help="fake Giphy search results")
    parser.add_argument("--snapshot", action="store_true", help="press Save snapshot at the end")
    parser.add_argument("--chrome", help="path to a Chromium binary, if Playwright's own is not installed")
    parser.add_argument("--challenge", action="store_true", help="play the daily challenge instead of free play")
    parser.add_argument("--prompts", default=VIDEO_ORDER, help="?prompts= for --challenge, or none for the real daily")
    parser.add_argument("--date", help="?date=YYYY-MM-DD")
    parser.add_argument("--camera", choices=["prompts", "video"], default="prompts",
                        help="with --challenge: frames that follow the prompt, or the looping video")
    parser.add_argument("--runs", type=int, default=1, help="with --challenge: play this many times in a row")
    parser.add_argument("--reload", action="store_true", help="with --challenge: reload at the end and print the daily panel")
    parser.add_argument("--face", action="store_true", help="with --challenge: tick include my face for the share image")
    asyncio.run(main(parser.parse_args()))

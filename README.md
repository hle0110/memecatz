# MemeCatz

Make a face, and a real cat or dog reacts with a meme.

## Try it in your browser

Go to **[memecatz.lhieu020304.workers.dev](https://memecatz.lhieu020304.workers.dev)** and press Start. Keep a relaxed face for two seconds, then smile, frown, raise an eyebrow, or try a thumbs up or a peace sign. Press s to save a picture. Nothing to install.

## Run it on your computer

Install Python from python.org. Then open this folder and run `./run.sh` on a Mac, or double click `run.bat` on Windows. The first start takes a minute or two to set up. Press s to save a picture, c to reset your face, and q to quit.

## Extras (optional)

For animated reactions that match your mood, get a free key at developers.giphy.com and start the app with `--giphy-key YOUR_KEY`. Add `--animal dog` for dogs instead of cats.

## Having trouble?

If the camera does not open, close other apps that use it. If setup fails, delete the `.venv` folder and start again.

## Privacy

Your camera picture stays on your device and is never uploaded. Saved pictures stay in your own snapshots folder. The only exception is the optional live captions on the desktop app: if you add an OpenAI key, a small photo of your face is sent to OpenAI.

## License

MIT, see LICENSE.

# MemeCatz

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

Your face. Their reaction. Live. From real cats and dogs.

## Try it in your browser

Open **[memecatz.lhieu020304.workers.dev](https://memecatz.lhieu020304.workers.dev)**, press Start, and hold a relaxed face for about two seconds. Then smile, frown, raise an eyebrow, or try a thumbs up, peace sign, open palm, fist, or pointing. Press c to recalibrate and s to save a snapshot. Nothing to install, and your camera never leaves your device.

## What the desktop app does

Watches your face and hands, reads your mood, and shows a real cat or dog reaction next to your webcam with a caption. Works with more than one person at once. Press s for a snapshot, c to recalibrate, q to quit.

## Getting started

Install Python 3.9 or newer. On an Intel Mac, use Python 3.12 or older. Run the launcher: `./run.sh` on Mac/Linux, `run.bat` on Windows, or `python run.py` anywhere. First run takes a minute to set up, every run after that is instant.

A free Giphy key gets animated, mood matched reactions. An OpenAI key turns on live captions (this needs an OpenAI account with paid credits). Both are optional. Set GIPHY_API_KEY and OPENAI_API_KEY, or pass `--giphy-key` and `--openai-key`. Use `--animal dog` for dog reactions, `--save-profile` to remember your calibration next time, and `--refresh-dataset` to fetch fresh backup photos. Run `python check_setup.py` to test your install.

## If something goes wrong

Install errors: delete .venv and rerun the launcher. Reaction stuck on "connecting": no internet yet, it fixes itself. Reactions feel random: add a Giphy key. Nothing opens: check your webcam is free and your terminal has camera permission.

## Privacy

Your webcam feed stays on your machine and is never uploaded. Reactions are downloaded from Giphy, The Cat API, or Dog CEO, which see your IP address like any website. If you set an OpenAI key, a small cropped photo of your face is sent to OpenAI for the optional mood boost. With `--save-profile`, a face encoding is stored in the profiles folder on your computer so it can recognize you next time; delete that folder to remove it. Snapshots stay in your own snapshots folder.

## License

MIT, see LICENSE. Built with OpenCV, MediaPipe, and LiteRT. Reactions from GIPHY, The Cat API, and the Dog CEO API.

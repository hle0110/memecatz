# MemeCatz

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.9 to 3.12](https://img.shields.io/badge/python-3.9%20to%203.12-blue.svg)](https://www.python.org/)

Your face. Their reaction. Live. From real cats and dogs.

## What it does

Watches your face and hands, reads your mood, and shows a real cat or dog reaction next to your webcam with a caption. Works with more than one person at once. Press s for a snapshot, c to recalibrate, q to quit.

## Getting started

Install Python 3.9 to 3.12. Python 3.13 is not recommended yet, mediapipe has a bug on it that can stop the camera view from loading. Run the launcher: `./run.sh` on Mac/Linux, `run.bat` on Windows, or `python run.py` anywhere. First run takes a minute to set up, every run after that is instant.

A free Giphy key gets mood matched reactions, and a free OpenAI key turns on live captions. Both optional. Set GIPHY_API_KEY and OPENAI_API_KEY, or pass `--giphy-key` and `--openai-key`. Use `--animal dog` for dog reactions. Run `python check_setup.py` to test your install.

## If something goes wrong

Install errors: delete .venv and rerun the launcher. Reaction stuck on "connecting": no internet yet, it fixes itself. Reactions feel random: add a Giphy key. Nothing opens: check your webcam is free and your terminal has camera permission.

## Privacy

Your webcam feed stays on your machine and is never uploaded anywhere. The only thing that ever leaves your computer is a small cropped photo of your face, and only if you set an OpenAI key for the optional mood boost. Snapshots you save with the s key stay in your own snapshots folder and are never sent anywhere.

## License

MIT, see LICENSE. Built with OpenCV, MediaPipe, and TensorFlow.

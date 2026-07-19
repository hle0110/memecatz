# MemeCatz

Your face. Their reaction. Live. From real cats and dogs.

## What it does

Watches your face and hands, reads your mood, and shows a real cat or dog reaction next to your webcam with a caption. Works with more than one person at once. Press s for a snapshot, c to recalibrate, q to quit.

## Getting started

Install Python 3.9 to 3.13. Run the launcher: `./run.sh` on Mac/Linux, `run.bat` on Windows, or `python run.py` anywhere. First run takes a minute to set up, every run after that is instant.

A free Giphy key gets mood matched reactions, and a free OpenAI key turns on live captions. Both optional. Set GIPHY_API_KEY and OPENAI_API_KEY, or pass `--giphy-key` and `--openai-key`. Use `--animal dog` for dog reactions. Run `python check_setup.py` to test your install.

## If something goes wrong

Install errors: delete .venv and rerun the launcher. Reaction stuck on "connecting": no internet yet, it fixes itself. Reactions feel random: add a Giphy key. Nothing opens: check your webcam is free and your terminal has camera permission.

## License

MIT, see LICENSE. Built with OpenCV, MediaPipe, and TensorFlow.

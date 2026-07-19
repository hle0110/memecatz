#!/usr/bin/env bash
cd "$(dirname "$0")"
if command -v python3 >/dev/null 2>&1; then
    python3 run.py "$@"
else
    python run.py "$@"
fi

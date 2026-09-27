#!/bin/zsh
# Double-click (or `open`) to preview every camera the Mac can see, incl. the iPhone via Continuity Camera.
# Runs in Terminal, which can ask macOS for camera access; a helper app cannot.
cd "$(dirname "$0")"
.venv/bin/python preview_cam.py
echo; echo "press any key to close"; read -k1

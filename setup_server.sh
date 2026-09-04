#!/bin/bash
# start_server.sh — starts the MuseTalk avatar WebSocket server.
#
# Run this from anywhere on the pod: bash start_server.sh
# It does NOT do the one-time env/weights/prep setup — that's runpod_setup.sh,
# run once. This script only (re)starts the actual live server, and is safe
# to run again any time you want to restart it (kills any old copy first).
#
# ─────────────────────────────────────────────────────────────────────────
# WHERE EVERYTHING MUST LIVE (fixed paths this script assumes):
#
#   /workspace/MuseTalk/                          <- the cloned repo root
#   /workspace/MuseTalk/mt_env/                   <- the python venv (from runpod_setup.sh)
#   /workspace/MuseTalk/avatar_ws_server.py        <- THIS is the server file.
#                                                     Must sit directly in the
#                                                     repo root (NOT in /workspace/,
#                                                     NOT in a subfolder) because it
#                                                     does `import scripts.realtime_inference`,
#                                                     which only resolves when run
#                                                     from inside MuseTalk/.
#   /workspace/MuseTalk/data/video/in_interviewer.mp4
#                                                   <- THE AVATAR VIDEO. Must be
#                                                     named exactly this and sit in
#                                                     data/video/ — it's the same
#                                                     path used both when you ran
#                                                     the one-time "preparation"
#                                                     pass (runpod_setup.sh step 6)
#                                                     AND now at server startup.
#                                                     If you ever swap in a
#                                                     different avatar video, you
#                                                     MUST re-run the preparation
#                                                     pass first — the server loads
#                                                     from the cached prepared data,
#                                                     not the raw video, at runtime.
#   /workspace/MuseTalk/models/                    <- all downloaded weights
#     musetalkV15/unet.pth, musetalk.json
#     sd-vae/config.json, diffusion_pytorch_model.bin
#     whisper/config.json, pytorch_model.bin, preprocessor_config.json
#     dwpose/dw-ll_ucoco_384.pth
#     face-parse-bisent/79999_iter.pth, resnet18-5c106cde.pth
#   /workspace/MuseTalk/results/v15/avatars/interviewer/
#                                                   <- the PREPARED avatar cache
#                                                     (latents.pt, coords.pkl, masks,
#                                                     full_imgs/) built once by
#                                                     runpod_setup.sh step 6. This
#                                                     script checks it exists before
#                                                     starting anything.
#
# There is no "audio" file to place anywhere — audio comes live from Gemini
# through the browser -> Node -> avatar_client.js -> this server's WebSocket,
# per-turn, at runtime. Nothing audio-related is stored on disk here except
# the tiny per-turn temp WAVs the server writes itself under --tmp_dir
# (default /tmp/avatar_turns) and deletes right after each turn finishes.
# ─────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Fixed locations — edit these if your layout genuinely differs ─────────
MUSETALK_DIR="/workspace/MuseTalk"
SERVER_SCRIPT="$MUSETALK_DIR/avatar_ws_server.py"
VENV_PYTHON="$MUSETALK_DIR/mt_env/bin/python"
AVATAR_VIDEO="$MUSETALK_DIR/data/video/in_interviewer.mp4"
PREPARED_AVATAR_DIR="$MUSETALK_DIR/results/v15/avatars/interviewer"
LOG_FILE="$MUSETALK_DIR/avatar_ws_server.log"

# ── Launch settings — safe to tweak ────────────────────────────────────────
HOST="0.0.0.0"
PORT="8765"
AVATAR_ID="interviewer"
BATCH_SIZE="4"
FPS="25"
CLIP_FRAMES="10"
JPEG_QUALITY="80"

# ── Sanity checks — fail loudly and early instead of a confusing crash ────
echo "── Checking everything is where it should be ──────────────────────"

if [ ! -f "$VENV_PYTHON" ]; then
  echo "[FATAL] venv python not found at $VENV_PYTHON"
  echo "        Run runpod_setup.sh first (it builds mt_env)."
  exit 1
fi
echo "  ✓ venv found: $VENV_PYTHON"

if [ ! -f "$SERVER_SCRIPT" ]; then
  echo "[FATAL] avatar_ws_server.py not found at $SERVER_SCRIPT"
  echo "        Copy it there (e.g. runpodctl send/receive) before running this."
  exit 1
fi
echo "  ✓ server script found: $SERVER_SCRIPT"

if [ ! -f "$AVATAR_VIDEO" ]; then
  echo "[FATAL] Avatar video not found at $AVATAR_VIDEO"
  echo "        Copy your approved in_interviewer.mp4 there first."
  exit 1
fi
echo "  ✓ avatar video found: $AVATAR_VIDEO"

if [ ! -d "$PREPARED_AVATAR_DIR" ] || [ -z "$(ls -A "$PREPARED_AVATAR_DIR" 2>/dev/null)" ]; then
  echo "[FATAL] Prepared avatar data not found at $PREPARED_AVATAR_DIR"
  echo "        Run the one-time preparation pass first (runpod_setup.sh step 6) —"
  echo "        the server loads cached avatar data, it does not build it itself."
  exit 1
fi
echo "  ✓ prepared avatar found: $PREPARED_AVATAR_DIR"

echo "── All checks passed ───────────────────────────────────────────────"

# ── Stop any previous instance so we never end up with two servers ────────
# fighting over the same GPU/port (the exact situation that caused the
# queue backlog you hit — an old server can silently keep running after
# you think you've restarted it).
if pgrep -f "avatar_ws_server.py" > /dev/null; then
  echo "Stopping existing avatar_ws_server.py instance(s)..."
  pkill -f "avatar_ws_server.py" || true
  sleep 3
fi

# ── Start it ────────────────────────────────────────────────────────────
cd "$MUSETALK_DIR"

echo "Starting server..."
nohup "$VENV_PYTHON" avatar_ws_server.py \
  --host "$HOST" --port "$PORT" \
  --version v15 \
  --avatar_id "$AVATAR_ID" \
  --video_path data/video/in_interviewer.mp4 \
  --unet_model_path models/musetalkV15/unet.pth \
  --unet_config models/musetalkV15/musetalk.json \
  --whisper_dir models/whisper \
  --extra_margin 14 --parsing_mode jaw \
  --left_cheek_width 90 --right_cheek_width 90 \
  --batch_size "$BATCH_SIZE" --fps "$FPS" \
  --clip_frames "$CLIP_FRAMES" --jpeg_quality "$JPEG_QUALITY" \
  > "$LOG_FILE" 2>&1 &

SERVER_PID=$!
echo "Server starting in background, PID $SERVER_PID."
echo "Log: $LOG_FILE"
echo ""
echo "Waiting for it to finish loading models (usually 60-90s)..."

# Poll the log for the ready line instead of a blind sleep, so this script
# tells you the real status instead of just guessing a wait time.
for i in $(seq 1 24); do
  if grep -q "Listening on ws://" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "── Server is up ────────────────────────────────────────────────"
    grep "Listening on ws://" "$LOG_FILE"
    echo ""
    echo "Set this pod's TCP-proxy address for port $PORT as AVATAR_WS_URL"
    echo "in the Node server's config (RunPod console -> Connect panel)."
    echo ""
    echo "Tail logs any time with:  tail -f $LOG_FILE"
    exit 0
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo ""
    echo "[FATAL] Server process died during startup. Last 40 log lines:"
    tail -40 "$LOG_FILE"
    exit 1
  fi
  sleep 5
done

echo ""
echo "[WARN] Still not ready after 2 minutes — check manually:"
echo "       tail -f $LOG_FILE"

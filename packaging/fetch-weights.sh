#!/bin/bash
# Resumable download of the MiniMax-Music3-MLX weights (curl -C -, one file at a time).
# No curl --retry: on a stream error it restarts from the original offset and truncates the
# file; the until-loop re-invokes curl so -C - resumes from the bytes on disk.
#   bash fetch-weights.sh [MODEL_DIR]      default $HOME/models/MiniMax-Music3-MLX
# A lock dir stops two copies writing the same file; prints "ALL_DONE" at the end.
set -u
D="${1:-$HOME/models/MiniMax-Music3-MLX}"
B=https://huggingface.co/PocketAiHub/MiniMax-Music3-MLX/resolve/main
mkdir -p "$D"
if ! mkdir "$D/.fetch.lock" 2>/dev/null; then echo "another fetch holds $D/.fetch.lock"; exit 2; fi
trap 'rmdir "$D/.fetch.lock" 2>/dev/null' EXIT
for f in vae/minimax_music3_dav.safetensors \
         diffusion_models/minimax_music3_dit_int8_convrot.safetensors \
         text_encoders/minimax_music3_text_encoder_pruned_int8_convrot.safetensors \
         examples/rock-and-roll-60s.wav; do
  mkdir -p "$D/$(dirname "$f")"
  until curl -L -C - --speed-limit 10000 --speed-time 60 -sS -o "$D/$f" "$B/$f"; do
    echo "retry $f $(date '+%H:%M:%S')"; sleep 5
  done
  echo "done $f $(stat -f %z "$D/$f") $(date '+%H:%M:%S')"
done
echo ALL_DONE

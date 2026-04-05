#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT_ROOT="${1:-$ROOT_DIR/results}"

bash "$ROOT_DIR/commands/run_video_best.sh" "$OUT_ROOT"
bash "$ROOT_DIR/commands/run_audio_best.sh" "$OUT_ROOT"

echo "[DONE] all runs finished."
echo "video summary: $OUT_ROOT/video_run/summary.json"
echo "audio summary: $OUT_ROOT/audio_run/summary.json"

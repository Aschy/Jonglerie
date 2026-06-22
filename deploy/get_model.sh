#!/usr/bin/env bash
# Exporte yolov8m.onnx (modèle de détection du ballon) dans models/.
# Étape ponctuelle : utilise ultralytics+torch dans un venv JETABLE pour ne pas
# alourdir le venv runtime. À relancer seulement si le modèle est absent.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="$ROOT/models/yolov8m.onnx"
mkdir -p "$ROOT/models"

if [ -f "$DEST" ]; then
  echo "[=] Modèle déjà présent : $DEST"; exit 0
fi

TMP="$(mktemp -d)"
echo "[+] venv temporaire d'export dans $TMP"
python3 -m venv "$TMP/venv"
"$TMP/venv/bin/pip" install -q --upgrade pip
"$TMP/venv/bin/pip" install -q "ultralytics>=8.0" onnx onnxruntime
echo "[+] export YOLOv8m -> ONNX (imgsz=640)…"
( cd "$TMP" && "$TMP/venv/bin/yolo" export model=yolov8m.pt format=onnx imgsz=640 )
mv "$TMP/yolov8m.onnx" "$DEST"
rm -rf "$TMP"
echo "[✓] Modèle prêt : $DEST"
ls -lh "$DEST"
